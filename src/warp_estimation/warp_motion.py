"""
WarpPositionMotionModel
=======================

Batched counterpart to :class:`PositionMotionModel`. Replaces the Python
``for robot in container.robots`` loop with vectorised writes to the
``(nworld, ...)`` Warp arrays plus a single ``mjw.step`` call.

Two state-write operations are exposed:

``propagate``
    Called every filter step with the *real* arm command. Broadcasts the
    command into all ``nworld`` worlds' ``ctrl`` arrays and advances the
    simulation by one step.

``change_internal_state``
    Called on filter init and after resampling. Writes the per-particle
    object pose (X, Y, optional θ) into the batched ``qpos`` array, zeroes
    the object's batched ``qvel``, and pushes the real arm pose into every
    world's arm slots. This is the "teleport the block to where this particle
    thinks it is" operation.

Why a Warp kernel for the qpos write
------------------------------------
The CPU model iterates N times and writes ``robot.data.qpos[...] = ...``.
On the GPU side, ``data.qpos`` is a single ``wp.array`` of shape
``(nworld, nq)``. We could ``wp.copy`` from a host buffer every step, but
the X/Y/θ slots are at fixed offsets — writing them with a small
``wp.kernel`` avoids constructing a full ``(nworld, nq)`` host tile when we
only want to overwrite a few columns.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
import warp as wp

import mujoco_warp as mjw

from src.estimation.motion import BaseMotionModel
from src.warp_estimation.warp_container import WarpRobotContainer


# ---------------------------------------------------------------------------
# Warp kernels
# ---------------------------------------------------------------------------
#
# Kernels are JIT-compiled on first launch. Keep the signature minimal —
# every extra parameter is a recompilation trigger if its type changes.


@wp.kernel
def _write_xy_kernel(
    qpos: wp.array2d(dtype=wp.float32),
    particles: wp.array2d(dtype=wp.float32),
    qpos_adr: int,
) -> None:
    """For each world i, write particles[i, 0:2] into qpos[i, qpos_adr:qpos_adr+2]."""
    i = wp.tid()
    qpos[i, qpos_adr + 0] = particles[i, 0]
    qpos[i, qpos_adr + 1] = particles[i, 1]


@wp.kernel
def _write_xy_theta_kernel(
    qpos: wp.array2d(dtype=wp.float32),
    particles: wp.array2d(dtype=wp.float32),
    qpos_adr: int,
) -> None:
    """For each world i, write X, Y and θ-as-quaternion (Z-axis rotation) into qpos.

    Quaternion layout in MuJoCo is [qw, qx, qy, qz], stored at offsets
    qpos_adr+3 .. qpos_adr+6 for a free joint (X, Y, Z at +0, +1, +2).
    """
    i = wp.tid()
    x = particles[i, 0]
    y = particles[i, 1]
    theta = particles[i, 2]

    qpos[i, qpos_adr + 0] = x
    qpos[i, qpos_adr + 1] = y
    # Z-axis quaternion: qw = cos(θ/2), qz = sin(θ/2), qx = qy = 0
    half = theta * float(0.5)
    qpos[i, qpos_adr + 3] = wp.cos(half)
    qpos[i, qpos_adr + 4] = float(0.0)
    qpos[i, qpos_adr + 5] = float(0.0)
    qpos[i, qpos_adr + 6] = wp.sin(half)


@wp.kernel
def _zero_obj_qvel_kernel(
    qvel: wp.array2d(dtype=wp.float32),
    dof_adr: int,
    dof_count: int,
) -> None:
    """Zero the object's velocity DOFs across all worlds (so it doesn't drift)."""
    i = wp.tid()
    for k in range(dof_count):
        qvel[i, dof_adr + k] = float(0.0)


@wp.kernel
def _broadcast_arm_qpos_kernel(
    qpos: wp.array2d(dtype=wp.float32),
    arm_qpos: wp.array(dtype=wp.float32),
    arm_dof: int,
) -> None:
    """For each world i, copy arm_qpos[0:arm_dof] into qpos[i, 0:arm_dof]."""
    i = wp.tid()
    for k in range(arm_dof):
        qpos[i, k] = arm_qpos[k]


@wp.kernel
def _broadcast_arm_qvel_kernel(
    qvel: wp.array2d(dtype=wp.float32),
    arm_qvel: wp.array(dtype=wp.float32),
    arm_dof: int,
) -> None:
    i = wp.tid()
    for k in range(arm_dof):
        qvel[i, k] = arm_qvel[k]


# ---------------------------------------------------------------------------
# Motion model
# ---------------------------------------------------------------------------


class WarpPositionMotionModel(BaseMotionModel):
    """Batched position-based motion model running on MuJoCo Warp.

    The particle state dimension determines which kernel writes the object
    pose. ``dim`` is detected from the first ``change_internal_state`` call:

    - ``dim == 2`` → write X and Y; leave object orientation at the model
      default (this matches the CPU model's behaviour).
    - ``dim == 3`` → write X, Y, and θ (encoded as a Z-axis quaternion).

    ``dim == 1`` is intentionally not supported here. The CPU model handled
    it by writing only the Y slot; if you need that, add a third kernel —
    don't reuse the 2D one or you'll silently overwrite the X slot with
    garbage when ``particles`` is shape ``(N, 1)``.
    """

    SUPPORTED_DIMS = (2, 3)

    def __init__(
        self,
        container: WarpRobotContainer,
        *,
        gripper_open_width: float = 0.08,
    ) -> None:
        self.container = container
        self.gripper_open_width = gripper_open_width

        # Cached host-side staging buffers for arm qpos/qvel. Reusing them
        # avoids per-step Python allocations; the actual H2D copy still
        # happens via wp.array(...) inside the kernel launches.
        self._arm_qpos_host: np.ndarray | None = None
        self._arm_qvel_host: np.ndarray | None = None

        # Cached Warp arrays for the broadcast inputs. Sized once, refilled
        # in-place each call to avoid reallocating GPU memory every step.
        self._arm_qpos_dev: wp.array | None = None
        self._arm_qvel_dev: wp.array | None = None

        # We don't know dim until the first change_internal_state. Store the
        # particle buffer on first use; reallocate if the caller ever resizes.
        self._particles_dev: wp.array | None = None
        self._particles_dev_shape: tuple[int, int] | None = None

    # -- BaseMotionModel ------------------------------------------------------

    def propagate(self, particles: np.ndarray, control_input: dict) -> np.ndarray:
        """Apply the current arm command to every world, then step once."""
        qpos_cmd = np.asarray(control_input["joints"], dtype=np.float32)
        gripper_width = float(control_input["gripper"])

        # move_joints sets self._ctrl across all worlds, copies it to device,
        # AND advances the sim once. That single mjw.step call is the entire
        # batched propagate.
        self.container.warp_robot.move_gripper(gripper_width)
        self.container.warp_robot.move_joints(qpos_cmd)

        # Particles are pure book-keeping for a position filter — the physics
        # is held inside Warp's qpos. The filter still needs the array back.
        return particles

    def change_internal_state(
        self, particles: np.ndarray, real_state: dict
    ) -> None:
        """Teleport object poses to ``particles`` and arm to ``real_state``."""
        if particles.ndim != 2:
            raise ValueError(
                f"WarpPositionMotionModel expects 2D particles of shape "
                f"(N, dim), got shape {particles.shape}. "
                "Reshape upstream — broadcasting a 1D array silently changes "
                "the kernel launch geometry."
            )

        n_worlds, dim = particles.shape
        if n_worlds != self.container.num_particles:
            raise ValueError(
                f"Particle count mismatch: container has "
                f"{self.container.num_particles} worlds but received "
                f"{n_worlds} particles. The batched Data object cannot be "
                "resized after construction."
            )
        if dim not in self.SUPPORTED_DIMS:
            raise ValueError(
                f"Unsupported particle dimension {dim}. "
                f"WarpPositionMotionModel supports {self.SUPPORTED_DIMS}; "
                "see class docstring for why 1D is rejected."
            )

        # Upload particles. Reuse the device buffer when shape matches.
        particles_f32 = np.ascontiguousarray(particles, dtype=np.float32)
        self._ensure_particles_dev(particles_f32)

        warp_data = self.container.warp_robot.warp_data
        qpos_adr = self.container.qpos_adr
        dof_adr = self.container.dof_adr

        # 1. Write object pose into batched qpos.
        if dim == 2:
            wp.launch(
                kernel=_write_xy_kernel,
                dim=n_worlds,
                inputs=[warp_data.qpos, self._particles_dev, qpos_adr],
            )
        else:  # dim == 3
            wp.launch(
                kernel=_write_xy_theta_kernel,
                dim=n_worlds,
                inputs=[warp_data.qpos, self._particles_dev, qpos_adr],
            )

        # 2. Zero the object's 6 velocity DOFs (free joint → 6 DOFs).
        wp.launch(
            kernel=_zero_obj_qvel_kernel,
            dim=n_worlds,
            inputs=[warp_data.qvel, dof_adr, 6],
        )

        # 3. Broadcast the real arm qpos/qvel into every world's arm slots.
        if real_state is not None:
            arm_qpos = np.asarray(real_state.get("qpos", []), dtype=np.float32)
            arm_qvel = np.asarray(real_state.get("qvel", []), dtype=np.float32)

            # Only write the arm DOFs the model actually has. We assume the
            # arm DOFs sit in qpos[0:arm_dof] — same convention as the CPU
            # path. If your XML puts the arm elsewhere, change this slice.
            arm_dof_qpos = min(len(arm_qpos), self.container.qpos_adr)
            arm_dof_qvel = min(len(arm_qvel), self.container.dof_adr)

            if arm_dof_qpos > 0:
                self._upload_arm_qpos(arm_qpos[:arm_dof_qpos])
                wp.launch(
                    kernel=_broadcast_arm_qpos_kernel,
                    dim=n_worlds,
                    inputs=[warp_data.qpos, self._arm_qpos_dev, arm_dof_qpos],
                )

            if arm_dof_qvel > 0:
                self._upload_arm_qvel(arm_qvel[:arm_dof_qvel])
                wp.launch(
                    kernel=_broadcast_arm_qvel_kernel,
                    dim=n_worlds,
                    inputs=[warp_data.qvel, self._arm_qvel_dev, arm_dof_qvel],
                )

        # 4. Refresh kinematics so subsequent sensor reads are valid.
        #    mjw.kinematics + mjw.com_pos etc. are bundled in forward(); but
        #    after a state edit, we want the same effect as mj_forward on CPU.
        mjw.forward(self.container.warp_robot.warp_model, warp_data)

        # The next sync_host() in the underlying robot will pull world_id 0
        # back to the host MjData. Mark dirty.
        self.container.warp_robot._host_dirty = True

    # -- buffer helpers -------------------------------------------------------

    def _ensure_particles_dev(self, particles_f32: np.ndarray) -> None:
        """Allocate or refill the device-side particle buffer."""
        shape = particles_f32.shape
        if self._particles_dev is None or self._particles_dev_shape != shape:
            kwargs: dict[str, Any] = {"dtype": wp.float32}
            if self.container.device is not None:
                kwargs["device"] = self.container.device
            self._particles_dev = wp.array(particles_f32, **kwargs)
            self._particles_dev_shape = shape
        else:
            # Reuse: refill in-place via a fresh host->device copy. wp.copy
            # accepts a wp.array source, so we wrap the host buffer.
            kwargs = {"dtype": wp.float32}
            if self.container.device is not None:
                kwargs["device"] = self.container.device
            src = wp.array(particles_f32, **kwargs)
            wp.copy(self._particles_dev, src)

    def _upload_arm_qpos(self, arm_qpos: np.ndarray) -> None:
        kwargs: dict[str, Any] = {"dtype": wp.float32}
        if self.container.device is not None:
            kwargs["device"] = self.container.device
        if self._arm_qpos_dev is None or self._arm_qpos_dev.shape[0] != arm_qpos.shape[0]:
            self._arm_qpos_dev = wp.array(arm_qpos, **kwargs)
        else:
            src = wp.array(arm_qpos, **kwargs)
            wp.copy(self._arm_qpos_dev, src)

    def _upload_arm_qvel(self, arm_qvel: np.ndarray) -> None:
        kwargs: dict[str, Any] = {"dtype": wp.float32}
        if self.container.device is not None:
            kwargs["device"] = self.container.device
        if self._arm_qvel_dev is None or self._arm_qvel_dev.shape[0] != arm_qvel.shape[0]:
            self._arm_qvel_dev = wp.array(arm_qvel, **kwargs)
        else:
            src = wp.array(arm_qvel, **kwargs)
            wp.copy(self._arm_qvel_dev, src)