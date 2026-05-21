"""
WarpBinaryContactMeasurementModel
=================================

Batched counterpart to :class:`BinaryContactMeasurementModel`.

The CPU model loops over N robots, reads each one's torque sensor, computes
``np.linalg.norm``, thresholds it, and writes one likelihood value per
particle. That's N device-independent CPU reads.

The Warp version reads the batched ``data.sensordata`` array (shape
``(nworld, nsensor)``) once, computes the per-world torque-norm on the GPU
with a kernel, applies the same negative-info / positive-info contact logic
the CPU model uses, and copies a single ``(nworld,)`` likelihood vector back
to the host.

This collapses N host-side sensor reads into one device→host transfer per
filter step.

Sensor lookup
-------------
MJWarp uses ``data.sensordata[world, address:address+dim]`` for each sensor,
where ``address`` and ``dim`` come from ``model.sensor_adr`` /
``model.sensor_dim``. We pull those from the host-side ``model`` once and
pass the resolved torque-sensor offsets to the kernel.

The container's robot is expected to have a sensor named ``"hand_torque"``
(matches the CPU ``MujocoWarpRobot``). If your XML uses a different name,
pass it via the constructor.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
import warp as wp

from src.estimation.measurement import BaseMeasurementModel
from src.estimation.motion import BaseMotionModel
from src.warp_estimation.warp_container import WarpRobotContainer


# ---------------------------------------------------------------------------
# Warp kernel
# ---------------------------------------------------------------------------


@wp.kernel
def _binary_contact_likelihood_kernel(
    sensordata: wp.array2d(dtype=wp.float32),
    likelihoods: wp.array(dtype=wp.float32),
    torque_adr: int,
    torque_dim: int,
    threshold: float,
    real_contact: int,
    lik_killed_zero: float,
    lik_killed_one: float,
) -> None:
    """Per-world contact-likelihood computation.

    Logic mirrors :class:`BinaryContactMeasurementModel.compute_likelihoods`:

      sim_contact = ||sensordata[i, torque_adr:torque_adr+torque_dim]|| > threshold

      if real == 0 and sim == 1 → likelihood = lik_killed_zero  (≈ 0.0)
      if real == 1 and sim == 0 → likelihood = lik_killed_one   (≈ 0.001)
      otherwise                 → likelihood = 1.0
    """
    i = wp.tid()

    # Compute ||torque_i||_2 over torque_dim components.
    s = float(0.0)
    for k in range(torque_dim):
        v = sensordata[i, torque_adr + k]
        s += v * v
    norm = wp.sqrt(s)

    sim_contact = int(0)
    if norm > threshold:
        sim_contact = int(1)

    if real_contact == 0 and sim_contact == 1:
        likelihoods[i] = lik_killed_zero
    elif real_contact == 1 and sim_contact == 0:
        likelihoods[i] = lik_killed_one
    else:
        likelihoods[i] = float(1.0)


# ---------------------------------------------------------------------------
# Measurement model
# ---------------------------------------------------------------------------


class WarpBinaryContactMeasurementModel(BaseMeasurementModel):
    """Batched binary-contact likelihood, computed on the GPU.

    Parameters
    ----------
    container
        The :class:`WarpRobotContainer` whose batched ``Data`` to read.
    contact_threshold
        Torque-norm threshold above which a world is considered "in contact".
    torque_sensor_name
        Name of the joint/site torque sensor in the MuJoCo XML. Defaults to
        ``"hand_torque"`` to match :class:`MujocoWarpRobot`.
    likelihood_negative_info
        Likelihood assigned to a world that registers contact when the real
        robot does not. Matches the CPU model's ``0.0``.
    likelihood_positive_info
        Likelihood assigned to a world that does *not* register contact when
        the real robot does. Matches the CPU model's ``0.001``.

    Notes on safety
    ---------------
    The CPU model uses a hard ``0.0`` for the negative-info case, which
    causes weight collapse if **every** world is wrong on the same step.
    The default here preserves that behaviour for compatibility; if you've
    been hitting "Particle Extinction" warnings, consider passing a tiny
    epsilon like ``1e-12`` instead.
    """

    def __init__(
        self,
        container: WarpRobotContainer,
        contact_threshold: float = 0.3,
        *,
        torque_sensor_name: str = "hand_torque",
        likelihood_negative_info: float = 0.0,
        likelihood_positive_info: float = 0.001,
    ) -> None:
        self.container = container
        self.threshold = float(contact_threshold)
        self.likelihood_negative_info = float(likelihood_negative_info)
        self.likelihood_positive_info = float(likelihood_positive_info)

        # Resolve sensor address/dim from the host model. These never change.
        model = container.warp_robot.model
        sensor_id = mujoco.mj_name2id(  # type: ignore[attr-defined]
            model, mujoco.mjtObj.mjOBJ_SENSOR, torque_sensor_name  # type: ignore[attr-defined]
        )
        if sensor_id == -1:
            raise ValueError(
                f"Torque sensor '{torque_sensor_name}' not found in the MuJoCo "
                "model. WarpBinaryContactMeasurementModel needs a torque sensor "
                "on the end effector; add one in the XML or pass "
                "torque_sensor_name=... with the correct name."
            )

        self.torque_adr = int(model.sensor_adr[sensor_id])
        self.torque_dim = int(model.sensor_dim[sensor_id])
        if self.torque_dim < 1:
            raise ValueError(
                f"Torque sensor '{torque_sensor_name}' reports dim={self.torque_dim}, "
                "which is degenerate. Check the sensor definition in the XML."
            )

        # Pre-allocated device-side likelihood buffer.
        kwargs: dict[str, Any] = {
            "dtype": wp.float32,
            "shape": (container.num_particles,),
        }
        if container.device is not None:
            kwargs["device"] = container.device
        self._likelihoods_dev: wp.array = wp.zeros(**kwargs)

    def compute_likelihoods(
        self, particles: np.ndarray, observation: dict
    ) -> np.ndarray:
        """Return a length-N array of per-particle likelihoods.

        ``particles`` is not read — the physics state lives inside the Warp
        ``Data``. We accept the argument to keep the :class:`BaseMeasurementModel`
        contract.
        """
        real_contact = int(observation.get("contact", 0))
        warp_data = self.container.warp_robot.warp_data

        wp.launch(
            kernel=_binary_contact_likelihood_kernel,
            dim=self.container.num_particles,
            inputs=[
                warp_data.sensordata,
                self._likelihoods_dev,
                self.torque_adr,
                self.torque_dim,
                self.threshold,
                real_contact,
                self.likelihood_negative_info,
                self.likelihood_positive_info,
            ],
        )

        # One device→host transfer. wp.synchronize() ensures the kernel has
        # finished before numpy() reads back; numpy() does this implicitly,
        # but being explicit makes the synchronisation point visible if you
        # ever profile this.
        wp.synchronize()
        return self._likelihoods_dev.numpy().astype(np.float64)