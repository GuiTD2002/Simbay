"""
WarpRobotContainer
==================

A drop-in replacement for ``RobotContainer`` that batches all particle worlds
into a single MuJoCo Warp simulation.

Design rationale
----------------
The CPU container holds ``num_particles`` independent ``MujocoRobot`` objects
and iterates over them in a Python ``for`` loop. That pattern does not survive
the move to Warp: the whole point of ``mjw.put_data(..., nworld=N)`` is that
**one** ``Data`` object holds all N worlds and **one** ``mjw.step`` advances
them in parallel on the GPU.

So this container holds a single :class:`MujocoWarpRobot` configured with
``nworld=num_particles`` and exposes the same metadata fields the CPU container
exposed (``obj_id``, ``jnt_adr``, ``qpos_adr``, ``dof_adr``), so motion and
measurement models that only need that metadata can stay shape-compatible with
the CPU versions.

What it deliberately does **not** expose
----------------------------------------
The ``robots: list`` field is gone. Code that used to iterate
``for robot in container.robots`` must be ported to operate on the batched
arrays held by ``container.warp_robot`` (one tensor of shape ``(nworld, ...)``
rather than N independent objects). That is the whole speedup.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any

import mujoco

from src.robots.warp_mujoco_robot import MujocoWarpRobot


@dataclass
class WarpRobotContainer:
    """Single-batched-world counterpart to :class:`RobotContainer`.

    Parameters
    ----------
    num_particles
        Number of particle worlds to simulate in parallel. Becomes ``nworld``
        on the underlying Warp ``Data``.
    props
        Object properties dict (same shape as the CPU container expects).
    dt
        Simulation timestep, in seconds.
    nconmax, njmax
        Optional Warp buffer sizes. ``nconmax`` is the per-world contact
        capacity, ``njmax`` is the per-world constraint capacity. If you see
        overflow warnings from MJWarp, raise these. Conservative defaults are
        applied when ``None``.
    device
        Warp device string (e.g. ``"cuda:0"``). ``None`` lets Warp pick.
    """

    num_particles: int
    props: dict[str, Any]
    dt: float
    # Per-world buffer caps. nconmax = max contacts per world; nccdmax = max
    # convex pairs per world (CCD/EPA). Total device alloc scales as cap * nworld,
    # so keep these tight — naccdmax dominates GPU memory.
    nconmax: int | None = None
    njmax: int | None = None
    nccdmax: int | None = None
    naccdmax: int | None = None
    ccd_iterations: int | None = None
    device: str | None = None

    # Generated in __post_init__ — not part of the constructor signature.
    warp_robot: MujocoWarpRobot = field(init=False)
    obj_id: int = field(init=False)
    jnt_adr: int = field(init=False)
    qpos_adr: int = field(init=False)
    dof_adr: int = field(init=False)

    def __post_init__(self) -> None:
        if self.num_particles < 1:
            raise ValueError(
                f"num_particles must be >= 1, got {self.num_particles}. "
                "A batched container with zero worlds has nothing to simulate."
            )

        self.warp_robot = MujocoWarpRobot(
            object_properties=self.props,
            dt=self.dt,
            nworld=self.num_particles,
            world_id=0,
            nconmax=self.nconmax,
            njmax=self.njmax,
            nccdmax=self.nccdmax,
            naccdmax=self.naccdmax,
            ccd_iterations=self.ccd_iterations,
            device=self.device,
        )

        # Resolve object/joint addresses from the host-side model. These are
        # constant for the lifetime of the container.
        base_model = self.warp_robot.model

        self.obj_id = mujoco.mj_name2id(  # type: ignore[attr-defined]
            base_model, mujoco.mjtObj.mjOBJ_BODY, "object"  # type: ignore[attr-defined]
        )
        if self.obj_id == -1:
            raise ValueError(
                "Body 'object' not found in the MuJoCo model. "
                "WarpRobotContainer requires a body named 'object' "
                "(this matches the CPU RobotContainer convention)."
            )

        self.jnt_adr = int(base_model.body_jntadr[self.obj_id])
        if self.jnt_adr == -1:
            raise ValueError(
                "Body 'object' has no joint attached. The batched motion model "
                "needs a free joint on 'object' to teleport particles in qpos."
            )

        self.qpos_adr = int(base_model.jnt_qposadr[self.jnt_adr])
        self.dof_adr = int(base_model.jnt_dofadr[self.jnt_adr])

    # -- Compatibility shims --------------------------------------------------
    #
    # The CPU RobotContainer exposes ``robots: list``. Some downstream code may
    # still iterate it. We expose a length-1 list whose only element is the
    # batched robot, so accidental iteration is loud (and incorrect-looking,
    # which is the point — the caller should be ported to use ``warp_robot``).

    @property
    def robots(self) -> list[MujocoWarpRobot]:
        """Compatibility shim. Returns ``[self.warp_robot]``.

        Iterating ``container.robots`` will visit the **batched** robot once.
        Any code that depended on visiting N independent robots must be ported
        to operate on batched tensors instead. Treat this as a migration aid,
        not a permanent surface.
        """
        return [self.warp_robot]