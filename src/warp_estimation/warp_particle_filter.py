"""
Warp particle-filter builders.

The normal path builds the filter in the current process. The Ray path keeps
the same small particle-filter interface but runs the Warp filter in a Ray
actor, which is useful when you want GPU process isolation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.estimation.particle_filter import ParticleFilterRegularized


def build_warp_particle_filter(
    num_particles: int,
    limits: tuple[Any, Any],
    object_props: dict[str, Any],
    dt: float,
    ess_threshold: float,
    *,
    nconmax: int | None = None,
    njmax: int | None = None,
    device: str | None = "cuda:0",
) -> ParticleFilterRegularized:
    """Build a Warp-backed particle filter in the current process."""
    # Imported lazily so the Ray builder doesn't drag warp/mujoco_warp (and their
    # CUDA driver init) into the local process when the filter only runs remotely.
    from src.warp_estimation.warp_container import WarpRobotContainer
    from src.warp_estimation.warp_measurement import WarpBinaryContactMeasurementModel
    from src.warp_estimation.warp_motion import WarpPositionMotionModel

    container = WarpRobotContainer(
        num_particles=num_particles,
        props=object_props,
        dt=dt,
        nconmax=nconmax,
        njmax=njmax,
        device=device,
    )

    return ParticleFilterRegularized(
        num_particles=num_particles,
        state_bounds=limits,
        motion_model=WarpPositionMotionModel(container),
        measurement_model=WarpBinaryContactMeasurementModel(container),
        ess_threshold_ratio=ess_threshold,
    )


def build_ray_warp_particle_filter(
    num_particles: int,
    limits: tuple[Any, Any],
    object_props: dict[str, Any],
    dt: float,
    ess_threshold: float,
    *,
    nconmax: int | None = None,
    njmax: int | None = None,
    device: str | None = "cuda:0",
    num_gpus: float = 1.0,
    ray_address: str | None = None,
    debug: bool = True,
) -> "RayWarpParticleFilter":
    """Build a Warp particle filter inside a Ray actor."""
    _ray_log(debug, "loading Ray")
    ray = _load_ray()

    owns_ray = False
    if not ray.is_initialized():
        target = ray_address or "local runtime"
        _ray_log(debug, f"initializing Ray connection ({target})")
        context = ray.init(address=ray_address, ignore_reinit_error=True)
        owns_ray = True
        address = _ray_address(context, target)
        _ray_log(debug, f"connected to Ray at {address}")
    else:
        _ray_log(debug, "using existing Ray runtime")

    _ray_log(
        debug,
        f"creating Warp particle-filter actor (num_gpus={num_gpus}, device={device})",
    )
    actor_class = ray.remote(num_gpus=num_gpus)(_WarpParticleFilterActor)
    actor = actor_class.remote(
        num_particles=num_particles,
        limits=limits,
        object_props=object_props,
        dt=dt,
        ess_threshold=ess_threshold,
        nconmax=nconmax,
        njmax=njmax,
        device=device,
        debug=debug,
    )
    particle_filter = RayWarpParticleFilter(
        ray,
        actor,
        debug=debug,
        owns_ray=owns_ray,
    )
    _ray_log(debug, f"actor ready with {particle_filter.N} particles")
    return particle_filter


class RayWarpParticleFilter:
    """Synchronous proxy for a Warp particle filter running in Ray."""

    def __init__(
        self,
        ray: Any,
        actor: Any,
        debug: bool = False,
        owns_ray: bool = False,
    ) -> None:
        self._ray = ray
        self._actor = actor
        self._debug = debug
        self._owns_ray = owns_ray
        self.N = 0
        self.ess_threshold_ratio = 0.0
        self.particles = np.empty((0, 0))
        self.weights = np.empty(0)
        self.history: dict[str, list[np.ndarray]] = {
            "particles": [],
            "estimates": [],
            "weights": [],
        }
        snapshot = self._request_snapshot("startup")
        self._sync(snapshot)

    def update_internal_state(self, state: dict[str, Any]) -> None:
        snapshot = self._ray.get(self._actor.update_internal_state.remote(state))
        self._sync(snapshot)

    def predict(self, control_input: dict[str, Any]) -> None:
        snapshot = self._ray.get(self._actor.predict.remote(control_input))
        self._sync(snapshot)

    def step(
        self,
        control_input: dict[str, Any],
        observation: dict[str, Any],
        current_state: dict[str, Any],
    ) -> None:
        snapshot = self._ray.get(
            self._actor.step.remote(control_input, observation, current_state)
        )
        self._sync(snapshot)

    def record_state(self) -> None:
        snapshot = self._ray.get(self._actor.record_state.remote())
        self._sync(snapshot)

    def estimate(self) -> np.ndarray:
        return self._ray.get(self._actor.estimate.remote())

    def reset(self, state: dict[str, Any]) -> None:
        snapshot = self._ray.get(self._actor.reset.remote(state))
        self._sync(snapshot)

    def close(self) -> None:
        _ray_log(self._debug, "stopping Ray particle-filter actor")
        self._ray.kill(self._actor)
        if self._owns_ray:
            _ray_log(self._debug, "shutting down Ray connection")
            self._ray.shutdown()

    def _sync(self, snapshot: dict[str, Any]) -> None:
        self.N = snapshot["N"]
        self.ess_threshold_ratio = snapshot["ess_threshold_ratio"]
        self.particles = snapshot["particles"]
        self.weights = snapshot["weights"]
        self.history = snapshot["history"]

    def _request_snapshot(self, label: str) -> dict[str, Any]:
        snapshot = self._ray.get(self._actor.snapshot.remote())
        return snapshot


class _WarpParticleFilterActor:
    def __init__(self, *args: Any, debug: bool = False, **kwargs: Any) -> None:
        self._debug = debug
        if debug:
            _ray_worker_log(debug, "building Warp particle filter")
        self.particle_filter = build_warp_particle_filter(*args, **kwargs)
        if debug:
            _ray_worker_log(debug, "Warp particle filter ready")

    def update_internal_state(self, state: dict[str, Any]) -> dict[str, Any]:
        self.particle_filter.update_internal_state(state)
        return self._snapshot()

    def predict(self, control_input: dict[str, Any]) -> dict[str, Any]:
        self.particle_filter.predict(control_input)
        return self._snapshot()

    def step(
        self,
        control_input: dict[str, Any],
        observation: dict[str, Any],
        current_state: dict[str, Any],
    ) -> dict[str, Any]:
        self.particle_filter.step(control_input, observation, current_state)
        return self._snapshot()

    def record_state(self) -> dict[str, Any]:
        self.particle_filter.record_state()
        return self._snapshot()

    def estimate(self) -> np.ndarray:
        return self.particle_filter.estimate()

    def reset(self, state: dict[str, Any]) -> dict[str, Any]:
        self.particle_filter.reset(state)
        return self._snapshot()

    def snapshot(self) -> dict[str, Any]:
        return self._snapshot()

    def _snapshot(self) -> dict[str, Any]:
        return {
            "N": self.particle_filter.N,
            "ess_threshold_ratio": self.particle_filter.ess_threshold_ratio,
            "particles": self.particle_filter.particles,
            "weights": self.particle_filter.weights,
            "history": self.particle_filter.history,
        }


def _load_ray() -> Any:
    try:
        import ray
    except ImportError as exc:
        raise RuntimeError(
            "Ray is not installed. Install ray or set USE_RAY = False."
        ) from exc
    return ray


def _ray_log(enabled: bool, message: str) -> None:
    if enabled:
        print(f"[Ray ⚡️] {message}", flush=True)


def _ray_worker_log(enabled: bool, message: str) -> None:
    if enabled:
        print(f"[Ray worker ⚡️] {message}", flush=True)


def _summarize(value: Any) -> str:
    if isinstance(value, np.ndarray):
        return f"ndarray(shape={value.shape}, dtype={value.dtype})"
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            parts.append(f"{key}={_summarize(item)}")
        return "{" + ", ".join(parts) + "}"
    if isinstance(value, (list, tuple)):
        return f"{type(value).__name__}(len={len(value)})"
    return repr(value)


def _ray_address(context: Any, fallback: str) -> str:
    address_info = getattr(context, "address_info", None)
    if not address_info:
        return fallback
    return (
        address_info.get("address")
        or address_info.get("gcs_address")
        or address_info.get("redis_address")
        or fallback
    )
