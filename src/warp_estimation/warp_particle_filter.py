"""
Warp particle-filter builders.

The normal path builds the filter in the current process. The Ray path keeps
the same small particle-filter interface but **shards the swarm across K Ray
actors**, each on its own slice of the GPU. The proxy mirrors the global
particle/weight vectors and runs all cross-shard math (normalize, ESS, cov,
cumsum, jitter, clip) driver-side; shards only run per-particle GPU work
(``mjw.step`` via the motion model, ``compute_likelihoods`` via the measurement
model).
"""

from __future__ import annotations

import logging
import math
import os
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
    nccdmax: int | None = None,
    naccdmax: int | None = None,
    ccd_iterations: int | None = None,
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
        nccdmax=nccdmax,
        naccdmax=naccdmax,
        ccd_iterations=ccd_iterations,
        device=device,
    )

    pf = ParticleFilterRegularized(
        num_particles=num_particles,
        state_bounds=limits,
        motion_model=WarpPositionMotionModel(container),
        measurement_model=WarpBinaryContactMeasurementModel(container),
        ess_threshold_ratio=ess_threshold,
    )

    # force lazy jit compilation so the first real sweep is faster (compilation cost)
    container.warp_robot.step(1)
    container.warp_robot.sync_host()
    return pf


def build_ray_warp_particle_filter(
    num_particles: int,
    limits: tuple[Any, Any],
    object_props: dict[str, Any],
    dt: float,
    ess_threshold: float,
    *,
    nconmax: int | None = None,
    njmax: int | None = None,
    nccdmax: int | None = None,
    naccdmax: int | None = None,
    ccd_iterations: int | None = None,
    device: str | None = "cuda:0",
    particles_per_actor: int = 500,
    num_gpus_per_actor: float = 0.25,
    num_cpus_per_actor: float = 1.0,
    ray_address: str | None = None,
    runtime_env: dict[str, Any] | None = None,
    debug: bool = True,
) -> RayWarpParticleFilter:
    """Build a Warp particle filter sharded across K Ray actors.

    ``K = ceil(num_particles / particles_per_actor)``. The last shard takes
    the remainder, e.g. ``num_particles=1200, particles_per_actor=500`` →
    ``sizes=[500, 500, 200]``.
    """
    _ray_log(debug, "loading Ray")
    ray = _load_ray()

    # The stable Docker image deliberately omits src/scripts, so we ship the
    # local repo to the cluster as the job's working_dir. Caller can override
    # via runtime_env.
    if runtime_env is None:
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
        )
        runtime_env = {
            "working_dir": repo_root,
            # Note: keep `models/` in the upload — MuJoCo XML/meshes are loaded
            # via the relative path "models/scene.xml" from CWD on the worker.
            "excludes": [
                ".git", ".venv", "__pycache__", "saved_plots",
                "outputs", "*.png", "*.mp4",
            ],
        }

    owns_ray = False
    if not ray.is_initialized():
        target = ray_address or "local runtime"
        _ray_log(debug, f"initializing Ray connection ({target})")
        context = ray.init(
            log_to_driver=debug, # see remote logs
            address=ray_address,
            ignore_reinit_error=True,
            runtime_env=runtime_env,
            logging_level=logging.INFO
        )
        owns_ray = True
        address = _ray_address(context, target)
        _ray_log(debug, f"connected to Ray at {address}")
    else:
        _ray_log(debug, "using existing Ray runtime")

    K = math.ceil(num_particles / particles_per_actor)
    sizes = [particles_per_actor] * (K - 1) + [num_particles - (K - 1) * particles_per_actor]

    _ray_log(
        debug,
        f"creating {K} Warp particle-filter actor(s) "
        f"(sizes={sizes}, num_cpus_per_actor={num_cpus_per_actor}, "
        f"num_gpus_per_actor={num_gpus_per_actor}, device={device})",
    )
    actor_cls = ray.remote(
        num_cpus=num_cpus_per_actor, num_gpus=num_gpus_per_actor
    )(_WarpParticleFilterActor)
    actors = [
        actor_cls.remote(
            num_particles=size,
            limits=limits,
            object_props=object_props,
            dt=dt,
            ess_threshold=ess_threshold,
            nconmax=nconmax,
            njmax=njmax,
            nccdmax=nccdmax,
            naccdmax=naccdmax,
            ccd_iterations=ccd_iterations,
            device=device,
            debug=debug,
        )
        for size in sizes
    ]

    particle_filter = RayWarpParticleFilter(
        ray,
        actors,
        sizes,
        limits,
        debug=debug,
        owns_ray=owns_ray,
    )
    _ray_log(debug, f"{K} actor(s) ready with {particle_filter.N} total particles")
    return particle_filter


class RayWarpParticleFilter:
    """Synchronous proxy that fans out across K Warp-particle-filter actors.

    The proxy mirrors the global ``particles``/``weights`` and runs all
    cross-shard math (normalize, ESS, covariance, cumsum, jitter, clip) on the
    driver. Shards only do per-particle GPU work via two RPCs: ``predict``
    (motion model step) and ``compute_likelihoods`` (measurement model).
    """

    def __init__(
        self,
        ray: Any,
        actors: list[Any],
        sizes: list[int],
        limits: tuple[Any, Any],
        debug: bool = False,
        owns_ray: bool = False,
    ) -> None:
        self._ray = ray
        self._actors = list(actors)
        self._sizes = list(sizes)
        self._debug = debug
        self._owns_ray = owns_ray
        self.N = sum(self._sizes)
        self.min_bound = np.atleast_1d(limits[0])
        self.max_bound = np.atleast_1d(limits[1])

        snapshots = self._ray.get([a.snapshot.remote() for a in self._actors])
        self.ess_threshold_ratio = snapshots[0]["ess_threshold_ratio"]
        self.particles = np.concatenate([s["particles"] for s in snapshots], axis=0)
        self.weights = np.ones(self.N) / self.N
        self.history: dict[str, list[np.ndarray]] = {
            "particles": [],
            "estimates": [],
            "weights": [],
        }

    def _scatter(self, arr: np.ndarray) -> list[np.ndarray]:
        return np.split(arr, np.cumsum(self._sizes)[:-1])

    def update_internal_state(self, state: dict[str, Any]) -> None:
        self._ray.get([a.update_internal_state.remote(state) for a in self._actors])

    def predict(self, control_input: dict[str, Any]) -> None:
        # Each shard runs its own mjw step. Particles are book-keeping for a
        # position filter (physics state lives inside Warp's qpos), so the
        # local mirror does not need to be re-fetched after predict.
        self._ray.get([a.predict.remote(control_input) for a in self._actors])

    def update(self, observation: dict[str, Any]) -> None:
        # Body mirrors ParticleFilterRegularized.update; only the likelihood
        # gather is sharded.
        likelihood_slices = self._ray.get(
            [a.compute_likelihoods.remote(observation) for a in self._actors]
        )
        likelihoods = np.concatenate(likelihood_slices)

        new_weights = self.weights * likelihoods

        if observation.get('contact', 0) == 1:
            max_weight = new_weights.max()
            perfect_mask = (
                (likelihoods >= 0.99)
                & (new_weights >= max_weight * 0.99)
                & (max_weight > 0)
            )
            num_perfect = int(perfect_mask.sum())
            if num_perfect > 0:
                print(f"🎯 BULLSEYE: Found {num_perfect} TRUE perfect particle(s) with no past penalties!")
            else:
                print(f"📉 No true perfect particles. Best surviving weight: {max_weight:.2e}")

        self.weights = new_weights
        sum_weights = self.weights.sum()
        if sum_weights == 0.0 or not np.isfinite(sum_weights):
            print("⚠️ CRITICAL WARNING: Particle Extinction Event! All weights collapsed to 0.0. Forcing a uniform reset.")
            self.weights = np.full_like(self.weights, 1.0 / self.N)
        else:
            self.weights /= sum_weights

    def resample(self, current_state: dict[str, Any]) -> None:
        # Body mirrors ParticleFilterRegularized.resample; only the final
        # change_internal_state is sharded.
        Neff = 1.0 / np.sum(self.weights ** 2)
        if Neff >= self.N * self.ess_threshold_ratio:
            return

        nx = self.particles.shape[1]
        S_k = np.cov(self.particles.T, aweights=self.weights, bias=True)
        S_k = np.atleast_2d(S_k)
        S_k += np.eye(nx) * 1e-8
        D = np.linalg.cholesky(S_k)

        u = np.random.rand()
        positions = (np.arange(self.N) + u) / self.N
        cumulative_sum = np.cumsum(self.weights)
        cumulative_sum[-1] = 1.0
        indexes = np.searchsorted(cumulative_sum, positions, side='right')
        resampled_particles = self.particles[indexes]

        A = (4.0 / (nx + 2.0)) ** (1.0 / (nx + 4.0))
        h_opt = A * (self.N ** (-1.0 / (nx + 4.0)))
        epsilon = np.random.randn(self.N, nx)
        jitter = h_opt * (D @ epsilon.T).T
        new_particles = resampled_particles + jitter
        new_particles = np.clip(new_particles, self.min_bound, self.max_bound)

        slices = self._scatter(new_particles)
        self._ray.get([
            a.set_particles.remote(s, current_state)
            for a, s in zip(self._actors, slices)
        ])

        self.particles = new_particles
        self.weights = np.ones(self.N) / self.N

    def step(
        self,
        control_input: dict[str, Any],
        observation: dict[str, Any],
        current_state: dict[str, Any],
    ) -> None:
        self.predict(control_input)
        self.update(observation)
        self.resample(current_state)

    def record_state(self) -> None:
        self.history['particles'].append(self.particles.copy())
        self.history['estimates'].append(self.estimate())
        self.history['weights'].append(self.weights.copy())

    def estimate(self) -> np.ndarray:
        return np.average(self.particles, weights=self.weights, axis=0)

    def reset(self, state: dict[str, Any]) -> None:
        snapshots = self._ray.get([a.reset.remote(state) for a in self._actors])
        self.particles = np.concatenate([s["particles"] for s in snapshots], axis=0)
        self.weights = np.ones(self.N) / self.N

    def close(self) -> None:
        _ray_log(
            self._debug,
            f"stopping {len(self._actors)} Ray particle-filter actor(s)",
        )
        for a in self._actors:
            self._ray.kill(a)
        if self._owns_ray:
            _ray_log(self._debug, "shutting down Ray connection")
            self._ray.shutdown()

    def get_history(self) -> dict[str, list[np.ndarray]]:
        return self.history


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

    def compute_likelihoods(self, observation: dict[str, Any]) -> np.ndarray:
        return self.particle_filter.measurement_model.compute_likelihoods(
            self.particle_filter.particles, observation
        )

    def set_particles(
        self,
        particles_slice: np.ndarray,
        current_state: dict[str, Any],
    ) -> dict[str, Any]:
        self.particle_filter.particles = particles_slice
        self.particle_filter.motion_model.change_internal_state(
            particles_slice, current_state
        )
        return self._snapshot()

    def reset(self, state: dict[str, Any]) -> dict[str, Any]:
        self.particle_filter.reset(state)
        return self._snapshot()

    def snapshot(self, include_history: bool = False) -> dict[str, Any]:
        return self._snapshot(include_history=include_history)

    def _snapshot(self, include_history: bool = False) -> dict[str, Any]:
        snapshot: dict[str, Any] = {
            "N": self.particle_filter.N,
            "ess_threshold_ratio": self.particle_filter.ess_threshold_ratio,
            "particles": self.particle_filter.particles,
            "weights": self.particle_filter.weights,
        }
        if include_history:
            snapshot["history"] = self.particle_filter.history
        return snapshot


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
