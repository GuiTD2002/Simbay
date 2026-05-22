from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.robots import BaseRobot
from src.utils.constants import DEFAULT_OBJECT_PROPS
from src.utils.constants import FRANKA_HOME_QPOS

try:
    import mujoco
    import mujoco_warp as mjw
    import warp as wp

    MJWARP_AVAILABLE = True
except ImportError:
    mujoco = None
    mjw = None
    wp = None
    MJWARP_AVAILABLE = False


class MujocoWarpRobot(BaseRobot):
    """
    Robot adapter backed by MuJoCo Warp.

    MJWarp keeps physics state on device. This class mirrors one selected world
    back into a host ``mujoco.MjData`` object when state or sensor reads are
    requested, while all simulation steps are executed through ``mjw.step``.
    """

    def __init__(
        self,
        model: Any | None = None,
        data: Any | None = None,
        *,
        xml_path: str | os.PathLike[str] | None = None,
        object_properties: dict[str, Any] | None = DEFAULT_OBJECT_PROPS,
        dt: float = 0.002,
        nworld: int = 1,
        world_id: int = 0,
        nconmax: int | None = None,
        nccdmax: int | None = None,
        njmax: int | None = 106,
        njmax_nnz: int | None = None,
        naconmax: int | None = None,
        naccdmax: int | None = None,
        device: str | None = None,
    ) -> None:
        _require_mjwarp()

        if nworld < 1:
            raise ValueError("nworld must be at least 1")
        if not 0 <= world_id < nworld:
            raise ValueError("world_id must refer to one of the simulated worlds")

        if device is not None:
            wp.set_device(device)  # type: ignore[union-attr]

        if model is None:
            model, data = _load_default_model(xml_path, object_properties, dt)
        elif data is None:
            data = mujoco.MjData(model)  # type: ignore[union-attr]
            data.qpos[:7] = FRANKA_HOME_QPOS
            data.ctrl[:7] = FRANKA_HOME_QPOS
            mujoco.mj_forward(model, data)  # type: ignore[union-attr]

        self.model = model
        self.data = data
        self.dt = dt
        self.nworld = nworld
        self.world_id = world_id
        self.device = device
        self.viewer = None

        self.warp_model = mjw.put_model(self.model)  # type: ignore[union-attr]
        self.warp_data = mjw.put_data(  # type: ignore[union-attr]
            self.model,
            self.data,
            nworld=nworld,
            nconmax=nconmax,
            nccdmax=nccdmax,
            njmax=njmax,
            njmax_nnz=njmax_nnz,
            naconmax=naconmax,
            naccdmax=naccdmax,
        )

        self._ctrl = np.repeat(np.asarray(self.data.ctrl, dtype=np.float32)[None, :], nworld, axis=0)
        self._copy_to_device(self.warp_data.ctrl, self._ctrl)
        self._host_dirty = False
        self._last_sync_time = time.perf_counter()

        self.force_sensor_id = mujoco.mj_name2id(  # type: ignore[union-attr]
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, "hand_force"  # type: ignore[union-attr]
        )
        self.force_address = self._sensor_address(self.force_sensor_id, "hand_force")

        self.torque_sensor_id = mujoco.mj_name2id(  # type: ignore[union-attr]
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, "hand_torque"  # type: ignore[union-attr]
        )
        self.torque_address = self._sensor_address(self.torque_sensor_id, "hand_torque")

        self.ee_site_id = mujoco.mj_name2id(  # type: ignore[union-attr]
            self.model, mujoco.mjtObj.mjOBJ_SITE, "pinch_site"  # type: ignore[union-attr]
        )
        if self.ee_site_id == -1:
            print("[Warning] 'pinch_site' not found. End-effector reads will return zeros.")

    def move_joints(self, pos: NDArray[np.float64]) -> None:
        arm_width = min(7, self.model.nu)
        self._ctrl[:, :arm_width] = np.asarray(pos[:arm_width], dtype=np.float32)
        self._set_host_ctrl()
        self._copy_to_device(self.warp_data.ctrl, self._ctrl)
        self.step()

    def move_gripper(self, width: float) -> None:
        if self.model.nu <= 7:
            return

        self._ctrl[:, 7] = np.float32(width * 255 / 0.08)
        self._set_host_ctrl()
        self._copy_to_device(self.warp_data.ctrl, self._ctrl)

    def step(self, n_steps: int = 1) -> None:
        for _ in range(n_steps):
            mjw.step(self.warp_model, self.warp_data)  # type: ignore[union-attr]
        self._host_dirty = True

    def sync_host(self) -> Any:
        if self._host_dirty:
            if hasattr(wp, "synchronize"):
                wp.synchronize()  # type: ignore[union-attr]
            mjw.get_data_into(self.data, self.model, self.warp_data, world_id=self.world_id)  # type: ignore[union-attr]
            self._host_dirty = False
        return self.data

    def get_joints_pos(self) -> NDArray[np.float64]:
        self.sync_host()
        return self.data.qpos[:7].copy()

    def get_joints_vel(self) -> NDArray[np.float64]:
        self.sync_host()
        return self.data.qvel[:7].copy()

    def get_ee_pos(self) -> NDArray[np.float64]:
        self.sync_host()
        if self.ee_site_id == -1:
            return np.zeros(3)
        return self.data.site_xpos[self.ee_site_id].copy()

    def get_torque_reads(self) -> NDArray[np.float64]:
        self.sync_host()
        local_torque = self._sensor_read(self.torque_address)
        return -(self._site_rotation() @ local_torque)

    def get_force_reads(self) -> NDArray[np.float64]:
        self.sync_host()
        local_force = self._sensor_read(self.force_address)
        return -(self._site_rotation() @ local_force)

    def sync(self) -> None:
        if self.viewer is not None:
            self.sync_host()
            self.viewer.sync()

        target_time = self._last_sync_time + self.dt
        while time.perf_counter() < target_time:
            pass
        self._last_sync_time = time.perf_counter()

    def move_trajectory(self, trajectory: list[NDArray[np.float64]], dt2: float | None = None) -> None:
        for pos in trajectory:
            self.move_joints(pos)
            self.sync()

    def move_trajectory_async(self, trajectory: list[NDArray[np.float64]], dt2: float | None = None) -> None:
        self.move_trajectory(trajectory, dt2)

    def wait_seconds(self, duration: float) -> None:
        n_steps = max(1, int(round(duration / self.dt)))
        for _ in range(n_steps):
            self.step()
            self.sync()

    def stop_arm(self) -> None:
        self.move_joints(self.get_joints_pos())

    def print_object_pos(self) -> None:
        self.sync_host()
        block_id = mujoco.mj_name2id(  # type: ignore[union-attr]
            self.model, mujoco.mjtObj.mjOBJ_BODY, "object"  # type: ignore[union-attr]
        )
        if block_id == -1:
            print("Object body not found.")
            return
        x_pos, y_pos, z_pos = self.data.xpos[block_id]
        print(f"Object Position -> X: {x_pos:.3f}, Y: {y_pos:.3f}, Z: {z_pos:.3f}")

    def _sensor_address(self, sensor_id: int, name: str) -> int | None:
        if sensor_id == -1:
            print(f"[Warning] '{name}' sensor not found. Reads will return zeros.")
            return None
        return int(self.model.sensor_adr[sensor_id])

    def _sensor_read(self, address: int | None) -> NDArray[np.float64]:
        if address is None:
            return np.zeros(3)
        return self.data.sensordata[address : address + 3].copy()

    def _site_rotation(self) -> NDArray[np.float64]:
        if self.ee_site_id == -1:
            return np.eye(3)
        return self.data.site_xmat[self.ee_site_id].reshape(3, 3)

    def _set_host_ctrl(self) -> None:
        self.data.ctrl[:] = self._ctrl[self.world_id]

    def _copy_to_device(self, destination: Any, values: NDArray[np.float32]) -> None:
        kwargs = {"dtype": wp.float32}  # type: ignore[union-attr]
        if self.device is not None:
            kwargs["device"] = self.device
        source = wp.array(values, **kwargs)  # type: ignore[union-attr]
        wp.copy(destination, source)  # type: ignore[union-attr]


def initialize_mujoco_warp_env(
    object_properties: dict[str, Any] | None = DEFAULT_OBJECT_PROPS,
    dt: float = 0.002,
    **kwargs: Any,
) -> MujocoWarpRobot:
    return MujocoWarpRobot(object_properties=object_properties, dt=dt, **kwargs)


def _load_default_model(
    xml_path: str | os.PathLike[str] | None,
    object_properties: dict[str, Any] | None,
    dt: float,
) -> tuple[Any, Any]:
    from src.utils.mujoco_utils import load_mujoco_model
    from src.utils.mujoco_utils import modify_object_properties

    resolved_path = os.fspath(xml_path or os.path.join("models", "scene.xml"))
    model, data = load_mujoco_model(resolved_path)
    model.opt.timestep = dt

    if object_properties is not None:
        modify_object_properties(model, data, "object", object_properties)

    return model, data


def _require_mjwarp() -> None:
    if MJWARP_AVAILABLE:
        return

    raise ImportError(
        "MujocoWarpRobot requires optional dependencies: mujoco, mujoco-warp, and warp. "
        "Install them with `pip install mujoco mujoco-warp` on a machine with a supported "
        "NVIDIA CUDA setup, or use Warp's CPU device for deterministic local checks."
    )
