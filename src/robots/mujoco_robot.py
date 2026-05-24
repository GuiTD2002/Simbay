import threading
import time

import mujoco
import mujoco.viewer
import numpy as np
from numpy.typing import NDArray

from .base import BaseRobot


class MujocoRobot(BaseRobot):
    def __init__(self, model, data, dt=0.002, viewer=None):
        self.model = model
        self.data = data
        self.viewer: mujoco.viewer.Handle | None = viewer
        self.dt = dt
        self._last_render_time = time.time()
        self._last_step_time = time.perf_counter()
        self._traj_stop: threading.Event | None = None
        self._traj_thread: threading.Thread | None = None

        # 1. Setup Force Sensor
        self.force_sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "hand_force") # type: ignore
        if self.force_sensor_id != -1:
            self.force_adress = model.sensor_adr[self.force_sensor_id]
        else:
            print("[Warning] 'hand_force' sensor not found in XML! Forces will read as 0.0")
            self.force_adress = None

        # 2. Setup Torque Sensor
        self.torque_sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "hand_torque") # type: ignore
        if self.torque_sensor_id != -1:
            self.torque_adress = model.sensor_adr[self.torque_sensor_id]
        else:
            print("[Warning] 'hand_torque' sensor not found in XML! Torques will read as 0.0")
            self.torque_adress = None

        # 3. Cache Site ID Once
        self.ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pinch_site") # type: ignore
        if self.ee_site_id == -1:
            print("[Warning] 'pinch_site' not found! End-effector kinematics will fail.")


    def move_joints(self, pos: NDArray[np.float64]):
        """ 
        Extracts the first 7 elements to control the arm joints, safely ignoring extra elements (e.g., gripper commands).
        """
        self.data.ctrl[:7] = pos[:7]          
        mujoco.mj_step(self.model, self.data)   # type: ignore


    def move_gripper(self, width):
        """Controls the gripper separately."""
        self.data.ctrl[7] = width * 255/0.08  
        # mujoco.mj_step(self.model, self.data)   # type: ignore     


    def get_joints_pos(self) -> NDArray[np.float64]:
        """Returns the current 7 joint angles of the robot."""
        # .copy() prevents accidental corruption of the physics state
        return self.data.qpos[:7].copy()
    
    def get_joints_vel(self) -> NDArray[np.float64]:
        """Returns the current 7 joint velocities of the robot."""
        return self.data.qvel[:7].copy()
    

    def get_ee_pos(self) -> NDArray[np.float64]:
        """Returns the Cartesian (XYZ) position of the pinch site."""
        # Uses the cached ID instead of a string lookup!
        return self.data.site_xpos[self.ee_site_id].copy()
    

    def get_torque_reads(self) -> NDArray[np.float64]:
        """Returns a 3-element array: [Tx, Ty, Tz] in the GLOBAL frame."""
        if self.torque_adress is not None:
            local_torque = self.data.sensordata[self.torque_adress : self.torque_adress + 3]
        else:
            local_torque = np.zeros(3)

        # Rotate local torques into the global frame
        rmat = self.data.site_xmat[self.ee_site_id].reshape(3, 3)
        global_torque = rmat @ local_torque
          
        # Flip the signs so the simulation measures the torque of the object ON the robot
        return -global_torque


    def get_force_reads(self) -> NDArray[np.float64]:
        """Returns a 3-element array: [Fx, Fy, Fz] in the GLOBAL frame."""
        if self.force_adress is not None:
            local_force = self.data.sensordata[self.force_adress : self.force_adress + 3]
        else:
            local_force = np.zeros(3)

        # Rotate local forces into the global frame
        rmat = self.data.site_xmat[self.ee_site_id].reshape(3, 3)
        global_force = rmat @ local_force
          
        # Flip the signs so the simulation measures the force of the object ON the robot
        return -global_force
    

    def sync(self):
        """
        Throttles graphics to 60 FPS AND paces the simulation 
        loop to match real-world time!
        """
        # 1. RENDER PACING (Save CPU, cap at 60 FPS)
        if hasattr(self, 'viewer') and self.viewer is not None:
            current_time = time.time()
            if (current_time - self._last_render_time) > 0.016: 
                self.viewer.sync()
                self._last_render_time = current_time
                
        # 2. TIME PACING (The Windows-Proof Spin Lock!)
        # Calculate exactly when this 1ms budget is supposed to end
        target_time = self._last_step_time + self.dt
        
        # Lock the CPU and wait for the exact nanosecond
        while time.perf_counter() < target_time:
            pass 
            
        # Reset the clock for the next loop
        self._last_step_time = time.perf_counter()

    def move_trajectory_async(self, trajectory, dt2=None):
        """
        Fires a trajectory on a background thread: steps the sim at `dt2`
        and ticks the viewer at ~60 Hz. Returns immediately so the main
        thread can poll measurements / run the filter independently.
        """
        dt2 = dt2 if dt2 is not None else self.dt

        if self._traj_stop is not None:
            self._traj_stop.set()
        if self._traj_thread is not None and self._traj_thread.is_alive():
            self._traj_thread.join()

        stop_evt = threading.Event()
        self._traj_stop = stop_evt

        def _run(traj, period, stop):
            last_render = time.perf_counter()
            for qpos in traj:
                if stop.is_set():
                    return
                target = time.perf_counter() + period
                self.data.ctrl[:7] = qpos[:7]
                mujoco.mj_step(self.model, self.data)  # type: ignore
                now = time.perf_counter()
                if self.viewer is not None and (now - last_render) > 0.016:
                    self.viewer.sync()
                    last_render = now
                while time.perf_counter() < target:
                    if stop.is_set():
                        return

        self._traj_thread = threading.Thread(
            target=_run, args=(trajectory, dt2, stop_evt), daemon=True
        )
        self._traj_thread.start()

    def stop_arm(self):
        """Stop any background trajectory thread started by move_trajectory_async."""
        if self._traj_stop is not None:
            self._traj_stop.set()
        if self._traj_thread is not None and self._traj_thread.is_alive():
            self._traj_thread.join()

    def wait_seconds(self, duration):
        time.sleep(duration)

    def print_object_pos(self):
        block_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'object') # type: ignore
        x_pos, y_pos, z_pos = self.data.xpos[block_id]
        print(f"Object Position -> X: {x_pos:.3f}, Y: {y_pos:.3f}, Z: {z_pos:.3f}")
