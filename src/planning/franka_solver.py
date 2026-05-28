import os

import mujoco
import numpy as np

from src.kinematics import MujocoPoseIK
from src.kinematics import StepMethods
from src.kinematics import solve_IKProblem
from src.utils import load_mujoco_model


class FrankaKinematics:
    _model = None
    _data = None
    _solver_instance: MujocoPoseIK
    _ee_site_name = "gripper" # Update if your FR3v2 XML uses 'hand_tcp'

    @classmethod
    def _initialize(cls):
        if cls._model is None:
            xml_path = os.path.join("models", "panda_nohand.xml")
            if not os.path.exists(xml_path):
                raise FileNotFoundError(f"Franka XML not found at {xml_path}")
            
            cls._model, cls._data = load_mujoco_model(xml_path)
            cls._solver_instance = MujocoPoseIK(cls._model, cls._data, cls._ee_site_name, StepMethods.SDLS)

    @classmethod
    def inverse(cls, start_joints, target_pos, target_quat, step_method=StepMethods.SDLS, tolerance=1e-6, max_iterations=500):
        """
        Inverse Kinematics: Calculates the required joints to reach a target XYZ + Quat.
        """
        cls._initialize()

        if cls._model is None or cls._data is None:
            raise RuntimeError("MuJoCo model failed to initialize!")
        
        cls._solver_instance.step_method = step_method

        target_pose = np.concatenate([target_pos, target_quat])

        # Solve using current_joints as the mathematical seed
        return solve_IKProblem(
            cls._solver_instance, 
            start_joints,
            target_pose, 
            tol=tolerance, 
            max_iter=max_iterations
        )

    @classmethod
    def forward(cls, joints):
        """
        Forward Kinematics: Calculates the exact XYZ and Quat of the end-effector 
        based on the provided joint angles.
        """
        cls._initialize()

        if cls._model is None or cls._data is None:
            raise RuntimeError("MuJoCo model failed to initialize!")
        
        # 1. Inject the joints into our cached mathematical model
        # Assuming the arm takes up the first 7 slots in qpos
        cls._data.qpos[:7] = joints 
        
        # 2. Update the spatial kinematics (Extremely fast, ignores physics/gravity)
        mujoco.mj_kinematics(cls._model, cls._data) # type: ignore
        
        # 3. Extract the End-Effector Position
        site_id = mujoco.mj_name2id(cls._model, mujoco.mjtObj.mjOBJ_SITE, cls._ee_site_name) # type: ignore
        pos = cls._data.site_xpos[site_id].copy()
        
        # 4. Extract Orientation (MuJoCo stores this as a 3x3 rotation matrix, convert to Quat)
        mat = cls._data.site_xmat[site_id]
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, mat) # type: ignore
        
        return pos, quat  

    @staticmethod
    def is_reachable(target_pos: np.ndarray, base_pos: np.ndarray = np.array([0.0, 0.0, 0.0])) -> bool:
        """
        Fast-reject geometric check to see if a Cartesian point is physically 
        within the robot's reachable workspace.
        """
        # Franka physical limits (in meters)
        MAX_REACH = 0.85  # Slightly less than true 0.855m for safety margin
        MIN_REACH = 0.15  # The robot's own base pillar
        MIN_Z = 0.0       # The floor / table surface

        # 1. Z-Height Check (Prevent smashing through the floor)
        if target_pos[2] < MIN_Z:
            print(f"Reachability Check Failed: Target Z ({target_pos[2]:.3f}) is below floor limit.")
            return False

        # 2. Distance Check (Spherical bounds)
        distance_from_base = np.linalg.norm(target_pos - base_pos)

        if distance_from_base > MAX_REACH:
            print(f"Reachability Check Failed: Target ({distance_from_base:.3f}m) exceeds max reach ({MAX_REACH}m).")
            return False
            
        if distance_from_base < MIN_REACH:
            print(f"Reachability Check Failed: Target ({distance_from_base:.3f}m) is inside the base pillar.")
            return False

        return True      
