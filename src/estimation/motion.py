from abc import ABC
from abc import abstractmethod

import mujoco
import numpy as np

from .container import RobotContainer


class BaseMotionModel(ABC): 
    @abstractmethod
    def propagate(self, particles: np.ndarray, control_input: dict) -> np.ndarray:
        ...

    @abstractmethod
    def change_internal_state(self, particles: np.ndarray, real_state: dict) -> None:
        """Forces the model's internal state to match the provided particles."""
        ...


class PositionMotionModel(BaseMotionModel):
    def __init__(self, container: RobotContainer):
        self.container = container
    
    def propagate(self, particles, control_input):
        qpos          = control_input['joints']
        gripper_width = control_input['gripper']

        for robot in self.container.robots: 
            robot.move_gripper(gripper_width)
            robot.move_joints(qpos)
            
            # VISUAL DEBUG: Update the screen if a viewer is attached to this internal robot
            if getattr(robot, 'viewer', None) is not None:
                robot.viewer.sync()
            
        return particles
 
    
    def change_internal_state(self, particles, real_state):
        qpos = real_state['qpos']
        qvel = real_state['qvel']

        # 1. Detect the dimension of the filter outside the loop
        dim = particles.shape[1] if particles.ndim > 1 else 1

        for i, robot in enumerate(self.container.robots):
            
            # Sync the internal robot's arm with the real world
            robot.data.qpos[:len(qpos)] = qpos
            robot.data.qvel[:len(qvel)] = qvel
            
            # ==========================================
            # 2. DIMENSION-AGNOSTIC QPOS MAPPING
            # ==========================================
            if dim == 1:
                # 1D (Y-Axis Only): Write specifically to the Y index (qpos_adr + 1)
                # The X index (qpos_adr) stays at its default physical position.
                robot.data.qpos[self.container.qpos_adr + 1] = particles[i, 0] if particles.ndim > 1 else particles[i]
                
            elif dim == 2:
                # 2D (X, Y): Overwrite both X and Y simultaneously
                robot.data.qpos[self.container.qpos_adr : self.container.qpos_adr + 2] = particles[i]
                
            elif dim == 3:
                # 3D (X, Y, Theta): Write X and Y, then convert Theta to a Quaternion
                robot.data.qpos[self.container.qpos_adr : self.container.qpos_adr + 2] = particles[i, :2]
                
                # Convert Theta (rotation around Z-axis) to [qw, qx, qy, qz]
                theta = particles[i, 2]
                robot.data.qpos[self.container.qpos_adr + 3 : self.container.qpos_adr + 7] = [
                    np.cos(theta / 2.0), 0.0, 0.0, np.sin(theta / 2.0)
                ]
            
            # Zero out object's velocities so it doesn't glide in the virtual world
            robot.data.qvel[self.container.dof_adr : self.container.dof_adr+6] = 0.0
            
            mujoco.mj_forward(robot.model, robot.data) # type: ignore
            
            # VISUAL DEBUG: Update the screen instantly when the block teleports
            if getattr(robot, 'viewer', None) is not None:
                robot.viewer.sync()
        
    

class MassMotionModel(BaseMotionModel):
    def __init__(self, container: RobotContainer):
        self.container = container

    def propagate(self, particles, control_input):
        qpos          = control_input['joints']
        gripper_width = control_input['gripper']

        for robot in self.container.robots: 
            robot.move_gripper(gripper_width)
            robot.move_joints(qpos)

            # VISUAL DEBUG: Update the screen if a viewer is attached to this internal robot
            if getattr(robot, 'viewer', None) is not None:
                robot.viewer.sync()

        return particles
    
    def change_internal_state(self, particles, real_state):
        if real_state is not None:
            qpos = real_state.get('qpos', [])
            qvel = real_state.get('qvel', [])
            
        for i, robot in enumerate(self.container.robots):
            mass_val = particles[i, 0] if particles.ndim == 2 else particles[i]
            
            robot.model.body_mass[self.container.obj_id] = mass_val
            
            # --- REBUILD PHYSICAL CONSTANTS ---
            # Because we altered `robot.model` instead of `robot.data`, we must tell 
            # MuJoCo to recalculate the inertia tensor. WARNING: This function resets 
            # data.qpos back to the XML default spawn location!
            saved_ctrl = robot.data.ctrl.copy()
            mujoco.mj_setConst(robot.model, robot.data) # type: ignore
            
            # Overwrite the state AFTER mj_setConst so the robot and object stay in their real positions
            if real_state is not None and len(qpos) > 0:
                robot.data.qpos[:len(qpos)] = qpos
                robot.data.qvel[:len(qvel)] = qvel
            robot.data.ctrl[:] = saved_ctrl

            # Recalculate kinematics just to be safe
            mujoco.mj_forward(robot.model, robot.data)  # type: ignore
    
    
    
class KinematicMotionModel(BaseMotionModel):
    def propagate(self, particles, control_input):
        """
        The object is static, and jitter is handled by the Regularized Resampler.
        We simply return the particles exactly as they are.
        """
        return particles
    
    def change_internal_state(self, particles, real_state):
        pass
