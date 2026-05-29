import matplotlib.pyplot as plt
import numpy as np

from src.planning import FrankaKinematics
from src.planning import plan_joints_trajectory
from src.robots import RealRobot
from src.skills import move_to_home
from src.utils import DEFAULT_OBJECT_PROPS
from src.utils import FRANKA_HOME_QPOS

# ==========================================
# 1. SETUP
# ==========================================
dt = 0.001  

real_robot = RealRobot(dt=dt)   
obj_pos = DEFAULT_OBJECT_PROPS['pos']
initial_pos = real_robot.get_joints_pos() 
print(initial_pos)
print(FRANKA_HOME_QPOS)


move_to_home(real_robot) 


# ==========================================
# 2. TRAJECTORY PLANNING
# ==========================================
target_quat = np.array([0.0, 1.0, 0.0, 0.0])

pre_grasp_pos = obj_pos + np.array([0.0, 0.0, 0.15])
current_joints = real_robot.get_joints_pos()
pre_grasp_q7 = FrankaKinematics.inverse(current_joints, pre_grasp_pos, target_quat)
#pre_grasp_q7 = current_joints + np.array([0.1, 0.0, 0.0, 0, 0.0, 0, 0])

grasp_q7 = FrankaKinematics.inverse(pre_grasp_q7, obj_pos, target_quat)


# ==========================================
# 3. EXECUTION
# ==========================================

# Phase 1: Move ABOVE the object
print("Moving to Approach position...")
traj1 = plan_joints_trajectory(current_joints, pre_grasp_q7, max_velocity=0.25, dt=dt)
arm_traj1 = [qpos[:7] for qpos in traj1] # Extract just the 7 arm joints
print(arm_traj1[0], arm_traj1[-1])
real_robot.original_move_trajectory(arm_traj1, dt) # Send the whole smooth map!



# Phase 2: Open the Gripper
print("Opening Gripper...")
real_robot.move_gripper(0.08) 
real_robot.wait_seconds(0.5) 

# Phase 3: Descend vertically to the object
print("Descending to grasp...")
current_joints = real_robot.get_joints_pos()
traj3 = plan_joints_trajectory(current_joints, grasp_q7, max_velocity=0.25, dt=dt)
arm_traj3 = [qpos[:7] for qpos in traj3]
real_robot.original_move_trajectory(arm_traj3, dt)

# Phase 4: Close the Gripper
print("Closing Gripper...")
real_robot.move_gripper(0.0) 
real_robot.wait_seconds(0.5) 

# Phase 5: Lift the object
print("Lifting the object...")
current_joints = real_robot.get_joints_pos()
traj5 = plan_joints_trajectory(current_joints, pre_grasp_q7, max_velocity=0.25, dt=dt)
arm_traj5 = [qpos[:7] for qpos in traj5]
real_robot.original_move_trajectory(arm_traj5, dt)

# Phase 6: Open the Gripper (Release)
print("Opening Gripper...")
real_robot.move_gripper(0.08) 
real_robot.wait_seconds(0.5) 

move_to_home(real_robot) 

print("Sequence complete. Shutting down...")
real_robot.shutdown()

