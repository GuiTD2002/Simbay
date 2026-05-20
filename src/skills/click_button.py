from os import name
import os
import sys

from src.planning import plan_joints_trajectory
from src.planning import plan_settle_trajectory
from src.planning import stitch_trajectories
from src.utils import FRANKA_HOME_QPOS
from src.utils import execute_trajectory
import numpy as np
from src.planning import FrankaKinematics



def click_button(robot, obj_pos, velocity=0.05, real = True):
    dt = robot.dt 

    # Close gripper
    print("Closing Gripper...")
    robot.move_gripper(0.00)  


    # Plan the move, then stitch a rigid settle at the end!
    target_quat = np.array([0.0, 1.0, 0.0, 0.0])
    initial_joints = robot.get_joints_pos() 
    target_pos = obj_pos + np.array([0.095, 0.035, 0.045])
    hover_pos = target_pos + np.array([0.0, 0.0, 0.05])

    target_q7 = FrankaKinematics.inverse(initial_joints, target_pos, target_quat)
    hover_q7 = FrankaKinematics.inverse(initial_joints, hover_pos, target_quat)

    print("Planning trajectory to click the button...")

    traj_move = plan_joints_trajectory(initial_joints, hover_q7, max_velocity=velocity, dt=dt)
    traj_settle = plan_settle_trajectory(traj_move[-1], time=0.5, dt=dt)
    full_home_traj = stitch_trajectories(traj_move, traj_settle)


    # Execute the trajectory
    print("Moving to Home...")
    if not real:
        execute_trajectory(robot, full_home_traj)
    else:
        robot.move_trajectory(full_home_traj, dt)


    # Plan the move, then stitch a rigid settle at the end!
    initial_joints = robot.get_joints_pos() 
    #print("Planning trajectory to click the button...")

    traj_move = plan_joints_trajectory(initial_joints, target_q7, max_velocity=velocity, dt=dt)
    traj_settle = plan_settle_trajectory(traj_move[-1], time=0.5, dt=dt)
    full_home_traj = stitch_trajectories(traj_move, traj_settle)


    # Execute the trajectory
    #print("Moving to Home...")
    if not real:
        execute_trajectory(robot, full_home_traj)
    else:
        robot.move_trajectory(full_home_traj, dt)
    
    #print("✅ Robot is perfectly Homed and Rigidlly Settled!")
