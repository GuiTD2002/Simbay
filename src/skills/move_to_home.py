from os import name
import os
import sys

from src.planning import plan_joints_trajectory
from src.planning import plan_settle_trajectory
from src.planning import stitch_trajectories
from src.utils import FRANKA_HOME_QPOS
from src.utils import execute_trajectory



def move_to_home(robot, velocity=0.5, real = True):
    dt = robot.dt 

    # Close gripper
    print("Closing Gripper...")
    robot.move_gripper(0.00)  


    # Plan the move, then stitch a rigid settle at the end!
    initial_joints = robot.get_joints_pos() 
    print("Planning trajectory to Home...")
    traj_move = plan_joints_trajectory(initial_joints, FRANKA_HOME_QPOS, max_velocity=velocity, dt=dt)
    traj_settle = plan_settle_trajectory(traj_move[-1], time=0.5, dt=dt)
    full_home_traj = stitch_trajectories(traj_move, traj_settle)


    # Execute the trajectory
    print("Moving to Home...")
    if not real:
        execute_trajectory(robot, full_home_traj)
    else:
        robot.original_move_trajectory(full_home_traj, dt)
    
    print("✅ Robot is perfectly Homed and Rigidlly Settled!")


