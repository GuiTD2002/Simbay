import numpy as np

from src.planning import FrankaKinematics
from src.planning import plan_joints_trajectory
from src.planning import plan_settle_trajectory
from src.planning import stitch_trajectories
from src.utils import execute_trajectory


def grab_and_lift(robot, obj_pos, lift_height, callback=None):
    lift_pos = obj_pos + lift_height * np.array([0.0, 0.0, 1.0])
    target_quat = np.array([0.0, 1.0, 0.0, 0.0])
    current_joints = robot.get_joints_pos()

    # Find target joints
    lift_qpos = FrankaKinematics.inverse(current_joints, lift_pos, target_quat)
    grab_qpos  = FrankaKinematics.inverse(lift_qpos, obj_pos, target_quat)

    # Plan the trajectories
    lift_move   = plan_joints_trajectory(current_joints, lift_qpos, max_velocity=2.5, dt=robot.dt)
    lift_settle = plan_settle_trajectory(lift_move[-1], time=0.5, dt=robot.dt)
    lift_traj   = stitch_trajectories(lift_move, lift_settle)

    grab_move   = plan_joints_trajectory(lift_move[-1], grab_qpos, max_velocity=2.5, dt=robot.dt)
    grab_settle = plan_settle_trajectory(grab_move[-1], time=0.5, dt=robot.dt)
    grab_traj   = stitch_trajectories(grab_move, grab_settle)

    # Open gripper
    robot.move_gripper(0.08) 
    if callback:
        callback()

    # Execute the lift trajectory
    print("Moving to approach position...", flush=True)
    execute_trajectory(robot, lift_traj, callback)

    # Execute the grab trajectory
    print("Going Down...", flush=True)
    execute_trajectory(robot, grab_traj, callback)

    # Close gripper
    print("Grabbing object...", flush=True)
    robot.move_gripper(0.00)
    if callback:
        callback()

    # Execute reverse trajectory
    print("Lifting...", flush=True)
    execute_trajectory(robot, grab_traj[::-1], callback)
    