import time

import numpy as np

from src.planning import plan_cartesian_trajectory
from src.planning import plan_settle_trajectory
from src.planning import stitch_trajectories
from src.utils import detect_contact
from src.utils import execute_trajectory
from src.utils import visualize_particles


def sweep_until_contact(robot, particle_filter, start_pos, end_pos, target_quat, sweep_vel, safety_distance, visualize=False):
    """
    Executes a complete robotic sweep skill: Hover -> Prep -> Start -> Sweep.

    The sweep trajectory is fired on a background thread (`move_trajectory_async`)
    so that the sim step + viewer tick at `robot.dt` independently of the main
    thread, which polls measurements and runs the particle filter — same
    structure as `real_sweep_until_contact`.
    """
    sweep_direction = (end_pos - start_pos) / np.linalg.norm(end_pos - start_pos)
    current_joints = _prepare_sweep(robot, robot.get_joints_pos(), start_pos, sweep_direction, target_quat, sweep_vel, safety_distance)

    # Plan Trajectory
    sweep_traj = plan_cartesian_trajectory(current_joints, end_pos, target_quat, sweep_vel, robot.dt)
    step_size = robot.dt * sweep_vel  # How far we move in Cartesian space each step

    state = {
        'qpos': robot.get_joints_pos(),
        'qvel': robot.get_joints_vel()
    }
    particle_filter.update_internal_state(state)

    total_duration = len(sweep_traj) * robot.dt
    robot.move_trajectory_async(sweep_traj, robot.dt)

    start_time = time.perf_counter()
    step = 0
    contact = 0
    timed_out = False

    # Mirror real_sweep_until_contact: predict every iter, update+resample
    # throttled to every 10 iters (or on contact), record_state every iter.
    while True:
        elapsed_time = time.perf_counter() - start_time
        if elapsed_time > total_duration:
            timed_out = True
            break

        planned_qpos = _get_interpolated_qpos(sweep_traj, robot.dt, elapsed_time)

        # ==========================================
        # 1. SENSE
        # ==========================================
        measurements = robot.get_torque_reads()
        contact = 1 if detect_contact(measurements) else 0

        # ==========================================
        # 2. ESTIMATE
        # ==========================================
        observation = {
            'torques'  : measurements,
            'contact'  : contact,
            'direction': sweep_direction,
            'arm_pos'  : robot.get_ee_pos(),
            'step_size': step_size
        }
        ctrl = {
            'joints' : planned_qpos,
            'gripper': 0.00
        }
        current_state = {
            'qpos': robot.get_joints_pos(),
            'qvel': robot.get_joints_vel()
        }

        particle_filter.predict(ctrl)

        if step % 10 == 0 or contact == 1:
            particle_filter.update(observation)
            particle_filter.resample(current_state, step=step)

            if visualize and hasattr(robot, 'viewer'):
                visualize_particles(robot.viewer, particle_filter.particles, particle_filter.weights)

        particle_filter.record_state()

        # ==========================================
        # 3. RESOLVE CONTACT
        # ==========================================
        if contact:
            print(f"✅ Object detected at step: {step}!")
            robot.stop_arm()
            robot.wait_seconds(0.2)
            _execute_safe_retreat(robot, sweep_direction)
            break

        step += 1

    if timed_out:
        robot.stop_arm()


def _get_interpolated_qpos(trajectory, dt, elapsed_time):
    """Return the trajectory waypoint interpolated to `elapsed_time` (stopwatch-paced)."""
    max_time = (len(trajectory) - 1) * dt
    t = max(0.0, min(elapsed_time, max_time))
    index_float = t / dt
    idx_low = int(np.floor(index_float))
    idx_high = min(int(np.ceil(index_float)), len(trajectory) - 1)
    if idx_low == idx_high:
        return trajectory[idx_low]
    weight = index_float - idx_low
    return (1.0 - weight) * np.array(trajectory[idx_low]) + weight * np.array(trajectory[idx_high])
    

def _prepare_sweep(robot, current_joints, start_pos, sweep_direction, target_quat, sweep_vel, safety_distance):
    """
    Executes the preparation phase of the sweep skill: Move to hover, then prep, then start.
    """ 
    # current_joints = np.asarray(current_joints, dtype=float)[:7]
    # if not np.all(np.isfinite(current_joints)):
    #     current_joints = np.zeros(7, dtype=float)

    # Get preparation and hover positions
    prep_pos = start_pos - (sweep_direction * safety_distance)
    hover_pos = prep_pos + np.array([0.0, 0.0, 0.1])
    
    # Plan hover trajectory
    hover_traj = plan_cartesian_trajectory(current_joints, hover_pos, target_quat, 0.2, robot.dt)
    hover_settle = plan_settle_trajectory(hover_traj[-1], 0.5, robot.dt)
    traj_hover = stitch_trajectories(hover_traj, hover_settle)

    # Plan preparation trajectory
    prep_traj = plan_cartesian_trajectory(traj_hover[-1], prep_pos, target_quat, 0.2, robot.dt)
    prep_settle = plan_settle_trajectory(prep_traj[-1], 0.5, robot.dt)
    traj_prep = stitch_trajectories(prep_traj, prep_settle)
    
    # Plan start trajectory
    start_traj = plan_cartesian_trajectory(traj_prep[-1], start_pos, target_quat, sweep_vel, robot.dt)

    
    print(f"\n--- Starting Sweep ({int(sweep_direction[0]), int(sweep_direction[1]), int(sweep_direction[2])}) ---")
    print("Moving to hover position...")
    execute_trajectory(robot, traj_hover)
    
    print("Moving to prep position...")
    execute_trajectory(robot, traj_prep)
    
    print("Moving to start position...")
    execute_trajectory(robot, start_traj)

    return start_traj[-1]

def _execute_safe_retreat(robot, sweep_direction):
    """Safely backs the robot away from the block in cartesian space."""
    safety_distance = 0.01
    current_joints = robot.get_joints_pos()[:7] 
    current_pos = robot.get_ee_pos()
    
    target_pos = current_pos - (safety_distance * sweep_direction) + np.array([0.0, 0.0, 0.1])
    
    # Automatically orient the wrist based on the sweep axis
    if sweep_direction[0] != 0: 
        target_quat = np.array([0.0, np.sqrt(2)/2, np.sqrt(2)/2, 0.0])
    else:
        target_quat = np.array([0.0, 1.0, 0.0, 0.0])
    
    retreat_traj = plan_cartesian_trajectory(current_joints, target_pos, target_quat, max_velocity=0.2, dt=robot.dt)
    settle_traj = plan_settle_trajectory(retreat_traj[-1], 0.5, robot.dt)
    full_exit_traj = stitch_trajectories(retreat_traj, settle_traj)

    execute_trajectory(robot, full_exit_traj)
