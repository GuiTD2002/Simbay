import time

import numpy as np

from src.planning import plan_cartesian_trajectory
from src.planning import plan_settle_trajectory
from src.planning import stitch_trajectories
from src.utils import detect_contact
from src.utils import execute_trajectory
from src.utils import visualize_particles


def sweep_until_contact(
    robot,
    particle_filter,
    start_pos,
    end_pos,
    target_quat,
    sweep_vel,
    safety_distance,
    visualize=False,
    gif_recorder=None,
):
    """
    Executes a complete robotic sweep skill: Hover -> Prep -> Start -> Sweep.
    """
    sweep_direction = (end_pos - start_pos) / np.linalg.norm(end_pos - start_pos)
    current_joints = _prepare_sweep(robot, robot.get_joints_pos(), start_pos, sweep_direction, target_quat, sweep_vel, safety_distance, gif_recorder=gif_recorder)

    # Plan Trajectory
    sweep_traj = plan_cartesian_trajectory(current_joints, end_pos, target_quat, sweep_vel, robot.dt)
    step_size = robot.dt * sweep_vel  # How far we move in Cartesian space each step 

    state = {
        'qpos': robot.get_joints_pos(),
        'qvel': robot.get_joints_vel()
    }

    particle_filter.update_internal_state(state)

    for step, qpos in enumerate(sweep_traj):   
        # ==========================================
        # 1. ACT & SENSE (Hardware Layer)
        # ==========================================
        # Move the joints instantly (Sim updates array, Real blasts network packet)
        robot.move_joints(qpos)
        
        # Read the current physical state
        measurements = robot.get_torque_reads()
        contact = 1 if detect_contact(measurements) else 0


        # ==========================================
        # 2. ESTIMATE (Math Layer)
        # ==========================================
        observation = {
            'torques'  : measurements,
            'contact'  : contact, 
            'direction': sweep_direction,
            'arm_pos'  : robot.get_ee_pos(), 
            'step_size': step_size
        }
        ctrl = {
            'joints' : qpos,
            'gripper': 0.00
        }

        current_state = {
            'qpos': robot.get_joints_pos(),
            'qvel': robot.get_joints_vel()
        }
        
        particle_filter.step(ctrl, observation, current_state)
        particle_filter.record_state()  # Save the current state for visualization later    

        # ==========================================
        # 3. RENDER & PACE (The Magic Synchronization)
        # ==========================================
        # If the user wants to see it AND the robot actually has a screen, draw the dots!
        if visualize and hasattr(robot, 'viewer') and step % 20 == 0: # Only update the dots at 50 Hz to avoid performance issues
            visualize_particles(robot.viewer, particle_filter.particles, particle_filter.weights)

        if gif_recorder is not None:
            gif_recorder.capture(robot)

        # Let the robot pace itself!
        # -> Sim Robot: Skips rendering if it's going too fast (>60 FPS) to save CPU.
        # -> Real Robot: Calculates exact math time and sleeps the remainder to lock at 1000 Hz.
        robot.sync() 

        # ==========================================
        # 4. RESOLVE CONTACT (Safety & Convergence)
        # ==========================================
        if contact:
            print(f"✅ Object detected at step: {step}!")    
            # Retreat safely
            _execute_safe_retreat(robot, sweep_direction, gif_recorder=gif_recorder)
            break 
    

def _prepare_sweep(robot, current_joints, start_pos, sweep_direction, target_quat, sweep_vel, safety_distance, gif_recorder=None):
    """
    Executes the preparation phase of the sweep skill: Move to hover, then prep, then start.
    """ 
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

    capture_cb = (lambda: gif_recorder.capture(robot)) if gif_recorder is not None else None

    print(f"\n--- Starting Sweep ({int(sweep_direction[0]), int(sweep_direction[1]), int(sweep_direction[2])}) ---")
    print("Moving to hover position...")
    execute_trajectory(robot, traj_hover, callback=capture_cb)

    print("Moving to prep position...")
    execute_trajectory(robot, traj_prep, callback=capture_cb)

    print("Moving to start position...")
    execute_trajectory(robot, start_traj, callback=capture_cb)

    return start_traj[-1]

def _execute_safe_retreat(robot, sweep_direction, gif_recorder=None):
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

    capture_cb = (lambda: gif_recorder.capture(robot)) if gif_recorder is not None else None
    execute_trajectory(robot, full_exit_traj, callback=capture_cb)
