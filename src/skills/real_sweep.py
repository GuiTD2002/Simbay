import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray

import time
import numpy as np
from collections import deque

from src.planning import plan_cartesian_trajectory
from src.planning import plan_settle_trajectory
from src.planning import stitch_trajectories
from src.utils import detect_contact

import matplotlib.pyplot as plt

class ParticleVisualizer(Node):
    def __init__(self):
        super().__init__('particle_visualizer')
        self.publisher = self.create_publisher(MarkerArray, '/particle_cloud', 10)
        
    def publish_particles(self, particles, weights, fixed_x=0.55, fixed_z=0.09):
        marker_array = MarkerArray()
        
        # 1. Wipe old RViz ghosts before drawing
        clear_marker = Marker()
        clear_marker.action = 3  # Marker.DELETEALL
        marker_array.markers.append(clear_marker)
        
        max_weight = max(weights) if len(weights) > 0 else 1.0
        if max_weight < 1e-8: max_weight = 1.0
        
        dim = particles.shape[1] if particles.ndim > 1 else 1

        for i in range(len(particles)):
            marker = Marker()
            marker.header.frame_id = "base" 
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "particles"
            marker.id = i
            
            # ==========================================
            # THE DOT VISUALS
            # ==========================================
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            # --- DOT SIZE (Meters) ---
            dot_diameter = 0.01  # 1 cm diameter sphere
            marker.scale.x = dot_diameter
            marker.scale.y = dot_diameter
            marker.scale.z = dot_diameter
            
            # --- DOT COLOR ---
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 1.0
            
            # Adjust transparency based on weight
            relative_weight = weights[i] / max_weight
            marker.color.a = max(0.1, float(relative_weight)) # Increased base alpha to 0.1 so small dots are visible
            # ==========================================
            
            # --- MAP MATH TO 3D SPACE ---
            if dim == 1:
                marker.pose.position.x = float(fixed_x)
                marker.pose.position.y = float(particles[i] if particles.ndim == 1 else particles[i, 0])
                marker.pose.position.z = 0.05
            elif dim == 2:
                marker.pose.position.x = float(particles[i, 0])
                marker.pose.position.y = float(particles[i, 1])
                marker.pose.position.z = 0.05
                
            marker.pose.orientation.w = 1.0
            
            # Put the dot into the array!
            marker_array.markers.append(marker)
            
        # Publish the full array
        self.publisher.publish(marker_array)


def real_sweep_until_contact(robot, particle_filter, start_pos, end_pos, target_quat, sweep_vel, safety_distance, visualize=False, visualizer=None):
    """
    DEDICATED REAL HARDWARE SCRIPT.
    Uses Transient Masking (Blind Distance) to ignore acceleration spikes, 
    but relies entirely on raw torque data for collision detection.
    """
    print("\n[REAL SWEEP] Starting physical sweep sequence...")
    
    sweep_direction = (end_pos - start_pos) / np.linalg.norm(end_pos - start_pos)
    _real_prepare_sweep(robot, robot.get_joints_pos(), start_pos, sweep_direction, target_quat, sweep_vel, safety_distance)

    current_joints = robot.get_joints_pos()
    sweep_traj = plan_cartesian_trajectory(current_joints, end_pos, target_quat, sweep_vel, robot.dt)
    step_size = robot.dt * sweep_vel

    state = {
        'qpos': robot.get_joints_pos(),
        'qvel': robot.get_joints_vel()
    }
    particle_filter.update_internal_state(state)

    total_duration = len(sweep_traj) * robot.dt
    print(f"[REAL SWEEP] Firing async trajectory. Listening for contact...")
    robot.move_trajectory_async(sweep_traj, robot.dt)

    # ==========================================
    # PRE-LOOP SETUP (Raw Data Mode)
    # ==========================================
    torque_history = deque(maxlen=50) # Standard 50-step history
    

    delta_threshold = 0.4

    
    # Distance Setup for Transient Masking
    actual_start_pos = robot.get_ee_pos() 
    safe_blind_distance = safety_distance
    
    recorded_distances = []
    recorded_torque_norms = []

    start_time = time.perf_counter()
    step = 0
    contact = 0

    # ==========================================
    # THE TIME-SYNCHRONIZED LOOP
    # ==========================================
    while True:
        elapsed_time = time.perf_counter() - start_time
        
        if elapsed_time > total_duration:
            break
            
        full_measurements = robot.get_torque_reads() 
        current_torque_norm = np.linalg.norm(full_measurements[:3]) 
        
        current_ee_pos = robot.get_ee_pos()
        distance_traveled = np.linalg.norm(current_ee_pos - actual_start_pos)

        # ==========================================
        # 4. TRANSIENT MASKING (No EMA Smoothing)
        # ==========================================
        if distance_traveled < safe_blind_distance:
            # BLIND PERIOD: We are accelerating. Do absolutely nothing.
            contact = 0
        else:
            # STEADY STATE: We are safely moving. Begin baseline math.
            if len(torque_history) > 10:
                baseline_torque = np.mean(torque_history)
                
                # --- 3-Tier Logic on RAW Torque ---
                if current_torque_norm > (baseline_torque + delta_threshold):
                    # 🚨 HIT DETECTED
                    contact = 1
                    
                else:
                    # 🟢 NORMAL: Update baseline
                    contact = 0
                    torque_history.append(current_torque_norm)
            else:
                # Still building initial history
                torque_history.append(current_torque_norm)

        # ==========================================
        # 5. ESTIMATE (Math Layer)
        # ==========================================
        real_qpos = robot.get_joints_pos()
        real_qvel = robot.get_joints_vel()
        
        observation = {
            'torques'  : full_measurements,
            'contact'  : contact,      
            'direction': sweep_direction,
            'arm_pos'  : current_ee_pos,
            'step_size': step_size
        }
        
        ctrl = {
            'joints' : real_qpos,
            'gripper': 0.00
        }
        
        current_state = {
            'qpos': real_qpos,
            'qvel': real_qvel
        }
        
        particle_filter.predict(ctrl) 
        
        if step % 10 == 0 or contact == 1:
            particle_filter.update(observation)
            particle_filter.resample(current_state)

            if visualizer is not None:
                visualizer.publish_particles(
                    particle_filter.particles, 
                    particle_filter.weights, 
                    fixed_x=actual_start_pos[0]
                )
                rclpy.spin_once(visualizer, timeout_sec=0.0)

        particle_filter.record_state()

        recorded_distances.append(distance_traveled) 
        recorded_torque_norms.append(current_torque_norm)

        # ==========================================
        # 6. RESOLVE CONTACT
        # ==========================================
        if contact:
            baseline_val = np.mean(torque_history) if len(torque_history) > 0 else 0
            
            print(f"✅ [REAL SWEEP] Physical contact detected at step: {step} (Time: {elapsed_time:.3f}s)")
            print(f"   -> Position: {robot.get_ee_pos()}")
            print(f"   -> Raw Torque: {current_torque_norm:.2f} Nm (Baseline was: {baseline_val:.2f} Nm)")    
            print(f"   -> Spike size: {current_torque_norm - baseline_val:.2f} Nm")
            
            robot.stop_arm()
            robot.wait_seconds(0.2) 
            _real_execute_safe_retreat(robot, sweep_direction)
            break 
            
        step += 1

    # ... (Plotting code remains the same) ...
    
    """
    # ==========================================
    # PLOT THE RESULTS AFTER RETREATING
    # ==========================================
    print("[REAL SWEEP] Plotting torque data...")
    plt.figure(figsize=(10, 5))
    
    # Plot using DISTANCE on the X-axis
    plt.plot(recorded_distances, recorded_torque_norms, label='Torque Norm (Nm)', color='blue', linewidth=2)
    
    # Shade from 0.0m to the safe blind distance (e.g., 0.02m)
    plt.axvspan(0, safe_blind_distance, color='gray', alpha=0.3, label=f'Blind Period ({safe_blind_distance}m)')
    
    if contact_distance is not None:
        plt.axvline(x=contact_distance, color='red', linestyle='--', linewidth=2, label='Contact Detected!')
        
    plt.title('End-Effector Torque Norm vs Physical Distance')
    plt.xlabel('Distance Traveled (Meters)') # <-- X-axis is now meters!
    plt.ylabel('Torque Magnitude (Nm)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.show()
    """


def _real_prepare_sweep(robot, current_joints, start_pos, sweep_direction, target_quat, sweep_vel, safety_distance):
    """Executes the preparation phase synchronously, waiting for each move to finish.""" 
    prep_pos = start_pos - (sweep_direction * safety_distance)
    hover_pos = prep_pos + np.array([0.0, 0.0, 0.1])
    
    hover_traj = plan_cartesian_trajectory(current_joints, hover_pos, target_quat, 0.05, robot.dt)
    hover_settle = plan_settle_trajectory(hover_traj[-1], 0.5, robot.dt)
    traj_hover = stitch_trajectories(hover_traj, hover_settle)

    prep_traj = plan_cartesian_trajectory(traj_hover[-1], prep_pos, target_quat, 0.05, robot.dt)
    prep_settle = plan_settle_trajectory(prep_traj[-1], 0.5, robot.dt)
    traj_prep = stitch_trajectories(prep_traj, prep_settle)
    
    start_traj = plan_cartesian_trajectory(traj_prep[-1], start_pos, target_quat, sweep_vel, robot.dt)

    
    print(f"   -> Moving to hover position...")
    robot.move_trajectory(traj_hover, robot.dt)
    
    print(f"   -> Moving to prep position...")
    robot.move_trajectory(traj_prep, robot.dt)
    
    #print(f"   -> Moving to start position...")
    #robot.move_trajectory(start_traj, robot.dt)



def _real_execute_safe_retreat(robot, sweep_direction):
    """Safely backs the robot away from the block in cartesian space."""
    safety_distance = 0.01
    current_joints = robot.get_joints_pos()[:7] 
    current_pos = robot.get_ee_pos()
    
    target_pos = current_pos - (safety_distance * sweep_direction) + np.array([0.0, 0.0, 0.1])
    
    if sweep_direction[0] != 0: 
        target_quat = np.array([0.0, np.sqrt(2)/2, np.sqrt(2)/2, 0.0])
    else:
        target_quat = np.array([0.0, 1.0, 0.0, 0.0])
    
    retreat_traj = plan_cartesian_trajectory(current_joints, target_pos, target_quat, max_velocity=0.1, dt=robot.dt)
    settle_traj = plan_settle_trajectory(retreat_traj[-1], 0.5, robot.dt)
    full_exit_traj = stitch_trajectories(retreat_traj, settle_traj)

    print(f"[REAL SWEEP] Safely retreating...")
    robot.move_trajectory(full_exit_traj, robot.dt)



def get_interpolated_qpos(trajectory, dt, elapsed_time):
    """
    Finds the exact mathematical target position based on a stopwatch.
    Interpolates between the two closest waypoints in the trajectory array.
    """
    # Total time of the trajectory
    max_time = (len(trajectory) - 1) * dt
    
    # Cap the elapsed time so we don't go out of bounds
    t = min(elapsed_time, max_time)
    t = max(t, 0.0) # Ensure it doesn't go negative
    
    # Find which two array indices we are currently between
    index_float = t / dt
    idx_low = int(np.floor(index_float))
    idx_high = int(np.ceil(index_float))
    
    # If we hit an exact index, return it
    if idx_low == idx_high:
        return trajectory[idx_low]
        
    # Cap upper index just to be absolutely safe against floating point rounding at the very end
    idx_high = min(idx_high, len(trajectory) - 1)
        
    # Otherwise, interpolate (blend) between the two closest points
    weight = index_float - idx_low
    
    pos_low = np.array(trajectory[idx_low])
    pos_high = np.array(trajectory[idx_high])
    
    return (1.0 - weight) * pos_low + (weight * pos_high)
