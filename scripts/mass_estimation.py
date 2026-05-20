import os
import sys

import matplotlib.pyplot as plt
import mujoco.viewer
import numpy as np

# Add the project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    
from src.estimation import MassMeasurementModel
from src.estimation import MassMotionModel
from src.estimation import ParticleFilterRegularized
from src.estimation import RobotContainer
from src.planning import FrankaKinematics
from src.planning import plan_joints_trajectory
from src.planning import plan_settle_trajectory
from src.planning import stitch_trajectories
from src.robots import RealRobot
from src.skills import grab_and_lift
from src.skills import sweep_until_contact
from src.utils import DEFAULT_OBJECT_PROPS2
from src.utils import execute_trajectory
from src.utils import initialize_mujoco_env
from src.utils import plot_particle_evolution

# ==========================================
# CONFIGURATION
# ==========================================
USE_REAL_ROBOT = False

# Filter Configuration
NUM_PARTICLES = 1
ESS_THRESHOLD = 0.5

# Workspace Limits (Y)
MIN_MASS, MAX_MASS = 0.1, 0.25

def main():
    print("Initializing Environment...")
    if USE_REAL_ROBOT:
        pass
        #robot = RealRobot() # type: ignore
        #robot.dt = 0.001 
    else:
        robot = initialize_mujoco_env(DEFAULT_OBJECT_PROPS2)
        robot.dt = 0.001 
        robot.viewer = mujoco.viewer.launch_passive(robot.model, robot.data)
    
    true_mass = DEFAULT_OBJECT_PROPS2['mass']
    if not USE_REAL_ROBOT: print(f"[Debug] Initial Ground Truth: Mass={true_mass:.3f}")

    limits = (MIN_MASS, MAX_MASS)
    container = RobotContainer(num_particles=NUM_PARTICLES, props=DEFAULT_OBJECT_PROPS2, dt=robot.dt)
    
    # Launch a separate viewer for the first particle in the container
    #if not USE_REAL_ROBOT:
        #particle_viewer = mujoco.viewer.launch_passive(container.robots[0].model, container.robots[0].data)
        #container.robots[0].viewer = particle_viewer

    particle_filter = ParticleFilterRegularized(
        num_particles=NUM_PARTICLES, state_bounds=limits, 
        motion_model=MassMotionModel(container), 
        measurement_model=MassMeasurementModel(container), 
        ess_threshold_ratio=ESS_THRESHOLD
    )
    state = {
        'qpos': robot.data.qpos.copy(),
        'qvel': robot.data.qvel.copy()
    }

    particle_filter.update_internal_state(state)

    # ==========================================
    # PHASE 1: Grab and Lift
    # ==========================================
    pos = DEFAULT_OBJECT_PROPS2['pos']
    lift_height = 0.1
    
    def prediction_callback():
        # Read the actual gripper command from the robot to sync the particle robots
        if not USE_REAL_ROBOT:
            current_gripper = robot.data.ctrl[7] * (0.08 / 255.0)
            target_joints = robot.data.ctrl[:7].copy()
        else:
            current_gripper = 0.00
            target_joints = robot.get_joints_pos()
            
        # Build the control dictionary required by the motion model
        control_input = {
            'joints': target_joints,
            'gripper': current_gripper
        }
        particle_filter.predict(control_input)
        
        # Continuously sync the full physical state so the object doesn't get left behind!
        current_state = {
            'qpos': robot.data.qpos.copy(),
            'qvel': robot.data.qvel.copy()
        }
        particle_filter.update_internal_state(current_state)
        
        if not USE_REAL_ROBOT:
            container.robots[0].sync()

    grab_and_lift(robot, pos, lift_height, callback=prediction_callback)


    # ==========================================
    # PHASE 2: Present Forward
    # ==========================================
    print("Rotating to point forward with vertical fingers...")
    forward_quat = np.array([0.5, 0.5, 0.5, 0.5])
    
    current_joints = robot.get_joints_pos()
    
    # Shift target position forward and up. This accommodates the length of the 
    # gripper as it rotates, preventing the elbow from swinging backward!
    target_pos = robot.get_ee_pos() + np.array([0.0, 0.15, 0.15])
    target_joints = FrankaKinematics.inverse(current_joints, target_pos, forward_quat)
    
    traj_move = plan_joints_trajectory(current_joints, target_joints, max_velocity=0.5, dt=robot.dt)
    traj_settle = plan_settle_trajectory(traj_move[-1], time=0.5, dt=robot.dt)
    traj_forward = stitch_trajectories(traj_move, traj_settle)
    
    execute_trajectory(robot, traj_forward, callback=prediction_callback)

    # ==========================================
    # PHASE 3: Turn On Particle Filter
    # ==========================================
    print("Turning on particle filter...")

    # Record the initial uniform spread BEFORE the first update collapses them!
    particle_filter.record_state()

    for _ in range(1000):
        # Dynamically read current state to prevent drift and jitter in the background viewers
        if not USE_REAL_ROBOT:
            current_gripper = robot.data.ctrl[7] * (0.08 / 255.0)
            target_joints = robot.data.ctrl[:7].copy()
        else:
            current_gripper = 0.00
            target_joints = robot.get_joints_pos()

        control_input = {
                'joints': target_joints,
                'gripper': current_gripper
            }
        
        state = {
            'qpos': robot.data.qpos.copy(),
            'qvel': robot.data.qvel.copy()
        }

        z_torque = robot.get_torque_reads()[1]
    

        # Add noise to Z-axis torque
        z_torque += np.random.normal(loc=0.0, scale=0.4)

        particle_filter.step(control_input, z_torque, state)
        particle_filter.record_state()
        
        # Step the simulated "real" robot forward in time so physics behaves normally
        if not USE_REAL_ROBOT:
            robot.move_joints(target_joints)
        
        # Forcefully sync the particle robots to glue them to the real robot's frame
        particle_filter.update_internal_state(state)
        
    
    print(z_torque)
    print(robot.get_torque_reads())
            
    # ==========================================
    # FINISH & RESULTS
    # ==========================================
    final_mass = particle_filter.estimate()[0] if isinstance(particle_filter.estimate(), np.ndarray) else particle_filter.estimate()
    
    print("\n" + "="*40)
    print("FINAL MASS ESTIMATION RESULTS")
    print("="*40)
    if not USE_REAL_ROBOT: print(f"True Object Mass : {true_mass:.3f} kg")
    print(f"Filter Est. Mass : {final_mass:.3f} kg")
    print("="*40 + "\n")
    
    output_folder = "saved_plots"
    plot_particle_evolution(particle_filter, dimension=0, ylabel='Estimated Mass (kg)', 
                            true_pos=true_mass, min_val=MIN_MASS, max_val=MAX_MASS, 
                            save_path=f"{output_folder}/mass_evolution.png")



if __name__ == "__main__":
    main()
