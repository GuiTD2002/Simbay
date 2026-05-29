
#!/usr/bin/env python
import os
import sys

import matplotlib.pyplot as plt
import mujoco.viewer
import numpy as np

# Add the project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    
from src.estimation import BinaryContactMeasurementModel
from src.estimation import ParticleFilterRegularized
from src.estimation import PositionMotionModel
from src.estimation import RobotContainer
from src.robots import RealRobot
from src.skills import move_to_home
from src.skills.sweep import sweep_until_contact
from src.utils import DEFAULT_OBJECT_PROPS
from src.utils import initialize_mujoco_env
from src.utils import plot_particle_evolution
from src.skills import click_button

# ==========================================
# CONFIGURATION
# ==========================================
USE_REAL_ROBOT = False

# Filter Configuration
NUM_PARTICLES = 50  
ESS_THRESHOLD = 0.5
DEBUG_PARTICLE_Y = 0.17

# Workspace Limits (Y)
MIN_Y, MAX_Y = 0.1, 0.2

# Sweep Parameters
FIXED_X = 0.55
FIXED_Z = 0.08
MAX_BLOCK_HALF_SIZE = 0.075 
SAFETY_DISTANCE = 0.01
SWEEP_VEL = 0.1
SWEEP_QUAT = np.array([0.0, 1.0, 0.0, 0.0])


def track_ground_truth(robot):
    if USE_REAL_ROBOT: return None
    block_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, 'object') # type: ignore
    return robot.data.xpos[block_id][1]


def main():
    print("Initializing Environment...")
    if USE_REAL_ROBOT:
        robot = RealRobot()
        robot.dt = 0.001 
    else:
        robot = initialize_mujoco_env()
        robot.dt = 0.001 
        robot.viewer = mujoco.viewer.launch_passive(robot.model, robot.data)
    
    true_y = track_ground_truth(robot)
    if not USE_REAL_ROBOT: print(f"[Debug] Initial Ground Truth: Y={true_y:.3f}")

    limits = (MIN_Y, MAX_Y)
    container = RobotContainer(num_particles=NUM_PARTICLES, props=DEFAULT_OBJECT_PROPS, dt=robot.dt)
    
    particle_filter = ParticleFilterRegularized(
        num_particles=NUM_PARTICLES, state_bounds=limits, 
        motion_model=PositionMotionModel(container), 
        measurement_model=BinaryContactMeasurementModel(container), 
        ess_threshold_ratio=ESS_THRESHOLD
    )

    #particle_filter.particles[0, 0] = DEBUG_PARTICLE_Y
    #particle_filter.update_internal_state({
        #'qpos': robot.get_joints_pos(),
        #'qvel': robot.get_joints_vel()
    #})
    #particle_viewer = mujoco.viewer.launch_passive(container.robots[0].model, container.robots[0].data)
    #container.robots[0].viewer = particle_viewer
    #print(f"[Debug] Particle 0 viewer pinned at Y={DEBUG_PARTICLE_Y:.3f}")


    # ==========================================
    # PHASE 1: SWEEP FORWARD (+Y)
    # ==========================================
    #MIN_Y2, MAX_Y2 = 0.1, 0.2
    print("\n--- Phase 1: Sweep Forward (+Y) ---")
    start_pos_y1 = np.array([FIXED_X, MIN_Y - MAX_BLOCK_HALF_SIZE - SAFETY_DISTANCE, FIXED_Z])
    end_pos_y1 = np.array([FIXED_X, MAX_Y, FIXED_Z])

    sweep_until_contact(
        robot=robot, particle_filter=particle_filter, start_pos=start_pos_y1,
        end_pos=end_pos_y1, target_quat=SWEEP_QUAT, sweep_vel=SWEEP_VEL,
        safety_distance=SAFETY_DISTANCE, visualize=True
    )

    if not USE_REAL_ROBOT: print(f"🛑 Ground Truth After Swipe 1: {track_ground_truth(robot):.3f}")


    # ==========================================
    # PHASE 2: SWEEP BACKWARD (-Y)
    # ==========================================
    print("\n--- Phase 2: Sweep Backward (-Y) ---")
    start_pos_y2 = np.array([FIXED_X, MAX_Y + MAX_BLOCK_HALF_SIZE + SAFETY_DISTANCE, FIXED_Z])
    end_pos_y2 = np.array([FIXED_X, MIN_Y, FIXED_Z])

    sweep_until_contact(
        robot=robot, particle_filter=particle_filter, start_pos=start_pos_y2,
        end_pos=end_pos_y2, target_quat=SWEEP_QUAT, sweep_vel=SWEEP_VEL,
        safety_distance=SAFETY_DISTANCE, visualize=True
    )

    if not USE_REAL_ROBOT: print(f"🛑 Ground Truth After Swipe 2: {track_ground_truth(robot):.3f}")


    # ==========================================
    # PHASE 3: EXTRACTION & HOMING
    # ==========================================
    
    # Extract purely from the geometric center of our padded bounds
    final_y = particle_filter.estimate()[0]

    print(final_y)
    obj_pos = np.array([FIXED_X, final_y, FIXED_Z])
    click_button(robot, obj_pos, real=False)
    move_to_home(robot, real=False)

    print("\n" + "="*40)
    print("FINAL 1D ESTIMATION RESULTS")
    print("="*40)
    print(f"True Object Position : {true_y:.3f}")
    print(f"Filter Center Est.   : {final_y:.3f}")
    print("="*40 + "\n")

    # Create a folder name (optional, helps keep things organized)
    output_folder = "saved_plots"

    # Plot Y
    plot_particle_evolution(particle_filter, axis='y', true_pos=true_y, 
                            min_val=-0.2, max_val=0.2, 
                            save_path=f"{output_folder}/y_axis_evolution.png")
    

if __name__ == "__main__":
    main()
