import os
import sys

import matplotlib.pyplot as plt
import mujoco.viewer
import numpy as np

import os
import sys

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

# ==========================================
# CONFIGURATION
# ==========================================
USE_REAL_ROBOT = False

NUM_PARTICLES = 100
ESS_THRESHOLD = 0.5

# Workspace Limits (X, Y)
MIN_X, MAX_X = 0.5, 0.6
MIN_Y, MAX_Y = 0.1, 0.2

# Sweep Parameters
FIXED_Z = 0.08
MAX_BLOCK_HALF_SIZE_X = 0.125 
MAX_BLOCK_HALF_SIZE_Y = 0.075
SAFETY_DISTANCE = 0.01
SWEEP_VEL = 0.1

def track_ground_truth(robot):
    if USE_REAL_ROBOT: return None, None
    block_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, 'object') # type: ignore
    return robot.data.xpos[block_id][0], robot.data.xpos[block_id][1]

def main():
    print("Initializing Environment...")
    if USE_REAL_ROBOT:
        robot = RealRobot()
        robot.dt = 0.001 
    else:
        robot = initialize_mujoco_env()
        robot.dt = 0.001
        viewer = mujoco.viewer.launch_passive(robot.model, robot.data)
        robot.viewer = viewer

    true_x, true_y = track_ground_truth(robot)
    if not USE_REAL_ROBOT: print(f"🛑 [Debug] Initial Ground Truth: X={true_x:.3f}, Y={true_y:.3f}")

    limits = (np.array([MIN_X, MIN_Y]), np.array([MAX_X, MAX_Y]))
    container = RobotContainer(num_particles=NUM_PARTICLES, props=DEFAULT_OBJECT_PROPS, dt=robot.dt)
    
    particle_filter = ParticleFilterRegularized(
        num_particles=NUM_PARTICLES, state_bounds=limits, 
        motion_model=PositionMotionModel(container), 
        measurement_model=BinaryContactMeasurementModel(container), 
        ess_threshold_ratio=ESS_THRESHOLD
    )

    mid_x = (MIN_X + MAX_X) / 2.0
    quat_y = np.array([0.0, 1.0, 0.0, 0.0])
    quat_x = np.array([0.0, np.sqrt(2)/2, np.sqrt(2)/2, 0.0])

    # ==========================================
    # PHASE 1: SWEEP FORWARD (+Y)
    # ==========================================
    print("\n--- Phase 1: Sweep Forward (+Y) ---")
    start_pos_y1 = np.array([mid_x, MIN_Y - MAX_BLOCK_HALF_SIZE_Y - SAFETY_DISTANCE, FIXED_Z])
    end_pos_y1 = np.array([mid_x, MAX_Y, FIXED_Z])

    sweep_until_contact(
        robot=robot, particle_filter=particle_filter, start_pos=start_pos_y1, 
        end_pos=end_pos_y1, target_quat=quat_y, sweep_vel=SWEEP_VEL, 
        safety_distance=SAFETY_DISTANCE, visualize=not USE_REAL_ROBOT
    )
    
    if not USE_REAL_ROBOT: print(f"🛑 [Debug] Ground Truth After Swipe 1: X={true_x:.3f}, Y={true_y:.3f}")

    # After Phase 1 sweep_until_contact finishes
    max_y = np.max(particle_filter.particles[:, 1])
    print(f"\n🛑 POST-PHASE 1 CHECK: The highest surviving particle is at Y = {max_y:.3f}")


    # ==========================================
    # PHASE 2: SWEEP BACKWARD (-Y)
    # ==========================================
    print("\n--- Phase 2: Sweep Backward (-Y) ---")
    start_pos_y2 = np.array([mid_x, MAX_Y + MAX_BLOCK_HALF_SIZE_Y + SAFETY_DISTANCE, FIXED_Z])
    end_pos_y2 = np.array([mid_x, MIN_Y, FIXED_Z])

    sweep_until_contact(
        robot=robot, particle_filter=particle_filter, start_pos=start_pos_y2, 
        end_pos=end_pos_y2, target_quat=quat_y, sweep_vel=SWEEP_VEL, 
        safety_distance=SAFETY_DISTANCE, visualize=not USE_REAL_ROBOT
    )

    if not USE_REAL_ROBOT: print(f"🛑 [Debug] Ground Truth After Swipe 2: X={true_x:.3f}, Y={true_y:.3f}")


    # ==========================================
    # PHASE 3: SWEEP BACKWARD (-X)
    # ==========================================
    print("\n--- Phase 3: Sweep Backward (-X) ---")
    estimate_y = particle_filter.estimate()[1]
    start_pos_y2 = np.array([mid_x, MAX_Y + MAX_BLOCK_HALF_SIZE_X + SAFETY_DISTANCE, FIXED_Z])
    start_pos_x1 = np.array([MAX_X + MAX_BLOCK_HALF_SIZE_X + SAFETY_DISTANCE, estimate_y, FIXED_Z])
    end_pos_x1 = np.array([MIN_X, estimate_y, FIXED_Z])

    sweep_until_contact(
        robot=robot, particle_filter=particle_filter, start_pos=start_pos_x1, 
        end_pos=end_pos_x1, target_quat=quat_x, sweep_vel=SWEEP_VEL, 
        safety_distance=SAFETY_DISTANCE, visualize=not USE_REAL_ROBOT
    )
    
    if not USE_REAL_ROBOT: print(f"🛑 [Debug] Ground Truth After Swipe 3: X={true_x:.3f}, Y={true_y:.3f}")


    # ==========================================
    # PHASE 4: SWEEP FORWARD (+X)
    # ==========================================
    print("\n--- Phase 4: Sweep Forward (+X) ---")
    estimate_y = particle_filter.estimate()[1]
    start_pos_x2 = np.array([MIN_X - MAX_BLOCK_HALF_SIZE_X - SAFETY_DISTANCE, estimate_y, FIXED_Z])
    end_pos_x2 = np.array([MAX_X, estimate_y, FIXED_Z])

    sweep_until_contact(
        robot=robot, particle_filter=particle_filter, start_pos=start_pos_x2, 
        end_pos=end_pos_x2, target_quat=quat_x, sweep_vel=SWEEP_VEL, 
        safety_distance=SAFETY_DISTANCE, visualize=not USE_REAL_ROBOT
    )
    
    if not USE_REAL_ROBOT: print(f"🛑 [Debug] Ground Truth After Swipe 4: X={true_x:.3f}, Y={true_y:.3f}")

    # ==========================================
    # FINISH & RESULTS
    # ==========================================
    move_to_home(robot, real=USE_REAL_ROBOT)

    # Final estimate mathematically perfectly balanced by the padded bounds
    final_x = particle_filter.estimate()[0]
    final_y = particle_filter.estimate()[1]

    print("\n" + "="*40)
    print("FINAL 2D ESTIMATION RESULTS")
    print("="*40)
    if not USE_REAL_ROBOT: print(f"True Object Position : ({true_x:.3f}, {true_y:.3f})")
    print(f"Filter Center Est.   : ({final_x:.3f}, {final_y:.3f})")
    print("="*40 + "\n")


    # Create a folder name (optional, helps keep things organized)
    output_folder = "saved_plots"

    # Plot Y
    plot_particle_evolution(particle_filter, axis='y', true_pos=true_y, 
                            min_val=-0.2, max_val=0.2, 
                            save_path=f"{output_folder}/y_axis_evolution.png")

    # Plot X
    plot_particle_evolution(particle_filter, axis='x', true_pos=true_x, 
                            min_val=0.3, max_val=0.6, 
                            save_path=f"{output_folder}/x_axis_evolution.png")

if __name__ == "__main__":
    main()
