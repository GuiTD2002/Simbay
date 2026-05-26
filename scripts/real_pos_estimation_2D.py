import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import mujoco.viewer
import numpy as np

from src.warp_estimation.warp_container import WarpRobotContainer
from src.warp_estimation.warp_measurement import WarpBinaryContactMeasurementModel
from src.warp_estimation.warp_motion import WarpPositionMotionModel
from src.warp_estimation.warp_particle_filter import build_ray_warp_particle_filter
from src.warp_estimation.warp_particle_filter import build_warp_particle_filter
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
from src.skills.real_sweep import real_sweep_until_contact
from src.utils import FRANKA_HOME_QPOS
from src.skills import click_button

import rclpy

# Import the new class you just added to real_sweep.py!
from src.skills.real_sweep import real_sweep_until_contact, ParticleVisualizer

# ==========================================
# RAY REMOTE COMPUTE (Optional GPU acceleration)
# ==========================================
# Use Ray for distributed GPU compute on remote machine.
# Disable to run locally: set USE_RAY=False  
# note: if USE_RAY=false it will use the CPU which is slower on the MujocoWarp so for testing/development <50 particles use the pos_estimation_2d.py
# script because it will run a small amount of particles faster. 
USE_RAY = True
RAY_ADDRESS = f"ray://{os.environ.get('SIMBAY_RAY_IP', 'localhost')}:10002"
RAY_NUM_GPUS = 1.0
RAY_DEBUG = True
WARP_DEVICE = "cuda:0" if USE_RAY else "cpu" # use the gpu on the remote computer 

# ==========================================
# CONFIGURATION
# ==========================================
USE_REAL_ROBOT = True

# Filter Configuration
NUM_PARTICLES = 1500
ESS_THRESHOLD = 0.5

# Workspace Limits (Y)
MIN_Y, MAX_Y = 0.10, 0.20
MIN_X, MAX_X = 0.5, 0.6

# Sweep Parameters
FIXED_Z = 0.08
MAX_BLOCK_HALF_SIZE_Y = 0.075 
MAX_BLOCK_HALF_SIZE_X = 0.125
SAFETY_DISTANCE = 0.02
SWEEP_VEL = 0.01


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
        robot.viewer = mujoco.viewer.launch_passive(robot.model, robot.data)

    initial_pos = robot.get_joints_pos() 
    if not np.all(initial_pos == FRANKA_HOME_QPOS):
        move_to_home(robot, velocity=0.3) 

    # --- NEW: INITIALIZE ROS 2 VISUALIZER ---
    print("Initializing RViz Particle Visualizer...")
    if not rclpy.ok():
        rclpy.init()
    rviz_node = ParticleVisualizer()
    
    true_x, true_y = track_ground_truth(robot)
    if not USE_REAL_ROBOT: print(f"[Debug] Initial Ground Truth: Y={true_y:.3f}")

    limits = (np.array([MIN_X, MIN_Y]), np.array([MAX_X, MAX_Y]))

    # uncomment to run the original version (version 2026-05-20, master)
    # container = RobotContainer(num_particles=NUM_PARTICLES, props=DEFAULT_OBJECT_PROPS, dt=robot.dt)
    
    # particle_filter = ParticleFilterRegularized(
    #     num_particles=NUM_PARTICLES, state_bounds=limits, 
    #     motion_model=PositionMotionModel(container), 
    #     measurement_model=BinaryContactMeasurementModel(container), 
    #     ess_threshold_ratio=ESS_THRESHOLD
    # )

    # new version 2026-05-22: Ray-enabled particle filter
    pf_kwargs = {
        "num_particles": NUM_PARTICLES,
        "limits": limits,
        "object_props": DEFAULT_OBJECT_PROPS,
        "dt": robot.dt,
        "ess_threshold": ESS_THRESHOLD,
        "nconmax": 64,           # per-world contacts (mjw asked for >=29, doubled for headroom)
        "njmax": 512,            # per-world constraint rows (mjw asked for ~250: nefc overflow, doubled for headroom)
        # mjw constraint: naccdmax <= naconmax (every CCD pair becomes a contact),
        # so nccdmax <= nconmax.
        "nccdmax": 64,
        "ccd_iterations": 12,    # shrinks EPA buffer width vs MuJoCo's higher default
        "device": WARP_DEVICE,
    }

    if USE_RAY:
        particle_filter = build_ray_warp_particle_filter(
            **pf_kwargs,
            num_gpus=RAY_NUM_GPUS,
            ray_address=RAY_ADDRESS,
            debug=RAY_DEBUG,
        )
    else:
        particle_filter = build_warp_particle_filter(**pf_kwargs)


    # ==========================================
    # PHASE 1: SWEEP FORWARD (+Y)
    # ==========================================
    SWEEP_QUAT = np.array([0.0, 1.0, 0.0, 0.0])
    FIXED_X = (MIN_X + MAX_X) / 2.0
    print("\n--- Phase 1: Sweep Forward (+Y) ---")
    start_pos_y1 = np.array([FIXED_X, MIN_Y - MAX_BLOCK_HALF_SIZE_Y - SAFETY_DISTANCE, FIXED_Z])
    end_pos_y1 = np.array([FIXED_X, MAX_Y, FIXED_Z])

    real_sweep_until_contact(
        robot=robot, particle_filter=particle_filter, start_pos=start_pos_y1,
        end_pos=end_pos_y1, target_quat=SWEEP_QUAT, sweep_vel=SWEEP_VEL,
        safety_distance=SAFETY_DISTANCE, visualize=not USE_REAL_ROBOT,
        visualizer = rviz_node
    )

    if not USE_REAL_ROBOT: print(f"🛑 Ground Truth After Swipe 1: {track_ground_truth(robot):.3f}")


    # ==========================================
    # PHASE 2: SWEEP BACKWARD (-Y)
    # ==========================================
    
    print("\n--- Phase 2: Sweep Backward (-Y) ---")
    start_pos_y2 = np.array([FIXED_X, MAX_Y + MAX_BLOCK_HALF_SIZE_Y + SAFETY_DISTANCE, FIXED_Z])
    end_pos_y2 = np.array([FIXED_X, MIN_Y, FIXED_Z])

    real_sweep_until_contact(
        robot=robot, particle_filter=particle_filter, start_pos=start_pos_y2,
        end_pos=end_pos_y2, target_quat=SWEEP_QUAT, sweep_vel=SWEEP_VEL,
        safety_distance=SAFETY_DISTANCE, visualize=not USE_REAL_ROBOT,
        visualizer = rviz_node
    )

    if not USE_REAL_ROBOT: print(f"🛑 Ground Truth After Swipe 2: {track_ground_truth(robot):.3f}")
    

    # ==========================================
    # PHASE 3: SWEEP BACKWARD (-X)
    # ==========================================
    
    print("\n--- Phase 3: Sweep Backward (-X) ---")
    SWEEP_QUAT = np.array([0.0, np.sqrt(2)/2, np.sqrt(2)/2, 0.0])
    FIXED_Y = particle_filter.estimate()[1]  # Use the best Y estimate from Phase 1 and 2
    start_pos_x2 = np.array([MAX_X + MAX_BLOCK_HALF_SIZE_X + SAFETY_DISTANCE, FIXED_Y, FIXED_Z])
    end_pos_x2 = np.array([MIN_X, FIXED_Y, FIXED_Z])

    real_sweep_until_contact(
        robot=robot, particle_filter=particle_filter, start_pos=start_pos_x2,
        end_pos=end_pos_x2, target_quat=SWEEP_QUAT, sweep_vel=SWEEP_VEL,
        safety_distance=SAFETY_DISTANCE, visualize=not USE_REAL_ROBOT,
        visualizer = rviz_node
    )                           

    if not USE_REAL_ROBOT: print(f"🛑 Ground Truth After Swipe 3: {track_ground_truth(robot):.3f}")
    
    
    # ==========================================
    # PHASE 4: SWEEP FORWARD (+X)
    # ==========================================
    print("\n--- Phase 4: Sweep Forward (+X) ---")
    #FIXED_Y = particle_filter.estimate()[1] 
    SWEEP_QUAT = np.array([0.0, np.sqrt(2)/2, np.sqrt(2)/2, 0.0])
    start_pos_x1 = np.array([MIN_X - MAX_BLOCK_HALF_SIZE_X - SAFETY_DISTANCE, FIXED_Y, FIXED_Z])
    end_pos_x1 = np.array([MAX_X, FIXED_Y, FIXED_Z])            

    real_sweep_until_contact(
        robot=robot, particle_filter=particle_filter, start_pos=start_pos_x1,
        end_pos=end_pos_x1, target_quat=SWEEP_QUAT, sweep_vel=SWEEP_VEL,
        safety_distance=SAFETY_DISTANCE, visualize=not USE_REAL_ROBOT,
        visualizer = rviz_node
    )       

    if not USE_REAL_ROBOT: print(f"🛑 Ground Truth After Swipe 4: {track_ground_truth(robot):.3f}")
    
    # ==========================================
    # PHASE 5: EXTRACTION & HOMING
    # ==========================================
    # Extract purely from the geometric center of our padded bounds
    final_x = particle_filter.estimate()[0]
    final_y = particle_filter.estimate()[1]     

    print(f"Final Estimate: X={final_x:.3f}, Y={final_y:.3f}")
    obj_pos = np.array([final_x, final_y, 0.05])
    click_button(robot, obj_pos, velocity=0.1)
    move_to_home(robot)
    robot.shutdown()

    # --- NEW: SHUTDOWN ROS 2 AT THE VERY END ---
    if rclpy.ok():
        rviz_node.destroy_node()
        rclpy.shutdown()

    print("\n" + "="*40)


    print("\n" + "="*40)
    print("FINAL 2D ESTIMATION RESULTS")
    print("="*40)
    #print(f"True Object Position : {true_y:.3f}")
    #print(f"Filter Center Est.   : {final_y:.3f}")
    print("="*40 + "\n")

    # Create a folder name (optional, helps keep things organized)
    output_folder = "saved_plots"

    if USE_RAY:
        # does one call to the remote to return the entire history
        particle_filter.get_history()


    # Plot Y
    plot_particle_evolution(particle_filter, axis='y', true_pos=true_y,
                            min_val=MIN_Y, max_val=MAX_Y,
                            save_path=f"{output_folder}/y_axis_evolution.png")
    
    # Plot X
    plot_particle_evolution(particle_filter, axis='x', true_pos=true_x,
                            min_val=MIN_X, max_val=MAX_X,
                            save_path=f"{output_folder}/x_axis_evolution.png")
    
    

if __name__ == "__main__":
    main()
