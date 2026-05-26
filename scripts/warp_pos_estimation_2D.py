import os
import sys

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco.viewer
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.robots import RealRobot
from src.skills import move_to_home
from src.skills.sweep import sweep_until_contact
from src.utils import DEFAULT_OBJECT_PROPS
from src.utils import initialize_mujoco_env
from src.utils import plot_particle_evolution
from src.warp_estimation.warp_particle_filter import RayWarpParticleFilter, build_ray_warp_particle_filter
from src.warp_estimation.warp_particle_filter import build_warp_particle_filter

# ==========================================
# RAY REMOTE COMPUTE (Optional GPU acceleration)
# ==========================================
# Use Ray for distributed GPU compute on remote machine.
# Disable to run locally: set USE_RAY=False
# note: if USE_RAY=false it will use the CPU which is slower on the MujocoWarp so for testing/development use the pos_estimation_2d.py
# script because it will run a small amount of particles faster. so use USE_RAY=true this with 200+ particles
USE_RAY = True
USE_GPU = True
RAY_ADDRESS = f"ray://{os.environ.get('SIMBAY_RAY_IP', 'localhost')}:10002"
RAY_NUM_GPUS = 1.0 
RAY_DEBUG = True # print ray orchestraion logs + remote worker logs
WARP_DEVICE = "cuda:0" if USE_GPU else "cpu" # use the gpu on the remote(USE_RAY=True) or local(USE_RAY=False) computer

# ==========================================
# CONFIGURATION
# ==========================================
USE_REAL_ROBOT = False
HEADLESS = False
NUM_PARTICLES = 1000
ESS_THRESHOLD = 0.5

# Workspace Limits (X, Y)
MIN_X, MAX_X = 0.5, 0.6
MIN_Y, MAX_Y = 0.0, 0.1

# Sweep Parameters
FIXED_Z = 0.09
MAX_BLOCK_HALF_SIZE = 0.125
SAFETY_DISTANCE = 0.01
SWEEP_VEL = 0.01

def track_ground_truth(robot):
    if USE_REAL_ROBOT: return None, None
    if hasattr(robot, "sync_host"):
        robot.sync_host()
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
        if not HEADLESS and os.environ.get("DISPLAY"):
            viewer = mujoco.viewer.launch_passive(robot.model, robot.data)
            robot.viewer = viewer

    true_x, true_y = track_ground_truth(robot)
    if not USE_REAL_ROBOT: print(f"🛑 [Debug] Initial Ground Truth: X={true_x:.3f}, Y={true_y:.3f}")

    limits = (np.array([MIN_X, MIN_Y]), np.array([MAX_X, MAX_Y]))

    # Initialize particle filter with Ray or local (mujoco warp)
    # NOTE: nconmax/njmax/nccdmax are PER-WORLD. mjw multiplies by nworld
    # (= NUM_PARTICLES) internally, so e.g. nconmax=8 ⇒ 8 * 500 = 4000 total
    # contact slots. Setting these to total-counts blows up GPU memory (the
    # EPA buffer in convex_narrowphase scales as naccdmax * ccd_iterations).
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

    mid_x = (MIN_X + MAX_X) / 2.0
    quat_y = np.array([0.0, 1.0, 0.0, 0.0])
    quat_x = np.array([0.0, np.sqrt(2)/2, np.sqrt(2)/2, 0.0])

    # ==========================================
    # PHASE 1: SWEEP FORWARD (+Y)
    # ==========================================
    print("\n--- Phase 1: Sweep Forward (+Y) ---")
    start_pos_y1 = np.array([mid_x, MIN_Y - MAX_BLOCK_HALF_SIZE - SAFETY_DISTANCE, FIXED_Z])
    end_pos_y1 = np.array([mid_x, MAX_Y, FIXED_Z])

    sweep_until_contact(
        robot=robot, particle_filter=particle_filter, start_pos=start_pos_y1,
        end_pos=end_pos_y1, target_quat=quat_y, sweep_vel=SWEEP_VEL,
        safety_distance=SAFETY_DISTANCE, visualize=not USE_REAL_ROBOT and not HEADLESS,
    )

    if not USE_REAL_ROBOT: print(f"🛑 [Debug] Ground Truth After Swipe 1: X={true_x:.3f}, Y={true_y:.3f}")

    # After Phase 1 sweep_until_contact finishes
    max_y = np.max(particle_filter.particles[:, 1])
    print(f"\n🛑 POST-PHASE 1 CHECK: The highest surviving particle is at Y = {max_y:.3f}")


    # ==========================================
    # PHASE 2: SWEEP BACKWARD (-Y)
    # ==========================================
    print("\n--- Phase 2: Sweep Backward (-Y) ---")
    start_pos_y2 = np.array([mid_x, MAX_Y + MAX_BLOCK_HALF_SIZE + SAFETY_DISTANCE, FIXED_Z])
    end_pos_y2 = np.array([mid_x, MIN_Y, FIXED_Z])

    sweep_until_contact(
        robot=robot, particle_filter=particle_filter, start_pos=start_pos_y2,
        end_pos=end_pos_y2, target_quat=quat_y, sweep_vel=SWEEP_VEL,
        safety_distance=SAFETY_DISTANCE, visualize=not USE_REAL_ROBOT and not HEADLESS,
    )

    if not USE_REAL_ROBOT: print(f"🛑 [Debug] Ground Truth After Swipe 2: X={true_x:.3f}, Y={true_y:.3f}")


    # ==========================================
    # PHASE 3: SWEEP BACKWARD (-X)
    # ==========================================
    print("\n--- Phase 3: Sweep Backward (-X) ---")
    estimate_y = particle_filter.estimate()[1]
    start_pos_x1 = np.array([MAX_X + MAX_BLOCK_HALF_SIZE + SAFETY_DISTANCE, estimate_y, FIXED_Z])
    end_pos_x1 = np.array([MIN_X, estimate_y, FIXED_Z])

    sweep_until_contact(
        robot=robot, particle_filter=particle_filter, start_pos=start_pos_x1,
        end_pos=end_pos_x1, target_quat=quat_x, sweep_vel=SWEEP_VEL,
        safety_distance=SAFETY_DISTANCE, visualize=not USE_REAL_ROBOT and not HEADLESS,
    )

    if not USE_REAL_ROBOT: print(f"🛑 [Debug] Ground Truth After Swipe 3: X={true_x:.3f}, Y={true_y:.3f}")


    # ==========================================
    # PHASE 4: SWEEP FORWARD (+X)
    # ==========================================
    print("\n--- Phase 4: Sweep Forward (+X) ---")
    estimate_y = particle_filter.estimate()[1]
    start_pos_x2 = np.array([MIN_X - MAX_BLOCK_HALF_SIZE - SAFETY_DISTANCE, estimate_y, FIXED_Z])
    end_pos_x2 = np.array([MAX_X, estimate_y, FIXED_Z])

    sweep_until_contact(
        robot=robot, particle_filter=particle_filter, start_pos=start_pos_x2,
        end_pos=end_pos_x2, target_quat=quat_x, sweep_vel=SWEEP_VEL,
        safety_distance=SAFETY_DISTANCE, visualize=not USE_REAL_ROBOT and not HEADLESS,
    )

    if not USE_REAL_ROBOT: print(f"🛑 [Debug] Ground Truth After Swipe 4: X={true_x:.3f}, Y={true_y:.3f}")

    # ==========================================
    # FINISH & RESULTS
    # ==========================================
    move_to_home(robot, real=USE_REAL_ROBOT)

    final_x = particle_filter.estimate()[0]
    final_y = particle_filter.estimate()[1]

    print("\n" + "="*40)
    print("FINAL 2D ESTIMATION RESULTS")
    print("="*40)
    if not USE_REAL_ROBOT: print(f"True Object Position : ({true_x:.3f}, {true_y:.3f})")
    print(f"Filter Center Est.   : ({final_x:.3f}, {final_y:.3f})")
    print("="*40 + "\n")

    output_folder = "saved_plots"

    if USE_RAY:
        # does one call to the remote to return the entire history
        particle_filter.get_history()

    plot_particle_evolution(particle_filter, axis='y', true_pos=true_y,
                            min_val=MIN_Y, max_val=MAX_Y,
                            save_path=f"{output_folder}/y_axis_evolution.png")

    plot_particle_evolution(particle_filter, axis='x', true_pos=true_x,
                            min_val=MIN_X, max_val=MAX_X,
                            save_path=f"{output_folder}/x_axis_evolution.png")

    if isinstance(particle_filter, RayWarpParticleFilter):
        particle_filter.close()

if __name__ == "__main__":
    main()
