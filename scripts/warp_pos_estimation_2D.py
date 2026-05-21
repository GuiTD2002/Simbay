import os
import sys

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco.viewer
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.estimation import BinaryContactMeasurementModel
from src.estimation import ParticleFilterRegularized
from src.estimation import PositionMotionModel
from src.estimation import RobotContainer
from src.robots import RealRobot
from src.robots.warp_mujoco_robot import initialize_mujoco_warp_env
from src.skills import move_to_home
from src.skills.sweep import sweep_until_contact
from src.utils import DEFAULT_OBJECT_PROPS
from src.utils import ViewerGifRecorder
from src.utils import plot_particle_evolution

"""
Example call-site
=================

How to switch your particle filter to the batched Warp backend.

The filter itself (``ParticleFilterRegularized``) is **unchanged**. Only
the container and the motion + measurement models are swapped.
"""

from src.estimation.particle_filter import ParticleFilterRegularized
from src.warp_estimation.warp_container import WarpRobotContainer
from src.warp_estimation.warp_motion import WarpPositionMotionModel
from src.warp_estimation.warp_measurement import WarpBinaryContactMeasurementModel


def build_warp_particle_filter(
    num_particles: int,
    limits,
    object_props: dict,
    dt: float,
    ess_threshold: float,
    *,
    nconmax: int | None = None,
    njmax: int | None = None,
    device: str | None = "cuda:0",
):
    """Construct a Warp-batched particle filter.

    Notes on the new parameters
    ---------------------------
    ``nconmax`` / ``njmax``
        Warp pre-allocates contact and constraint buffers. If you see
        overflow messages from MJWarp during stepping, raise these.
        Sensible starting points: ``nconmax = num_particles * 8``,
        ``njmax = 200``. Tune from there.

    ``device``
        Pass ``"cuda:0"`` for a single-GPU setup. With Ray + 1 GPU there's
        nothing to gain from sharding the filter across devices — keep
        ``device`` fixed and let Ray schedule the actor onto that GPU.
    """
    container = WarpRobotContainer( 
        num_particles=num_particles,
        props=object_props,
        dt=dt,
        nconmax=nconmax,
        njmax=njmax,
        device=device,
    )

    return ParticleFilterRegularized(
        num_particles=num_particles,
        state_bounds=limits,
        motion_model=WarpPositionMotionModel(container),
        measurement_model=WarpBinaryContactMeasurementModel(container),
        ess_threshold_ratio=ess_threshold,
    )


# ---------------------------------------------------------------------------
# Optional: Ray actor wrapper
# ---------------------------------------------------------------------------
#
# With one GPU, Ray's value here is mainly process isolation — keeping the
# Warp kernels off the main process's CUDA context so a viewer thread or
# other GPU consumer doesn't fight for the runtime.
#
# Uncomment if you want this.
#
# import ray
#
# @ray.remote(num_gpus=1)
# class WarpParticleFilterActor:
#     def __init__(self, *args, **kwargs):
#         self.pf = build_warp_particle_filter(*args, **kwargs)
#
#     def step(self, control_input, observation, current_state):
#         self.pf.step(control_input, observation, current_state)
#         return self.pf.estimate()
#
#     def reset(self, state):
#         self.pf.reset(state)

# ==========================================
# VIEWER / GIF QUALITY
# ==========================================
GIF_PATH = "saved_plots/gpu_pos_estimation_2D.gif"
GIF_WIDTH = 1280
GIF_HEIGHT = 960
GIF_FPS = 10
GIF_INTERVAL = 10
VIEWER_CAMERA_NAME = "frontal"
VIEWER_CAMERA_X = 3

# ==========================================
# CONFIGURATION
# ==========================================
USE_REAL_ROBOT = False
HEADLESS = True

NUM_PARTICLES = 100
ESS_THRESHOLD = 0.5

# Workspace Limits (X, Y)
MIN_X, MAX_X = 0.5, 0.6
MIN_Y, MAX_Y = 0.1, 0.2

# Sweep Parameters
FIXED_Z = 0.08
MAX_BLOCK_HALF_SIZE = 0.125
SAFETY_DISTANCE = 0.01
SWEEP_VEL = 0.5

def track_ground_truth(robot):
    if USE_REAL_ROBOT: return None, None
    if hasattr(robot, "sync_host"):
        robot.sync_host()
    block_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, 'object') # type: ignore
    return robot.data.xpos[block_id][0], robot.data.xpos[block_id][1]


def configure_named_camera_x(robot, camera_name, x):
    camera_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name) # type: ignore
    if camera_id < 0:
        raise ValueError(f"Camera '{camera_name}' was not found in the MuJoCo model.")
    robot.model.cam_pos[camera_id, 0] = float(x)
    return int(camera_id)


def use_fixed_viewer_camera(viewer, camera_id):
    if viewer is None:
        return
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED  # type: ignore
    viewer.cam.fixedcamid = int(camera_id)

def main():
    print("Initializing Environment...")
    if USE_REAL_ROBOT:
        robot = RealRobot()
        robot.dt = 0.001
    else:
        robot = initialize_mujoco_warp_env(dt=0.001, device="cuda")
        robot.sync_host()
        if not HEADLESS and os.environ.get("DISPLAY"):
            viewer = mujoco.viewer.launch_passive(robot.model, robot.data)
            robot.viewer = viewer

    true_x, true_y = track_ground_truth(robot)
    if not USE_REAL_ROBOT: print(f"🛑 [Debug] Initial Ground Truth: X={true_x:.3f}, Y={true_y:.3f}")

    limits = (np.array([MIN_X, MIN_Y]), np.array([MAX_X, MAX_Y]))
    container = RobotContainer(num_particles=NUM_PARTICLES, props=DEFAULT_OBJECT_PROPS, dt=robot.dt)

    # particle_filter = ParticleFilterRegularized(
    #     num_particles=NUM_PARTICLES, state_bounds=limits,
    #     motion_model=PositionMotionModel(container),
    #     measurement_model=BinaryContactMeasurementModel(container),
    #     ess_threshold_ratio=ESS_THRESHOLD
    # )

    particle_filter = build_warp_particle_filter(
        num_particles=NUM_PARTICLES,
        limits=limits,
        object_props=DEFAULT_OBJECT_PROPS,
        dt=robot.dt,
        ess_threshold=ESS_THRESHOLD,
        nconmax=NUM_PARTICLES * 8,
        njmax=300,
        device="cuda:0", # not simbay specific - this is what tells the MujucoWarp it to use the GPU
    )

    gif_recorder = ViewerGifRecorder(
        save_path=GIF_PATH,
        capture_interval=GIF_INTERVAL,
        fps=GIF_FPS,
        width=GIF_WIDTH,
        height=GIF_HEIGHT,
        particle_filter=particle_filter,
        camera_name=VIEWER_CAMERA_NAME,
        camera_x=VIEWER_CAMERA_X,
    )

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
        gif_recorder=gif_recorder,
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
        gif_recorder=gif_recorder,
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
        gif_recorder=gif_recorder,
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
        gif_recorder=gif_recorder,
    )

    if not USE_REAL_ROBOT: print(f"🛑 [Debug] Ground Truth After Swipe 4: X={true_x:.3f}, Y={true_y:.3f}")

    # ==========================================
    # FINISH & RESULTS
    # ==========================================
    move_to_home(robot)

    # Final estimate mathematically perfectly balanced by the padded bounds
    final_x = particle_filter.estimate()[0]
    final_y = particle_filter.estimate()[1]

    print("\n" + "="*40)
    print("FINAL 2D ESTIMATION RESULTS")
    print("="*40)
    if not USE_REAL_ROBOT: print(f"True Object Position : ({true_x:.3f}, {true_y:.3f})")
    print(f"Filter Center Est.   : ({final_x:.3f}, {final_y:.3f})")
    print("="*40 + "\n")

    gif_recorder.save()
    gif_recorder.close()

    # Create a folder name (optional, helps keep things organized)
    output_folder = "saved_plots"

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
