import os

import matplotlib.pyplot as plt
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np


class ViewerGifRecorder:
    def __init__(
        self,
        save_path,
        capture_interval=200,
        fps=10,
        width=640,
        height=480,
        enabled=True,
        particle_filter=None,
        camera_name=None,
        camera_x=None,
    ):
        self.save_path = save_path
        self.capture_interval = capture_interval
        self.fps = fps
        self.width = width
        self.height = height
        self.enabled = enabled
        self.particle_filter = particle_filter
        self.camera_name = camera_name
        self.camera_x = camera_x
        self.frames = []
        self.step_count = 0
        self._renderer = None

    def capture(self, robot):
        if not self.enabled:
            return

        self.step_count += 1
        if self.step_count % self.capture_interval != 0:
            return

        frame, self._renderer = capture_viewer_frame(
            robot,
            renderer=self._renderer,
            width=self.width,
            height=self.height,
            particle_filter=self.particle_filter,
            camera_name=self.camera_name,
        )
        if frame is not None:
            self.frames.append(frame)

    def save(self):
        save_viewer_gif(self.frames, self.save_path, fps=self.fps)

    def close(self):
        if self._renderer is not None:
            if hasattr(self._renderer, "close"):
                self._renderer.close()
            self._renderer = None


def capture_viewer_frame(robot, renderer=None, width=640, height=480, particle_filter=None, camera_name=None, camera_x=None):
    """
    Captures the robot's current MuJoCo scene as an RGB frame.

    If a passive viewer is attached, its camera and scene options are reused.
    For MJWarp robots, host data is synchronized before rendering.
    """
    if getattr(robot, "model", None) is None or getattr(robot, "data", None) is None:
        return None, renderer

    if hasattr(robot, "sync_host"):
        robot.sync_host()

    if renderer is None:
        renderer = _create_renderer(robot.model, width=width, height=height)

    viewer = getattr(robot, "viewer", None)
    camera = getattr(viewer, "cam", None)
    scene_option = getattr(viewer, "opt", None)

    try:
        if camera_name is not None:
            if camera_x is not None:
                camera_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
                if camera_id >= 0:
                    robot.model.cam_pos[camera_id, 0] = float(camera_x)
            renderer.update_scene(robot.data, camera=camera_name)
        else:
            renderer.update_scene(robot.data, camera=camera, scene_option=scene_option)
    except (TypeError, ValueError):
        if camera_name is not None:
            renderer.update_scene(robot.data, camera=camera_name)
        elif camera is not None:
            renderer.update_scene(robot.data, camera=camera)
        else:
            renderer.update_scene(robot.data)

    if particle_filter is not None:
        draw_particle_geoms(
            renderer.scene,
            particle_filter.particles,
            particle_filter.weights,
            fixed_z=0.04,
            fixed_x=0.55,
            clear=False,
        )

    return renderer.render().copy(), renderer


def draw_particle_geoms(scene, particles, weights, fixed_z=0.04, fixed_x=0.55, clear=False):
    """
    Draw particle markers into an MjvScene.
    """
    if scene is None or len(particles) == 0:
        return

    num_p = len(particles)
    dim = particles.shape[1] if len(particles.shape) > 1 else 1

    max_w = np.max(weights) if np.max(weights) > 1e-10 else 1e-10
    norm_w = np.clip(weights / max_w, 0.0, 1.0)

    rgba = np.zeros((num_p, 4))
    rgba[:, 3] = 1.0
    mask = norm_w < 0.5
    rgba[mask, 1] = norm_w[mask] * 2.0
    rgba[mask, 2] = 1.0 - (norm_w[mask] * 2.0)
    rgba[~mask, 0] = (norm_w[~mask] - 0.5) * 2.0
    rgba[~mask, 1] = 1.0 - ((norm_w[~mask] - 0.5) * 2.0)

    pos_3d = np.zeros((num_p, 3))
    if dim == 1:
        pos_3d[:, 0] = fixed_x
        pos_3d[:, 1] = particles[:, 0]
    else:
        pos_3d[:, 0] = particles[:, 0]
        pos_3d[:, 1] = particles[:, 1]
    pos_3d[:, 2] = fixed_z

    mat_3d = np.zeros((num_p, 9))
    if dim == 3:
        thetas = particles[:, 2]
        cos_t = np.cos(thetas)
        sin_t = np.sin(thetas)
        mat_3d[:, 0] = cos_t
        mat_3d[:, 1] = -sin_t
        mat_3d[:, 3] = sin_t
        mat_3d[:, 4] = cos_t
        mat_3d[:, 8] = 1.0
        geom_type = mujoco.mjtGeom.mjGEOM_BOX            # type: ignore
        geom_size = np.array([0.015, 0.002, 0.002])
        # rgba[:, 3] = 0.2
    else:
        mat_3d[:, 0] = 1.0
        mat_3d[:, 4] = 1.0
        mat_3d[:, 8] = 1.0
        geom_type = mujoco.mjtGeom.mjGEOM_SPHERE         # type: ignore
        geom_size = np.array([0.005, 0.0, 0.0])

    if clear:
        scene.ngeom = 0

    for i in range(num_p):
        if scene.ngeom >= scene.maxgeom:
            break
        mujoco.mjv_initGeom(                              # type: ignore
            scene.geoms[scene.ngeom],
            type=geom_type,
            size=geom_size,
            pos=pos_3d[i],
            mat=mat_3d[i],
            rgba=rgba[i],
        )
        scene.ngeom += 1


def _create_renderer(model, width=640, height=480):
    """
    Create a MuJoCo renderer that works in both interactive and headless runs.

    In headless mode MuJoCo still needs an OpenGL context before Renderer is
    constructed, so we create a small offscreen context and keep it alive on the
    renderer object.
    """
    if model.vis.global_.offwidth < width:
        model.vis.global_.offwidth = width
    if model.vis.global_.offheight < height:
        model.vis.global_.offheight = height
    return mujoco.Renderer(model, height=height, width=width)


def save_viewer_gif(frames, save_path, fps=10):
    """
    Saves captured RGB frames to a GIF.
    """
    if not frames:
        print(f"[GIF] No viewer frames captured; skipping {save_path}")
        return

    output_dir = os.path.dirname(save_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        from PIL import Image
    except ImportError as error:
        raise ImportError("Saving viewer GIFs requires Pillow (`pip install pillow`).") from error

    duration_ms = int(1000 / fps)
    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(
        save_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )
    print(f"[GIF] Saved {len(frames)} viewer frames to: {save_path}")


def visualize_particles(viewer, particles, weights, fixed_z=0.04, fixed_x=0.55):
    """
    Optimized & Dimension-Agnostic: Renders spheres for 1D/2D, 
    and oriented directional markers (boxes) for 3DOF.
    """
    if viewer is None or len(particles) == 0:
        return

    draw_particle_geoms(viewer.user_scn, particles, weights, fixed_z=fixed_z, fixed_x=fixed_x, clear=True)


def plot_particle_evolution(
    particle_filter,
    dimension=0,
    ylabel=None,
    axis=None,
    true_pos=None,
    min_val=None,
    max_val=None,
    save_path=None,
):
    """
    Universal plotter for 1D, 2D, or 3DOF particle filters.
    """
    # 1. Convert history to numpy arrays
    particles_np = np.array(particle_filter.history['particles']) 
    estimates_np = np.array(particle_filter.history['estimates']) 
    weights_np = np.array(particle_filter.history['weights'])     

    if len(particles_np) == 0:
        print("[Plotting] Warning: No data in filter history to plot.")
        return

    num_steps = len(particles_np)
    ess_vals = np.asarray(particle_filter.history.get('ess', []), dtype=float)
    if ess_vals.size == 0:
        ess_vals = 1.0 / np.sum(weights_np**2, axis=1)
    
    # ==========================================
    # 2. AXIS CONFIGURATION
    # ==========================================
    if axis is not None:
        axis = axis.lower()
        title_name = f"{axis.upper()}-Axis"
        if axis == 'x':
            dimension = 0
            if ylabel is None: ylabel = 'Estimated X-Position (meters)'
        elif axis == 'y':
            dimension = 1
            if ylabel is None: ylabel = 'Estimated Y-Position (meters)'
        elif axis == 'theta':
            dimension = 2
            if ylabel is None: ylabel = 'Estimated Angle (degrees)'
        else:
            print("Invalid axis. Choose 'x', 'y', or 'theta'.")
            return
    else:
        title_name = f"Dimension {dimension}"
        if ylabel is None:
            ylabel = f'Estimated Value (Dim {dimension})'
    
    print(f"[Plotting] Generating Evolution Plot for {title_name}...")
    
    idx = dimension
        
    # ==========================================
    # 3. DIMENSION-AGNOSTIC EXTRACTION
    # ==========================================
    if particles_np.ndim == 2:
        num_particles = particles_np.shape[1]
        p_vals = particles_np.flatten()
        e_vals = estimates_np.flatten()
    else:
        num_particles = particles_np.shape[1]
        dim = particles_np.shape[2]
        
        # ---> THE 1D FAIL-SAFE <---
        # If the filter is 1D, ignore the axis string and force it to look at index 0
        if dim == 1:
            idx = 0
            
        if idx >= dim:
            print(f"Error: Cannot plot {title_name}. Data only has {dim} dimensions.")
            return
            
        p_vals = particles_np[:, :, idx].flatten()
        e_vals = estimates_np if estimates_np.ndim == 1 else estimates_np[:, idx]

    # ==========================================
    # 4. THETA CONVERSION (Radians to Degrees)
    # ==========================================
    if axis == 'theta' and dim > 2:
        p_vals = np.degrees(p_vals)
        e_vals = np.degrees(e_vals)

    c_vals = weights_np.flatten()
    x_vals = np.repeat(np.arange(num_steps), num_particles)

    # ==========================================
    # 5. MATPLOTLIB RENDERING
    # ==========================================
    fig, (ax_main, ax_ess) = plt.subplots(
        2,
        1,
        figsize=(12, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1]},
    )
    sc = ax_main.scatter(x_vals, p_vals, c=c_vals, cmap='viridis',
                         alpha=0.6, s=15, edgecolors='none')
    cbar = fig.colorbar(sc, ax=ax_main)
    cbar.set_label('Particle Weight (Probability)', fontsize=12, fontweight='bold')

    ax_main.plot(range(num_steps), e_vals, color='red', linewidth=3, label='Filter Estimate (Mean)')
    
    if true_pos is not None:
        ax_main.axhline(y=true_pos, color='red', linestyle='--', linewidth=2, label=f'True value ({true_pos:.3f})')

    ax_main.set_title(f'Particle Filter: {title_name} Evolution ({num_particles} particles)', fontsize=14, fontweight='bold')
    ax_main.set_ylabel(ylabel, fontsize=12)
    
    if min_val is not None and max_val is not None:
        ax_main.set_ylim(min_val, max_val)
        
    ess_threshold = particle_filter.N * particle_filter.ess_threshold_ratio
    ax_ess.plot(range(len(ess_vals)), ess_vals, color="teal", linewidth=2, label="ESS")
    ax_ess.axhline(y=ess_threshold, color="gray", linestyle="--", linewidth=1.5, label="Resampling Threshold")
    ax_ess.set_ylabel("ESS", fontsize=12)
    ax_ess.set_xlabel("Simulation Step", fontsize=12)

    ax_main.legend(loc='upper right')
    ax_ess.legend(loc='upper right')
    ax_main.grid(True, linestyle=':', alpha=0.7)
    ax_ess.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Plotting] Saved {title_name} plot to: {save_path}")
        
    plt.show()
        
        
