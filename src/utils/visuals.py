import os

import matplotlib.pyplot as plt
import mujoco
import numpy as np


def visualize_particles(viewer, particles, weights, fixed_z=0.04, fixed_x=0.55):
    """
    Optimized & Dimension-Agnostic: Renders spheres for 1D/2D, 
    and oriented directional markers (boxes) for 3DOF.
    """
    if viewer is None or len(particles) == 0:
        return

    viewer.user_scn.ngeom = 0 
    num_p = len(particles)
    dim = particles.shape[1] if len(particles.shape) > 1 else 1
    
    # ==========================================
    # 1. VECTORIZED COLOR MATH
    # ==========================================
    max_w = np.max(weights) if np.max(weights) > 1e-10 else 1e-10
    norm_w = np.clip(weights / max_w, 0.0, 1.0)
    
    rgba = np.zeros((num_p, 4))
    rgba[:, 3] = 1.0 
    
    mask = norm_w < 0.5
    rgba[mask, 1] = norm_w[mask] * 2.0         
    rgba[mask, 2] = 1.0 - (norm_w[mask] * 2.0) 
    rgba[~mask, 0] = (norm_w[~mask] - 0.5) * 2.0     
    rgba[~mask, 1] = 1.0 - ((norm_w[~mask] - 0.5) * 2.0) 

    # ==========================================
    # 2. VECTORIZED POSITION MATH
    # ==========================================
    pos_3d = np.zeros((num_p, 3))
    
    if dim == 1:
        pos_3d[:, 0] = fixed_x
        pos_3d[:, 1] = particles[:, 0]
    else:
        pos_3d[:, 0] = particles[:, 0]
        pos_3d[:, 1] = particles[:, 1]
        
    pos_3d[:, 2] = fixed_z 

    # ==========================================
    # 3. VECTORIZED ROTATION MATH (The 3DOF Upgrade)
    # ==========================================
    # MuJoCo expects a flat 9-element rotation matrix [R11, R12, R13, R21...]
    mat_3d = np.zeros((num_p, 9))
    
    if dim == 3:
        # Calculate Sin and Cos for all particles at once
        thetas = particles[:, 2]
        cos_t = np.cos(thetas)
        sin_t = np.sin(thetas)
        
        # Fill the Z-axis rotation matrix
        mat_3d[:, 0] = cos_t   # R11
        mat_3d[:, 1] = -sin_t  # R12
        mat_3d[:, 3] = sin_t   # R21
        mat_3d[:, 4] = cos_t   # R22
        mat_3d[:, 8] = 1.0     # R33 (Z-axis scale)
        
        geom_type = mujoco.mjtGeom.mjGEOM_BOX           # type: ignore
        geom_size = np.array([0.015, 0.002, 0.002])     # Thin directional needle
    else:
        # 1D/2D: Just use standard Identity Matrix
        mat_3d[:, 0] = 1.0
        mat_3d[:, 4] = 1.0
        mat_3d[:, 8] = 1.0
        
        geom_type = mujoco.mjtGeom.mjGEOM_SPHERE        # type: ignore
        geom_size = np.array([0.005, 0.0, 0.0])         # Standard sphere

    # ==========================================
    # 4. FAST RENDERING LOOP
    # ==========================================
    for i in range(num_p):
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
            break
            
        mujoco.mjv_initGeom(                            # type: ignore
            viewer.user_scn.geoms[viewer.user_scn.ngeom],
            type=geom_type,
            size=geom_size,
            pos=pos_3d[i],
            mat=mat_3d[i], 
            rgba=rgba[i] 
        )
        viewer.user_scn.ngeom += 1


def plot_particle_evolution(particle_filter, dimension=0, ylabel=None, axis=None, true_pos=None, min_val=None, max_val=None, save_path=None):
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
    plt.figure(figsize=(12, 7))
    sc = plt.scatter(x_vals, p_vals, c=c_vals, cmap='viridis', 
                     alpha=0.6, s=15, edgecolors='none')
    cbar = plt.colorbar(sc)
    cbar.set_label('Particle Weight (Probability)', fontsize=12, fontweight='bold')

    plt.plot(range(num_steps), e_vals, color='red', linewidth=3, label='Filter Estimate (Mean)')
    
    if true_pos is not None:
        plt.axhline(y=true_pos, color='red', linestyle='--', linewidth=2, label=f'True value ({true_pos:.3f})')

    plt.title(f'Particle Filter: {title_name} Evolution', fontsize=14, fontweight='bold')
    plt.xlabel('Simulation Step', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    
    if min_val is not None and max_val is not None:
        plt.ylim(min_val, max_val) 
        
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Plotting] Saved {title_name} plot to: {save_path}")
        
    plt.show()
        
        