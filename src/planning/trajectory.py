import numpy as np

from src.planning import FrankaKinematics


def plan_joints_trajectory(start_joints: np.ndarray, target_joints: np.ndarray, max_velocity: float, dt: float) -> np.ndarray: 
    """
    Calculates a straight-line trajectory in joint space.
    Returns a NumPy array of intermediate joint states.
    
    Args:
        start_q (array-like): Starting joint positions.
        target_q (array-like): Target joint positions.
        max_velocity (float): Maximum velocity for any joint (radians/sec).
        dt (float): Time step between trajectory points (seconds).
    """
    # Calculate distance and bottleneck time
    distance = np.abs(target_joints - start_joints)
    duration = np.max(distance / max_velocity)
    
    # Use CEIL to ensure any fractional step rounds UP to a full step
    total_steps = int(np.ceil(duration / dt))

    # Generate total_steps + 1 points, and slice off the start_pos
    return np.linspace(start_joints, target_joints, total_steps + 1)[1:]


import numpy as np

def quaternion_slerp(q0: np.ndarray, q1: np.ndarray, fraction: float) -> np.ndarray:
    """
    Spherical linear interpolation between two quaternions.
    Smoothly blends the rotation from q0 to q1 based on the fraction (0.0 to 1.0).
    """
    # Ensure unit quaternions
    q0 = np.array(q0) / np.linalg.norm(q0)
    q1 = np.array(q1) / np.linalg.norm(q1)
    
    dot = np.dot(q0, q1)
    
    # If the dot product is negative, the quaternions have opposite handedness.
    # We reverse one so the robot takes the shortest rotational path.
    if dot < 0.0:
        q1 = -q1
        dot = -dot
        
    # If the rotations are virtually identical, linearly interpolate to avoid division by zero
    if dot > 0.9995:
        result = q0 + fraction * (q1 - q0)
        return result / np.linalg.norm(result)
        
    # Standard SLERP math
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta_fraction = theta_0 * fraction
    sin_theta_fraction = np.sin(theta_fraction)
    
    s0 = np.cos(theta_fraction) - dot * sin_theta_fraction / sin_theta_0
    s1 = sin_theta_fraction / sin_theta_0
    
    return (s0 * q0) + (s1 * q1)


def plan_cartesian_trajectory(start_joints: np.ndarray, target_pos: np.ndarray, target_quat: np.ndarray, max_velocity: float, dt: float) -> np.ndarray:
    """
    Calculates a straight-line trajectory in Cartesian space, 
    smoothly interpolating BOTH position and orientation.
    """
    # 1. Derive the starting Cartesian state
    start_pos, start_quat = FrankaKinematics.forward(start_joints)

    # 2. Calculate timing based purely on Cartesian distance
    distance = np.linalg.norm(target_pos - start_pos)
    
    # Safety Catch: If distance is ~0 but we still need to rotate, guarantee at least 1 second of movement.
    if distance < 1e-4:
        duration = 1.0
    else:
        duration = distance / max_velocity
        
    total_steps = int(np.ceil(duration / dt))

    # 3. Generate intermediate fractions from 0.0 to 1.0
    # (We use fractions so we can easily interpolate both position and quaternion)
    fractions = np.linspace(0.0, 1.0, total_steps + 1)[1:]

    joints_traj = []
    current_joints = start_joints 
    
    for frac in fractions:
        # Interpolate Position
        waypoint_pos = start_pos + frac * (target_pos - start_pos)
        
        # Interpolate Orientation (THE FIX)
        waypoint_quat = quaternion_slerp(start_quat, target_quat, frac)

        # 4. Solve IK for the synchronized waypoint
        next_joints = FrankaKinematics.inverse(current_joints, waypoint_pos, waypoint_quat)
        joints_traj.append(next_joints)
        
        # Advance the loop
        current_joints = next_joints

    return np.array(joints_traj)


def plan_settle_trajectory(joints: np.ndarray, time: float, dt: float) -> np.ndarray:
    steps = int(time / dt)
    return np.tile(joints, (steps, 1))


def stitch_trajectories(*trajectories: np.ndarray) -> np.ndarray:
    """
    Combines multiple trajectory arrays into a single continuous sequence.
    Using *args allows you to pass 2, 3, or 10 trajectories at once!
    """
    return np.concatenate(trajectories, axis=0)
