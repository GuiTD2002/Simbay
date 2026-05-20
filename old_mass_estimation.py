
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np

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
from src.utils import FRANKA_HOME_QPOS
from src.utils import execute_trajectory
from src.utils import initialize_mujoco_env
from src.utils import plot_particle_evolution

# ==========================================
# 1. SETUP
# ==========================================
# Setup "real" robot
real_robot = initialize_mujoco_env(DEFAULT_OBJECT_PROPS2)
viewer = mujoco.viewer.launch_passive(real_robot.model, real_robot.data)
real_robot.viewer = viewer
dt = real_robot.dt

obj_pos = DEFAULT_OBJECT_PROPS2['pos']
true_mass = DEFAULT_OBJECT_PROPS2['mass']

# Initiate Particle Filter
NUM_PARTICLES = 100
ESS_THRESHOLD = 0.5
limits = (0.0, 0.3)



container = RobotContainer(num_particles=NUM_PARTICLES, props=DEFAULT_OBJECT_PROPS2, dt=real_robot.dt)

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

# ==========================================
# 2. TRAJECTORY PLANNING
# ==========================================
target_quat = np.array([0.0, 1.0, 0.0, 0.0])

pre_grasp_pos = obj_pos + np.array([0.0, 0.0, 0.15])
pre_grasp_q7 = FrankaKinematics.inverse(FRANKA_HOME_QPOS, pre_grasp_pos, target_quat)

grasp_q7 = FrankaKinematics.inverse(pre_grasp_q7, obj_pos, target_quat)

lift_pos = obj_pos + np.array([0.0, 0.0, 0.2])
lift_q7 = FrankaKinematics.inverse(grasp_q7, lift_pos, target_quat)

OPEN = 255
CLOSED = 0
q_home = np.append(FRANKA_HOME_QPOS, OPEN)
q_pre_grasp = np.append(pre_grasp_q7, OPEN)
q_grasp_open = np.append(grasp_q7, OPEN)
q_grasp_closed = np.append(grasp_q7, CLOSED)
q_lift_closed = np.append(lift_q7, CLOSED)

# ==========================================
# 3. EXECUTION
# ==========================================

# Phase 1: Move ABOVE the object (No PF updates, just predict to stay synced)
print("Moving to Approach position...")
traj1 = plan_joints_trajectory(q_home, q_pre_grasp, max_velocity=1.0, dt=dt)
for qpos in traj1:
    real_robot.move_gripper(0.08)
    real_robot.move_joints(qpos)
    real_robot.sync()

    control_input = {
            'joints': qpos,
            'gripper': 0.08
        }

    particle_filter.predict(control_input) 


# Phase 2: Descend vertically to the object (No PF updates)
print("Descending to grasp...")
traj2 = plan_joints_trajectory(q_pre_grasp, q_grasp_open, max_velocity=0.5, dt=dt)
for qpos in traj2:
    real_robot.move_gripper(0.08)
    real_robot.move_joints(qpos)
    real_robot.sync()

    control_input = {
            'joints': qpos,
            'gripper': 0.08
        }

    particle_filter.predict(control_input) 


# Phase 3: Close the Gripper (No PF updates)
print("Closing Gripper...")
traj3 = plan_joints_trajectory(q_grasp_closed, q_grasp_closed, max_velocity=500, dt=dt) # we close directly and only use the settle_time
for qpos in traj3:
    real_robot.move_gripper(0.00)
    real_robot.move_joints(qpos)
    real_robot.sync()

    control_input = {
            'joints': qpos,
            'gripper': 0.00
        }

    particle_filter.predict(control_input) 


# Phase 4: Lift straight up (OBJECT IS GRASPED - START TRACKING MASS)
print("Lifting object and running Particle Filter...")
traj4 = plan_joints_trajectory(q_grasp_closed, q_lift_closed, max_velocity=0.5, dt=dt) 

# <--- Initialize lists to hold the historical data for the graph --->
history_particles = []
history_estimates = []

for step, qpos in enumerate(traj4):
    real_robot.move_joints(qpos)
    real_robot.sync()

    control_input = {
            'joints': qpos,
            'gripper': 0.00
        }

    
    # 1. Step the particles forward
    particle_filter.predict(control_input) 
    
    # 2. Get the real measurement
    measurements = real_robot.get_torque_reads()
    real_ft_reading = measurements
    
    # 3. INJECT NOISE to simulate physical hardware (e.g., 0.5N of sensor noise)
    noisy_ft_reading = real_ft_reading + np.random.normal(0, 0.5, size=3)

    observation = {'torques': noisy_ft_reading}
    
    # 4. Update beliefs based on the noisy real reading and resample
    particle_filter.update(observation)
    particle_filter.resample(None)

    # <--- Save the state of the particles at this exact timestep --->
    history_particles.append(particle_filter.particles.copy())
    history_estimates.append(particle_filter.estimate())


print("\nSequence complete. Press Enter to close.")
input()
print(f"Final Mass Prediction: {particle_filter.estimate():.4f} kg")
print(f"Final Error: {abs (true_mass - particle_filter.estimate()) * 100:.2f} %")
# You should also print the real mass here to see if the filter got it right!


# ==========================================
# 4. GRAPH GENERATION
# ==========================================
print("Generating Particle Filter Evolution Plot...")


plt.figure(figsize=(10, 6))

# Plot the 100 particles over time as highly transparent blue dots
num_steps = len(history_particles)
for t in range(num_steps):
    plt.scatter([t] * NUM_PARTICLES, history_particles[t], color='blue', alpha=0.05, s=15)

# Plot the Filter's official guess (Mean)
plt.plot(range(num_steps), history_estimates, color='red', linewidth=3, label='Filter Estimate (Mean)')

# Plot the True Mass
plt.axhline(y=true_mass, color='green', linestyle='--', linewidth=2, label=f'True Mass ({true_mass} kg)')

# Formatting
plt.title('Particle Filter: Mass Estimation Evolution', fontsize=14, fontweight='bold')
plt.xlabel('Simulation Step (Lifting Phase)', fontsize=12)
plt.ylabel('Estimated Mass (kg)', fontsize=12)
plt.ylim(limits[0], limits[1]) # Lock the Y-axis to physical limits
plt.legend(loc='upper right')
plt.grid(True, linestyle=':', alpha=0.7)

plt.show()