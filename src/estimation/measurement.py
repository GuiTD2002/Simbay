from abc import ABC
from abc import abstractmethod

import mujoco
import numpy as np

from .container import RobotContainer


class BaseMeasurementModel(ABC):
    @abstractmethod
    def compute_likelihoods(self, particles: np.ndarray, observation: dict) -> np.ndarray:
        pass


class MassMeasurementModel(BaseMeasurementModel):
    def __init__(self, container: RobotContainer):
        self.container = container

    def compute_likelihoods(self, particles, observation):   
        sim_measurements = np.array([robot.get_torque_reads()[1] for robot in self.container.robots])
        #print(sim_measurements)
        for robot in self.container.robots:
            continue
            print(robot.model.body_mass[self.container.obj_id])
        
        # Calculate the difference against the real torque observation
        diff = observation['torques'] - sim_measurements
        
        # Measurement noise covariance (R)
        R = 0.001
        dist_sq = diff ** 2
        
        # Calculate likelihood using Gaussian kernel
        likelihoods = np.exp(-0.5 * dist_sq / R)
        
        return likelihoods
    

class BinaryContactMeasurementModel(BaseMeasurementModel):
    def __init__(self, container: RobotContainer, contact_threshold=0.3):
        self.container = container
        self.threshold = contact_threshold
        

    def compute_likelihoods(self, particles, observation):   
        torque_vector = observation['torques']
        real_norm = np.linalg.norm(torque_vector)
        real_contact = observation['contact']
        
        likelihoods = np.ones(self.container.num_particles)

        for i, robot in enumerate(self.container.robots):
            norm = np.linalg.norm(robot.get_torque_reads())
            contact = 1 if norm > self.threshold else 0 
            #print(robot.print_object_pos())
            
            # ==========================================
            # Negative Info (Kill Partiles hit before real_robot)
            # ==========================================
            if real_contact == 0 and contact == 1:  # Negative Info (Kill Partiles hit before real_robot)
                likelihoods[i] = 0.01
                #print("Killed because contact")

            elif real_contact == 1 and contact == 0: # Positive Info (Kill Particles that haven't hit when real_robot has)
                likelihoods[i] = 0.01
                #print("killed because no contact")


            # ==========================================
            # DEBUGGING
            # ==========================================
            #if real_contact == 1 and contact == 0:
                
                # THE PREMATURE EXECUTION TRIPWIRE
                # We only care if it's killing a "perfect" particle.
                # (You can extract Y_pos the same way you did for the green circles)
                #block_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, 'object') # type: ignore
                #y_pos = robot.data.xpos[block_id][1]
                
                #if 0.150 < y_pos < 0.152:
                    #print(f"⚠️ TRAP TRIGGERED: Killing perfect particle at Y={y_pos:.3f}!")
                    #print(f"   -> Real Torque: {real_norm:.3f} | Virt Torque: {norm:.3f} | Threshold: {self.threshold}")

                    #arm_pos = observation['arm_pos']
                    #print(f" real arm position {arm_pos[1]:.3f}, sim position {robot.get_ee_pos()[1]:.3f}")


            
            #if real_contact == 1 and contact ==1:
                #block_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, 'object') # type: ignore
  
                #print("🟢 Both Real and Sim Contact: Particle with Y_pos {} is consistent.".format(round(robot.data.xpos[block_id][1], 3)))
                #arm_pos = observation['arm_pos']
                #print(f" real arm position {arm_pos[1]:.3f}, sim position {robot.get_ee_pos()[1]:.3f}")
        
            #arm_pos = observation['arm_pos']
            #print(f" real arm position {arm_pos[1]:.3f}, sim position {robot.get_ee_pos()[1]:.3f}")
            
        return likelihoods
