#!/usr/bin/env python3

import lxros
import rclpy
from std_msgs.msg import Header
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState

last_joint_states = JointState()


def create_trajectory(array,secs=2.0):
    traj = JointTrajectory()
    #traj.joint_names = ['fr3_joint1','fr3_joint2','fr3_joint3','fr3_joint4','fr3_joint5','fr3_joint6','fr3_joint7']
    traj.joint_names = ['fr3_joint'+str(i) for i in range(1,8)]

    traj.header.stamp = rclpy.time.Time().to_msg()

    point = JointTrajectoryPoint()
    point.positions = array
    point.time_from_start = rclpy.time.Duration(seconds=secs).to_msg()

    traj.points.append(point)
    return traj
   



class SimbayFRPick():
    def __init__(self):
        self.node1 = lxros.init_node("simbay_pick_test_node")
        self.pub1 = self.node1.pub("/fr3_arm_controller/joint_trajectory", JointTrajectory)
        self.sub1 = self.node1.sub("/franka/joint_states", JointState, self.jointstate_callback)
        

    def jointstate_callback(self,msg):
        global last_joint_states
        last_joint_states = msg
        print(f'Recebido Joint States: {msg.position}')


    def update_joint_states(self):
        lxros.spin_once(timeout=0.1)

    def send_trajectory(self, traj):
        self.update_joint_states()
        self.pub1.publish(traj)
        print(f'Published Trajectory: {[traj.points[i].positions for i in range(len(traj.points))]}')
        self.update_joint_states()
        

import random

def main():

    picker=SimbayFRPick()
    
    traj = create_trajectory([1.0, -0.8, 0.0,  -2.2, 0.0, 1.6, 0.8], secs=2)
    picker.send_trajectory(traj)



if __name__ == "__main__":
    main()
