# type: ignore
import time

import numpy as np

from .base import BaseRobot

# --- CONDITIONAL IMPORTS ---
try:
    import rclpy
    from franka_msgs.action import Move
    from geometry_msgs.msg import WrenchStamped
    from rclpy.action import ActionClient
    from rclpy.duration import Duration
    from sensor_msgs.msg import JointState
    from trajectory_msgs.msg import JointTrajectory
    from trajectory_msgs.msg import JointTrajectoryPoint
    from tf2_ros import Buffer, TransformListener
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    print("Warning: ROS libraries not found. RealRobot will not work, but SimRobot is fine.")


class RealRobot(BaseRobot):
    def __init__(self, dt=0.001):
        """
        Args:
            dt (float): The control loop timestep. Must match the 'dt' used 
                        in your trajectory generator to ensure smooth motion!
        """
        if not ROS_AVAILABLE:
            raise ImportError("Cannot use RealRobot: ROS libraries are missing!")
        
        self.dt = dt
        
        # 1. Setup Joint Names
        self.joint_names = [
            "fr3_joint1", "fr3_joint2", "fr3_joint3", 
            "fr3_joint4", "fr3_joint5", "fr3_joint6", "fr3_joint7"
        ]
        self.current_joints = None
        self.current_velocities = np.zeros(7)
        
        # 2. Setup Sensor Data Holder
        self.current_wrench = np.zeros(6)
        
        # 3. Connect to ROS 2
        self._setup_ros()

    def _setup_ros(self):
        """Initializes the ROS 2 node, publishers, subscribers, and action clients."""
        if not rclpy.ok():
            rclpy.init()
        self.node1 = rclpy.create_node("simbay_real_robot_node")
        
        # Arm Publisher & Subscriber
        self.pub1 = self.node1.create_publisher(JointTrajectory, "/fr3_arm_controller/joint_trajectory", 10)
        self.sub1 = self.node1.create_subscription(JointState, "/joint_states", self.jointstate_callback, 10)
        
        # Force/Torque Sensor Subscriber
        self.sub_wrench = self.node1.create_subscription(
            WrenchStamped,
            "/franka_robot_state_broadcaster/external_wrench_in_base_frame",
            self.wrench_callback,
            10
        )
        
        # Gripper Action Client
        self.gripper_client = ActionClient(self.node1, Move, '/franka_gripper/move')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node1)
        
        print("Waiting for robot connection and Action Servers...")
        
        # Block until we receive the first valid joint state message
        while self.current_joints is None:
            rclpy.spin_once(self.node1, timeout_sec=0.1)
            
        # Wait for gripper server (5 seconds timeout so it doesn't hang forever)
        if not self.gripper_client.wait_for_server(timeout_sec=5.0):
            print("⚠️ Warning: Gripper Action Server not found. Gripper commands may fail.")
            
        print("✅ Robot Connected!")

    def jointstate_callback(self, msg):
        """Safely extracts joint positions AND velocities by their exact names."""
        try:
            positions = []
            velocities = []
            for name in self.joint_names:
                idx = msg.name.index(name)
                positions.append(msg.position[idx])
                
                # Safely grab velocity if it exists in the message
                if len(msg.velocity) > idx:
                    velocities.append(msg.velocity[idx])
                else:
                    velocities.append(0.0)
                    
            self.current_joints = positions
            self.current_velocities = velocities
        except ValueError:
            pass # Ignore messages that don't contain our 7 arm joints


    def wrench_callback(self, msg):
        """Automatically updates every time the Franka sends new force/torque data."""
        self.current_wrench = np.array([
            msg.wrench.force.x,
            msg.wrench.force.y,
            msg.wrench.force.z,
            msg.wrench.torque.x,
            msg.wrench.torque.y,
            msg.wrench.torque.z
        ])


    def move_joints(self, pos):
        """
        Blindly streams a single waypoint to the controller.
        Expects to be called in a loop, running exactly every self.dt seconds.
        """
        msg = JointTrajectory()
        msg.joint_names = self.joint_names
        msg.header.stamp = self.node1.get_clock().now().to_msg()
        
        point = JointTrajectoryPoint()
        point.positions = list(pos)
        
        point.time_from_start = Duration(seconds=self.dt).to_msg() 
        msg.points = [point]

        # Publish the movement and return instantly
        self.pub1.publish(msg)

    def move_gripper(self, width):
        """
        Sends a blocking command to the physical gripper action server.
        Expects width in meters (e.g., 0.0 for closed, ~0.08 for fully open).
        """
        if not self.gripper_client.server_is_ready():
            print("❌ Gripper server offline. Cannot move gripper.")
            return

        goal_msg = Move.Goal()
        goal_msg.width = float(width)
        goal_msg.speed = 0.1 # Safe default speed

        # Send goal and wait for acceptance
        future = self.gripper_client.send_goal_async(goal_msg)
        # Safely wait for the gripper action to finish without crashing the background thread!
        while not future.done():
            try:
                rclpy.spin_once(self.node1, timeout_sec=0.01)
            except RuntimeError as e:
                # If the background thread is currently grabbing sensor data, just wait a few milliseconds and try again.
                if 'already spinning' in str(e):
                    time.sleep(0.005)
                else:
                    raise e
        goal_handle = future.result()

        if goal_handle and goal_handle.accepted:
            # Wait for the physical fingers to finish moving
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self.node1, result_future)
        else:
            print("❌ Gripper goal rejected by hardware!")


    def get_joints_pos(self):
        """Returns the current 7-joint positions as a numpy array."""
        try:
            rclpy.spin_once(self.node1, timeout_sec=0.01)
        except RuntimeError as e:
            # If the main thread is moving the arm, ignore the spin collision!
            if 'already spinning' in str(e):
                pass
            else:
                raise e
        return np.array(self.current_joints)
    
    def get_joints_vel(self):
        """Returns the current 7-joint velocities as a numpy array."""
        try:
            rclpy.spin_once(self.node1, timeout_sec=0.01)
        except RuntimeError as e:
            if 'already spinning' in str(e):
                pass
            else:
                raise e
        return np.array(self.current_velocities)


    def get_ee_pos(self):
        """
        Returns the current end-effector position [x, y, z] as a numpy array.
        Uses TF2 to look up the transform from the robot's base to its hand.
        """
        try:
            # We spin once to ensure the TF buffer is fed with the latest data
            try:
                rclpy.spin_once(self.node1, timeout_sec=0.001)
            except RuntimeError as e:
                if 'already spinning' not in str(e): raise e

            # Lookup transform from base ('fr3_link0') to end-effector ('fr3_link8' or 'fr3_hand')
            # Note: If your lab uses different TF names, adjust these strings!
            trans = self.tf_buffer.lookup_transform(
                "fr3_link0", 
                "fr3_link8", # Alternatively "fr3_hand" or "fr3_EE" depending on your launch file
                rclpy.time.Time()
            )
            
            return np.array([
                trans.transform.translation.x,
                trans.transform.translation.y,
                trans.transform.translation.z - 0.1
            ])
            
        except Exception as e:
            # If TF fails (usually just on the very first loop before data arrives), return zeros safely
            return np.zeros(3)


    def get_torque_reads(self):
        """Returns the 6-element array: [Fx, Fy, Fz, Tx, Ty, Tz] in the GLOBAL frame."""
        try:
            # Try to process ROS messages (works fine during time.sleep)
            rclpy.spin_once(self.node1, timeout_sec=0.001)
        except RuntimeError as e:
            # If we get "Executor is already spinning", it means the main thread 
            # is currently moving the arm or gripper. 
            # We can safely ignore this because the main thread's spin is 
            # already updating our sensor callbacks anyway!
            if 'already spinning' in str(e):
                pass
            else:
                raise e
        return self.current_wrench.copy()[3:]
    

    def get_force_reads(self):
        """Returns the 6-element array: [Fx, Fy, Fz, Tx, Ty, Tz] in the GLOBAL frame."""
        try:
            # Try to process ROS messages (works fine during time.sleep)
            rclpy.spin_once(self.node1, timeout_sec=0.001)
        except RuntimeError as e:
            # If we get "Executor is already spinning", it means the main thread 
            # is currently moving the arm or gripper. 
            # We can safely ignore this because the main thread's spin is 
            # already updating our sensor callbacks anyway!
            if 'already spinning' in str(e):
                pass
            else:
                raise e
        return self.current_wrench.copy()[0:3]
    
    def sync(self):
        """
        Calculates exact math/execution time and sleeps the remainder 
        to lock the control loop strictly to the self.dt frequency.
        """
        # Initialize the timer on the very first call
        if not hasattr(self, '_last_sync_time'):
            self._last_sync_time = time.perf_counter()
            return

        # 1. Calculate how long the math and ROS operations took
        current_time = time.perf_counter()
        elapsed = current_time - self._last_sync_time
        
        # 2. Calculate remaining time to hit our target timestep (dt)
        remainder = self.dt - elapsed
        
        # 3. Sleep the exact remainder
        if remainder > 0:
            time.sleep(remainder)
        # else:
            # Optional: Uncomment to debug if your math is taking longer than 1ms!
            # print(f"⚠️ Loop overrun by {-remainder:.5f}s")
            
        # 4. Reset the clock for the start of the next loop
        self._last_sync_time = time.perf_counter()
    
    
    def move_trajectory(self, trajectory, dt2):
        msg = JointTrajectory()
        msg.joint_names = self.joint_names
        msg.header.stamp = self.node1.get_clock().now().to_msg()
        t = 0.01
        for pos in trajectory:
            point = JointTrajectoryPoint()
            point.positions = list(pos)
            point.time_from_start = Duration(nanoseconds=int(t * 1e9)).to_msg()
            t += dt2
            msg.points.append(point)

        # Publish the movement and return instantly
        self.pub1.publish(msg)
        self.wait_seconds(t - dt2 + 0.1) # Wait for the whole trajectory to finish plus a small buffer. Don't delete!!!!


    def wait_seconds(self, duration):
        start_time = time.perf_counter()
        
        while time.perf_counter() - start_time < duration:
            try:
                # Constantly drain the ROS queue while waiting
                rclpy.spin_once(self.node1, timeout_sec=0.005)
            except RuntimeError as e:
                # If the main thread is busy, just sleep for a tiny fraction
                if 'already spinning' in str(e):
                    time.sleep(0.005)
                else:
                    raise e

    def shutdown(self):
        """Clean up the ROS node before exiting to prevent zombie processes."""
        self.node1.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    def move_trajectory_async(self, trajectory, dt2):
        """
        Asynchronous trajectory execution. 
        Fires the command to the hardware and returns to Python instantly.
        """
        msg = JointTrajectory()
        msg.joint_names = self.joint_names
        msg.header.stamp = self.node1.get_clock().now().to_msg()
        
        t = 0.01 # Small initial buffer
        for pos in trajectory:
            point = JointTrajectoryPoint()
            point.positions = list(pos)
            
            # Safely convert float time to ROS 2 nanoseconds
            point.time_from_start = Duration(nanoseconds=int(t * 1e9)).to_msg() 
            
            t += dt2
            msg.points.append(point)

        # Publish the full path to the Trajectory Controller
        self.pub1.publish(msg)
        
        # CRITICAL DIFFERENCE: 
        # We do NOT call self.wait_seconds() here. 
        # We just let the function end so Python goes straight back to your loop!

    def stop_arm(self):
        """
        Instantly halts the arm by preempting the current trajectory 
        with a command to hold its current physical position.
        """
        print("🛑 [HARDWARE] Emergency software brakes triggered!")
        
        # 1. Grab the absolute latest physical position
        current_joints = self.get_joints_pos()
        
        # 2. Create the preemptive trajectory message
        msg = JointTrajectory()
        msg.joint_names = self.joint_names
        msg.header.stamp = self.node1.get_clock().now().to_msg()
        
        # 3. Create a single point telling the robot to freeze
        point = JointTrajectoryPoint()
        point.positions = list(current_joints)
        
        # Explicitly telling the controller we want 0 velocity forces it 
        # to calculate a safe deceleration curve instantly!
        point.velocities = [0.0] * 7 
        point.accelerations = [0.0] * 7
        
        # Command it to achieve this "stopped" state almost immediately (50ms).
        # We use 50ms instead of 0.0s to give the internal math time to smoothly brake
        # without triggering a violent jerk or Reflex Error.
        point.time_from_start = Duration(nanoseconds=int(0.05 * 1e9)).to_msg()
        
        msg.points = [point]
        
        # 4. Fire the stop command!
        # (Note: Double check if your publisher is named self.pub1 or self.pub_traj 
        # based on your __init__ setup, and adjust if necessary!)
        self.pub1.publish(msg)

    
