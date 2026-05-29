import rclpy
from rclpy.node import Node
from franka_msgs.msg import FrankaState
from geometry_msgs.msg import WrenchStamped

class ExternalWrenchPublisher(Node):
    def __init__(self):
        super().__init__('external_wrench_publisher')

        # Declare parameter for publishing frequency (default: 100 Hz)
        self.declare_parameter('publish_frequency', 100.0)
        frequency = self.get_parameter('publish_frequency').value
        timer_period = 1.0 / frequency

        # Variable to store the latest incoming state message
        self.latest_state_msg = None

        # Subscriber for the Franka robot state
        self.subscription = self.create_subscription(
            FrankaState,
            '/franka_robot_state_broadcaster/robot_state',
            self.state_callback,
            10
        )

        # Publisher for the extracted wrench
        self.publisher = self.create_publisher(
            WrenchStamped,
            'franka_robot_state_broadcaster/external_wrench_in_base_frame',
            10
        )

        # Timer to publish at the chosen frequency
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.get_logger().info(f"Wrench publisher started at {frequency} Hz.")

    def state_callback(self, msg: FrankaState):
        """Simply store the latest message when it arrives."""
        self.latest_state_msg = msg

    def timer_callback(self):
        """Extract the wrench and publish at the timer's frequency."""
        if self.latest_state_msg is None:
            # Do nothing if we haven't received a state message yet
            return

        wrench_msg = WrenchStamped()
        
        # Populate Header
        wrench_msg.header.stamp = self.get_clock().now().to_msg()
        # Defaulting to standard franka base frame. Adjust to 'panda_link0' if necessary.
        wrench_msg.header.frame_id = 'panda_link0'

        # Extract o_f_ext_hat_k (float64[6] array representing [Fx, Fy, Fz, Tx, Ty, Tz])
        ext_wrench = self.latest_state_msg.o_f_ext_hat_k

        # Populate Wrench
        wrench_msg.wrench.force.x = float(ext_wrench[0])
        wrench_msg.wrench.force.y = float(ext_wrench[1])
        wrench_msg.wrench.force.z = float(ext_wrench[2])
        
        wrench_msg.wrench.torque.x = float(ext_wrench[3])
        wrench_msg.wrench.torque.y = float(ext_wrench[4])
        wrench_msg.wrench.torque.z = float(ext_wrench[5])

        self.publisher.publish(wrench_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ExternalWrenchPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt, shutting down.")
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()
