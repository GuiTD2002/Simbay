#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import WrenchStamped
from std_srvs.srv import Trigger

from VIS_FT import visFTDriver
import numpy as np
import time

import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class VisFTNode(Node):
    def __init__(self):
        super().__init__('vis_ft_node')

        # Parameters
        self.declare_parameter('frame_id', 'vis_ft_sensor')
        self.declare_parameter('publish_rate', 120.0)  # Hz

        self.frame_id = self.get_parameter('frame_id').value
        rate = self.get_parameter('publish_rate').value
        self.bias = np.zeros(6)

        # Publisher
        self.publisher_ = self.create_publisher(
            WrenchStamped,
            'vis_ft/wrench',
            10
        )
        # Service
        self.bias_service_ = self.create_service(
            Trigger,
            'vis_ft/bias',
            self.bias_callback
        )
        # Image publisher
        self.image_pub_ = self.create_publisher(
            Image,
            'vis_ft/image',
            10
        )

        self.bridge = CvBridge()
        # Sensor setup
        self.get_logger().info('Initiating VIS_FT sensor connection...')
        self.visft_driver = visFTDriver()

        try:
            self.visft_driver.start()
            self.get_logger().info('Zeroing VIS_FT sensor...')
            self.visft_driver.zero()
            self.get_logger().info('Connected to VIS_FT sensor!')
        except Exception as e:
            self.get_logger().error(f'Failed to connect to VIS_FT sensor: {e}')
            raise RuntimeError('VIS_FT initialization failed')

        # Timer
        self.timer = self.create_timer(1.0 / rate, self.timer_callback)

    def timer_callback(self):
        try:
            vis_ft_val, frame = self.visft_driver.read(drift_correct=False)
            vis_ft_val = np.array(vis_ft_val, dtype=float) - self.bias

            msg = WrenchStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.frame_id

            # Force (N)
            msg.wrench.force.x = float(vis_ft_val[0])
            msg.wrench.force.y = float(vis_ft_val[1])
            msg.wrench.force.z = float(vis_ft_val[2])

            # Torque (Nm)
            msg.wrench.torque.x = float(vis_ft_val[3])
            msg.wrench.torque.y = float(vis_ft_val[4])
            msg.wrench.torque.z = float(vis_ft_val[5])

            self.publisher_.publish(msg)


            image_msg = self.bridge.cv2_to_imgmsg(
                frame,
                encoding='bgr8'   # most OpenCV images
            )
            image_msg.header.stamp = msg.header.stamp
            image_msg.header.frame_id = self.frame_id

            self.image_pub_.publish(image_msg)

        except Exception as e:
            self.get_logger().warn(f'Read failed: {e}')


        except Exception as e:
            self.get_logger().warn(f'VIS_FT read failed: {e}')

    def bias_callback(self, request, response):
        self.get_logger().info('Biasing (zeroing) VIS_FT sensor...')
        try:
            #self.visft_driver.zero()
            self.bias,_=self.visft_driver.read(drift_correct=False)
            
            response.success = True
            response.message = 'VIS_FT sensor biased successfully'
            self.get_logger().info(response.message)
        except Exception as e:
            response.success = False
            response.message = f'Biasing failed: {e}'
            self.get_logger().error(response.message)

        return response



    def destroy_node(self):
        self.get_logger().info('Shutting down VIS_FT sensor...')
        try:
            self.visft_driver.shutdown()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VisFTNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
