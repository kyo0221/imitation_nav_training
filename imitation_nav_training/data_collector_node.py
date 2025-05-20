import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

from cv_bridge import CvBridge
import cv2
import torch
import numpy as np
from collections import deque

class DataCollector(Node):
    def __init__(self):
        super().__init__('data_collector')

        self.declare_parameter('image_topic', '/image_raw')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('image_width', 64)
        self.declare_parameter('image_height', 48)
        self.declare_parameter('output_path', 'dataset.pt')
        self.declare_parameter('max_data_count', 10000)

        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').get_parameter_value().string_value
        self.img_width = self.get_parameter('image_width').get_parameter_value().integer_value
        self.img_height = self.get_parameter('image_height').get_parameter_value().integer_value
        self.output_path = self.get_parameter('output_path').get_parameter_value().string_value
        self.max_data_count = self.get_parameter('max_data_count').get_parameter_value().integer_value

        self.bridge = CvBridge()
        self.images = []
        self.ang_vels = []
        self.last_ang_vel = 0.0

        self.image_sub = self.create_subscription(Image, self.image_topic, self.image_callback, 10)
        self.cmd_sub = self.create_subscription(Twist, self.cmd_vel_topic, self.cmd_callback, 10)

        self.get_logger().info(f"Subscribed to {self.image_topic} and {self.cmd_vel_topic}")
        self.get_logger().info(f"Saving to {self.output_path} with max count {self.max_data_count}")

    def cmd_callback(self, msg):
        self.last_ang_vel = msg.angular.z

    def image_callback(self, msg):
        try:
            if len(self.images) >= self.max_data_count:
                self.get_logger().info("Max data count reached. Ignoring further data.")
                return

            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            resized = cv2.resize(cv_image, (self.img_width, self.img_height))
            tensor_image = torch.tensor(resized, dtype=torch.float32).permute(2, 0, 1) / 255.0

            self.images.append(tensor_image)
            self.ang_vels.append(torch.tensor([self.last_ang_vel], dtype=torch.float32))
        except Exception as e:
            self.get_logger().error(f"Image processing failed: {e}")

    def save_data(self):
        if len(self.images) == 0:
            self.get_logger().warn("No data to save.")
            return
        images_tensor = torch.stack(self.images)
        ang_vels_tensor = torch.stack(self.ang_vels)
        torch.save({'images': images_tensor, 'angles': ang_vels_tensor}, self.output_path)
        self.get_logger().info(f"✨ Saved {len(self.images)} samples to {self.output_path}")

def main(args=None):
    rclpy.init(args=args)
    node = DataCollector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted. Saving collected data...")
        node.save_data()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
