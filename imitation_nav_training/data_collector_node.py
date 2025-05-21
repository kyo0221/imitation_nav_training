import os
import sys
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

from cv_bridge import CvBridge
import cv2
import torch
import numpy as np
from collections import deque

from ament_index_python.packages import get_package_share_directory

class DataCollector(Node):
    def __init__(self):
        super().__init__('data_collector')
        pkg_dir = os.path.dirname(os.path.realpath(__file__))

        self.declare_parameter('image_topic', '/image_raw')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('image_width', 640)
        self.declare_parameter('image_height', 480)
        self.declare_parameter('log_name', 'dataset.pt')
        self.declare_parameter('max_data_count', 10000)

        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').get_parameter_value().string_value
        self.img_width = self.get_parameter('image_width').get_parameter_value().integer_value
        self.img_height = self.get_parameter('image_height').get_parameter_value().integer_value
        self.log_name = self.get_parameter('log_name').get_parameter_value().string_value
        self.max_data_count = self.get_parameter('max_data_count').get_parameter_value().integer_value

        self.save_log_path = os.path.join(pkg_dir, '..', 'logs')
        self.save_path = os.path.abspath(self.save_log_path) + '/' + self.log_name

        self.bridge = CvBridge()
        self.images = []
        self.ang_vels = []
        self.last_ang_vel = 0.0

        self.save_flag = False
        self.data_saved = False

        self.save_flag_sub = self.create_subscription(Bool, '/save', self.save_callback, 10)
        self.image_sub = self.create_subscription(Image, self.image_topic, self.image_callback, 10)
        self.cmd_sub = self.create_subscription(Twist, self.cmd_vel_topic, self.cmd_callback, 10)

        self.get_logger().info(f"Subscribed to {self.image_topic} and {self.cmd_vel_topic}")
        self.get_logger().info(f"Saving to {self.save_log_path} with max count {self.max_data_count}")

    def cmd_callback(self, msg):
        self.last_ang_vel = msg.angular.z

    def save_callback(self, msg):
        if msg.data and not self.data_saved:
            self.get_logger().info("💾 Save flag received, saving data...")
        
        if len(self.images) and not msg.data:
            self.save_data()
            self.data_saved = True

    def image_callback(self, msg):
        if self.save_flag:
            return

        try:
            if len(self.images) >= self.max_data_count:
                self.get_logger().info("Max data count reached. Ignoring further data.")
                return

            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            resized = cv2.resize(cv_image, (self.img_width, self.img_height))
            tensor_image = torch.tensor(resized, dtype=torch.float32).permute(2, 0, 1) / 255.0

            self.images.append(tensor_image)
            if not len(self.images):
                print("📝 docs: add dataset")
            self.ang_vels.append(torch.tensor([self.last_ang_vel], dtype=torch.float32))
        except Exception as e:
            self.get_logger().error(f"Image processing failed: {e}")

    def save_data(self):
        if len(self.images) == 0:
            self.get_logger().warn("No data to save.")
            return

        images_tensor = torch.stack(self.images)
        ang_vels_tensor = torch.stack(self.ang_vels)
        torch.save({'images': images_tensor, 'angles': ang_vels_tensor}, self.save_path)

        if rclpy.ok():
            self.get_logger().info(f"✨ Saved {len(self.images)} samples to {self.save_path}")

def main(args=None):
    rclpy.init(args=args)
    node = DataCollector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n[INFO] Graceful shutdown requested by Ctrl+C.")
        sys.exit(0)
    finally:
        node.destroy_node()
        rclpy.shutdown()