import os
import sys
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

from cv_bridge import CvBridge
import cv2
import numpy as np


class DataCollector(Node):
    def __init__(self):
        super().__init__('data_collector')
        pkg_dir = os.path.dirname(os.path.realpath(__file__))

        self.declare_parameter('image_topic', '/image_raw')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('image_width', 224)
        self.declare_parameter('image_height', 224)
        self.declare_parameter('log_name', 'dataset')
        self.declare_parameter('max_data_count', 50000)

        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').get_parameter_value().string_value
        self.img_width = self.get_parameter('image_width').get_parameter_value().integer_value
        self.img_height = self.get_parameter('image_height').get_parameter_value().integer_value
        self.log_name = self.get_parameter('log_name').get_parameter_value().string_value
        self.max_data_count = self.get_parameter('max_data_count').get_parameter_value().integer_value

        self.save_log_path = os.path.join(pkg_dir, '..', 'logs')
        self.dataset_dir = os.path.join(self.save_log_path, self.log_name)
        self.images_dir = os.path.join(self.dataset_dir, 'images')
        self.angle_dir = os.path.join(self.dataset_dir, 'angle')
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.angle_dir, exist_ok=True)

        self.bridge = CvBridge()
        self.last_ang_vel = 0.0
        self.save_flag = False
        self.data_count = 0

        self.image_sub = self.create_subscription(Image, self.image_topic, self.image_callback, 10)
        self.cmd_sub = self.create_subscription(Twist, self.cmd_vel_topic, self.cmd_callback, 10)
        self.save_flag_sub = self.create_subscription(Bool, '/save', self.save_callback, 10)

        self.get_logger().info(f"Saving dataset to: {self.dataset_dir}")


    def cmd_callback(self, msg):
        self.last_ang_vel = msg.angular.z

    def save_callback(self, msg):
        if msg.data:
            self.save_flag = True
            self.get_logger().info("Save flag received, shutting down data collection.")
        else:
            self.save_flag = False

    def image_callback(self, msg):
        if not self.save_flag or self.data_count >= self.max_data_count:
            return

        cv_image = self.bridge.imgmsg_to_cv2(msg)
        resized = cv2.resize(cv_image, (self.img_width, self.img_height))

        idx_str = f"{self.data_count + 1:05d}"

        # 保存パス
        img_path = os.path.join(self.images_dir, f"{idx_str}.png")
        angle_path = os.path.join(self.angle_dir, f"{idx_str}.csv")

        # 保存
        cv2.imwrite(img_path, resized)
        np.savetxt(angle_path, np.array([self.last_ang_vel]), delimiter=",")

        self.data_count += 1
        self.get_logger().info(f"📸 Sample saved: {idx_str}, angle: {self.last_ang_vel:.3f}")

    def destroy_node(self):
        self.get_logger().info(f"Total samples collected: {self.data_count}")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DataCollector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n[INFO] Graceful shutdown by Ctrl+C.")
        sys.exit(0)
    finally:
        node.destroy_node()
        rclpy.shutdown()
