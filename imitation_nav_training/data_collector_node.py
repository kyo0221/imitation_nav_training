import os
import sys
import yaml
import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty, Bool, String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

from cv_bridge import CvBridge
import cv2
import torch
import numpy as np

from .topomap_creator_node import TopologicalMapCreator
from .placenet import PlaceNet

class DataCollector(Node):
    def __init__(self):
        super().__init__('data_collector')
        pkg_dir = os.path.dirname(os.path.realpath(__file__))

        self.declare_parameter('image_topic', '/image_raw')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('image_width', 200)
        self.declare_parameter('image_height', 88)
        self.declare_parameter('log_name', 'dataset.pt')
        self.declare_parameter('max_data_count', 50000)
        self.declare_parameter('velocity_mps', 0.5)  # assumed average velocity
        self.declare_parameter('distance_threshold', 5.0)  # meters to travel before saving

        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').get_parameter_value().string_value
        self.img_width = self.get_parameter('image_width').get_parameter_value().integer_value
        self.img_height = self.get_parameter('image_height').get_parameter_value().integer_value
        self.log_name = self.get_parameter('log_name').get_parameter_value().string_value
        self.max_data_count = self.get_parameter('max_data_count').get_parameter_value().integer_value
        self.velocity_mps = self.get_parameter('velocity_mps').get_parameter_value().double_value
        self.distance_threshold = self.get_parameter('distance_threshold').get_parameter_value().double_value

        self.save_log_path = os.path.join(pkg_dir, '..', 'logs')
        self.save_path = os.path.abspath(self.save_log_path) + '/' + self.log_name

        self.topo_map_dir = os.path.join(self.save_log_path, 'topo_map')
        os.makedirs(self.topo_map_dir, exist_ok=True)
        self.topo_map_yaml = os.path.join(self.topo_map_dir, 'topomap.yaml')
        self.image_dir = os.path.join(self.topo_map_dir, 'images')
        os.makedirs(self.image_dir, exist_ok=True)
        self.weight_path = os.path.join(pkg_dir, '..', 'weights', 'efficientnet_85x85.pth')

        self.map_creator = TopologicalMapCreator(self.topo_map_yaml, self.image_dir, self.weight_path)

        self.bridge = CvBridge()
        self.images = []
        self.ang_vels = []
        self.actions = []

        self.action_to_index = {"straight": 0, "left": 1, "right": 2}
        self.command_mode = "straight"
        self.last_ang_vel = 0.0

        self.save_flag = False
        self.data_saved = False
        self.latest_image_msg = None

        self.cv_resized_image = None
        self.image_save_counter = 1
        self.accumulated_distance = 0.0

        self.save_flag_sub = self.create_subscription(Bool, '/save', self.save_callback, 10)
        self.cmd_route_sub = self.create_subscription(String, "/cmd_route", self.command_mode_callback, 10)
        self.cmd_save_image_sub = self.create_subscription(Empty, "/save_image", self.save_image_callback, 10)

        self.image_sub = self.create_subscription(Image, self.image_topic, self.image_callback, 10)
        self.cmd_sub = self.create_subscription(Twist, self.cmd_vel_topic, self.cmd_callback, 10)

        self.get_logger().info(f"Subscribed to {self.image_topic}, {self.cmd_vel_topic}, and /cmd_route")
        self.get_logger().info(f"Saving to {self.save_log_path} with max count {self.max_data_count}")

        self.timer = self.create_timer(0.1, self.periodic_collect)

    def image_callback(self, msg):
        self.latest_image_msg = msg

    def cmd_callback(self, msg):
        self.last_ang_vel = msg.angular.z
        lin_speed = msg.linear.x
        self.accumulated_distance += lin_speed * 0.1

    def save_callback(self, msg):
        if msg.data and not self.data_saved:
            self.get_logger().info("💾 Save flag received, saving data...")

        if len(self.images) and not msg.data:
            self.data_augmentation()
            self.save_data()
            self.data_saved = True

    def command_mode_callback(self, msg):
        if msg.data in self.action_to_index:
            self.command_mode = msg.data
        else:
            self.get_logger().warn(f"Unknown command_mode received: {msg.data}, defaulting to 'straight'")
            self.command_mode = "straight"

    def save_image_callback(self, msg: Empty):
        self.save_image()

    def save_image(self):
        if self.cv_resized_image is None:
            self.get_logger().warn("No resized image available to save.")
            return

        img_name = f"img{self.image_save_counter}.png"
        img_path = os.path.join(self.image_dir, img_name)

        try:
            cv2.imwrite(img_path, self.cv_resized_image)
            self.get_logger().info(f"💾 Saved image to {img_path}")

            self.map_creator.add_node(img_name, self.command_mode)
            self.image_save_counter += 1
            self.accumulated_distance = 0.0  # reset distance after saving

        except Exception as e:
            self.get_logger().error(f"❌ Failed to save image or add node: {e}")

    def periodic_collect(self):
        if self.save_flag or self.latest_image_msg is None:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(self.latest_image_msg, desired_encoding='bgr8')
            resized = cv2.resize(cv_image, (self.img_width, self.img_height))
            self.cv_resized_image = (resized * 1).astype(np.uint8)
            tensor_image = torch.tensor(resized, dtype=torch.float32).permute(2, 0, 1) / 255.0

            self.images.append(tensor_image)
            self.ang_vels.append(torch.tensor([self.last_ang_vel], dtype=torch.float32))
            self.actions.append(torch.tensor([self.action_to_index[self.command_mode]], dtype=torch.long))

            if self.accumulated_distance >= self.distance_threshold:
                self.save_image()

            if len(self.images) >= self.max_data_count:
                self.get_logger().info("Max data count reached.")
                self.save_flag = True

        except Exception as e:
            self.get_logger().error(f"[collect] Image processing failed: {e}")

    def data_augmentation(self):
        num_augmented = 0
        angle_offset_deg = 5
        vel_offset = 0.2

        original_images = self.images.copy()
        original_vels = self.ang_vels.copy()
        original_actions = self.actions.copy()

        for img_tensor, ang_vel, action in zip(original_images, original_vels, original_actions):
            img = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            for sign in [-1, -0.5, 0.5, 1]:
                rot_mat = cv2.getRotationMatrix2D((img.shape[1] // 2, img.shape[0] // 2), sign * angle_offset_deg, 1.0)
                rotated_img = cv2.warpAffine(img, rot_mat, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)

                rotated_tensor = torch.tensor(rotated_img, dtype=torch.float32).permute(2, 0, 1) / 255.0
                corrected_vel = ang_vel + torch.tensor([-sign * vel_offset], dtype=torch.float32)

                self.images.append(rotated_tensor)
                self.ang_vels.append(corrected_vel)
                self.actions.append(action.clone())
                num_augmented += 1

        self.get_logger().info(f"📈 Data augmentation complete. {num_augmented} samples added.")

    def save_data(self):
        if len(self.images) == 0:
            self.get_logger().warn("No data to save.")
            return

        images_tensor = torch.stack(self.images)
        ang_vels_tensor = torch.stack(self.ang_vels)
        actions_tensor = torch.stack(self.actions)

        torch.save({
            'images': images_tensor,
            'angles': ang_vels_tensor,
            'actions': actions_tensor,
            'action_classes': list(self.action_to_index.keys())
        }, self.save_path)

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
        node.map_creator.save_map()
        node.destroy_node()
        rclpy.shutdown()
