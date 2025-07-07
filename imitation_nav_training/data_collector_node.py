import os
import sys
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String, Empty
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

from cv_bridge import CvBridge
import cv2
import numpy as np
import matplotlib.pyplot as plt

from .topomap_creator_node import TopologicalMapCreator
from .placenet import PlaceNet


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
        self.declare_parameter('save_node_freq', 5)
        self.declare_parameter('show_histogram', True)

        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').get_parameter_value().string_value
        self.img_width = self.get_parameter('image_width').get_parameter_value().integer_value
        self.img_height = self.get_parameter('image_height').get_parameter_value().integer_value
        self.log_name = self.get_parameter('log_name').get_parameter_value().string_value
        self.max_data_count = self.get_parameter('max_data_count').get_parameter_value().integer_value
        self.save_node_freq = self.get_parameter('save_node_freq').get_parameter_value().integer_value
        self.show_histogram = self.get_parameter('show_histogram').get_parameter_value().bool_value

        self.save_log_path = os.path.join(pkg_dir, '..', 'logs')
        self.dataset_dir = os.path.join(self.save_log_path, self.log_name)
        self.images_dir = os.path.join(self.dataset_dir, 'images')
        self.angle_dir = os.path.join(self.dataset_dir, 'angle')
        self.action_dir = os.path.join(self.dataset_dir, 'action')
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.angle_dir, exist_ok=True)
        os.makedirs(self.action_dir, exist_ok=True)

        self.topo_map_dir = os.path.join(self.save_log_path, 'topo_map')
        os.makedirs(self.topo_map_dir, exist_ok=True)
        self.topo_map_yaml = os.path.join(self.topo_map_dir, 'topomap.yaml')
        self.image_dir = os.path.join(self.topo_map_dir, 'images')
        os.makedirs(self.image_dir, exist_ok=True)
        self.weight_path = os.path.join(pkg_dir, '..', 'weights', 'efficientnet_85x85.pth')
        self.map_creator = TopologicalMapCreator(self.topo_map_yaml, self.image_dir, self.weight_path)

        self.bridge = CvBridge()
        self.last_ang_vel = 0.0
        self.command_mode = "roadside"
        self.action_to_index = {"roadside": 0, "straight": 1, "left": 2, "right": 3}
        self.save_flag = False
        self.image_save_counter = 1
        self.data_count = 0
        self.action_counts = {"roadside": 0, "straight": 0, "left": 0, "right": 0}
        self.latest_image_ = np.array([])
        
        # ヒストグラム表示用の図を初期化（パラメータで有効な場合のみ）
        self.fig = None
        self.ax = None
        if self.show_histogram:
            plt.ion()  # インタラクティブモードON
            self.fig, self.ax = plt.subplots(figsize=(10, 6))

        self.image_sub = self.create_subscription(Image, self.image_topic, self.image_callback, 10)
        self.cmd_sub = self.create_subscription(Twist, self.cmd_vel_topic, self.cmd_callback, 10)
        self.cmd_route_sub = self.create_subscription(String, "/cmd_route", self.command_mode_callback, 10)
        self.cmd_save_image_sub = self.create_subscription(Empty, "/save_image", self.save_image_callback, 10)
        self.save_flag_sub = self.create_subscription(Bool, '/save', self.save_callback, 10)

        self.get_logger().info(f"Saving dataset to: {self.dataset_dir}")
        self.create_timer(self.save_node_freq, self.save_topomap_periodic)


    def cmd_callback(self, msg):
        self.last_ang_vel = msg.angular.z

    def command_mode_callback(self, msg):
        if msg.data in self.action_to_index:
            self.command_mode = msg.data
        else:
            self.get_logger().warn(f"Unknown command: {msg.data}, using 'roadside' instead.")
            self.command_mode = "roadside"

    def save_callback(self, msg):
        if msg.data:
            self.save_flag = True
            self.get_logger().info("Save flag received, shutting down data collection.")
        else:
            self.save_flag = False

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg)
            self.latest_image_ = cv_image.copy()
            
            if not self.save_flag or self.data_count >= self.max_data_count:
                return
                
            resized = cv2.resize(cv_image, (self.img_width, self.img_height))

            idx_str = f"{self.data_count + 1:05d}"

            # 保存パス
            img_path = os.path.join(self.images_dir, f"{idx_str}.png")
            angle_path = os.path.join(self.angle_dir, f"{idx_str}.csv")
            action_path = os.path.join(self.action_dir, f"{idx_str}.csv")

            # 保存
            cv2.imwrite(img_path, resized)
            np.savetxt(angle_path, np.array([self.last_ang_vel]), delimiter=",")
            np.savetxt(action_path, np.array([self.action_to_index[self.command_mode]]), fmt='%d', delimiter=",")

            self.action_counts[self.command_mode] += 1
            self.data_count += 1
            
            # ヒストグラムを毎回更新（save_flagがTrueかつshow_histogramが有効な場合のみ）
            if self.save_flag and self.show_histogram:
                self.display_histogram()
            
            self.get_logger().info(f"📸 Sample saved: {idx_str}, action: {self.command_mode}, angle: {self.last_ang_vel:.3f}")

        except Exception as e:
            self.get_logger().error(f"[image_callback] Failed to save data: {e}")

    def save_image_callback(self, msg):
        if self.data_count >= self.max_data_count:
            return

        try:
            if not self.latest_image_.empty():
                idx_str = f"{self.image_save_counter:05d}"
                img_path = os.path.join(self.image_dir, f"{idx_str}.png")
                cv2.imwrite(img_path, self.latest_image_)
                self.image_save_counter += 1
                self.get_logger().info(f"🗺️ Topological map image saved: {idx_str}")
        except Exception as e:
            self.get_logger().error(f"[save_image_callback] Failed to save image: {e}")

    def save_topomap_image(self, image):
        try:
            self.map_creator.add_node(image)
        except Exception as e:
            self.get_logger().error(f"[save_topomap_image] Failed to add node: {e}")

    def save_topomap_periodic(self):
        try:
            if hasattr(self, 'latest_image_') and not self.latest_image_.empty():
                self.save_topomap_image(self.latest_image_)
        except Exception as e:
            self.get_logger().error(f"[save_topomap_periodic] Failed to save periodic: {e}")

    def display_histogram(self):
        try:
            if self.fig is None or self.ax is None:
                return
                
            self.ax.clear()
            actions = list(self.action_counts.keys())
            counts = list(self.action_counts.values())
            
            bars = self.ax.bar(actions, counts, color=['blue', 'green', 'orange', 'red'])
            self.ax.set_title(f'Action Distribution (Total: {self.data_count})')
            self.ax.set_ylabel('Count')
            
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                self.ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{count}', ha='center', va='bottom')
            
            plt.draw()
            plt.pause(0.001)
        except Exception as e:
            self.get_logger().error(f"[display_histogram] Failed to update histogram: {e}")

    def destroy_node(self):
        self.get_logger().info(f"Total samples collected: {self.data_count}")
        self.get_logger().info(f"Action distribution: {self.action_counts}")
        if self.fig is not None:
            plt.close(self.fig)
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
