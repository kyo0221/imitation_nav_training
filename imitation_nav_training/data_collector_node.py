import os
import sys
import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty, Bool, String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

from cv_bridge import CvBridge
import cv2
import numpy as np
import matplotlib.pyplot as plt
import webdataset as wds
import io
import json

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
        self.declare_parameter('log_name', 'dataset')
        self.declare_parameter('max_data_count', 50000)
        self.declare_parameter('save_node_freq', 5)
        self.declare_parameter('show_histogram', True)
        self.declare_parameter('samples_per_shard', 1000)
        self.declare_parameter('enable_compression', True)
        self.declare_parameter('swap_rb_channels', False)

        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').get_parameter_value().string_value
        self.img_width = self.get_parameter('image_width').get_parameter_value().integer_value
        self.img_height = self.get_parameter('image_height').get_parameter_value().integer_value
        self.log_name = self.get_parameter('log_name').get_parameter_value().string_value
        self.max_data_count = self.get_parameter('max_data_count').get_parameter_value().integer_value
        self.save_node_freq = self.get_parameter('save_node_freq').get_parameter_value().integer_value
        self.show_histogram = self.get_parameter('show_histogram').get_parameter_value().bool_value
        self.samples_per_shard = self.get_parameter('samples_per_shard').get_parameter_value().integer_value
        self.enable_compression = self.get_parameter('enable_compression').get_parameter_value().bool_value
        self.swap_rb_channels = self.get_parameter('swap_rb_channels').get_parameter_value().bool_value

        self.save_log_path = os.path.join(pkg_dir, '..', 'logs')
        self.dataset_dir = os.path.join(self.save_log_path, self.log_name)
        self.webdataset_dir = os.path.join(self.dataset_dir, 'webdataset')
        os.makedirs(self.webdataset_dir, exist_ok=True)

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
        
        # WebDataset用の設定
        self.current_shard = 0
        self.current_shard_count = 0
        self.shard_writer = None
        self.total_data_size = 0
        self._init_shard_writer()
        
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

        self.get_logger().info(f"Saving webdataset to: {self.webdataset_dir}")
        self.get_logger().info(f"Samples per shard: {self.samples_per_shard}")
        self.get_logger().info(f"Compression enabled: {self.enable_compression}")
        self.get_logger().info(f"Swap RB channels: {self.swap_rb_channels}")
        self.get_logger().info("Save format: numpy (fixed)")
        self.create_timer(self.save_node_freq, self.save_topomap_periodic)

    def _swap_rb_channels(self, image):
        """RとBチャンネルを入れ替える"""
        if self.swap_rb_channels and len(image.shape) == 3 and image.shape[2] >= 3:
            # BGRからRGBに変換（OpenCVのデフォルトはBGR）
            swapped_image = image.copy()
            swapped_image[:, :, 0], swapped_image[:, :, 2] = image[:, :, 2], image[:, :, 0]
            return swapped_image
        return image

    def _init_shard_writer(self):
        """新しいシャードライターを初期化"""
        if self.shard_writer is not None:
            self.shard_writer.close()
        
        # WebDatasetのShardWriterは%形式のパターンを期待する
        if self.enable_compression:
            shard_pattern = os.path.join(self.webdataset_dir, "shard_%06d.tar.gz")
        else:
            shard_pattern = os.path.join(self.webdataset_dir, "shard_%06d.tar")
        
        self.shard_writer = wds.ShardWriter(shard_pattern, maxcount=self.samples_per_shard)
        self.current_shard_count = 0
        
        # 実際のファイル名を表示用に生成
        actual_filename = shard_pattern % self.current_shard
        self.get_logger().info(f"🗂️ 新しいシャードを開始: {actual_filename}")

    def _get_current_shard_size(self):
        """現在のシャードのサイズを取得"""
        if self.shard_writer is None:
            return 0
        
        # 現在のシャードのファイル名を生成
        if self.enable_compression:
            shard_filename = os.path.join(self.webdataset_dir, f"shard_{self.current_shard:06d}.tar.gz")
        else:
            shard_filename = os.path.join(self.webdataset_dir, f"shard_{self.current_shard:06d}.tar")
        
        try:
            return os.path.getsize(shard_filename)
        except OSError:
            return 0

    def _log_data_stats(self):
        """データ統計をログ出力"""
        current_shard_size = self._get_current_shard_size()
        self.total_data_size += current_shard_size
        
        self.get_logger().info(
            f"📊 データ統計: "
            f"合計サンプル数={self.data_count}, "
            f"現在のシャード={self.current_shard}, "
            f"シャード内サンプル数={self.current_shard_count}, "
            f"累積データサイズ={self.total_data_size / 1024 / 1024:.2f}MB"
        )

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
        if not self.save_flag or self.data_count >= self.max_data_count:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg)
            resized = cv2.resize(cv_image, (self.img_width, self.img_height))
            
            # RとBチャンネルを反転（パラメータに応じて）
            processed_image = self._swap_rb_channels(resized)

            # 最後に処理した画像を記録（TopoMap用）
            self._last_processed_image = processed_image.copy()

            # WebDataset形式で保存
            self._save_webdataset_sample(processed_image, self.last_ang_vel, self.action_to_index[self.command_mode], msg)

            self.action_counts[self.command_mode] += 1
            self.data_count += 1
            self.current_shard_count += 1
            
            # シャードが満杯になったら新しいシャードを開始
            if self.current_shard_count >= self.samples_per_shard:
                self._log_data_stats()
                self.current_shard += 1
                self._init_shard_writer()
            
            # ヒストグラムを毎回更新（save_flagがTrueかつshow_histogramが有効な場合のみ）
            if self.save_flag and self.show_histogram:
                self.display_histogram()
            
            if self.data_count % 100 == 0:
                self._log_data_stats()
            
            self.get_logger().info(f"📸 Sample saved: {self.data_count:05d}")

        except Exception as e:
            self.get_logger().error(f"[image_callback] Failed to save data: {e}")

    def _save_webdataset_sample(self, image, angle, action, msg):
        """WebDataset形式でサンプルを保存（numpy形式固定）"""
        if self.shard_writer is None:
            self.get_logger().error("ShardWriter is not initialized")
            return
        
        # numpy形式で保存
        img_data = image.tobytes()
        img_ext = "npy"
        
        # メタデータを作成
        metadata = {
            'angle': float(angle),
            'action': int(action),
            'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9 if hasattr(msg, 'header') else 0,
            'image_width': self.img_width,
            'image_height': self.img_height,
            'save_format': 'numpy',
            'image_shape': list(image.shape),
            'image_dtype': str(image.dtype)
        }
        
        # WebDatasetに書き込み
        sample_key = f"{self.data_count:06d}"
        sample_data = {
            "__key__": sample_key,
            img_ext: img_data,
            "angle.json": json.dumps(metadata),
            "action.json": json.dumps({"action": int(action)})
        }
        
        self.shard_writer.write(sample_data)

    def save_image_callback(self, msg: Empty):
        self.save_topomap_image()

    def save_topomap_image(self):
        if self.data_count == 0:
            self.get_logger().warn("No image available yet for topomap.")
            return

        try:
            # 最新のデータカウントから画像を取得
            # WebDatasetからの画像取得は複雑なので、現在のサンプルから直接使用
            dst_name = f"img{self.image_save_counter:05d}.png"
            dst_path = os.path.join(self.image_dir, dst_name)
            
            # 最後に処理した画像を使用（すでにRB反転済み）
            if hasattr(self, '_last_processed_image') and self._last_processed_image is not None:
                resized_img = cv2.resize(self._last_processed_image, (85, 85))
                cv2.imwrite(dst_path, resized_img)
                self.map_creator.add_node(dst_name, self.command_mode)
                self.image_save_counter += 1
                self.get_logger().info(f"🗺️ Topomap image saved: {dst_name} (RB swap: {self.swap_rb_channels})")
            else:
                self.get_logger().warn("No processed image available for topomap")
                
        except Exception as e:
            self.get_logger().error(f"[save_topomap_image] Failed to save topomap image: {e}")

    def save_topomap_periodic(self):
        if self.save_flag and self.data_count > 0:
            self.save_topomap_image()

    def display_histogram(self):
        # ヒストグラム表示が無効またはfig/axが初期化されていない場合は何もしない
        if not self.show_histogram or self.fig is None or self.ax is None:
            return
            
        actions = list(self.action_counts.keys())
        counts = list(self.action_counts.values())
        
        # 既存の図をクリアして更新
        self.ax.clear()
        
        bars = self.ax.bar(actions, counts, color=['orange', 'blue', 'green', 'red'])
        self.ax.set_xlabel('Action Type')
        self.ax.set_ylabel('Count')
        self.ax.set_title(f'Data Collection Histogram (Total: {self.data_count})')
        
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            self.ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{count}', ha='center', va='bottom')
        
        # 図を更新して表示
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
        
        # ファイル保存
        histogram_path = os.path.join(self.dataset_dir, 'data_histogram.png')
        self.fig.savefig(histogram_path)
        
        self.get_logger().info(f"📊 Histogram updated: roadside={counts[0]}, straight={counts[1]}, left={counts[2]}, right={counts[3]}")

    def destroy_node(self):
        if self.show_histogram:
            self.display_histogram()
            plt.close('all')  # matplotlib ウィンドウを閉じる
        
        # WebDatasetの保存を完了
        if self.shard_writer is not None:
            self._log_data_stats()
            self.shard_writer.close()
            self.get_logger().info("🗂️ WebDataset保存完了")
        
        self.map_creator.save_map()
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
