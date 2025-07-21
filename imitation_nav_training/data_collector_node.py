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
import time
import threading

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
        self.shard_lock = threading.Lock()  # スレッドセーフティのためのロック
        self._last_processed_image = None
        
        # シャードファイル管理
        self.completed_shards = []
        
        self._init_shard_writer()
        
        # ヒストグラム表示用の図を初期化（パラメータで有効な場合のみ）
        self.fig = None
        self.ax = None
        if self.show_histogram:
            try:
                plt.ion()  # インタラクティブモードON
                self.fig, self.ax = plt.subplots(figsize=(10, 6))
            except Exception as e:
                self.get_logger().warn(f"ヒストグラム表示の初期化に失敗: {e}")
                self.show_histogram = False

        self.image_sub = self.create_subscription(Image, self.image_topic, self.image_callback, 10)
        self.cmd_sub = self.create_subscription(Twist, self.cmd_vel_topic, self.cmd_callback, 10)
        self.cmd_route_sub = self.create_subscription(String, "/cmd_route", self.command_mode_callback, 10)
        self.cmd_save_image_sub = self.create_subscription(Empty, "/save_image", self.save_image_callback, 10)
        self.save_flag_sub = self.create_subscription(Bool, '/save', self.save_callback, 10)

        self.get_logger().info(f"Saving webdataset to: {self.webdataset_dir}")
        self.get_logger().info(f"Samples per shard: {self.samples_per_shard}")
        self.get_logger().info(f"Compression enabled: {self.enable_compression}")
        self.get_logger().info("Save format: numpy array")
        self.create_timer(self.save_node_freq, self.save_topomap_periodic)

    def _get_shard_filename(self, shard_id):
        """シャードファイル名を生成"""
        if self.enable_compression:
            return os.path.join(self.webdataset_dir, f"shard_{shard_id:06d}.tar.gz")
        else:
            return os.path.join(self.webdataset_dir, f"shard_{shard_id:06d}.tar")

    def _close_current_shard_and_start_next(self):
        """現在のシャードを閉じて次のシャードを開始"""
        with self.shard_lock:
            # WebDatasetのShardWriterは自動でシャード切り替えを行う
            # 手動でのシャード管理をやめて、WebDatasetに任せる
            self.current_shard += 1
            self.current_shard_count = 0
            self.get_logger().info(f"🗂️ シャード {self.current_shard-1} が満杯になりました。WebDatasetが自動で次のシャードに切り替えます")

    def _init_shard_writer(self):
        """最初のシャードライターを初期化"""
        with self.shard_lock:
            try:
                # WebDatasetのShardWriterは%形式のパターンを期待する
                if self.enable_compression:
                    shard_pattern = os.path.join(self.webdataset_dir, "shard_%06d.tar.gz")
                else:
                    shard_pattern = os.path.join(self.webdataset_dir, "shard_%06d.tar")
                
                self.shard_writer = wds.ShardWriter(shard_pattern, maxcount=self.samples_per_shard)
                self.current_shard_count = 0
                
                # 実際のファイル名を表示用に生成
                actual_filename = shard_pattern % self.current_shard
                self.get_logger().info(f"🗂️ 初期シャードを開始: {actual_filename} (maxcount={self.samples_per_shard})")
                
            except Exception as e:
                self.get_logger().error(f"初期シャードライター初期化エラー: {e}")
                self.save_error_count += 1
                self.shard_writer = None

    def _get_current_shard_size(self):
        """現在のシャードのサイズを取得"""
        shard_filename = self._get_shard_filename(self.current_shard)
        
        try:
            if os.path.exists(shard_filename):
                return os.path.getsize(shard_filename)
            else:
                return 0
        except OSError as e:
            self.get_logger().warn(f"シャードサイズ取得エラー: {e}")
            return 0

    def _log_data_stats(self):
        """データ統計をログ出力"""
        current_shard_size = self._get_current_shard_size()
        
        self.get_logger().info(
            f"📊 データ統計: "
            f"合計サンプル数={self.data_count}, "
            f"現在のシャード={self.current_shard}, "
            f"シャード内サンプル数={self.current_shard_count}, "
            f"現在のシャードサイズ={current_shard_size / 1024:.2f}KB, "
            f"完了したシャード数={len(self.completed_shards)}"
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
            self.get_logger().info("Save flag received, starting data collection.")
        else:
            self.save_flag = False
            self.get_logger().info("Save flag disabled, stopping data collection.")

    def image_callback(self, msg):
        if not self.save_flag:  return

        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        resized = cv2.resize(cv_image, (self.img_width, self.img_height))

        # 最後に処理した画像を記録（TopoMap用）
        self._last_processed_image = resized.copy()

        # WebDataset形式で保存
        save_success = self._save_webdataset_sample(resized, self.last_ang_vel, self.action_to_index[self.command_mode], msg)
        
        if save_success:
            self.action_counts[self.command_mode] += 1
            self.data_count += 1
            self.current_shard_count += 1
            
            # シャードが満杯になったら新しいシャードを開始
            # WebDatasetのShardWriterが自動でシャード切り替えを行うため、カウンターのリセットのみ
            if self.current_shard_count >= self.samples_per_shard:
                self._close_current_shard_and_start_next()
            
            # ヒストグラムを定期的に更新
            if self.save_flag and self.show_histogram and self.data_count % 10 == 0:    self.display_histogram()
            if self.data_count % 100 == 0:  self._log_data_stats()
            if self.data_count % 10 == 0:   self.get_logger().info(f"📸 Sample saved: {self.data_count:05d}")
        else:
            self.get_logger().warn("サンプル保存に失敗しました")


    def _save_webdataset_sample(self, image, angle, action, msg):
        if self.shard_writer is None:
            self.get_logger().error("ShardWriter is not initialized")
            return False
        
        try:
            with self.shard_lock:
                img_buffer = io.BytesIO()
                np.save(img_buffer, image)
                img_data = img_buffer.getvalue()
                img_ext = "npy"
                
                # メタデータを作成
                metadata = {
                    'angle': float(angle),
                    'action': int(action),
                    'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9 if hasattr(msg, 'header') else time.time(),
                    'image_width': self.img_width,
                    'image_height': self.img_height,
                    'save_format': 'numpy',
                    'image_shape': list(image.shape),
                    'image_dtype': str(image.dtype),
                }
                
                # WebDatasetに書き込み
                sample_key = f"{self.data_count:06d}"
                sample_data = {
                    "__key__": sample_key,
                    img_ext: img_data,
                    "metadata.json": json.dumps(metadata),
                    "action.json": json.dumps({"action": int(action), "angle": float(angle)})
                }
                
                self.shard_writer.write(sample_data)
                return True
                
        except Exception as e:
            self.get_logger().error(f"WebDataset保存エラー: {e}")
            return False

    def save_image_callback(self, msg: Empty):
        self.save_topomap_image()

    def save_topomap_image(self):
        if self.data_count == 0:
            self.get_logger().warn("No image available yet for topomap.")
            return

        try:
            dst_name = f"{self.image_save_counter:05d}.png"
            dst_path = os.path.join(self.image_dir, dst_name)
            
            if self._last_processed_image is not None:
                resized_img = cv2.resize(self._last_processed_image, (85, 85))
                bgr_img = cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR) # rgb to bgr
                cv2.imwrite(dst_path, bgr_img)
                self.map_creator.add_node(dst_name, self.command_mode)
                self.image_save_counter += 1
                self.get_logger().info(f"🗺️ Topomap image saved: {dst_name}")
            else:
                self.get_logger().warn("No processed image available for topomap")
                
        except Exception as e:
            self.get_logger().error(f"[save_topomap_image] Failed to save topomap image: {e}")

    def save_topomap_periodic(self):
        if self.save_flag and self.data_count > 0:
            self.save_topomap_image()

    def display_histogram(self):
        """ヒストグラムを表示"""
        if not self.show_histogram or self.fig is None or self.ax is None:
            return
            
        try:
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
            
        except Exception as e:
            self.get_logger().warn(f"ヒストグラム表示エラー: {e}")
    
    def _save_dataset_stats(self):
        """データセット統計情報を保存"""
        # 最終シャードサイズを計算
        total_size = sum(os.path.getsize(shard) for shard in self.completed_shards if os.path.exists(shard))
        current_shard_size = self._get_current_shard_size()
        total_size += current_shard_size
        
        stats = {
            "total_samples": self.data_count,
            "total_shards": len(self.completed_shards) + (1 if self.current_shard_count > 0 else 0),
            "completed_shards": len(self.completed_shards),
            "current_shard_samples": self.current_shard_count,
            "samples_per_shard": self.samples_per_shard,
            "compression_enabled": self.enable_compression,
            "save_format": "numpy",
            "action_distribution": dict(self.action_counts),
            "dataset_directory": self.webdataset_dir,
            "image_size": [self.img_height, self.img_width],
            "max_data_count": self.max_data_count,
            "total_dataset_size_bytes": total_size,
            "save_error_count": self.save_error_count,
            "shard_files": [
                {"filename": os.path.basename(shard), "size": os.path.getsize(shard)}
                for shard in self.completed_shards if os.path.exists(shard)
            ]
        }
        
        # 現在のシャードも追加
        if self.current_shard_count > 0:
            current_shard_file = self._get_shard_filename(self.current_shard)
            if os.path.exists(current_shard_file):
                stats["shard_files"].append({
                    "filename": os.path.basename(current_shard_file),
                    "size": os.path.getsize(current_shard_file)
                })
        
        stats_file = os.path.join(self.webdataset_dir, "dataset_stats.json")
        try:
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            self.get_logger().info(f"📊 統計情報を保存: {stats_file}")
        except Exception as e:
            self.get_logger().error(f"統計情報保存エラー: {e}")

    def destroy_node(self):
        """ノードを安全に終了"""
        self.get_logger().info("🛑 データ収集ノードを終了します...")
        
        # ヒストグラムを最終更新
        if self.show_histogram:
            try:
                self.display_histogram()
                if self.fig is not None:
                    plt.close(self.fig)
                    self.fig = None
            except:
                pass
        
        # WebDatasetの保存を完了
        if self.shard_writer is not None:
            self._log_data_stats()
            try:
                with self.shard_lock:
                    self.shard_writer.close()
                    final_shard = self._get_shard_filename(self.current_shard)
                    
                    # ファイルシステムの同期を待つ
                    
                    if os.path.exists(final_shard):
                        self.completed_shards.append(final_shard)
                        file_size = os.path.getsize(final_shard)
                        self.get_logger().info(f"🗂️ 最終シャード {self.current_shard} を保存: {final_shard} ({file_size} bytes)")
                    else:
                        self.get_logger().warn(f"⚠️ 最終シャードファイルが見つかりません: {final_shard}")
            except Exception as e:
                self.get_logger().error(f"最終シャード保存エラー: {e}")
        
        # 統計情報ファイルを保存
        self._save_dataset_stats()
        
        # 完了したシャードの一覧を表示
        self.get_logger().info(f"🗂️ 完了したシャード数: {len(self.completed_shards)}")
        for i, shard in enumerate(self.completed_shards):
            if os.path.exists(shard):
                size = os.path.getsize(shard)
                self.get_logger().info(f"  {i+1}. {os.path.basename(shard)} ({size} bytes)")
        
        # TopoMapを保存
        try:
            self.map_creator.save_map()
            self.get_logger().info("🗺️ TopoMap保存完了")
        except Exception as e:
            self.get_logger().error(f"TopoMap保存エラー")
        
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DataCollector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n[INFO] Graceful shutdown by Ctrl+C.")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(0)