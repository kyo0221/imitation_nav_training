#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import numpy as np
import threading
import time

class CmdVelVisualizer(Node):
    def __init__(self):
        super().__init__('cmd_vel_visualizer')
        
        # ROS2サブスクライバー
        self.subscription = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )
        
        # データ保存用のdeque（最大1000個のデータポイント）
        self.max_points = 1000
        self.timestamps = deque(maxlen=self.max_points)
        self.angular_z = deque(maxlen=self.max_points)
        
        # 初期時刻
        self.start_time = time.time()
        
        # 最新のcmd_velデータ
        self.latest_angular_z = 0.0
        
        # データロック
        self.data_lock = threading.Lock()
        
        self.get_logger().info('cmd_vel visualizer started')
        self.get_logger().info('Subscribing to /cmd_vel topic')
    
    def cmd_vel_callback(self, msg):
        current_time = time.time() - self.start_time
        
        with self.data_lock:
            # タイムスタンプとangular.zデータを追加
            self.timestamps.append(current_time)
            self.angular_z.append(msg.angular.z)
            
            # 最新データを更新
            self.latest_angular_z = msg.angular.z

class RealtimePlotter:
    def __init__(self, visualizer_node):
        self.node = visualizer_node
        
        # matplotlib設定
        plt.style.use('dark_background')
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 5))
        self.fig.suptitle('cmd_vel Angular Z Real-time Visualization', fontsize=16, color='white')
        
        # 各軸の設定
        self.setup_axes()
        
        # プロット線の初期化
        self.lines = {}
        self.init_lines()
        
        # アニメーション設定
        self.ani = animation.FuncAnimation(
            self.fig, self.update_plot, interval=50, blit=False
        )
        
    def setup_axes(self):
        # Angular Z速度のプロット設定
        self.ax1.set_title('Angular Z Velocity', color='white')
        self.ax1.set_xlabel('Time [s]', color='white')
        self.ax1.set_ylabel('Angular Velocity [rad/s]', color='white')
        self.ax1.grid(True, alpha=0.3)
        
        # 現在値表示用のテキストエリア
        self.ax2.set_title('Current Values', color='white')
        self.ax2.axis('off')
        self.current_text = self.ax2.text(0.1, 0.5, '', fontsize=14, color='white', 
                                         transform=self.ax2.transAxes, family='monospace')
        
    def init_lines(self):
        # Angular Z速度の線
        self.lines['angular_z'], = self.ax1.plot([], [], 'cyan', linewidth=3)
        
    def update_plot(self, frame):
        with self.node.data_lock:
            if len(self.node.timestamps) == 0:
                return
            
            # データをリストに変換
            times = list(self.node.timestamps)
            angular_z = list(self.node.angular_z)
            latest_angular_z = self.node.latest_angular_z
        
        # Angular Z速度のプロット更新
        self.lines['angular_z'].set_data(times, angular_z)
        
        # 軸の範囲を自動調整
        if times:
            time_min, time_max = min(times), max(times)
            
            # X軸の範囲設定（最新10秒間を表示）
            display_min = max(0, time_max - 10)
            display_max = max(10, time_max)
            
            self.ax1.set_xlim(display_min, display_max)
            
            # Y軸の範囲を動的に調整
            self.auto_scale_y_axis(self.ax1, times, [angular_z], display_min, display_max)
        
        # 現在値のテキスト更新
        text = f"""Angular Velocity Z:
  Current: {latest_angular_z:8.4f} rad/s
  Degrees: {np.degrees(latest_angular_z):8.2f} deg/s

Data Points: {len(times)}
Time Range: {times[-1] if times else 0:.1f} seconds

Max: {max(angular_z) if angular_z else 0:8.4f} rad/s
Min: {min(angular_z) if angular_z else 0:8.4f} rad/s
Avg: {np.mean(angular_z) if angular_z else 0:8.4f} rad/s
"""
        self.current_text.set_text(text)
        
        return list(self.lines.values()) + [self.current_text]
    
    def auto_scale_y_axis(self, ax, times, data_lists, t_min, t_max):
        visible_data = []
        for data_list in data_lists:
            for t, val in zip(times, data_list):
                if t_min <= t <= t_max:
                    visible_data.append(val)
        
        if visible_data:
            y_min, y_max = min(visible_data), max(visible_data)
            y_range = y_max - y_min
            if y_range == 0:
                y_range = 1
            margin = y_range * 0.1
            ax.set_ylim(y_min - margin, y_max + margin)
        else:
            ax.set_ylim(-1, 1)

def main():
    rclpy.init()
    
    try:
        # ROS2ノードを作成
        visualizer = CmdVelVisualizer()
        
        # ROS2のスピニングを別スレッドで実行
        ros_thread = threading.Thread(target=lambda: rclpy.spin(visualizer), daemon=True)
        ros_thread.start()
        
        # リアルタイムプロッターを作成
        plotter = RealtimePlotter(visualizer)
        
        # matplotlibのGUIを表示
        plt.tight_layout()
        plt.show()
        
    except KeyboardInterrupt:
        print("Shutting down...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'visualizer' in locals():
            visualizer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()