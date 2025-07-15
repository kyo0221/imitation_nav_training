#!/usr/bin/env python3
"""
WebDataset解析スクリプト
画像・角速度の分布とactionの統計を表示
"""

import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import webdataset as wds
import cv2
from collections import defaultdict, Counter
from tqdm import tqdm
import glob
import sys

# WebDatasetLoaderをインポート
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from augment.webdataset_loader import WebDatasetLoader


class WebDatasetAnalyzer:
    def __init__(self, dataset_dir, output_dir=None):
        """
        WebDatasetを解析するクラス
        
        Args:
            dataset_dir: WebDatasetのシャードファイルが含まれるディレクトリ
            output_dir: 解析結果を保存するディレクトリ
        """
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir or os.path.join(dataset_dir, 'analysis')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 統計情報を格納する辞書
        self.stats = {
            'total_samples': 0,
            'actions': defaultdict(int),
            'angles': [],
            'image_stats': {
                'mean_rgb': [],
                'std_rgb': [],
                'brightness': [],
                'contrast': []
            },
            'metadata': {}
        }
        
        # WebDatasetLoaderを使用してデータセットを読み込み
        self.webdataset_loader = WebDatasetLoader(
            dataset_dir=self.dataset_dir,
            input_size=(88, 200),  # デフォルトサイズ
            shift_aug=False,       # 拡張なし
            yaw_aug=False,         # 拡張なし
            n_action_classes=4
        )
        
        print(f"Loaded WebDataset from: {self.dataset_dir}")
        print(f"Total samples: {len(self.webdataset_loader)}")
        
        # 統計情報ファイルを読み込み
        self._load_dataset_stats()
    
    
    def _load_dataset_stats(self):
        """dataset_stats.jsonを読み込み"""
        stats_file = os.path.join(self.dataset_dir, "dataset_stats.json")
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                self.stats['metadata'] = json.load(f)
                print(f"Loaded dataset stats: {self.stats['metadata']}")
        else:
            print("dataset_stats.json not found")
    
    def _process_image_array(self, img_array):
        """画像配列を処理して統計を計算"""
        # 画像統計を計算
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            # BGRからRGBに変換
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            
            # 平均と標準偏差を計算
            mean_rgb = np.mean(img_rgb, axis=(0, 1))
            std_rgb = np.std(img_rgb, axis=(0, 1))
            
            # 明度を計算（グレースケール変換）
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            return {
                'mean_rgb': mean_rgb,
                'std_rgb': std_rgb,
                'brightness': brightness,
                'contrast': contrast
            }
        
        return None
    
    def analyze(self, sample_limit=None):
        """WebDatasetを解析"""
        print(f"Analyzing WebDataset in {self.dataset_dir}")
        
        # WebDatasetLoaderからサンプルを取得
        total_samples = len(self.webdataset_loader)
        analyze_count = sample_limit if sample_limit and sample_limit < total_samples else total_samples
        
        # 内部キャッシュを使用してサンプルを取得
        if not hasattr(self.webdataset_loader, '_samples_cache'):
            self.webdataset_loader._samples_cache = list(self.webdataset_loader.dataset)
        
        samples = self.webdataset_loader._samples_cache[:analyze_count]
        
        sample_count = 0
        for sample_data in tqdm(samples, desc="Analyzing samples"):
            try:
                # サンプルデータを取得
                img_data, angle_data, action_data = sample_data
                
                # JSONデータを解析
                angle_info = json.loads(angle_data)
                action_info = json.loads(action_data)
                
                # 角速度を記録
                angle = angle_info.get('angle', 0.0)
                self.stats['angles'].append(angle)
                
                # アクションを記録
                action = action_info.get('action', 0)
                self.stats['actions'][action] += 1
                
                # 画像データを処理（PIL Imageから変換）
                if hasattr(img_data, 'mode'):  # PIL Image
                    img_array = np.array(img_data)
                    if len(img_array.shape) == 3:
                        # RGBからBGRに変換してOpenCVと統一
                        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    
                    img_stats = self._process_image_array(img_array)
                    if img_stats:
                        self.stats['image_stats']['mean_rgb'].append(img_stats['mean_rgb'])
                        self.stats['image_stats']['std_rgb'].append(img_stats['std_rgb'])
                        self.stats['image_stats']['brightness'].append(img_stats['brightness'])
                        self.stats['image_stats']['contrast'].append(img_stats['contrast'])
                
                sample_count += 1
                
            except Exception as e:
                print(f"Error processing sample {sample_count}: {e}")
                continue
        
        self.stats['total_samples'] = sample_count
        print(f"Analyzed {sample_count} samples")
        
        # 統計を計算
        self._calculate_statistics()
        
        # 結果を保存
        self._save_results()
        
        # 可視化
        self._create_visualizations()
    
    def _calculate_statistics(self):
        """統計情報を計算"""
        # 角速度の統計
        if self.stats['angles']:
            angles = np.array(self.stats['angles'])
            self.stats['angle_stats'] = {
                'mean': float(np.mean(angles)),
                'std': float(np.std(angles)),
                'min': float(np.min(angles)),
                'max': float(np.max(angles)),
                'median': float(np.median(angles))
            }
        
        # 画像統計の平均
        if self.stats['image_stats']['mean_rgb']:
            mean_rgb = np.array(self.stats['image_stats']['mean_rgb'])
            self.stats['image_summary'] = {
                'mean_rgb_overall': np.mean(mean_rgb, axis=0).tolist(),
                'mean_brightness': float(np.mean(self.stats['image_stats']['brightness'])),
                'mean_contrast': float(np.mean(self.stats['image_stats']['contrast']))
            }
    
    def _save_results(self):
        """解析結果をJSONファイルに保存"""
        # NumPy配列をリストに変換
        results = {
            'total_samples': self.stats['total_samples'],
            'actions': dict(self.stats['actions']),
            'angle_stats': self.stats.get('angle_stats', {}),
            'image_summary': self.stats.get('image_summary', {}),
            'metadata': self.stats['metadata']
        }
        
        output_file = os.path.join(self.output_dir, 'analysis_results.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Analysis results saved to: {output_file}")
    
    def _create_visualizations(self):
        """可視化を作成"""
        # 1. アクション分布のヒストグラム
        self._plot_action_distribution()
        
        # 2. 角速度分布のヒストグラム
        self._plot_angle_distribution()
        
        # 3. 画像統計の分布
        self._plot_image_statistics()
        
        # 4. 統計サマリー
        self._create_summary_plot()
    
    def _plot_action_distribution(self):
        """アクション分布をプロット"""
        if not self.stats['actions']:
            return
        
        plt.figure(figsize=(10, 6))
        
        actions = list(self.stats['actions'].keys())
        counts = list(self.stats['actions'].values())
        
        # アクションラベルのマッピング
        action_labels = {0: 'roadside', 1: 'straight', 2: 'left', 3: 'right'}
        labels = [action_labels.get(action, f'action_{action}') for action in actions]
        
        bars = plt.bar(labels, counts, color=['orange', 'blue', 'green', 'red'])
        plt.title(f'Action Distribution (Total: {sum(counts)})')
        plt.xlabel('Action Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # 各バーの上に数値を表示
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{count}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'action_distribution.png'), dpi=300)
        plt.close()
    
    def _plot_angle_distribution(self):
        """角速度分布をプロット"""
        if not self.stats['angles']:
            return
        
        plt.figure(figsize=(12, 8))
        
        angles = np.array(self.stats['angles'])
        
        # ヒストグラム
        plt.subplot(2, 2, 1)
        plt.hist(angles, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Angular Velocity Distribution')
        plt.xlabel('Angular Velocity (rad/s)')
        plt.ylabel('Frequency')
        
        # 累積分布
        plt.subplot(2, 2, 2)
        sorted_angles = np.sort(angles)
        y = np.arange(len(sorted_angles)) / len(sorted_angles)
        plt.plot(sorted_angles, y, linewidth=2)
        plt.title('Cumulative Distribution')
        plt.xlabel('Angular Velocity (rad/s)')
        plt.ylabel('Cumulative Probability')
        
        # ボックスプロット
        plt.subplot(2, 2, 3)
        plt.boxplot(angles, vert=True)
        plt.title('Angular Velocity Box Plot')
        plt.ylabel('Angular Velocity (rad/s)')
        
        # 統計情報テキスト
        plt.subplot(2, 2, 4)
        plt.axis('off')
        stats_text = f"""
        Mean: {self.stats['angle_stats']['mean']:.4f}
        Std: {self.stats['angle_stats']['std']:.4f}
        Min: {self.stats['angle_stats']['min']:.4f}
        Max: {self.stats['angle_stats']['max']:.4f}
        Median: {self.stats['angle_stats']['median']:.4f}
        """
        plt.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'angle_distribution.png'), dpi=300)
        plt.close()
    
    def _plot_image_statistics(self):
        """画像統計をプロット"""
        if not self.stats['image_stats']['brightness']:
            return
        
        plt.figure(figsize=(15, 10))
        
        # 明度分布
        plt.subplot(2, 3, 1)
        plt.hist(self.stats['image_stats']['brightness'], bins=30, alpha=0.7, color='yellow', edgecolor='black')
        plt.title('Brightness Distribution')
        plt.xlabel('Brightness')
        plt.ylabel('Frequency')
        
        # コントラスト分布
        plt.subplot(2, 3, 2)
        plt.hist(self.stats['image_stats']['contrast'], bins=30, alpha=0.7, color='purple', edgecolor='black')
        plt.title('Contrast Distribution')
        plt.xlabel('Contrast')
        plt.ylabel('Frequency')
        
        # RGB平均値分布
        if self.stats['image_stats']['mean_rgb']:
            mean_rgb = np.array(self.stats['image_stats']['mean_rgb'])
            
            plt.subplot(2, 3, 3)
            plt.hist(mean_rgb[:, 0], bins=30, alpha=0.7, color='red', label='Red', edgecolor='black')
            plt.hist(mean_rgb[:, 1], bins=30, alpha=0.7, color='green', label='Green', edgecolor='black')
            plt.hist(mean_rgb[:, 2], bins=30, alpha=0.7, color='blue', label='Blue', edgecolor='black')
            plt.title('RGB Mean Value Distribution')
            plt.xlabel('Mean Value')
            plt.ylabel('Frequency')
            plt.legend()
            
            # RGB標準偏差分布
            std_rgb = np.array(self.stats['image_stats']['std_rgb'])
            
            plt.subplot(2, 3, 4)
            plt.hist(std_rgb[:, 0], bins=30, alpha=0.7, color='red', label='Red', edgecolor='black')
            plt.hist(std_rgb[:, 1], bins=30, alpha=0.7, color='green', label='Green', edgecolor='black')
            plt.hist(std_rgb[:, 2], bins=30, alpha=0.7, color='blue', label='Blue', edgecolor='black')
            plt.title('RGB Standard Deviation Distribution')
            plt.xlabel('Standard Deviation')
            plt.ylabel('Frequency')
            plt.legend()
        
        # 明度 vs コントラスト散布図
        plt.subplot(2, 3, 5)
        plt.scatter(self.stats['image_stats']['brightness'], self.stats['image_stats']['contrast'], 
                   alpha=0.5, s=1)
        plt.title('Brightness vs Contrast')
        plt.xlabel('Brightness')
        plt.ylabel('Contrast')
        
        # 画像統計サマリー
        plt.subplot(2, 3, 6)
        plt.axis('off')
        summary_text = f"""
        Image Statistics Summary:
        
        Brightness:
        Mean: {np.mean(self.stats['image_stats']['brightness']):.2f}
        Std: {np.std(self.stats['image_stats']['brightness']):.2f}
        
        Contrast:
        Mean: {np.mean(self.stats['image_stats']['contrast']):.2f}
        Std: {np.std(self.stats['image_stats']['contrast']):.2f}
        
        RGB Mean (overall):
        R: {self.stats['image_summary']['mean_rgb_overall'][0]:.2f}
        G: {self.stats['image_summary']['mean_rgb_overall'][1]:.2f}
        B: {self.stats['image_summary']['mean_rgb_overall'][2]:.2f}
        """
        plt.text(0.05, 0.95, summary_text, fontsize=10, verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'image_statistics.png'), dpi=300)
        plt.close()
    
    def _create_summary_plot(self):
        """統計サマリーを作成"""
        plt.figure(figsize=(16, 10))
        
        # 全体のサマリー情報
        plt.subplot(2, 2, 1)
        plt.axis('off')
        summary_text = f"""
        Dataset Analysis Summary
        
        Total Samples: {self.stats['total_samples']}
        
        Actions:
        {chr(10).join([f"  {k}: {v}" for k, v in self.stats['actions'].items()])}
        
        Angular Velocity:
        Mean: {self.stats.get('angle_stats', {}).get('mean', 0):.4f}
        Std: {self.stats.get('angle_stats', {}).get('std', 0):.4f}
        Range: [{self.stats.get('angle_stats', {}).get('min', 0):.4f}, {self.stats.get('angle_stats', {}).get('max', 0):.4f}]
        
        Dataset Info:
        {chr(10).join([f"  {k}: {v}" for k, v in self.stats['metadata'].items() if k != 'input_directory'])}
        """
        plt.text(0.05, 0.95, summary_text, fontsize=12, verticalalignment='top')
        
        # アクション分布（円グラフ）
        plt.subplot(2, 2, 2)
        if self.stats['actions']:
            action_labels = {0: 'roadside', 1: 'straight', 2: 'left', 3: 'right'}
            labels = [action_labels.get(action, f'action_{action}') for action in self.stats['actions'].keys()]
            sizes = list(self.stats['actions'].values())
            colors = ['orange', 'blue', 'green', 'red']
            
            plt.pie(sizes, labels=labels, colors=colors[:len(sizes)], autopct='%1.1f%%', startangle=90)
            plt.title('Action Distribution')
        
        # 角速度分布（簡略版）
        plt.subplot(2, 2, 3)
        if self.stats['angles']:
            angles = np.array(self.stats['angles'])
            plt.hist(angles, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            plt.title('Angular Velocity Distribution')
            plt.xlabel('Angular Velocity (rad/s)')
            plt.ylabel('Frequency')
        
        # 明度分布（簡略版）
        plt.subplot(2, 2, 4)
        if self.stats['image_stats']['brightness']:
            plt.hist(self.stats['image_stats']['brightness'], bins=30, alpha=0.7, color='yellow', edgecolor='black')
            plt.title('Image Brightness Distribution')
            plt.xlabel('Brightness')
            plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'dataset_summary.png'), dpi=300)
        plt.close()
    
    def print_summary(self):
        """解析結果の要約を表示"""
        print("\n" + "="*50)
        print("WEBDATASET ANALYSIS SUMMARY")
        print("="*50)
        
        print(f"Total Samples: {self.stats['total_samples']}")
        
        print("\nAction Distribution:")
        action_labels = {0: 'roadside', 1: 'straight', 2: 'left', 3: 'right'}
        for action, count in self.stats['actions'].items():
            label = action_labels.get(action, f'action_{action}')
            percentage = (count / self.stats['total_samples']) * 100 if self.stats['total_samples'] > 0 else 0
            print(f"  {label}: {count} ({percentage:.1f}%)")
        
        if 'angle_stats' in self.stats:
            print("\nAngular Velocity Statistics:")
            print(f"  Mean: {self.stats['angle_stats']['mean']:.4f} rad/s")
            print(f"  Std: {self.stats['angle_stats']['std']:.4f} rad/s")
            print(f"  Range: [{self.stats['angle_stats']['min']:.4f}, {self.stats['angle_stats']['max']:.4f}] rad/s")
        
        if 'image_summary' in self.stats:
            print("\nImage Statistics:")
            print(f"  Mean brightness: {self.stats['image_summary']['mean_brightness']:.2f}")
            print(f"  Mean contrast: {self.stats['image_summary']['mean_contrast']:.2f}")
            rgb = self.stats['image_summary']['mean_rgb_overall']
            print(f"  RGB means: R={rgb[0]:.2f}, G={rgb[1]:.2f}, B={rgb[2]:.2f}")
        
        print(f"\nAnalysis results saved to: {self.output_dir}")
        print("="*50)
    
    def view_images_interactive(self, sample_limit=100):
        """画像を対話的に表示"""
        print("\n" + "="*50)
        print("INTERACTIVE IMAGE VIEWER")
        print("="*50)
        print("Controls:")
        print("  → (Right Arrow): Next image")
        print("  ← (Left Arrow): Previous image")
        print("  'q': Quit")
        print("  'a': Toggle action display")
        print("  'r': Reset to first image")
        print("="*50)
        
        # WebDatasetLoaderからサンプルを取得
        if not hasattr(self.webdataset_loader, '_samples_cache'):
            self.webdataset_loader._samples_cache = list(self.webdataset_loader.dataset)
        
        samples = self.webdataset_loader._samples_cache
        total_samples = len(samples)
        display_samples = min(sample_limit, total_samples) if sample_limit else total_samples
        
        print(f"Loaded {display_samples} samples for viewing")
        
        current_idx = 0
        show_actions = True
        
        # matplotlib設定
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 8))
        
        def update_display():
            """現在の画像を表示"""
            if current_idx >= display_samples:
                return
            
            try:
                # サンプルデータを取得
                img_data, angle_data, action_data = samples[current_idx]
                
                # JSONデータを解析
                angle_info = json.loads(angle_data)
                action_info = json.loads(action_data)
                
                # 画像データを処理
                if hasattr(img_data, 'mode'):  # PIL Image
                    img_array = np.array(img_data)
                    # PIL ImageはRGBなのでそのまま表示
                    display_img = img_array
                else:
                    display_img = img_data
                
                # 画像を表示
                ax.clear()
                ax.imshow(display_img)
                ax.axis('off')
                
                # タイトルを設定
                angle = angle_info.get('angle', 0.0)
                action = action_info.get('action', 0)
                action_labels = {0: 'roadside', 1: 'straight', 2: 'left', 3: 'right'}
                action_label = action_labels.get(action, f'action_{action}')
                
                if show_actions:
                    title = f"Sample {current_idx + 1}/{display_samples} | Angle: {angle:.4f} | Action: {action_label}"
                else:
                    title = f"Sample {current_idx + 1}/{display_samples} | Angle: {angle:.4f}"
                
                ax.set_title(title, fontsize=12, pad=20)
                
                # 画面を更新
                fig.canvas.draw()
                fig.canvas.flush_events()
                
            except Exception as e:
                print(f"Error displaying sample {current_idx}: {e}")
        
        def on_key_press(event):
            """キー入力イベントを処理"""
            nonlocal current_idx, show_actions
            
            if event.key == 'right':  # 次の画像
                current_idx = min(current_idx + 1, display_samples - 1)
                update_display()
            elif event.key == 'left':  # 前の画像
                current_idx = max(current_idx - 1, 0)
                update_display()
            elif event.key == 'q':  # 終了
                plt.close(fig)
                return
            elif event.key == 'a':  # アクション表示切り替え
                show_actions = not show_actions
                update_display()
            elif event.key == 'r':  # 最初に戻る
                current_idx = 0
                update_display()
        
        # キー入力イベントを設定
        fig.canvas.mpl_connect('key_press_event', on_key_press)
        
        # 最初の画像を表示
        update_display()
        
        print("\nImage viewer started. Use arrow keys to navigate.")
        
        # イベントループ
        try:
            plt.show(block=True)
        except KeyboardInterrupt:
            print("\nViewer closed by user.")
        finally:
            plt.close('all')


def main():
    parser = argparse.ArgumentParser(description="Analyze WebDataset with interactive image viewer")
    parser.add_argument("dataset_dir", help="Directory containing WebDataset shard files (parent directory)")
    parser.add_argument("--output_dir", help="Output directory for analysis results")
    parser.add_argument("--sample_limit", type=int, help="Limit number of samples to analyze")
    parser.add_argument("--stats_only", action="store_true", help="Only run statistical analysis (skip image viewer)")
    parser.add_argument("--view_limit", type=int, default=100, help="Limit number of images to load for viewing")
    
    args = parser.parse_args()
    
    # train.pyと同じ方式でWebDatasetディレクトリを構築
    webdataset_dir = os.path.join(args.dataset_dir, 'webdataset')
    if not os.path.exists(webdataset_dir):
        print(f"Error: WebDataset directory {webdataset_dir} does not exist")
        print(f"Expected structure: {args.dataset_dir}/webdataset/")
        return 1
    
    # 解析を実行
    analyzer = WebDatasetAnalyzer(webdataset_dir, args.output_dir)
    
    if args.stats_only:
        # 統計解析のみを実行
        analyzer.analyze(sample_limit=args.sample_limit)
        analyzer.print_summary()
    else:
        # デフォルト: 画像ビューアーを起動
        analyzer.view_images_interactive(sample_limit=args.view_limit)
    
    return 0


if __name__ == "__main__":
    exit(main())