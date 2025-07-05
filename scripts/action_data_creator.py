#!/usr/bin/env python3
"""
データセットのactionを手動で修正するインタラクティブスクリプト

画像を表示してキーボード入力でactionを修正します。
指定したactionフィルターに基づいて特定のactionのみを対象にできます。

キーボード操作:
- s: roadside (0)
- w: straight (1)
- a: left (2)
- d: right (3)
- n: next (変更せずに次へ)
- b: back (前の画像に戻る)
- q: quit (終了)
- r: reset (元のactionに戻す)

実行例:
- 全てのactionを対象: python action_data_creator.py /path/to/dataset
- roadsideのみ対象: python action_data_creator.py /path/to/dataset --filter roadside
- 複数のactionを対象: python action_data_creator.py /path/to/dataset --filter roadside,straight
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Button
from pathlib import Path
from datetime import datetime
import json
from PIL import Image


class ActionDataCreator:
    def __init__(self, dataset_dir, action_filter=None, backup=True):
        """
        ActionDataCreatorの初期化
        
        Args:
            dataset_dir (str): データセットディレクトリ
            action_filter (list): 対象とするactionのリスト（Noneの場合は全て）
            backup (bool): バックアップを作成するかどうか
        """
        self.dataset_path = Path(dataset_dir)
        self.images_dir = self.dataset_path / "images"
        self.angle_dir = self.dataset_path / "angle" 
        self.action_dir = self.dataset_path / "action"
        
        # ディレクトリの存在確認
        self._validate_dataset()
        
        # actionマッピング
        self.action_map = {
            's': 0,  # roadside
            'w': 1,  # straight
            'a': 2,  # left
            'd': 3   # right
        }
        
        self.action_names = {
            0: "roadside",
            1: "straight", 
            2: "left",
            3: "right"
        }
        
        self.filter_actions = action_filter
        
        # 辞書の初期化
        self.original_actions = {}  # 元のactionを保存
        self.modified_actions = {}  # 修正したaction
        
        # データの読み込み
        self.data_list = self._load_dataset()
        
        # バックアップの作成
        if backup and self.data_list:
            self._create_backup()
        
        # 状態変数
        self.current_index = 0
        self.total_items = len(self.data_list)
        
        print(f"Dataset loaded: {self.total_items} items")
        if self.filter_actions:
            print(f"Filter applied: {', '.join(self.filter_actions)}")
        else:
            print("No filter applied (all actions)")
    
    def _validate_dataset(self):
        """データセットディレクトリの検証"""
        missing_dirs = []
        if not self.images_dir.exists():
            missing_dirs.append("images")
        if not self.angle_dir.exists():
            missing_dirs.append("angle")
        if not self.action_dir.exists():
            missing_dirs.append("action")
        
        if missing_dirs:
            raise FileNotFoundError(f"Missing directories: {', '.join(missing_dirs)}")
    
    def _load_dataset(self):
        """データセットの読み込み"""
        print("Loading dataset...")
        
        # 画像ファイルの一覧を取得
        image_files = sorted(self.images_dir.glob("*.png"))
        data_list = []
        
        for image_file in image_files:
            base_name = image_file.stem
            angle_file = self.angle_dir / f"{base_name}.csv"
            action_file = self.action_dir / f"{base_name}.csv"
            
            # 対応するファイルが存在するかチェック
            if not (angle_file.exists() and action_file.exists()):
                continue
            
            try:
                # actionの読み込み
                action_value = int(np.loadtxt(action_file, delimiter=",", ndmin=1)[0])
                action_name = self.action_names.get(action_value, "unknown")
                
                # フィルターが指定されている場合はチェック
                if self.filter_actions and action_name not in self.filter_actions:
                    continue
                
                # angleの読み込み
                angle_value = float(np.loadtxt(angle_file, delimiter=",", ndmin=1)[0])
                
                data_item = {
                    'base_name': base_name,
                    'image_file': image_file,
                    'angle_file': angle_file,
                    'action_file': action_file,
                    'action_value': action_value,
                    'action_name': action_name,
                    'angle_value': angle_value
                }
                
                data_list.append(data_item)
                self.original_actions[base_name] = action_value
                
            except Exception as e:
                print(f"Warning: Failed to load {base_name}: {e}")
                continue
        
        return data_list
    
    def _create_backup(self):
        """actionディレクトリのバックアップを作成"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir = self.dataset_path / f"action_backup_{timestamp}"
        
        try:
            import shutil
            shutil.copytree(self.action_dir, backup_dir)
            print(f"Backup created: {backup_dir}")
        except Exception as e:
            print(f"Warning: Failed to create backup: {e}")
    
    def _load_image(self, data_item):
        """画像の読み込み"""
        try:
            image = Image.open(data_item['image_file'])
            return np.array(image)
        except Exception as e:
            raise ValueError(f"Failed to load image: {data_item['image_file']}: {e}")
    
    def _get_info_text(self, data_item):
        """画像情報のテキストを生成"""
        current_action = self.modified_actions.get(data_item['base_name'], data_item['action_value'])
        current_action_name = self.action_names.get(current_action, "unknown")
        
        info_text = f"""File: {data_item['base_name']} ({self.current_index + 1}/{self.total_items})
Original Action: {data_item['action_name']} ({data_item['action_value']})
Current Action: {current_action_name} ({current_action})
Angle: {data_item['angle_value']:.3f}

Controls: [s]roadside [w]straight [a]left [d]right
         [n]next [b]back [r]reset [q]quit"""
        
        return info_text
    
    def _save_action(self, data_item, new_action):
        """actionファイルを保存"""
        try:
            np.savetxt(data_item['action_file'], np.array([new_action]), fmt='%d', delimiter=",")
            self.modified_actions[data_item['base_name']] = new_action
            return True
        except Exception as e:
            print(f"Error saving action for {data_item['base_name']}: {e}")
            return False
    
    def _reset_action(self, data_item):
        """actionを元の値に戻す"""
        original_action = self.original_actions[data_item['base_name']]
        if self._save_action(data_item, original_action):
            if data_item['base_name'] in self.modified_actions:
                del self.modified_actions[data_item['base_name']]
            return True
        return False
    
    def _show_statistics(self):
        """修正統計を表示"""
        if not self.modified_actions:
            print("No modifications made.")
            return
        
        print("\nModification Statistics:")
        print("=" * 50)
        
        action_changes = {}
        for base_name, new_action in self.modified_actions.items():
            original_action = self.original_actions[base_name]
            change_key = f"{self.action_names[original_action]} -> {self.action_names[new_action]}"
            action_changes[change_key] = action_changes.get(change_key, 0) + 1
        
        for change, count in sorted(action_changes.items()):
            print(f"  {change}: {count} files")
        
        print(f"\nTotal modified files: {len(self.modified_actions)}")
    
    def run(self):
        """メインの実行ループ"""
        if not self.data_list:
            print("No data to process.")
            return
        
        print("\nStarting interactive action modification...")
        print("Use keyboard controls to modify actions. Press 'q' to quit.")
        
        # matplotlibの設定
        plt.ion()  # インタラクティブモードを有効化
        fig, (ax_img, ax_text) = plt.subplots(1, 2, figsize=(15, 8))
        fig.suptitle('Action Data Creator', fontsize=16)
        
        # キーボードイベントハンドラ
        def on_key_press(event):
            if event.key == 'q':  # 終了
                plt.close('all')
                return
            elif event.key == 'n':  # 次へ
                self.current_index = min(self.current_index + 1, self.total_items - 1)
                self._update_display(ax_img, ax_text)
            elif event.key == 'b':  # 前へ
                self.current_index = max(0, self.current_index - 1)
                self._update_display(ax_img, ax_text)
            elif event.key == 'r':  # リセット
                data_item = self.data_list[self.current_index]
                if self._reset_action(data_item):
                    print(f"Reset action for {data_item['base_name']}")
                    self._update_display(ax_img, ax_text)
            elif event.key in ['s', 'w', 'a', 'd']:  # action変更
                data_item = self.data_list[self.current_index]
                new_action = self.action_map[event.key]
                new_action_name = self.action_names[new_action]
                
                if self._save_action(data_item, new_action):
                    print(f"Changed {data_item['base_name']}: {data_item['action_name']} -> {new_action_name}")
                    self.current_index = min(self.current_index + 1, self.total_items - 1)  # 自動で次に進む
                    self._update_display(ax_img, ax_text)
        
        # キーボードイベントを登録
        fig.canvas.mpl_connect('key_press_event', on_key_press)
        
        # 初期表示
        self._update_display(ax_img, ax_text)
        
        try:
            # メインループ
            while plt.get_fignums():  # 図が開いている間
                plt.pause(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            plt.close('all')
        
        self._show_statistics()
        
        # 最終確認
        if self.modified_actions:
            save_summary = input(f"\nSave summary to file? (y/N): ")
            if save_summary.lower() == 'y':
                self._save_modification_summary()
    
    def _update_display(self, ax_img, ax_text):
        """画面表示を更新"""
        if self.current_index >= self.total_items:
            return
            
        data_item = self.data_list[self.current_index]
        
        try:
            # 画像の読み込みと表示
            image = self._load_image(data_item)
            
            ax_img.clear()
            ax_img.imshow(image)
            ax_img.set_title(f"Image: {data_item['base_name']}")
            ax_img.axis('off')
            
            # 情報テキストの表示
            info_text = self._get_info_text(data_item)
            ax_text.clear()
            ax_text.text(0.05, 0.95, info_text, transform=ax_text.transAxes, 
                        fontsize=12, verticalalignment='top', fontfamily='monospace')
            ax_text.axis('off')
            
            plt.draw()
            
        except Exception as e:
            print(f"Error displaying {data_item['base_name']}: {e}")
            self.current_index += 1
            if self.current_index < self.total_items:
                self._update_display(ax_img, ax_text)
    
    def _save_modification_summary(self):
        """修正内容のサマリーを保存"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_file = self.dataset_path / f"action_modifications_{timestamp}.json"
        
        summary_data = {
            'timestamp': timestamp,
            'dataset_dir': str(self.dataset_path),
            'filter_actions': self.filter_actions,
            'total_processed': self.total_items,
            'total_modified': len(self.modified_actions),
            'modifications': {}
        }
        
        for base_name, new_action in self.modified_actions.items():
            original_action = self.original_actions[base_name]
            summary_data['modifications'][base_name] = {
                'original_action': original_action,
                'original_action_name': self.action_names[original_action],
                'new_action': new_action,
                'new_action_name': self.action_names[new_action]
            }
        
        try:
            with open(summary_file, 'w') as f:
                json.dump(summary_data, f, indent=2)
            print(f"Modification summary saved: {summary_file}")
        except Exception as e:
            print(f"Failed to save summary: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive action data modification tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Keyboard Controls:
  s - Set action to roadside (0)
  w - Set action to straight (1)  
  a - Set action to left (2)
  d - Set action to right (3)
  n - Next image (no change)
  b - Previous image
  r - Reset to original action
  q - Quit
  ESC - Quit

Examples:
  # Modify all actions:
  python action_data_creator.py /path/to/dataset
  
  # Modify only roadside actions:
  python action_data_creator.py /path/to/dataset --filter roadside
  
  # Modify multiple specific actions:
  python action_data_creator.py /path/to/dataset --filter roadside,straight
  
  # No backup:
  python action_data_creator.py /path/to/dataset --no-backup
        """
    )
    
    parser.add_argument('dataset_dir', type=str,
                       help='Path to dataset directory')
    parser.add_argument('--filter', type=str, default=None,
                       help='Filter actions to modify (comma-separated: roadside,straight,left,right)')
    parser.add_argument('--no-backup', action='store_true',
                       help='Do not create backup of action files')
    parser.add_argument('--start-index', type=int, default=0,
                       help='Start from specific index (0-based)')
    
    args = parser.parse_args()
    
    # 入力検証
    if not os.path.exists(args.dataset_dir):
        print(f"Error: Dataset directory not found: {args.dataset_dir}")
        return 1
    
    # フィルターの処理
    action_filter = None
    if args.filter:
        valid_actions = ['roadside', 'straight', 'left', 'right']
        action_filter = [action.strip() for action in args.filter.split(',')]
        
        # フィルターの検証
        invalid_actions = [action for action in action_filter if action not in valid_actions]
        if invalid_actions:
            print(f"Error: Invalid actions in filter: {invalid_actions}")
            print(f"Valid actions: {valid_actions}")
            return 1
    
    try:
        # ActionDataCreatorの実行
        creator = ActionDataCreator(
            dataset_dir=args.dataset_dir,
            action_filter=action_filter,
            backup=not args.no_backup
        )
        
        # 開始インデックスの設定
        if args.start_index > 0:
            creator.current_index = min(args.start_index, creator.total_items - 1)
        
        creator.run()
        
        print("Action modification completed.")
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())