#!/usr/bin/env python3
"""
既存のデータセットのactionファイルを新しい4つの制御入力形式に変換するスクリプト

従来の形式:
- straight: 0
- left: 1  
- right: 2

新しい形式:
- roadside: 0 (従来のstraightを変換)
- straight: 1 (新規追加)
- left: 2 (従来のleftを維持)
- right: 3 (従来のrightを変換)
"""

import os
import argparse
import numpy as np
from pathlib import Path
import shutil
from datetime import datetime


def convert_action_value(old_value):
    """
    古い行動値を新しい行動値に変換
    
    Args:
        old_value (int): 古い行動値 (0: straight, 1: left, 2: right)
    
    Returns:
        int: 新しい行動値 (0: roadside, 1: straight, 2: left, 3: right)
    """
    if old_value == 0:  # 従来のstraight -> roadside
        return 0
    elif old_value == 1:  # 従来のleft -> left
        return 2
    elif old_value == 2:  # 従来のright -> right
        return 3
    else:
        print(f"Warning: Unknown action value {old_value}, converting to roadside (0)")
        return 0


def convert_dataset(dataset_dir, output_dir, backup=True, dry_run=False):
    """
    データセット全体のactionファイルを変換
    
    Args:
        dataset_dir (str): 変換対象のデータセットディレクトリ
        output_dir (str): 出力先ディレクトリ（Noneの場合は上書き）
        backup (bool): バックアップを作成するかどうか
        dry_run (bool): 実際の変換を行わず、統計のみ表示
    """
    dataset_path = Path(dataset_dir)
    action_dir = dataset_path / "action"
    
    if not action_dir.exists():
        raise FileNotFoundError(f"Action directory not found: {action_dir}")
    
    # 出力先の設定
    if output_dir:
        output_path = Path(output_dir)
        output_action_dir = output_path / "action"
        # 他のディレクトリもコピー
        if not dry_run:
            output_path.mkdir(parents=True, exist_ok=True)
            if (dataset_path / "images").exists():
                shutil.copytree(dataset_path / "images", output_path / "images", dirs_exist_ok=True)
            if (dataset_path / "angle").exists():
                shutil.copytree(dataset_path / "angle", output_path / "angle", dirs_exist_ok=True)
            output_action_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_action_dir = action_dir
        output_path = dataset_path
    
    # バックアップの作成
    if backup and not output_dir and not dry_run:
        backup_dir = dataset_path / f"action_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copytree(action_dir, backup_dir)
        print(f"Backup created: {backup_dir}")
    
    # actionファイルの収集
    action_files = list(action_dir.glob("*.csv"))
    if not action_files:
        raise FileNotFoundError(f"No CSV files found in {action_dir}")
    
    # 統計情報の初期化
    stats = {
        'total_files': len(action_files),
        'old_actions': {0: 0, 1: 0, 2: 0},  # straight, left, right
        'new_actions': {0: 0, 1: 0, 2: 0, 3: 0},  # roadside, straight, left, right
        'converted_files': 0,
        'errors': []
    }
    
    print(f"Found {len(action_files)} action files")
    print(f"Converting from {dataset_dir} to {output_path}")
    
    if dry_run:
        print("DRY RUN MODE - No files will be modified")
    
    # ファイルごとの変換処理
    for action_file in sorted(action_files):
        try:
            # 古い行動値を読み込み
            old_action = int(np.loadtxt(action_file, delimiter=",", ndmin=1)[0])
            
            # 統計更新
            if old_action in stats['old_actions']:
                stats['old_actions'][old_action] += 1
            
            # 新しい行動値に変換
            new_action = convert_action_value(old_action)
            stats['new_actions'][new_action] += 1
            
            # ファイル保存（dry_runでない場合のみ）
            if not dry_run:
                output_file = output_action_dir / action_file.name
                np.savetxt(output_file, np.array([new_action]), fmt='%d', delimiter=",")
                stats['converted_files'] += 1
            
        except Exception as e:
            error_msg = f"Error processing {action_file}: {e}"
            stats['errors'].append(error_msg)
            print(f"ERROR: {error_msg}")
    
    # 結果の表示
    print("\n" + "="*60)
    print("CONVERSION SUMMARY")
    print("="*60)
    
    print(f"Total files processed: {stats['total_files']}")
    if not dry_run:
        print(f"Successfully converted: {stats['converted_files']}")
    print(f"Errors encountered: {len(stats['errors'])}")
    
    print("\nOLD ACTION DISTRIBUTION:")
    old_labels = {0: "straight", 1: "left", 2: "right"}
    for action, count in stats['old_actions'].items():
        percentage = (count / stats['total_files']) * 100 if stats['total_files'] > 0 else 0
        print(f"  {old_labels[action]}: {count} files ({percentage:.1f}%)")
    
    print("\nNEW ACTION DISTRIBUTION:")
    new_labels = {0: "roadside", 1: "straight", 2: "left", 3: "right"}
    for action, count in stats['new_actions'].items():
        percentage = (count / stats['total_files']) * 100 if stats['total_files'] > 0 else 0
        print(f"  {new_labels[action]}: {count} files ({percentage:.1f}%)")
    
    print("\nCONVERSION MAPPING:")
    print("  straight (0) -> roadside (0)")
    print("  left (1) -> left (2)")
    print("  right (2) -> right (3)")
    
    if stats['errors']:
        print(f"\nERRORS ({len(stats['errors'])}):")
        for error in stats['errors']:
            print(f"  {error}")
    
    if not dry_run:
        print(f"\nOutput directory: {output_path}")
        
        # 変換後の検証
        print("\nValidating converted files...")
        validate_converted_dataset(output_path)
    
    return stats


def validate_converted_dataset(dataset_dir):
    """
    変換後のデータセットの整合性を検証
    
    Args:
        dataset_dir (str): 検証対象のデータセットディレクトリ
    """
    dataset_path = Path(dataset_dir)
    action_dir = dataset_path / "action"
    images_dir = dataset_path / "images"
    angle_dir = dataset_path / "angle"
    
    if not action_dir.exists():
        print("  ERROR: Action directory not found")
        return
    
    action_files = list(action_dir.glob("*.csv"))
    image_files = list(images_dir.glob("*.png")) if images_dir.exists() else []
    angle_files = list(angle_dir.glob("*.csv")) if angle_dir.exists() else []
    
    print(f"  Action files: {len(action_files)}")
    print(f"  Image files: {len(image_files)}")
    print(f"  Angle files: {len(angle_files)}")
    
    # ファイル数の整合性チェック
    if len(action_files) != len(image_files) or len(action_files) != len(angle_files):
        print("  WARNING: File count mismatch between action, image, and angle directories")
    
    # 行動値の範囲チェック
    invalid_actions = []
    for action_file in action_files:
        try:
            action = int(np.loadtxt(action_file, delimiter=",", ndmin=1)[0])
            if action not in [0, 1, 2, 3]:
                invalid_actions.append((action_file.name, action))
        except Exception as e:
            invalid_actions.append((action_file.name, f"read_error: {e}"))
    
    if invalid_actions:
        print(f"  ERROR: Found {len(invalid_actions)} files with invalid action values:")
        for filename, action in invalid_actions[:10]:  # 最初の10個のみ表示
            print(f"    {filename}: {action}")
        if len(invalid_actions) > 10:
            print(f"    ... and {len(invalid_actions) - 10} more")
    else:
        print("  ✓ All action values are valid (0-3)")


def main():
    parser = argparse.ArgumentParser(
        description="Convert existing dataset action files to new 4-action format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (preview only):
  python convert_action_format.py /path/to/dataset --dry-run
  
  # Convert in-place with backup:
  python convert_action_format.py /path/to/dataset --backup
  
  # Convert to new directory:
  python convert_action_format.py /path/to/dataset -o /path/to/new_dataset
  
  # Convert without backup (dangerous):
  python convert_action_format.py /path/to/dataset --no-backup
        """
    )
    
    parser.add_argument('dataset_dir', type=str, 
                       help='Path to the dataset directory containing action/ subdirectory')
    parser.add_argument('-o', '--output', type=str, default=None,
                       help='Output directory (if not specified, converts in-place)')
    parser.add_argument('--backup', action='store_true', default=True,
                       help='Create backup before conversion (default: True)')
    parser.add_argument('--no-backup', action='store_true',
                       help='Do not create backup (overrides --backup)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Preview conversion without making changes')
    
    args = parser.parse_args()
    
    # バックアップ設定の処理
    backup = args.backup and not args.no_backup
    
    if not os.path.exists(args.dataset_dir):
        print(f"Error: Dataset directory not found: {args.dataset_dir}")
        return 1
    
    try:
        stats = convert_dataset(
            dataset_dir=args.dataset_dir,
            output_dir=args.output,
            backup=backup,
            dry_run=args.dry_run
        )
        
        if args.dry_run:
            print("\nTo perform the actual conversion, run without --dry-run flag")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())