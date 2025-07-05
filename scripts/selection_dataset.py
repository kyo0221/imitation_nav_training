#!/usr/bin/env python3
"""
データセットから指定範囲のデータを抽出して新しいデータセットを作成するスクリプト

指定した範囲（start~end）のデータを抽出し、ファイル名を1から連番で振り直した
新しいデータセットを作成します。

例：
- 元データセット: 00001.png ~ 01000.png
- 指定範囲: 50~150
- 抽出結果: 00001.png ~ 00101.png (101個のファイル)
  - 元の00050.png -> 新の00001.png
  - 元の00051.png -> 新の00002.png
  - ...
  - 元の00150.png -> 新の00101.png
"""

import os
import argparse
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime
import cv2


def extract_dataset(source_dir, output_dir, start_idx, end_idx, dry_run=False):
    """
    指定範囲のデータセットを抽出
    
    Args:
        source_dir (str): 元データセットのディレクトリ
        output_dir (str): 出力先ディレクトリ
        start_idx (int): 開始インデックス（1-based）
        end_idx (int): 終了インデックス（1-based、inclusive）
        dry_run (bool): 実際のコピーを行わず、統計のみ表示
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # 必要なディレクトリの存在確認
    images_dir = source_path / "images"
    angle_dir = source_path / "angle"
    action_dir = source_path / "action"
    
    missing_dirs = []
    if not images_dir.exists():
        missing_dirs.append("images")
    if not angle_dir.exists():
        missing_dirs.append("angle")
    if not action_dir.exists():
        missing_dirs.append("action")
    
    if missing_dirs:
        raise FileNotFoundError(f"Missing directories in source: {', '.join(missing_dirs)}")
    
    # 出力ディレクトリの作成
    if not dry_run:
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "images").mkdir(exist_ok=True)
        (output_path / "angle").mkdir(exist_ok=True)
        (output_path / "action").mkdir(exist_ok=True)
    
    # 抽出統計の初期化
    stats = {
        'total_requested': end_idx - start_idx + 1,
        'found_files': 0,
        'missing_files': [],
        'copied_files': 0,
        'errors': [],
        'action_distribution': {0: 0, 1: 0, 2: 0, 3: 0}  # roadside, straight, left, right
    }
    
    print(f"Extracting data from index {start_idx} to {end_idx}")
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Expected files: {stats['total_requested']}")
    
    if dry_run:
        print("DRY RUN MODE - No files will be copied")
    
    # ファイルの抽出処理
    new_idx = 1
    for original_idx in range(start_idx, end_idx + 1):
        # 元ファイル名（5桁ゼロパディング）
        original_name = f"{original_idx:05d}"
        
        # 元ファイルパス
        original_image = images_dir / f"{original_name}.png"
        original_angle = angle_dir / f"{original_name}.csv"
        original_action = action_dir / f"{original_name}.csv"
        
        # 新ファイル名（5桁ゼロパディング）
        new_name = f"{new_idx:05d}"
        
        # 新ファイルパス
        new_image = output_path / "images" / f"{new_name}.png"
        new_angle = output_path / "angle" / f"{new_name}.csv"
        new_action = output_path / "action" / f"{new_name}.csv"
        
        # ファイル存在確認
        missing_files = []
        if not original_image.exists():
            missing_files.append("image")
        if not original_angle.exists():
            missing_files.append("angle")
        if not original_action.exists():
            missing_files.append("action")
        
        if missing_files:
            stats['missing_files'].append({
                'index': original_idx,
                'missing': missing_files
            })
            continue
        
        stats['found_files'] += 1
        
        try:
            # action値の統計取得
            if original_action.exists():
                action_value = int(np.loadtxt(original_action, delimiter=",", ndmin=1)[0])
                if action_value in stats['action_distribution']:
                    stats['action_distribution'][action_value] += 1
            
            # ファイルコピー（dry_runでない場合のみ）
            if not dry_run:
                # 画像ファイルのコピー
                shutil.copy2(original_image, new_image)
                
                # angleファイルのコピー
                shutil.copy2(original_angle, new_angle)
                
                # actionファイルのコピー
                shutil.copy2(original_action, new_action)
                
                stats['copied_files'] += 1
            
            # 進行状況の表示（100ファイルごと）
            if new_idx % 100 == 0:
                print(f"Processed: {new_idx}/{stats['total_requested']} files")
            
            new_idx += 1
            
        except Exception as e:
            error_msg = f"Error copying index {original_idx}: {e}"
            stats['errors'].append(error_msg)
            print(f"ERROR: {error_msg}")
    
    # 結果の表示
    print("\n" + "="*60)
    print("EXTRACTION SUMMARY")
    print("="*60)
    
    print(f"Requested range: {start_idx} to {end_idx} ({stats['total_requested']} files)")
    print(f"Found complete sets: {stats['found_files']}")
    print(f"Missing file sets: {len(stats['missing_files'])}")
    
    if not dry_run:
        print(f"Successfully copied: {stats['copied_files']}")
    
    print(f"Errors encountered: {len(stats['errors'])}")
    
    # 行動分布の表示
    if stats['found_files'] > 0:
        print("\nACTION DISTRIBUTION IN EXTRACTED DATA:")
        action_labels = {0: "roadside", 1: "straight", 2: "left", 3: "right"}
        for action, count in stats['action_distribution'].items():
            percentage = (count / stats['found_files']) * 100 if stats['found_files'] > 0 else 0
            print(f"  {action_labels[action]}: {count} files ({percentage:.1f}%)")
    
    # 不足ファイルの詳細表示
    if stats['missing_files']:
        print(f"\nMISSING FILES ({len(stats['missing_files'])}):")
        for i, missing_info in enumerate(stats['missing_files']):
            if i < 10:  # 最初の10個のみ表示
                print(f"  Index {missing_info['index']:05d}: missing {', '.join(missing_info['missing'])}")
            elif i == 10:
                print(f"  ... and {len(stats['missing_files']) - 10} more missing file sets")
                break
    
    # エラーの詳細表示
    if stats['errors']:
        print(f"\nERRORS ({len(stats['errors'])}):")
        for error in stats['errors'][:10]:  # 最初の10個のみ表示
            print(f"  {error}")
        if len(stats['errors']) > 10:
            print(f"  ... and {len(stats['errors']) - 10} more errors")
    
    if not dry_run and stats['copied_files'] > 0:
        print(f"\nOutput directory: {output_path}")
        
        # 抽出後の検証
        print("\nValidating extracted dataset...")
        validate_extracted_dataset(output_path)
    
    return stats


def validate_extracted_dataset(dataset_dir):
    """
    抽出後のデータセットの整合性を検証
    
    Args:
        dataset_dir (str): 検証対象のデータセットディレクトリ
    """
    dataset_path = Path(dataset_dir)
    
    images_dir = dataset_path / "images"
    angle_dir = dataset_path / "angle"
    action_dir = dataset_path / "action"
    
    # ファイル数の確認
    image_files = sorted(images_dir.glob("*.png"))
    angle_files = sorted(angle_dir.glob("*.csv"))
    action_files = sorted(action_dir.glob("*.csv"))
    
    print(f"  Image files: {len(image_files)}")
    print(f"  Angle files: {len(angle_files)}")
    print(f"  Action files: {len(action_files)}")
    
    # ファイル数の整合性
    if len(image_files) == len(angle_files) == len(action_files):
        print("  ✓ File counts match across all directories")
    else:
        print("  ✗ File count mismatch detected")
    
    # 連番の確認
    expected_count = len(image_files)
    if expected_count > 0:
        missing_indices = []
        for i in range(1, expected_count + 1):
            expected_name = f"{i:05d}"
            
            image_file = images_dir / f"{expected_name}.png"
            angle_file = angle_dir / f"{expected_name}.csv"
            action_file = action_dir / f"{expected_name}.csv"
            
            if not (image_file.exists() and angle_file.exists() and action_file.exists()):
                missing_indices.append(i)
        
        if missing_indices:
            print(f"  ✗ Missing indices: {missing_indices[:10]}")  # 最初の10個のみ表示
            if len(missing_indices) > 10:
                print(f"      ... and {len(missing_indices) - 10} more")
        else:
            print(f"  ✓ Sequential numbering is correct (1 to {expected_count})")
    
    # 画像ファイルの簡単な検証
    corrupted_images = []
    for image_file in image_files[:10]:  # 最初の10個のみ検証
        try:
            img = cv2.imread(str(image_file))
            if img is None:
                corrupted_images.append(image_file.name)
        except Exception:
            corrupted_images.append(image_file.name)
    
    if corrupted_images:
        print(f"  ✗ Potentially corrupted images: {corrupted_images}")
    else:
        print("  ✓ Sample images are readable")


def main():
    parser = argparse.ArgumentParser(
        description="Extract a range of data from dataset and renumber from 1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract data from index 50 to 150 (preview only):
  python selection_dataset.py /path/to/source_dataset /path/to/output_dataset --start 50 --end 150 --dry-run
  
  # Extract data from index 50 to 150:
  python selection_dataset.py /path/to/source_dataset /path/to/output_dataset --start 50 --end 150
  
  # Extract first 100 samples:
  python selection_dataset.py /path/to/source_dataset /path/to/output_dataset --start 1 --end 100
  
  # Extract specific range with verbose output:
  python selection_dataset.py ../saituyo_dataset ../extracted_dataset --start 200 --end 300
        """
    )
    
    parser.add_argument('source_dir', type=str,
                       help='Path to source dataset directory')
    parser.add_argument('output_dir', type=str,
                       help='Path to output dataset directory')
    parser.add_argument('--start', type=int, required=True,
                       help='Start index (1-based, inclusive)')
    parser.add_argument('--end', type=int, required=True,
                       help='End index (1-based, inclusive)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Preview extraction without copying files')
    parser.add_argument('--force', action='store_true',
                       help='Overwrite output directory if it exists')
    
    args = parser.parse_args()
    
    # 入力検証
    if not os.path.exists(args.source_dir):
        print(f"Error: Source directory not found: {args.source_dir}")
        return 1
    
    if args.start < 1:
        print("Error: Start index must be >= 1")
        return 1
    
    if args.end < args.start:
        print("Error: End index must be >= start index")
        return 1
    
    # 出力ディレクトリの存在確認
    if os.path.exists(args.output_dir) and not args.force and not args.dry_run:
        response = input(f"Output directory {args.output_dir} already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Operation cancelled")
            return 1
    
    try:
        stats = extract_dataset(
            source_dir=args.source_dir,
            output_dir=args.output_dir,
            start_idx=args.start,
            end_idx=args.end,
            dry_run=args.dry_run
        )
        
        if args.dry_run:
            print("\nTo perform the actual extraction, run without --dry-run flag")
        elif stats['copied_files'] > 0:
            print(f"\n✓ Successfully extracted {stats['copied_files']} data samples")
        else:
            print("\n✗ No files were extracted")
            return 1
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())