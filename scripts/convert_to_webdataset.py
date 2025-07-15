#!/usr/bin/env python3
"""
既存の個別ファイル形式のデータセットをWebDataset形式に変換するスクリプト
"""

import os
import argparse
import json
import cv2
import numpy as np
import webdataset as wds
from tqdm import tqdm
import glob


def convert_to_webdataset(input_dir, output_dir, samples_per_shard=1000, enable_compression=True):
    """
    個別ファイル形式のデータセットをWebDataset形式に変換（numpy形式固定）
    
    Args:
        input_dir: 入力ディレクトリ（images/, angle/, action/が含まれる）
        output_dir: 出力ディレクトリ
        samples_per_shard: 1シャードあたりのサンプル数
        enable_compression: 圧縮を有効にするかどうか
    """
    
    # 入力ディレクトリの確認
    images_dir = os.path.join(input_dir, 'images')
    angle_dir = os.path.join(input_dir, 'angle')
    action_dir = os.path.join(input_dir, 'action')
    
    if not all(os.path.exists(d) for d in [images_dir, angle_dir, action_dir]):
        raise ValueError(f"Required directories not found in {input_dir}")
    
    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 画像ファイルのリストを取得
    image_files = sorted(glob.glob(os.path.join(images_dir, "*.png")))
    
    if not image_files:
        raise ValueError(f"No image files found in {images_dir}")
    
    print(f"Found {len(image_files)} samples to convert")
    
    # WebDatasetの作成
    current_shard = 0
    sample_count = 0
    shard_writer = None
    
    def create_new_shard():
        nonlocal shard_writer, current_shard
        if shard_writer is not None:
            shard_writer.close()
        
        if enable_compression:
            shard_pattern = os.path.join(output_dir, "shard_%06d.tar.gz")
        else:
            shard_pattern = os.path.join(output_dir, "shard_%06d.tar")
        
        shard_writer = wds.ShardWriter(shard_pattern, maxcount=samples_per_shard)
        actual_filename = shard_pattern % current_shard
        print(f"Creating shard: {actual_filename}")
        current_shard += 1
        return shard_writer
    
    # 最初のシャードを作成
    shard_writer = create_new_shard()
    
    try:
        for image_file in tqdm(image_files, desc="Converting"):
            # ファイル名からベース名を取得
            base_name = os.path.splitext(os.path.basename(image_file))[0]
            
            # 対応するangleとactionファイルのパスを構築
            angle_file = os.path.join(angle_dir, f"{base_name}.csv")
            action_file = os.path.join(action_dir, f"{base_name}.csv")
            
            # ファイルの存在確認
            if not os.path.exists(angle_file) or not os.path.exists(action_file):
                print(f"Warning: Missing files for {base_name}, skipping...")
                continue
            
            # データの読み込み
            try:
                # 画像データ
                img = cv2.imread(image_file)
                if img is None:
                    print(f"Warning: Could not read image {image_file}, skipping...")
                    continue
                
                # numpy形式で保存（バイト列として）
                img_data = img.tobytes()
                img_ext = "npy"
                
                # 角度データ
                angle = float(np.loadtxt(angle_file, delimiter=",", ndmin=1))
                
                # アクションデータ
                action = int(np.loadtxt(action_file, delimiter=",", ndmin=1))
                
                # メタデータの作成
                metadata = {
                    'angle': float(angle),
                    'action': int(action),
                    'original_file': base_name,
                    'image_width': img.shape[1],
                    'image_height': img.shape[0],
                    'save_format': 'numpy',
                    'image_shape': list(img.shape),
                    'image_dtype': str(img.dtype)
                }
                
                # WebDatasetサンプルの作成
                sample_key = f"{sample_count:06d}"
                sample = {
                    "__key__": sample_key,
                    img_ext: img_data,
                    "angle.json": json.dumps(metadata),
                    "action.json": json.dumps({"action": int(action)})
                }
                
                # シャードに書き込み
                shard_writer.write(sample)
                sample_count += 1
                
                # 必要に応じて新しいシャードを作成
                if sample_count % samples_per_shard == 0:
                    shard_writer = create_new_shard()
                    
            except Exception as e:
                print(f"Error processing {base_name}: {e}")
                continue
    
    finally:
        # 最後のシャードを閉じる
        if shard_writer is not None:
            shard_writer.close()
    
    print(f"Conversion completed:")
    print(f"  - Total samples: {sample_count}")
    print(f"  - Total shards: {current_shard}")
    print(f"  - Output directory: {output_dir}")
    
    # 統計情報ファイルの作成
    stats = {
        "total_samples": sample_count,
        "total_shards": current_shard,
        "samples_per_shard": samples_per_shard,
        "compression_enabled": enable_compression,
        "save_format": "numpy",
        "input_directory": input_dir,
        "output_directory": output_dir
    }
    
    stats_file = os.path.join(output_dir, "dataset_stats.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Statistics saved to: {stats_file}")


def main():
    parser = argparse.ArgumentParser(description="Convert dataset to WebDataset format")
    parser.add_argument("input_dir", help="Input directory containing images/, angle/, action/")
    parser.add_argument("output_dir", help="Output directory for WebDataset shards")
    parser.add_argument("--samples_per_shard", type=int, default=1000, 
                       help="Number of samples per shard (default: 1000)")
    parser.add_argument("--no_compression", action="store_true",
                       help="Disable compression")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        return 1
    
    enable_compression = not args.no_compression
    
    try:
        convert_to_webdataset(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            samples_per_shard=args.samples_per_shard,
            enable_compression=enable_compression
        )
        print("Conversion successful!")
        return 0
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        return 1


if __name__ == "__main__":
    exit(main())