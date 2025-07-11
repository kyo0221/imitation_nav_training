#!/usr/bin/env python3

import os
import sys
import re
import numpy as np
from PIL import Image
import argparse

class ImageRBSwapper:
    def __init__(self, input_dir: str, output_dir: str = None):
        self.input_dir = input_dir
        self.output_dir = output_dir if output_dir else os.path.join(input_dir, 'rb_swapped')
        
        # 出力ディレクトリが存在しない場合は作成（overrideモードでない場合）
        if output_dir or not hasattr(self, 'override_mode'):
            os.makedirs(self.output_dir, exist_ok=True)
    
    def _is_target_image_file(self, filename: str) -> bool:
        """XXXXX.png形式のファイルかどうかをチェック"""
        pattern1 = r'^\d{5}\.png$'
        pattern2 = r'^img\d{5}\.png$'
        return bool(re.match(pattern1, filename.lower()) or re.match(pattern2, filename.lower()))
    
    def _get_image_files(self) -> list:
        """入力ディレクトリからXXXXX.png形式のファイルのリストを取得"""
        image_files = []
        for filename in os.listdir(self.input_dir):
            if self._is_target_image_file(filename):
                image_files.append(filename)
        return sorted(image_files)
    
    def _swap_rb_channels(self, input_path: str, output_path: str) -> bool:
        """画像のRとBチャンネルを反転して保存"""
        try:
            # 画像を読み込み
            with Image.open(input_path) as img:
                # RGBモードに変換（必要に応じて）
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # NumPy配列に変換
                img_array = np.array(img)
                
                # RとBチャンネルを入れ替え
                # RGB -> BGR
                swapped_array = img_array.copy()
                swapped_array[:, :, 0] = img_array[:, :, 2]  # R <- B
                swapped_array[:, :, 2] = img_array[:, :, 0]  # B <- R
                # Gチャンネル（インデックス1）はそのまま
                
                # PIL Imageに戻す
                swapped_img = Image.fromarray(swapped_array, 'RGB')
                
                # PNG形式で保存
                swapped_img.save(output_path, 'PNG')
                
                return True
                
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
            return False
    
    def process_images(self, override_mode: bool = False) -> tuple:
        """すべての画像を処理"""
        self.override_mode = override_mode
        image_files = self._get_image_files()
        
        if not image_files:
            print(f"No XXXXX.png files found in {self.input_dir}")
            return 0, 0
        
        print(f"Found {len(image_files)} image(s) in {self.input_dir}")
        print(f"Target format: XXXXX.png")
        print(f"Operation: Swapping R and B channels (RGB -> BGR)")
        
        if override_mode:
            print("Mode: Override (replacing original files)")
        else:
            print(f"Output directory: {self.output_dir}")
        print()
        
        success_count = 0
        skip_count = 0
        
        for i, filename in enumerate(image_files, 1):
            input_path = os.path.join(self.input_dir, filename)
            
            if override_mode:
                # 元ファイルを上書き
                output_path = input_path
            else:
                # 別ディレクトリに保存
                output_path = os.path.join(self.output_dir, filename)
                
                # ファイルが既に存在する場合はスキップ
                if os.path.exists(output_path):
                    print(f"[{i:3d}/{len(image_files)}] Skipping {filename} (already exists)")
                    skip_count += 1
                    continue
            
            print(f"[{i:3d}/{len(image_files)}] Processing {filename}...", end=' ')
            
            if self._swap_rb_channels(input_path, output_path):
                print("✓")
                success_count += 1
            else:
                print("✗")
        
        return success_count, skip_count
    
    def show_summary(self, success_count: int, skip_count: int, total_count: int, override_mode: bool):
        """処理結果のサマリーを表示"""
        print()
        print("=" * 50)
        print("PROCESSING SUMMARY")
        print("=" * 50)
        print(f"Total files found:    {total_count}")
        print(f"Successfully processed: {success_count}")
        
        if not override_mode:
            print(f"Skipped:              {skip_count}")
            
        failed_count = total_count - success_count - skip_count
        if failed_count > 0:
            print(f"Failed:               {failed_count}")
            
        if override_mode:
            print(f"Mode:                 Override (original files replaced)")
        else:
            print(f"Output directory:     {self.output_dir}")
        print("=" * 50)

def main():
    parser = argparse.ArgumentParser(
        description="Swap R and B channels in XXXXX.png images (RGB -> BGR)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 image_rb_swap.py images/
  python3 image_rb_swap.py images/ --output swapped_images/
  python3 image_rb_swap.py images/ --override
        """
    )
    
    parser.add_argument(
        'input_dir',
        help='Input directory containing XXXXX.png images'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output directory (default: input_dir/rb_swapped)',
        default=None
    )
    
    parser.add_argument(
        '--override',
        action='store_true',
        help='Override original files (replace with RB-swapped versions)'
    )
    
    args = parser.parse_args()
    
    # 入力ディレクトリの存在チェック
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        sys.exit(1)
    
    if not os.path.isdir(args.input_dir):
        print(f"Error: '{args.input_dir}' is not a directory")
        sys.exit(1)
    
    # overrideモードでoutputが指定されている場合は警告
    if args.override and args.output:
        print("Warning: --output option is ignored when --override is used")
    
    try:
        # ImageRBSwapperを初期化
        swapper = ImageRBSwapper(
            input_dir=args.input_dir,
            output_dir=args.output
        )
        
        # 画像を処理
        success_count, skip_count = swapper.process_images(override_mode=args.override)
        
        # 結果を表示
        total_files = len([f for f in os.listdir(args.input_dir) 
                          if swapper._is_target_image_file(f)])
        swapper.show_summary(success_count, skip_count, total_files, args.override)
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()