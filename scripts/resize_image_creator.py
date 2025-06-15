#!/usr/bin/env python3

import os
import sys
import re
from PIL import Image
import argparse

class ImageResizeCreator:
    def __init__(self, input_dir: str, output_dir: str = None, target_size: tuple = (85, 85)):
        self.input_dir = input_dir
        self.output_dir = output_dir if output_dir else os.path.join(input_dir, 'resized')
        self.target_size = target_size
        
        # 出力ディレクトリが存在しない場合は作成（overrideモードでない場合）
        if output_dir or not hasattr(self, 'override_mode'):
            os.makedirs(self.output_dir, exist_ok=True)
    
    def _is_target_image_file(self, filename: str) -> bool:
        """imgXXXXX.png形式のファイルかどうかをチェック"""
        pattern = r'^img\d{5}\.png$'
        return bool(re.match(pattern, filename.lower()))
    
    def _get_image_files(self) -> list:
        """入力ディレクトリからimgXXXXX.png形式のファイルのリストを取得"""
        image_files = []
        for filename in os.listdir(self.input_dir):
            if self._is_target_image_file(filename):
                image_files.append(filename)
        return sorted(image_files)
    
    def _resize_image(self, input_path: str, output_path: str) -> bool:
        """画像をリサイズして保存"""
        try:
            with Image.open(input_path) as img:
                # 既に85x85の場合はスキップ
                if img.size == self.target_size:
                    if input_path != output_path:
                        img.save(output_path, 'PNG')
                    return True
                
                # RGBモードに変換（必要に応じて）
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # リサイズ（高品質なランチョス法を使用）
                # PIL/Pillowのバージョン互換性のため
                try:
                    # Pillow >= 10.0.0
                    resized_img = img.resize(self.target_size, Image.Resampling.LANCZOS)
                except AttributeError:
                    # Pillow < 10.0.0
                    resized_img = img.resize(self.target_size, Image.LANCZOS)
                
                # PNG形式で保存
                resized_img.save(output_path, 'PNG')
                
                return True
                
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
            return False
    
    def process_images(self, override_mode: bool = False) -> tuple:
        """すべての画像を処理"""
        self.override_mode = override_mode
        image_files = self._get_image_files()
        
        if not image_files:
            print(f"No imgXXXXX.png files found in {self.input_dir}")
            return 0, 0
        
        print(f"Found {len(image_files)} image(s) in {self.input_dir}")
        print(f"Target format: imgXXXXX.png")
        print(f"Resizing to {self.target_size[0]}x{self.target_size[1]} pixels")
        
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
            
            if self._resize_image(input_path, output_path):
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
        print(f"Successfully resized: {success_count}")
        
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
        description="Resize imgXXXXX.png images to 85x85 pixels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 image_resize_creator.py images/
  python3 image_resize_creator.py images/ --output resized_images/
  python3 image_resize_creator.py images/ --size 128 128
  python3 image_resize_creator.py images/ --override
        """
    )
    
    parser.add_argument(
        'input_dir',
        help='Input directory containing imgXXXXX.png images'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output directory (default: input_dir/resized)',
        default=None
    )
    
    parser.add_argument(
        '--size', '-s',
        nargs=2,
        type=int,
        help='Target size as width height (default: 85 85)',
        default=[85, 85],
        metavar=('WIDTH', 'HEIGHT')
    )
    
    parser.add_argument(
        '--override',
        action='store_true',
        help='Override original files (replace with resized versions)'
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
        # ImageResizeCreatorを初期化
        resizer = ImageResizeCreator(
            input_dir=args.input_dir,
            output_dir=args.output,
            target_size=tuple(args.size)
        )
        
        # 画像を処理
        success_count, skip_count = resizer.process_images(override_mode=args.override)
        
        # 結果を表示
        total_files = len([f for f in os.listdir(args.input_dir) 
                          if resizer._is_target_image_file(f)])
        resizer.show_summary(success_count, skip_count, total_files, args.override)
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()