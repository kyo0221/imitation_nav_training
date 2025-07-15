#!/usr/bin/env python3
import os
import argparse
import numpy as np
import shutil
from pathlib import Path
import csv
from tqdm import tqdm


def convert_dataset_to_timeseries(input_dir, output_dir, future_steps=20):
    """
    従来のデータセットを時系列考慮ネットワーク用に変換
    
    Args:
        input_dir: 入力データセットディレクトリ（images/, angle/, action/を含む）
        output_dir: 出力データセットディレクトリ
        future_steps: 未来のステップ数（デフォルト20）
    """
    
    # 入力ディレクトリの確認
    input_images_dir = os.path.join(input_dir, "images")
    input_angle_dir = os.path.join(input_dir, "angle")
    input_action_dir = os.path.join(input_dir, "action")
    
    if not all(os.path.exists(d) for d in [input_images_dir, input_angle_dir, input_action_dir]):
        raise ValueError(f"入力ディレクトリに必要なフォルダが存在しません: {input_dir}")
    
    # 出力ディレクトリの作成
    output_images_dir = os.path.join(output_dir, "images")
    output_angle_dir = os.path.join(output_dir, "angle")
    output_action_dir = os.path.join(output_dir, "action")
    
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_angle_dir, exist_ok=True)
    os.makedirs(output_action_dir, exist_ok=True)
    
    # 入力ファイルリストの取得
    image_files = sorted([f for f in os.listdir(input_images_dir) if f.endswith(".png")])
    print(f"入力画像ファイル数: {len(image_files)}")
    
    # 時系列データの作成
    converted_count = 0
    
    for i, image_file in enumerate(tqdm(image_files, desc="データセット変換中")):
        # 未来のstep分のデータが存在するかチェック
        if i + future_steps >= len(image_files):
            break
            
        base_name = image_file[:-4]  # .png を除去
        
        # 現在のフレームの画像とアクションをコピー
        input_image_path = os.path.join(input_images_dir, image_file)
        input_action_path = os.path.join(input_action_dir, base_name + ".csv")
        
        if not os.path.exists(input_action_path):
            continue
            
        output_image_path = os.path.join(output_images_dir, image_file)
        output_action_path = os.path.join(output_action_dir, base_name + ".csv")
        
        # 画像とアクションをコピー
        shutil.copy2(input_image_path, output_image_path)
        shutil.copy2(input_action_path, output_action_path)
        
        # 未来のstep分のangleを収集
        future_angles = []
        valid_sequence = True
        
        for j in range(future_steps):
            future_image_file = image_files[i + j]
            future_base_name = future_image_file[:-4]
            future_angle_path = os.path.join(input_angle_dir, future_base_name + ".csv")
            
            if not os.path.exists(future_angle_path):
                valid_sequence = False
                break
                
            try:
                angle = float(np.loadtxt(future_angle_path, delimiter=",", ndmin=1))
                future_angles.append(angle)
            except Exception as e:
                print(f"角度読み込みエラー: {future_angle_path}, {e}")
                valid_sequence = False
                break
        
        # 有効なシーケンスの場合のみ出力
        if valid_sequence and len(future_angles) == future_steps:
            output_angle_path = os.path.join(output_angle_dir, base_name + ".csv")
            
            # 未来の角度を1つのCSVファイルに保存
            with open(output_angle_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(future_angles)
            
            converted_count += 1
    
    print(f"変換完了: {converted_count}個のサンプルを出力しました")
    print(f"出力ディレクトリ: {output_dir}")
    
    # 変換結果の統計情報を保存
    info_path = os.path.join(output_dir, "conversion_info.txt")
    with open(info_path, 'w') as f:
        f.write(f"変換日時: {np.datetime64('now')}\n")
        f.write(f"入力ディレクトリ: {input_dir}\n")
        f.write(f"出力ディレクトリ: {output_dir}\n")
        f.write(f"未来ステップ数: {future_steps}\n")
        f.write(f"入力サンプル数: {len(image_files)}\n")
        f.write(f"出力サンプル数: {converted_count}\n")
        f.write(f"変換率: {converted_count/len(image_files)*100:.1f}%\n")


def validate_converted_dataset(dataset_dir, future_steps=20):
    """
    変換後のデータセットの妥当性チェック
    
    Args:
        dataset_dir: 検証するデータセットディレクトリ
        future_steps: 期待される未来ステップ数
    """
    
    images_dir = os.path.join(dataset_dir, "images")
    angle_dir = os.path.join(dataset_dir, "angle")
    action_dir = os.path.join(dataset_dir, "action")
    
    if not all(os.path.exists(d) for d in [images_dir, angle_dir, action_dir]):
        print("❌ 必要なディレクトリが存在しません")
        return False
    
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(".png")])
    
    print(f"📊 データセット検証中...")
    print(f"サンプル数: {len(image_files)}")
    
    # サンプルのangleファイルをチェック
    sample_count = min(10, len(image_files))
    valid_count = 0
    
    for i in range(sample_count):
        image_file = image_files[i]
        base_name = image_file[:-4]
        angle_path = os.path.join(angle_dir, base_name + ".csv")
        
        if os.path.exists(angle_path):
            try:
                angles = np.loadtxt(angle_path, delimiter=",", ndmin=1)
                if len(angles) == future_steps:
                    valid_count += 1
                else:
                    print(f"⚠️  {angle_path}: 期待されるステップ数と異なります (期待: {future_steps}, 実際: {len(angles)})")
            except Exception as e:
                print(f"❌ {angle_path}: 読み込みエラー - {e}")
    
    success_rate = valid_count / sample_count * 100
    print(f"✅ 検証完了: {valid_count}/{sample_count} サンプルが有効 ({success_rate:.1f}%)")
    
    return success_rate > 90


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="従来データセットを時系列考慮ネットワーク用に変換")
    parser.add_argument("input_dir", type=str, help="入力データセットディレクトリ")
    parser.add_argument("output_dir", type=str, help="出力データセットディレクトリ")
    parser.add_argument("--future_steps", type=int, default=20, help="未来のステップ数（デフォルト: 20）")
    parser.add_argument("--validate", action="store_true", help="変換後にデータセットの妥当性チェックを実行")
    
    args = parser.parse_args()
    
    # 入力ディレクトリの存在確認
    if not os.path.exists(args.input_dir):
        print(f"❌ 入力ディレクトリが存在しません: {args.input_dir}")
        exit(1)
    
    # 出力ディレクトリが存在する場合の確認
    if os.path.exists(args.output_dir):
        response = input(f"出力ディレクトリが既に存在します: {args.output_dir}\n上書きしますか? (y/N): ")
        if response.lower() != 'y':
            print("変換を中止しました")
            exit(0)
    
    try:
        print("🚀 データセット変換を開始します...")
        convert_dataset_to_timeseries(args.input_dir, args.output_dir, args.future_steps)
        
        if args.validate:
            print("\n🔍 データセットの妥当性チェックを実行します...")
            is_valid = validate_converted_dataset(args.output_dir, args.future_steps)
            if is_valid:
                print("✅ データセットの変換が正常に完了しました")
            else:
                print("⚠️  データセットの変換に問題がある可能性があります")
        else:
            print("✅ データセットの変換が完了しました")
            print("💡 --validate オプションを使用して変換結果を検証することを推奨します")
            
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        exit(1)
