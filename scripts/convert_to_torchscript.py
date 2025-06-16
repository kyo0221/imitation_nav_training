#!/usr/bin/env python3
"""
時系列モデルをTorchScriptに変換するスクリプト
"""

import torch
import argparse
import os
import sys
import yaml
from pathlib import Path

# プロジェクトのルートパスを追加
sys.path.append(str(Path(__file__).parent))

from models.temporal_model import TemporalConditionalAnglePredictor

def convert_temporal_model_to_torchscript(model_path, output_path, config_path):
    """
    時系列モデルをTorchScriptに変換
    
    Args:
        model_path (str): PyTorchモデルファイルのパス
        output_path (str): 出力TorchScriptファイルのパス
        config_path (str): 設定ファイルのパス
    """
    
    # 設定ファイル読み込み
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # モデルパラメータ
    params = config['model']
    temporal_params = config.get('temporal', {})
    
    # モデル初期化
    model = TemporalConditionalAnglePredictor(
        n_channel=params['n_channel'],
        n_out=params['n_out'],
        input_height=params['image_height'],
        input_width=params['image_width'],
        n_action_classes=params['n_action_classes'],
        sequence_length=temporal_params.get('sequence_length', 10),
        hidden_size=temporal_params.get('hidden_size', 256),
        num_layers=temporal_params.get('num_layers', 2),
        prediction_horizon=temporal_params.get('prediction_horizon', 3),
        rnn_type=temporal_params.get('rnn_type', 'LSTM')
    )
    
    # 学習済み重みをロード
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Model loaded from: {model_path}")
    else:
        print(f"Warning: Model file not found at {model_path}. Using random weights.")
    
    model.eval()
    
    # サンプル入力を作成
    sequence_length = temporal_params.get('sequence_length', 10)
    batch_size = 1
    
    # 画像シーケンス: [batch, seq_len, channels, height, width]
    sample_images = torch.randn(
        batch_size, 
        sequence_length, 
        params['n_channel'], 
        params['image_height'], 
        params['image_width']
    )
    
    # コマンドシーケンス: [batch, seq_len, n_action_classes]
    sample_commands = torch.randn(
        batch_size, 
        sequence_length, 
        params['n_action_classes']
    )
    
    try:
        # TorchScriptに変換
        traced_model = torch.jit.trace(model, (sample_images, sample_commands))
        
        # 保存
        traced_model.save(output_path)
        print(f"TorchScript model saved to: {output_path}")
        
        # 変換後のモデルをテスト
        with torch.no_grad():
            original_output = model(sample_images, sample_commands)
            traced_output = traced_model(sample_images, sample_commands)
            
            # 出力の差を確認
            if isinstance(original_output, tuple):
                diff = torch.abs(original_output[0] - traced_output[0]).max()
            else:
                diff = torch.abs(original_output - traced_output).max()
            
            print(f"Max difference between original and traced model: {diff.item()}")
            
            if diff < 1e-5:
                print("✓ Conversion successful - outputs match")
            else:
                print("⚠ Warning: Outputs differ significantly")
        
    except Exception as e:
        print(f"Error during TorchScript conversion: {e}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Convert temporal model to TorchScript')
    parser.add_argument('model_path', help='Path to the PyTorch model file')
    parser.add_argument('--output', '-o', help='Output TorchScript file path', 
                       default='temporal_model.pt')
    parser.add_argument('--config', '-c', help='Config file path',
                       default='../config/train_params.yaml')
    
    args = parser.parse_args()
    
    # パスの解決
    model_path = Path(args.model_path).resolve()
    output_path = Path(args.output).resolve()
    config_path = Path(args.config).resolve()
    
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return 1
    
    print(f"Converting model: {model_path}")
    print(f"Output path: {output_path}")
    print(f"Config path: {config_path}")
    
    success = convert_temporal_model_to_torchscript(
        str(model_path), 
        str(output_path), 
        str(config_path)
    )
    
    if success:
        print("Conversion completed successfully!")
        return 0
    else:
        print("Conversion failed!")
        return 1

if __name__ == '__main__':
    exit(main())