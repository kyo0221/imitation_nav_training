import os
import torch
import random
import numpy as np
import cv2
import albumentations as A


def create_albumentations_augmented_dataset(base_dataset, num_augmented_samples=1,
                                            brightness_limit=0.2, contrast_limit=0.2,
                                            saturation_limit=0.2, hue_limit=0.1,
                                            blur_limit=3, h_flip_prob=0.5,
                                            visualize=False, visualize_dir=None):
    """
    WebDataset対応のAlbumentations拡張データセット作成
    
    Args:
        base_dataset: ベースとなるWebDataset
        num_augmented_samples: 1つのサンプルから生成する拡張サンプル数
        brightness_limit: 明度変更の範囲 (±brightness_limit)
        contrast_limit: コントラスト変更の範囲 (±contrast_limit)  
        saturation_limit: 彩度変更の範囲 (±saturation_limit)
        hue_limit: 色相変更の範囲 (±hue_limit)
        blur_limit: ガウシアンブラーのカーネルサイズ上限
        h_flip_prob: 水平反転の確率
        visualize: 拡張画像の可視化フラグ
        visualize_dir: 可視化画像の保存先ディレクトリ
    """
    if visualize and visualize_dir:
        os.makedirs(visualize_dir, exist_ok=True)

    # WebDatasetのcompose()機能を活用
    return base_dataset.compose(lambda source: _apply_albumentations_augmentation(
        source, num_augmented_samples, brightness_limit, contrast_limit,
        saturation_limit, hue_limit, blur_limit, h_flip_prob,
        visualize, visualize_dir
    ))


def _apply_albumentations_augmentation(source, num_augmented_samples,
                                       brightness_limit, contrast_limit,
                                       saturation_limit, hue_limit,
                                       blur_limit, h_flip_prob,
                                       visualize, visualize_dir):
    """Albumentations拡張処理のコア実装"""
    visualized_count = 0
    visualize_limit = 100
    
    # Albumentationsトランスフォーム定義
    transform = A.Compose([
        A.ColorJitter(
            brightness=brightness_limit,
            contrast=contrast_limit, 
            saturation=saturation_limit,
            hue=hue_limit,
            p=0.8
        ),
        # ガウシアンブラー
        A.GaussianBlur(
            blur_limit=(3, blur_limit),
            p=0.3
        ),
        # ランダムノイズ
        A.GaussNoise(
            var_limit=(10, 50),
            mean=0,
            p=0.2
        ),
        # シャープネス調整
        A.Sharpen(
            alpha=(0.1, 0.3),
            lightness=(0.8, 1.2),
            p=0.2
        ),
        # 軽微な回転（車両制御に影響しない範囲）
        A.Rotate(
            limit=3,
            p=0.3,
            border_mode=cv2.BORDER_CONSTANT,
            value=0
        ),
    ])

    for image, action_onehot, angle in source:
        # 元のサンプルを出力
        yield image, action_onehot, angle
        
        # 拡張サンプルを生成
        for aug_idx in range(num_augmented_samples):
            # PyTorchテンソルからnumpy配列に変換 (C, H, W) -> (H, W, C)
            img_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            
            augmented_angle = angle
            
            # 水平反転を個別に処理（角度調整のため）
            if random.random() < h_flip_prob:
                img_np = cv2.flip(img_np, 1)  # 水平反転
                augmented_angle = -angle  # 角度を反転
            
            # Albumentationsで他の拡張を適用
            augmented = transform(image=img_np)
            aug_img = augmented['image']
            
            # 可視化用画像保存
            if visualize and visualize_dir and visualized_count < visualize_limit:
                save_path = os.path.join(
                    visualize_dir, 
                    f"{visualized_count:05d}_aug{aug_idx}_albu.png"
                )
                cv2.imwrite(save_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
                visualized_count += 1
            
            # numpy配列からPyTorchテンソルに変換
            with torch.no_grad():
                augmented_image = torch.tensor(aug_img, dtype=torch.float32).permute(2, 0, 1) / 255.0
            
            yield augmented_image, action_onehot, augmented_angle