import os
import torch
import random
import numpy as np
import cv2
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class AlbumentationsWrapperDataset(Dataset):
    def __init__(self, base_dataset, num_augmented_samples=1, 
                 brightness_limit=0.2, contrast_limit=0.2, saturation_limit=0.2, 
                 hue_limit=0.1, blur_limit=3, h_flip_prob=0.5,
                 visualize=False, visualize_dir=None):
        """
        Albumentationsを使用したデータ拡張Wrapper
        
        Args:
            base_dataset: ベースとなるデータセット
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
        self.base_dataset = base_dataset
        self.num_augmented_samples = num_augmented_samples
        self.h_flip_prob = h_flip_prob
        
        self.transform = A.Compose([
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
        
        self.visualize = visualize
        self.visualize_dir = visualize_dir
        self.visualized_count = 0
        self.visualize_limit = 100
        self.total_augmented = len(base_dataset) * num_augmented_samples
        self.visualize_prob = min(1.0, self.visualize_limit / self.total_augmented)
        
        if self.visualize and self.visualize_dir:
            os.makedirs(self.visualize_dir, exist_ok=True)
            
        self.original_length = len(base_dataset)
    
    def __len__(self):
        return self.original_length * (1 + self.num_augmented_samples)
    
    def __getitem__(self, index):
        base_idx = index // (1 + self.num_augmented_samples)
        is_aug = index % (1 + self.num_augmented_samples) != 0
        
        image, action_onehot, angle = self.base_dataset[base_idx]
        
        if is_aug:
            # PyTorchテンソルからnumpy配列に変換 (C, H, W) -> (H, W, C)
            img_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            
            # 水平反転を個別に処理（角度調整のため）
            flip_applied = False
            if random.random() < self.h_flip_prob:
                img_np = cv2.flip(img_np, 1)  # 水平反転
                flip_applied = True
                angle = -angle  # 角度を反転
            
            # Albumentationsで他の拡張を適用
            augmented = self.transform(image=img_np)
            aug_img = augmented['image']
            
            # 可視化用画像保存
            if self.visualize and self.visualized_count < self.visualize_limit:
                if random.random() < self.visualize_prob:
                    save_path = os.path.join(
                        self.visualize_dir, 
                        f"{base_idx:05d}_aug{index}_albu.png"
                    )
                    # OpenCVはBGR形式なので変換
                    cv2.imwrite(save_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
                    self.visualized_count += 1
            
            # numpy配列からPyTorchテンソルに変換
            with torch.no_grad():
                image = torch.tensor(aug_img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        
        return image, action_onehot, angle