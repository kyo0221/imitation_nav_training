import os

from torch.utils.data import Dataset
import numpy as np
import torch
import random
import cv2
import torchvision.transforms.functional as F


class GammaWrapperDataset(Dataset):
    def __init__(self, base_dataset, gamma_range=(0.9, 1.1), num_augmented_samples=1,
                 contrast_range=(0.8, 1.2), visualize=False, visualize_dir=None):
        self.base_dataset = base_dataset
        self.gamma_range = gamma_range
        self.contrast_range = contrast_range
        self.num_augmented_samples = num_augmented_samples

        self.visualize = visualize
        self.visualize_dir = visualize_dir
        self.visualize_limit = 100
        self.visualized_count = 0
        self.total_augmented = len(base_dataset) * num_augmented_samples
        self.visualize_prob = min(1.0, self.visualize_limit / self.total_augmented)

        if self.visualize and self.visualize_dir:
            os.makedirs(self.visualize_dir, exist_ok=True)

        self.original_length = len(base_dataset)

    def __len__(self):
        return self.original_length * (1 + self.num_augmented_samples)

    def __getitem__(self, index):
        base_idx = index // (1 + self.num_augmented_samples)
        aug_idx = index % (1 + self.num_augmented_samples)
        is_aug = aug_idx != 0

        image, action_onehot, angle = self.base_dataset[base_idx]

        if is_aug:
            gamma = random.uniform(*self.gamma_range)
            contrast = random.uniform(*self.contrast_range)
            image = F.adjust_gamma(image, gamma)
            image = F.adjust_contrast(image, contrast)
            aug_type = f"gamma{gamma:.2f}_contrast{contrast:.2f}"

            if self.visualize and self.visualized_count < self.visualize_limit:
                if random.random() < self.visualize_prob:
                    img_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    save_path = os.path.join(
                        self.visualize_dir, f"{base_idx:05d}_aug{index}_{aug_type}.png"
                    )
                    cv2.imwrite(save_path, img_np)
                    self.visualized_count += 1

        return image, action_onehot, angle