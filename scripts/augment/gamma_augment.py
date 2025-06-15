import os

from torch.utils.data import Dataset
import numpy as np
import torch
import random
import cv2


class GammaWrapperDataset(Dataset):
    def __init__(self, base_dataset, gamma_range=(0.9, 1.1), num_augmented_samples=1,
                 visualize=False, visualize_dir=None):
        self.base_dataset = base_dataset
        self.gamma_range = gamma_range
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
        is_aug = index % (1 + self.num_augmented_samples) != 0

        image, action_onehot, angle = self.base_dataset[base_idx]

        if is_aug:
            img_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            gamma = random.uniform(*self.gamma_range)
            inv_gamma = 1.0 / gamma
            table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
            gamma_img = cv2.LUT(img_np, table)

            if self.visualize and self.visualized_count < self.visualize_limit:
                if random.random() < self.visualize_prob:
                    save_path = os.path.join(
                        self.visualize_dir, f"{base_idx:05d}_aug{index}_gamma{gamma:.2f}.png"
                    )
                    cv2.imwrite(save_path, gamma_img)
                    self.visualized_count += 1

            with torch.no_grad():
                image = torch.tensor(gamma_img, dtype=torch.float32).permute(2, 0, 1) / 255.0

        return image, action_onehot, angle