import os
import yaml
import torch
import random
import numpy as np
import cv2
from tqdm import tqdm
from ament_index_python.packages import get_package_share_directory

class GammaAugmentor:
    def __init__(self, config_path='config/train_params.yaml', input_dataset=None):
        self.package_dir = os.path.dirname(os.path.realpath(__file__))
        self.logs_dir = os.path.abspath(os.path.join(self.package_dir, '..', '..', 'logs'))
        self.config_path = os.path.abspath(os.path.join(self.package_dir, '..', '..', config_path))
        self.visualize_dir = os.path.join(self.logs_dir, 'visualize')

        self._load_config()

        if input_dataset is not None:
            self.input_dataset = input_dataset

        if self.visualize_flag:
            os.makedirs(self.visualize_dir, exist_ok=True)

    def _load_config(self):
        with open(self.config_path, 'r') as f:
            params = yaml.safe_load(f)['gamma']

        self.gamma_range = params['gamma_range']
        self.num_augmented_samples = params['num_augmented_samples']
        self.visualize_flag = params.get('visualize_image', False)

    def _apply_gamma_correction(self, image: np.ndarray, gamma: float) -> np.ndarray:
        inv_gamma = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255
            for i in range(256)
        ]).astype("uint8")
        return cv2.LUT(image, table)

    def augment(self):
        print(f"📦 Running GammaAugmentor on {len(self.input_dataset)} samples")

        new_images = []
        new_actions = []
        new_angles = []

        for idx in tqdm(range(len(self.input_dataset)), desc="GammaAugmenting"):
            image, action_onehot, angle = self.input_dataset[idx]

            new_images.append(image)
            new_actions.append(action_onehot)
            new_angles.append(angle)

            img_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            for i in range(self.num_augmented_samples):
                gamma = random.uniform(*self.gamma_range)
                gamma_img = self._apply_gamma_correction(img_np, gamma)
                gamma_tensor = torch.tensor(gamma_img, dtype=torch.float32).permute(2, 0, 1) / 255.0

                new_images.append(gamma_tensor)
                new_actions.append(action_onehot.clone())
                new_angles.append(angle.clone())

                if self.visualize_flag:
                    save_path = os.path.join(self.visualize_dir, f"{idx:05d}_aug{i}_gamma{gamma:.2f}.png")
                    cv2.imwrite(save_path, gamma_img)

        print(f"✅ Gamma augmentation complete: {len(new_images)} total samples")
        return torch.utils.data.TensorDataset(
            torch.stack(new_images),
            torch.stack(new_actions),
            torch.stack(new_angles)
        )


if __name__ == '__main__':
    print("🔧 Please use this module by importing GammaAugmentor(input_dataset=...) from training script.")
