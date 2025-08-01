import os

import numpy as np
import random
import cv2
import torchvision.transforms.functional as F


def create_gamma_augmented_dataset(base_dataset, gamma_range=(0.9, 1.1),
                                   num_augmented_samples=1,
                                   contrast_range=(0.8, 1.2), visualize=False,
                                   visualize_dir=None):
    if visualize and visualize_dir:
        os.makedirs(visualize_dir, exist_ok=True)

    # WebDatasetのcompose()機能を活用
    return base_dataset.compose(lambda source: _apply_gamma_augmentation(
        source, gamma_range, contrast_range, num_augmented_samples,
        visualize, visualize_dir
    ))


def _apply_gamma_augmentation(source, gamma_range, contrast_range,
                              num_augmented_samples, visualize,
                              visualize_dir):
    visualized_count = 0
    visualize_limit = 100

    for image, action_onehot, angle in source:
        yield image, action_onehot, angle

        for _ in range(num_augmented_samples):
            gamma = random.uniform(*gamma_range)
            contrast = random.uniform(*contrast_range)
            augmented_image = F.adjust_gamma(image, gamma)
            augmented_image = F.adjust_contrast(augmented_image, contrast)

            if visualize and visualize_dir and visualized_count < visualize_limit:
                img_np = (augmented_image.permute(1, 2, 0).numpy() * 255).astype(
                    np.uint8)
                bgr_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                save_path = os.path.join(
                    visualize_dir,
                    f"{visualized_count:05d}_gamma{gamma:.2f}_"
                    f"contrast{contrast:.2f}.png"
                )
                cv2.imwrite(save_path, bgr_img)
                visualized_count += 1

            yield augmented_image, action_onehot, angle