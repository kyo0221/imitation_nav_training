import os
import yaml
import torch
import random
import numpy as np
import cv2
from tqdm import tqdm
from ament_index_python.packages import get_package_share_directory

from torch.utils.data import Dataset


# AugMix component operations
class augmentations:
    @staticmethod
    def apply_op(img, op_name, severity):
        op_map = {
            'autocontrast': augmentations.autocontrast,
            'equalize': augmentations.equalize,
            'posterize': augmentations.posterize,
            'rotate': augmentations.rotate,
            'solarize': augmentations.solarize,
            'shear_x': augmentations.shear_x,
            'shear_y': augmentations.shear_y,
            'translate_x': augmentations.translate_x,
            'translate_y': augmentations.translate_y,
            'color': augmentations.color,
            'contrast': augmentations.contrast,
            'brightness': augmentations.brightness,
            'sharpness': augmentations.sharpness,
        }
        return op_map[op_name](img, severity)

    @staticmethod
    def autocontrast(img, _):
        return cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    @staticmethod
    def equalize(img, severity):
        img_yuv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

    @staticmethod
    def posterize(img, severity):
        bits = 8 - int(severity)
        shift = 8 - bits
        return np.right_shift(np.left_shift(img, shift), shift)

    @staticmethod
    def rotate(img, severity):
        degrees = random.uniform(-1, 1) * severity * 5
        h, w = img.shape[:2]
        mat = cv2.getRotationMatrix2D((w/2, h/2), degrees, 1.0)
        return cv2.warpAffine(img, mat, (w, h), borderMode=cv2.BORDER_REFLECT)

    @staticmethod
    def solarize(img, severity):
        threshold = 256 - severity * 20
        return np.where(img < threshold, img, 255 - img).astype(np.uint8)

    @staticmethod
    def shear_x(img, severity):
        factor = random.uniform(-1, 1) * severity * 0.1
        M = np.array([[1, factor, 0], [0, 1, 0]], dtype=np.float32)
        return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)

    @staticmethod
    def shear_y(img, severity):
        factor = random.uniform(-1, 1) * severity * 0.1
        M = np.array([[1, 0, 0], [factor, 1, 0]], dtype=np.float32)
        return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)

    @staticmethod
    def translate_x(img, severity):
        shift = int(random.uniform(-1, 1) * severity * 5)
        M = np.array([[1, 0, shift], [0, 1, 0]], dtype=np.float32)
        return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)

    @staticmethod
    def translate_y(img, severity):
        shift = int(random.uniform(-1, 1) * severity * 5)
        M = np.array([[1, 0, 0], [0, 1, shift]], dtype=np.float32)
        return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)

    @staticmethod
    def color(img, severity):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] *= 1.0 + random.uniform(-1, 1) * severity * 0.1
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    @staticmethod
    def contrast(img, severity):
        factor = 1.0 + random.uniform(-1, 1) * severity * 0.1
        return cv2.convertScaleAbs(img, alpha=factor, beta=0)

    @staticmethod
    def brightness(img, severity):
        factor = int(random.uniform(-1, 1) * severity * 10)
        return cv2.convertScaleAbs(img, alpha=1, beta=factor)

    @staticmethod
    def sharpness(img, severity):
        kernel = np.array([[0, -1, 0], [-1, 5 + severity, -1], [0, -1, 0]])
        return cv2.filter2D(img, -1, kernel)

    @staticmethod
    def augmix(image, severity, width, depth, allowed_ops=None, alpha=1.0):
        ws = np.float32(np.random.dirichlet([alpha] * width))
        mix = np.zeros_like(image, dtype=np.float32)

        for i in range(width):
            img_aug = image.copy().astype(np.float32)
            d = depth if depth > 0 else np.random.randint(1, 4)
            ops = random.choices(allowed_ops, k=d)
            for op in ops:
                img_aug = augmentations.apply_op(img_aug, op, severity)
            mix += ws[i] * img_aug

        mixed = np.clip(mix, 0, 255).astype(np.uint8)
        return mixed

    
class AugMixWrapperDataset(Dataset):
    def __init__(self, base_dataset, num_augmented_samples=1, severity=3, width=3, depth=-1,
                 allowed_ops=None, alpha=1.0, visualize=False, visualize_dir=None):
        self.base_dataset = base_dataset
        self.num_augmented_samples = num_augmented_samples
        self.severity = severity
        self.width = width
        self.depth = depth
        self.allowed_ops = allowed_ops or ['rotate', 'contrast', 'brightness', 'sharpness']
        self.alpha = alpha

        self.visualize = visualize
        self.visualize_dir = visualize_dir
        self.visualized_count = 0
        self.visualize_limit = 100
        self.total_augmented = len(base_dataset) * num_augmented_samples
        self.visualize_prob = min(1.0, self.visualize_limit / self.total_augmented)

        if self.visualize and self.visualize_dir:
            os.makedirs(self.visualize_dir, exist_ok=True)

        self.original_length = len(self.base_dataset)

    def __len__(self):
        return self.original_length * (1 + self.num_augmented_samples)

    def __getitem__(self, index):
        base_idx = index // (1 + self.num_augmented_samples)
        is_aug = index % (1 + self.num_augmented_samples) != 0

        image, action_onehot, angle = self.base_dataset[base_idx]

        if is_aug:
            img_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            aug_img = self._apply_augmix(img_np)

            if self.visualize and self.visualized_count < self.visualize_limit:
                if random.random() < self.visualize_prob:
                    save_path = os.path.join(self.visualize_dir, f"{base_idx:05d}_aug{index}.png")
                    cv2.imwrite(save_path, aug_img[:, :, ::-1])  # RGB → BGR
                    self.visualized_count += 1

            with torch.no_grad():
                image = torch.tensor(aug_img, dtype=torch.float32).permute(2, 0, 1) / 255.0

        return image, action_onehot, angle

    def _apply_augmix(self, image: np.ndarray) -> np.ndarray:
        ws = np.float32(np.random.dirichlet([self.alpha] * self.width))
        mix = np.zeros_like(image, dtype=np.float32)

        for i in range(self.width):
            img_aug = image.copy().astype(np.float32)
            d = self.depth if self.depth > 0 else np.random.randint(1, 4)
            ops = random.choices(self.allowed_ops, k=d)
            for op in ops:
                img_aug = self._apply_op(img_aug, op, self.severity)
            mix += ws[i] * img_aug

        mixed = np.clip(mix, 0, 255).astype(np.uint8)
        return mixed

    def _apply_op(self, img, op_name, severity):
        return augmentations.apply_op(img, op_name, severity)