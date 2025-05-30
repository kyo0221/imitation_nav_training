import os
import yaml
import torch
import random
import numpy as np
import cv2
from tqdm import tqdm
from ament_index_python.packages import get_package_share_directory


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


class AugMixAugmentor:
    def __init__(self, input_dataset, config_path='config/train_params.yaml'):
        self.input_dataset = input_dataset
        self.package_dir = os.path.dirname(os.path.realpath(__file__))
        self.logs_dir = os.path.abspath(os.path.join(self.package_dir, '..', '..', 'logs'))
        self.config_path = os.path.abspath(os.path.join(self.package_dir, '..', '..', config_path))
        self.visualize_dir = os.path.join(self.logs_dir, 'visualize')

        self._load_config()
        if self.visualize_flag:
            os.makedirs(self.visualize_dir, exist_ok=True)

    def _load_config(self):
        with open(self.config_path, 'r') as f:
            params = yaml.safe_load(f)['augmix']

        self.num_augmented_samples = params['num_augmented_samples']
        self.severity = params.get('severity', 3)
        self.width = params.get('width', 3)
        self.depth = params.get('depth', -1)
        self.allowed_ops = params.get('operations', ['rotate', 'contrast', 'brightness', 'sharpness', 'blur'])
        self.visualize_flag = params.get('visualize_image', False)
        self.alpha = params.get('alpha', 1.0)

    def augment(self):
        print(f"📦 Running AugMix on {len(self.input_dataset)} samples")
        new_images = []
        new_actions = []
        new_angles = []

        for idx in tqdm(range(len(self.input_dataset)), desc="AugMixing"):
            image, action_onehot, angle = self.input_dataset[idx]

            new_images.append(image)
            new_actions.append(action_onehot)
            new_angles.append(angle)

            img_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            for i in range(self.num_augmented_samples):
                aug_img = augmentations.augmix(
                    img_np, self.severity, self.width, self.depth,
                    self.allowed_ops, self.alpha
                )
                aug_tensor = torch.tensor(aug_img, dtype=torch.float32).permute(2, 0, 1) / 255.0

                new_images.append(aug_tensor)
                new_actions.append(action_onehot.clone())
                new_angles.append(angle.clone())

                if self.visualize_flag:
                    save_path = os.path.join(self.visualize_dir, f"{idx:05d}_aug{i}_augmix.png")
                    cv2.imwrite(save_path, aug_img)

        return torch.utils.data.TensorDataset(
            torch.stack(new_images),
            torch.stack(new_actions),
            torch.stack(new_angles)
        )


if __name__ == '__main__':
    augmentor = AugMixAugmentor()
    augmentor.augment()
