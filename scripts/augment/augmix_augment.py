import os
import torch
import random
import numpy as np
import cv2


def create_augmix_augmented_dataset(base_dataset, num_augmented_samples=1,
                                    severity=3, width=3, depth=-1,
                                    allowed_ops=None, alpha=1.0,
                                    visualize=False, visualize_dir=None):
    """
    WebDataset対応のAugMix拡張データセット作成
    
    Args:
        base_dataset: ベースとなるWebDataset
        num_augmented_samples: 1つのサンプルから生成する拡張サンプル数
        severity: 拡張の強度 (1-10)
        width: 混合する拡張チェーンの数
        depth: 各チェーンの拡張操作数 (-1で1-3のランダム)
        allowed_ops: 使用する拡張操作のリスト
        alpha: Dirichlet分布のパラメータ
        visualize: 拡張画像の可視化フラグ
        visualize_dir: 可視化画像の保存先ディレクトリ
    """
    if visualize and visualize_dir:
        os.makedirs(visualize_dir, exist_ok=True)
    
    if allowed_ops is None:
        allowed_ops = ['rotate', 'contrast', 'brightness', 'sharpness']

    # WebDatasetのcompose()機能を活用
    return base_dataset.compose(lambda source: _apply_augmix_augmentation(
        source, num_augmented_samples, severity, width, depth,
        allowed_ops, alpha, visualize, visualize_dir
    ))


def _apply_augmix_augmentation(source, num_augmented_samples, severity,
                               width, depth, allowed_ops, alpha,
                               visualize, visualize_dir):
    """AugMix拡張処理のコア実装"""
    visualized_count = 0
    visualize_limit = 100

    for image, action_onehot, angle in source:
        # 元のサンプルを出力
        yield image, action_onehot, angle
        
        # 拡張サンプルを生成
        for aug_idx in range(num_augmented_samples):
            # PyTorchテンソルからnumpy配列に変換 (C, H, W) -> (H, W, C)
            img_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            
            # AugMix拡張を適用
            aug_img = _apply_augmix(img_np, severity, width, depth, allowed_ops, alpha)
            
            # 可視化用画像保存
            if visualize and visualize_dir and visualized_count < visualize_limit:
                save_path = os.path.join(
                    visualize_dir, 
                    f"{visualized_count:05d}_aug{aug_idx}_augmix.png"
                )
                cv2.imwrite(save_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
                visualized_count += 1
            
            # numpy配列からPyTorchテンソルに変換
            with torch.no_grad():
                augmented_image = torch.tensor(aug_img, dtype=torch.float32).permute(2, 0, 1) / 255.0
            
            yield augmented_image, action_onehot, angle


def _apply_augmix(image, severity, width, depth, allowed_ops, alpha):
    """AugMix拡張の実装"""
    ws = np.float32(np.random.dirichlet([alpha] * width))
    mix = np.zeros_like(image, dtype=np.float32)

    for i in range(width):
        img_aug = image.copy().astype(np.float32)
        d = depth if depth > 0 else np.random.randint(1, 4)
        ops = random.choices(allowed_ops, k=d)
        for op in ops:
            img_aug = _apply_op(img_aug, op, severity)
        mix += ws[i] * img_aug

    mixed = np.clip(mix, 0, 255).astype(np.uint8)
    return mixed


def _apply_op(img, op_name, severity):
    """AugMix拡張操作を適用"""
    op_map = {
        'autocontrast': _autocontrast,
        'equalize': _equalize,
        'posterize': _posterize,
        'rotate': _rotate,
        'solarize': _solarize,
        'shear_x': _shear_x,
        'shear_y': _shear_y,
        'translate_x': _translate_x,
        'translate_y': _translate_y,
        'color': _color,
        'contrast': _contrast,
        'brightness': _brightness,
        'sharpness': _sharpness,
    }
    return op_map[op_name](img, severity)


def _autocontrast(img, _):
    """オートコントラスト調整"""
    return cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)


def _equalize(img, severity):
    """ヒストグラム均等化"""
    img_yuv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)


def _posterize(img, severity):
    """ポスタライゼーション（色数削減）"""
    bits = 8 - int(severity)
    shift = 8 - bits
    return np.right_shift(np.left_shift(img, shift), shift)


def _rotate(img, severity):
    """軽微な回転変換"""
    degrees = random.uniform(-1, 1) * severity * 5
    h, w = img.shape[:2]
    mat = cv2.getRotationMatrix2D((w/2, h/2), degrees, 1.0)
    return cv2.warpAffine(img, mat, (w, h), borderMode=cv2.BORDER_REFLECT)


def _solarize(img, severity):
    """ソラリゼーション（階調反転）"""
    threshold = 256 - severity * 20
    return np.where(img < threshold, img, 255 - img).astype(np.uint8)


def _shear_x(img, severity):
    """X軸方向のせん断変換"""
    factor = random.uniform(-1, 1) * severity * 0.1
    M = np.array([[1, factor, 0], [0, 1, 0]], dtype=np.float32)
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)


def _shear_y(img, severity):
    """Y軸方向のせん断変換"""
    factor = random.uniform(-1, 1) * severity * 0.1
    M = np.array([[1, 0, 0], [factor, 1, 0]], dtype=np.float32)
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)


def _translate_x(img, severity):
    """X軸方向の平行移動"""
    shift = int(random.uniform(-1, 1) * severity * 5)
    M = np.array([[1, 0, shift], [0, 1, 0]], dtype=np.float32)
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)


def _translate_y(img, severity):
    """Y軸方向の平行移動"""
    shift = int(random.uniform(-1, 1) * severity * 5)
    M = np.array([[1, 0, 0], [0, 1, shift]], dtype=np.float32)
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)


def _color(img, severity):
    """彩度調整"""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] *= 1.0 + random.uniform(-1, 1) * severity * 0.1
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def _contrast(img, severity):
    """コントラスト調整"""
    factor = 1.0 + random.uniform(-1, 1) * severity * 0.1
    return cv2.convertScaleAbs(img, alpha=factor, beta=0)


def _brightness(img, severity):
    """明度調整"""
    factor = int(random.uniform(-1, 1) * severity * 10)
    return cv2.convertScaleAbs(img, alpha=1, beta=factor)


def _sharpness(img, severity):
    """シャープネス調整"""
    kernel = np.array([[0, -1, 0], [-1, 5 + severity, -1], [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)