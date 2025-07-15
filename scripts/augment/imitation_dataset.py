import os
import torch
import numpy as np
from torch.utils.data import Dataset
import cv2
import random


class ImitationDataset(Dataset):
    def __init__(self, dataset_dir, input_size=(88, 200), shift_aug=True, yaw_aug=True,shift_offset=5, vel_offset=0.2, n_action_classes=4, visualize_dir=None, future_steps=20):
        self.dataset_dir = dataset_dir
        self.image_dir = os.path.join(dataset_dir, "images")
        self.angle_dir = os.path.join(dataset_dir, "angle")
        self.action_dir = os.path.join(dataset_dir, "action")
        self.input_size = input_size
        self.shift_aug = shift_aug
        self.yaw_aug = yaw_aug
        self.shift_offset = shift_offset
        self.vel_offset = vel_offset
        self.n_action_classes = n_action_classes
        self.visualize_dir = visualize_dir
        self.future_steps = future_steps

        self.samples = self._gather_sample_paths()
        if self.visualize_dir:
            os.makedirs(self.visualize_dir, exist_ok=True)

    def _gather_sample_paths(self):
        filenames = sorted(os.listdir(self.image_dir))
        samples = []

        for fname in filenames:
            if not fname.endswith(".png"):
                continue
                
            base = fname[:-4]
            image_path = os.path.join(self.image_dir, base + ".png")
            angle_path = os.path.join(self.angle_dir, base + ".csv")
            action_path = os.path.join(self.action_dir, base + ".csv")

            if os.path.exists(image_path) and os.path.exists(angle_path) and os.path.exists(action_path):
                # 時系列データセットでは、angle_pathに未来のステップ分のデータが含まれている
                try:
                    angles = np.loadtxt(angle_path, delimiter=",", ndmin=1)
                    if len(angles) == self.future_steps:
                        samples.append((image_path, action_path, [angle_path]))
                    else:
                        print(f"警告: {angle_path} の時系列データ長が不正です（期待: {self.future_steps}, 実際: {len(angles)}）")
                except Exception as e:
                    print(f"警告: {angle_path} の読み込みに失敗しました: {e}")

        return samples

    def __len__(self):
        if self.shift_aug:
            return len(self.samples) * 7
        else:
            return len(self.samples)
        
    def apply_yaw_projection(image, yaw_deg, fov_deg=150):
        h, w = image.shape[:2]
        f = w / (2 * np.tan(np.deg2rad(fov_deg / 2)))

        # カメラ行列
        K = np.array([
            [f, 0, w / 2],
            [0, f, h / 2],
            [0, 0,     1]
        ])

        yaw = np.deg2rad(yaw_deg)
        R = np.array([
            [np.cos(yaw), 0, np.sin(yaw)],
            [0,           1, 0],
            [-np.sin(yaw), 0, np.cos(yaw)]
        ])

        K_inv = np.linalg.inv(K)
        H = K @ R @ K_inv
        return cv2.warpPerspective(image, H, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


    def __getitem__(self, idx):
        shift_signs = [-2.0, -1.0, 0.0, 1.0, 2.0] if self.shift_aug else [0.0]
        yaw_signs = [0.0] if self.yaw_aug else [0.0]

        base_idx = idx // 7 if self.shift_aug else idx
        aug_idx = idx % 7 if self.shift_aug else 0

        image_path, action_path, future_angle_paths = self.samples[base_idx]

        img = cv2.imread(image_path)
        img = cv2.resize(img, self.input_size[::-1])
        
        # 時系列angleデータを読み込み（未来20ステップ分）
        angle_path = future_angle_paths[0]  # 最初のファイルに時系列データが含まれている
        future_angles = np.loadtxt(angle_path, delimiter=",", ndmin=1).tolist()
        
        # 時系列データの妥当性チェック
        if len(future_angles) != self.future_steps:
            raise ValueError(f"時系列データの長さが不正です。期待: {self.future_steps}, 実際: {len(future_angles)}, ファイル: {angle_path}")
        
        action = int(np.loadtxt(action_path, delimiter=",", ndmin=1))

        # 全行動で同じパディング*射影変換を適用
        aug_combinations = [(s, y) for s in shift_signs for y in yaw_signs]
        
        if aug_idx < len(aug_combinations):
            shift_sign, yaw_deg = aug_combinations[aug_idx]
        else:
            shift_sign, yaw_deg = 0.0, 0.0

        # パディング処理（x方向にシフト）
        if shift_sign != 0.0:
            shift = int(shift_sign * self.input_size[0] * 0.1)
            h, w = img.shape[:2]
            trans_mat = np.array([[1, 0, shift], [0, 1, 0]], dtype=np.float32)
            img = cv2.warpAffine(img, trans_mat, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            # 未来フレームに進むほど緩和されるオフセットを適用
            for i in range(len(future_angles)):
                decay_factor = 1.0 - (i / len(future_angles))  # 1.0 -> 0.0に線形減衰
                future_angles[i] = future_angles[i] - shift_sign * self.vel_offset * decay_factor

        # 射影変換（yaw方向に回転）- 全行動に適用
        if yaw_deg != 0.0:
            img = ImitationDataset.apply_yaw_projection(img, yaw_deg=yaw_deg)
            yaw_offset = 0.4
            if yaw_deg < 0:
                for i in range(len(future_angles)):
                    decay_factor = 1.0 - (i / len(future_angles))  # 1.0 -> 0.0に線形減衰
                    future_angles[i] = future_angles[i] + yaw_offset * decay_factor
            elif yaw_deg > 0:
                for i in range(len(future_angles)):
                    decay_factor = 1.0 - (i / len(future_angles))  # 1.0 -> 0.0に線形減衰
                    future_angles[i] = future_angles[i] - yaw_offset * decay_factor

        if self.visualize_dir and idx < 100:
            save_path = os.path.join(self.visualize_dir, f"{idx:05d}_a{future_angles[0]:.2f}_c{action}.png")
            cv2.imwrite(save_path, img)

        img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        future_angles_tensor = torch.tensor(future_angles, dtype=torch.float32)
        action_tensor = torch.tensor(action, dtype=torch.long)
        action_onehot = torch.nn.functional.one_hot(action_tensor, num_classes=self.n_action_classes).squeeze().float()

        return img_tensor, action_onehot, future_angles_tensor