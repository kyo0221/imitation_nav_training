import os
import torch
import numpy as np
from torch.utils.data import Dataset
import cv2


class ImitationDataset(Dataset):
    def __init__(self, dataset_dir, input_size=(88, 200), rotate_aug=True, angle_offset_deg=5, vel_offset=0.2, n_action_classes=3):
        self.dataset_dir = dataset_dir
        self.image_dir = os.path.join(dataset_dir, "images")
        self.angle_dir = os.path.join(dataset_dir, "angle")
        self.action_dir = os.path.join(dataset_dir, "action")
        self.input_size = input_size
        self.rotate_aug = rotate_aug
        self.angle_offset_deg = angle_offset_deg
        self.vel_offset = vel_offset
        self.n_action_classes = n_action_classes

        self.data = []
        self.data_augmentation()

    def data_augmentation(self):
        filenames = sorted(os.listdir(self.image_dir))
        for fname in filenames:
            if not fname.endswith(".png"):
                continue
            base = fname[:-4]

            image_path = os.path.join(self.image_dir, base + ".png")
            angle_path = os.path.join(self.angle_dir, base + ".csv")
            action_path = os.path.join(self.action_dir, base + ".csv")

            if not (os.path.exists(image_path) and os.path.exists(angle_path) and os.path.exists(action_path)):
                continue

            img = cv2.imread(image_path)
            img = cv2.resize(img, self.input_size[::-1])
            img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0

            angle = torch.tensor(np.loadtxt(angle_path, delimiter=",", ndmin=1), dtype=torch.float32)
            action = torch.tensor(np.loadtxt(action_path, delimiter=",", ndmin=1), dtype=torch.long)

            self.data.append((img_tensor, action, angle))

            if self.rotate_aug:
                h, w = img.shape[:2]

                for sign in [-1.5, -1, -0.5, 0.5, 1, 1.5]:
                    shift = int(sign * self.input_size[0] * 0.1)
                    trans_mat = np.array([[1, 0, shift], [0, 1, 0]], dtype=np.float32)
                    shifted_img = cv2.warpAffine(img, trans_mat, (w, h), borderMode=cv2.BORDER_REFLECT)
                    shifted_tensor = torch.tensor(shifted_img, dtype=torch.float32).permute(2, 0, 1) / 255.0
                    corrected_vel = angle + torch.tensor([-sign * self.vel_offset], dtype=torch.float32)
                    self.data.append((shifted_tensor, action, corrected_vel))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, action, angle = self.data[idx]
        action_onehot = torch.nn.functional.one_hot(action, num_classes=self.n_action_classes).squeeze().float()
        return image, action_onehot, angle
