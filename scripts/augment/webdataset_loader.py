import os
import torch
import numpy as np
from torch.utils.data import Dataset
import cv2
import json
import random
import webdataset as wds
from io import BytesIO
from PIL import Image


class WebDatasetLoader(Dataset):
    def __init__(self, dataset_dir, input_size=(224, 224), shift_aug=True, yaw_aug=False, 
                 shift_offset=5, vel_offset=0.2, n_action_classes=4, 
                 shift_signs=None, yaw_signs=None, visualize_dir=None):
        self.dataset_dir = dataset_dir
        self.input_size = input_size
        self.shift_aug = shift_aug
        self.yaw_aug = yaw_aug
        self.shift_offset = shift_offset
        self.vel_offset = vel_offset
        self.n_action_classes = n_action_classes
        self.shift_signs = shift_signs if shift_signs is not None else [-2.0, -1.0, 0.0, 1.0, 2.0]
        self.yaw_signs = yaw_signs if yaw_signs is not None else [0.0]
        self.visualize_dir = visualize_dir
        
        import glob
        shard_files = glob.glob(os.path.join(dataset_dir, "shard_*.tar*"))
        if not shard_files:
            raise ValueError(f"No shard files found in {dataset_dir}")
        self.shard_pattern = sorted(shard_files)
        
        self.samples_count = self._count_samples_efficient()
        self._build_index_mapping()
        
        if self.visualize_dir:
            os.makedirs(self.visualize_dir, exist_ok=True)
    
    def _handle_sample_format(self, sample):
        if "metadata.json" in sample:
            metadata_data = sample["metadata.json"]
            action_data = sample["action.json"]
            if isinstance(metadata_data, (str, bytes)):
                metadata_info = json.loads(metadata_data)
            else:
                metadata_info = metadata_data
            angle_data = json.dumps({"angle": metadata_info.get("angle", 0.0)})
        else:
            raise ValueError("Missing metadata.json in sample")
        
        img_array = sample["npy"]
        if isinstance(img_array, bytes):
            img_array = np.load(BytesIO(img_array))
        
        return (img_array, angle_data, action_data)
    
    def _build_index_mapping(self):
        self.shard_samples = []
        self.cumulative_samples = [0]
        
        stats_file = os.path.join(self.dataset_dir, "dataset_stats.json")
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                stats = json.load(f)
                if 'shard_info' in stats:
                    for shard_info in stats['shard_info']:
                        samples_in_shard = shard_info.get('samples', 0)
                        self.shard_samples.append(samples_in_shard)
                        self.cumulative_samples.append(self.cumulative_samples[-1] + samples_in_shard)
                    return
        
        # フォールバック: 均等分散と仮定
        estimated_per_shard = max(1, self.samples_count // len(self.shard_pattern))
        for i in range(len(self.shard_pattern)):
            self.shard_samples.append(estimated_per_shard)
            self.cumulative_samples.append(self.cumulative_samples[-1] + estimated_per_shard)
    
    def _count_samples_efficient(self):
        stats_file = os.path.join(self.dataset_dir, "dataset_stats.json")
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                stats = json.load(f)
                return stats['total_samples']
        
        count = 0
        dataset = wds.WebDataset(self.shard_pattern, shardshuffle=False)
        for _ in dataset:
            count += 1
        return count
    
    def __len__(self):
        if self.shift_aug:
            # 拡張係数の数 × yaw拡張の数
            aug_factor = len(self.shift_signs) * len(self.yaw_signs)
            return self.samples_count * aug_factor
        else:
            return self.samples_count
    
    def _apply_horizontal_shift_and_crop(self, img, shift_sign):
        h, w = img.shape[:2]
        crop_size = h
        center_x = w // 2
        max_shift = (w - crop_size) // 2
        x_offset = int(shift_sign * max_shift / 2.0)
        x_start = center_x - crop_size // 2 + x_offset
        x_start = max(0, min(x_start, w - crop_size))
        cropped_img = img[:, x_start:x_start + crop_size]
        return cropped_img
    
    def _get_sample_from_shard(self, shard_idx, sample_idx_in_shard):
        shard_file = self.shard_pattern[shard_idx]
        dataset = wds.WebDataset([shard_file], shardshuffle=False).map(self._handle_sample_format)
        
        for i, sample in enumerate(dataset):
            if i == sample_idx_in_shard:
                return sample
        
        for sample in dataset:
            return sample
        
        raise IndexError(f"Sample not found in shard {shard_idx}")
    
    def _find_shard_for_index(self, base_idx):
        if base_idx >= self.samples_count:
            base_idx = base_idx % self.samples_count
        
        for i in range(len(self.cumulative_samples) - 1):
            if self.cumulative_samples[i] <= base_idx < self.cumulative_samples[i + 1]:
                shard_idx = i
                sample_idx_in_shard = base_idx - self.cumulative_samples[i]
                return shard_idx, sample_idx_in_shard
        
        return len(self.shard_samples) - 1, 0

    def __getitem__(self, idx):
        shift_signs = self.shift_signs if self.shift_aug else [0.0]
        yaw_signs = [0.0]  # yaw_aug無効化

        aug_factor = len(shift_signs)
        base_idx = idx // aug_factor if self.shift_aug else idx
        aug_idx = idx % aug_factor if self.shift_aug else 0

        try:
            shard_idx, sample_idx_in_shard = self._find_shard_for_index(base_idx)
            img_array, angle_data, action_data = self._get_sample_from_shard(shard_idx, sample_idx_in_shard)
            
            angle_info = json.loads(angle_data)
            action_info = json.loads(action_data)
            
            angle = float(angle_info['angle'])
            action = int(action_info['action'])
            
            if aug_idx < len(shift_signs):
                shift_sign = shift_signs[aug_idx]
            else:
                shift_sign = 0.0

            img_square = self._apply_horizontal_shift_and_crop(img_array, shift_sign)
            img = cv2.resize(img_square, self.input_size[::-1])
            
            if shift_sign != 0.0:
                angle += shift_sign * self.vel_offset

            if self.visualize_dir and idx < 100:
                save_path = os.path.join(self.visualize_dir, f"{idx:05d}_shift{shift_sign:.1f}_a{angle:.3f}_c{action}.png")
                cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().div_(255.0)
            angle_tensor = torch.tensor([angle], dtype=torch.float32)
            action_tensor = torch.tensor(action, dtype=torch.long)
            action_onehot = torch.nn.functional.one_hot(action_tensor, num_classes=self.n_action_classes).squeeze().float()

            return img_tensor, action_onehot, angle_tensor
            
        except Exception as e:
            print(f"Error loading sample {idx} (base_idx: {base_idx}, shard: {shard_idx if 'shard_idx' in locals() else 'unknown'}): {e}")
            raise e


class WebDatasetIterableLoader:
    def __init__(self, dataset_dir, input_size=(88, 200), batch_size=32, 
                 shift_aug=True, yaw_aug=True, shift_offset=5, vel_offset=0.2, 
                 n_action_classes=4, shuffle=True):
        self.dataset_dir = dataset_dir
        self.input_size = input_size
        self.batch_size = batch_size
        self.shift_aug = shift_aug
        self.yaw_aug = yaw_aug
        self.shift_offset = shift_offset
        self.vel_offset = vel_offset
        self.n_action_classes = n_action_classes
        self.shuffle = shuffle
        
        # WebDatasetのパターンを設定
        import glob
        shard_files = glob.glob(os.path.join(dataset_dir, "shard_*.tar*"))
        if not shard_files:
            raise ValueError(f"No shard files found in {dataset_dir}")
        self.shard_pattern = shard_files
    
    def create_dataloader(self):
        """DataLoaderを作成"""
        dataset = (
            wds.WebDataset(self.shard_pattern)
            .shuffle(1000 if self.shuffle else 0)
            .to_tuple("npy", "angle.json", "action.json")
            .map(self._process_sample)
        )
        
        if self.shift_aug:
            dataset = dataset.map(self._apply_augmentation)
        
        return torch.utils.data.DataLoader(
            dataset.batched(self.batch_size),
            batch_size=None,
            num_workers=0
        )
    
    def _process_sample(self, sample):
        """サンプルを処理（numpy専用）"""
        img_bytes, angle_data, action_data = sample
        
        # JSONデータを解析
        angle_info = json.loads(angle_data)
        action_info = json.loads(action_data)
        
        # メタデータから画像情報を取得
        image_shape = angle_info.get('image_shape')
        image_dtype = angle_info.get('image_dtype', 'uint8')
        
        if image_shape is None:
            raise ValueError("Missing image shape in numpy format")
        
        # バイト列からnumpy配列に変換
        img_array = np.frombuffer(img_bytes, dtype=image_dtype).reshape(image_shape)
        
        # リサイズ
        img = cv2.resize(img_array, self.input_size[::-1])
        
        angle = float(angle_info['angle'])
        action = int(action_info['action'])
        
        # テンソルに変換
        img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        angle_tensor = torch.tensor([angle], dtype=torch.float32)
        action_tensor = torch.tensor(action, dtype=torch.long)
        action_onehot = torch.nn.functional.one_hot(action_tensor, num_classes=self.n_action_classes).squeeze().float()
        
        return img_tensor, action_onehot, angle_tensor
    
    def _apply_augmentation(self, sample):
        """拡張を適用"""
        # TODO: 必要に応じて拡張処理を実装
        return sample