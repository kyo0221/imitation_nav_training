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
    def __init__(self, dataset_dir, input_size=(88, 200), shift_aug=True, yaw_aug=True, 
                 shift_offset=5, vel_offset=0.2, n_action_classes=4, 
                 shift_signs=None, yaw_signs=None, visualize_dir=None):
        """
        WebDataset形式のデータを読み込むデータセットクラス
        
        Args:
            dataset_dir: WebDatasetファイルがあるディレクトリパス
            input_size: 入力画像サイズ (height, width)
            shift_aug: シフト拡張を行うかどうか
            yaw_aug: Yaw拡張を行うかどうか
            shift_offset: シフト拡張のオフセット
            vel_offset: 速度オフセット
            n_action_classes: アクション数
            shift_signs: シフト拡張の係数リスト（デフォルト: [-2.0, -1.0, 0.0, 1.0, 2.0]）
            yaw_signs: Yaw拡張の角度リスト（デフォルト: [0.0]）
            visualize_dir: 可視化用保存ディレクトリ
        """
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
        
        # WebDatasetのパターンを設定
        import glob
        shard_files = glob.glob(os.path.join(dataset_dir, "shard_*.tar*"))
        if not shard_files:
            raise ValueError(f"No shard files found in {dataset_dir}")
        self.shard_pattern = shard_files
        
        # データセットを初期化
        self.dataset = self._create_webdataset()
        
        # サンプル数を計算
        self.samples_count = self._count_samples()
        
        if self.visualize_dir:
            os.makedirs(self.visualize_dir, exist_ok=True)
    
    def _handle_sample_format(self, sample):
        # メタデータを取得
        if "metadata.json" in sample:
            # data_collector_node.pyの新しい形式に対応
            metadata_data = sample["metadata.json"]
            action_data = sample["action.json"]
            # metadataからangle情報を抽出
            if isinstance(metadata_data, (str, bytes)):
                metadata_info = json.loads(metadata_data)
            else:
                metadata_info = metadata_data
            angle_data = json.dumps({"angle": metadata_info.get("angle", 0.0)})
        else:
            raise ValueError("Missing metadata.json in sample")
        
        # numpy形式の画像データを直接取得（PIL変換なし）
        img_array = sample["npy"]  # 既にnumpy配列として提供される
        
        # webdatasetがbytesとして提供する場合はnp.loadで読み込み
        if isinstance(img_array, bytes):
            img_array = np.load(BytesIO(img_array))
        
        # RGB形式のnumpy配列として直接返す（PIL変換不要）
        return (img_array, angle_data, action_data)
    
    def _create_webdataset(self):
        """WebDatasetを作成"""
        return (
            wds.WebDataset(self.shard_pattern, shardshuffle=False)
            .map(self._handle_sample_format)
        )
    
    def _count_samples(self):
        """サンプル数をカウント"""
        # シャードファイルから統計情報を読み込み
        stats_file = os.path.join(self.dataset_dir, "dataset_stats.json")
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                stats = json.load(f)
                return stats['total_samples']
        
        # 統計情報がない場合は実際にサンプルをカウント
        count = 0
        for sample in self.dataset:
            count += 1
        return count
    
    def __len__(self):
        if self.shift_aug:
            # 拡張係数の数 × yaw拡張の数
            aug_factor = len(self.shift_signs) * len(self.yaw_signs)
            return self.samples_count * aug_factor
        else:
            return self.samples_count
    
    @staticmethod
    def apply_yaw_projection(image, yaw_deg, fov_deg=150):
        """Yaw方向の射影変換を適用"""
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
        shift_signs = self.shift_signs if self.shift_aug else [0.0]
        yaw_signs = self.yaw_signs if self.yaw_aug else [0.0]

        aug_factor = len(shift_signs) * len(yaw_signs)
        base_idx = idx // aug_factor if self.shift_aug else idx
        aug_idx = idx % aug_factor if self.shift_aug else 0

        # WebDatasetから対応するサンプルを取得
        try:
            # データセットを再作成してイテレート
            if not hasattr(self, '_samples_cache'):
                print(f"Loading WebDataset samples into cache...")
                self._samples_cache = list(self.dataset)
                print(f"Successfully cached {len(self._samples_cache)} samples")
            
            samples = self._samples_cache
            if base_idx >= len(samples):
                base_idx = base_idx % len(samples)
            
            img_array, angle_data, action_data = samples[base_idx]
            
            # JSONデータを解析
            angle_info = json.loads(angle_data)
            action_info = json.loads(action_data)
            
            # 画像データを直接使用（numpy配列のまま、色変換不要）
            img = cv2.resize(img_array, self.input_size[::-1])
            
            angle = float(angle_info['angle'])
            action = int(action_info['action'])
            
            # 拡張の組み合わせを適用
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
                angle -= shift_sign * self.vel_offset

            # 射影変換（yaw方向に回転）
            if yaw_deg != 0.0:
                img = self.apply_yaw_projection(img, yaw_deg=yaw_deg)
                if yaw_deg < 0:
                    angle += 0.4
                elif yaw_deg > 0:
                    angle -= 0.4

            # 可視化用保存
            if self.visualize_dir and idx < 100:
                save_path = os.path.join(self.visualize_dir, f"{idx:05d}_a{angle:.2f}_c{action}.png")
                cv2.imwrite(save_path, img)

            # テンソルに変換（メモリ効率向上）
            img_tensor = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0
            angle_tensor = torch.tensor([angle], dtype=torch.float32)
            action_tensor = torch.tensor(action, dtype=torch.long)
            action_onehot = torch.nn.functional.one_hot(action_tensor, num_classes=self.n_action_classes).squeeze().float()

            return img_tensor, action_onehot, angle_tensor
            
        except Exception as e:
            print(f"Error loading sample {idx} (base_idx: {base_idx}): {e}")
            raise e


class WebDatasetIterableLoader:
    """
    反復可能なWebDatasetローダー（大規模データセット用）
    """
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