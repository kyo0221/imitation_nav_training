import os
import torch
import numpy as np
import cv2
import json
import webdataset as wds
from io import BytesIO
from tqdm import tqdm
import glob


def create_webdataset_streaming_loader(dataset_dir, input_size=(224, 224),
                                       vel_offset=0.2, n_action_classes=4,
                                       gaussian_shift_params=None,
                                       visualize_dir=None, shuffle_buffer=1000):
    # ガウス分布パラメータの設定
    if gaussian_shift_params is None:
        gaussian_shift_params = {
            'mean': 0.0,      # 分布の中心値（0.0で中央基準）
            'std': 0.67,      # 標準偏差（3σ≈2.0で従来範囲[-2.0,2.0]をカバー）
            'clip_range': [-2.0, 2.0]  # 値域制限（極端な値を防ぐ）
        }

    shard_files = glob.glob(os.path.join(dataset_dir, "shard_*.tar*"))
    if not shard_files:
        raise ValueError(f"No shard files found in {dataset_dir}")
    shard_pattern = sorted(shard_files)

    if visualize_dir:
        os.makedirs(visualize_dir, exist_ok=True)

    # WebDatasetを直接作成し、compose()で処理を統合
    dataset = (wds.WebDataset(shard_pattern, shardshuffle=True,
                              empty_check=False)
               .shuffle(shuffle_buffer)
               .map(lambda sample: _handle_sample_format(sample))
               .compose(lambda source: _process_streaming_samples(
                   source, input_size, vel_offset, n_action_classes,
                   gaussian_shift_params, visualize_dir)))

    return dataset


def _handle_sample_format(sample):
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


def _process_streaming_samples(source, input_size, vel_offset,
                               n_action_classes, gaussian_shift_params,
                               visualize_dir):
    vis_counter = 0

    for img_array, angle_data, action_data in source:
        angle_info = json.loads(angle_data)
        action_info = json.loads(action_data)

        angle = float(angle_info['angle'])
        action = int(action_info['action'])

        # ガウス分布から複数のshift値をサンプリング（従来の5個と同様）
        num_samples = 5  # 従来のshift_signsの数と同じ
        shift_values = []
        for _ in range(num_samples):
            shift_val = np.random.normal(
                gaussian_shift_params['mean'],
                gaussian_shift_params['std']
            )
            # 値域制限（クリッピング）
            clip_min, clip_max = gaussian_shift_params['clip_range']
            shift_val = np.clip(shift_val, clip_min, clip_max)
            shift_values.append(shift_val)

        for shift_sign in shift_values:
            img_square = _apply_horizontal_shift_and_crop(img_array,
                                                          shift_sign)
            img = cv2.resize(img_square, input_size[::-1])

            adjusted_angle = angle
            # ガウス分布では常に角度調整を適用
            adjusted_angle += shift_sign * vel_offset

            if visualize_dir and vis_counter < 100:
                filename = (f"{vis_counter:05d}_shift{shift_sign:.1f}_"
                            f"a{adjusted_angle:.3f}_c{action}.png")
                save_path = os.path.join(visualize_dir, filename)
                cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                vis_counter += 1

            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().div_(
                255.0)
            angle_tensor = torch.tensor([adjusted_angle], dtype=torch.float32)
            action_tensor = torch.tensor(action, dtype=torch.long)
            action_onehot = torch.nn.functional.one_hot(
                action_tensor, num_classes=n_action_classes).squeeze().float()

            yield (img_tensor, action_onehot, angle_tensor)


def _apply_horizontal_shift_and_crop(img, shift_sign):
    h, w = img.shape[:2]
    crop_size = h
    center_x = w // 2
    max_shift = (w - crop_size) // 2
    x_offset = int(shift_sign * max_shift / 2.0)
    x_start = center_x - crop_size // 2 + x_offset
    x_start = max(0, min(x_start, w - crop_size))
    cropped_img = img[:, x_start:x_start + crop_size]
    return cropped_img


def _count_samples_efficient(dataset_dir, shard_pattern):
    stats_file = os.path.join(dataset_dir, "dataset_stats.json")
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            stats = json.load(f)
            return stats['total_samples']

    print(f"Counting samples across {len(shard_pattern)} shards...")
    count = 0
    dataset = wds.WebDataset(shard_pattern, shardshuffle=False,
                             empty_check=False)

    with tqdm(desc="Counting samples", unit="samples",
              dynamic_ncols=True) as pbar:
        for sample in dataset:
            count += 1
            pbar.update(1)

    return count


# 後方互換性のためのクラス（廃止予定）
class WebDatasetStreamingLoader:
    def __init__(self, *args, **kwargs):
        print("Warning: WebDatasetStreamingLoader class is deprecated. "
              "Use create_webdataset_streaming_loader() function instead.")
        self.dataset = create_webdataset_streaming_loader(*args, **kwargs)

    def __iter__(self):
        return iter(self.dataset)


# 後方互換性のためのエイリアス
WebDatasetLoader = WebDatasetStreamingLoader


class WebDatasetIterableLoader:
    def __init__(self, dataset_dir, input_size=(88, 200), batch_size=32,
                 vel_offset=0.2, n_action_classes=4, shuffle=True):
        self.dataset_dir = dataset_dir
        self.input_size = input_size
        self.batch_size = batch_size
        self.vel_offset = vel_offset
        self.n_action_classes = n_action_classes
        self.shuffle = shuffle

        # WebDatasetのパターンを設定
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

        # 拡張処理はメインクラスで実行

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
        img_array = np.frombuffer(img_bytes,
                                  dtype=image_dtype).reshape(image_shape)

        # リサイズ
        img = cv2.resize(img_array, self.input_size[::-1])

        angle = float(angle_info['angle'])
        action = int(action_info['action'])

        # テンソルに変換
        img_tensor = torch.tensor(img, dtype=torch.float32).permute(
            2, 0, 1) / 255.0
        angle_tensor = torch.tensor([angle], dtype=torch.float32)
        action_tensor = torch.tensor(action, dtype=torch.long)
        action_onehot = torch.nn.functional.one_hot(
            action_tensor, num_classes=self.n_action_classes).squeeze().float()

        return img_tensor, action_onehot, angle_tensor

    def _apply_augmentation(self, sample):
        """拡張を適用"""
        # TODO: 必要に応じて拡張処理を実装
        return sample
