import torch
import numpy as np
from torch.utils.data import Dataset
from collections import Counter
from tqdm import tqdm
import time


class ResamplingWrapperDataset(Dataset):
    def __init__(self, base_dataset, enable_resampling=True):
        """
        オーバーサンプリングによる各行動クラスのバランシングWrapper
        
        Args:
            base_dataset: ベースとなるデータセット
            enable_resampling: リサンプリングを有効にするかどうか
        """
        self.base_dataset = base_dataset
        self.enable_resampling = enable_resampling
        
        if self.enable_resampling:
            self._create_resampled_indices()
        else:
            self.resampled_indices = list(range(len(base_dataset)))
    
    def _create_resampled_indices(self):
        """各行動クラスのオーバーサンプリングインデックスを作成"""
        print("🔍 Analyzing action class distribution...")
        
        # 各行動クラスのサンプルインデックスを収集
        action_indices = {i: [] for i in range(self.n_action_classes)}
        
        # プログレスバー付きでデータセット解析
        for idx in tqdm(range(len(self.base_dataset)), desc="Scanning dataset", unit="samples"):
            _, action_onehot, _ = self.base_dataset[idx]
            action = torch.argmax(action_onehot).item()
            action_indices[action].append(idx)
        
        # 各行動クラスのサンプル数を確認
        action_counts = {action: len(indices) for action, indices in action_indices.items()}
        
        # 行動が一つもないクラスを除外
        valid_actions = {action: indices for action, indices in action_indices.items() if len(indices) > 0}
        
        if not valid_actions:
            print("Warning: No valid action classes found. Using original dataset.")
            self.resampled_indices = list(range(len(self.base_dataset)))
            return
        
        # 最大サンプル数に合わせてオーバーサンプリング
        max_samples = max(len(indices) for indices in valid_actions.values())
        
        print(f"\n📊 Action class distribution analysis:")
        print("=" * 60)
        action_names = ["Straight", "Left", "Right"]
        
        for action, count in action_counts.items():
            name = action_names[action] if action < len(action_names) else f"Action{action}"
            if count > 0:
                percentage = (count / sum(action_counts.values())) * 100
                bar_length = int(count / max_samples * 40) if max_samples > 0 else 0
                bar = "█" * bar_length + "░" * (40 - bar_length)
                print(f"  {name:8} [{bar}] {count:6d} samples ({percentage:5.1f}%)")
            else:
                print(f"  {name:8} [{'░' * 40}] {count:6d} samples (excluded)")
        
        print("=" * 60)
        print(f"🎯 Target samples per class: {max_samples:,}")
        print(f"⚡ Starting oversampling process...")
        
        self.resampled_indices = []
        
        # 各行動クラスのオーバーサンプリングを進捗表示付きで実行
        with tqdm(total=len(valid_actions) * max_samples, desc="Resampling", unit="samples") as pbar:
            for action, indices in valid_actions.items():
                action_name = action_names[action] if action < len(action_names) else f"Action{action}"
                pbar.set_postfix(action=action_name)
                
                current_count = len(indices)
                resampled_action_indices = indices.copy()
                pbar.update(current_count)
                
                # 不足分を補うためのランダムサンプリング
                additional_needed = max_samples - current_count
                if additional_needed > 0:
                    # バッチ処理で効率化
                    batch_size = min(1000, additional_needed)
                    for i in range(0, additional_needed, batch_size):
                        current_batch = min(batch_size, additional_needed - i)
                        additional_indices = np.random.choice(indices, size=current_batch, replace=True)
                        resampled_action_indices.extend(additional_indices.tolist())
                        pbar.update(current_batch)
                        time.sleep(0.001)  # 進捗表示を見やすくするための小さな遅延
                
                self.resampled_indices.extend(resampled_action_indices)
        
        print("🔀 Shuffling resampled indices...")
        # インデックスをシャッフル
        np.random.shuffle(self.resampled_indices)
        
        # リサンプリング後の統計情報
        print(f"\n✅ Resampling completed successfully!")
        print("=" * 60)
        print(f"📈 Final resampling statistics:")
        print(f"  Original dataset size: {len(self.base_dataset):,} samples")
        print(f"  Resampled dataset size: {len(self.resampled_indices):,} samples")
        print(f"  Expansion ratio: {len(self.resampled_indices) / len(self.base_dataset):.2f}x")
        print(f"  Valid action classes: {len(valid_actions)}")
        print("=" * 60)
    
    def __len__(self):
        return len(self.resampled_indices)
    
    def __getitem__(self, idx):
        # リサンプリングされたインデックスを使用してベースデータセットから取得
        base_idx = self.resampled_indices[idx]
        return self.base_dataset[base_idx]