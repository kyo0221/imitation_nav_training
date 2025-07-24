import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from ament_index_python.packages import get_package_share_directory
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchvision.models as models

from augment.gamma_augment import GammaWrapperDataset
from augment.augmix_augment import AugMixWrapperDataset
from augment.albumentations_augment import AlbumentationsWrapperDataset
from augment.resampling_dataset import ResamplingWrapperDataset
from augment.webdataset_loader import WebDatasetLoader


class Config:
    def __init__(self):
        package_dir = get_package_share_directory('imitation_nav_training')
        config_path = os.path.join(package_dir, 'config', 'train_params.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)['train']

        self.package_dir = os.path.dirname(os.path.realpath(__file__))
        self.result_dir = os.path.join(self.package_dir, '..', 'logs', 'result')
        os.makedirs(self.result_dir, exist_ok=True)

        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.learning_rate = config['learning_rate']
        self.shuffle = config.get('shuffle', True)
        self.image_height = config['image_height']
        self.image_width = config['image_width']
        self.model_filename = config['model_filename']
        self.class_names = [name.strip() for name in config['action_classes'][0].split(',')]
        self.augment_method = config['augment']
        self.resample = config.get('resample', False)
        self.freeze_resnet_backbone = config.get('freeze_resnet_backbone', True)
        self.use_pretrained_resnet = config.get('use_pretrained_resnet', True)
        self.shift_signs = config.get('shift_signs', [-2.0, -1.0, 0.0, 1.0, 2.0])
        self.yaw_signs = config.get('yaw_signs', [0.0])

class AugMixConfig:
    def __init__(self):
        package_dir = get_package_share_directory('imitation_nav_training')
        config_path = os.path.join(package_dir, 'config', 'train_params.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)['augmix']

        self.num_augmented_samples = config['num_augmented_samples']
        self.severity = config['severity']
        self.width = config['width']
        self.depth = config['depth']
        self.alpha = config['alpha']
        self.operations = config['operations']
        self.visualize_image = config['visualize_image']

class GammaConfig:
    def __init__(self):
        package_dir = get_package_share_directory('imitation_nav_training')
        config_path = os.path.join(package_dir, 'config', 'train_params.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)['gamma']

        self.num_augmented_samples = config['num_augmented_samples']
        self.gamma_range = config['gamma_range']
        self.contrast_range = config['contrast_range']
        self.visualize_image = config['visualize_image']

class AlbumentationsConfig:
    def __init__(self):
        package_dir = get_package_share_directory('imitation_nav_training')
        config_path = os.path.join(package_dir, 'config', 'train_params.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)['albumentations']

        self.num_augmented_samples = config['num_augmented_samples']
        self.brightness_limit = config['brightness_limit']
        self.contrast_limit = config['contrast_limit']
        self.saturation_limit = config['saturation_limit']
        self.hue_limit = config['hue_limit']
        self.blur_limit = config['blur_limit']
        self.h_flip_prob = config['h_flip_prob']
        self.visualize_image = config['visualize_image']


class ConditionalAnglePredictor(nn.Module):
    def __init__(self, n_channel, n_out, input_height, input_width, n_action_classes, freeze_resnet_backbone=True, use_pretrained_resnet=True):
        super(ConditionalAnglePredictor, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.dropout_fc = nn.Dropout(p=0.5)

        # ResNet18 backbone
        resnet18 = models.resnet18(pretrained=use_pretrained_resnet)
        if n_channel != 3:
            resnet18.conv1 = nn.Conv2d(n_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 最後の全結合層とAvgPoolを削除してバックボーンを取得
        self.resnet_backbone = nn.Sequential(*list(resnet18.children())[:-2])

        # Display ResNet initialization mode
        if use_pretrained_resnet:
            print("📦 Using pretrained ResNet18 weights from ImageNet")
        else:
            print("🌱 Initializing ResNet18 from scratch (no pretrained weights)")
        
        # Freeze ResNet backbone if specified
        if freeze_resnet_backbone:
            for param in self.resnet_backbone.parameters():
                param.requires_grad = False
            print("🔒 ResNet backbone parameters frozen. Only MLP layers will be trained.")
        else:
            print("🔓 ResNet backbone parameters trainable. Full model will be trained.")
        
        # ResNet18の出力次元を計算
        with torch.no_grad():
            dummy_input = torch.zeros(1, n_channel, input_height, input_width)
            x = self.resnet_backbone(dummy_input)
            x = nn.AdaptiveAvgPool2d((1, 1))(x)
            x = self.flatten(x)
            resnet_features = x.shape[1]  # ResNet18では512
        
        # MLP部分（ResNet18の出力次元に合わせて）
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(resnet_features, 512)
        self.fc2 = nn.Linear(512, 512)

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 256),
                self.relu,
                nn.Linear(256, n_out)
            ) for _ in range(n_action_classes)
        ])

        self.cnn_layer = nn.Sequential(
            self.resnet_backbone,
            self.adaptive_pool,
            self.flatten
        )

        # LSTMは維持
        self.lstm = nn.LSTM(input_size=512, hidden_size=512, num_layers=1, batch_first=True)


    def forward(self, image, action_onehot):
        # ResNet18で特徴抽出
        features = self.cnn_layer(image)
        x = self.relu(self.fc1(features))
        x = self.dropout_fc(x)
        fc_out = self.relu(self.fc2(x))

        # 条件付き模倣学習のブランチ処理（変更なし）
        batch_size = image.size(0)
        action_indices = torch.argmax(action_onehot, dim=1)

        output = torch.zeros(batch_size, self.branches[0][-1].out_features, device=image.device, dtype=fc_out.dtype)
        for idx, branch in enumerate(self.branches):
            selected_idx = (action_indices == idx).nonzero().squeeze(1)
            if selected_idx.numel() > 0:
                output[selected_idx] = branch(fc_out[selected_idx])

        return output

class Training:
    def __init__(self, config, dataset):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.loader = DataLoader(dataset, batch_size=config.batch_size, num_workers=os.cpu_count() // 20, pin_memory=True, shuffle=config.shuffle)
        self.model = ConditionalAnglePredictor(3, 1, config.image_height, config.image_width, len(config.class_names), config.freeze_resnet_backbone, config.use_pretrained_resnet).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.writer = SummaryWriter(log_dir=config.result_dir)
        self.loss_log = []

    def train(self):
        scaler = torch.cuda.amp.GradScaler()
        torch.backends.cudnn.benchmark = True

        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            batch_iter = tqdm(self.loader, desc=f"Epoch {epoch+1}/{self.config.epochs}", leave=False)
            for i, batch in enumerate(batch_iter):
                images, action_onehots, targets = [x.to(self.device) for x in batch]

                self.optimizer.zero_grad()

                with torch.cuda.amp.autocast():
                    preds = self.model(images, action_onehots)
                    loss = self.criterion(preds, targets)

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                self.loss_log.append(loss.item())
                epoch_loss += loss.item()
                batch_iter.set_postfix(loss=loss.item())

            avg_loss = epoch_loss / len(self.loader)
            self.writer.add_scalar('Loss/epoch_avg', avg_loss, epoch)
            self.writer.flush()

            # Save model every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_intermediate_model(epoch + 1)

        self.save_results()
        self.writer.close()

    def save_intermediate_model(self, epoch):
        scripted_model = torch.jit.script(self.model)
        base_filename = os.path.splitext(self.config.model_filename)[0]
        extension = os.path.splitext(self.config.model_filename)[1]
        intermediate_filename = f"{base_filename}_{epoch}ep{extension}"
        scripted_path = os.path.join(self.config.result_dir, intermediate_filename)
        scripted_model.save(scripted_path)
        print(f"🐜 中間モデルを保存しました: {scripted_path}")

    def save_results(self):
        scripted_model = torch.jit.script(self.model)
        scripted_path = os.path.join(self.config.result_dir, self.config.model_filename)
        scripted_model.save(scripted_path)
        print(f"🐜 学習済みモデルを保存しました: {scripted_path}")

        plt.figure()
        plt.plot(self.loss_log)
        plt.title("Training Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.savefig(os.path.join(self.config.result_dir, 'loss_curve.png'))
        print("📈 学習曲線を保存しました")


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help='Path to dataset directory (contains webdataset/)')
    parser.add_argument('visualize_dir', nargs='?', default=None, help='Optional directory to save visualized samples')
    args = parser.parse_args()

    config = Config()
    dataset_dir = args.dataset

    # WebDatasetを使用
    webdataset_dir = os.path.join(dataset_dir, 'webdataset')
    if not os.path.exists(webdataset_dir):
        raise ValueError(f"WebDataset directory not found: {webdataset_dir}")
    
    # 拡張なしのベースデータセットを作成（生データセットサイズ用）
    raw_dataset = WebDatasetLoader(
        dataset_dir=webdataset_dir,
        input_size=(config.image_height, config.image_width),
        shift_aug=False,  # 拡張なし
        yaw_aug=False,    # 拡張なし
        shift_offset=5,
        vel_offset=0.2,
        n_action_classes=len(config.class_names),
        shift_signs=config.shift_signs,
        yaw_signs=config.yaw_signs,
        visualize_dir=None
    )
    
    # データ拡張ありのベースデータセットを作成
    base_dataset = WebDatasetLoader(
        dataset_dir=webdataset_dir,
        input_size=(config.image_height, config.image_width),
        shift_offset=5,
        vel_offset=0.2,
        n_action_classes=len(config.class_names),
        shift_signs=config.shift_signs,
        yaw_signs=config.yaw_signs,
        visualize_dir=args.visualize_dir
    )
    
    print(f"Using WebDataset from: {webdataset_dir}")
    print(f"Raw dataset size (no augmentation): {len(raw_dataset)} samples")
    print(f"Base dataset size (with shift/yaw augmentation): {len(base_dataset)} samples")

    if config.augment_method == "gamma":
        gamma_config = GammaConfig()
        dataset = GammaWrapperDataset(
            base_dataset=base_dataset,
            gamma_range=gamma_config.gamma_range,
            contrast_range=gamma_config.contrast_range,
            num_augmented_samples=gamma_config.num_augmented_samples,
            visualize=gamma_config.visualize_image,
            visualize_dir=os.path.join(config.result_dir, "gamma")
        )
    elif config.augment_method == "augmix":
        augmix_config = AugMixConfig()
        dataset = AugMixWrapperDataset(
            base_dataset=base_dataset,
            num_augmented_samples=augmix_config.num_augmented_samples,
            severity=augmix_config.severity,
            width=augmix_config.width,
            depth=augmix_config.depth,
            allowed_ops=augmix_config.operations,
            alpha=augmix_config.alpha,
            visualize=augmix_config.visualize_image,
            visualize_dir=os.path.join(config.result_dir, "augmix")
        )
    elif config.augment_method == "albumentations":
        albumentations_config = AlbumentationsConfig()
        dataset = AlbumentationsWrapperDataset(
            base_dataset=base_dataset,
            num_augmented_samples=albumentations_config.num_augmented_samples,
            brightness_limit=albumentations_config.brightness_limit,
            contrast_limit=albumentations_config.contrast_limit,
            saturation_limit=albumentations_config.saturation_limit,
            hue_limit=albumentations_config.hue_limit,
            blur_limit=albumentations_config.blur_limit,
            h_flip_prob=albumentations_config.h_flip_prob,
            visualize=albumentations_config.visualize_image,
            visualize_dir=os.path.join(config.result_dir, "albumentations")
        )
    elif config.augment_method in ["none", "None"]:
        dataset = base_dataset
    else:
        raise ValueError(f"Unknown augmentation method: {config.augment_method}")

    print(f"Dataset size after {config.augment_method} augmentation: {len(dataset)} samples")
    
    # データセットサイズの比較表示
    raw_size = len(raw_dataset)
    base_size = len(base_dataset)
    augmented_size = len(dataset)
    
    print("\n" + "="*50)
    print("DATASET SIZE SUMMARY")
    print("="*50)
    print(f"Raw dataset (no augmentation):      {raw_size:,} samples")
    print(f"Base dataset (shift/yaw aug):       {base_size:,} samples (x{base_size/raw_size:.1f})")
    print(f"After {config.augment_method} augmentation:        {augmented_size:,} samples (x{augmented_size/raw_size:.1f})")
    print("="*50)

    if config.resample:
        print("Applying action class resampling...")
        pre_resample_size = len(dataset)
        dataset = ResamplingWrapperDataset(
            base_dataset=dataset,
            n_action_classes=len(config.class_names),
            enable_resampling=True
        )
        post_resample_size = len(dataset)
        print(f"Resampling: {pre_resample_size:,} -> {post_resample_size:,} samples")
    else:
        print("Resampling disabled.")

    final_size = len(dataset)
    print(f"Final dataset size after resampling: {final_size:,} samples")
    
    # 最終的なデータセットサイズの比較
    if config.resample:
        print(f"Total augmentation factor: x{final_size/raw_size:.1f} (from {raw_size:,} to {final_size:,} samples)")
    else:
        print(f"Total augmentation factor: x{augmented_size/raw_size:.1f} (from {raw_size:,} to {augmented_size:,} samples)")
    print("")

    trainer = Training(config, dataset)
    trainer.train()
