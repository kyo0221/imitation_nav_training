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

from augment.gamma_augment import GammaWrapperDataset
from augment.augmix_augment import AugMixWrapperDataset
from augment.albumentations_augment import AlbumentationsWrapperDataset
from augment.imitation_dataset import ImitationDataset
import timm


class Config:
    def __init__(self):
        package_dir = get_package_share_directory('imitation_nav_training')
        config_path = os.path.join(package_dir, 'config', 'train_params.yaml')
        with open(config_path, 'r') as f:
            full_config = yaml.safe_load(f)
            config = full_config['train']
            aug_config = full_config.get('augmentation', {})

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
        self.augment_method = config['augment']
        
        # データ拡張パラメータ
        self.shift_signs = aug_config.get('shift_signs', [0.0])
        self.yaw_signs = aug_config.get('yaw_signs', [0.0])
        self.shift_offset = aug_config.get('shift_offset', 5)
        self.vel_offset = aug_config.get('vel_offset', 0.2)
        self.yaw_base_deg = aug_config.get('yaw_base_deg', 5.0)
        self.angular_offset_per_deg = aug_config.get('angular_offset_per_deg', 0.04)

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


class ViTAnglePredictor(nn.Module):
    def __init__(self, n_channel, n_out, input_height, input_width):
        super().__init__()
        # ImageNet事前学習済みViT-Base（特徴抽出用、分類ヘッドなし）
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        vit_features = 768  # ViT-Baseの特徴次元
        
        # 回帰用のヘッド
        self.regression_head = nn.Sequential(
            nn.LayerNorm(vit_features),
            nn.Linear(vit_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, n_out)
        )

    def forward(self, image):
        features = self.vit(image)
        return self.regression_head(features)

class Training:
    def __init__(self, config, dataset):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.loader = DataLoader(dataset, batch_size=config.batch_size, num_workers=os.cpu_count() // 20, pin_memory=True, shuffle=config.shuffle)
        self.model = ViTAnglePredictor(3, 1, config.image_height, config.image_width).to(self.device)
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
                images, targets = [x.to(self.device) for x in batch]

                self.optimizer.zero_grad()

                with torch.cuda.amp.autocast():
                    preds = self.model(images)
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
    parser.add_argument('dataset', type=str, help='Path to dataset directory (contains images/, angle/, action/)')
    parser.add_argument('visualize_dir', nargs='?', default=None, help='Optional directory to save visualized samples')
    args = parser.parse_args()

    config = Config()
    dataset_dir = args.dataset

    base_dataset = ImitationDataset(
        dataset_dir=dataset_dir,
        input_size=(config.image_height, config.image_width),
        shift_signs=config.shift_signs,
        yaw_signs=config.yaw_signs,
        shift_offset=config.shift_offset,
        vel_offset=config.vel_offset,
        yaw_base_deg=config.yaw_base_deg,
        angular_offset_per_deg=config.angular_offset_per_deg,
        visualize_dir=args.visualize_dir
    )

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

    print(f"Base dataset size (after rotate_aug): {len(base_dataset)} samples")
    print(f"Dataset size after {config.augment_method} augmentation: {len(dataset)} samples")

    trainer = Training(config, dataset)
    trainer.train()
