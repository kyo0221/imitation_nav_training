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

from augment.gamma_augment import create_gamma_augmented_dataset
from augment.augmix_augment import create_augmix_augmented_dataset
from augment.albumentations_augment import create_albumentations_augmented_dataset
from augment.resampling_dataset import ResamplingWrapperDataset
from augment.webdataset_loader import create_webdataset_streaming_loader


class BaseConfig:
    def __init__(self, config_section):
        package_dir = get_package_share_directory('imitation_nav_training')
        config_path = os.path.join(package_dir, 'config', 'train_params.yaml')
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)[config_section]


class Config:
    def __init__(self):
        base = BaseConfig('train')
        config = base.config

        self.package_dir = os.path.dirname(os.path.realpath(__file__))
        self.result_dir = os.path.join(self.package_dir, '..', 'logs', 'result')
        os.makedirs(self.result_dir, exist_ok=True)

        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.learning_rate = config['learning_rate']
        self.image_height = config['image_height']
        self.image_width = config['image_width']
        self.model_filename = config['model_filename']
        self.class_names = [name.strip() for name in config['action_classes'][0].split(',')]
        self.augment_method = config['augment']
        self.freeze_resnet_backbone = config.get('freeze_resnet_backbone', True)
        self.use_pretrained_resnet = config.get('use_pretrained_resnet', True)
        self.shift_signs = config.get('shift_signs', [-2.0, -1.0, 0.0, 1.0, 2.0])
        self.resample = config.get('resample', False)


class ConditionalAnglePredictor(nn.Module):
    def __init__(self, n_channel, n_out, input_height, input_width, n_action_classes, freeze_resnet_backbone=True, use_pretrained_resnet=True):
        super(ConditionalAnglePredictor, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.dropout_fc = nn.Dropout(p=0.5)

        resnet18 = models.resnet18(pretrained=use_pretrained_resnet)
        if n_channel != 3:
            resnet18.conv1 = nn.Conv2d(n_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.resnet_backbone = nn.Sequential(*list(resnet18.children())[:-2])

        print(f"📦 ResNet18 {'pretrained' if use_pretrained_resnet else 'from scratch'}")
        
        if freeze_resnet_backbone:
            for param in self.resnet_backbone.parameters():
                param.requires_grad = False
            print("🔒 ResNet backbone parameters frozen. Only MLP layers will be trained.")
        else:
            print("🔓 ResNet backbone parameters trainable. Full model will be trained.")
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, n_channel, input_height, input_width)
            x = self.resnet_backbone(dummy_input)
            x = nn.AdaptiveAvgPool2d((1, 1))(x)
            x = self.flatten(x)
            resnet_features = x.shape[1]
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(resnet_features, 512)
        self.fc2 = nn.Linear(512, 512)

        self.branches = nn.ModuleList([
            nn.Sequential(nn.Linear(512, 256), self.relu, nn.Linear(256, n_out))
            for _ in range(n_action_classes)
        ])


    def forward(self, image, action_onehot):
        features = self.flatten(self.adaptive_pool(self.resnet_backbone(image)))
        x = self.relu(self.fc1(features))
        x = self.dropout_fc(x)
        fc_out = self.relu(self.fc2(x))
        
        batch_size = image.size(0)
        action_indices = torch.argmax(action_onehot, dim=1)
        output = torch.zeros(batch_size, self.branches[0][-1].out_features, 
                           device=image.device, dtype=fc_out.dtype)
        
        for idx, branch in enumerate(self.branches):
            mask = (action_indices == idx)
            if mask.any():
                output[mask] = branch(fc_out[mask])
        
        return output

class Training:
    def __init__(self, config, dataset):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.loader = DataLoader(dataset, batch_size=config.batch_size, 
                               num_workers=18, pin_memory=True, shuffle=False)
        self.model = ConditionalAnglePredictor(
            3, 1, config.image_height, config.image_width, 
            len(config.class_names), config.freeze_resnet_backbone, 
            config.use_pretrained_resnet).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.writer = SummaryWriter(log_dir=config.result_dir)
        self.loss_log = []

    def _save_model(self, filename):
        torch.jit.script(self.model).save(os.path.join(self.config.result_dir, filename))

    def train(self):
        scaler = torch.cuda.amp.GradScaler()
        torch.backends.cudnn.benchmark = True

        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            pbar = tqdm(desc=f"🚀 Epoch {epoch+1}/{self.config.epochs}", unit="batch")
            
            for batch in self.loader:
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
                batch_count += 1
                
                pbar.update(1)
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            pbar.close()

            if batch_count > 0:
                avg_loss = epoch_loss / batch_count
                self.writer.add_scalar('Loss/epoch_avg', avg_loss, epoch)
                self.writer.flush()
                print(f"📊 Epoch {epoch+1}/{self.config.epochs} - Avg Loss: {avg_loss:.4f}")

            if (epoch + 1) % 10 == 0:
                filename = f"{os.path.splitext(self.config.model_filename)[0]}_{epoch+1}ep.pt"
                self._save_model(filename)
                print(f"🐜 中間モデル保存: {filename}")

        self.save_results()
        self.writer.close()

    def save_results(self):
        self._save_model(self.config.model_filename)
        print(f"🐜 学習済みモデル保存: {self.config.model_filename}")

        plt.figure()
        plt.plot(self.loss_log)
        plt.title("Training Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.savefig(os.path.join(self.config.result_dir, 'loss_curve.png'))
        print("📈 学習曲線を保存しました")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help='Path to dataset directory (contains webdataset/)')
    parser.add_argument('visualize_dir', nargs='?', default=None, help='Optional directory to save visualized samples')
    args = parser.parse_args()

    config = Config()
    webdataset_dir = os.path.join(args.dataset, 'webdataset')
    if not os.path.exists(webdataset_dir):
        raise ValueError(f"WebDataset directory not found: {webdataset_dir}")
    
    base_dataset = create_webdataset_streaming_loader(
        dataset_dir=webdataset_dir,
        input_size=(config.image_height, config.image_width),
        vel_offset=0.2,
        n_action_classes=len(config.class_names),
        shift_signs=config.shift_signs,
        visualize_dir=args.visualize_dir
    )
    
    print(f"Dataset loaded from: {webdataset_dir}, augmentation: {config.augment_method}")

    # Augmentation factory
    augment_configs = {
        'gamma': BaseConfig('gamma').config,
        'augmix': BaseConfig('augmix').config,
    }
    
    # albumentationsセクションが存在する場合のみ追加
    try:
        augment_configs['albumentations'] = BaseConfig('albumentations').config
    except KeyError:
        pass  # albumentationsセクションが設定ファイルにない場合はスキップ

    if config.augment_method == "gamma":
        cfg = augment_configs['gamma']
        dataset = create_gamma_augmented_dataset(
            base_dataset=base_dataset,
            gamma_range=cfg['gamma_range'],
            contrast_range=cfg['contrast_range'],
            num_augmented_samples=cfg['num_augmented_samples'],
            visualize=cfg['visualize_image'],
            visualize_dir=os.path.join(config.result_dir, "gamma")
        )
    elif config.augment_method == "augmix":
        cfg = augment_configs['augmix']
        dataset = create_augmix_augmented_dataset(
            base_dataset=base_dataset,
            num_augmented_samples=cfg['num_augmented_samples'],
            severity=cfg['severity'],
            width=cfg['width'],
            depth=cfg['depth'],
            allowed_ops=cfg['operations'],
            alpha=cfg['alpha'],
            visualize=cfg['visualize_image'],
            visualize_dir=os.path.join(config.result_dir, "augmix")
        )
    elif config.augment_method == "albumentations":
        cfg = augment_configs['albumentations']
        dataset = create_albumentations_augmented_dataset(
            base_dataset=base_dataset,
            num_augmented_samples=cfg['num_augmented_samples'],
            brightness_limit=cfg['brightness_limit'],
            contrast_limit=cfg['contrast_limit'],
            saturation_limit=cfg['saturation_limit'],
            hue_limit=cfg['hue_limit'],
            blur_limit=cfg['blur_limit'],
            h_flip_prob=cfg['h_flip_prob'],
            visualize=cfg['visualize_image'],
            visualize_dir=os.path.join(config.result_dir, "albumentations")
        )
    elif config.augment_method in ["none", "None"]:
        dataset = base_dataset
    else:
        raise ValueError(f"Unknown augmentation method: {config.augment_method}")

    print(f"Final dataset: {len(dataset):,} samples" if hasattr(dataset, '__len__') else "Using streaming dataset")

    if config.resample:
        dataset = ResamplingWrapperDataset(
            base_dataset=dataset,
            n_action_classes=len(config.class_names),
            enable_resampling=True
        )
        print(f"With resampling: {len(dataset):,} samples")

    trainer = Training(config, dataset)
    trainer.train()
