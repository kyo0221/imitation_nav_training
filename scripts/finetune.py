import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, ConcatDataset
from ament_index_python.packages import get_package_share_directory
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from augment.gamma_augment import create_gamma_augmented_dataset
from augment.augmix_augment import create_augmix_augmented_dataset
from augment.albumentations_augment import create_albumentations_augmented_dataset
from augment.imitation_dataset import ImitationDataset
from augment.resampling_dataset import ResamplingWrapperDataset


class FineTuneConfig:
    def __init__(self):
        package_dir = get_package_share_directory('imitation_nav_training')
        config_path = os.path.join(package_dir, 'config', 'finetune_params.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)['finetune']

        self.package_dir = os.path.dirname(os.path.realpath(__file__))
        self.result_dir = os.path.join(self.package_dir, '..', 'logs', 'finetune_result')
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
        self.freeze_layers = config.get('freeze_layers', [])
        self.learning_rate_scheduler = config.get('learning_rate_scheduler', False)
        self.weight_decay = config.get('weight_decay', 0.0)
        self.new_skill_ratio = config.get('new_skill_ratio', 3.0)

class AugMixConfig:
    def __init__(self):
        package_dir = get_package_share_directory('imitation_nav_training')
        config_path = os.path.join(package_dir, 'config', 'finetune_params.yaml')
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
        config_path = os.path.join(package_dir, 'config', 'finetune_params.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)['gamma']

        self.num_augmented_samples = config['num_augmented_samples']
        self.gamma_range = config['gamma_range']
        self.contrast_range = config['contrast_range']
        self.visualize_image = config['visualize_image']

class AlbumentationsConfig:
    def __init__(self):
        package_dir = get_package_share_directory('imitation_nav_training')
        config_path = os.path.join(package_dir, 'config', 'finetune_params.yaml')
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
    def __init__(self, n_channel, n_out, input_height, input_width, n_action_classes):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.dropout_conv = nn.Dropout2d(p=0.2)
        self.dropout_fc = nn.Dropout(p=0.5)

        def conv_block(in_channels, out_channels, kernel_size, stride, apply_bn=True):
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2),
                nn.BatchNorm2d(out_channels) if apply_bn else nn.Identity(),
                self.relu,
                self.dropout_conv
            ]
            return nn.Sequential(*layers)

        self.conv1 = conv_block(n_channel, 32, kernel_size=5, stride=2)
        self.conv2 = conv_block(32, 48, kernel_size=3, stride=1)
        self.conv3 = conv_block(48, 64, kernel_size=3, stride=2)
        self.conv4 = conv_block(64, 96, kernel_size=3, stride=1)
        self.conv5 = conv_block(96, 128, kernel_size=3, stride=2)
        self.conv6 = conv_block(128, 160, kernel_size=3, stride=1)
        self.conv7 = conv_block(160, 192, kernel_size=3, stride=1)
        self.conv8 = conv_block(192, 256, kernel_size=3, stride=1)

        with torch.no_grad():
            dummy_input = torch.zeros(1, n_channel, input_height, input_width)
            x = self.conv1(dummy_input)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.conv5(x)
            x = self.conv6(x)
            x = self.conv7(x)
            x = self.conv8(x)
            x = self.flatten(x)
            flattened_size = x.shape[1]

        self.fc1 = nn.Linear(flattened_size, 512)
        self.fc2 = nn.Linear(512, 512)

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 256),
                self.relu,
                nn.Linear(256, n_out)
            ) for _ in range(n_action_classes)
        ])

        self.cnn_layer = nn.Sequential(
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4,
            self.conv5,
            self.conv6,
            self.conv7,
            self.conv8,
            self.flatten
        )

    def forward(self, image, action_onehot):
        features = self.cnn_layer(image)
        x = self.relu(self.fc1(features))
        x = self.dropout_fc(x)
        fc_out = self.relu(self.fc2(x))

        batch_size = image.size(0)
        action_indices = torch.argmax(action_onehot, dim=1)

        output = torch.zeros(batch_size, self.branches[0][-1].out_features, device=image.device, dtype=fc_out.dtype)
        for idx, branch in enumerate(self.branches):
            selected_idx = (action_indices == idx).nonzero().squeeze(1)
            if selected_idx.numel() > 0:
                output[selected_idx] = branch(fc_out[selected_idx])

        return output

class WeightedConcatDataset(ConcatDataset):
    def __init__(self, datasets, weights):
        super().__init__(datasets)
        self.weights = weights
        self.weighted_lengths = [int(len(d) * w) for d, w in zip(datasets, weights)]
        
        # Create weighted indices
        self.weighted_indices = []
        for i, (dataset, weighted_len) in enumerate(zip(datasets, self.weighted_lengths)):
            original_len = len(dataset)
            # Repeat indices to match weighted length
            dataset_indices = [j for j in range(original_len)] * (weighted_len // original_len + 1)
            dataset_indices = dataset_indices[:weighted_len]  # Trim to exact weighted length
            # Add dataset offset
            offset = sum(len(d) for d in datasets[:i])
            weighted_dataset_indices = [idx + offset for idx in dataset_indices]
            self.weighted_indices.extend(weighted_dataset_indices)
    
    def __len__(self):
        return len(self.weighted_indices)
    
    def __getitem__(self, idx):
        actual_idx = self.weighted_indices[idx]
        return super().__getitem__(actual_idx)

class FineTuning:
    def __init__(self, config, dataset, pretrained_model_path):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.loader = DataLoader(dataset, batch_size=config.batch_size, num_workers=os.cpu_count() // 20, pin_memory=True, shuffle=config.shuffle)
        
        # Load pretrained model
        self.model = torch.jit.load(pretrained_model_path, map_location=self.device)
        self.model.train()
        
        # Freeze specified layers
        self._freeze_layers()
        
        self.criterion = nn.MSELoss()
        
        # Get parameters that require gradients
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(trainable_params, lr=config.learning_rate, weight_decay=config.weight_decay)
        
        # Learning rate scheduler
        if config.learning_rate_scheduler:
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        else:
            self.scheduler = None
            
        self.writer = SummaryWriter(log_dir=config.result_dir)
        self.loss_log = []

    def _freeze_layers(self):
        for name, param in self.model.named_parameters():
            param.requires_grad = True
            
        for layer_name in self.config.freeze_layers:
            for name, param in self.model.named_parameters():
                if layer_name in name:
                    param.requires_grad = False
                    print(f"🔒 Frozen layer: {name}")

    def train(self):
        scaler = torch.cuda.amp.GradScaler()
        torch.backends.cudnn.benchmark = True

        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            batch_iter = tqdm(self.loader, desc=f"FineTune Epoch {epoch+1}/{self.config.epochs}", leave=False)
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
            self.writer.add_scalar('FineTune_Loss/epoch_avg', avg_loss, epoch)
            
            if self.scheduler:
                self.scheduler.step()
                self.writer.add_scalar('Learning_Rate', self.scheduler.get_last_lr()[0], epoch)
                
            self.writer.flush()

            # Save model every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_intermediate_model(epoch + 1)

        self.save_results()
        self.writer.close()

    def save_intermediate_model(self, epoch):
        base_filename = os.path.splitext(self.config.model_filename)[0]
        extension = os.path.splitext(self.config.model_filename)[1]
        intermediate_filename = f"{base_filename}_finetune_{epoch}ep{extension}"
        scripted_path = os.path.join(self.config.result_dir, intermediate_filename)
        self.model.save(scripted_path)
        print(f"🦋 中間ファインチューニングモデルを保存しました: {scripted_path}")

    def save_results(self):
        scripted_path = os.path.join(self.config.result_dir, self.config.model_filename)
        self.model.save(scripted_path)
        print(f"🦋 ファインチューニング済みモデルを保存しました: {scripted_path}")

        plt.figure()
        plt.plot(self.loss_log)
        plt.title("Fine-tuning Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.savefig(os.path.join(self.config.result_dir, 'finetune_loss_curve.png'))
        print("📈 ファインチューニング学習曲線を保存しました")


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument('pretrained_model', type=str, help='Path to pretrained model file (.pt)')
    parser.add_argument('new_skill_dataset', type=str, help='Path to new skill dataset directory (contains images/, angle/, action/)')
    parser.add_argument('original_dataset', type=str, help='Path to original dataset directory (contains images/, angle/, action/)')
    parser.add_argument('visualize_dir', nargs='?', default=None, help='Optional directory to save visualized samples')
    args = parser.parse_args()

    config = FineTuneConfig()
    pretrained_model_path = args.pretrained_model
    new_skill_dataset_dir = args.new_skill_dataset
    original_dataset_dir = args.original_dataset

    # Create new skill dataset (higher priority)
    new_skill_base_dataset = ImitationDataset(
        dataset_dir=new_skill_dataset_dir,
        input_size=(config.image_height, config.image_width),
        shift_offset=5,
        vel_offset=0.2,
        n_action_classes=len(config.class_names),
        visualize_dir=os.path.join(args.visualize_dir, 'new_skill') if args.visualize_dir else None
    )
    
    # Create original dataset (for maintaining existing skills)
    original_base_dataset = ImitationDataset(
        dataset_dir=original_dataset_dir,
        input_size=(config.image_height, config.image_width),
        shift_offset=5,
        vel_offset=0.2,
        n_action_classes=len(config.class_names),
        visualize_dir=os.path.join(args.visualize_dir, 'original') if args.visualize_dir else None
    )

    # Apply data augmentation to both datasets
    def apply_augmentation(base_dataset, dataset_name):
        if config.augment_method == "gamma":
            gamma_config = GammaConfig()
            return create_gamma_augmented_dataset(
                base_dataset=base_dataset,
                gamma_range=gamma_config.gamma_range,
                contrast_range=gamma_config.contrast_range,
                num_augmented_samples=gamma_config.num_augmented_samples,
                visualize=gamma_config.visualize_image,
                visualize_dir=os.path.join(config.result_dir, f"gamma_{dataset_name}")
            )
        elif config.augment_method == "augmix":
            augmix_config = AugMixConfig()
            return create_augmix_augmented_dataset(
                base_dataset=base_dataset,
                num_augmented_samples=augmix_config.num_augmented_samples,
                severity=augmix_config.severity,
                width=augmix_config.width,
                depth=augmix_config.depth,
                allowed_ops=augmix_config.operations,
                alpha=augmix_config.alpha,
                visualize=augmix_config.visualize_image,
                visualize_dir=os.path.join(config.result_dir, f"augmix_{dataset_name}")
            )
        elif config.augment_method == "albumentations":
            albumentations_config = AlbumentationsConfig()
            return create_albumentations_augmented_dataset(
                base_dataset=base_dataset,
                num_augmented_samples=albumentations_config.num_augmented_samples,
                brightness_limit=albumentations_config.brightness_limit,
                contrast_limit=albumentations_config.contrast_limit,
                saturation_limit=albumentations_config.saturation_limit,
                hue_limit=albumentations_config.hue_limit,
                blur_limit=albumentations_config.blur_limit,
                h_flip_prob=albumentations_config.h_flip_prob,
                visualize=albumentations_config.visualize_image,
                visualize_dir=os.path.join(config.result_dir, f"albumentations_{dataset_name}")
            )
        elif config.augment_method in ["none", "None"]:
            return base_dataset
        else:
            raise ValueError(f"Unknown augmentation method: {config.augment_method}")
    
    new_skill_dataset = apply_augmentation(new_skill_base_dataset, "new_skill")
    original_dataset = apply_augmentation(original_base_dataset, "original")

    print(f"New skill dataset size (after rotate_aug): {len(new_skill_base_dataset)} samples")
    print(f"New skill dataset size after {config.augment_method} augmentation: {len(new_skill_dataset)} samples")
    print(f"Original dataset size (after rotate_aug): {len(original_base_dataset)} samples")
    print(f"Original dataset size after {config.augment_method} augmentation: {len(original_dataset)} samples")

    # Apply resampling if enabled
    if config.resample:
        print("Applying action class resampling...")
        new_skill_dataset = ResamplingWrapperDataset(
            base_dataset=new_skill_dataset,
            n_action_classes=len(config.class_names),
            enable_resampling=True
        )
        original_dataset = ResamplingWrapperDataset(
            base_dataset=original_dataset,
            n_action_classes=len(config.class_names),
            enable_resampling=True
        )
    else:
        print("Resampling disabled.")

    # Create weighted combined dataset (new_skill_ratio times more new skill data)
    weights = [config.new_skill_ratio, 1.0]  # [new_skill_weight, original_weight]
    dataset = WeightedConcatDataset([new_skill_dataset, original_dataset], weights)
    
    print(f"New skill dataset final size: {len(new_skill_dataset)} samples")
    print(f"Original dataset final size: {len(original_dataset)} samples")
    print(f"Combined weighted dataset size: {len(dataset)} samples")
    print(f"New skill data ratio: {config.new_skill_ratio}:1 (new_skill:original)")

    trainer = FineTuning(config, dataset, pretrained_model_path)
    trainer.train()