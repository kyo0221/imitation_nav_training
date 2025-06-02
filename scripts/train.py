import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from ament_index_python.packages import get_package_share_directory

from augment.gamma_augment import GammaWrapperDataset
from augment.augmix_augment import AugMixWrapperDataset
from augment.imitation_dataset import ImitationDataset


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
        self.visualize_image = config['visualize_image']


class ConditionalAnglePredictor(nn.Module):
    def __init__(self, n_channel, n_out, input_height, input_width, n_action_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channel, 32, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=2)

        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()

        with torch.no_grad():
            dummy_input = torch.zeros(1, n_channel, input_height, input_width)
            x = self.relu(self.conv1(dummy_input))
            x = self.relu(self.conv2(x))
            x = self.relu(self.conv3(x))
            x = self.relu(self.conv4(x))
            x = self.relu(self.conv5(x))
            x = self.flatten(x)
            flattened_size = x.shape[1]

        self.fc4 = nn.Linear(flattened_size, 512)
        self.fc5 = nn.Linear(512, 512)
        self.fc6 = nn.Linear(512, 512)

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 256),
                self.relu,
                nn.Linear(256, n_out)
            ) for _ in range(n_action_classes)
        ])

        self.cnn_layer = nn.Sequential(
            self.conv1, self.relu,
            self.conv2, self.relu,
            self.conv3, self.relu,
            self.conv4, self.relu,
            self.conv5, self.relu,
            self.flatten
        )

    def forward(self, image, action_onehot):
        features = self.cnn_layer(image)
        x = self.relu(self.fc4(features))
        x = self.relu(self.fc5(x))
        fc_out = self.relu(self.fc6(x))

        batch_size = image.size(0)
        action_indices = torch.argmax(action_onehot, dim=1)

        output = torch.zeros(batch_size, self.branches[0][-1].out_features, device=image.device)
        for idx, branch in enumerate(self.branches):
            selected_idx = (action_indices == idx).nonzero().squeeze(1)
            if selected_idx.numel() > 0:
                output[selected_idx] = branch(fc_out[selected_idx])

        return output


class Training:
    def __init__(self, config, dataset):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=config.shuffle)
        self.model = ConditionalAnglePredictor(3, 1, config.image_height, config.image_width, len(config.class_names)).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.loss_log = []

    def train(self):
        total_batches = len(self.loader) * self.config.epochs
        current_batch = 0

        for epoch in range(self.config.epochs):
            for batch in self.loader:
                images, action_onehots, targets = [x.to(self.device) for x in batch]
                preds = self.model(images, action_onehots)
                loss = self.criterion(preds, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.loss_log.append(loss.item())
                current_batch += 1
                progress = (current_batch / total_batches) * 100
                print(f"Epoch {epoch+1}/{self.config.epochs}, Loss: {loss.item():.4f}, Progress: {progress:.1f}%")

        self.save_results()

    def save_results(self):
        scripted_model = torch.jit.script(self.model)
        scripted_path = os.path.join(self.config.result_dir, self.config.model_filename)
        scripted_model.save(scripted_path)
        print(f"\U0001f41c 学習済みモデルを保存しました: {scripted_path}")

        plt.figure()
        plt.plot(self.loss_log)
        plt.title("Training Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.savefig(os.path.join(self.config.result_dir, 'loss_curve.png'))
        print("\U0001f4c8 学習曲線を保存しました")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help='Path to dataset directory (contains images/, angle/, action/)')
    args = parser.parse_args()

    config = Config()
    dataset_dir = args.dataset

    dataset = ImitationDataset(
        dataset_dir=dataset_dir,
        input_size=(config.image_height, config.image_width),
        rotate_aug=True,
        angle_offset_deg=5,
        vel_offset=0.2,
        n_action_classes=len(config.class_names)
    )

    if config.augment_method == "gamma":
        gamma_config = GammaConfig()  # 共通で使うならこのクラス名でもOK
        dataset = GammaWrapperDataset(
            base_dataset=dataset,
            gamma_range=gamma_config.gamma_range,
            num_augmented_samples=gamma_config.num_augmented_samples,
            visualize=gamma_config.visualize_image,
            visualize_dir=os.path.join(config.result_dir, "gamma")
        )
    elif config.augment_method == "augmix":
        augmix_config = AugMixConfig()
        dataset = AugMixWrapperDataset(
            base_dataset=dataset,
            num_augmented_samples=augmix_config.num_augmented_samples,
            severity=augmix_config.severity,
            width=augmix_config.width,
            depth=augmix_config.depth,
            allowed_ops=augmix_config.operations,
            alpha=augmix_config.alpha,
            visualize=augmix_config.visualize_image,
            visualize_dir=os.path.join(config.result_dir, "augmix")
        )
    elif config.augment_method in ["none", "None"]:
        pass
    else:
        raise ValueError(f"Unknown augmentation method: {config.augment_method}")

    trainer = Training(config, dataset)
    trainer.train()
