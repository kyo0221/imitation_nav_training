import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from ament_index_python.packages import get_package_share_directory
from augment.gamma_augment import GammaAugmentor


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
    def __init__(self, config, dataset_path):
        self.config = config
        self.dataset_path = dataset_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        data = torch.load(self.dataset_path)
        images, angles, actions = data['images'], data['angles'], data['actions']
        n_action_classes = len(config.class_names)

        onehot_actions = torch.nn.functional.one_hot(actions, num_classes=n_action_classes).float()
        dataset = TensorDataset(images, onehot_actions, angles)
        self.loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=config.shuffle)

        self.model = ConditionalAnglePredictor(3, 1, config.image_height, config.image_width, n_action_classes).to(self.device)
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
        # dummy_image = torch.randn(1, 3, self.config.image_height, self.config.image_width).to(self.device)
        # dummy_action = torch.zeros(1, len(self.config.class_names)).to(self.device)
        # dummy_action[0, 0] = 1

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


class Sampling:
    def resample_dataset_balanced(dataset_path: str, class_names: list) -> str:
        data = torch.load(dataset_path)
        images, angles, actions = data['images'], data['angles'], data['actions']

        class_to_indices = {
            cls: (actions == idx).nonzero(as_tuple=True)[0]
            for idx, cls in enumerate(class_names)
            if (actions == idx).sum().item() > 0
        }

        if not class_to_indices:
            raise ValueError("❌ 全ての行動クラスにデータが存在しません")

        max_count = max(len(idxs) for idxs in class_to_indices.values())

        balanced_images = []
        balanced_angles = []
        balanced_actions = []

        for idx, cls in enumerate(class_names):
            if cls not in class_to_indices:
                continue
            indices = class_to_indices[cls]
            repeats = max_count // len(indices)
            remainder = max_count % len(indices)

            resampled_idxs = indices.repeat(repeats)
            remainder_idxs = indices[torch.randperm(len(indices))[:remainder]]
            final_idxs = torch.cat([resampled_idxs, remainder_idxs])

            balanced_images.append(images[final_idxs])
            balanced_angles.append(angles[final_idxs])
            balanced_actions.append(actions[final_idxs])

        balanced_data = {
            'images': torch.cat(balanced_images),
            'angles': torch.cat(balanced_angles),
            'actions': torch.cat(balanced_actions),
            'action_classes': class_names
        }

        new_path = dataset_path.replace('.pt', '_resampled.pt')
        torch.save(balanced_data, new_path)
        print(f"📊 リサンプリング済みデータ保存: {new_path}")
        return new_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help='Path to dataset .pt file')
    args = parser.parse_args()

    config = Config()
    dataset_path = args.dataset

    yaml_path = os.path.join(config.package_dir, '..', 'config', 'train_params.yaml')
    config_dict = yaml.safe_load(open(yaml_path))

    if config_dict['train'].get('resample', False):
        dataset_path = Sampling.resample_dataset_balanced(dataset_path, config.class_names)

    augmentor = GammaAugmentor(input_dataset_path=dataset_path)
    augmentor.augment()
    dataset_path = augmentor.output_dataset

    trainer = Training(config, dataset_path)
    trainer.train()
