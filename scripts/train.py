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


class ConditionalAnglePredictor(nn.Module):
    def __init__(self, n_channel, n_out, input_height, input_width, n_action_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channel, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()

        with torch.no_grad():
            dummy_input = torch.zeros(1, n_channel, input_height, input_width)
            x = self.conv1(dummy_input)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.relu(x)
            x = self.conv3(x)
            x = self.relu(x)
            x = self.flatten(x)
            flattened_size = x.shape[1]

        self.fc4 = nn.Linear(flattened_size + n_action_classes, 512)
        self.fc5 = nn.Linear(512, n_out)

        self.cnn_layer = nn.Sequential(
            self.conv1, self.relu,
            self.conv2, self.relu,
            self.conv3, self.relu,
            self.flatten
        )

    def forward(self, image, action_class_onehot):
        features = self.cnn_layer(image)
        x = torch.cat([features, action_class_onehot], dim=1)
        x = self.relu(self.fc4(x))
        return self.fc5(x)


class Training:
    def __init__(self, config, dataset_path):
        self.config = config
        self.dataset_path = dataset_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        data = torch.load(self.dataset_path)
        images, angles, actions = data['images'], data['angles'], data['actions']
        n_action_classes = max(actions).item() + 1

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
        dummy_image = torch.randn(1, 3, self.config.image_height, self.config.image_width).to(self.device)
        dummy_action = torch.zeros(1, 3).to(self.device)
        dummy_action[0, 0] = 1
        scripted_model = torch.jit.trace(self.model, (dummy_image, dummy_action))

        scripted_path = os.path.join(self.config.result_dir, self.config.model_filename)
        scripted_model.save(scripted_path)
        print(f"🐜 学習済みモデルを保存しました: {scripted_path}")

        plt.figure()
        plt.plot(self.loss_log)
        plt.title("Training Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        loss_plot_path = os.path.join(self.config.result_dir, 'loss_curve.png')
        plt.savefig(loss_plot_path)
        print(f"📈 损失推移グラフを保存しました: {loss_plot_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help='Path to dataset .pt file')
    args = parser.parse_args()

    augmentor = GammaAugmentor(input_dataset_path=args.dataset)
    augmentor.augment()
    dataset_path = augmentor.output_dataset

    config = Config()
    trainer = Training(config, dataset_path)
    trainer.train()
