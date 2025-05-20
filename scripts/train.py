import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from ament_index_python.packages import get_package_share_directory


class Config:
    def __init__(self):
        package_dir = get_package_share_directory('imitation_nav_training')
        config_path = os.path.join(package_dir, 'config', 'train_params.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)['train']

        self.package_dir = package_dir
        self.result_dir = os.path.join(package_dir, 'logs', 'result')
        os.makedirs(self.result_dir, exist_ok=True)

        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.learning_rate = config['learning_rate']
        self.shuffle = config.get('shuffle', True)


class AnglePredictor(nn.Module):
    def __init__(self, n_channel, n_out):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channel, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(960, 512)
        self.fc5 = nn.Linear(512, n_out)
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()

        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        torch.nn.init.kaiming_normal_(self.conv3.weight)
        torch.nn.init.kaiming_normal_(self.fc4.weight)

        self.cnn_layer = nn.Sequential(
            self.conv1, self.relu,
            self.conv2, self.relu,
            self.conv3, self.relu,
            self.flatten
        )
        self.fc_layer = nn.Sequential(
            self.fc4, self.relu, self.fc5
        )

    def forward(self, x):
        x = self.cnn_layer(x)
        x = self.fc_layer(x)
        return x


class Training:
    def __init__(self, config, dataset_path):
        self.config = config
        self.dataset_path = dataset_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load dataset
        data = torch.load(self.dataset_path)
        images, angles = data['images'], data['angles']
        dataset = TensorDataset(images, angles)
        self.loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=config.shuffle)

        # Setup model
        self.model = AnglePredictor(3, 1).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.loss_log = []

    def train(self):
        total_batches = len(self.loader) * self.config.epochs
        current_batch = 0

        for epoch in range(self.config.epochs):
            for batch in self.loader:
                inputs, targets = [x.to(self.device) for x in batch]
                preds = self.model(inputs)
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
        # Save model
        model_path = os.path.join(self.config.result_dir, 'angle_predictor.pt')
        torch.save(self.model.state_dict(), model_path)
        print(f"\n✅ モデルを保存しました: {model_path}")

        # Plot loss curve
        plt.figure()
        plt.plot(self.loss_log)
        plt.title("Training Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        loss_plot_path = os.path.join(self.config.result_dir, 'loss_curve.png')
        plt.savefig(loss_plot_path)
        print(f"📈 損失推移グラフを保存しました: {loss_plot_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help='Path to dataset .pt file')
    args = parser.parse_args()

    config = Config()
    trainer = Training(config, args.dataset)
    trainer.train()
