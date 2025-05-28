import torch
import torch.nn as nn
import torch.nn.functional as F


class PlaceNet(nn.Module):
    def __init__(self, output_dim=128):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # → (64, 23, 9)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),  # → (128, 11, 4)
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 11 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

    def forward(self, x: torch.Tensor):
        x = self.conv_layers(x)
        x = self.fc(x)
        x = F.normalize(x, dim=1)
        return x
