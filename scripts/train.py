import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

# データセット読み込み
data = torch.load('dataset.pt')
images = data['images']
angles = data['angles']

dataset = TensorDataset(images, angles)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# シンプルなCNNモデル
class AnglePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, 5, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 25 * 25, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# モデル、損失関数、最適化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AnglePredictor().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 学習ループ
for epoch in range(10):
    for batch in loader:
        inputs, targets = [x.to(device) for x in batch]
        preds = model(inputs)
        loss = criterion(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}: loss = {loss.item():.4f}")

# モデル保存
torch.save(model.state_dict(), 'angle_predictor.pt')
print("✅ モデル angle_predictor.pt を保存しました。")
