import torch
import sys
import os

# このファイルの親ディレクトリを取得（= scripts/ の親）
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(base_dir, 'imitation_nav_training'))
from placenet import PlaceNet

def export_to_torchscript(output_path="placenet.pt"):
    model = PlaceNet()
    model.eval()

    dummy_input = torch.randn(1, 3, 88, 200)

    # TorchScript 形式に変換
    traced_script_module = torch.jit.trace(model, dummy_input)

    traced_script_module.save(output_path)
    print(f"✅ TorchScript model exported to {output_path}")

if __name__ == "__main__":
    export_to_torchscript("placenet.pt")