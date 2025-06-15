import torch
import sys
import os

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, base_dir)

from imitation_nav_training.placenet import PlaceNet

def export_to_torchscript(output_path="placenet.pt"):
    config = {
        'backbone': 'EfficientNet_B0',
        'fc_output_dim': 512,
        'checkpoint_path': os.path.join(base_dir, "weights", "efficientnet_85x85.pth"),
    }

    model = PlaceNet(config)
    model.eval()

    # BaseModel経由では data: dict -> PlaceNet.forward -> net(image) の構造なので
    # model.net を直接 trace する
    dummy_input = torch.randn(1, 3, 85, 85)  # 入力解像度に注意
    traced_script_module = torch.jit.trace(model.net, dummy_input)

    traced_script_module.save(output_path)
    print(f"✅ TorchScript model exported to {output_path}")

if __name__ == "__main__":
    export_to_torchscript("placenet.pt")
