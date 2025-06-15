import os
import yaml
import torch
import math
from PIL import Image
from nav_msgs.msg import Odometry
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from .placenet import PlaceNet

class TopologicalMapCreator:
    def __init__(self, map_path: str, image_dir: str, weight_path: str):
        self.map_path = map_path
        self.image_dir = image_dir
        self.weight_path = weight_path
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(os.path.dirname(map_path), exist_ok=True)
        self.node_id = 0
        self.nodes = []

        self.config = {'checkpoint_path':self.weight_path}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = PlaceNet(self.config)
        self.model.to(self.device)
        self.model.eval()

        # 入力画像の前処理 85x85<-学習済みモデルと同様のサイズに調整
        self.transform = Compose([
            Resize((85, 85)),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def add_node(self, image_filename: str, action: str):
        image_path = os.path.join(self.image_dir, image_filename)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model({'image': tensor})
            feature_tensor = outputs['global_descriptor']
            feature = feature_tensor.cpu().squeeze().tolist()

        node_entry = {
            'id': self.node_id,
            'image': image_filename,
            'feature': feature,
            'edges': [
                {
                    'target': self.node_id + 1,
                    'action': action
                }
            ]
        }
        self.nodes.append(node_entry)
        self.node_id += 1

    def save_map(self):
        map_data = {'nodes': self.nodes}
        with open(self.map_path, 'w') as f:
            yaml.dump(map_data, f, sort_keys=False)

    @staticmethod
    def get_pose_from_odom(msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return [x, y], yaw