import os
import yaml
import math
from nav_msgs.msg import Odometry

class TopologicalMapCreator:
    def __init__(self, map_path: str, image_dir: str):
        self.map_path = map_path
        self.image_dir = image_dir
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(os.path.dirname(map_path), exist_ok=True)
        self.node_id = 0
        self.nodes = []

    def add_node(self, image_filename: str, position: list, yaw: float, action: str):
        node_entry = {
            'id': self.node_id,
            'image': image_filename,
            'position': position,
            'yaw': yaw,
            'edges': [
                {
                    'target': self.node_id + 1,  # provisional target, may be revised in post-processing
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
        # convert quaternion to yaw
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return [x, y], yaw
