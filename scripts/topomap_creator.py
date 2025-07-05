#!/usr/bin/env python3

import os
import sys
import yaml
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from imitation_nav_training.placenet import PlaceNet

class TopologicalMapCreator:
    def __init__(self, image_dir: str, existing_map_path: str = None):
        pkg_dir = os.path.dirname(os.path.realpath(__file__))
        self.image_dir = image_dir
        self.weight_path = os.path.join(pkg_dir, '..', 'weights', 'efficientnet_85x85.pth')
        self.node_id = 0
        self.nodes = []
        self.existing_nodes = {}  # 既存のマップからのノード情報
        self.current_image_index = 0
        self.image_files = []
        self.existing_map_path = existing_map_path
        
        # 画像ファイルのリストを取得
        self._load_image_files()
        
        if not self.image_files:
            raise ValueError(f"No images found in {image_dir}")
        
        # 既存のマップを読み込み
        self._load_existing_map()
        
        # PlaceNetの初期化
        self._initialize_model()
        
    def _load_image_files(self):
        """画像ファイルのリストを読み込み、ソートする"""
        for filename in os.listdir(self.image_dir):
            if filename.lower().endswith('.png') and filename.startswith('img'):
                self.image_files.append(filename)
        self.image_files.sort()
    
    def _load_existing_map(self):
        """既存のマップファイルを読み込む"""
        if not self.existing_map_path or not os.path.exists(self.existing_map_path):
            print("No existing map found, starting from scratch")
            return
        
        try:
            with open(self.existing_map_path, 'r') as f:
                map_data = yaml.safe_load(f)
            
            if map_data and 'nodes' in map_data:
                for node in map_data['nodes']:
                    # 画像ファイル名をキーとして既存ノード情報を保存
                    if 'image' in node:
                        self.existing_nodes[node['image']] = node
                print(f"Loaded existing map with {len(self.existing_nodes)} nodes")
            else:
                print("Existing map file is empty or invalid")
                
        except Exception as e:
            print(f"Error loading existing map: {e}")
            print("Starting from scratch")
        
    def _initialize_model(self):
        """PlaceNetモデルを初期化"""
        self.config = {'checkpoint_path': self.weight_path} if self.weight_path else {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            self.model = PlaceNet(self.config)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Warning: Could not initialize PlaceNet model: {e}")
            print("Continuing without feature extraction...")
            self.model = None
        
        # 入力画像の前処理
        self.transform = Compose([
            Resize((85, 85)),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def _extract_feature(self, image_path: str):
        """画像から特徴量を抽出"""
        if self.model is None:
            # モデルが利用できない場合はダミーの特徴量を返す
            print("No self.model")
            return [0.0] * 128
            
        try:
            image = Image.open(image_path).convert("RGB")
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model({'image': tensor})
                feature_tensor = outputs['global_descriptor']
                feature = feature_tensor.cpu().squeeze().tolist()
                
            return feature
        except Exception as e:
            print(f"Error extracting features: {e}")
            return [0.0] * 128
    
    def _display_image(self, image_path: str):
        """画像を表示"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not load image: {image_path}")
                return False
                
            # 画像を拡大して表示
            display_image = cv2.resize(image, (340, 340), interpolation=cv2.INTER_NEAREST)
            
            # 画像情報を表示
            filename = os.path.basename(image_path)
            info_text = f"{filename} ({self.current_image_index + 1}/{len(self.image_files)})"
            cv2.putText(display_image, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Topological Map Creator', display_image)
            return True
        except Exception as e:
            print(f"Error displaying image: {e}")
            return False
    
    def _add_node(self, image_filename: str, action: str):
        """ノードをマップに追加"""
        image_path = os.path.join(self.image_dir, image_filename)
        
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return False
        
        # 特徴量を抽出
        feature = self._extract_feature(image_path)
        
        # ノードエントリを作成
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
        
        print(f"Added node {self.node_id - 1}: {image_filename} with action '{action}'")
        return True
    
    def _remove_last_node(self):
        """最後のノードを削除"""
        if self.nodes:
            removed_node = self.nodes.pop()
            self.node_id -= 1
            print(f"Removed node {removed_node['id']}: {removed_node['image']}")
            return True
        return False
    
    def _save_map(self, output_path: str = "topomap.yaml"):
        """マップをYAMLファイルに保存"""
        if self.nodes:
            last_node = self.nodes[-1]
            if last_node['edges'] and last_node['edges'][0]['target'] >= len(self.nodes):
                last_node['edges'] = [
                    {
                        'target': len(self.nodes),
                        'action': 'roadside'
                    }
                ]
        
        map_data = {'nodes': self.nodes}
        
        try:
            with open(output_path, 'w') as f:
                yaml.dump(map_data, f, sort_keys=False, default_flow_style=False)
            print(f"Map saved to {output_path}")
            return True
        except Exception as e:
            print(f"Error saving map: {e}")
            return False
    
    def run(self):
        """メインループを実行"""
        print("Topological Map Creator")
        print("Controls:")
        print("  r: roadside (道なり)")
        print("  w: straight (直進)")
        print("  a: left (左折)") 
        print("  d: right (右折)")
        print("  n: next (既存マップの記述を使用、なければスキップ)")
        print("  s: go back (undo last input)")
        print("  q: quit and save")
        print("  ESC: quit without saving")
        print()
        
        cv2.namedWindow('Topological Map Creator', cv2.WINDOW_AUTOSIZE)
        
        while self.current_image_index < len(self.image_files):
            current_image = self.image_files[self.current_image_index]
            image_path = os.path.join(self.image_dir, current_image)
            
            if not self._display_image(image_path):
                break
            
            print(f"Current image: {current_image}")
            print("Enter action (r/w/a/d) or n to skip, s to go back, q to quit:")
            
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('r'):
                # Roadside
                if self._add_node(current_image, 'roadside'):
                    self.current_image_index += 1
                    
            elif key == ord('w'):
                # Straight
                if self._add_node(current_image, 'straight'):
                    self.current_image_index += 1
                    
            elif key == ord('a'):
                # Left
                if self._add_node(current_image, 'left'):
                    self.current_image_index += 1
                    
            elif key == ord('d'):
                # Right
                if self._add_node(current_image, 'right'):
                    self.current_image_index += 1
                    
            elif key == ord('n'):
                # Next (use existing node if available, otherwise skip)
                if current_image in self.existing_nodes:
                    # 既存のノード情報を使用
                    existing_node = self.existing_nodes[current_image].copy()
                    existing_node['id'] = self.node_id  # IDを更新
                    
                    # エッジのターゲットIDを調整
                    if 'edges' in existing_node:
                        for edge in existing_node['edges']:
                            if 'target' in edge:
                                edge['target'] = self.node_id + 1
                    
                    self.nodes.append(existing_node)
                    self.node_id += 1
                    
                    action = existing_node.get('edges', [{}])[0].get('action', 'unknown')
                    print(f"Used existing node: {current_image} with action '{action}'")
                else:
                    print(f"No existing node found for {current_image}, skipped")
                
                self.current_image_index += 1
                    
            elif key == ord('s'):
                # Go back
                if self.current_image_index > 0:
                    self.current_image_index -= 1
                    self._remove_last_node()
                    print("Went back to previous image")
                else:
                    print("Already at the first image")
                    
            elif key == ord('q'):
                # Quit and save
                print("Saving map and quitting...")
                self._save_map()
                break
                
            elif key == 27:  # ESC key
                # Quit without saving
                print("Quitting without saving...")
                break
                
            else:
                print("Invalid key. Use r/w/a/d for actions, n to skip, s to go back, q to quit")
        
        cv2.destroyAllWindows()
        
        if self.current_image_index >= len(self.image_files):
            print("Processed all images!")
            self._save_map()

def main():
    if len(sys.argv) < 2:
        print("Topological Map Creator - Interactive image labeling tool")
        print("Usage: python3 topomap_creator.py <image_directory> [existing_map.yaml]")
        print()
        print("Controls during execution:")
        print("  r: roadside (道なり)")
        print("  w: straight (直進)")
        print("  a: left (左折)")
        print("  d: right (右折)")
        print("  n: next (既存マップの記述を使用、なければスキップ)")
        print("  s: go back (undo last input)")
        print("  q: quit and save")
        print("  ESC: quit without saving")
        print()
        print("Examples:")
        print("  python3 topomap_creator.py ../logs/topo_map/images/")
        print("  python3 topomap_creator.py ../logs/topo_map/images/ topomap.yaml")
        sys.exit(1)
    
    image_dir = sys.argv[1]
    existing_map_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(image_dir):
        print(f"Error: Image directory '{image_dir}' does not exist")
        sys.exit(1)
    
    if existing_map_path and not os.path.exists(existing_map_path):
        print(f"Warning: Existing map file '{existing_map_path}' does not exist")
        existing_map_path = None
    
    try:
        creator = TopologicalMapCreator(image_dir, existing_map_path)
        creator.run()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()