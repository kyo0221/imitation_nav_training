#!/usr/bin/env python3

import argparse
import os
import json
import sys
import cv2
import numpy as np
from pathlib import Path
import webdataset as wds
from PIL import Image
import torch
import yaml
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import io

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from imitation_nav_training.placenet import PlaceNet

class TopomapFromWebdataset:
    def __init__(self, weight_path, output_dir):
        self.weight_path = weight_path
        self.output_dir = output_dir
        self.image_dir = os.path.join(output_dir, 'images')
        self.topomap_yaml = os.path.join(output_dir, 'topomap.yaml')
        
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        self.node_id = 0
        self.nodes = []
        self.action_names = ["roadside", "straight", "left", "right"]
        
        # PlaceNet model setup
        self.config = {'checkpoint_path': self.weight_path}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PlaceNet(self.config)
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing (85x85 to match trained model)
        self.transform = Compose([
            Resize((85, 85)),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        print(f"Initialized topomap creator with model: {weight_path}")
        print(f"Output directory: {output_dir}")
        print(f"Using device: {self.device}")

    def load_webdataset_samples(self, dataset_path):
        """Load webdataset samples"""
        print(f"Loading webdataset from: {dataset_path}")
        
        dataset_dir = Path(dataset_path)
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
        
        webdataset_dir = dataset_dir / "webdataset"  
        if not webdataset_dir.exists():
            raise FileNotFoundError(f"Webdataset directory not found: {webdataset_dir}")
        
        shard_files = list(webdataset_dir.glob("shard_*.tar*"))
        if not shard_files:
            raise FileNotFoundError(f"No shard files found in: {webdataset_dir}")
        
        print(f"Found {len(shard_files)} shard files")
        
        # Create webdataset loader
        urls = [str(f) for f in sorted(shard_files)]
        dataset = wds.WebDataset(urls).decode()
        
        samples = []
        print("Loading samples...")
        for i, sample in enumerate(dataset):
            try:
                # Extract image (numpy array in RGB format)
                image = sample["npy"]
                if isinstance(image, bytes):
                    image_buffer = io.BytesIO(image)
                    image = np.load(image_buffer)
                
                # Extract metadata
                metadata = sample["metadata.json"]
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)
                
                samples.append({
                    "image": image,
                    "action": metadata["action"],
                    "key": sample["__key__"],
                    "sample_id": i
                })
                
                if (i + 1) % 1000 == 0:
                    print(f"Loaded {i + 1} samples...")
                    
            except Exception as e:
                print(f"Error loading sample {i}: {e}")
                continue
        
        print(f"Loaded {len(samples)} samples total")
        return samples

    def should_create_node(self, current_idx, current_action, prev_action):
        """Determine if a node should be created"""
        # Create node every 100 images
        if current_idx % 100 == 0:
            return True
        
        # Create node when action changes from/to left or right
        if (current_action in [2, 3] or prev_action in [2, 3]) and current_action != prev_action:
            return True
        
        return False

    def create_feature_vector(self, image_array):
        """Create feature vector using PlaceNet"""
        # Convert numpy array to PIL Image for preprocessing
        if image_array.max() <= 1.0:
            image_array = (image_array * 255).astype(np.uint8)
        
        pil_image = Image.fromarray(image_array).convert("RGB")
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model({'image': tensor})
            feature_tensor = outputs['global_descriptor']
            feature = feature_tensor.cpu().squeeze().tolist()
        
        return feature

    def add_node(self, image_array, action, sample_id):
        """Add a node to the topomap"""
        # Save image
        image_id = self.node_id + 1
        image_filename = f"img{image_id:05d}.png"
        image_path = os.path.join(self.image_dir, image_filename)
        
        # Resize to 85x85 for consistency
        resized_img = cv2.resize(image_array, (85, 85))
        if resized_img.shape[2] == 3:  # RGB to BGR for cv2
            resized_img = cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(image_path, resized_img)
        
        # Create feature vector
        feature = self.create_feature_vector(image_array)
        
        # Create node entry
        node_entry = {
            'id': self.node_id,
            'image': image_filename,
            'feature': feature,
            'edges': [
                {
                    'target': self.node_id + 1,
                    'action': self.action_names[action]
                }
            ]
        }
        
        self.nodes.append(node_entry)
        print(f"Added node {self.node_id}: sample {sample_id}, action {self.action_names[action]}")
        self.node_id += 1

    def create_topomap(self, dataset_path):
        """Create topomap from webdataset"""
        samples = self.load_webdataset_samples(dataset_path)
        
        if not samples:
            raise ValueError("No samples found in dataset")
        
        print(f"Creating topomap from {len(samples)} samples...")
        
        prev_action = samples[0]['action']
        nodes_created = 0
        
        for i, sample in enumerate(samples):
            current_action = sample['action']
            
            if self.should_create_node(i, current_action, prev_action):
                self.add_node(sample['image'], current_action, sample['sample_id'])
                nodes_created += 1
            
            prev_action = current_action
            
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{len(samples)} samples, created {nodes_created} nodes")
        
        print(f"Created {nodes_created} nodes total")
        
        # Save topomap
        self.save_topomap()

    def save_topomap(self):
        """Save topomap to YAML file"""
        map_data = {'nodes': self.nodes}
        with open(self.topomap_yaml, 'w') as f:
            yaml.dump(map_data, f, sort_keys=False)
        
        print(f"Saved topomap with {len(self.nodes)} nodes to: {self.topomap_yaml}")
        print(f"Node images saved to: {self.image_dir}")

def main():
    parser = argparse.ArgumentParser(description="Create topomap from webdataset")
    parser.add_argument("dataset_path", help="Path to dataset directory containing webdataset folder")
    parser.add_argument("weight_path", help="Path to PlaceNet model weights")
    parser.add_argument("--output_dir", default="./topomap_output", 
                       help="Output directory for topomap (default: ./topomap_output)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.weight_path):
        print(f"Error: Weight file not found: {args.weight_path}")
        sys.exit(1)
    
    try:
        creator = TopomapFromWebdataset(args.weight_path, args.output_dir)
        creator.create_topomap(args.dataset_path)
        print("Topomap creation completed successfully!")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()