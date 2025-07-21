#!/usr/bin/env python3

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import webdataset as wds
import cv2
import sys
import os
from pathlib import Path

class WebDatasetAnalyzer:
    def __init__(self, dataset_path, fps=50):
        self.dataset_path = dataset_path
        self.fps = fps
        self.current_index = 0
        self.samples = []
        self.fig = None
        self.ax_image = None
        self.ax_panel = None
        self.animation = None
        
        # Action mapping from data_collector_node.py
        self.action_names = ["roadside", "straight", "left", "right"]
        self.action_colors = ["orange", "green", "blue", "red"]
        
        # Load dataset
        self._load_dataset()
        
        # Setup visualization
        self._setup_visualization()
    
    def _load_dataset(self):
        """Load webdataset samples into memory for random access"""
        print(f"Loading dataset from: {self.dataset_path}")
        
        # Find shard files
        dataset_dir = Path(self.dataset_path)
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_path}")
        
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
        
        # Load all samples
        print("Loading samples...")
        for i, sample in enumerate(dataset):
            try:
                # Extract image (already numpy array in RGB format)
                image = sample["npy"]
                if isinstance(image, bytes):
                    image = np.frombuffer(image, dtype=np.uint8)
                # Image is already in RGB format from data_collector_node.py
                
                # Extract metadata (already decoded by webdataset)
                metadata = sample["metadata.json"]
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)
                angle = metadata["angle"]
                action = metadata["action"]
                
                self.samples.append({
                    "image": image,
                    "angle": angle,
                    "action": action,
                    "key": sample["__key__"]
                })
                
                if (i + 1) % 100 == 0:
                    print(f"Loaded {i + 1} samples...")
                    
            except Exception as e:
                print(f"Error loading sample {i}: {e}")
                continue
        
        print(f"Loaded {len(self.samples)} samples total")
        
        if not self.samples:
            raise ValueError("No valid samples found in dataset")
    
    def _setup_visualization(self):
        """Setup matplotlib figure and axes"""
        self.fig = plt.figure(figsize=(12, 6))
        
        # Image display area (left side)
        self.ax_image = plt.subplot2grid((2, 3), (0, 0), colspan=2, rowspan=2)
        self.ax_image.set_title("Camera Image")
        self.ax_image.axis('off')
        
        # Action panel area (right side)
        self.ax_panel = plt.subplot2grid((2, 3), (0, 2), rowspan=2)
        self.ax_panel.set_title("Action & Angle")
        self.ax_panel.set_xlim(0, 2)
        self.ax_panel.set_ylim(0, 5)
        self.ax_panel.axis('off')
        
        # Setup keyboard event handling
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        
        # Initial display
        self._update_display()
        
        # Setup animation for auto-advance
        self.animation = FuncAnimation(
            self.fig, 
            self._animate, 
            interval=1000//self.fps,  # 50Hz = 20ms
            blit=False
        )
    
    def _update_display(self):
        """Update the visualization with current sample"""
        if not self.samples:
            return
        
        # Clamp index to valid range
        self.current_index = max(0, min(self.current_index, len(self.samples) - 1))
        
        sample = self.samples[self.current_index]
        
        # Update image
        self.ax_image.clear()
        self.ax_image.imshow(sample["image"])
        self.ax_image.set_title(f"Image {self.current_index + 1}/{len(self.samples)} (Key: {sample['key']})")
        self.ax_image.axis('off')
        
        # Update action panel
        self.ax_panel.clear()
        self.ax_panel.set_xlim(0, 2)
        self.ax_panel.set_ylim(0, 5)
        self.ax_panel.axis('off')
        self.ax_panel.set_title("Action & Angle")
        
        # Draw action buttons
        button_positions = [
            (0.1, 4.0, "roadside"),    # top-left
            (1.1, 4.0, "straight"),    # top-right  
            (0.1, 3.0, "left"),        # bottom-left
            (1.1, 3.0, "right")        # bottom-right
        ]
        
        for i, (x, y, name) in enumerate(button_positions):
            # Determine if this action is active
            is_active = (sample["action"] == i)
            color = self.action_colors[i] if is_active else 'lightgray'
            alpha = 1.0 if is_active else 0.3
            
            # Draw button rectangle
            rect = patches.Rectangle(
                (x, y), 0.8, 0.6, 
                linewidth=2, 
                edgecolor='black', 
                facecolor=color, 
                alpha=alpha
            )
            self.ax_panel.add_patch(rect)
            
            # Add text
            text_color = 'white' if is_active else 'black'
            self.ax_panel.text(
                x + 0.4, y + 0.3, name, 
                ha='center', va='center', 
                fontsize=8, fontweight='bold', 
                color=text_color
            )
        
        # Display angle value
        self.ax_panel.text(
            1.0, 2.0, f"Angle: {sample['angle']:.3f} rad/s", 
            ha='center', va='center', 
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue")
        )
        
        # Display navigation help
        self.ax_panel.text(
            1.0, 0.8, "Controls:\n← : Back 50\n→ : Forward 100\nSpace: Pause/Resume", 
            ha='center', va='center', 
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow")
        )
    
    def _animate(self, frame):
        """Animation function for auto-advance"""
        if hasattr(self, '_paused') and self._paused:
            return
            
        self.current_index = (self.current_index + 1) % len(self.samples)
        self._update_display()
    
    def _on_key_press(self, event):
        """Handle keyboard events"""
        if event.key == 'left':
            # Go back 50 images
            self.current_index = max(0, self.current_index - 50)
            self._update_display()
            print(f"Moved to index: {self.current_index}")
            
        elif event.key == 'right':
            # Go forward 100 images
            self.current_index = min(len(self.samples) - 1, self.current_index + 100)
            self._update_display()
            print(f"Moved to index: {self.current_index}")
            
        elif event.key == ' ':
            # Toggle pause/resume
            if hasattr(self, '_paused'):
                self._paused = not self._paused
            else:
                self._paused = True
            print(f"Animation {'paused' if self._paused else 'resumed'}")
            
        elif event.key == 'q' or event.key == 'escape':
            # Quit
            plt.close('all')
            sys.exit(0)
    
    def show(self):
        """Display the visualization"""
        print("\nWebDataset Analyzer")
        print("Controls:")
        print("  ← : Go back 50 images")
        print("  → : Go forward 100 images") 
        print("  Space : Pause/Resume auto-advance")
        print("  Q/Esc : Quit")
        print(f"\nDisplaying {len(self.samples)} samples at {self.fps} Hz")
        
        plt.tight_layout()
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Analyze and visualize webdataset data")
    parser.add_argument("dataset_path", help="Path to dataset directory containing webdataset folder")
    parser.add_argument("--fps", type=int, default=50, help="Display refresh rate (default: 50 Hz)")
    
    args = parser.parse_args()
    
    try:
        analyzer = WebDatasetAnalyzer(args.dataset_path, args.fps)
        analyzer.show()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()