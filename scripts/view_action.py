#!/usr/bin/env python3
"""
Dataset Viewer for Imitation Learning
Displays images and corresponding control actions from dataset
Usage: python3 view_action.py ~/dataset/
"""

import os
import sys
import glob
import numpy as np
import argparse
import time
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from matplotlib.patches import Rectangle
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class DatasetViewer:
    def __init__(self, dataset_path, step=1):
        self.dataset_path = Path(dataset_path)
        self.image_dir = self.dataset_path / "images"
        self.action_dir = self.dataset_path / "action"
        self.step = step
        self.interval = 0.05  # Fixed interval
        
        # Action mapping for conditional imitation learning
        self.action_mapping = {
            0: "STRAIGHT",
            1: "LEFT", 
            2: "RIGHT"
        }
        
        # Determine which backend to use
        if MATPLOTLIB_AVAILABLE:
            self.backend = 'matplotlib'
            print("Using matplotlib backend")
        elif PIL_AVAILABLE:
            self.backend = 'pil'
            print("Using PIL backend")
        elif CV2_AVAILABLE:
            self.backend = 'cv2'
            print("Using OpenCV backend (may have display issues in headless environment)")
        else:
            raise ImportError("No suitable image library found. Install matplotlib, PIL, or OpenCV.")
        
        # Load all image and action files
        self.load_dataset()
        
        # Current index
        self.current_index = 0
        
        # Initialize matplotlib figure if using matplotlib
        if self.backend == 'matplotlib':
            plt.ion()  # Turn on interactive mode
            self.fig, self.ax = plt.subplots(figsize=(12, 8))
        
        print(f"Dataset loaded: {len(self.image_files)} samples")
        print(f"Auto-play mode: {self.interval:.3f} seconds per image (step size: {self.step})")
        print("Controls: Press Ctrl+C to stop")
        
    def load_dataset(self):
        """Load all image and action file paths"""
        # Find all image files
        image_patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        self.image_files = []
        
        for pattern in image_patterns:
            self.image_files.extend(glob.glob(str(self.image_dir / pattern)))
        
        # Sort by filename to ensure consistent ordering
        self.image_files.sort()
        
        # Verify corresponding action files exist
        self.action_files = []
        valid_pairs = []
        
        for img_path in self.image_files:
            img_name = Path(img_path).stem
            action_path = self.action_dir / f"{img_name}.csv"
            
            if action_path.exists():
                valid_pairs.append((img_path, str(action_path)))
        
        if not valid_pairs:
            raise FileNotFoundError(f"No matching image-action pairs found in {self.dataset_path}")
        
        self.image_files, self.action_files = zip(*valid_pairs)
        self.image_files = list(self.image_files)
        self.action_files = list(self.action_files)
        
        print(f"Found {len(self.image_files)} valid image-action pairs")
        
    def load_action(self, action_file):
        """Load action value from CSV file"""
        try:
            with open(action_file, 'r') as f:
                content = f.read().strip()
                if content:
                    action_value = int(content)
                    return action_value
                else:
                    return 0  # Default to straight if empty
        except (ValueError, FileNotFoundError):
            return 0  # Default to straight if error
    
    def update_display(self):
        """Update the current display"""
        if self.current_index >= len(self.image_files):
            print("Reached end of dataset")
            return False
        
        if self.current_index < 0:
            self.current_index = 0
        
        # Load image
        img_path = self.image_files[self.current_index]
        
        # Load action
        action_file = self.action_files[self.current_index]
        action_value = self.load_action(action_file)
        action_name = self.action_mapping.get(action_value, f"UNKNOWN({action_value})")
        
        if self.backend == 'matplotlib':
            return self._update_matplotlib(img_path, action_value, action_name)
        else:
            return self._display_other(img_path, action_value, action_name)
    
    def _update_matplotlib(self, img_path, action_value, action_name):
        """Update matplotlib display without creating new figure"""
        try:
            # Load image
            image = mpimg.imread(img_path)
            
            # Clear previous content
            self.ax.clear()
            self.ax.imshow(image)
            self.ax.axis('off')
            
            # Add title with action info
            info_text = (
                f"Sample: {self.current_index + 1}/{len(self.image_files)} | "
                f"Image: {Path(img_path).name} | "
                f"Action: {action_name} ({action_value})"
            )
            self.ax.set_title(info_text, fontsize=12, pad=20)
            
            # Add action indicator bar
            height, width = image.shape[:2]
            bar_height = height * 0.05
            bar_y = height * 0.9
            
            # Action colors and positions (LEFT, STRAIGHT, RIGHT)
            action_colors = ['blue', 'green', 'red']
            action_labels = ['LEFT', 'STRAIGHT', 'RIGHT']
            action_positions = [1, 0, 2]  # Map action values to display positions
            
            section_width = width / 3
            for i in range(3):
                # Determine if this section should be highlighted
                is_active = (action_positions[action_value] == i) if action_value in [0, 1, 2] else False
                color = action_colors[i] if is_active else 'lightgray'
                alpha = 0.8 if is_active else 0.3
                
                rect = Rectangle((i * section_width, bar_y), section_width, bar_height, 
                               facecolor=color, alpha=alpha, edgecolor='black')
                self.ax.add_patch(rect)
                
                # Add label
                self.ax.text(i * section_width + section_width/2, bar_y + bar_height/2, 
                           action_labels[i], ha='center', va='center', 
                           fontsize=10, fontweight='bold', color='white' if is_active else 'black')
            
            # Update display
            plt.tight_layout()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
            return True
            
        except Exception as e:
            print(f"Error displaying image with matplotlib: {e}")
            return True
    
    def _display_other(self, img_path, action_value, action_name):
        """Display using non-matplotlib backends"""
        print(f"Sample: {self.current_index + 1}/{len(self.image_files)}")
        print(f"Image: {Path(img_path).name}")
        print(f"Action: {action_name} ({action_value})")
        print("Press Enter to continue...")
        return True
    
    def navigate(self, direction):
        """Navigate through dataset"""
        if direction == 'next':
            self.current_index += 1
        elif direction == 'prev':
            self.current_index -= 1
        elif direction == 'skip10':
            self.current_index += 10
        elif direction == 'back10':
            self.current_index -= 10
        elif direction == 'skip100':
            self.current_index += 100
        elif direction == 'back100':
            self.current_index -= 100
        elif direction == 'skip1000':
            self.current_index += 1000
        elif direction == 'back1000':
            self.current_index -= 1000
        
        # Bounds checking
        if self.current_index >= len(self.image_files):
            self.current_index = len(self.image_files) - 1
            print("Reached end of dataset")
        elif self.current_index < 0:
            self.current_index = 0
            print("At beginning of dataset")
        
        return self.update_display()
    
    def run(self):
        """Main viewer loop"""
        if not self.update_display():
            return
        
        if self.backend == 'matplotlib':
            self._run_matplotlib()
        else:
            self._run_other()
    
    def _run_matplotlib(self):
        """Run viewer with matplotlib backend - auto-play mode"""
        try:
            # Show the plot window first
            plt.show(block=False)
            
            while self.current_index < len(self.image_files):
                # Update display
                if not self.update_display():
                    break
                
                # Show progress
                print(f"Displaying: {self.current_index + 1}/{len(self.image_files)} - "
                      f"{Path(self.image_files[self.current_index]).name}")
                
                # Auto-advance to next image after interval
                time.sleep(self.interval)
                self.current_index += self.step
                
            print(f"Completed viewing all {len(self.image_files)} images")
                    
        except KeyboardInterrupt:
            print(f"\nViewer stopped at image {self.current_index + 1}/{len(self.image_files)}")
        finally:
            plt.ioff()  # Turn off interactive mode
            plt.close('all')
    
    def _run_other(self):
        """Run viewer with other backends - auto-play mode"""
        try:
            while self.current_index < len(self.image_files):
                # Display current sample
                if not self.update_display():
                    break
                
                # Show progress
                print(f"Displaying: {self.current_index + 1}/{len(self.image_files)} - "
                      f"{Path(self.image_files[self.current_index]).name}")
                
                # Auto-advance to next image after interval
                time.sleep(self.interval)
                self.current_index += self.step
                
            print(f"Completed viewing all {len(self.image_files)} images")
                    
        except KeyboardInterrupt:
            print(f"\nViewer stopped at image {self.current_index + 1}/{len(self.image_files)}")


def main():
    parser = argparse.ArgumentParser(description='View dataset images and actions')
    parser.add_argument('dataset_path', help='Path to dataset directory')
    parser.add_argument('--step', '-s', type=int, default=1, 
                      help='Step size for image advancement (default: 1)')
    
    args = parser.parse_args()
    
    # Validate dataset path
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"Error: Dataset path does not exist: {dataset_path}")
        sys.exit(1)
    
    image_dir = dataset_path / "images"
    action_dir = dataset_path / "action"
    
    if not image_dir.exists():
        print(f"Error: Images directory not found: {image_dir}")
        sys.exit(1)
    
    if not action_dir.exists():
        print(f"Error: Action directory not found: {action_dir}")
        sys.exit(1)
    
    try:
        viewer = DatasetViewer(dataset_path, args.step)
        viewer.run()
    except KeyboardInterrupt:
        print("\nViewer interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()