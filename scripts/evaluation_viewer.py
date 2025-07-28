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
import torch
import torch.nn.functional as F
from collections import deque
from tqdm import tqdm

class E2EEvaluationViewer:
    def __init__(self, dataset_path, model_path, fps=10, history_size=100, input_size=None):
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.fps = fps
        self.history_size = history_size
        self.input_size = input_size  # (width, height) tuple
        self.current_index = 0
        self.samples = []
        self.fig = None
        self.axes = {}
        self.animation = None
        self._paused = False
        
        # History for graphs
        self.time_history = deque(maxlen=history_size)
        self.ground_truth_history = deque(maxlen=history_size)
        self.prediction_history = deque(maxlen=history_size)
        
        # Action mapping from data_collector_node.py
        self.action_names = ["roadside", "straight", "left", "right"]
        self.action_colors = ["orange", "green", "blue", "red"]
        
        # Get input size interactively if not provided
        if self.input_size is None:
            self.input_size = self._get_input_size_interactively()
        
        print(f"Using input size: {self.input_size[0]}x{self.input_size[1]} (width x height)")
        
        # Load model and dataset
        self._load_model()
        self._load_dataset()
        
        # Setup visualization
        self._setup_visualization()
    
    def _get_input_size_interactively(self):
        """Get input size from user interactively"""
        print("\nModel input size configuration:")
        print("Common sizes:")
        print("  1. 200x88  (train_params.yaml default)")  
        print("  2. 200x66  (updated train_params.yaml)")
        print("  3. Custom size")
        
        while True:
            try:
                choice = input("Select option (1-3): ").strip()
                
                if choice == "1":
                    return (200, 88)
                elif choice == "2": 
                    return (200, 66)
                elif choice == "3":
                    width = int(input("Enter width: ").strip())
                    height = int(input("Enter height: ").strip())
                    if width > 0 and height > 0:
                        return (width, height)
                    else:
                        print("Width and height must be positive integers.")
                else:
                    print("Invalid choice. Please select 1, 2, or 3.")
                    
            except ValueError:
                print("Invalid input. Please enter valid numbers.")
            except KeyboardInterrupt:
                print("\nOperation cancelled.")
                sys.exit(0)
    
    def _load_model(self):
        """Load PyTorch model for inference"""
        print(f"Loading model from: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            # Load TorchScript model
            self.model = torch.jit.load(self.model_path, map_location=self.device)
            self.model.eval()
            print(f"Model loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
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
        dataset = wds.WebDataset(urls, shardshuffle=False).decode()
        
        # Load all samples from webdataset
        print("Loading all samples from webdataset...")
        print("⚠️  Loading complete dataset - this may take some time and use significant memory")
        
        # First pass to count total samples
        print("Counting total samples...")
        sample_count = 0
        for _ in wds.WebDataset(urls, shardshuffle=False).decode():
            sample_count += 1
        print(f"Found {sample_count} total samples")
        
        # Second pass to load all samples with progress bar
        dataset = wds.WebDataset(urls, shardshuffle=False).decode()
        sample_iter = tqdm(enumerate(dataset), total=sample_count, desc="Loading samples")
        
        for i, sample in sample_iter:
            try:
                # Extract image (already numpy array in RGB format)
                image = sample["npy"]
                if isinstance(image, bytes):
                    image = np.frombuffer(image, dtype=np.uint8)
                
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
                    
            except Exception as e:
                sample_iter.write(f"Error loading sample {i}: {e}")
                continue
        
        print(f"Loaded {len(self.samples)} samples total")
        
        if not self.samples:
            raise ValueError("No valid samples found in dataset")
    
    def _preprocess_image(self, image):
        """Preprocess image for model inference"""
        # Resize to specified input size (width, height)
        resized = cv2.resize(image, self.input_size)
        
        # Convert to tensor and normalize
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0)  # Add batch dimension
        
        return tensor.to(self.device)
    
    def _predict_angle(self, image, action):
        """Predict angle using the loaded model"""
        try:
            # Preprocess image
            image_tensor = self._preprocess_image(image)
            
            # Create action one-hot encoding
            action_onehot = torch.zeros(1, 4, device=self.device)
            action_onehot[0, action] = 1.0
            
            # Model inference
            with torch.no_grad():
                prediction = self.model(image_tensor, action_onehot)
                
            return prediction.cpu().numpy()[0, 0]  # Extract scalar value
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return 0.0
    
    def _setup_visualization(self):
        """Setup matplotlib figure with 2x2 layout"""
        self.fig = plt.figure(figsize=(16, 10))
        
        # Create 2x2 grid
        # Top-left: Camera Image
        self.axes['image'] = plt.subplot2grid((2, 2), (0, 0))
        self.axes['image'].set_title("Camera Image")
        self.axes['image'].axis('off')
        
        # Top-right: Action Display
        self.axes['action'] = plt.subplot2grid((2, 2), (0, 1))
        self.axes['action'].set_title("Action & Control Input")
        self.axes['action'].set_xlim(0, 2)
        self.axes['action'].set_ylim(0, 5)
        self.axes['action'].axis('off')
        
        # Bottom-left: Ground Truth Angular Velocity
        self.axes['ground_truth'] = plt.subplot2grid((2, 2), (1, 0))
        self.axes['ground_truth'].set_title("Ground Truth Angular Velocity")
        self.axes['ground_truth'].set_xlabel("Time Step")
        self.axes['ground_truth'].set_ylabel("Angular Velocity (rad/s)")
        self.axes['ground_truth'].grid(True)
        
        # Bottom-right: Predicted Angular Velocity
        self.axes['prediction'] = plt.subplot2grid((2, 2), (1, 1))
        self.axes['prediction'].set_title("Predicted Angular Velocity")
        self.axes['prediction'].set_xlabel("Time Step")
        self.axes['prediction'].set_ylabel("Angular Velocity (rad/s)")
        self.axes['prediction'].grid(True)
        
        # Setup keyboard event handling
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        
        # Initial display
        self._update_display()
        
        # Setup animation for auto-advance
        self.animation = FuncAnimation(
            self.fig, 
            self._animate, 
            interval=1000//self.fps,
            blit=False
        )
    
    def _update_display(self):
        """Update the visualization with current sample"""
        if not self.samples:
            return
        
        # Clamp index to valid range
        self.current_index = max(0, min(self.current_index, len(self.samples) - 1))
        
        sample = self.samples[self.current_index]
        
        # Get prediction
        predicted_angle = self._predict_angle(sample["image"], sample["action"])
        
        # Update history
        self.time_history.append(self.current_index)
        self.ground_truth_history.append(sample["angle"])
        self.prediction_history.append(predicted_angle)
        
        # Update camera image
        self._update_camera_image(sample)
        
        # Update action display
        self._update_action_display(sample)
        
        # Update ground truth graph
        self._update_ground_truth_graph()
        
        # Update prediction graph
        self._update_prediction_graph()
        
        # Update main title
        self.fig.suptitle(f"E2E Autonomous Driving Evaluation - Sample {self.current_index + 1}/{len(self.samples)}", fontsize=14)
    
    def _update_camera_image(self, sample):
        """Update camera image display"""
        self.axes['image'].clear()
        self.axes['image'].imshow(sample["image"])
        self.axes['image'].set_title(f"Camera Image (Key: {sample['key']})")
        self.axes['image'].axis('off')
    
    def _update_action_display(self, sample):
        """Update action and control input display"""
        self.axes['action'].clear()
        self.axes['action'].set_xlim(0, 2)
        self.axes['action'].set_ylim(0, 5)
        self.axes['action'].axis('off')
        self.axes['action'].set_title("Action & Control Input")
        
        # Draw action buttons (same as analyze_webdataset.py)
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
            self.axes['action'].add_patch(rect)
            
            # Add text
            text_color = 'white' if is_active else 'black'
            self.axes['action'].text(
                x + 0.4, y + 0.3, name, 
                ha='center', va='center', 
                fontsize=10, fontweight='bold', 
                color=text_color
            )
        
        # Display control input information
        self.axes['action'].text(
            1.0, 2.0, f"Ground Truth: {sample['angle']:.4f} rad/s", 
            ha='center', va='center', 
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen")
        )
        
        # Get current prediction
        predicted_angle = self.prediction_history[-1] if self.prediction_history else 0.0
        self.axes['action'].text(
            1.0, 1.3, f"Predicted: {predicted_angle:.4f} rad/s", 
            ha='center', va='center', 
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue")
        )
        
        # Display error
        error = abs(sample['angle'] - predicted_angle)
        self.axes['action'].text(
            1.0, 0.6, f"Error: {error:.4f} rad/s", 
            ha='center', va='center', 
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral")
        )
    
    def _update_ground_truth_graph(self):
        """Update ground truth angular velocity graph"""
        self.axes['ground_truth'].clear()
        self.axes['ground_truth'].set_title("Ground Truth Angular Velocity")
        self.axes['ground_truth'].set_xlabel("Time Step")
        self.axes['ground_truth'].set_ylabel("Angular Velocity (rad/s)")
        self.axes['ground_truth'].grid(True)
        
        if len(self.time_history) > 0:
            self.axes['ground_truth'].plot(
                list(self.time_history), 
                list(self.ground_truth_history), 
                'g-', linewidth=2, label='Ground Truth'
            )
            self.axes['ground_truth'].legend()
            
            # Highlight current point
            if len(self.ground_truth_history) > 0:
                self.axes['ground_truth'].plot(
                    self.time_history[-1], 
                    self.ground_truth_history[-1], 
                    'go', markersize=8
                )
    
    def _update_prediction_graph(self):
        """Update predicted angular velocity graph"""
        self.axes['prediction'].clear()
        self.axes['prediction'].set_title("Predicted Angular Velocity")
        self.axes['prediction'].set_xlabel("Time Step")
        self.axes['prediction'].set_ylabel("Angular Velocity (rad/s)")
        self.axes['prediction'].grid(True)
        
        if len(self.time_history) > 0:
            self.axes['prediction'].plot(
                list(self.time_history), 
                list(self.prediction_history), 
                'b-', linewidth=2, label='Prediction'
            )
            self.axes['prediction'].legend()
            
            # Highlight current point
            if len(self.prediction_history) > 0:
                self.axes['prediction'].plot(
                    self.time_history[-1], 
                    self.prediction_history[-1], 
                    'bo', markersize=8
                )
    
    def _animate(self, frame):
        """Animation function for auto-advance"""
        if self._paused:
            return
            
        self.current_index = (self.current_index + 1) % len(self.samples)
        self._update_display()
    
    def _on_key_press(self, event):
        """Handle keyboard events"""
        if event.key == 'left':
            # Go back 10 samples
            self.current_index = max(0, self.current_index - 10)
            self._update_display()
            print(f"Moved to index: {self.current_index}")
            
        elif event.key == 'right':
            # Go forward 10 samples
            self.current_index = min(len(self.samples) - 1, self.current_index + 10)
            self._update_display()
            print(f"Moved to index: {self.current_index}")
            
        elif event.key == ' ':
            # Toggle pause/resume
            self._paused = not self._paused
            print(f"Animation {'paused' if self._paused else 'resumed'}")
            
        elif event.key == 'r':
            # Reset to beginning
            self.current_index = 0
            self.time_history.clear()
            self.ground_truth_history.clear()
            self.prediction_history.clear()
            self._update_display()
            print("Reset to beginning")
            
        elif event.key == 's':
            # Save current state
            self._save_current_state()
            
        elif event.key == 'q' or event.key == 'escape':
            # Quit
            plt.close('all')
            sys.exit(0)
    
    def _save_current_state(self):
        """Save current evaluation state to file"""
        try:
            output_dir = Path("evaluation_results")
            output_dir.mkdir(exist_ok=True)
            
            # Save current frame
            sample = self.samples[self.current_index]
            cv2.imwrite(str(output_dir / f"frame_{self.current_index:06d}.png"), 
                       cv2.cvtColor(sample["image"], cv2.COLOR_RGB2BGR))
            
            # Save graph data
            data = {
                "time_steps": list(self.time_history),
                "ground_truth": list(self.ground_truth_history),
                "predictions": list(self.prediction_history),
                "current_index": self.current_index
            }
            
            with open(output_dir / f"evaluation_data_{self.current_index:06d}.json", 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"Saved evaluation state at index {self.current_index} to {output_dir}")
            
        except Exception as e:
            print(f"Error saving state: {e}")
    
    def show(self):
        """Display the evaluation viewer"""
        print("\nE2E Autonomous Driving Evaluation Viewer")
        print("Controls:")
        print("  ← : Go back 10 samples")
        print("  → : Go forward 10 samples") 
        print("  Space : Pause/Resume auto-advance")
        print("  R : Reset to beginning")
        print("  S : Save current state")
        print("  Q/Esc : Quit")
        print(f"\nEvaluating {len(self.samples)} samples at {self.fps} Hz")
        print(f"Model: {self.model_path}")
        print(f"Dataset: {self.dataset_path}")
        
        plt.tight_layout()
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Evaluate E2E autonomous driving model performance")
    parser.add_argument("dataset_path", help="Path to dataset directory containing webdataset folder")
    parser.add_argument("model_path", help="Path to PyTorch model file (.pt)")
    parser.add_argument("--fps", type=int, default=10, help="Display refresh rate (default: 10 Hz)")
    parser.add_argument("--history", type=int, default=100, help="History size for graphs (default: 100)")
    parser.add_argument("--input-size", nargs=2, type=int, metavar=('WIDTH', 'HEIGHT'), 
                       help="Model input image size (width height), e.g., --input-size 200 66")
    
    args = parser.parse_args()
    
    # Convert input_size to tuple if provided
    input_size = tuple(args.input_size) if args.input_size else None
    
    try:
        viewer = E2EEvaluationViewer(args.dataset_path, args.model_path, args.fps, args.history, input_size)
        viewer.show()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()