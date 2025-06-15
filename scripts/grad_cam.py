#!/usr/bin/env python3

import sys
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class GradCAMNode(Node):
    def __init__(self, model_path, num_actions=None):
        super().__init__('gradcam_node')
        
        # CvBridge for converting ROS images to OpenCV format
        self.bridge = CvBridge()
        
        # Load the trained model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f'Using device: {self.device}')
        
        # Set default number of actions (you may need to adjust this)
        self.num_actions = num_actions if num_actions is not None else 3  # Common for robot control (left, straight, right)
        
        try:
            # Load the model (handle both regular PyTorch and TorchScript models)
            if model_path.endswith('.pt') or model_path.endswith('.pth'):
                try:
                    # Try loading as TorchScript first
                    self.model = torch.jit.load(model_path, map_location=self.device)
                    self.is_torchscript = True
                    self.get_logger().info(f'TorchScript model loaded successfully from {model_path}')
                except:
                    # Fallback to regular PyTorch model
                    self.model = torch.load(model_path, map_location=self.device)
                    self.is_torchscript = False
                    self.get_logger().info(f'PyTorch model loaded successfully from {model_path}')
            else:
                self.model = torch.load(model_path, map_location=self.device)
                self.is_torchscript = False
                self.get_logger().info(f'Model loaded successfully from {model_path}')
            
            self.model.eval()
            
            # Check if model requires action_onehot
            self.requires_action = self._check_if_requires_action()
            if self.requires_action:
                self.get_logger().info(f'Model requires action_onehot input with {self.num_actions} actions')
            
        except Exception as e:
            self.get_logger().error(f'Failed to load model from {model_path}: {str(e)}')
            sys.exit(1)
        
        # Setup Grad-CAM
        try:
            target_layers = []
            
            if self.is_torchscript:
                # For TorchScript models, we need to find conv layers differently
                self.get_logger().info('Analyzing TorchScript model structure...')
                
                # Print model structure for debugging
                print("TorchScript model graph:")
                print(self.model.graph)
                
                # Create a wrapper to make TorchScript model compatible with Grad-CAM
                class TorchScriptWrapper(torch.nn.Module):
                    def __init__(self, torchscript_model, requires_action=True):
                        super().__init__()
                        self.model = torchscript_model
                        self.requires_action = requires_action
                        
                    def forward(self, x, action_onehot=None):
                        if self.requires_action and action_onehot is not None:
                            return self.model(x, action_onehot)
                        else:
                            # For TorchScript models that require action, create default straight action
                            if self.requires_action:
                                default_action = torch.zeros(1, 3).to(x.device)
                                default_action[0, 1] = 1.0  # Straight action
                                return self.model(x, default_action)
                            else:
                                return self.model(x)
                
                # Wrap the TorchScript model
                wrapped_model = TorchScriptWrapper(self.model, self.requires_action)
                
                # For TorchScript, we'll use a hook-based approach
                # This is a workaround since we can't easily access internal layers
                self.activation = {}
                def get_activation(name):
                    def hook(model, input, output):
                        self.activation[name] = output
                    return hook
                
                # Register hook on the model
                # Note: This is a simplified approach for TorchScript models
                self.model = wrapped_model
                target_layers = [wrapped_model]  # Use the whole model as target
                
            else:
                # For regular PyTorch models
                # For ResNet-like architectures, use the last convolutional layer
                if hasattr(self.model, 'layer4'):
                    target_layers = [self.model.layer4[-1]]
                elif hasattr(self.model, 'features'):
                    # For VGG-like architectures
                    target_layers = [self.model.features[-1]]
                elif hasattr(self.model, 'conv_layers'):
                    # For custom architectures
                    target_layers = [self.model.conv_layers[-1]]
                else:
                    # Try to find the last convolutional layer automatically
                    conv_layers = []
                    for name, module in self.model.named_modules():
                        if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
                            conv_layers.append((name, module))
                    
                    if conv_layers:
                        # Use the last convolutional layer found
                        last_conv_name, last_conv_layer = conv_layers[-1]
                        target_layers = [last_conv_layer]
                        self.get_logger().info(f'Using layer: {last_conv_name}')
                    else:
                        # If no conv layers found, print model structure for debugging
                        self.get_logger().info('Model structure:')
                        for name, module in self.model.named_modules():
                            self.get_logger().info(f'  {name}: {type(module)}')
                        raise ValueError("Could not find suitable target layer for Grad-CAM")
            
            if not target_layers:
                if self.is_torchscript:
                    # For TorchScript models, skip pytorch-gradcam and use simple gradient method
                    self.get_logger().info('TorchScript model detected. Using simple gradient-based visualization.')
                    # Don't initialize cam for TorchScript models
                    pass
                else:
                    raise ValueError("Could not find suitable target layer for Grad-CAM")
            else:
                self.cam = GradCAM(model=self.model, target_layers=target_layers)
                self.get_logger().info('Grad-CAM initialized successfully')
            
        except Exception as e:
            self.get_logger().error(f'Failed to initialize Grad-CAM: {str(e)}')
            sys.exit(1)
        
        # ROS2 subscribers and publishers
        self.image_subscription = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            10
        )
        
        self.gradcam_publisher = self.create_publisher(
            Image,
            '/gradcam_image',
            10
        )
        
        self.get_logger().info('GradCAM node initialized. Waiting for images...')
    
    def _check_if_requires_action(self):
        """Check if the model requires action_onehot input"""
        try:
            # Create dummy inputs
            dummy_image = torch.randn(1, 3, 88, 200).to(self.device)
            
            # Try without action_onehot first
            try:
                with torch.no_grad():
                    _ = self.model(dummy_image)
                return False
            except:
                # Model requires action_onehot
                return True
        except:
            # Assume it requires action if we can't test
            return True
    
    def create_action_onehot(self, action_idx=1):
        """Create action onehot tensor - default to straight (index 1)"""
        action_onehot = torch.zeros(1, self.num_actions).to(self.device)
        action_onehot[0, action_idx] = 1.0
        return action_onehot
    
    def preprocess_image(self, cv_image):
        """Preprocess the input image for the model"""
        # Resize to 200x88
        resized = cv2.resize(cv_image, (200, 88))
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1] for visualization
        rgb_normalized = rgb_image.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        # Assuming the model expects input in the range [0, 1]
        # You may need to adjust normalization based on your training
        tensor_image = torch.from_numpy(rgb_image).permute(2, 0, 1).float() / 255.0
        tensor_image = tensor_image.unsqueeze(0).to(self.device)
        
        return tensor_image, rgb_normalized
    
    def generate_simple_gradcam(self, input_tensor):
        """Generate Grad-CAM using gradient computation - always use straight action"""
        input_tensor.requires_grad_(True)
        
        # Create action_onehot for straight action (index 1)
        action_onehot = self.create_action_onehot(action_idx=1)
        
        # Forward pass
        if self.requires_action:
            output = self.model(input_tensor, action_onehot)
        else:
            output = self.model(input_tensor)
        
        # Determine target for backprop
        if len(output.shape) > 1 and output.shape[1] > 1:
            target_class = output.argmax(dim=1).item()
            target = output[0, target_class]
        else:
            # For regression models, use the output directly
            target = output.sum()
        
        # Backward pass
        self.model.zero_grad()
        target.backward()
        
        # Get gradients and activations
        gradients = input_tensor.grad.data
        
        # Compute Grad-CAM
        # This is a simplified version using input gradients
        gradcam = torch.abs(gradients).mean(dim=1, keepdim=True)
        gradcam = F.interpolate(gradcam, size=(88, 200), mode='bilinear', align_corners=False)
        gradcam = gradcam.squeeze().cpu().numpy()
        
        # Normalize to [0, 1]
        gradcam = (gradcam - gradcam.min()) / (gradcam.max() - gradcam.min() + 1e-8)
        
        return gradcam

    def image_callback(self, msg):
        """Callback function for processing incoming images"""
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Preprocess the image
            input_tensor, rgb_normalized = self.preprocess_image(cv_image)
            
            # Generate visualization
            try:
                if hasattr(self, 'cam'):
                    # Use pytorch-gradcam for regular models
                    action_onehot = None
                    if self.requires_action:
                        action_onehot = self.create_action_onehot(action_idx=1)
                    
                    with torch.no_grad():
                        if self.requires_action:
                            output = self.model(input_tensor, action_onehot)
                        else:
                            output = self.model(input_tensor)
                        
                        if len(output.shape) > 1 and output.shape[1] > 1:
                            predicted_class = output.argmax(dim=1).item()
                            targets = [ClassifierOutputTarget(predicted_class)]
                        else:
                            targets = None
                    
                    # Note: pytorch-gradcam might not work well with conditional models
                    # So we'll skip it and use our custom method
                    raise Exception("Using custom method for conditional model")
                    
                else:
                    # Use our custom gradient method with straight action only
                    grayscale_cam = self.generate_simple_gradcam(input_tensor.clone())
                
            except Exception as cam_error:
                self.get_logger().info(f'Using custom gradient method: {cam_error}')
                # Use our custom gradient-based visualization with straight action
                grayscale_cam = self.generate_simple_gradcam(input_tensor.clone())
            
            # Create visualization
            visualization = show_cam_on_image(rgb_normalized, grayscale_cam, use_rgb=True)
            
            # Convert back to BGR for ROS
            visualization_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
            
            # Convert to ROS Image message
            gradcam_msg = self.bridge.cv2_to_imgmsg(visualization_bgr, encoding='bgr8')
            gradcam_msg.header = msg.header  # Preserve timestamp and frame_id
            
            # Publish the Grad-CAM result
            self.gradcam_publisher.publish(gradcam_msg)
            
            self.get_logger().info('Grad-CAM visualization published')
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='ROS2 Grad-CAM Node')
    parser.add_argument('model_path', help='Path to the trained PyTorch model (.pt file)')
    parser.add_argument('--num_actions', type=int, default=3, 
                       help='Number of actions for conditional model (default: 3)')
    
    if len(sys.argv) < 2:
        parser.print_help()
        print("\nError: Model path is required!")
        print("Usage: python3 grad_cam.py logs/model.pt [--num_actions N]")
        sys.exit(1)
    
    args = parser.parse_args()
    
    # Check if model file exists
    import os
    if not os.path.exists(args.model_path):
        print(f"Error: Model file '{args.model_path}' not found!")
        sys.exit(1)
    
    # Initialize ROS2
    rclpy.init()
    
    try:
        # Create and run the node
        gradcam_node = GradCAMNode(args.model_path, args.num_actions)
        rclpy.spin(gradcam_node)
        
    except KeyboardInterrupt:
        print("Shutting down...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Cleanup
        if 'gradcam_node' in locals():
            gradcam_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()