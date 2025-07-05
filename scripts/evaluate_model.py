import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")

from augment.imitation_dataset import ImitationDataset


class ConditionalAnglePredictor(nn.Module):
    def __init__(self, n_channel, n_out, input_height, input_width, n_action_classes):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.dropout_conv = nn.Dropout2d(p=0.2)
        self.dropout_fc = nn.Dropout(p=0.5)

        def conv_block(in_channels, out_channels, kernel_size, stride, apply_bn=True):
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2),
                nn.BatchNorm2d(out_channels) if apply_bn else nn.Identity(),
                self.relu,
                self.dropout_conv
            ]
            return nn.Sequential(*layers)

        self.conv1 = conv_block(n_channel, 32, kernel_size=5, stride=2)
        self.conv2 = conv_block(32, 48, kernel_size=3, stride=1)
        self.conv3 = conv_block(48, 64, kernel_size=3, stride=2)
        self.conv4 = conv_block(64, 96, kernel_size=3, stride=1)
        self.conv5 = conv_block(96, 128, kernel_size=3, stride=2)
        self.conv6 = conv_block(128, 160, kernel_size=3, stride=1)
        self.conv7 = conv_block(160, 192, kernel_size=3, stride=1)
        self.conv8 = conv_block(192, 256, kernel_size=3, stride=1)

        with torch.no_grad():
            dummy_input = torch.zeros(1, n_channel, input_height, input_width)
            x = self.conv1(dummy_input)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.conv5(x)
            x = self.conv6(x)
            x = self.conv7(x)
            x = self.conv8(x)
            x = self.flatten(x)
            flattened_size = x.shape[1]

        self.fc1 = nn.Linear(flattened_size, 512)
        self.fc2 = nn.Linear(512, 512)

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 256),
                self.relu,
                nn.Linear(256, n_out)
            ) for _ in range(n_action_classes)
        ])

        self.cnn_layer = nn.Sequential(
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4,
            self.conv5,
            self.conv6,
            self.conv7,
            self.conv8,
            self.flatten
        )

    def forward(self, image, action_onehot):
        features = self.cnn_layer(image)
        x = self.relu(self.fc1(features))
        x = self.dropout_fc(x)
        fc_out = self.relu(self.fc2(x))

        batch_size = image.size(0)
        action_indices = torch.argmax(action_onehot, dim=1)

        output = torch.zeros(batch_size, self.branches[0][-1].out_features, device=image.device, dtype=fc_out.dtype)
        for idx, branch in enumerate(self.branches):
            selected_idx = (action_indices == idx).nonzero().squeeze(1)
            if selected_idx.numel() > 0:
                output[selected_idx] = branch(fc_out[selected_idx])

        return output


class ModelEvaluator:
    def __init__(self, model_path, dataset_dir, input_size=(88, 200), n_action_classes=4, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.n_action_classes = n_action_classes
        
        # Load model
        print(f"Loading model from: {model_path}")
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        
        # Load dataset
        print(f"Loading dataset from: {dataset_dir}")
        self.dataset = ImitationDataset(
            dataset_dir=dataset_dir,
            input_size=input_size,
            shift_aug=False,  # Disable augmentation for evaluation
            yaw_aug=False,    # Disable augmentation for evaluation
            n_action_classes=n_action_classes
        )
        
        self.dataloader = DataLoader(self.dataset, batch_size=32, shuffle=False, num_workers=4)
        
        # Initialize results storage
        self.predictions = []
        self.ground_truths = []
        self.actions = []
        self.image_ids = []
        
    def evaluate(self):
        """Perform model evaluation"""
        print("Evaluating model...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.dataloader, desc="Evaluating")):
                images, action_onehots, targets = [x.to(self.device) for x in batch]
                
                predictions = self.model(images, action_onehots)
                
                # Store results
                self.predictions.extend(predictions.cpu().numpy().flatten())
                self.ground_truths.extend(targets.cpu().numpy().flatten())
                self.actions.extend(torch.argmax(action_onehots, dim=1).cpu().numpy())
                
                # Generate image IDs for this batch
                start_idx = batch_idx * self.dataloader.batch_size
                batch_size = len(images)
                batch_ids = [f"{start_idx + i:05d}" for i in range(batch_size)]
                self.image_ids.extend(batch_ids)
        
        # Convert to numpy arrays for analysis
        self.predictions = np.array(self.predictions)
        self.ground_truths = np.array(self.ground_truths)
        self.actions = np.array(self.actions)
        
        print(f"Evaluation completed. Analyzed {len(self.predictions)} samples.")
        
    def compute_metrics(self):
        """Compute comprehensive evaluation metrics"""
        print("Computing evaluation metrics...")
        
        # Basic metrics
        mae = mean_absolute_error(self.ground_truths, self.predictions)
        mse = mean_squared_error(self.ground_truths, self.predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.ground_truths, self.predictions)
        
        # Correlation coefficient
        correlation, p_value = pearsonr(self.ground_truths, self.predictions)
        
        # Temporal smoothness (if sequential data)
        temporal_smoothness_gt = np.mean(np.abs(np.diff(self.ground_truths)))
        temporal_smoothness_pred = np.mean(np.abs(np.diff(self.predictions)))
        
        # Zero-crossing frequency
        zero_cross_gt = np.sum(np.diff(np.sign(self.ground_truths)) != 0)
        zero_cross_pred = np.sum(np.diff(np.sign(self.predictions)) != 0)
        zero_cross_freq_gt = zero_cross_gt / len(self.ground_truths)
        zero_cross_freq_pred = zero_cross_pred / len(self.predictions)
        
        # Sign agreement (positive/negative consistency)
        sign_agreement = np.mean(np.sign(self.ground_truths) == np.sign(self.predictions))
        
        # Maximum absolute error
        max_error = np.max(np.abs(self.predictions - self.ground_truths))
        
        # 95th percentile error
        percentile_95_error = np.percentile(np.abs(self.predictions - self.ground_truths), 95)
        
        metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R²_Score': r2,
            'Correlation': correlation,
            'Correlation_P_Value': p_value,
            'Temporal_Smoothness_GT': temporal_smoothness_gt,
            'Temporal_Smoothness_Pred': temporal_smoothness_pred,
            'Zero_Cross_Freq_GT': zero_cross_freq_gt,
            'Zero_Cross_Freq_Pred': zero_cross_freq_pred,
            'Sign_Agreement': sign_agreement,
            'Max_Absolute_Error': max_error,
            '95th_Percentile_Error': percentile_95_error
        }
        
        return metrics
    
    def compute_action_specific_metrics(self):
        """Compute metrics for each action class"""
        print("Computing action-specific metrics...")
        
        action_metrics = {}
        
        for action_class in range(self.n_action_classes):
            mask = self.actions == action_class
            if np.sum(mask) == 0:
                continue
                
            gt_action = self.ground_truths[mask]
            pred_action = self.predictions[mask]
            
            mae = mean_absolute_error(gt_action, pred_action)
            mse = mean_squared_error(gt_action, pred_action)
            rmse = np.sqrt(mse)
            r2 = r2_score(gt_action, pred_action)
            
            if len(gt_action) > 1:
                correlation, p_value = pearsonr(gt_action, pred_action)
            else:
                correlation, p_value = 0.0, 1.0
            
            action_metrics[f'Action_{action_class}'] = {
                'Sample_Count': np.sum(mask),
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'R²_Score': r2,
                'Correlation': correlation,
                'Max_Error': np.max(np.abs(pred_action - gt_action))
            }
        
        return action_metrics
    
    def create_visualizations(self, output_dir):
        """Create comprehensive visualizations"""
        print("Creating visualizations...")
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Time series plot
        plt.figure(figsize=(15, 6))
        indices = np.arange(len(self.ground_truths))
        plt.plot(indices, self.ground_truths, label='Ground Truth', alpha=0.7, linewidth=1)
        plt.plot(indices, self.predictions, label='Predictions', alpha=0.7, linewidth=1)
        plt.title('Time Series: Ground Truth vs Predictions')
        plt.xlabel('Sample Index (Image ID)')
        plt.ylabel('Angular Velocity')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'time_series_plot.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Scatter plot (GT vs Pred)
        plt.figure(figsize=(10, 10))
        plt.scatter(self.ground_truths, self.predictions, alpha=0.6, s=20)
        
        # Perfect prediction line
        min_val = min(np.min(self.ground_truths), np.min(self.predictions))
        max_val = max(np.max(self.ground_truths), np.max(self.predictions))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        plt.xlabel('Ground Truth')
        plt.ylabel('Predictions')
        plt.title('Scatter Plot: Ground Truth vs Predictions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'scatter_plot.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Error distribution histogram
        errors = self.predictions - self.ground_truths
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(np.mean(errors), color='red', linestyle='--', linewidth=2, label=f'Mean Error: {np.mean(errors):.4f}')
        plt.axvline(np.median(errors), color='green', linestyle='--', linewidth=2, label=f'Median Error: {np.median(errors):.4f}')
        plt.xlabel('Prediction Error (Pred - GT)')
        plt.ylabel('Frequency')
        plt.title('Error Distribution Histogram')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'error_histogram.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Absolute error over time
        abs_errors = np.abs(errors)
        plt.figure(figsize=(15, 6))
        plt.plot(indices, abs_errors, alpha=0.7, linewidth=1)
        plt.axhline(np.mean(abs_errors), color='red', linestyle='--', linewidth=2, label=f'Mean Abs Error: {np.mean(abs_errors):.4f}')
        plt.xlabel('Sample Index (Image ID)')
        plt.ylabel('Absolute Error')
        plt.title('Absolute Error Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'absolute_error_plot.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Action-specific scatter plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for action_class in range(min(self.n_action_classes, 4)):
            mask = self.actions == action_class
            if np.sum(mask) == 0:
                continue
                
            ax = axes[action_class]
            ax.scatter(self.ground_truths[mask], self.predictions[mask], alpha=0.6, s=20)
            
            # Perfect prediction line
            gt_action = self.ground_truths[mask]
            pred_action = self.predictions[mask]
            min_val = min(np.min(gt_action), np.min(pred_action))
            max_val = max(np.max(gt_action), np.max(pred_action))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
            
            ax.set_xlabel('Ground Truth')
            ax.set_ylabel('Predictions')
            ax.set_title(f'Action Class {action_class} (n={np.sum(mask)})')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'action_specific_scatter.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. Box plot of errors by action
        fig, ax = plt.subplots(figsize=(10, 6))
        error_by_action = []
        action_labels = []
        
        for action_class in range(self.n_action_classes):
            mask = self.actions == action_class
            if np.sum(mask) > 0:
                error_by_action.append(errors[mask])
                action_labels.append(f'Action {action_class}')
        
        if error_by_action:
            ax.boxplot(error_by_action, labels=action_labels)
            ax.set_ylabel('Prediction Error')
            ax.set_title('Error Distribution by Action Class')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'error_boxplot_by_action.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to: {output_dir}")
    
    def save_results(self, output_dir, metrics, action_metrics):
        """Save detailed results to files"""
        print("Saving results...")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save overall metrics
        with open(os.path.join(output_dir, 'evaluation_metrics.txt'), 'w') as f:
            f.write("=== Overall Evaluation Metrics ===\n")
            f.write(f"Dataset size: {len(self.predictions)} samples\n")
            f.write(f"Model device: {self.device}\n\n")
            
            for metric_name, value in metrics.items():
                f.write(f"{metric_name}: {value:.6f}\n")
            
            f.write("\n=== Action-Specific Metrics ===\n")
            for action_name, action_data in action_metrics.items():
                f.write(f"\n{action_name}:\n")
                for metric_name, value in action_data.items():
                    if metric_name == 'Sample_Count':
                        f.write(f"  {metric_name}: {value}\n")
                    else:
                        f.write(f"  {metric_name}: {value:.6f}\n")
        
        # Save detailed data as CSV
        results_df = pd.DataFrame({
            'Image_ID': self.image_ids,
            'Ground_Truth': self.ground_truths,
            'Prediction': self.predictions,
            'Action_Class': self.actions,
            'Absolute_Error': np.abs(self.predictions - self.ground_truths),
            'Error': self.predictions - self.ground_truths
        })
        
        results_df.to_csv(os.path.join(output_dir, 'detailed_results.csv'), index=False)
        
        print(f"Results saved to: {output_dir}")
    
    def print_summary(self, metrics, action_metrics):
        """Print evaluation summary"""
        print("\n" + "="*60)
        print("MODEL EVALUATION SUMMARY")
        print("="*60)
        
        print(f"Dataset size: {len(self.predictions)} samples")
        print(f"Model device: {self.device}")
        
        print("\n📊 OVERALL METRICS:")
        print(f"  MAE (Mean Absolute Error):     {metrics['MAE']:.6f}")
        print(f"  RMSE (Root Mean Square Error): {metrics['RMSE']:.6f}")
        print(f"  R² Score:                      {metrics['R²_Score']:.6f}")
        print(f"  Correlation:                   {metrics['Correlation']:.6f}")
        print(f"  Sign Agreement:                {metrics['Sign_Agreement']:.6f}")
        print(f"  Max Absolute Error:            {metrics['Max_Absolute_Error']:.6f}")
        
        print("\n📈 BEHAVIORAL METRICS:")
        print(f"  Temporal Smoothness (GT):      {metrics['Temporal_Smoothness_GT']:.6f}")
        print(f"  Temporal Smoothness (Pred):    {metrics['Temporal_Smoothness_Pred']:.6f}")
        print(f"  Zero-Cross Frequency (GT):     {metrics['Zero_Cross_Freq_GT']:.6f}")
        print(f"  Zero-Cross Frequency (Pred):   {metrics['Zero_Cross_Freq_Pred']:.6f}")
        
        print("\n🔍 ACTION-SPECIFIC PERFORMANCE:")
        for action_name, action_data in action_metrics.items():
            print(f"  {action_name} (n={action_data['Sample_Count']}):")
            print(f"    MAE: {action_data['MAE']:.6f}")
            print(f"    R²:  {action_data['R²_Score']:.6f}")
            print(f"    Max Error: {action_data['Max_Error']:.6f}")
        
        print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description='Evaluate imitation learning model')
    parser.add_argument('model_path', type=str, help='Path to trained model (.pt file)')
    parser.add_argument('dataset_dir', type=str, help='Path to dataset directory (contains images/, angle/, action/)')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', 
                        help='Output directory for results (default: evaluation_results)')
    parser.add_argument('--input_height', type=int, default=88, help='Input image height')
    parser.add_argument('--input_width', type=int, default=200, help='Input image width')
    parser.add_argument('--n_action_classes', type=int, default=4, help='Number of action classes')
    
    args = parser.parse_args()
    
    # Check if paths exist
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        return
    
    if not os.path.exists(args.dataset_dir):
        print(f"Error: Dataset directory not found: {args.dataset_dir}")
        return
    
    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        model_path=args.model_path,
        dataset_dir=args.dataset_dir,
        input_size=(args.input_height, args.input_width),
        n_action_classes=args.n_action_classes
    )
    
    # Perform evaluation
    evaluator.evaluate()
    
    # Compute metrics
    overall_metrics = evaluator.compute_metrics()
    action_metrics = evaluator.compute_action_specific_metrics()
    
    # Create visualizations
    evaluator.create_visualizations(output_dir)
    
    # Save results
    evaluator.save_results(output_dir, overall_metrics, action_metrics)
    
    # Print summary
    evaluator.print_summary(overall_metrics, action_metrics)
    
    print(f"\n✅ Evaluation completed successfully!")
    print(f"📁 All results saved to: {output_dir}")


if __name__ == '__main__':
    main()