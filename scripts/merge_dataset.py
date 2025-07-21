#!/usr/bin/env python3
"""
WebDataset Merger for Imitation Learning
Merges multiple webdataset directories into a single unified webdataset
Usage: python3 merge_dataset.py dataset1/webdataset/ dataset2/webdataset/ -o merged_dataset/
"""

import os
import sys
import json
import glob
import time
import argparse
from pathlib import Path

try:
    import webdataset as wds
    WEBDATASET_AVAILABLE = True
except ImportError:
    WEBDATASET_AVAILABLE = False

def merge_webdatasets(dataset_paths, output_path=None, samples_per_shard=1000, enable_compression=True):
    """
    Merge multiple webdatasets into a single unified webdataset.
    All samples are combined and renumbered sequentially.
    """
    if not WEBDATASET_AVAILABLE:
        print("Error: webdataset library is required. Install with: pip install webdataset")
        return False
    
    # Validate input paths
    valid_datasets = []
    for dataset_path in dataset_paths:
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            print(f"Error: Dataset path does not exist: {dataset_path}")
            return False
        
        # Check for webdataset files
        tar_files = list(dataset_path.glob("*.tar")) + list(dataset_path.glob("*.tar.gz"))
        if not tar_files:
            print(f"Error: No webdataset files found in: {dataset_path}")
            return False
        
        valid_datasets.append((dataset_path, tar_files))
    
    # Set output path
    if output_path is None:
        output_path = Path("merged_dataset")
    else:
        output_path = Path(output_path)
    
    # Create output directory structure (following data_collector_node.py pattern)
    output_path.mkdir(exist_ok=True)
    webdataset_dir = output_path / "webdataset"
    webdataset_dir.mkdir(exist_ok=True)
    
    print(f"Merging {len(valid_datasets)} webdatasets:")
    for i, (dataset_path, tar_files) in enumerate(valid_datasets):
        print(f"  Dataset {i+1}: {dataset_path} ({len(tar_files)} shards)")
    print(f"  Output: {output_path}")
    print(f"  WebDataset directory: {webdataset_dir}")
    print(f"  Samples per shard: {samples_per_shard}")
    print(f"  Compression: {'enabled' if enable_compression else 'disabled'}")
    
    # Initialize output shard writer
    if enable_compression:
        shard_pattern = str(webdataset_dir / "shard_%06d.tar.gz")
    else:
        shard_pattern = str(webdataset_dir / "shard_%06d.tar")
    
    shard_writer = wds.ShardWriter(shard_pattern, maxcount=samples_per_shard)
    
    # Statistics
    total_samples = 0
    datasets_stats = []
    action_counts = {"roadside": 0, "straight": 0, "left": 0, "right": 0}
    action_to_index = {"roadside": 0, "straight": 1, "left": 2, "right": 3}
    index_to_action = {0: "roadside", 1: "straight", 2: "left", 3: "right"}
    
    try:
        # Process each dataset
        for dataset_idx, (dataset_path, tar_files) in enumerate(valid_datasets):
            print(f"\nProcessing dataset {dataset_idx + 1}: {dataset_path}")
            
            # Load dataset statistics if available
            stats_file = dataset_path / "dataset_stats.json"
            dataset_stats = {}
            if stats_file.exists():
                try:
                    with open(stats_file, 'r') as f:
                        dataset_stats = json.load(f)
                    print(f"  Loaded stats: {dataset_stats.get('total_samples', 'unknown')} samples")
                except:
                    print("  Warning: Could not load dataset stats")
            
            # Create webdataset from tar files
            tar_files_str = [str(f) for f in tar_files]
            dataset = wds.WebDataset(tar_files_str)
            
            dataset_samples = 0
            dataset_action_counts = {"roadside": 0, "straight": 0, "left": 0, "right": 0}
            
            # Process each sample
            for sample in dataset:
                try:
                    # Extract sample data
                    sample_key = f"{total_samples:06d}"
                    
                    # Get action data
                    action_data = {}
                    if 'action.json' in sample:
                        action_json = json.loads(sample['action.json'].decode('utf-8'))
                        action_value = action_json.get('action', 0)
                        angle = action_json.get('angle', 0.0)
                    else:
                        action_value = 0
                        angle = 0.0
                    
                    # Update action counts
                    action_name = index_to_action.get(action_value, "roadside")
                    action_counts[action_name] += 1
                    dataset_action_counts[action_name] += 1
                    
                    # Get metadata
                    metadata = {}
                    if 'metadata.json' in sample:
                        metadata = json.loads(sample['metadata.json'].decode('utf-8'))
                    
                    # Update metadata with merge information
                    metadata.update({
                        'original_dataset': str(dataset_path),
                        'original_key': sample.get('__key__', 'unknown'),
                        'merged_timestamp': time.time(),
                        'dataset_index': dataset_idx
                    })
                    
                    # Prepare sample for output
                    output_sample = {
                        "__key__": sample_key,
                        "metadata.json": json.dumps(metadata),
                        "action.json": json.dumps({"action": action_value, "angle": angle})
                    }
                    
                    # Copy image data
                    if 'npy' in sample:
                        output_sample['npy'] = sample['npy']
                    else:
                        print(f"Warning: No image data found in sample {sample.get('__key__', 'unknown')}")
                        continue
                    
                    # Write sample to output
                    shard_writer.write(output_sample)
                    
                    dataset_samples += 1
                    total_samples += 1
                    
                    if dataset_samples % 100 == 0:
                        print(f"  Processed {dataset_samples} samples from dataset {dataset_idx + 1}")
                        
                except Exception as e:
                    print(f"  Error processing sample: {e}")
                    continue
            
            # Store dataset statistics
            datasets_stats.append({
                "path": str(dataset_path),
                "samples": dataset_samples,
                "action_distribution": dict(dataset_action_counts),
                "original_stats": dataset_stats
            })
            
            print(f"  Completed dataset {dataset_idx + 1}: {dataset_samples} samples")
        
        # Close the shard writer
        shard_writer.close()
        
        # Create merged dataset statistics
        merged_stats = {
            "total_samples": total_samples,
            "merged_datasets": len(valid_datasets),
            "samples_per_shard": samples_per_shard,
            "compression_enabled": enable_compression,
            "save_format": "numpy",
            "action_distribution": dict(action_counts),
            "merge_timestamp": time.time(),
            "individual_datasets": datasets_stats,
            "output_directory": str(output_path),
            "webdataset_directory": str(webdataset_dir)
        }
        
        # Save merged statistics
        stats_file = webdataset_dir / "dataset_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(merged_stats, f, indent=2)
        
        print(f"\nMerge completed successfully!")
        print(f"Total samples: {total_samples}")
        print(f"Action distribution: {dict(action_counts)}")
        print(f"Output directory: {output_path}")
        print(f"WebDataset directory: {webdataset_dir}")
        print(f"Statistics saved to: {stats_file}")
        
        return True
        
    except Exception as e:
        print(f"Error during merge: {e}")
        if shard_writer:
            try:
                shard_writer.close()
            except:
                pass
        return False

def main():
    parser = argparse.ArgumentParser(description='Merge multiple webdataset directories into a single unified webdataset')
    parser.add_argument('datasets', nargs='+', help='Paths to webdataset directories (supports multiple datasets)')
    parser.add_argument('-o', '--output', help='Output directory path (default: merged_dataset)')
    parser.add_argument('--samples-per-shard', type=int, default=1000, 
                      help='Number of samples per shard (default: 1000)')
    parser.add_argument('--no-compression', action='store_true', 
                      help='Disable compression (use .tar instead of .tar.gz)')
    
    args = parser.parse_args()
    
    # Validate minimum dataset count
    if len(args.datasets) < 2:
        print("Error: At least 2 datasets are required for merging")
        sys.exit(1)
    
    enable_compression = not args.no_compression
    
    success = merge_webdatasets(
        args.datasets, 
        args.output, 
        args.samples_per_shard, 
        enable_compression
    )
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()