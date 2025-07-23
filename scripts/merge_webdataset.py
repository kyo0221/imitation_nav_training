#!/usr/bin/env python3
"""
WebDataset Merger for Imitation Learning
Merges multiple webdataset directories into a single unified webdataset
Usage: python3 merge_webdataset.py dataset1/ dataset2/ [dataset3/ dataset4/ dataset5/] -o output_dataset/
"""

import os
import sys
import json
import glob
import time
import argparse
from pathlib import Path
import numpy as np
from io import BytesIO

try:
    import webdataset as wds
    WEBDATASET_AVAILABLE = True
except ImportError:
    WEBDATASET_AVAILABLE = False

def merge_webdatasets(dataset_paths, output_path, samples_per_shard=1000, enable_compression=True):
    """
    Merge multiple webdatasets into a single unified webdataset.
    All samples are combined and renumbered sequentially.
    Preserves numpy array format for images (RGB uint8).
    """
    if not WEBDATASET_AVAILABLE:
        print("Error: webdataset library is required. Install with: pip install webdataset")
        return False
    
    # Validate input paths and extract webdataset directories
    valid_datasets = []
    for dataset_path in dataset_paths:
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            print(f"Error: Dataset path does not exist: {dataset_path}")
            return False
        
        # Check for webdataset subdirectory (following train.py pattern)
        webdataset_dir = dataset_path / 'webdataset'
        if not webdataset_dir.exists():
            print(f"Error: webdataset subdirectory not found in: {dataset_path}")
            return False
        
        # Check for webdataset files
        tar_files = list(webdataset_dir.glob("*.tar")) + list(webdataset_dir.glob("*.tar.gz"))
        if not tar_files:
            print(f"Error: No webdataset files found in: {webdataset_dir}")
            return False
        
        valid_datasets.append((webdataset_dir, tar_files))
    
    # Create output directory structure
    output_path = Path(output_path)
    output_webdataset_dir = output_path / 'webdataset'
    output_webdataset_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Merging {len(valid_datasets)} datasets into: {output_webdataset_dir}")
    
    # Merge all datasets
    sample_count = 0
    
    # Determine file extension
    file_ext = ".tar.gz" if enable_compression else ".tar"
    
    # Initialize shard writer using WebDataset's ShardWriter pattern
    shard_pattern = str(output_webdataset_dir / f"shard_%06d{file_ext}")
    sink = wds.ShardWriter(shard_pattern, maxcount=samples_per_shard)
    
    # Collect statistics
    stats = {
        'total_samples': 0,
        'datasets': [],
        'merge_time': time.time()
    }
    
    def handle_sample_format(sample):
        """Extract image and metadata from webdataset sample"""
        # Handle numpy format images (preserving original format)
        img_array = None
        if "npy" in sample:
            img_data = sample["npy"]
            if isinstance(img_data, bytes):
                img_array = np.load(BytesIO(img_data))
            else:
                img_array = img_data
        else:
            print(f"Warning: No numpy image data found in sample, skipping")
            return None, None
        
        # Extract metadata
        metadata = None
        action_data = None
        
        if "metadata.json" in sample:
            metadata_raw = sample["metadata.json"]
            if isinstance(metadata_raw, (str, bytes)):
                metadata = json.loads(metadata_raw)
            else:
                metadata = metadata_raw
                
        if "action.json" in sample:
            action_raw = sample["action.json"]
            if isinstance(action_raw, (str, bytes)):
                action_data = json.loads(action_raw)
            else:
                action_data = action_raw
        
        if metadata is None or action_data is None:
            print(f"Warning: Missing metadata or action data in sample, skipping")
            return None, None
            
        return img_array, metadata, action_data
    
    for dataset_idx, (dataset_dir, tar_files) in enumerate(valid_datasets):
        print(f"Processing dataset {dataset_idx + 1}/{len(valid_datasets)}: {dataset_dir}")
        dataset_samples = 0
        
        # Create dataset from tar files - use absolute paths
        tar_file_paths = [str(f) for f in tar_files]
        dataset = wds.WebDataset(tar_file_paths, shardshuffle=False)
        
        for sample in dataset:
            img_array, metadata, action_data = handle_sample_format(sample)
            
            if img_array is None or metadata is None or action_data is None:
                continue
            
            # Convert numpy array back to bytes for storage
            img_buffer = BytesIO()
            np.save(img_buffer, img_array)
            img_data = img_buffer.getvalue()
            
            # Write sample with new sequential key, preserving numpy format
            sample_key = f"{sample_count:06d}"
            
            sample_data = {
                "__key__": sample_key,
                "npy": img_data,  # Keep numpy format
                "metadata.json": json.dumps(metadata),
                "action.json": json.dumps(action_data)
            }
            
            sink.write(sample_data)
            
            sample_count += 1
            dataset_samples += 1
        
        stats['datasets'].append({
            'path': str(dataset_dir),
            'samples': dataset_samples
        })
        
        print(f"  Processed {dataset_samples} samples from {dataset_dir}")
    
    # Close shard writer
    sink.close()
    
    # Update statistics
    stats['total_samples'] = sample_count
    stats['merge_time'] = time.time() - stats['merge_time']
    
    # Save statistics
    stats_path = output_path / 'merge_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nMerge completed successfully!")
    print(f"Total samples: {sample_count}")
    print(f"Output directory: {output_webdataset_dir}")
    print(f"Statistics saved to: {stats_path}")
    print(f"Processing time: {stats['merge_time']:.2f} seconds")
    print(f"Image format preserved: RGB numpy arrays (uint8)")
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description='Merge multiple webdataset directories into a single unified webdataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 merge_webdataset.py dataset1/ dataset2/ -o merged_dataset/
  python3 merge_webdataset.py dataset1/ dataset2/ dataset3/ dataset4/ dataset5/ -o output/
        """
    )
    
    parser.add_argument('datasets', nargs='+', help='Paths to dataset directories (2-5 datasets supported, each should contain webdataset/ subdirectory)')
    parser.add_argument('-o', '--output', required=True, help='Output directory path')
    parser.add_argument('--samples-per-shard', type=int, default=1000, 
                       help='Number of samples per shard (default: 1000)')
    parser.add_argument('--no-compression', action='store_true', 
                       help='Disable compression (use .tar instead of .tar.gz)')
    
    args = parser.parse_args()
    
    # Validate number of datasets
    if len(args.datasets) < 2:
        print("Error: At least 2 datasets are required for merging")
        sys.exit(1)
    
    if len(args.datasets) > 5:
        print("Error: Maximum 5 datasets are supported")
        sys.exit(1)
    
    print(f"Merging {len(args.datasets)} datasets:")
    for i, dataset in enumerate(args.datasets, 1):
        print(f"  {i}. {dataset}")
    
    success = merge_webdatasets(
        args.datasets, 
        args.output,
        args.samples_per_shard,
        not args.no_compression
    )
    
    if not success:
        sys.exit(1)
    
    print("\nMerge operation completed successfully!")

if __name__ == "__main__":
    main()