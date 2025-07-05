#!/usr/bin/env python3

import os
import sys
import shutil
import argparse
from pathlib import Path

def merge_datasets(dataset1_path, dataset2_path, output_path=None):
    """
    Merge two datasets containing images, action, and angle directories.
    All files are combined and renumbered sequentially.
    """
    dataset1 = Path(dataset1_path)
    dataset2 = Path(dataset2_path)
    
    if not dataset1.exists() or not dataset2.exists():
        print("Error: One or both dataset paths do not exist")
        return False
    
    # Check required directories exist
    required_dirs = ['images', 'action', 'angle']
    for dataset in [dataset1, dataset2]:
        for dir_name in required_dirs:
            if not (dataset / dir_name).exists():
                print(f"Error: Required directory '{dir_name}' not found in {dataset}")
                return False
    
    # Set output path
    if output_path is None:
        output_path = dataset1.parent / "merged_dataset"
    else:
        output_path = Path(output_path)
    
    # Create output directories
    output_path.mkdir(exist_ok=True)
    for dir_name in required_dirs:
        (output_path / dir_name).mkdir(exist_ok=True)
    
    print(f"Merging datasets:")
    print(f"  Dataset 1: {dataset1}")
    print(f"  Dataset 2: {dataset2}")
    print(f"  Output: {output_path}")
    
    # Process each directory type
    file_counter = 1
    
    for dir_name in required_dirs:
        print(f"\nProcessing {dir_name} directory...")
        
        # Get all files from both datasets
        files1 = sorted((dataset1 / dir_name).glob("*"))
        files2 = sorted((dataset2 / dir_name).glob("*"))
        
        # Determine file extension
        if files1:
            ext = files1[0].suffix
        elif files2:
            ext = files2[0].suffix
        else:
            print(f"Warning: No files found in {dir_name} directories")
            continue
        
        # Reset counter for each directory type
        counter = 1
        
        # Copy files from dataset1
        for file_path in files1:
            if file_path.is_file():
                new_name = f"{counter:05d}{ext}"
                shutil.copy2(file_path, output_path / dir_name / new_name)
                counter += 1
        
        # Copy files from dataset2
        for file_path in files2:
            if file_path.is_file():
                new_name = f"{counter:05d}{ext}"
                shutil.copy2(file_path, output_path / dir_name / new_name)
                counter += 1
        
        print(f"  Merged {len(files1)} + {len(files2)} = {counter-1} files")
    
    print(f"\nDataset merge completed successfully!")
    print(f"Merged dataset saved to: {output_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Merge two imitation navigation datasets')
    parser.add_argument('dataset1', help='Path to first dataset directory')
    parser.add_argument('dataset2', help='Path to second dataset directory')
    parser.add_argument('-o', '--output', help='Output directory path (optional)')
    
    args = parser.parse_args()
    
    success = merge_datasets(args.dataset1, args.dataset2, args.output)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()