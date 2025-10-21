"""
Download a subset of BloodMNIST dataset from MedMNIST.
This script downloads a limited number of samples to keep storage requirements low.
"""

import os
import urllib.request
import numpy as np
from PIL import Image
import json
from pathlib import Path

def download_bloodmnist(output_dir='data/raw/bloodmnist', sample_limit=None):
    """
    Download BloodMNIST dataset and extract samples.
    
    Args:
        output_dir: Directory to save the dataset
        sample_limit: Number of samples to extract per split (train/val/test).
                     If None, extracts all samples.
    """
    
    print("=" * 60)
    print("BloodMNIST Dataset Downloader")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Download URLs
    train_url = "https://zenodo.org/records/10519652/files/bloodmnist.npz?download=1"
    
    npz_path = os.path.join(output_dir, 'bloodmnist.npz')
    
    # Download the dataset if not exists
    if not os.path.exists(npz_path):
        print(f"\nDownloading BloodMNIST dataset...")
        print(f"URL: {train_url}")
        print("This may take a few minutes...")
        
        try:
            urllib.request.urlretrieve(train_url, npz_path)
            print(f"✓ Downloaded successfully to {npz_path}")
        except Exception as e:
            print(f"✗ Error downloading: {e}")
            print("\nAlternative: Please manually download from:")
            print("https://zenodo.org/records/10519652/files/bloodmnist.npz")
            print(f"and place it in {output_dir}/")
            return
    else:
        print(f"\n✓ Dataset already exists at {npz_path}")
    
    # Load the dataset
    print("\nLoading dataset...")
    data = np.load(npz_path)
    
    # Class names for BloodMNIST
    class_names = [
        'basophil',
        'eosinophil', 
        'erythroblast',
        'ig',  # immature granulocytes
        'lymphocyte',
        'monocyte',
        'neutrophil',
        'platelet'
    ]
    
    # Create class mapping
    class_info = {i: name for i, name in enumerate(class_names)}
    
    # Save class info
    with open(os.path.join(output_dir, 'class_info.json'), 'w') as f:
        json.dump(class_info, f, indent=2)
    
    print(f"\nDataset info:")
    print(f"  Classes: {len(class_names)}")
    print(f"  Class names: {', '.join(class_names)}")
    
    # Extract and save images
    for split in ['train', 'val', 'test']:
        images = data[f'{split}_images']
        labels = data[f'{split}_labels'].flatten()
        
        print(f"\n{split.upper()} split:")
        print(f"  Total samples: {len(images)}")
        print(f"  Image shape: {images.shape[1:]}")
        
        # Create directories for each class
        split_dir = os.path.join(output_dir, split)
        for class_name in class_names:
            os.makedirs(os.path.join(split_dir, class_name), exist_ok=True)
        
        # Save images (sample a subset or all)
        class_counts = {i: 0 for i in range(len(class_names))}
        
        if sample_limit is None:
            # Extract all samples
            samples_per_class = len(images) // len(class_names)
            print(f"  Extracting ALL samples (~{samples_per_class} per class)...")
        else:
            # Limit samples
            num_samples = min(sample_limit, len(images))
            samples_per_class = num_samples // len(class_names)
            print(f"  Extracting {samples_per_class} samples per class (total ~{samples_per_class * len(class_names)})...")
        
        saved_count = 0
        for idx in range(len(images)):
            label = labels[idx]
            
            # Check if we need more samples from this class
            if class_counts[label] < samples_per_class:
                img_array = images[idx]
                
                # Convert to PIL Image
                if len(img_array.shape) == 2:
                    img = Image.fromarray(img_array, mode='L')
                else:
                    img = Image.fromarray(img_array, mode='RGB')
                
                # Save image
                class_name = class_names[label]
                img_filename = f"{class_name}_{class_counts[label]:04d}.png"
                img_path = os.path.join(split_dir, class_name, img_filename)
                img.save(img_path)
                
                class_counts[label] += 1
                saved_count += 1
            
            # Check if we have enough samples
            if all(count >= samples_per_class for count in class_counts.values()):
                break
        
        print(f"  ✓ Saved {saved_count} images to {split_dir}/")
        print(f"  Distribution: {dict((class_names[k], v) for k, v in class_counts.items())}")
    
    print("\n" + "=" * 60)
    print("✓ BloodMNIST subset download complete!")
    print("=" * 60)
    print(f"\nDataset location: {output_dir}")
    print(f"Structure:")
    print(f"  {output_dir}/")
    print(f"    ├── train/")
    print(f"    │   ├── basophil/")
    print(f"    │   ├── eosinophil/")
    print(f"    │   └── ...")
    print(f"    ├── val/")
    print(f"    └── test/")
    print(f"\nYou can now use this data with PyTorch ImageFolder!")
    print("\nNote: This is a subset for quick prototyping.")
    print("For full dataset training, increase sample_limit parameter.")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Download BloodMNIST dataset')
    parser.add_argument('--output_dir', type=str, default='data/raw/bloodmnist',
                       help='Output directory for dataset')
    parser.add_argument('--sample_limit', type=int, default=None,
                       help='Number of samples to extract per split (None = all samples)')
    
    args = parser.parse_args()
    
    download_bloodmnist(args.output_dir, args.sample_limit)
