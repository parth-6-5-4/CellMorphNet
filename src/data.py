"""
Data loading and preprocessing utilities for CellMorphNet.
Handles BCCD, LISC, and BloodMNIST datasets.
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import json
import shutil

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class BCCDPreprocessor:
    """Preprocess BCCD dataset from VOC format to ImageFolder format."""
    
    def __init__(self, bccd_root: str, output_root: str):
        """
        Args:
            bccd_root: Path to BCCD directory containing Annotations and JPEGImages
            output_root: Path to output processed images
        """
        self.bccd_root = Path(bccd_root)
        self.annotations_dir = self.bccd_root / 'Annotations'
        self.images_dir = self.bccd_root / 'JPEGImages'
        self.output_root = Path(output_root)
        
    def parse_voc_xml(self, xml_path: str) -> List[Dict]:
        """Parse VOC format XML annotation."""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        filename = root.find('filename').text
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        
        objects = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            bbox = obj.find('bndbox')
            xmin = int(float(bbox.find('xmin').text))
            ymin = int(float(bbox.find('ymin').text))
            xmax = int(float(bbox.find('xmax').text))
            ymax = int(float(bbox.find('ymax').text))
            
            objects.append({
                'filename': filename,
                'class': name,
                'bbox': (xmin, ymin, xmax, ymax),
                'img_width': width,
                'img_height': height
            })
        
        return objects
    
    def process_dataset(self, min_size: int = 20, target_size: int = 128):
        """
        Process BCCD dataset: crop cells and organize into ImageFolder structure.
        
        Args:
            min_size: Minimum bounding box size to keep
            target_size: Target size for cropped images
        """
        print("Processing BCCD dataset...")
        
        # Parse all annotations
        all_objects = []
        for xml_file in self.annotations_dir.glob('*.xml'):
            objects = self.parse_voc_xml(str(xml_file))
            all_objects.extend(objects)
        
        print(f"Found {len(all_objects)} annotated objects")
        
        # Count classes
        class_counts = {}
        for obj in all_objects:
            cls = obj['class']
            class_counts[cls] = class_counts.get(cls, 0) + 1
        
        print(f"Class distribution: {class_counts}")
        
        # Create output directories
        for split in ['train', 'val', 'test']:
            for cls in class_counts.keys():
                (self.output_root / split / cls).mkdir(parents=True, exist_ok=True)
        
        # Crop and save images
        class_file_counts = {cls: {'train': 0, 'val': 0, 'test': 0} for cls in class_counts.keys()}
        
        for obj in all_objects:
            img_path = self.images_dir / obj['filename']
            if not img_path.exists():
                continue
            
            # Load image
            img = Image.open(img_path).convert('RGB')
            
            # Crop bounding box
            xmin, ymin, xmax, ymax = obj['bbox']
            
            # Check minimum size
            if (xmax - xmin) < min_size or (ymax - ymin) < min_size:
                continue
            
            # Add small padding
            pad = 5
            xmin = max(0, xmin - pad)
            ymin = max(0, ymin - pad)
            xmax = min(obj['img_width'], xmax + pad)
            ymax = min(obj['img_height'], ymax + pad)
            
            cropped = img.crop((xmin, ymin, xmax, ymax))
            
            # Resize to target size
            cropped = cropped.resize((target_size, target_size), Image.BILINEAR)
            
            # Determine split (80/10/10)
            rand = np.random.random()
            if rand < 0.8:
                split = 'train'
            elif rand < 0.9:
                split = 'val'
            else:
                split = 'test'
            
            # Save
            cls = obj['class']
            idx = class_file_counts[cls][split]
            out_path = self.output_root / split / cls / f"{cls}_{idx:04d}.png"
            cropped.save(out_path)
            
            class_file_counts[cls][split] += 1
        
        print(f"\n✓ BCCD processing complete!")
        print(f"Output directory: {self.output_root}")
        for split in ['train', 'val', 'test']:
            total = sum(class_file_counts[cls][split] for cls in class_counts.keys())
            print(f"  {split}: {total} images")


class LISCPreprocessor:
    """Preprocess LISC dataset to ImageFolder format."""
    
    def __init__(self, lisc_root: str, output_root: str):
        """
        Args:
            lisc_root: Path to LISC "Main Dataset" directory
            output_root: Path to output processed images
        """
        self.lisc_root = Path(lisc_root)
        self.output_root = Path(output_root)
    
    def process_dataset(self, target_size: int = 128, train_ratio: float = 0.8, val_ratio: float = 0.1):
        """
        Process LISC dataset: copy and organize images with train/val/test split.
        
        Args:
            target_size: Target size for images
            train_ratio: Ratio of training data
            val_ratio: Ratio of validation data
        """
        print("Processing LISC dataset...")
        
        # Get all classes (subdirectories)
        classes = [d.name for d in self.lisc_root.iterdir() if d.is_dir()]
        print(f"Found classes: {classes}")
        
        # Create output directories
        for split in ['train', 'val', 'test']:
            for cls in classes:
                (self.output_root / split / cls).mkdir(parents=True, exist_ok=True)
        
        # Process each class
        for cls in classes:
            class_dir = self.lisc_root / cls
            images = list(class_dir.glob('*.bmp')) + list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
            
            print(f"\nClass '{cls}': {len(images)} images")
            
            if len(images) == 0:
                continue
            
            # Split data
            train_val, test = train_test_split(images, test_size=(1 - train_ratio - val_ratio), random_state=42)
            train, val = train_test_split(train_val, test_size=val_ratio / (train_ratio + val_ratio), random_state=42)
            
            splits = {'train': train, 'val': val, 'test': test}
            
            for split_name, split_files in splits.items():
                for idx, img_path in enumerate(split_files):
                    # Load and resize
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize((target_size, target_size), Image.BILINEAR)
                    
                    # Save
                    out_path = self.output_root / split_name / cls / f"{cls}_{idx:04d}.png"
                    img.save(out_path)
                
                print(f"  {split_name}: {len(split_files)} images")
        
        print(f"\n✓ LISC processing complete!")
        print(f"Output directory: {self.output_root}")


def get_data_loaders(
    data_dir: str,
    batch_size: int = 16,
    img_size: int = 224,
    num_workers: int = 2,
    augmentation_transform: Optional[transforms.Compose] = None
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Create train, val, and test data loaders.
    
    Args:
        data_dir: Root directory with train/val/test subdirectories
        batch_size: Batch size
        img_size: Image size
        num_workers: Number of data loading workers
        augmentation_transform: Optional custom augmentation transform
    
    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    
    # Default transforms
    if augmentation_transform is None:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = augmentation_transform
    
    val_test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
    val_dataset = ImageFolder(os.path.join(data_dir, 'val'), transform=val_test_transform)
    test_dataset = ImageFolder(os.path.join(data_dir, 'test'), transform=val_test_transform)
    
    # Get class names
    class_names = train_dataset.classes
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Data loaders created:")
    print(f"  Train: {len(train_dataset)} images, {len(train_loader)} batches")
    print(f"  Val: {len(val_dataset)} images, {len(val_loader)} batches")
    print(f"  Test: {len(test_dataset)} images, {len(test_loader)} batches")
    print(f"  Classes ({len(class_names)}): {class_names}")
    
    return train_loader, val_loader, test_loader, class_names


def get_class_weights(data_dir: str, num_classes: int) -> torch.Tensor:
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        data_dir: Root directory with train subdirectory
        num_classes: Number of classes
    
    Returns:
        Class weights tensor
    """
    train_dataset = ImageFolder(os.path.join(data_dir, 'train'))
    
    # Count samples per class
    class_counts = np.zeros(num_classes)
    for _, label in train_dataset.samples:
        class_counts[label] += 1
    
    # Calculate weights (inverse frequency)
    total = sum(class_counts)
    class_weights = total / (num_classes * class_counts)
    
    return torch.FloatTensor(class_weights)


if __name__ == '__main__':
    """Example usage for preprocessing datasets."""
    
    # Example: Process BCCD
    # bccd_processor = BCCDPreprocessor(
    #     bccd_root='archive (1)/BCCD',
    #     output_root='data/processed/bccd'
    # )
    # bccd_processor.process_dataset(target_size=128)
    
    # Example: Process LISC
    # lisc_processor = LISCPreprocessor(
    #     lisc_root='LISC Database/Main Dataset',
    #     output_root='data/processed/lisc'
    # )
    # lisc_processor.process_dataset(target_size=128)
    
    # Example: Create data loaders
    # train_loader, val_loader, test_loader, classes = get_data_loaders(
    #     data_dir='data/raw/bloodmnist',
    #     batch_size=32,
    #     img_size=224
    # )
    
    print("Data utilities loaded. See docstrings for usage examples.")
