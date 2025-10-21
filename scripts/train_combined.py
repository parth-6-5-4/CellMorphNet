"""
Professional training pipeline for CellMorphNet using combined datasets.
Supports BCCD, LISC, and BloodMNIST with advanced training strategies.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.backbones import get_model
from src.train import Trainer


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)


class CombinedDatasetConfig:
    """Configuration for combined dataset training."""
    
    def __init__(self):
        self.datasets = {
            'bloodmnist': {
                'path': 'data/raw/bloodmnist',
                'enabled': True,
                'weight': 1.0,
                'num_classes': 8,
                'classes': ['basophil', 'eosinophil', 'erythroblast', 'ig', 
                           'lymphocyte', 'monocyte', 'neutrophil', 'platelet']
            },
            'bccd': {
                'path': 'data/processed/bccd',
                'enabled': False,
                'weight': 1.5,
                'num_classes': 3,
                'classes': ['RBC', 'WBC', 'Platelets']
            },
            'lisc': {
                'path': 'data/processed/lisc',
                'enabled': False,
                'weight': 1.2,
                'num_classes': 6,
                'classes': ['Basophil', 'Eosinophil', 'Lymphocyte', 
                           'Monocyte', 'Neutrophil', 'Mixed']
            }
        }
    
    def get_enabled_datasets(self):
        """Get list of enabled datasets."""
        return [name for name, config in self.datasets.items() if config['enabled']]
    
    def enable_dataset(self, name: str):
        """Enable a dataset."""
        if name in self.datasets:
            self.datasets[name]['enabled'] = True
            logger.info(f"Enabled dataset: {name}")
    
    def disable_dataset(self, name: str):
        """Disable a dataset."""
        if name in self.datasets:
            self.datasets[name]['enabled'] = False
            logger.info(f"Disabled dataset: {name}")


def check_dataset_availability(config: CombinedDatasetConfig) -> dict:
    """
    Check which datasets are available on disk.
    
    Returns:
        Dictionary with dataset availability status
    """
    availability = {}
    
    for name, dataset_config in config.datasets.items():
        path = Path(dataset_config['path'])
        train_dir = path / 'train'
        
        is_available = train_dir.exists() and any(train_dir.iterdir())
        availability[name] = is_available
        
        if is_available:
            num_classes = len([d for d in train_dir.iterdir() if d.is_dir()])
            logger.info(f"✓ {name}: Available at {path} ({num_classes} classes)")
        else:
            logger.warning(f"✗ {name}: Not found at {path}")
    
    return availability


def get_data_transforms(img_size: int = 224, augment: bool = True):
    """
    Get data transforms for training and validation.
    
    Args:
        img_size: Target image size
        augment: Whether to apply augmentation
    
    Returns:
        Tuple of (train_transform, val_transform)
    """
    # Normalization values (ImageNet)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=45),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            normalize
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize
    ])
    
    return train_transform, val_transform


def load_combined_datasets(config: CombinedDatasetConfig, img_size: int = 224, 
                          augment: bool = True):
    """
    Load and combine multiple datasets.
    
    Args:
        config: Dataset configuration
        img_size: Target image size
        augment: Whether to apply augmentation
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_names)
    """
    train_transform, val_transform = get_data_transforms(img_size, augment)
    
    train_datasets = []
    val_datasets = []
    test_datasets = []
    all_class_names = []
    
    enabled_datasets = config.get_enabled_datasets()
    
    if not enabled_datasets:
        raise ValueError("No datasets enabled! Please enable at least one dataset.")
    
    logger.info(f"Loading {len(enabled_datasets)} datasets...")
    
    for dataset_name in enabled_datasets:
        dataset_config = config.datasets[dataset_name]
        dataset_path = Path(dataset_config['path'])
        
        # Load datasets
        train_dir = dataset_path / 'train'
        val_dir = dataset_path / 'val'
        test_dir = dataset_path / 'test'
        
        if not train_dir.exists():
            logger.warning(f"Skipping {dataset_name}: train directory not found")
            continue
        
        # Create datasets
        train_dataset = ImageFolder(str(train_dir), transform=train_transform)
        val_dataset = ImageFolder(str(val_dir), transform=val_transform) if val_dir.exists() else None
        test_dataset = ImageFolder(str(test_dir), transform=val_transform) if test_dir.exists() else None
        
        # Add to lists
        train_datasets.append(train_dataset)
        if val_dataset:
            val_datasets.append(val_dataset)
        if test_dataset:
            test_datasets.append(test_dataset)
        
        # Collect class names
        class_names = train_dataset.classes
        all_class_names.extend([f"{dataset_name}_{cls}" for cls in class_names])
        
        logger.info(f"  {dataset_name}: {len(train_dataset)} train, "
                   f"{len(val_dataset) if val_dataset else 0} val, "
                   f"{len(test_dataset) if test_dataset else 0} test")
    
    # Combine datasets
    if len(train_datasets) > 1:
        train_combined = ConcatDataset(train_datasets)
        val_combined = ConcatDataset(val_datasets) if val_datasets else None
        test_combined = ConcatDataset(test_datasets) if test_datasets else None
        logger.info(f"Combined datasets: {len(train_combined)} train samples")
    else:
        train_combined = train_datasets[0]
        val_combined = val_datasets[0] if val_datasets else None
        test_combined = test_datasets[0] if test_datasets else None
    
    # Create data loaders
    train_loader = DataLoader(
        train_combined,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=False  # Disable for MPS
    )
    
    val_loader = DataLoader(
        val_combined,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=False
    ) if val_combined else None
    
    test_loader = DataLoader(
        test_combined,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=False
    ) if test_combined else None
    
    return train_loader, val_loader, test_loader, all_class_names


def setup_training_environment(args):
    """
    Setup training environment with device selection and logging.
    
    Args:
        args: Command line arguments
    
    Returns:
        torch.device object
    """
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Select device
    if torch.cuda.is_available() and not args.force_cpu:
        device = torch.device('cuda')
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available() and not args.force_cpu:
        device = torch.device('mps')
        logger.info("Using MPS (Apple Silicon) device for acceleration")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU device")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_path = output_dir / 'training_config.json'
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    logger.info(f"Saved training configuration to {config_path}")
    
    return device


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(
        description='Train CellMorphNet on combined datasets',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset arguments
    parser.add_argument('--use-bloodmnist', action='store_true', default=True,
                       help='Use BloodMNIST dataset')
    parser.add_argument('--use-bccd', action='store_true',
                       help='Use BCCD dataset')
    parser.add_argument('--use-lisc', action='store_true',
                       help='Use LISC dataset')
    parser.add_argument('--bloodmnist-samples', type=int, default=None,
                       help='Number of BloodMNIST samples (None = all)')
    
    # Model arguments
    parser.add_argument('--backbone', type=str, default='efficientnet_lite0',
                       choices=['efficientnet_lite0', 'mobilenetv3_small', 'resnet18', 'resnet34'],
                       help='Backbone architecture')
    parser.add_argument('--num-classes', type=int, default=8,
                       help='Number of output classes')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained weights')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--accumulation-steps', type=int, default=2,
                       help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--img-size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--augment', action='store_true', default=True,
                       help='Apply data augmentation')
    
    # Hardware arguments
    parser.add_argument('--force-cpu', action='store_true',
                       help='Force CPU usage (disable GPU/MPS)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='models/combined_exp',
                       help='Output directory for checkpoints and logs')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Print header
    logger.info("=" * 80)
    logger.info("CellMorphNet Combined Dataset Training")
    logger.info("=" * 80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup environment
    device = setup_training_environment(args)
    
    # Configure datasets
    dataset_config = CombinedDatasetConfig()
    
    # Enable/disable datasets based on arguments
    if not args.use_bloodmnist:
        dataset_config.disable_dataset('bloodmnist')
    if args.use_bccd:
        dataset_config.enable_dataset('bccd')
    if args.use_lisc:
        dataset_config.enable_dataset('lisc')
    
    # Check dataset availability
    availability = check_dataset_availability(dataset_config)
    
    # Disable unavailable datasets
    for name, available in availability.items():
        if not available and dataset_config.datasets[name]['enabled']:
            dataset_config.disable_dataset(name)
            logger.warning(f"Disabled {name} dataset (not available)")
    
    # Load datasets
    try:
        train_loader, val_loader, test_loader, class_names = load_combined_datasets(
            dataset_config,
            img_size=args.img_size,
            augment=args.augment
        )
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        logger.info("\nTo prepare datasets, run:")
        logger.info("  1. BloodMNIST: python scripts/download_bloodmnist.py --sample_limit 2000")
        logger.info("  2. BCCD: python -c \"from src.data import BCCDPreprocessor; BCCDPreprocessor('archive (1)/BCCD', 'data/processed/bccd').process_dataset()\"")
        logger.info("  3. LISC: python -c \"from src.data import LISCPreprocessor; LISCPreprocessor('LISC Database/Main Dataset', 'data/processed/lisc').process_dataset()\"")
        return
    
    # Determine number of classes
    if len(dataset_config.get_enabled_datasets()) == 1:
        # Single dataset
        dataset_name = dataset_config.get_enabled_datasets()[0]
        num_classes = dataset_config.datasets[dataset_name]['num_classes']
    else:
        # Multiple datasets - use class names length
        num_classes = len(class_names)
    
    logger.info(f"Total classes: {num_classes}")
    
    # Create model
    logger.info(f"Creating model with backbone: {args.backbone}")
    model = get_model(
        backbone=args.backbone,
        num_classes=num_classes,
        pretrained=args.pretrained
    )
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {trainable_params:,} trainable / {total_params:,} total")
    
    # Training configuration
    config = {
        'backbone': args.backbone,
        'num_classes': num_classes,
        'img_size': args.img_size,
        'batch_size': args.batch_size,
        'accumulation_steps': args.accumulation_steps,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'epochs': args.epochs,
        'device': str(device),
        'datasets': dataset_config.get_enabled_datasets()
    }
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        output_dir=args.output_dir,
        class_names=class_names
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.checkpoint and os.path.exists(args.checkpoint):
        logger.info(f"Resuming from checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        logger.info(f"Resuming from epoch {start_epoch}")
    
    # Train model
    logger.info("=" * 80)
    logger.info("Starting training...")
    logger.info("=" * 80)
    
    try:
        history = trainer.train(epochs=args.epochs, start_epoch=start_epoch)
        
        logger.info("=" * 80)
        logger.info("Training complete!")
        logger.info(f"Best validation F1: {max(history['val_f1']):.4f}")
        logger.info(f"Final validation accuracy: {history['val_acc'][-1]:.4f}")
        logger.info(f"Saved to: {args.output_dir}")
        logger.info("=" * 80)
        
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        logger.info("Saving checkpoint...")
        trainer.save_checkpoint(epoch=trainer.current_epoch, is_best=False)
        logger.info("Checkpoint saved. Training can be resumed with --checkpoint")
    
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    
    # Final evaluation on test set
    if test_loader is not None:
        logger.info("\nEvaluating on test set...")
        test_loss, test_acc, test_f1, test_metrics = trainer.validate(test_loader)
        logger.info(f"Test Loss: {test_loss:.4f}")
        logger.info(f"Test Accuracy: {test_acc:.4f}")
        logger.info(f"Test F1: {test_f1:.4f}")
        
        # Save test results
        test_results = {
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_f1': test_f1,
            'per_class_metrics': test_metrics
        }
        
        results_path = Path(args.output_dir) / 'test_results.json'
        with open(results_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        logger.info(f"Test results saved to {results_path}")
    
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
