"""
Training script for CellMorphNet with AMP, gradient accumulation, and checkpointing.
Optimized for Mac M2 with limited memory.
"""

import os
import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

# Import local modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import get_data_loaders, get_class_weights
from src.models.backbones import get_model, count_parameters
from src.augment import AlbumentationsTransform, get_train_augmentation, get_val_test_augmentation


class Trainer:
    """Training manager for CellMorphNet."""
    
    def __init__(self, config):
        """
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = self._get_device()
        
        print(f"Using device: {self.device}")
        
        # Create output directories
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'checkpoints').mkdir(exist_ok=True)
        
        # Initialize model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Print model info
        trainable, total = count_parameters(self.model)
        print(f"\nModel: {config['backbone']}")
        print(f"  Trainable parameters: {trainable:,}")
        print(f"  Total parameters: {total:,}")
        
        # Create data loaders
        self.train_loader, self.val_loader, self.test_loader, self.class_names = self._create_data_loaders()
        
        print(f"\nClasses ({len(self.class_names)}): {self.class_names}")
        
        # Setup loss function
        self.criterion = self._create_criterion()
        
        # Setup optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Setup AMP
        self.scaler = GradScaler() if config.get('use_amp', True) and self.device.type == 'cuda' else None
        
        # Training state
        self.current_epoch = 0
        self.best_val_f1 = 0.0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': []
        }
    
    def _get_device(self):
        """Get training device (CUDA, MPS, or CPU)."""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    def _create_model(self):
        """Create model."""
        return get_model(
            backbone=self.config['backbone'],
            num_classes=self.config['num_classes'],
            pretrained=self.config.get('pretrained', True),
            freeze_backbone=self.config.get('freeze_backbone', False)
        )
    
    def _create_data_loaders(self):
        """Create data loaders."""
        # Get augmentations
        if self.config.get('use_albumentations', False):
            train_aug = AlbumentationsTransform(
                get_train_augmentation(
                    img_size=self.config['img_size'],
                    stain_norm=False
                )
            )
            val_aug = AlbumentationsTransform(
                get_val_test_augmentation(
                    img_size=self.config['img_size'],
                    stain_norm=False
                )
            )
        else:
            train_aug = None
            val_aug = None
        
        return get_data_loaders(
            data_dir=self.config['data_dir'],
            batch_size=self.config['batch_size'],
            img_size=self.config['img_size'],
            num_workers=self.config.get('num_workers', 2),
            augmentation_transform=train_aug
        )
    
    def _create_criterion(self):
        """Create loss function."""
        if self.config.get('use_class_weights', False):
            class_weights = get_class_weights(
                self.config['data_dir'],
                self.config['num_classes']
            )
            class_weights = class_weights.to(self.device)
            print(f"\nUsing class weights: {class_weights.cpu().numpy()}")
            return nn.CrossEntropyLoss(weight=class_weights)
        else:
            return nn.CrossEntropyLoss()
    
    def _create_optimizer(self):
        """Create optimizer."""
        if self.config.get('optimizer', 'adamw').lower() == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config['lr'],
                weight_decay=self.config.get('weight_decay', 1e-4)
            )
        else:
            return optim.SGD(
                self.model.parameters(),
                lr=self.config['lr'],
                momentum=0.9,
                weight_decay=self.config.get('weight_decay', 1e-4)
            )
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        scheduler_type = self.config.get('scheduler', 'cosine')
        
        if scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['epochs'],
                eta_min=1e-6
            )
        elif scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get('step_size', 10),
                gamma=0.1
            )
        else:
            return None
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        accumulation_steps = self.config.get('accumulation_steps', 1)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}/{self.config['epochs']}")
        
        self.optimizer.zero_grad()
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Mixed precision training
            if self.scaler is not None:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    loss = loss / accumulation_steps
                
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss = loss / accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # Statistics
            running_loss += loss.item() * accumulation_steps
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'lr': self.optimizer.param_groups[0]['lr']
            })
        
        # Calculate metrics
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate model."""
        self.model.eval()
        
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validating"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                if self.scaler is not None and self.device.type == 'cuda':
                    with autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        val_loss = running_loss / len(self.val_loader)
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_preds, average=None, zero_division=0
        )
        
        print(f"\nPer-class metrics:")
        for i, class_name in enumerate(self.class_names):
            print(f"  {class_name}: P={precision[i]:.3f}, R={recall[i]:.3f}, F1={f1[i]:.3f}, N={support[i]}")
        
        return val_loss, val_acc, val_f1
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_f1': self.best_val_f1,
            'history': self.history,
            'config': self.config,
            'class_names': self.class_names
        }
        
        # Save latest checkpoint
        checkpoint_path = self.output_dir / 'checkpoints' / 'latest.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / 'checkpoints' / 'best.pth'
            torch.save(checkpoint, best_path)
            print(f"  Saved best model (F1: {self.best_val_f1:.4f})")
    
    def train(self):
        """Main training loop."""
        print(f"\n{'=' * 60}")
        print("Starting training...")
        print(f"{'=' * 60}\n")
        
        for epoch in range(self.config['epochs']):
            self.current_epoch = epoch
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, val_f1 = self.validate()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)
            
            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{self.config['epochs']}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
            
            # Save checkpoint
            is_best = val_f1 > self.best_val_f1
            if is_best:
                self.best_val_f1 = val_f1
            self.save_checkpoint(is_best)
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Unfreeze backbone after initial epochs
            if self.config.get('freeze_backbone', False) and epoch == self.config.get('unfreeze_after', 3):
                print("\n  Unfreezing backbone...")
                for param in self.model.parameters():
                    param.requires_grad = True
                # Recreate optimizer with lower LR
                self.config['lr'] = self.config['lr'] * 0.1
                self.optimizer = self._create_optimizer()
                self.scheduler = self._create_scheduler()
        
        # Save training history
        history_path = self.output_dir / 'history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\n{'=' * 60}")
        print(f"Training complete!")
        print(f"Best validation F1: {self.best_val_f1:.4f}")
        print(f"Saved to: {self.output_dir}")
        print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(description='Train CellMorphNet')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='models/experiment', help='Output directory')
    
    # Model arguments
    parser.add_argument('--backbone', type=str, default='efficientnet_lite0',
                       choices=['efficientnet_lite0', 'mobilenet_v3_small', 'mobilenet_v3_large', 'resnet18', 'resnet34'],
                       help='Backbone architecture')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of classes')
    parser.add_argument('--pretrained', action='store_true', default=True, help='Use pretrained weights')
    parser.add_argument('--freeze_backbone', action='store_true', help='Freeze backbone initially')
    parser.add_argument('--unfreeze_after', type=int, default=3, help='Unfreeze backbone after N epochs')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'sgd'], help='Optimizer')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'step', 'none'], help='LR scheduler')
    
    # Data arguments
    parser.add_argument('--img_size', type=int, default=224, help='Image size')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of data loading workers')
    parser.add_argument('--use_class_weights', action='store_true', help='Use class weights for imbalanced data')
    parser.add_argument('--use_albumentations', action='store_true', help='Use Albumentations for augmentation')
    
    # Other
    parser.add_argument('--use_amp', action='store_true', default=True, help='Use automatic mixed precision')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Convert args to config dict
    config = vars(args)
    
    # Create trainer and train
    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
