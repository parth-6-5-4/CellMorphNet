"""
Lightweight CNN backbones for CellMorphNet.
Includes EfficientNet-Lite, MobileNetV3, and ResNet variants.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


def get_efficientnet_lite(num_classes: int, pretrained: bool = True, freeze_backbone: bool = False):
    """
    Get EfficientNet-Lite0 model (lightweight variant).
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        freeze_backbone: Whether to freeze backbone layers
    
    Returns:
        Model
    """
    # Use EfficientNet-B0 as EfficientNet-Lite equivalent
    model = models.efficientnet_b0(weights='IMAGENET1K_V1' if pretrained else None)
    
    # Freeze backbone if requested
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
    
    # Replace classifier
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, num_classes)
    )
    
    return model


def get_mobilenet_v3(num_classes: int, variant: str = 'small', pretrained: bool = True, freeze_backbone: bool = False):
    """
    Get MobileNetV3 model.
    
    Args:
        num_classes: Number of output classes
        variant: 'small' or 'large'
        pretrained: Whether to use pretrained weights
        freeze_backbone: Whether to freeze backbone layers
    
    Returns:
        Model
    """
    if variant == 'small':
        model = models.mobilenet_v3_small(weights='IMAGENET1K_V1' if pretrained else None)
    else:
        model = models.mobilenet_v3_large(weights='IMAGENET1K_V1' if pretrained else None)
    
    # Freeze backbone if requested
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
    
    # Replace classifier
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.Hardswish(inplace=True),
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(512, num_classes)
    )
    
    return model


def get_resnet18(num_classes: int, pretrained: bool = True, freeze_backbone: bool = False):
    """
    Get ResNet18 model.
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        freeze_backbone: Whether to freeze backbone layers
    
    Returns:
        Model
    """
    model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
    
    # Freeze backbone if requested
    if freeze_backbone:
        for name, param in model.named_parameters():
            if 'fc' not in name:  # Don't freeze final FC layer
                param.requires_grad = False
    
    # Replace classifier
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes)
    )
    
    return model


def get_resnet34(num_classes: int, pretrained: bool = True, freeze_backbone: bool = False):
    """
    Get ResNet34 model (slightly larger than ResNet18).
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        freeze_backbone: Whether to freeze backbone layers
    
    Returns:
        Model
    """
    model = models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
    
    # Freeze backbone if requested
    if freeze_backbone:
        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
    
    # Replace classifier
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes)
    )
    
    return model


def get_model(backbone: str, num_classes: int, pretrained: bool = True, freeze_backbone: bool = False):
    """
    Factory function to get model by name.
    
    Args:
        backbone: Model name ('efficientnet_lite0', 'mobilenet_v3_small', 'mobilenet_v3_large', 'resnet18', 'resnet34')
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        freeze_backbone: Whether to freeze backbone layers
    
    Returns:
        Model
    """
    backbone = backbone.lower()
    
    if backbone == 'efficientnet_lite0' or backbone == 'efficientnet':
        return get_efficientnet_lite(num_classes, pretrained, freeze_backbone)
    elif backbone == 'mobilenet_v3_small':
        return get_mobilenet_v3(num_classes, 'small', pretrained, freeze_backbone)
    elif backbone == 'mobilenet_v3_large':
        return get_mobilenet_v3(num_classes, 'large', pretrained, freeze_backbone)
    elif backbone == 'resnet18':
        return get_resnet18(num_classes, pretrained, freeze_backbone)
    elif backbone == 'resnet34':
        return get_resnet34(num_classes, pretrained, freeze_backbone)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")


def count_parameters(model):
    """Count trainable and total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


if __name__ == '__main__':
    """Test model creation."""
    
    num_classes = 8
    
    print("Testing model creation...")
    print("=" * 60)
    
    for backbone in ['efficientnet_lite0', 'mobilenet_v3_small', 'resnet18']:
        print(f"\nBackbone: {backbone}")
        model = get_model(backbone, num_classes, pretrained=True, freeze_backbone=False)
        trainable, total = count_parameters(model)
        print(f"  Trainable parameters: {trainable:,}")
        print(f"  Total parameters: {total:,}")
        print(f"  Model size: ~{total * 4 / 1024 / 1024:.2f} MB (FP32)")
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 224, 224)
        output = model(dummy_input)
        print(f"  Output shape: {output.shape}")
        assert output.shape == (1, num_classes), f"Expected output shape (1, {num_classes}), got {output.shape}"
    
    print("\n✓ All models tested successfully!")
