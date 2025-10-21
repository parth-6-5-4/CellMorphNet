"""
Morphology attention module for CellMorphNet.
Adds spatial and channel attention to capture cell shape and appearance features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Channel attention module (what features to focus on)."""
    
    def __init__(self, in_channels: int, reduction: int = 16):
        """
        Args:
            in_channels: Number of input channels
            reduction: Reduction ratio for bottleneck
        """
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial attention module (where to focus in the image)."""
    
    def __init__(self, kernel_size: int = 7):
        """
        Args:
            kernel_size: Convolution kernel size
        """
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Compute channel-wise statistics
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and convolve
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    Combines channel and spatial attention.
    
    Reference: "CBAM: Convolutional Block Attention Module" (Woo et al., ECCV 2018)
    """
    
    def __init__(self, in_channels: int, reduction: int = 16, kernel_size: int = 7):
        """
        Args:
            in_channels: Number of input channels
            reduction: Channel reduction ratio
            kernel_size: Spatial attention kernel size
        """
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        # Apply channel attention
        x = x * self.channel_attention(x)
        
        # Apply spatial attention
        x = x * self.spatial_attention(x)
        
        return x


class MorphologyAttentionBlock(nn.Module):
    """
    Custom morphology-aware attention block.
    Designed to capture both texture (appearance) and shape (geometric) features of cells.
    """
    
    def __init__(self, in_channels: int, reduction: int = 16):
        """
        Args:
            in_channels: Number of input channels
            reduction: Reduction ratio
        """
        super().__init__()
        
        # Texture branch (fine-grained appearance)
        self.texture_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True)
        )
        
        # Shape branch (coarse geometric features)
        self.shape_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 2, 3, padding=2, dilation=2),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True)
        )
        
        # Attention mechanism
        self.attention = CBAM(in_channels, reduction)
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Extract texture and shape features
        texture = self.texture_branch(x)
        shape = self.shape_branch(x)
        
        # Concatenate
        combined = torch.cat([texture, shape], dim=1)
        
        # Apply attention
        attended = self.attention(combined)
        
        # Fuse with residual connection
        out = self.fusion(attended) + x
        
        return out


class AttentionHead(nn.Module):
    """
    Classification head with morphology attention.
    """
    
    def __init__(self, backbone_out_channels: int, num_classes: int, use_attention: bool = True):
        """
        Args:
            backbone_out_channels: Number of channels from backbone
            num_classes: Number of output classes
            use_attention: Whether to use morphology attention
        """
        super().__init__()
        
        self.use_attention = use_attention
        
        if use_attention:
            self.attention_block = MorphologyAttentionBlock(backbone_out_channels)
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(backbone_out_channels, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        if self.use_attention:
            x = self.attention_block(x)
        
        x = self.global_pool(x)
        x = self.classifier(x)
        
        return x


class CellMorphNet(nn.Module):
    """
    Complete CellMorphNet model with backbone and morphology attention.
    """
    
    def __init__(self, backbone: nn.Module, backbone_out_channels: int, num_classes: int, use_attention: bool = True):
        """
        Args:
            backbone: Feature extraction backbone (e.g., EfficientNet, MobileNet)
            backbone_out_channels: Number of output channels from backbone
            num_classes: Number of output classes
            use_attention: Whether to use morphology attention
        """
        super().__init__()
        
        self.backbone = backbone
        self.head = AttentionHead(backbone_out_channels, num_classes, use_attention)
    
    def forward(self, x):
        features = self.backbone(x)
        out = self.head(features)
        return out


def create_cellmorphnet_from_backbone(
    backbone_name: str,
    num_classes: int,
    pretrained: bool = True,
    use_attention: bool = True
):
    """
    Create CellMorphNet from a backbone name.
    
    Args:
        backbone_name: Name of backbone ('efficientnet', 'mobilenet', 'resnet18')
        num_classes: Number of classes
        pretrained: Whether to use pretrained backbone
        use_attention: Whether to add morphology attention
    
    Returns:
        CellMorphNet model
    """
    import torchvision.models as models
    
    # Get backbone and extract features
    if 'efficientnet' in backbone_name.lower():
        model = models.efficientnet_b0(weights='IMAGENET1K_V1' if pretrained else None)
        backbone = model.features
        out_channels = 1280
    elif 'mobilenet' in backbone_name.lower():
        model = models.mobilenet_v3_small(weights='IMAGENET1K_V1' if pretrained else None)
        backbone = model.features
        out_channels = 576
    elif 'resnet18' in backbone_name.lower():
        model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        # Remove FC and avgpool
        backbone = nn.Sequential(*list(model.children())[:-2])
        out_channels = 512
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")
    
    # Create CellMorphNet
    cellmorphnet = CellMorphNet(backbone, out_channels, num_classes, use_attention)
    
    return cellmorphnet


if __name__ == '__main__':
    """Test attention modules."""
    
    print("Testing morphology attention modules...")
    print("=" * 60)
    
    # Test CBAM
    print("\nTesting CBAM:")
    cbam = CBAM(in_channels=64)
    x = torch.randn(2, 64, 28, 28)
    out = cbam(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    assert out.shape == x.shape
    
    # Test MorphologyAttentionBlock
    print("\nTesting MorphologyAttentionBlock:")
    morph_att = MorphologyAttentionBlock(in_channels=128)
    x = torch.randn(2, 128, 14, 14)
    out = morph_att(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    assert out.shape == x.shape
    
    # Test CellMorphNet
    print("\nTesting CellMorphNet (EfficientNet + Attention):")
    model = create_cellmorphnet_from_backbone('efficientnet', num_classes=8, pretrained=False, use_attention=True)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    assert out.shape == (2, 8)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    print("\n✓ All attention modules tested successfully!")
