"""
Inference and explainability utilities for CellMorphNet.
Includes Grad-CAM for visualizing model attention.
"""

import os
import argparse
from pathlib import Path
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

# Import local modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.backbones import get_model


class GradCAM:
    """
    Grad-CAM: Gradient-weighted Class Activation Mapping.
    Generates heatmaps showing which regions of the image the model focuses on.
    """
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Args:
            model: Trained model
            target_layer: Target layer for Grad-CAM (usually last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        """Hook to save forward activation."""
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward gradient."""
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_image: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_image: Input image tensor (1, C, H, W)
            target_class: Target class index (if None, use predicted class)
        
        Returns:
            Heatmap as numpy array (H, W)
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_image)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        class_score = output[0, target_class]
        class_score.backward()
        
        # Get gradients and activations
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Calculate weights (global average pooling of gradients)
        weights = gradients.mean(dim=(1, 2))  # (C,)
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU (only positive influences)
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.cpu().numpy()
    
    def overlay_heatmap(
        self,
        heatmap: np.ndarray,
        original_image: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Overlay heatmap on original image.
        
        Args:
            heatmap: Grad-CAM heatmap (H, W)
            original_image: Original image (H, W, 3) in RGB
            alpha: Transparency for overlay
            colormap: OpenCV colormap
        
        Returns:
            Overlayed image (H, W, 3) in RGB
        """
        # Resize heatmap to match original image
        heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        
        # Convert heatmap to color
        heatmap = (heatmap * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay
        overlayed = (alpha * heatmap + (1 - alpha) * original_image).astype(np.uint8)
        
        return overlayed


def get_target_layer(model, backbone: str):
    """
    Get the target layer for Grad-CAM based on backbone architecture.
    
    Args:
        model: Model
        backbone: Backbone name
    
    Returns:
        Target layer
    """
    if 'efficientnet' in backbone.lower():
        return model.features[-1]
    elif 'mobilenet' in backbone.lower():
        return model.features[-1]
    elif 'resnet' in backbone.lower():
        return model.layer4[-1]
    else:
        raise ValueError(f"Unknown backbone: {backbone}")


def load_checkpoint(checkpoint_path: str, device: torch.device):
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model to
    
    Returns:
        model, config, class_names
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint['config']
    class_names = checkpoint['class_names']
    
    # Create model
    model = get_model(
        backbone=config['backbone'],
        num_classes=config['num_classes'],
        pretrained=False
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch'] + 1}")
    print(f"Best validation F1: {checkpoint['best_val_f1']:.4f}")
    print(f"Classes: {class_names}")
    
    return model, config, class_names


def preprocess_image(image_path: str, img_size: int = 224) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Preprocess image for inference.
    
    Args:
        image_path: Path to image
        img_size: Target image size
    
    Returns:
        Preprocessed tensor, original image (RGB)
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    original = np.array(image)
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    tensor = transform(image).unsqueeze(0)
    
    return tensor, original


def predict_image(
    model: nn.Module,
    image_tensor: torch.Tensor,
    class_names: List[str],
    device: torch.device
) -> Tuple[str, float, torch.Tensor]:
    """
    Predict class for image.
    
    Args:
        model: Model
        image_tensor: Image tensor (1, C, H, W)
        class_names: List of class names
        device: Device
    
    Returns:
        predicted_class, confidence, probabilities
    """
    model.eval()
    
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probabilities, 1)
    
    predicted_class = class_names[pred_idx.item()]
    confidence = confidence.item()
    
    return predicted_class, confidence, probabilities[0]


def visualize_prediction(
    image_path: str,
    model: nn.Module,
    config: dict,
    class_names: List[str],
    device: torch.device,
    save_path: Optional[str] = None,
    use_gradcam: bool = True
):
    """
    Visualize prediction with optional Grad-CAM.
    
    Args:
        image_path: Path to image
        model: Model
        config: Configuration
        class_names: List of class names
        device: Device
        save_path: Optional path to save visualization
        use_gradcam: Whether to generate Grad-CAM
    """
    # Preprocess image
    image_tensor, original_image = preprocess_image(image_path, config['img_size'])
    
    # Predict
    predicted_class, confidence, probabilities = predict_image(model, image_tensor, class_names, device)
    
    # Create figure
    fig, axes = plt.subplots(1, 2 if use_gradcam else 1, figsize=(12 if use_gradcam else 6, 5))
    if not use_gradcam:
        axes = [axes]
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title(f"Prediction: {predicted_class}\nConfidence: {confidence:.2%}")
    axes[0].axis('off')
    
    # Grad-CAM
    if use_gradcam:
        target_layer = get_target_layer(model, config['backbone'])
        gradcam = GradCAM(model, target_layer)
        
        heatmap = gradcam.generate(image_tensor.to(device))
        overlayed = gradcam.overlay_heatmap(heatmap, original_image, alpha=0.5)
        
        axes[1].imshow(overlayed)
        axes[1].set_title("Grad-CAM Visualization")
        axes[1].axis('off')
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    # Print probabilities
    print(f"\nClass probabilities:")
    probs_sorted, indices = torch.sort(probabilities, descending=True)
    for prob, idx in zip(probs_sorted[:5], indices[:5]):
        print(f"  {class_names[idx]}: {prob.item():.2%}")


def batch_inference(
    model: nn.Module,
    image_dir: str,
    config: dict,
    class_names: List[str],
    device: torch.device,
    output_dir: Optional[str] = None,
    use_gradcam: bool = False
):
    """
    Run inference on a directory of images.
    
    Args:
        model: Model
        image_dir: Directory containing images
        config: Configuration
        class_names: List of class names
        device: Device
        output_dir: Optional output directory for visualizations
        use_gradcam: Whether to generate Grad-CAM
    """
    image_dir = Path(image_dir)
    image_files = list(image_dir.glob('*.png')) + list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.jpeg'))
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for image_file in tqdm(image_files, desc="Processing images"):
        try:
            # Preprocess and predict
            image_tensor, _ = preprocess_image(str(image_file), config['img_size'])
            predicted_class, confidence, probabilities = predict_image(model, image_tensor, class_names, device)
            
            results.append({
                'filename': image_file.name,
                'predicted_class': predicted_class,
                'confidence': confidence
            })
            
            # Generate visualization if requested
            if output_dir:
                save_path = output_dir / f"{image_file.stem}_prediction.png"
                visualize_prediction(
                    str(image_file),
                    model,
                    config,
                    class_names,
                    device,
                    save_path=str(save_path),
                    use_gradcam=use_gradcam
                )
        
        except Exception as e:
            print(f"Error processing {image_file.name}: {e}")
    
    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Processed {len(results)} images")
    print(f"{'=' * 60}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='CellMorphNet Inference')
    
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--image', type=str, help='Path to single image for inference')
    parser.add_argument('--image_dir', type=str, help='Path to directory of images')
    parser.add_argument('--output_dir', type=str, help='Output directory for visualizations')
    parser.add_argument('--gradcam', action='store_true', help='Generate Grad-CAM visualizations')
    parser.add_argument('--no_gradcam', action='store_true', help='Disable Grad-CAM')
    
    args = parser.parse_args()
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Load model
    model, config, class_names = load_checkpoint(args.checkpoint, device)
    
    use_gradcam = args.gradcam or (not args.no_gradcam)
    
    # Single image inference
    if args.image:
        print(f"\nProcessing single image: {args.image}")
        visualize_prediction(
            args.image,
            model,
            config,
            class_names,
            device,
            save_path=args.output_dir if args.output_dir else None,
            use_gradcam=use_gradcam
        )
    
    # Batch inference
    elif args.image_dir:
        print(f"\nProcessing directory: {args.image_dir}")
        batch_inference(
            model,
            args.image_dir,
            config,
            class_names,
            device,
            output_dir=args.output_dir,
            use_gradcam=use_gradcam
        )
    
    else:
        print("Please provide either --image or --image_dir")


if __name__ == '__main__':
    main()
