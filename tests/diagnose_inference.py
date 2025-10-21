"""
Diagnostic script to analyze inference issues.
Tests uploaded images and shows detailed predictions.
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from src.models.backbones import get_model


def load_model(checkpoint_path):
    """Load model from checkpoint."""
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    class_names = checkpoint['class_names']
    
    model = get_model(
        backbone=config['backbone'],
        num_classes=config['num_classes'],
        pretrained=False
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, config, class_names, device


def preprocess_image(image_path, img_size=224):
    """Preprocess image - exactly as in training."""
    # Load image
    img = Image.open(image_path).convert('RGB')
    
    print(f"\nImage: {Path(image_path).name}")
    print(f"  Original size: {img.size}")
    print(f"  Original mode: {img.mode}")
    
    # Check if image is very small
    if img.size[0] < 64 or img.size[1] < 64:
        print(f"  WARNING: Image is very small! This may affect quality.")
    
    # Standard ImageNet preprocessing (same as training)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Also create non-normalized version for visualization
    transform_viz = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    
    tensor = transform(img).unsqueeze(0)
    tensor_viz = transform_viz(img)
    
    return tensor, tensor_viz, img


def analyze_image_statistics(image_path):
    """Analyze image statistics."""
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    
    print(f"\n  Image Statistics:")
    print(f"    Shape: {img_array.shape}")
    print(f"    Dtype: {img_array.dtype}")
    print(f"    Min: {img_array.min()}, Max: {img_array.max()}")
    print(f"    Mean: {img_array.mean():.2f}")
    print(f"    Std: {img_array.std():.2f}")
    
    # Check each channel
    for i, channel in enumerate(['R', 'G', 'B']):
        channel_data = img_array[:, :, i]
        print(f"    {channel}: mean={channel_data.mean():.2f}, std={channel_data.std():.2f}")
    
    return img_array


def predict_with_details(model, image_tensor, class_names, device):
    """Make prediction with detailed output."""
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        
        # Get top-5 predictions
        top5_probs, top5_indices = torch.topk(probabilities, k=min(5, len(class_names)))
        
        print(f"\n  Predictions (Top 5):")
        for i, (prob, idx) in enumerate(zip(top5_probs, top5_indices)):
            marker = "★" if i == 0 else " "
            print(f"    {marker} {class_names[idx]:15s}: {prob:.4f} ({prob*100:.2f}%)")
        
        predicted_class = class_names[top5_indices[0]]
        confidence = top5_probs[0].item()
        
        return predicted_class, confidence, probabilities.cpu().numpy()


def visualize_results(image_paths, predictions, save_path='test_images_analysis.png'):
    """Visualize test images with predictions."""
    n_images = len(image_paths)
    fig, axes = plt.subplots(1, n_images, figsize=(6*n_images, 6))
    
    if n_images == 1:
        axes = [axes]
    
    for idx, (img_path, pred_info) in enumerate(zip(image_paths, predictions)):
        img = Image.open(img_path).convert('RGB')
        
        axes[idx].imshow(img)
        axes[idx].axis('off')
        
        title = f"Predicted: {pred_info['predicted']}\n"
        title += f"Confidence: {pred_info['confidence']:.2%}"
        
        # Color based on confidence
        color = 'green' if pred_info['confidence'] > 0.8 else 'orange' if pred_info['confidence'] > 0.5 else 'red'
        axes[idx].set_title(title, fontsize=12, fontweight='bold', color=color)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {save_path}")
    plt.close()


def compare_with_training_data(test_image_path, model, class_names, device):
    """Compare test image with training data samples."""
    print("\n" + "="*70)
    print("COMPARING WITH TRAINING DATA")
    print("="*70)
    
    # Check if training data exists
    train_dir = Path('data/raw/bloodmnist_full/train')
    if not train_dir.exists():
        print("Training data not found. Skipping comparison.")
        return
    
    # Get predicted class
    tensor, _, _ = preprocess_image(test_image_path)
    predicted, confidence, _ = predict_with_details(model, tensor, class_names, device)
    
    print(f"\nPredicted class: {predicted}")
    print(f"Looking for samples in: {train_dir / predicted}")
    
    # List some training samples from predicted class
    predicted_class_dir = train_dir / predicted
    if predicted_class_dir.exists():
        samples = list(predicted_class_dir.glob('*.png'))[:5]
        print(f"Found {len(list(predicted_class_dir.glob('*.png')))} training samples")
        print(f"Example files: {[s.name for s in samples]}")
    else:
        print(f"No training data found for class: {predicted}")


def main():
    print("="*70)
    print("CELLMORPHNET INFERENCE DIAGNOSTIC")
    print("="*70)
    
    # Load model
    checkpoint_path = Path('models/bloodmnist_full_exp/checkpoints/best.pth')
    print(f"\nLoading model from: {checkpoint_path}")
    
    model, config, class_names, device = load_model(checkpoint_path)
    
    print(f"\nModel Configuration:")
    print(f"  Backbone: {config['backbone']}")
    print(f"  Image Size: {config['img_size']}")
    print(f"  Num Classes: {config['num_classes']}")
    print(f"  Device: {device}")
    
    print(f"\nClass Names: {class_names}")
    
    # Find test images
    test_dir = Path('test images')
    test_images = sorted(test_dir.glob('*.png'))
    
    if not test_images:
        print(f"\nNo test images found in '{test_dir}'")
        return
    
    print(f"\nFound {len(test_images)} test images:")
    for img in test_images:
        print(f"  - {img.name}")
    
    # Analyze each image
    predictions = []
    
    for img_path in test_images:
        print("\n" + "="*70)
        print(f"ANALYZING: {img_path.name}")
        print("="*70)
        
        # Image statistics
        img_array = analyze_image_statistics(img_path)
        
        # Preprocess and predict
        tensor, tensor_viz, original_img = preprocess_image(img_path, config['img_size'])
        predicted, confidence, probs = predict_with_details(model, tensor, class_names, device)
        
        predictions.append({
            'image': img_path.name,
            'predicted': predicted,
            'confidence': confidence,
            'probabilities': probs
        })
        
        # Compare with training data
        compare_with_training_data(img_path, model, class_names, device)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for i, pred in enumerate(predictions, 1):
        print(f"\n{i}. {pred['image']}")
        print(f"   Predicted: {pred['predicted']}")
        print(f"   Confidence: {pred['confidence']:.2%}")
        
        # Check if prediction is suspicious (low confidence or uniform distribution)
        if pred['confidence'] < 0.5:
            print(f"   ⚠️  WARNING: Low confidence!")
        
        # Check for uniform distribution (confused model)
        probs = pred['probabilities']
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(len(class_names))
        relative_entropy = entropy / max_entropy
        
        if relative_entropy > 0.8:
            print(f"   ⚠️  WARNING: Model is very uncertain (high entropy: {relative_entropy:.2f})")
    
    # Create visualization
    visualize_results(test_images, predictions)
    
    print("\n" + "="*70)
    print("DIAGNOSTIC COMPLETE")
    print("="*70)
    
    # Recommendations
    print("\n📋 RECOMMENDATIONS:")
    print("\n1. Check if test images are:")
    print("   - Blood cell microscopy images (not screenshots/other images)")
    print("   - Similar quality/resolution to training data")
    print("   - Properly cropped (cell in center, minimal background)")
    
    print("\n2. If images are screenshots, they may contain:")
    print("   - UI elements, text, borders")
    print("   - Multiple cells or non-cell regions")
    print("   - Different color profile than training data")
    
    print("\n3. For best results, test images should be:")
    print("   - Single cell microscopy images")
    print("   - 28x28 or larger resolution")
    print("   - RGB color (not grayscale)")
    print("   - Similar staining to BloodMNIST dataset")


if __name__ == '__main__':
    main()
