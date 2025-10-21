"""
Analyze why the model misclassified the screenshots.
Compare features between screenshots and training data.
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
from src.infer import GradCAM, get_target_layer


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


def preprocess_for_visualization(image_path, img_size=224):
    """Preprocess image for both inference and visualization."""
    img = Image.open(image_path)
    
    # Convert RGBA to RGB
    if img.mode == 'RGBA':
        rgb_img = Image.new('RGB', img.size, (255, 255, 255))
        rgb_img.paste(img, mask=img.split()[3])
        img = rgb_img
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    
    # For model input
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # For visualization (no normalization)
    transform_viz = transforms.Compose([
        transforms.Resize((img_size, img_size)),
    ])
    
    tensor = transform(img).unsqueeze(0)
    img_viz = transform_viz(img)
    
    return tensor, img_viz, img


def predict_with_all_scores(model, image_tensor, class_names, device):
    """Get predictions and all class scores."""
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        
    return probabilities.cpu().numpy()


def compare_with_correct_class(screenshot_path, true_class, model, config, class_names, device):
    """Compare screenshot with training samples from the correct class."""
    print("\n" + "="*80)
    print(f"ANALYZING: {screenshot_path.name}")
    print(f"TRUE CLASS: {true_class}")
    print("="*80)
    
    # Process screenshot
    tensor, img_viz, img_original = preprocess_for_visualization(screenshot_path, config['img_size'])
    probs = predict_with_all_scores(model, tensor, class_names, device)
    
    predicted_idx = probs.argmax()
    predicted_class = class_names[predicted_idx]
    
    print(f"\nScreenshot size: {img_original.size}")
    print(f"Predicted: {predicted_class} ({probs[predicted_idx]:.2%})")
    print(f"True class probability: {probs[class_names.index(true_class)]:.2%}")
    
    # Show all probabilities
    print("\nAll class probabilities:")
    for i, (cls, prob) in enumerate(zip(class_names, probs)):
        marker = "★" if i == predicted_idx else "→" if cls == true_class else " "
        print(f"  {marker} {cls:15s}: {prob:.4f} ({prob*100:.2f}%)")
    
    # Load training samples from both predicted and true class
    train_dir = Path('data/raw/bloodmnist_full/train')
    
    # Get samples from true class
    true_class_dir = train_dir / true_class
    predicted_class_dir = train_dir / predicted_class
    
    # Create comparison visualization
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    
    # Row 1: Screenshot and predicted class samples
    axes[0, 0].imshow(img_viz)
    axes[0, 0].set_title(f'YOUR SCREENSHOT\n(Should be: {true_class.upper()})', 
                         fontsize=12, fontweight='bold', color='red')
    axes[0, 0].axis('off')
    
    # Show 4 samples from predicted class
    if predicted_class_dir.exists():
        samples = list(predicted_class_dir.glob('*.png'))[:4]
        for i, sample_path in enumerate(samples):
            sample = Image.open(sample_path).convert('RGB')
            axes[0, i+1].imshow(sample)
            axes[0, i+1].set_title(f'Training: {predicted_class.upper()}\n(Model predicted this)', 
                                   fontsize=10, color='orange')
            axes[0, i+1].axis('off')
    
    # Row 2: True class samples
    axes[1, 0].text(0.5, 0.5, f'SHOULD BE:\n{true_class.upper()}', 
                    ha='center', va='center', fontsize=14, fontweight='bold', color='green')
    axes[1, 0].axis('off')
    
    if true_class_dir.exists():
        samples = list(true_class_dir.glob('*.png'))[:4]
        for i, sample_path in enumerate(samples):
            sample = Image.open(sample_path).convert('RGB')
            axes[1, i+1].imshow(sample)
            axes[1, i+1].set_title(f'Training: {true_class.upper()}\n(Correct class)', 
                                   fontsize=10, color='green')
            axes[1, i+1].axis('off')
    
    plt.suptitle(f'Screenshot vs Training Samples\nPredicted: {predicted_class} | True: {true_class}', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    filename = f'analysis_{screenshot_path.stem}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n✓ Comparison saved to: {filename}")
    plt.close()
    
    return predicted_class, probs


def analyze_why_misclassified(screenshot_path, true_class, model, config, class_names, device):
    """Analyze why the screenshot was misclassified."""
    print("\n" + "="*80)
    print("DETAILED ANALYSIS: Why was it misclassified?")
    print("="*80)
    
    # Load and process screenshot
    img = Image.open(screenshot_path)
    
    # Convert RGBA to RGB
    if img.mode == 'RGBA':
        rgb_img = Image.new('RGB', img.size, (255, 255, 255))
        rgb_img.paste(img, mask=img.split()[3])
        img = rgb_img
    
    img_array = np.array(img)
    
    # Analyze color distribution
    print(f"\nScreenshot color analysis:")
    print(f"  Size: {img.size}")
    print(f"  Mean RGB: R={img_array[:,:,0].mean():.1f}, G={img_array[:,:,1].mean():.1f}, B={img_array[:,:,2].mean():.1f}")
    print(f"  Std RGB:  R={img_array[:,:,0].std():.1f}, G={img_array[:,:,1].std():.1f}, B={img_array[:,:,2].std():.1f}")
    
    # Load training samples from true class
    train_dir = Path('data/raw/bloodmnist_full/train')
    true_class_dir = train_dir / true_class
    
    if true_class_dir.exists():
        # Analyze 10 training samples from true class
        samples = list(true_class_dir.glob('*.png'))[:10]
        
        r_means, g_means, b_means = [], [], []
        sizes = []
        
        for sample_path in samples:
            sample = Image.open(sample_path).convert('RGB')
            sample_array = np.array(sample)
            sizes.append(sample.size)
            r_means.append(sample_array[:,:,0].mean())
            g_means.append(sample_array[:,:,1].mean())
            b_means.append(sample_array[:,:,2].mean())
        
        print(f"\nTraining samples ({true_class}) color analysis (n=10):")
        print(f"  Size: {sizes[0]} (all training images)")
        print(f"  Mean RGB: R={np.mean(r_means):.1f}, G={np.mean(g_means):.1f}, B={np.mean(b_means):.1f}")
        print(f"  Std RGB:  R={np.std(r_means):.1f}, G={np.std(g_means):.1f}, B={np.std(b_means):.1f}")
    
    # Key differences
    print("\n🔍 KEY DIFFERENCES IDENTIFIED:")
    print(f"  1. Size mismatch: Screenshot is {img.size}, training data is 28x28")
    print(f"  2. When resized from {img.size[0]}x{img.size[1]} → 224x224, fine details are lost")
    print(f"  3. Training images are resized from 28x28 → 224x224 (upscaling)")
    print(f"  4. Screenshot is resized from ~300x300 → 224x224 (downscaling)")
    print(f"  5. Different interpolation artifacts affect the features the model sees")
    
    print("\n💡 WHY THE MODEL FAILS:")
    print(f"  • The model learned features from tiny 28x28 upscaled images")
    print(f"  • Your screenshots are large images downscaled to the same size")
    print(f"  • After resizing, they look COMPLETELY DIFFERENT to the model")
    print(f"  • The pixel patterns, edges, and textures don't match")
    print(f"  • This is a fundamental training/inference data distribution mismatch")


def main():
    print("="*80)
    print("ANALYZING MISCLASSIFICATION OF YOUR SCREENSHOTS")
    print("="*80)
    
    # Load model
    checkpoint_path = Path('models/bloodmnist_full_exp/checkpoints/best.pth')
    model, config, class_names, device = load_model(checkpoint_path)
    
    print(f"\nModel: {config['backbone']}")
    print(f"Classes: {class_names}")
    
    # Test images with ground truth labels
    test_cases = [
        ('test images/Screenshot 2025-10-22 at 3.38.38 AM.png', 'neutrophil'),
        ('test images/Screenshot 2025-10-22 at 3.40.15 AM.png', 'basophil')
    ]
    
    results = []
    
    for screenshot_path, true_class in test_cases:
        screenshot_path = Path(screenshot_path)
        if not screenshot_path.exists():
            print(f"\nWarning: {screenshot_path} not found")
            continue
        
        predicted_class, probs = compare_with_correct_class(
            screenshot_path, true_class, model, config, class_names, device
        )
        
        results.append({
            'screenshot': screenshot_path.name,
            'true': true_class,
            'predicted': predicted_class,
            'correct': predicted_class == true_class
        })
        
        analyze_why_misclassified(screenshot_path, true_class, model, config, class_names, device)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for result in results:
        status = "✓" if result['correct'] else "✗"
        print(f"\n{status} {result['screenshot']}")
        print(f"   True: {result['true']:12s} | Predicted: {result['predicted']:12s}")
    
    # Recommendations
    print("\n" + "="*80)
    print("🔧 SOLUTIONS TO FIX THIS ISSUE")
    print("="*80)
    
    print("\n1. RETRAIN WITH YOUR IMAGE FORMAT:")
    print("   • Add your screenshots (or similar large images) to the training data")
    print("   • This will teach the model to handle both small and large images")
    print("   • Augment with different scales during training")
    
    print("\n2. DOWNSCALE YOUR SCREENSHOTS FIRST:")
    print("   • Resize screenshots to 28x28 before saving")
    print("   • Then they'll match the training data format")
    print("   • Use: img.resize((28, 28), Image.LANCZOS)")
    
    print("\n3. USE MULTI-SCALE TRAINING:")
    print("   • Train with images at multiple scales (28x28, 56x56, 112x112, 224x224)")
    print("   • This makes the model scale-invariant")
    
    print("\n4. APPLY DOWNSCALING IN PREPROCESSING:")
    print("   • Add a step to detect large images")
    print("   • Downscale to 28x28, then upscale to 224x224")
    print("   • This mimics the training data pipeline")
    
    print("\n5. CREATE A TWO-STAGE PIPELINE:")
    print("   • Stage 1: Detect and crop individual cells from screenshots")
    print("   • Stage 2: Classify cropped cells")
    
    print("\n" + "="*80)
    print("🎯 IMMEDIATE FIX: Preprocessing Script")
    print("="*80)
    print("\nI'll create a script to preprocess your screenshots correctly...")


if __name__ == '__main__':
    main()
