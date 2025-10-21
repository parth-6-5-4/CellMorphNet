"""
Test the model with actual blood cell images from the test set.
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import random

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
    """Preprocess image."""
    img = Image.open(image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform(img).unsqueeze(0), img


def predict(model, image_tensor, class_names, device):
    """Make prediction."""
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        
        predicted_idx = probabilities.argmax().item()
        predicted_class = class_names[predicted_idx]
        confidence = probabilities[predicted_idx].item()
        
        return predicted_class, confidence, probabilities.cpu().numpy()


def main():
    print("="*70)
    print("TESTING WITH ACTUAL BLOOD CELL IMAGES FROM TEST SET")
    print("="*70)
    
    # Load model
    checkpoint_path = Path('models/bloodmnist_full_exp/checkpoints/best.pth')
    model, config, class_names, device = load_model(checkpoint_path)
    
    print(f"\nModel: {config['backbone']}")
    print(f"Classes: {class_names}")
    
    # Get test images (2 random samples per class)
    test_dir = Path('data/raw/bloodmnist_full/test')
    
    if not test_dir.exists():
        print(f"\nTest directory not found: {test_dir}")
        print("Please run: python scripts/download_bloodmnist.py")
        return
    
    # Collect test samples
    test_samples = []
    for class_name in class_names:
        class_dir = test_dir / class_name
        if class_dir.exists():
            images = list(class_dir.glob('*.png'))
            if images:
                # Take 2 random samples
                samples = random.sample(images, min(2, len(images)))
                test_samples.extend([(img, class_name) for img in samples])
    
    print(f"\nTesting on {len(test_samples)} images...")
    
    # Test each image
    results = []
    correct = 0
    total = 0
    
    for img_path, true_class in test_samples:
        tensor, original_img = preprocess_image(img_path, config['img_size'])
        predicted, confidence, probs = predict(model, tensor, class_names, device)
        
        is_correct = predicted == true_class
        if is_correct:
            correct += 1
        total += 1
        
        results.append({
            'image': img_path,
            'true_class': true_class,
            'predicted': predicted,
            'confidence': confidence,
            'correct': is_correct,
            'original_img': original_img
        })
        
        status = "✓ CORRECT" if is_correct else "✗ WRONG"
        color = '\033[92m' if is_correct else '\033[91m'
        reset = '\033[0m'
        
        print(f"{color}{status}{reset} | True: {true_class:12s} | Predicted: {predicted:12s} | Confidence: {confidence:.2%}")
    
    # Summary
    accuracy = correct / total if total > 0 else 0
    print("\n" + "="*70)
    print(f"ACCURACY: {correct}/{total} = {accuracy:.2%}")
    print("="*70)
    
    # Visualize results (4x4 grid)
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.flatten()
    
    for idx, result in enumerate(results[:16]):
        ax = axes[idx]
        ax.imshow(result['original_img'])
        ax.axis('off')
        
        true_class = result['true_class']
        predicted = result['predicted']
        confidence = result['confidence']
        
        if result['correct']:
            title = f"✓ {true_class.upper()}\n{confidence:.1%}"
            color = 'green'
        else:
            title = f"✗ True: {true_class}\nPred: {predicted}\n{confidence:.1%}"
            color = 'red'
        
        ax.set_title(title, fontsize=10, fontweight='bold', color=color)
    
    # Hide empty subplots
    for idx in range(len(results), 16):
        axes[idx].axis('off')
    
    plt.suptitle(f'Model Testing on Actual Blood Cells\nAccuracy: {accuracy:.2%}', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('proper_test_results.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Results saved to: proper_test_results.png")
    
    # Now test with the user's screenshots
    print("\n" + "="*70)
    print("TESTING YOUR SCREENSHOTS")
    print("="*70)
    
    test_screenshots = Path('test images')
    screenshots = list(test_screenshots.glob('*.png'))
    
    if screenshots:
        print(f"\nFound {len(screenshots)} screenshots")
        print("\nWARNING: These are screenshots, not microscopy images!")
        print("The model was trained on 28x28 pixel microscopy images of single cells.")
        print("Screenshots may contain multiple cells, UI elements, or different image properties.\n")
        
        for img_path in screenshots:
            # Convert RGBA to RGB if needed
            img = Image.open(img_path)
            if img.mode == 'RGBA':
                # Create white background
                rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                rgb_img.paste(img, mask=img.split()[3])  # Use alpha channel as mask
                img = rgb_img
            
            tensor = transforms.Compose([
                transforms.Resize((config['img_size'], config['img_size'])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])(img).unsqueeze(0)
            
            predicted, confidence, probs = predict(model, tensor, class_names, device)
            
            print(f"\n{img_path.name}:")
            print(f"  Original size: {Image.open(img_path).size}")
            print(f"  Predicted: {predicted}")
            print(f"  Confidence: {confidence:.2%}")
            print(f"  ⚠️  Result may be unreliable due to image format mismatch")


if __name__ == '__main__':
    random.seed(42)
    main()
