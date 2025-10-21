"""
Fix preprocessing to handle screenshot scale mismatch.
Converts large screenshots to match training data format.
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import glob

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


def preprocess_normal(img, img_size=224):
    """Normal preprocessing (what we're currently using)."""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)


def preprocess_with_downscale(img, img_size=224):
    """Preprocess by first downscaling to 28x28 (like training data), then upscaling."""
    # Step 1: Downscale to training data size
    img_small = img.resize((28, 28), Image.LANCZOS)
    
    # Step 2: Apply same transform as training
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img_small).unsqueeze(0)


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
    print("="*80)
    print("TESTING PREPROCESSING STRATEGIES FOR SCREENSHOTS")
    print("="*80)
    
    # Load model
    checkpoint_path = Path('models/bloodmnist_full_exp/checkpoints/best.pth')
    model, config, class_names, device = load_model(checkpoint_path)
    
    print(f"\nModel: {config['backbone']}")
    print(f"Classes: {class_names}")
    
    # Find screenshots using glob
    test_images = sorted(glob.glob('test images/*.png'))
    
    if not test_images:
        print("\nNo test images found!")
        return
    
    print(f"\nFound {len(test_images)} test images")
    
    # Ground truth labels (provide actual labels from user)
    ground_truth = {
        'Screenshot 2025-10-22 at 3.38.38\u202fAM.png': 'neutrophil',
        'Screenshot 2025-10-22 at 3.40.15\u202fAM.png': 'basophil'
    }
    
    print("\nGround truth labels:")
    for name, label in ground_truth.items():
        print(f"  - {name}: {label}")
    
    results = []
    
    for img_path in test_images:
        img_name = Path(img_path).name
        true_class = ground_truth.get(img_name, 'unknown')
        
        print(f"\n" + "="*80)
        print(f"IMAGE: {img_name}")
        print(f"TRUE CLASS: {true_class}")
        print("="*80)
        
        # Load image
        img = Image.open(img_path)
        print(f"\nOriginal: {img.size}, {img.mode}")
        
        # Convert RGBA to RGB if needed
        if img.mode == 'RGBA':
            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
            rgb_img.paste(img, mask=img.split()[3])
            img = rgb_img
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Strategy 1: Normal preprocessing
        print("\n📊 STRATEGY 1: Normal preprocessing (current)")
        print("   " + "-"*70)
        tensor1 = preprocess_normal(img, config['img_size'])
        pred1, conf1, probs1 = predict(model, tensor1, class_names, device)
        
        print(f"   Predicted: {pred1:15s} Confidence: {conf1:.2%}")
        if true_class != 'unknown':
            print(f"   True class ({true_class}) probability: {probs1[class_names.index(true_class)]:.2%}")
        
        correct1 = pred1 == true_class if true_class != 'unknown' else False
        if correct1:
            print("   ✓ CORRECT!")
        else:
            print(f"   ✗ WRONG (predicted {pred1}, should be {true_class})")
        
        # Strategy 2: Downscale first
        print("\n📊 STRATEGY 2: Downscale to 28x28 first (match training data)")
        print("   " + "-"*70)
        tensor2 = preprocess_with_downscale(img, config['img_size'])
        pred2, conf2, probs2 = predict(model, tensor2, class_names, device)
        
        print(f"   Predicted: {pred2:15s} Confidence: {conf2:.2%}")
        if true_class != 'unknown':
            print(f"   True class ({true_class}) probability: {probs2[class_names.index(true_class)]:.2%}")
        
        correct2 = pred2 == true_class if true_class != 'unknown' else False
        if correct2:
            print("   ✓ CORRECT!")
        else:
            print(f"   ✗ WRONG (predicted {pred2}, should be {true_class})")
        
        results.append({
            'image': img_name,
            'true': true_class,
            'normal_pred': pred1,
            'normal_conf': conf1,
            'normal_correct': correct1,
            'downscale_pred': pred2,
            'downscale_conf': conf2,
            'downscale_correct': correct2
        })
        
        # Show top-3 predictions for both strategies
        print("\n   Top-3 predictions comparison:")
        print("   " + "-"*70)
        print(f"   {'Class':<15s} {'Normal':<12s} {'Downscale':<12s}")
        print("   " + "-"*70)
        
        top3_1 = np.argsort(probs1)[-3:][::-1]
        top3_2 = np.argsort(probs2)[-3:][::-1]
        
        shown_classes = set(top3_1) | set(top3_2)
        for cls_idx in sorted(shown_classes, key=lambda x: max(probs1[x], probs2[x]), reverse=True)[:5]:
            cls_name = class_names[cls_idx]
            marker = "→" if cls_name == true_class else " "
            print(f"   {marker} {cls_name:<15s} {probs1[cls_idx]:>6.2%}       {probs2[cls_idx]:>6.2%}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    normal_correct = sum(1 for r in results if r['normal_correct'])
    downscale_correct = sum(1 for r in results if r['downscale_correct'])
    
    print(f"\nNormal preprocessing:      {normal_correct}/{len(results)} correct")
    print(f"Downscale preprocessing:   {downscale_correct}/{len(results)} correct")
    
    if downscale_correct > normal_correct:
        print("\n✓ DOWNSCALING IMPROVES ACCURACY!")
        print("  Recommendation: Update preprocessing to downscale screenshots first")
    elif downscale_correct == normal_correct:
        print("\n⚠️  Both strategies perform equally")
    else:
        print("\n⚠️  Normal preprocessing is better (but still may not be perfect)")
    
    # Detailed results
    print("\nDetailed results:")
    print("-"*80)
    for r in results:
        print(f"\n{r['image']}:")
        print(f"  True class: {r['true']}")
        print(f"  Normal:     {r['normal_pred']:15s} ({r['normal_conf']:.1%}) {'✓' if r['normal_correct'] else '✗'}")
        print(f"  Downscale:  {r['downscale_pred']:15s} ({r['downscale_conf']:.1%}) {'✓' if r['downscale_correct'] else '✗'}")
    
    # Visualization
    fig, axes = plt.subplots(len(test_images), 4, figsize=(16, 5*len(test_images)))
    if len(test_images) == 1:
        axes = [axes]
    
    for idx, (img_path, result) in enumerate(zip(test_images, results)):
        img = Image.open(img_path)
        if img.mode == 'RGBA':
            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
            rgb_img.paste(img, mask=img.split()[3])
            img = rgb_img
        
        # Original
        axes[idx][0].imshow(img)
        axes[idx][0].set_title(f'Original Screenshot\nTrue: {result["true"].upper()}', 
                               fontsize=10, fontweight='bold')
        axes[idx][0].axis('off')
        
        # Downscaled to 28x28
        img_28 = img.resize((28, 28), Image.LANCZOS)
        axes[idx][1].imshow(img_28)
        axes[idx][1].set_title('Downscaled to 28x28\n(Training data size)', fontsize=10)
        axes[idx][1].axis('off')
        
        # Normal preprocessing result
        color1 = 'green' if result['normal_correct'] else 'red'
        axes[idx][2].imshow(img.resize((224, 224)))
        axes[idx][2].set_title(f'Normal → {result["normal_pred"]}\n{result["normal_conf"]:.1%}', 
                               fontsize=10, color=color1, fontweight='bold')
        axes[idx][2].axis('off')
        
        # Downscale preprocessing result
        color2 = 'green' if result['downscale_correct'] else 'red'
        axes[idx][3].imshow(img_28.resize((224, 224)))
        axes[idx][3].set_title(f'Downscale → {result["downscale_pred"]}\n{result["downscale_conf"]:.1%}', 
                               fontsize=10, color=color2, fontweight='bold')
        axes[idx][3].axis('off')
    
    plt.suptitle('Preprocessing Strategy Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('preprocessing_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: preprocessing_comparison.png")


if __name__ == '__main__':
    main()
