"""
Visual comparison between test images and training samples.
"""

import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import numpy as np

# Load test images
test_dir = Path('test images')
test_images = sorted(test_dir.glob('*.png'))

# Training data directory
train_dir = Path('data/raw/bloodmnist_full/train')

# Get samples from each class
classes = ['basophil', 'eosinophil', 'erythroblast', 'ig', 'lymphocyte', 'monocyte', 'neutrophil', 'platelet']

# Create figure
fig = plt.figure(figsize=(20, 12))

# Plot test images (larger)
for idx, test_img_path in enumerate(test_images[:2]):
    ax = plt.subplot(4, 5, idx*5 + 1)
    img = Image.open(test_img_path)
    ax.imshow(img)
    ax.set_title(f'TEST IMAGE {idx+1}\n{test_img_path.name}\nSize: {img.size}', 
                 fontsize=10, fontweight='bold', color='red')
    ax.axis('off')

# Plot training samples
row = 0
for class_idx, class_name in enumerate(classes):
    col = (class_idx % 4) + 1
    if class_idx == 4:
        row = 1
    
    ax = plt.subplot(4, 5, row*10 + col + 1)
    
    class_dir = train_dir / class_name
    if class_dir.exists():
        samples = list(class_dir.glob('*.png'))
        if samples:
            # Load first sample
            sample_img = Image.open(samples[0])
            ax.imshow(sample_img)
            ax.set_title(f'{class_name.upper()}\n(Training Sample)\nSize: {sample_img.size}', 
                        fontsize=8)
    else:
        ax.text(0.5, 0.5, 'No samples', ha='center', va='center')
        ax.set_title(class_name.upper(), fontsize=8)
    
    ax.axis('off')

plt.suptitle('TEST IMAGES vs TRAINING SAMPLES', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('test_vs_training_comparison.png', dpi=200, bbox_inches='tight')
print("✓ Comparison saved to: test_vs_training_comparison.png")
plt.close()

# Now create a detailed view
fig, axes = plt.subplots(2, 5, figsize=(20, 8))

# Show test images in first row
for idx, test_img_path in enumerate(test_images[:2]):
    ax = axes[0, idx*2]
    img = Image.open(test_img_path)
    axes[0, idx*2].imshow(img)
    axes[0, idx*2].set_title(f'TEST {idx+1}: {test_img_path.name}', fontsize=10, fontweight='bold')
    axes[0, idx*2].axis('off')
    
    # Show predicted class sample next to it
    predicted_class = 'erythroblast'  # From diagnostic
    ax = axes[0, idx*2 + 1]
    class_dir = train_dir / predicted_class
    if class_dir.exists():
        samples = list(class_dir.glob('*.png'))
        if samples:
            sample_img = Image.open(samples[0])
            ax.imshow(sample_img)
            ax.set_title(f'Predicted: {predicted_class.upper()}\n(Training Sample)', fontsize=10, color='blue')
    ax.axis('off')

# Show some other class samples in second row for comparison
other_classes = ['eosinophil', 'lymphocyte', 'neutrophil', 'monocyte', 'platelet']
for idx, class_name in enumerate(other_classes):
    ax = axes[1, idx]
    class_dir = train_dir / class_name
    if class_dir.exists():
        samples = list(class_dir.glob('*.png'))
        if samples:
            sample_img = Image.open(samples[0])
            ax.imshow(sample_img)
            ax.set_title(f'{class_name.upper()}', fontsize=10)
    ax.axis('off')

plt.suptitle('DETAILED COMPARISON: Test Images vs Training Data', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('detailed_comparison.png', dpi=200, bbox_inches='tight')
print("✓ Detailed comparison saved to: detailed_comparison.png")
plt.close()

print("\n" + "="*70)
print("KEY OBSERVATIONS TO CHECK:")
print("="*70)
print("\n1. Are your test images SCREENSHOTS of blood cells?")
print("   - Training data: Pure microscopy images of single cells (28x28 pixels)")
print("   - Your images: Appear to be larger screenshots (290x308 and 306x274 pixels)")
print("\n2. Do your test images contain:")
print("   - Multiple cells?")
print("   - UI elements, borders, or text?")
print("   - Different background colors?")
print("\n3. BloodMNIST training images are:")
print("   - Very small (28x28 pixels)")
print("   - Tightly cropped around single cells")
print("   - Standardized color/contrast")
print("\n4. For accurate predictions, test images should:")
print("   - Be actual microscopy images")
print("   - Show a single cell")
print("   - Have similar staining/coloring to training data")
print("   - Be cropped tightly around the cell")
print("\n" + "="*70)
