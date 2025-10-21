#!/usr/bin/env python3
"""
Test script to verify CellMorphNet installation and basic functionality.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

def test_imports():
    """Test that all required packages are installed."""
    print("Testing imports...")
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('PIL', 'Pillow'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('sklearn', 'scikit-learn'),
        ('matplotlib', 'Matplotlib'),
        ('tqdm', 'tqdm'),
    ]
    
    optional_packages = [
        ('albumentations', 'Albumentations'),
        ('cv2', 'OpenCV'),
        ('coremltools', 'CoreML Tools'),
        ('streamlit', 'Streamlit'),
        ('fastapi', 'FastAPI'),
    ]
    
    failed = []
    
    print("\n✓ Required packages:")
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - MISSING")
            failed.append(name)
    
    print("\n✓ Optional packages:")
    for package, name in optional_packages:
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ⚠ {name} - Not installed (optional)")
    
    if failed:
        print(f"\n✗ Missing required packages: {', '.join(failed)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("\n✓ All required packages installed!")
    return True


def test_torch_device():
    """Test PyTorch device availability."""
    import torch
    
    print("\nTesting PyTorch device...")
    
    print(f"  PyTorch version: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print(f"  ✓ MPS (Apple Silicon) available")
        device = "mps"
    else:
        print(f"  ✓ CPU only")
        device = "cpu"
    
    # Test tensor creation
    try:
        x = torch.randn(2, 3, 224, 224).to(device)
        print(f"  ✓ Successfully created tensor on {device}")
        return True
    except Exception as e:
        print(f"  ✗ Failed to create tensor: {e}")
        return False


def test_model_creation():
    """Test model creation."""
    import torch
    from src.models.backbones import get_model, count_parameters
    
    print("\nTesting model creation...")
    
    try:
        model = get_model('efficientnet_lite0', num_classes=8, pretrained=False)
        trainable, total = count_parameters(model)
        print(f"  ✓ EfficientNet-Lite0 created")
        print(f"    Parameters: {total:,} ({trainable:,} trainable)")
        
        # Test forward pass
        x = torch.randn(1, 3, 224, 224)
        y = model(x)
        assert y.shape == (1, 8), f"Expected shape (1, 8), got {y.shape}"
        print(f"  ✓ Forward pass successful: {y.shape}")
        
        return True
    except Exception as e:
        print(f"  ✗ Model creation failed: {e}")
        return False


def test_data_loading():
    """Test data loading."""
    from pathlib import Path
    
    print("\nTesting data availability...")
    
    data_dir = Path("data/raw/bloodmnist")
    
    if not data_dir.exists():
        print(f"  ⚠ Dataset not found at {data_dir}")
        print(f"    Run: python scripts/download_bloodmnist.py")
        return False
    
    # Check structure
    splits = ['train', 'val', 'test']
    for split in splits:
        split_dir = data_dir / split
        if split_dir.exists():
            num_classes = len([d for d in split_dir.iterdir() if d.is_dir()])
            num_images = sum(1 for d in split_dir.rglob('*.png'))
            print(f"  ✓ {split}: {num_classes} classes, {num_images} images")
        else:
            print(f"  ✗ {split} directory not found")
            return False
    
    return True


def test_training_setup():
    """Test training setup without actual training."""
    import torch
    from src.models.backbones import get_model
    
    print("\nTesting training setup...")
    
    try:
        # Create model, optimizer, loss
        model = get_model('mobilenet_v3_small', num_classes=8, pretrained=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        criterion = torch.nn.CrossEntropyLoss()
        
        print(f"  ✓ Model, optimizer, and loss function created")
        
        # Simulate training step
        model.train()
        x = torch.randn(2, 3, 224, 224)
        y = torch.randint(0, 8, (2,))
        
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        print(f"  ✓ Training step successful (loss: {loss.item():.4f})")
        
        return True
    except Exception as e:
        print(f"  ✗ Training setup failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("CellMorphNet Installation Test")
    print("=" * 60)
    
    tests = [
        ("Package imports", test_imports),
        ("PyTorch device", test_torch_device),
        ("Model creation", test_model_creation),
        ("Data loading", test_data_loading),
        ("Training setup", test_training_setup),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} failed with exception: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests passed! You're ready to use CellMorphNet.")
        print("\nNext steps:")
        print("1. Download dataset: python scripts/download_bloodmnist.py")
        print("2. Train model: python src/train.py --data_dir data/raw/bloodmnist --num_classes 8 --epochs 20")
        print("3. Run demo: streamlit run demos/streamlit_app.py")
    else:
        print("\n⚠ Some tests failed. Please fix the issues above.")
        sys.exit(1)


if __name__ == '__main__':
    main()
