# Test Scripts

This directory contains diagnostic and testing scripts for validating model performance and debugging issues.

## Scripts Overview

### `analyze_misclassification.py`
**Purpose**: Analyzes misclassified images to understand failure patterns

**Usage**:
```bash
python tests/analyze_misclassification.py
```

**Features**:
- Compares test images with training samples
- Analyzes color statistics and image properties
- Generates visual comparisons
- Helps identify distribution mismatches

**Output**: `test_images_analysis.png` with side-by-side comparisons

---

### `compare_images.py`
**Purpose**: Visual comparison tool for test images vs training images

**Usage**:
```bash
python tests/compare_images.py
```

**Features**:
- Side-by-side display of test vs training samples
- Helpful for understanding dataset differences
- Generates comparison visualizations

**Output**: Comparison plots showing image differences

---

### `diagnose_inference.py`
**Purpose**: Comprehensive inference diagnostics and debugging

**Usage**:
```bash
python tests/diagnose_inference.py
```

**Features**:
- Detailed inference analysis
- Preprocessing pipeline inspection
- Model output diagnostics
- Confidence score analysis

**Output**: Diagnostic reports and visualizations

---

### `proper_test.py`
**Purpose**: Validates model performance on actual microscopy images

**Usage**:
```bash
python tests/proper_test.py
```

**Features**:
- Tests model on real blood cell images from test set
- Verifies model accuracy on training-distribution images
- Confirms model is working correctly
- Generates per-class accuracy reports

**Expected Results**: 
- 95%+ accuracy on BloodMNIST test images
- High confidence scores (>90%)

**Output**: `proper_test_results.png` with predictions and confidence scores

---

### `test_preprocessing_fix.py`
**Purpose**: Tests and compares different preprocessing strategies

**Usage**:
```bash
python tests/test_preprocessing_fix.py
```

**Features**:
- Compares normal vs downscale preprocessing
- Tests on large images and screenshots
- Generates visualization comparing results
- Validates the downscale fix solution

**Key Insight**: 
Large images (>100x100) should be downscaled to 28x28 first, then upscaled to 224x224 to match training data format.

**Output**: `preprocessing_comparison.png` showing both strategies

---

## Common Testing Workflows

### Testing Model on New Images

1. **For microscopy images** (similar to training data):
   ```bash
   python tests/proper_test.py
   ```

2. **For screenshots or large images**:
   ```bash
   python tests/test_preprocessing_fix.py
   ```

### Debugging Misclassifications

1. Run full diagnostic:
   ```bash
   python tests/diagnose_inference.py
   ```

2. Analyze specific failures:
   ```bash
   python tests/analyze_misclassification.py
   ```

3. Compare with training distribution:
   ```bash
   python tests/compare_images.py
   ```

### Validating Model Performance

```bash
# Quick validation on test set
python tests/proper_test.py

# Full evaluation
python src/infer.py \
    --checkpoint models/bloodmnist_full_exp/checkpoints/best.pth \
    --image_dir data/raw/bloodmnist_full/test \
    --output_dir results/validation
```

---

## Known Issues and Solutions

### Issue: Screenshots Misclassified

**Root Cause**: Scale mismatch between training data (28x28) and screenshots (200x200+)

**Solution**: Use downscale preprocessing fix
```python
# Enable in Streamlit: Check "Enable downscale fix"
# Or use test_preprocessing_fix.py to validate
```

### Issue: Low Confidence Scores

**Possible Causes**:
1. Image not from training distribution
2. Image quality issues (blur, artifacts)
3. Incorrect preprocessing

**Debug Steps**:
```bash
python tests/diagnose_inference.py  # Check preprocessing
python tests/compare_images.py      # Compare with training data
```

---

## Adding New Tests

To add a new test script:

1. Create script in `tests/` directory
2. Follow naming convention: `test_*.py` or `*_test.py`
3. Include docstring explaining purpose
4. Add usage example in this README
5. Generate outputs to `results/` or `results/tests/`

Example template:
```python
"""
Test script for [specific functionality]

Purpose: [What this test validates]
Usage: python tests/test_new_feature.py
Output: [Description of outputs]
"""

import torch
from src.models.backbones import get_model

def test_feature():
    """Test specific feature"""
    # Test implementation
    pass

if __name__ == "__main__":
    test_feature()
    print("Test completed successfully!")
```

---

## Test Data Requirements

Most test scripts expect:
- Trained model checkpoint in `models/bloodmnist_full_exp/checkpoints/best.pth`
- Test images in `test images/` directory (for screenshot tests)
- BloodMNIST test set in `data/raw/bloodmnist_full/test/` (for validation)

Ensure these paths exist before running tests.

---

## Performance Benchmarks

Expected performance on various hardware:

| Hardware | proper_test.py | test_preprocessing_fix.py |
|----------|----------------|---------------------------|
| M1/M2 Mac | ~5 seconds | ~3 seconds |
| NVIDIA RTX 3090 | ~2 seconds | ~1 second |
| CPU (Intel i7) | ~15 seconds | ~8 seconds |

---

## Troubleshooting

### Import Errors

Make sure you're running from project root:
```bash
cd /path/to/CellMorphNet
python tests/test_script.py
```

### Missing Dependencies

Install all requirements:
```bash
pip install -r requirements.txt
```

### File Not Found Errors

Check that model checkpoint exists:
```bash
ls -lh models/bloodmnist_full_exp/checkpoints/best.pth
```

---

For more information, see the main [README.md](../README.md) and [documentation](../docs/).
