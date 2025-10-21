# Getting Started with CellMorphNet

This guide will help you get started with CellMorphNet in under 10 minutes.

## Installation

### Prerequisites
- Python 3.10+
- pip or conda

### Step 1: Clone the Repository

```bash
git clone https://github.com/parth-6-5-4/CellMorphNet.git
cd CellMorphNet
```

### 2. Download Dataset (1-2 minutes)

```bash
# Download BloodMNIST subset (296 images per split)
python scripts/download_bloodmnist.py --sample_limit 300
```

This downloads a lightweight subset perfect for quick experimentation.

### 3. Train Your First Model (30-40 minutes)

**Option A: Quick training script (recommended)**

```bash
./scripts/train_quick.sh
```

This runs a complete training pipeline with sensible defaults.

**Option B: Manual command**

```bash
python src/train.py \
    --data_dir data/raw/bloodmnist \
    --num_classes 8 \
    --backbone efficientnet_lite0 \
    --epochs 20 \
    --batch_size 8 \
    --accumulation_steps 4 \
    --output_dir models/my_first_model
```

**What's happening:**
- Training EfficientNet-Lite0 (~4M parameters)
- Using mixed precision training (AMP)
- Gradient accumulation for effective batch size of 32
- Early stopping on validation F1 score
- Saving checkpoints to `models/my_first_model/checkpoints/`

**Expected output:**
```
Epoch 1/20: Train Loss: 1.2345, Val Loss: 0.9876, Val F1: 0.8234
Epoch 2/20: Train Loss: 0.8765, Val Loss: 0.7654, Val F1: 0.9012
...
Best validation F1: 0.9543
```

### 4. Test Inference (1 minute)

**Single image prediction:**

```bash
python src/infer.py \
    --checkpoint models/my_first_model/checkpoints/best.pth \
    --image data/raw/bloodmnist/test/basophil/basophil_0000.png \
    --gradcam
```

This will:
- Load the trained model
- Classify the image
- Generate Grad-CAM visualization
- Display results

**Batch inference:**

```bash
python src/infer.py \
    --checkpoint models/my_first_model/checkpoints/best.pth \
    --image_dir data/raw/bloodmnist/test/neutrophil/ \
    --output_dir results/test_predictions \
    --gradcam
```

### 5. Launch Web Demo (30 seconds)

**Streamlit UI:**

```bash
streamlit run demos/streamlit_app.py
```

Open http://localhost:8501 in your browser and:
- Upload an image
- Click "Classify"
- View predictions and Grad-CAM

**FastAPI Server:**

```bash
python demos/fastapi_server.py
```

Visit http://localhost:8000/docs for interactive API documentation.

Test with curl:

```bash
curl -X POST "http://localhost:8000/predict" \
    -F "file=@data/raw/bloodmnist/test/basophil/basophil_0000.png"
```

### 6. Export Model (1 minute)

Export to multiple formats:

```bash
python src/export_coreml.py \
    --checkpoint models/my_first_model/checkpoints/best.pth \
    --output_dir models/exported \
    --all \
    --quantize \
    --benchmark
```

This creates:
- `cellmorphnet.pt` - TorchScript (for Python/C++)
- `cellmorphnet.onnx` - ONNX (cross-platform)
- `cellmorphnet.mlmodel` - CoreML (for iOS/macOS)

---

## Common Issues & Solutions

### Issue: "RuntimeError: MPS backend out of memory"

**Solution:** Reduce batch size and increase accumulation steps

```bash
python src/train.py \
    --batch_size 4 \
    --accumulation_steps 8 \
    # ... other args
```

### Issue: "Import cv2 could not be resolved"

**Solution:** Install OpenCV

```bash
pip install opencv-python
```

### Issue: "Checkpoint not found"

**Solution:** Make sure training completed successfully. Check:

```bash
ls -lh models/my_first_model/checkpoints/
```

You should see `best.pth` and `latest.pth`.

### Issue: "Slow training on Mac"

**Causes & solutions:**
1. **Not using MPS**: PyTorch should auto-detect MPS. Check logs for "Using device: mps"
2. **Too large batch size**: Reduce to 4-8
3. **Large images**: Use `--img_size 128` for faster training
4. **Background processes**: Close other applications

---

## Next Steps

### Train on Full Dataset

Download complete BloodMNIST:

```bash
python scripts/download_bloodmnist.py --sample_limit 100000
```

Train for better results:

```bash
python src/train.py \
    --data_dir data/raw/bloodmnist \
    --num_classes 8 \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-4 \
    --use_class_weights
```

### Try Different Backbones

```bash
# MobileNetV3-Small (fastest)
python src/train.py --backbone mobilenet_v3_small --batch_size 32

# ResNet18 (higher accuracy)
python src/train.py --backbone resnet18 --batch_size 16

# MobileNetV3-Large (balanced)
python src/train.py --backbone mobilenet_v3_large --batch_size 16
```

### Use Other Datasets

See [DATASETS.md](DATASETS.md) for:
- BCCD dataset setup
- LISC dataset setup
- Custom dataset format

### Hyperparameter Tuning

Key parameters to experiment with:
- `--lr`: Learning rate (try 1e-3, 1e-4, 1e-5)
- `--batch_size`: Batch size (4, 8, 16, 32)
- `--img_size`: Image resolution (128, 224, 384)
- `--freeze_backbone`: Transfer learning strategy
- `--weight_decay`: Regularization (1e-3, 1e-4, 1e-5)

### Advanced Features

**Albumentations augmentation:**

```bash
python src/train.py --use_albumentations --epochs 30
```

**Class weighting for imbalanced data:**

```bash
python src/train.py --use_class_weights
```

**Progressive unfreezing:**

```bash
python src/train.py --freeze_backbone --unfreeze_after 5 --epochs 30
```

---

## Performance Benchmarks

Expected results on Mac M2 8GB:

| Model | Training Time | Val Accuracy | Inference Time | Model Size |
|-------|--------------|--------------|----------------|------------|
| EfficientNet-Lite0 | 40 min | 95%+ | 30ms | 16MB |
| MobileNetV3-Small | 30 min | 93%+ | 20ms | 10MB |
| ResNet18 | 60 min | 96%+ | 40ms | 45MB |

*For 20 epochs on BloodMNIST subset (888 images)*

---

## Learning Resources

- **Understanding the code**: Start with `src/data.py` → `src/models/backbones.py` → `src/train.py`
- **Modifying models**: See `src/models/morph_attention.py` for attention mechanisms
- **Custom augmentations**: Check `src/augment.py` for Albumentations examples
- **Deployment**: Review `src/export_coreml.py` for model conversion

---

## Get Help

-  **Documentation**: See README.md and DATASETS.md
-  **Issues**: Open a GitHub issue
-  **Discussions**: Use GitHub Discussions for questions

---

## Congratulations! 🎉

You've successfully:
-  Set up CellMorphNet
-  Downloaded a dataset
-  Trained a model
-  Run inference with Grad-CAM
-  Launched a web demo
-  Exported models for deployment

You're now ready to:
- Experiment with different architectures
- Try other datasets
- Deploy to production
- Customize for your use case

Happy coding! 
