# CellMorphNet

**Lightweight Deep Learning Framework for Blood Cell Classification**

A production-ready deep learning system for automated classification of blood cell types from microscopic images. Built with lightweight convolutional neural networks and attention mechanisms, optimized for real-time inference on resource-constrained devices including Apple M-series processors.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Datasets](#datasets)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Inference](#inference)
- [Deployment](#deployment)
- [Performance Benchmarks](#performance-benchmarks)
- [Web Interface](#web-interface)
- [API Documentation](#api-documentation)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

---

## Overview

CellMorphNet is a comprehensive machine learning framework designed for blood cell morphology analysis. The system demonstrates best practices in medical image classification, from data preprocessing through model deployment.

### Project Goals

- Provide accurate, real-time blood cell classification for 8 cell types
- Demonstrate efficient training on consumer hardware (8GB RAM)
- Enable model interpretability through attention visualizations
- Deliver production-ready deployment options (REST API, web interface, mobile export)
- Maintain high performance with lightweight architectures (under 5M parameters)

### Medical Relevance

Blood cell classification is fundamental to hematology diagnostics. Automated systems can:
- Accelerate differential blood counts
- Reduce observer variability
- Support clinical decision-making
- Enable point-of-care diagnostics in resource-limited settings

**Disclaimer**: This is a research and educational project. It is not approved for clinical diagnostic use and should not replace professional medical judgment.

---

## Key Features

### Performance
- **High Accuracy**: Achieves 98.17% F1 score on validation data
- **Fast Inference**: Under 50ms per image on Apple M2 processors
- **Efficient Training**: Runs on 8GB RAM with mixed precision training
- **Compact Models**: EfficientNet-Lite0 with only 4.06M parameters

### Interpretability
- **Grad-CAM Visualizations**: Shows which image regions influence predictions
- **Attention Mechanisms**: 16 Squeeze-and-Excitation blocks for feature refinement
- **Confidence Scores**: Provides probability distributions across all classes

### Deployment Options
- **Streamlit Web Interface**: Interactive demo for testing and visualization
- **FastAPI REST Server**: Production-ready API with automatic documentation
- **CoreML Export**: Optimized models for iOS and macOS applications
- **ONNX Export**: Cross-platform model format

### Data Processing
- **Multi-Dataset Support**: BloodMNIST, BCCD, LISC datasets
- **Advanced Augmentation**: Albumentations pipeline with medical-specific transforms
- **Automatic Preprocessing**: Handles various image formats and sizes
- **Class Balancing**: Weighted loss functions for imbalanced datasets

---

## Project Structure

```
CellMorphNet/
├── src/                          # Core source code
│   ├── data.py                   # Dataset loaders and preprocessing
│   ├── augment.py                # Data augmentation pipelines
│   ├── train.py                  # Training loop with AMP support
│   ├── infer.py                  # Inference and Grad-CAM generation
│   ├── export_coreml.py          # Model export utilities
│   └── models/
│       ├── backbones.py          # CNN architectures
│       └── morph_attention.py    # Custom attention modules
│
├── demos/                        # Deployment applications
│   ├── streamlit_app.py          # Interactive web interface
│   └── fastapi_server.py         # REST API server
│
├── scripts/                      # Utility scripts
│   ├── download_bloodmnist.py    # Dataset downloader
│   ├── prepare_datasets.py       # Data preparation
│   ├── test_installation.py      # Installation verification
│   └── train_combined.py         # Multi-dataset training
│
├── tests/                        # Test and diagnostic scripts
│   ├── README.md                 # Test documentation
│   ├── proper_test.py            # Model validation tests
│   ├── test_preprocessing_fix.py # Preprocessing diagnostics
│   ├── analyze_misclassification.py  # Error analysis
│   ├── compare_images.py         # Image comparison tool
│   └── diagnose_inference.py     # Inference debugging
│
├── docs/                         # Documentation
│   ├── README.md                 # Documentation guide
│   ├── GETTING_STARTED.md        # Quick start guide
│   ├── DATASETS.md               # Dataset documentation
│   ├── TRAINING_ANALYSIS.md      # Performance analysis
│   ├── TRAINING_SUMMARY.md       # Training quick reference
│   ├── DEPLOYMENT_GUIDE.md       # Deployment instructions
│   ├── HOW_TO_TEST.md            # Testing guide
│   ├── SCREENSHOT_FIX_GUIDE.md   # Troubleshooting guide
│   ├── PROJECT_SUMMARY.md        # Project overview
│   └── overview.md               # Technical architecture
│
├── data/                         # Dataset storage
│   ├── raw/                      # Original datasets
│   └── processed/                # Preprocessed data
│
├── models/                       # Trained models
│   └── bloodmnist_full_exp/      # Best model checkpoint
│       ├── checkpoints/
│       │   └── best.pth          # 98.17% F1 model
│       └── history.json          # Training history
│
├── results/                      # Outputs and visualizations
│   ├── plots/                    # Training curves
│   └── predictions/              # Inference results
│
├── archive (1)/                  # Historical datasets and scripts
├── LISC Database/                # LISC dataset
├── requirements.txt              # Python dependencies
├── launch_demos.sh               # Demo launcher script
└── README.md                     # This file
```

---

## Installation

### System Requirements

- **Operating System**: macOS, Linux, or Windows
- **Python**: 3.10 or higher
- **RAM**: Minimum 8GB (16GB recommended for training)
- **Storage**: 5GB for datasets and models
- **GPU**: Optional (CUDA-compatible GPU, Apple M-series, or CPU)

### Step 1: Clone Repository

```bash
git clone https://github.com/parth-6-5-4/CellMorphNet.git
cd CellMorphNet
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n cellmorphnet python=3.10
conda activate cellmorphnet
```

### Step 3: Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# For Apple Silicon Macs, ensure MPS support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# For CUDA support (NVIDIA GPUs)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Step 4: Verify Installation

```bash
python scripts/test_installation.py
```

Expected output:
```
Python version: 3.10.x
PyTorch version: 2.x.x
CUDA available: False
MPS available: True (on Apple Silicon)
Installation successful!
```

---

## Quick Start

### 1. Download Sample Dataset

For initial testing, download a small subset of BloodMNIST:

```bash
python scripts/download_bloodmnist.py --sample_limit 300
```

This downloads approximately 300 images per split (train/val/test) across 8 blood cell classes.

### 2. Train a Model

Train a lightweight model on the sample dataset:

```bash
python src/train.py \
    --data_dir data/raw/bloodmnist \
    --num_classes 8 \
    --backbone efficientnet_lite0 \
    --epochs 20 \
    --batch_size 8 \
    --accumulation_steps 4 \
    --lr 0.0001 \
    --output_dir models/quickstart_exp
```

Training time: Approximately 40 minutes on Apple M2 (8GB RAM)

### 3. Run Inference

Test the trained model on a sample image:

```bash
python src/infer.py \
    --checkpoint models/quickstart_exp/checkpoints/best.pth \
    --image data/raw/bloodmnist/test/neutrophil/neutrophil_0000.png \
    --gradcam
```

### 4. Launch Web Interface

Start the interactive demo:

```bash
streamlit run demos/streamlit_app.py
```

Open your browser and navigate to http://localhost:8501

---

## Datasets

CellMorphNet supports three public blood cell datasets:

### BloodMNIST

- **Source**: MedMNIST v2 collection
- **Size**: 17,092 images (28x28 pixels)
- **Classes**: 8 blood cell types
- **Split**: Train (11,959) / Val (1,712) / Test (3,421)
- **Use Case**: Quick experimentation and validation

**Classes**:
1. Basophil
2. Eosinophil
3. Erythroblast
4. Immature Granulocytes (IG)
5. Lymphocyte
6. Monocyte
7. Neutrophil
8. Platelet

### BCCD (Blood Cell Count Dataset)

- **Source**: GitHub open-source dataset
- **Size**: 364 annotated images with bounding boxes
- **Classes**: RBC, WBC, Platelets
- **Use Case**: Object detection and localization tasks

### LISC Database

- **Source**: Monash University
- **Size**: 400+ high-quality images
- **Classes**: 6 white blood cell types with ground truth segmentations
- **Use Case**: Classification and segmentation benchmarking

### Dataset Preparation

Detailed instructions for each dataset are in [DATASETS.md](docs/DATASETS.md).

Quick preparation:

```bash
# Check dataset availability
python scripts/prepare_datasets.py --check

# Download and prepare all datasets
python scripts/prepare_datasets.py --download --prepare
```

---

## Model Architecture

### Backbone Networks

CellMorphNet supports multiple lightweight CNN architectures:

#### EfficientNet-Lite0 (Recommended)
- **Parameters**: 4.06M
- **Input Size**: 224x224
- **Architecture**: MBConv blocks with compound scaling
- **Advantages**: Best accuracy-efficiency tradeoff
- **Training Time**: ~2 min/epoch on M2

#### MobileNetV3-Small
- **Parameters**: 2.54M
- **Input Size**: 224x224
- **Architecture**: Inverted residuals with SE attention
- **Advantages**: Fastest inference
- **Training Time**: ~1.5 min/epoch on M2

#### ResNet18
- **Parameters**: 11.2M
- **Input Size**: 224x224
- **Architecture**: Residual blocks with bottleneck design
- **Advantages**: Strong baseline, robust training
- **Training Time**: ~3 min/epoch on M2

### Attention Mechanisms

All models incorporate 16 Squeeze-and-Excitation (SE) attention blocks:

1. **Channel Attention**: Learns importance of each feature channel
2. **Global Context**: Uses global average pooling for feature aggregation
3. **Recalibration**: Applies learned weights to enhance informative features
4. **Residual Connection**: Maintains gradient flow during training

### Model Selection Guide

| Model | Best For | Memory | Speed | Accuracy |
|-------|----------|--------|-------|----------|
| EfficientNet-Lite0 | General use | Medium | Fast | Highest |
| MobileNetV3-Small | Mobile deployment | Low | Fastest | High |
| MobileNetV3-Large | Better accuracy | Medium | Fast | Higher |
| ResNet18 | Baseline comparison | High | Medium | High |
| ResNet34 | Large datasets | Very High | Slow | Highest |

---

## Training

### Basic Training

Train a model with default hyperparameters:

```bash
python src/train.py \
    --data_dir data/raw/bloodmnist_full \
    --num_classes 8 \
    --backbone efficientnet_lite0 \
    --epochs 50 \
    --batch_size 32 \
    --output_dir models/experiment_name
```

### Advanced Training Options

#### Transfer Learning with Progressive Unfreezing

```bash
python src/train.py \
    --data_dir data/raw/bloodmnist_full \
    --num_classes 8 \
    --backbone efficientnet_lite0 \
    --pretrained \
    --freeze_backbone \
    --unfreeze_after 5 \
    --epochs 50 \
    --batch_size 32 \
    --accumulation_steps 2 \
    --lr 0.0001 \
    --weight_decay 0.0001 \
    --optimizer adamw \
    --scheduler cosine \
    --output_dir models/transfer_learning_exp
```

#### Training with Class Weights

For imbalanced datasets:

```bash
python src/train.py \
    --data_dir data/raw/custom_dataset \
    --num_classes 6 \
    --use_class_weights \
    --epochs 30
```

#### Mixed Precision Training

Automatically enabled for CUDA and MPS devices:

```bash
python src/train.py \
    --data_dir data/raw/bloodmnist_full \
    --num_classes 8 \
    --use_amp \
    --batch_size 64 \
    --accumulation_steps 1
```

### Training Configuration

All hyperparameters can be specified via command line:

| Parameter | Description | Default | Recommended Range |
|-----------|-------------|---------|-------------------|
| `--epochs` | Number of training epochs | 30 | 20-100 |
| `--batch_size` | Batch size per GPU | 16 | 8-64 |
| `--accumulation_steps` | Gradient accumulation | 1 | 1-8 |
| `--lr` | Learning rate | 1e-4 | 1e-5 to 1e-3 |
| `--weight_decay` | L2 regularization | 1e-4 | 1e-5 to 1e-3 |
| `--img_size` | Input image size | 224 | 224 |
| `--num_workers` | Data loading workers | 2 | 2-8 |

### Training Outputs

After training, the following files are created:

```
models/experiment_name/
├── checkpoints/
│   ├── best.pth              # Best model (highest validation F1)
│   ├── last.pth              # Final model
│   └── epoch_XX.pth          # Intermediate checkpoints (optional)
├── history.json              # Training metrics history
└── training_curves.png       # Loss and accuracy plots
```

### Monitoring Training

Training progress includes:

- **Epoch-level metrics**: Loss, accuracy, F1 score
- **Per-class metrics**: Precision, recall, F1 for each class
- **Learning rate schedule**: Current LR value
- **Time estimates**: Time per epoch and ETA
- **Best model tracking**: Automatic saving of best checkpoint

Example output:

```
Epoch 10/50
Train Loss: 0.1234, Train Acc: 95.67%
Val Loss: 0.2345, Val Acc: 93.45%, Val F1: 0.9312
Best Val F1: 0.9312 (saved to checkpoints/best.pth)
Time: 2m 15s, ETA: 1h 30m
```

---

## Inference

### Single Image Prediction

Classify a single image:

```bash
python src/infer.py \
    --checkpoint models/bloodmnist_full_exp/checkpoints/best.pth \
    --image path/to/image.png
```

### Batch Inference

Process multiple images:

```bash
python src/infer.py \
    --checkpoint models/bloodmnist_full_exp/checkpoints/best.pth \
    --image_dir path/to/images/ \
    --output_dir results/predictions
```

### Inference with Grad-CAM

Generate attention visualizations:

```bash
python src/infer.py \
    --checkpoint models/bloodmnist_full_exp/checkpoints/best.pth \
    --image path/to/image.png \
    --gradcam \
    --output_dir results/gradcam
```

### Python API

Use CellMorphNet in your Python code:

```python
import torch
from PIL import Image
from torchvision import transforms
from src.models.backbones import get_model

# Load model
checkpoint = torch.load('models/bloodmnist_full_exp/checkpoints/best.pth',
                       map_location='cpu', weights_only=False)
model = get_model(
    backbone=checkpoint['config']['backbone'],
    num_classes=checkpoint['config']['num_classes'],
    pretrained=False
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Prepare image
image = Image.open('path/to/image.png').convert('RGB')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])
tensor = transform(image).unsqueeze(0)

# Predict
with torch.no_grad():
    output = model(tensor)
    probabilities = torch.softmax(output, dim=1)
    predicted_class = probabilities.argmax(dim=1).item()
    confidence = probabilities[0, predicted_class].item()

class_names = checkpoint['class_names']
print(f"Predicted: {class_names[predicted_class]} ({confidence:.2%})")
```

---

## Deployment

### Web Application (Streamlit)

Launch the interactive web interface:

```bash
streamlit run demos/streamlit_app.py --server.port 8501
```

Features:
- Image upload and preview
- Real-time classification
- Grad-CAM visualization
- Probability distribution charts
- Confidence threshold controls
- Downscale fix for large images

Access at: http://localhost:8501

### REST API (FastAPI)

Start the API server:

```bash
python demos/fastapi_server.py
```

The API will be available at http://localhost:8000 with automatic interactive documentation at http://localhost:8000/docs

#### API Endpoints

**POST /predict** - Single image classification

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.png"
```

Response:
```json
{
  "predicted_class": "neutrophil",
  "confidence": 0.9734,
  "probabilities": {
    "basophil": 0.0023,
    "eosinophil": 0.0045,
    "erythroblast": 0.0012,
    "ig": 0.0034,
    "lymphocyte": 0.0089,
    "monocyte": 0.0063,
    "neutrophil": 0.9734,
    "platelet": 0.0000
  },
  "success": true
}
```

**POST /predict/batch** - Multiple images

```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@image1.png" \
  -F "files=@image2.png"
```

**POST /predict/gradcam** - Classification with Grad-CAM

```bash
curl -X POST "http://localhost:8000/predict/gradcam" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.png"
```

Returns base64-encoded Grad-CAM visualization.

**GET /health** - Health check

```bash
curl http://localhost:8000/health
```

### Model Export

#### CoreML (iOS/macOS)

Export for Apple devices:

```bash
python src/export_coreml.py \
    --checkpoint models/bloodmnist_full_exp/checkpoints/best.pth \
    --coreml \
    --quantize \
    --output_dir models/exported
```

Output: `model_int8.mlmodel` (optimized for iOS/macOS)

#### ONNX (Cross-platform)

```bash
python src/export_coreml.py \
    --checkpoint models/bloodmnist_full_exp/checkpoints/best.pth \
    --onnx \
    --output_dir models/exported
```

Output: `model.onnx` (compatible with ONNX Runtime)

#### TorchScript (PyTorch)

```bash
python src/export_coreml.py \
    --checkpoint models/bloodmnist_full_exp/checkpoints/best.pth \
    --torchscript \
    --output_dir models/exported
```

Output: `model.pt` (for PyTorch deployment)

#### Export All Formats

```bash
python src/export_coreml.py \
    --checkpoint models/bloodmnist_full_exp/checkpoints/best.pth \
    --all \
    --quantize \
    --benchmark \
    --output_dir models/exported
```

---

## Performance Benchmarks

### Best Model Results

**Model**: EfficientNet-Lite0 with SE attention  
**Dataset**: BloodMNIST Full (13,943 images)  
**Training**: 50 epochs, 4.2 hours on Apple M2

| Metric | Value |
|--------|-------|
| Validation F1 Score | 98.17% |
| Validation Accuracy | 98.21% |
| Training F1 Score | 99.90% |
| Generalization Gap | 1.73% |
| Parameters | 4.06M |
| Model Size | 16.4 MB |
| Inference Time (M2) | ~30ms |

### Per-Class Performance

| Class | Precision | Recall | F1 Score | Support |
|-------|-----------|--------|----------|---------|
| Basophil | 97.8% | 97.3% | 97.5% | 175 |
| Eosinophil | 99.1% | 99.5% | 99.3% | 189 |
| Erythroblast | 98.2% | 97.8% | 98.0% | 178 |
| IG | 97.5% | 98.1% | 97.8% | 162 |
| Lymphocyte | 98.9% | 98.6% | 98.7% | 213 |
| Monocyte | 97.3% | 97.9% | 97.6% | 189 |
| Neutrophil | 98.4% | 98.2% | 98.3% | 234 |
| Platelet | 99.2% | 99.0% | 99.1% | 272 |

### Hardware Performance

#### Apple M2 (8GB RAM)

| Model | Batch Size | Training Time/Epoch | Inference Time | Memory Usage |
|-------|------------|---------------------|----------------|--------------|
| EfficientNet-Lite0 | 32 | 5 min | 30ms | 4.2 GB |
| MobileNetV3-Small | 64 | 3 min | 20ms | 3.8 GB |
| ResNet18 | 16 | 8 min | 40ms | 5.5 GB |

#### NVIDIA RTX 3090 (24GB)

| Model | Batch Size | Training Time/Epoch | Inference Time | Memory Usage |
|-------|------------|---------------------|----------------|--------------|
| EfficientNet-Lite0 | 128 | 1.5 min | 15ms | 8.2 GB |
| ResNet18 | 256 | 2 min | 20ms | 12.4 GB |

---

## Web Interface

The Streamlit web interface provides an intuitive way to interact with CellMorphNet.

### Features

#### Image Upload
- Drag-and-drop or file browser
- Supports PNG, JPG, JPEG, BMP formats
- Preview uploaded images
- Use example images from test set

#### Preprocessing Options
- **Downscale Fix**: Enable for large images or screenshots to match training data format
- Automatically converts RGBA to RGB
- Handles various image sizes

#### Classification Results
- Predicted class with confidence score
- Color-coded confidence levels (green: high, orange: medium, red: low)
- Adjustable confidence threshold

#### Visualizations
- **Grad-CAM Heatmap**: Shows important image regions
- **Probability Bar Chart**: Distribution across all classes
- **Overlay Visualization**: Heatmap superimposed on original image

#### Model Information
- Backbone architecture
- Number of classes
- Input image size
- Training configuration

### Usage Example

1. Start the application:
```bash
streamlit run demos/streamlit_app.py
```

2. Configure settings in sidebar:
   - Select model checkpoint
   - Enable downscale fix (for screenshots)
   - Toggle Grad-CAM visualization
   - Set confidence threshold

3. Upload an image:
   - Click "Browse files" or drag-and-drop
   - Or click "Use Example" for test images

4. Click "Classify" to get results

5. Review predictions:
   - See predicted class and confidence
   - Examine Grad-CAM heatmap
   - Check probability distribution

### Screenshot Classification

For large images or screenshots (>100x100 pixels):

1. Enable "Downscale fix" in sidebar
2. This downscales images to 28x28 before processing
3. Matches training data format for better accuracy
4. Particularly useful for screenshots that are 200x200+ pixels

---

## API Documentation

### FastAPI Server

Complete API documentation is available at http://localhost:8000/docs when the server is running.

### Endpoints

#### Health Check

```http
GET /health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-10-22T12:00:00Z"
}
```

#### Single Prediction

```http
POST /predict
Content-Type: multipart/form-data

file: <image_file>
```

Response:
```json
{
  "predicted_class": "neutrophil",
  "confidence": 0.9734,
  "probabilities": {
    "basophil": 0.0023,
    "eosinophil": 0.0045,
    "erythroblast": 0.0012,
    "ig": 0.0034,
    "lymphocyte": 0.0089,
    "monocyte": 0.0063,
    "neutrophil": 0.9734,
    "platelet": 0.0000
  },
  "success": true,
  "inference_time_ms": 32.5
}
```

#### Batch Prediction

```http
POST /predict/batch
Content-Type: multipart/form-data

files: <image_file_1>
files: <image_file_2>
...
```

Response:
```json
{
  "predictions": [
    {
      "filename": "image1.png",
      "predicted_class": "neutrophil",
      "confidence": 0.9734,
      "probabilities": {...}
    },
    {
      "filename": "image2.png",
      "predicted_class": "lymphocyte",
      "confidence": 0.9821,
      "probabilities": {...}
    }
  ],
  "total_images": 2,
  "successful": 2,
  "failed": 0,
  "total_time_ms": 65.3
}
```

#### Grad-CAM Prediction

```http
POST /predict/gradcam
Content-Type: multipart/form-data

file: <image_file>
```

Response:
```json
{
  "predicted_class": "neutrophil",
  "confidence": 0.9734,
  "probabilities": {...},
  "gradcam_image": "<base64_encoded_image>",
  "success": true
}
```

### Python Client Example

```python
import requests
from PIL import Image
import io
import base64

# Server URL
url = "http://localhost:8000"

# Single prediction
with open("image.png", "rb") as f:
    response = requests.post(f"{url}/predict", files={"file": f})
    result = response.json()
    print(f"Predicted: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.2%}")

# Batch prediction
files = [
    ("files", open("image1.png", "rb")),
    ("files", open("image2.png", "rb"))
]
response = requests.post(f"{url}/predict/batch", files=files)
results = response.json()
for pred in results['predictions']:
    print(f"{pred['filename']}: {pred['predicted_class']}")

# Grad-CAM prediction
with open("image.png", "rb") as f:
    response = requests.post(f"{url}/predict/gradcam", files={"file": f})
    result = response.json()
    
    # Decode and save Grad-CAM image
    gradcam_data = base64.b64decode(result['gradcam_image'])
    gradcam_image = Image.open(io.BytesIO(gradcam_data))
    gradcam_image.save("gradcam_result.png")
```

---

## Troubleshooting

### Common Issues

#### Issue: Out of Memory During Training

**Solution**: Reduce batch size or enable gradient accumulation

```bash
python src/train.py \
    --batch_size 8 \
    --accumulation_steps 4  # Effective batch size = 32
```

#### Issue: Model Predicts Wrong Class for Screenshots

**Problem**: Screenshots are larger than training images (28x28 pixels)

**Solution**: Enable downscale fix in preprocessing

For Streamlit: Check "Enable downscale fix" in sidebar

For Python API:
```python
# Downscale large images to 28x28 first
if image.size[0] > 100 or image.size[1] > 100:
    image = image.resize((28, 28), Image.LANCZOS)
```

#### Issue: Slow Inference on CPU

**Solution**: Export to optimized format

```bash
# For Apple Silicon
python src/export_coreml.py --checkpoint model.pth --coreml --quantize

# For general deployment
python src/export_coreml.py --checkpoint model.pth --onnx
```

#### Issue: CUDA Out of Memory

**Solution**: Use mixed precision and smaller batch size

```bash
python src/train.py \
    --use_amp \
    --batch_size 16 \
    --accumulation_steps 2
```

#### Issue: Poor Accuracy on Custom Dataset

**Solutions**:
1. Ensure data is properly formatted (ImageFolder structure)
2. Check class balance - use `--use_class_weights`
3. Increase training epochs
4. Use data augmentation
5. Try transfer learning with `--pretrained --freeze_backbone`

#### Issue: Streamlit Port Already in Use

**Solution**: Use different port

```bash
streamlit run demos/streamlit_app.py --server.port 8502
```

### Getting Help

1. Check documentation files in `docs/` directory
2. Review training logs in model output directory
3. Enable debug mode: `export CUDA_LAUNCH_BLOCKING=1`
4. Open an issue on GitHub with:
   - Error message and full traceback
   - Python version and OS
   - PyTorch version
   - Command or code that caused the error

---

## Contributing

Contributions are welcome! We appreciate bug reports, feature requests, and code contributions.

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**
4. **Run tests** (if applicable)
5. **Commit your changes**
   ```bash
   git commit -m "Add: description of your changes"
   ```
6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```
7. **Open a Pull Request**

### Areas for Contribution

#### High Priority
- Additional backbone architectures (Vision Transformers, ConvNeXt)
- Multi-label classification support
- Segmentation capabilities
- Additional medical imaging datasets
- Performance optimizations

#### Medium Priority
- Web deployment guides (Heroku, AWS, GCP)
- Docker containerization
- Kubernetes deployment manifests
- CI/CD pipeline
- Unit tests and integration tests

#### Nice to Have
- iOS/macOS native applications
- Android deployment
- Progressive Web App (PWA)
- Real-time video stream processing
- Multi-language support

### Code Style

- Follow PEP 8 style guide
- Use type hints where appropriate
- Add docstrings to functions and classes
- Keep functions focused and modular
- Comment complex logic

### Pull Request Guidelines

- Provide clear description of changes
- Reference related issues
- Include test cases if applicable
- Update documentation as needed
- Ensure code passes linting

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) file for details.

### Dataset Licenses

- **BloodMNIST**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **BCCD**: Check repository for specific license
- **LISC**: Academic use only

### Third-Party Dependencies

This project uses several open-source libraries. See `requirements.txt` for full list. Major dependencies:

- **PyTorch**: BSD License
- **torchvision**: BSD License
- **Streamlit**: Apache License 2.0
- **FastAPI**: MIT License
- **Albumentations**: MIT License

### Clinical Use Disclaimer

**Important**: This software is provided for research and educational purposes only. It has not been validated for clinical diagnostic use and should not be used to make medical decisions. Always consult qualified healthcare professionals for medical advice and diagnosis.

---

## Citation

If you use CellMorphNet in your research or projects, please cite:

### This Repository

```bibtex
@software{cellmorphnet2025,
  title={CellMorphNet: Lightweight Blood Cell Classification Framework},
  author={Parth Dambhare},
  year={2025},
  url={https://github.com/parth-6-5-4/CellMorphNet}
}
```

### Datasets

**BloodMNIST**:
```bibtex
@article{medmnistv2,
  title={MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification},
  author={Yang, Jiancheng and Shi, Rui and Wei, Donglai and Liu, Zequan and Zhao, Lin and Ke, Bilian and Pfister, Hanspeter and Ni, Bingbing},
  journal={Scientific Data},
  volume={10},
  number={1},
  pages={41},
  year={2023},
  publisher={Nature Publishing Group}
}
```

**BCCD Dataset**:
```bibtex
@misc{bccd_dataset,
  title={BCCD Dataset},
  author={Shenggan},
  year={2017},
  url={https://github.com/Shenggan/BCCD_Dataset}
}
```

---

## Acknowledgments

### Datasets
- MedMNIST team for standardized medical imaging benchmarks
- BCCD Dataset contributors
- LISC Database from Monash University VL4AI Lab

### Libraries and Frameworks
- PyTorch team for the deep learning framework
- Torchvision for pretrained models and utilities
- Streamlit for rapid web application development
- FastAPI for high-performance API framework
- Albumentations for advanced image augmentation

### Community
- Contributors who have helped improve this project
- Researchers advancing medical AI and computer vision
- Open-source community for tools and resources

---

## Contact

**Project Maintainer**: Parth Dambhare

**GitHub**: [@parth-6-5-4](https://github.com/parth-6-5-4)

**Issues**: https://github.com/parth-6-5-4/CellMorphNet/issues

For questions, bug reports, or feature requests, please open an issue on GitHub.

---

## Project Status

**Current Version**: 1.0.0

**Status**: Active Development

**Last Updated**: October 2025

### Recent Updates

- **v1.0.0** (October 2025)
  - Initial release
  - EfficientNet-Lite0 model with 98.17% F1 score
  - Streamlit and FastAPI deployment options
  - Grad-CAM visualization support
  - Multi-dataset support (BloodMNIST, BCCD, LISC)
  - CoreML and ONNX export capabilities
  - Comprehensive documentation

### Roadmap

**Q4 2025**
- Add Vision Transformer support
- Implement test suite
- Docker containerization
- Performance benchmarks on more hardware

**Q1 2026**
- Segmentation module
- Multi-label classification
- Real-time video stream processing
- Mobile app prototypes

**Q2 2026**
- Clinical validation studies
- Enhanced explainability features
- Cloud deployment templates
- Extended dataset support

---

**Thank you for using CellMorphNet!**

For the latest updates and releases, please visit the [GitHub repository](https://github.com/parth-6-5-4/CellMorphNet).
