# Datasets for CellMorphNet

This document provides information about the datasets used in CellMorphNet and instructions for downloading and preprocessing them.

## Overview

CellMorphNet supports three public blood cell datasets:

1. **BCCD** - Blood Cell Count and Detection Dataset
2. **LISC** - Leukocyte Image Segmentation and Classification
3. **BloodMNIST** - Standardized blood cell images from MedMNIST

---

## 1. BCCD Dataset

### Description
- **Source**: [GitHub - Shenggan/BCCD_Dataset](https://github.com/Shenggan/BCCD_Dataset)
- **Format**: VOC-style annotations with JPEG images
- **Classes**: RBC (Red Blood Cells), WBC (White Blood Cells), Platelets
- **Size**: ~300 images with bounding box annotations
- **Use Case**: Object detection and cell classification

### Download

**Option A: Direct download from GitHub**
```bash
git clone https://github.com/Shenggan/BCCD_Dataset.git archive/BCCD
```

**Option B: Kaggle**
```bash
# Install Kaggle CLI
pip install kaggle

# Download (requires Kaggle API credentials)
kaggle datasets download -d konstantinazov/bccd-dataset
unzip bccd-dataset.zip -d archive/BCCD
```

### Preprocessing

Process BCCD from VOC format to ImageFolder format:

```python
from src.data import BCCDPreprocessor

processor = BCCDPreprocessor(
    bccd_root='archive (1)/BCCD',
    output_root='data/processed/bccd'
)

processor.process_dataset(target_size=128)
```

This will:
- Parse VOC XML annotations
- Crop individual cells using bounding boxes
- Organize into train/val/test splits (80/10/10)
- Resize images to target size

---

## 2. LISC Dataset

### Description
- **Source**: [VL4AI - LISC Dataset](https://vl4ai.erc.monash.edu/pages/LISC.html)
- **Format**: BMP images with segmentation masks
- **Classes**: Basophil, Eosinophil, Lymphocyte, Monocyte, Neutrophil, Mixed
- **Size**: Hundreds of high-quality WBC images
- **Use Case**: Cell classification and segmentation

### Download

1. Visit: https://vl4ai.erc.monash.edu/pages/LISC.html
2. Download "Main Dataset" folder
3. Extract to `LISC Database/Main Dataset/`

### Preprocessing

Process LISC to ImageFolder format:

```python
from src.data import LISCPreprocessor

processor = LISCPreprocessor(
    lisc_root='LISC Database/Main Dataset',
    output_root='data/processed/lisc'
)

processor.process_dataset(target_size=128)
```

This will:
- Load images from class subdirectories
- Split into train/val/test (80/10/10)
- Resize images to target size
- Organize in ImageFolder structure

---

## 3. BloodMNIST Dataset

### Description
- **Source**: [MedMNIST](https://medmnist.com/)
- **Format**: NPZ file with preprocessed 28×28 RGB images
- **Classes**: 8 blood cell types (basophil, eosinophil, erythroblast, IG, lymphocyte, monocyte, neutrophil, platelet)
- **Size**: ~17,000 images (11,959 train, 1,712 val, 3,421 test)
- **Use Case**: Quick prototyping and baseline experiments

### Download

**Automated download** (recommended):

```bash
python scripts/download_bloodmnist.py --output_dir data/raw/bloodmnist --sample_limit 300
```

This script will:
- Download BloodMNIST.npz from Zenodo
- Extract a subset of images (300 samples per split by default)
- Organize into train/val/test ImageFolder structure
- Save class information

**Manual download**:

1. Visit: https://zenodo.org/records/10519652
2. Download `bloodmnist.npz`
3. Place in `data/raw/bloodmnist/`
4. Run the download script to extract images

### Full Dataset Download

For training on the complete dataset:

```bash
python scripts/download_bloodmnist.py --sample_limit 100000
```

This will extract all available images.

---

## Dataset Structure

After preprocessing, all datasets follow PyTorch ImageFolder structure:

```
data/
├── raw/                          # Raw downloaded datasets
│   ├── bloodmnist/
│   │   ├── bloodmnist.npz
│   │   ├── train/
│   │   │   ├── basophil/
│   │   │   │   ├── basophil_0000.png
│   │   │   │   └── ...
│   │   │   ├── eosinophil/
│   │   │   └── ...
│   │   ├── val/
│   │   └── test/
│   └── ...
│
└── processed/                    # Preprocessed datasets
    ├── bccd/
    │   ├── train/
    │   │   ├── RBC/
    │   │   ├── WBC/
    │   │   └── Platelet/
    │   ├── val/
    │   └── test/
    └── lisc/
        ├── train/
        ├── val/
        └── test/
```

---

## Quick Start

### 1. Download BloodMNIST (fastest to get started)

```bash
python scripts/download_bloodmnist.py --sample_limit 300
```

### 2. Train a model

```bash
python src/train.py \
    --data_dir data/raw/bloodmnist \
    --num_classes 8 \
    --backbone efficientnet_lite0 \
    --epochs 20 \
    --batch_size 16
```

### 3. Run inference

```bash
python src/infer.py \
    --checkpoint models/experiment/checkpoints/best.pth \
    --image data/raw/bloodmnist/test/basophil/basophil_0000.png \
    --gradcam
```

---

## Dataset Comparison

| Dataset | Size | Resolution | Classes | Best For |
|---------|------|------------|---------|----------|
| **BCCD** | ~300 images | Variable (640×480 typical) | 3 (RBC, WBC, Platelet) | Quick detection tasks |
| **LISC** | Hundreds | High (varied) | 6 WBC types | High-quality classification |
| **BloodMNIST** | ~17K images | 28×28 (small) | 8 cell types | Fast prototyping, baselines |

---

## Dataset Citations

### BCCD
```bibtex
@misc{bccd_dataset,
  author = {Shenggan},
  title = {BCCD Dataset},
  year = {2017},
  publisher = {GitHub},
  url = {https://github.com/Shenggan/BCCD_Dataset}
}
```

### LISC
```bibtex
@article{lisc_dataset,
  title={LISC: A Large-scale Image Set for Leukocyte Images Segmentation and Classification},
  author={Monash University},
  url={https://vl4ai.erc.monash.edu/pages/LISC.html}
}
```

### BloodMNIST
```bibtex
@article{medmnistv2,
  title={MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification},
  author={Yang, Jiancheng and Shi, Rui and Wei, Donglai and Liu, Zequan and Zhao, Lin and Ke, Bilian and Pfister, Hanspeter and Ni, Bingbing},
  journal={Scientific Data},
  year={2023},
  url={https://medmnist.com/}
}
```

---

## Troubleshooting

### Issue: "Dataset not found"
- Check that you've extracted the datasets to the correct directories
- Verify the directory structure matches the ImageFolder format
- Run preprocessing scripts if needed

### Issue: "Download failed"
- For Kaggle datasets, ensure your API credentials are configured: `~/.kaggle/kaggle.json`
- For manual downloads, check your internet connection and try again
- Consider downloading from alternative sources (GitHub, Kaggle, Zenodo)

### Issue: "Out of memory during preprocessing"
- Reduce `target_size` parameter in preprocessing scripts
- Process datasets in batches
- Use a machine with more RAM or cloud resources

---

## License and Usage

- **BCCD**: Check repository license
- **LISC**: Academic use, cite original authors
- **BloodMNIST**: CC BY 4.0 License

**Important**: These datasets are for research and educational purposes only. Do not use for clinical diagnosis without proper validation and regulatory approval.

---

## Need Help?

- Check the [main README](README.md) for general setup instructions
- Review dataset-specific documentation on their original websites
- Open an issue on GitHub if you encounter problems
