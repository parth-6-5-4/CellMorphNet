#!/bin/bash
# Quick training script for CellMorphNet on Mac M2

echo "=================================================="
echo "CellMorphNet - Quick Training Script"
echo "=================================================="
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Virtual environment not activated!"
    echo "Please run: source venv/bin/activate"
    exit 1
fi

echo "✓ Virtual environment active: $VIRTUAL_ENV"
echo ""

# Check if data exists
if [ ! -d "data/raw/bloodmnist/train" ]; then
    echo "📥 Downloading BloodMNIST dataset..."
    python scripts/download_bloodmnist.py --sample_limit 300
    echo ""
else
    echo "✓ Dataset found"
    echo ""
fi

# Train model
echo "🚀 Starting training..."
echo "Model: EfficientNet-Lite0"
echo "Dataset: BloodMNIST (subset)"
echo "Device: Mac M2 (MPS)"
echo ""

python src/train.py \
    --data_dir data/raw/bloodmnist \
    --num_classes 8 \
    --backbone efficientnet_lite0 \
    --epochs 20 \
    --batch_size 8 \
    --accumulation_steps 4 \
    --lr 1e-4 \
    --img_size 224 \
    --output_dir models/bloodmnist_quick \
    --freeze_backbone \
    --unfreeze_after 3 \
    --use_class_weights

echo ""
echo "=================================================="
echo "✓ Training complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Run inference: python src/infer.py --checkpoint models/bloodmnist_quick/checkpoints/best.pth --image <path>"
echo "2. Export model: python src/export_coreml.py --checkpoint models/bloodmnist_quick/checkpoints/best.pth --all"
echo "3. Launch demo: streamlit run demos/streamlit_app.py"
echo ""
