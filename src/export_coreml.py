"""
Export CellMorphNet models to CoreML for M-series Mac inference.
"""

import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import coremltools as ct
from coremltools.models.neural_network import quantization_utils

# Import local modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.backbones import get_model


def load_checkpoint(checkpoint_path: str):
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
    
    Returns:
        model, config, class_names
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    config = checkpoint['config']
    class_names = checkpoint['class_names']
    
    # Create model
    model = get_model(
        backbone=config['backbone'],
        num_classes=config['num_classes'],
        pretrained=False
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded checkpoint:")
    print(f"  Epoch: {checkpoint['epoch'] + 1}")
    print(f"  Best F1: {checkpoint['best_val_f1']:.4f}")
    print(f"  Backbone: {config['backbone']}")
    print(f"  Classes: {class_names}")
    
    return model, config, class_names


def export_to_torchscript(
    model: nn.Module,
    img_size: int,
    output_path: str
):
    """
    Export model to TorchScript.
    
    Args:
        model: PyTorch model
        img_size: Input image size
        output_path: Output file path
    """
    model.eval()
    
    # Create example input
    example_input = torch.randn(1, 3, img_size, img_size)
    
    # Trace the model
    traced_model = torch.jit.trace(model, example_input)
    
    # Save
    traced_model.save(output_path)
    print(f"✓ Exported TorchScript model to {output_path}")
    
    # Test
    with torch.no_grad():
        original_output = model(example_input)
        traced_output = traced_model(example_input)
        
        # Check outputs match
        if torch.allclose(original_output, traced_output, atol=1e-5):
            print("  ✓ TorchScript model validated")
        else:
            print("  ⚠ Warning: TorchScript output differs from original")


def export_to_onnx(
    model: nn.Module,
    img_size: int,
    output_path: str
):
    """
    Export model to ONNX format.
    
    Args:
        model: PyTorch model
        img_size: Input image size
        output_path: Output file path
    """
    model.eval()
    
    # Create example input
    example_input = torch.randn(1, 3, img_size, img_size)
    
    # Export
    torch.onnx.export(
        model,
        example_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"✓ Exported ONNX model to {output_path}")


def export_to_coreml(
    model: nn.Module,
    img_size: int,
    class_names: list,
    output_path: str,
    quantize: bool = False
):
    """
    Export model to CoreML format.
    
    Args:
        model: PyTorch model
        img_size: Input image size
        class_names: List of class names
        output_path: Output file path
        quantize: Whether to quantize model (FP16)
    """
    model.eval()
    
    print(f"\nExporting to CoreML...")
    
    # Create example input
    example_input = torch.randn(1, 3, img_size, img_size)
    
    # Trace the model first
    traced_model = torch.jit.trace(model, example_input)
    
    # Convert to CoreML
    try:
        # Define input
        image_input = ct.ImageType(
            name="image",
            shape=(1, 3, img_size, img_size),
            scale=1/255.0,
            bias=[0, 0, 0]
        )
        
        # Convert
        coreml_model = ct.convert(
            traced_model,
            inputs=[image_input],
            classifier_config=ct.ClassifierConfig(class_names),
            minimum_deployment_target=ct.target.macOS13,
            compute_precision=ct.precision.FLOAT16 if quantize else ct.precision.FLOAT32
        )
        
        # Set metadata
        coreml_model.author = "CellMorphNet"
        coreml_model.short_description = "Blood cell classification model"
        coreml_model.version = "1.0"
        
        # Save
        coreml_model.save(output_path)
        print(f"✓ Exported CoreML model to {output_path}")
        
        # Get model size
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  Model size: {file_size:.2f} MB")
        
        if quantize:
            print(f"  Quantization: FP16")
        
    except Exception as e:
        print(f"✗ CoreML export failed: {e}")
        print("\nTrying alternative method (via ONNX)...")
        
        # Export to ONNX first
        onnx_path = output_path.replace('.mlmodel', '.onnx')
        export_to_onnx(model, img_size, onnx_path)
        
        # Convert ONNX to CoreML
        try:
            coreml_model = ct.convert(
                onnx_path,
                minimum_deployment_target=ct.target.macOS13,
                compute_precision=ct.precision.FLOAT16 if quantize else ct.precision.FLOAT32
            )
            coreml_model.save(output_path)
            print(f"✓ Exported CoreML model to {output_path} (via ONNX)")
        except Exception as e2:
            print(f"✗ Alternative method also failed: {e2}")


def benchmark_coreml(model_path: str, img_size: int):
    """
    Benchmark CoreML model inference speed.
    
    Args:
        model_path: Path to CoreML model
        img_size: Input image size
    """
    import time
    import numpy as np
    
    print(f"\nBenchmarking CoreML model...")
    
    # Load model
    model = ct.models.MLModel(model_path)
    
    # Create dummy input
    dummy_image = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
    
    # Warm up
    for _ in range(5):
        model.predict({'image': dummy_image})
    
    # Benchmark
    num_runs = 50
    times = []
    
    for _ in range(num_runs):
        start = time.time()
        model.predict({'image': dummy_image})
        times.append((time.time() - start) * 1000)  # Convert to ms
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"  Average inference time: {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"  Throughput: {1000 / avg_time:.2f} images/second")


def main():
    parser = argparse.ArgumentParser(description='Export CellMorphNet to CoreML')
    
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='models/exported', help='Output directory')
    parser.add_argument('--output_name', type=str, default='cellmorphnet', help='Output model name')
    
    parser.add_argument('--torchscript', action='store_true', help='Export to TorchScript')
    parser.add_argument('--onnx', action='store_true', help='Export to ONNX')
    parser.add_argument('--coreml', action='store_true', help='Export to CoreML')
    parser.add_argument('--all', action='store_true', help='Export to all formats')
    
    parser.add_argument('--quantize', action='store_true', help='Quantize CoreML model to FP16')
    parser.add_argument('--benchmark', action='store_true', help='Benchmark CoreML model')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("CellMorphNet Model Export")
    print("=" * 60)
    
    # Load model
    model, config, class_names = load_checkpoint(args.checkpoint)
    
    img_size = config['img_size']
    
    # Determine what to export
    export_all = args.all or not (args.torchscript or args.onnx or args.coreml)
    
    # Export TorchScript
    if args.torchscript or export_all:
        print("\n" + "-" * 60)
        print("Exporting to TorchScript...")
        print("-" * 60)
        torchscript_path = output_dir / f"{args.output_name}.pt"
        export_to_torchscript(model, img_size, str(torchscript_path))
    
    # Export ONNX
    if args.onnx or export_all:
        print("\n" + "-" * 60)
        print("Exporting to ONNX...")
        print("-" * 60)
        onnx_path = output_dir / f"{args.output_name}.onnx"
        export_to_onnx(model, img_size, str(onnx_path))
    
    # Export CoreML
    if args.coreml or export_all:
        print("\n" + "-" * 60)
        print("Exporting to CoreML...")
        print("-" * 60)
        coreml_path = output_dir / f"{args.output_name}.mlmodel"
        export_to_coreml(model, img_size, class_names, str(coreml_path), quantize=args.quantize)
        
        # Benchmark if requested
        if args.benchmark and os.path.exists(coreml_path):
            benchmark_coreml(str(coreml_path), img_size)
    
    print("\n" + "=" * 60)
    print("✓ Export complete!")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")


if __name__ == '__main__':
    main()
