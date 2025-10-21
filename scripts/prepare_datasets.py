#!/usr/bin/env python3
"""
Comprehensive dataset preparation script for CellMorphNet.
Prepares all three datasets (BloodMNIST, BCCD, LISC) for training.
"""

import os
import sys
from pathlib import Path
import argparse
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data import BCCDPreprocessor, LISCPreprocessor
from scripts.download_bloodmnist import download_bloodmnist


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def prepare_bloodmnist(output_dir='data/raw/bloodmnist', sample_limit=None):
    """
    Download and prepare BloodMNIST dataset.
    
    Args:
        output_dir: Output directory
        sample_limit: Number of samples per split (None = all)
    """
    logger.info("=" * 60)
    logger.info("Preparing BloodMNIST Dataset")
    logger.info("=" * 60)
    
    try:
        download_bloodmnist(output_dir=output_dir, sample_limit=sample_limit)
        logger.info("✓ BloodMNIST dataset prepared successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to prepare BloodMNIST: {e}")
        return False


def prepare_bccd(bccd_root='archive (1)/BCCD', output_dir='data/processed/bccd', 
                target_size=224):
    """
    Process BCCD dataset from VOC format.
    
    Args:
        bccd_root: Path to raw BCCD directory
        output_dir: Output directory
        target_size: Target image size
    """
    logger.info("=" * 60)
    logger.info("Preparing BCCD Dataset")
    logger.info("=" * 60)
    
    bccd_path = Path(bccd_root)
    if not bccd_path.exists():
        logger.warning(f"✗ BCCD directory not found: {bccd_root}")
        logger.info("Please download BCCD from:")
        logger.info("  https://github.com/Shenggan/BCCD_Dataset")
        logger.info(f"  and place it in: {bccd_root}")
        return False
    
    try:
        processor = BCCDPreprocessor(bccd_root=bccd_root, output_root=output_dir)
        processor.process_dataset(target_size=target_size)
        logger.info("✓ BCCD dataset prepared successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to prepare BCCD: {e}")
        return False


def prepare_lisc(lisc_root='LISC Database/Main Dataset', 
                output_dir='data/processed/lisc', target_size=224):
    """
    Process LISC dataset.
    
    Args:
        lisc_root: Path to raw LISC directory
        output_dir: Output directory
        target_size: Target image size
    """
    logger.info("=" * 60)
    logger.info("Preparing LISC Dataset")
    logger.info("=" * 60)
    
    lisc_path = Path(lisc_root)
    if not lisc_path.exists():
        logger.warning(f"✗ LISC directory not found: {lisc_root}")
        logger.info("Please download LISC from:")
        logger.info("  https://vl4ai.erc.monash.edu/pages/LISC.html")
        logger.info(f"  and place it in: {lisc_root}")
        return False
    
    try:
        processor = LISCPreprocessor(lisc_root=lisc_root, output_root=output_dir)
        processor.process_dataset(target_size=target_size)
        logger.info("✓ LISC dataset prepared successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to prepare LISC: {e}")
        return False


def check_dataset_status():
    """Check status of all datasets."""
    logger.info("=" * 60)
    logger.info("Dataset Status Check")
    logger.info("=" * 60)
    
    datasets = {
        'BloodMNIST': {
            'path': 'data/raw/bloodmnist/train',
            'description': 'Standardized blood cell images (MedMNIST)'
        },
        'BCCD (Processed)': {
            'path': 'data/processed/bccd/train',
            'description': 'Blood Cell Count and Detection'
        },
        'LISC (Processed)': {
            'path': 'data/processed/lisc/train',
            'description': 'Leukocyte Image Segmentation'
        }
    }
    
    available_datasets = []
    
    for name, info in datasets.items():
        path = Path(info['path'])
        if path.exists():
            num_classes = len([d for d in path.iterdir() if d.is_dir()])
            num_images = sum(len(list(d.glob('*.*'))) 
                           for d in path.iterdir() if d.is_dir())
            logger.info(f"✓ {name}: Available")
            logger.info(f"    Path: {path}")
            logger.info(f"    Classes: {num_classes}, Images: {num_images}")
            logger.info(f"    Description: {info['description']}")
            available_datasets.append(name)
        else:
            logger.info(f"✗ {name}: Not available")
            logger.info(f"    Expected path: {path}")
            logger.info(f"    Description: {info['description']}")
        logger.info("")
    
    return available_datasets


def main():
    """Main dataset preparation pipeline."""
    parser = argparse.ArgumentParser(
        description='Prepare datasets for CellMorphNet training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset selection
    parser.add_argument('--all', action='store_true',
                       help='Prepare all available datasets')
    parser.add_argument('--bloodmnist', action='store_true',
                       help='Prepare BloodMNIST dataset')
    parser.add_argument('--bccd', action='store_true',
                       help='Prepare BCCD dataset')
    parser.add_argument('--lisc', action='store_true',
                       help='Prepare LISC dataset')
    parser.add_argument('--check', action='store_true',
                       help='Check status of datasets')
    
    # BloodMNIST options
    parser.add_argument('--bloodmnist-samples', type=int, default=None,
                       help='BloodMNIST samples per split (None = all, ~11k total)')
    parser.add_argument('--bloodmnist-output', type=str, 
                       default='data/raw/bloodmnist',
                       help='BloodMNIST output directory')
    
    # BCCD options
    parser.add_argument('--bccd-root', type=str, default='archive (1)/BCCD',
                       help='BCCD raw data directory')
    parser.add_argument('--bccd-output', type=str, default='data/processed/bccd',
                       help='BCCD output directory')
    
    # LISC options
    parser.add_argument('--lisc-root', type=str, 
                       default='LISC Database/Main Dataset',
                       help='LISC raw data directory')
    parser.add_argument('--lisc-output', type=str, default='data/processed/lisc',
                       help='LISC output directory')
    
    # Common options
    parser.add_argument('--img-size', type=int, default=224,
                       help='Target image size for preprocessing')
    
    args = parser.parse_args()
    
    # Check status if requested
    if args.check:
        available = check_dataset_status()
        logger.info("=" * 60)
        logger.info(f"Summary: {len(available)}/3 datasets available")
        logger.info("=" * 60)
        return
    
    # Determine which datasets to prepare
    prepare_all = args.all or not (args.bloodmnist or args.bccd or args.lisc)
    
    results = {}
    
    # Prepare BloodMNIST
    if prepare_all or args.bloodmnist:
        results['BloodMNIST'] = prepare_bloodmnist(
            output_dir=args.bloodmnist_output,
            sample_limit=args.bloodmnist_samples
        )
    
    # Prepare BCCD
    if prepare_all or args.bccd:
        results['BCCD'] = prepare_bccd(
            bccd_root=args.bccd_root,
            output_dir=args.bccd_output,
            target_size=args.img_size
        )
    
    # Prepare LISC
    if prepare_all or args.lisc:
        results['LISC'] = prepare_lisc(
            lisc_root=args.lisc_root,
            output_dir=args.lisc_output,
            target_size=args.img_size
        )
    
    # Summary
    logger.info("=" * 60)
    logger.info("Dataset Preparation Summary")
    logger.info("=" * 60)
    
    for dataset, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        logger.info(f"{dataset}: {status}")
    
    successful = sum(1 for s in results.values() if s)
    logger.info("")
    logger.info(f"Prepared {successful}/{len(results)} datasets successfully")
    
    if successful > 0:
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Check dataset status: python scripts/prepare_datasets.py --check")
        logger.info("  2. Train model: python scripts/train_combined.py --use-bloodmnist --use-bccd --use-lisc")
    
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
