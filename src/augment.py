"""
Advanced augmentation pipelines for medical microscopy images using Albumentations.
Includes stain normalization and medical-image-specific augmentations.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from typing import Optional


class StainNormalization:
    """
    Reinhard stain normalization for microscopy images.
    Normalizes H&E stained images to a reference color distribution.
    """
    
    def __init__(self, target_means: Optional[np.ndarray] = None, target_stds: Optional[np.ndarray] = None):
        """
        Args:
            target_means: Target LAB means [L, A, B]
            target_stds: Target LAB stds [L, A, B]
        """
        # Default target statistics (computed from typical blood smear images)
        self.target_means = target_means if target_means is not None else np.array([148.60, 41.56, 20.57])
        self.target_stds = target_stds if target_stds is not None else np.array([41.56, 15.57, 10.12])
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Apply stain normalization.
        
        Args:
            image: RGB image (H x W x 3)
        
        Returns:
            Normalized RGB image
        """
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Split channels
        l, a, b = cv2.split(lab)
        
        # Compute source statistics
        l_mean, l_std = l.mean(), l.std()
        a_mean, a_std = a.mean(), a.std()
        b_mean, b_std = b.mean(), b.std()
        
        # Normalize to target statistics
        l = ((l - l_mean) / (l_std + 1e-8)) * self.target_stds[0] + self.target_means[0]
        a = ((a - a_mean) / (a_std + 1e-8)) * self.target_stds[1] + self.target_means[1]
        b = ((b - b_mean) / (b_std + 1e-8)) * self.target_stds[2] + self.target_means[2]
        
        # Clip values
        l = np.clip(l, 0, 255)
        a = np.clip(a, 0, 255)
        b = np.clip(b, 0, 255)
        
        # Merge and convert back to RGB
        lab = cv2.merge([l.astype(np.uint8), a.astype(np.uint8), b.astype(np.uint8)])
        rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return rgb


def get_train_augmentation(img_size: int = 224, stain_norm: bool = False) -> A.Compose:
    """
    Get training augmentation pipeline.
    
    Args:
        img_size: Target image size
        stain_norm: Whether to apply stain normalization
    
    Returns:
        Albumentations composition
    """
    
    transforms_list = [
        A.Resize(img_size, img_size),
    ]
    
    # Add stain normalization if requested
    if stain_norm:
        transforms_list.append(
            A.Lambda(image=StainNormalization(), name='stain_norm')
        )
    
    # Geometric augmentations
    transforms_list.extend([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.15,
            rotate_limit=45,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.5
        ),
    ])
    
    # Elastic deformation (useful for cell morphology)
    transforms_list.append(
        A.ElasticTransform(
            alpha=1,
            sigma=50,
            alpha_affine=50,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.3
        )
    )
    
    # Color augmentations (careful with medical images)
    transforms_list.extend([
        A.OneOf([
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0),
        ], p=0.5),
    ])
    
    # Noise and blur
    transforms_list.extend([
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
            A.MotionBlur(blur_limit=7, p=1.0),
        ], p=0.3),
        
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
        ], p=0.2),
    ])
    
    # Coarse dropout (simulates occlusions)
    transforms_list.append(
        A.CoarseDropout(
            max_holes=8,
            max_height=img_size // 10,
            max_width=img_size // 10,
            min_holes=1,
            fill_value=0,
            p=0.3
        )
    )
    
    # Normalization and tensor conversion
    transforms_list.extend([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])
    
    return A.Compose(transforms_list)


def get_val_test_augmentation(img_size: int = 224, stain_norm: bool = False) -> A.Compose:
    """
    Get validation/test augmentation pipeline (minimal augmentations).
    
    Args:
        img_size: Target image size
        stain_norm: Whether to apply stain normalization
    
    Returns:
        Albumentations composition
    """
    
    transforms_list = [
        A.Resize(img_size, img_size),
    ]
    
    # Add stain normalization if requested
    if stain_norm:
        transforms_list.append(
            A.Lambda(image=StainNormalization(), name='stain_norm')
        )
    
    # Normalization and tensor conversion
    transforms_list.extend([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])
    
    return A.Compose(transforms_list)


def get_light_augmentation(img_size: int = 224) -> A.Compose:
    """
    Get lightweight augmentation pipeline (for quick training on limited hardware).
    
    Args:
        img_size: Target image size
    
    Returns:
        Albumentations composition
    """
    
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, p=0.3),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])


def get_tta_augmentation(img_size: int = 224, n_augmentations: int = 5) -> list:
    """
    Get Test-Time Augmentation (TTA) pipeline.
    Returns a list of augmentation pipelines for ensemble prediction.
    
    Args:
        img_size: Target image size
        n_augmentations: Number of TTA variants
    
    Returns:
        List of Albumentations compositions
    """
    
    base_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])
    
    tta_transforms = [base_transform]  # Original
    
    # Add flips and rotations
    if n_augmentations >= 2:
        tta_transforms.append(A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]))
    
    if n_augmentations >= 3:
        tta_transforms.append(A.Compose([
            A.Resize(img_size, img_size),
            A.VerticalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]))
    
    if n_augmentations >= 4:
        tta_transforms.append(A.Compose([
            A.Resize(img_size, img_size),
            A.Rotate(limit=90, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]))
    
    if n_augmentations >= 5:
        tta_transforms.append(A.Compose([
            A.Resize(img_size, img_size),
            A.Rotate(limit=180, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]))
    
    return tta_transforms[:n_augmentations]


class AlbumentationsTransform:
    """Wrapper to use Albumentations with PyTorch Dataset."""
    
    def __init__(self, augmentation: A.Compose):
        self.augmentation = augmentation
    
    def __call__(self, image):
        """
        Args:
            image: PIL Image
        
        Returns:
            Augmented tensor
        """
        # Convert PIL to numpy
        image = np.array(image)
        
        # Apply augmentations
        augmented = self.augmentation(image=image)
        
        return augmented['image']


if __name__ == '__main__':
    """Example usage."""
    
    # Create augmentation pipelines
    train_aug = get_train_augmentation(img_size=224, stain_norm=False)
    val_aug = get_val_test_augmentation(img_size=224, stain_norm=False)
    light_aug = get_light_augmentation(img_size=224)
    
    print("Augmentation pipelines created:")
    print(f"  Train: {len(train_aug.transforms)} transforms")
    print(f"  Val/Test: {len(val_aug.transforms)} transforms")
    print(f"  Light: {len(light_aug.transforms)} transforms")
    
    # Test TTA
    tta_augs = get_tta_augmentation(img_size=224, n_augmentations=5)
    print(f"  TTA: {len(tta_augs)} augmentation variants")
