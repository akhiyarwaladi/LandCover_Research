#!/usr/bin/env python3
"""
Data Preparation Module - Deep Learning Support
================================================

This module prepares data for deep learning models (ResNet, ViT, U-Net, etc.)
Converts rasterized land cover data into patch-based datasets with proper
augmentation and DataLoader creation.

Functions:
    - extract_patches: Extract image patches from raster data
    - create_patch_dataset: Create PyTorch dataset from patches
    - get_data_loaders: Create train/val/test DataLoaders
    - apply_augmentation: Data augmentation for training

Design Philosophy:
    - Framework-agnostic where possible (can use PyTorch or TensorFlow)
    - Extensible for different DL architectures (patch-based, semantic segmentation)
    - Follows same modular pattern as other modules

Author: Claude Sonnet 4.5
Date: 2026-01-01
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# PATCH EXTRACTION
# ============================================================================

def extract_patches(features, labels, patch_size=32, stride=16,
                   max_patches=None, random_state=42, verbose=False):
    """
    Extract image patches from feature and label rasters.

    This function extracts fixed-size patches from the full raster images.
    Patches with no valid labels (all -1) are excluded.

    Parameters
    ----------
    features : numpy.ndarray
        Feature raster with shape (n_bands, height, width)
    labels : numpy.ndarray
        Label raster with shape (height, width)
    patch_size : int, default=32
        Size of square patches to extract (patch_size x patch_size)
    stride : int, default=16
        Stride for sliding window (use stride < patch_size for overlap)
    max_patches : int, optional
        Maximum number of patches to extract (for memory constraints)
    random_state : int, default=42
        Random seed for reproducible patch sampling
    verbose : bool, default=False
        Print progress information

    Returns
    -------
    X_patches : numpy.ndarray
        Patch features with shape (n_patches, n_bands, patch_size, patch_size)
    y_patches : numpy.ndarray
        Patch labels with shape (n_patches,) - center pixel label

    Notes
    -----
    - Only patches with valid labels (not -1) are kept
    - Uses center pixel as patch label (for patch-based classification)
    - For semantic segmentation, modify to return full patch labels

    Examples
    --------
    >>> X_patches, y_patches = extract_patches(
    ...     features, labels,
    ...     patch_size=32,
    ...     stride=16,
    ...     max_patches=50000
    ... )
    >>> print(f"Extracted {len(X_patches)} patches")
    """

    if verbose:
        print(f"\n📦 Extracting patches:")
        print(f"   Patch size: {patch_size}x{patch_size}")
        print(f"   Stride: {stride}")
        print(f"   Input shape: {features.shape}")

    n_bands, height, width = features.shape

    # Calculate number of patches
    n_patches_h = (height - patch_size) // stride + 1
    n_patches_w = (width - patch_size) // stride + 1
    total_patches = n_patches_h * n_patches_w

    if verbose:
        print(f"   Total possible patches: {total_patches:,}")

    # Preallocate arrays
    X_patches = []
    y_patches = []

    # Extract patches using sliding window
    for i in range(0, height - patch_size + 1, stride):
        for j in range(0, width - patch_size + 1, stride):
            # Extract patch
            patch = features[:, i:i+patch_size, j:j+patch_size]

            # Get center pixel label
            center_i = i + patch_size // 2
            center_j = j + patch_size // 2
            label = labels[center_i, center_j]

            # Skip patches with no label
            if label == -1:
                continue

            X_patches.append(patch)
            y_patches.append(label)

    # Convert to numpy arrays
    X_patches = np.array(X_patches)
    y_patches = np.array(y_patches)

    if verbose:
        print(f"   Valid patches: {len(X_patches):,}")

    # Sample if max_patches specified
    if max_patches is not None and len(X_patches) > max_patches:
        if verbose:
            print(f"   Sampling {max_patches:,} patches...")

        np.random.seed(random_state)
        indices = np.random.choice(len(X_patches), max_patches, replace=False)
        X_patches = X_patches[indices]
        y_patches = y_patches[indices]

    # Remap non-sequential labels to sequential [0,1,2,3,4,5...]
    # This is CRITICAL for PyTorch which expects class indices to be in range [0, n_classes)
    unique_labels = np.unique(y_patches)
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}

    if verbose:
        print(f"\n   ⚠️  Remapping non-sequential labels for PyTorch:")
        for old, new in label_mapping.items():
            print(f"     {old} → {new}")

    # Apply mapping
    y_patches_remapped = np.array([label_mapping[label] for label in y_patches])

    if verbose:
        print(f"\n   Final patches: {len(X_patches):,}")
        print(f"   Patch shape: {X_patches[0].shape}")

        # Class distribution
        unique, counts = np.unique(y_patches_remapped, return_counts=True)
        print(f"\n   Class distribution (after remapping):")
        for cls, count in zip(unique, counts):
            print(f"     Class {cls}: {count:,} ({count/len(y_patches_remapped)*100:.1f}%)")

    return X_patches, y_patches_remapped


# ============================================================================
# PYTORCH DATASET CLASS
# ============================================================================

class LandCoverPatchDataset(Dataset):
    """
    PyTorch Dataset for land cover patches.

    This dataset handles land cover image patches with optional augmentation.
    Compatible with PyTorch DataLoader for batch training.

    Parameters
    ----------
    X : numpy.ndarray
        Feature patches with shape (n_patches, n_bands, H, W)
    y : numpy.ndarray
        Labels with shape (n_patches,)
    transform : callable, optional
        Transformation/augmentation to apply to patches
    normalize : bool, default=True
        Normalize patches to [0, 1] range

    Examples
    --------
    >>> dataset = LandCoverPatchDataset(X_train, y_train, transform=train_transform)
    >>> loader = DataLoader(dataset, batch_size=32, shuffle=True)
    """

    def __init__(self, X, y, transform=None, normalize=True):
        self.X = X
        self.y = y
        self.transform = transform
        self.normalize = normalize

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Get patch and label
        patch = self.X[idx]
        label = self.y[idx]

        # Convert to tensor
        patch = torch.from_numpy(patch).float()
        label = torch.tensor(label, dtype=torch.long)

        # Normalize if requested
        if self.normalize:
            # Normalize to [0, 1] range (assume input is already roughly in this range)
            patch = (patch - patch.min()) / (patch.max() - patch.min() + 1e-8)

        # Apply transformations
        if self.transform:
            patch = self.transform(patch)

        return patch, label


# ============================================================================
# DATA AUGMENTATION
# ============================================================================

def get_augmentation_transforms(mode='train'):
    """
    Get data augmentation transforms for training/validation.

    Parameters
    ----------
    mode : str, default='train'
        Either 'train' (with augmentation) or 'val'/'test' (no augmentation)

    Returns
    -------
    transform : torchvision.transforms.Compose
        Composition of transforms to apply

    Notes
    -----
    Training augmentation includes:
        - Random horizontal flip
        - Random vertical flip
        - Random rotation (90, 180, 270 degrees)

    Validation/test uses no augmentation (only normalization)

    Examples
    --------
    >>> train_transform = get_augmentation_transforms('train')
    >>> val_transform = get_augmentation_transforms('val')
    """

    if mode == 'train':
        # Training augmentation
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            # Random rotation by 90 degrees increments
            transforms.RandomApply([
                transforms.RandomRotation((90, 90)),
            ], p=0.5),
        ])
    else:
        # No augmentation for validation/test
        return None


# ============================================================================
# DATALOADER CREATION
# ============================================================================

def get_data_loaders(X, y, batch_size=32, val_size=0.15, test_size=0.15,
                    random_state=42, num_workers=4, verbose=False):
    """
    Create PyTorch DataLoaders for training, validation, and testing.

    Parameters
    ----------
    X : numpy.ndarray
        Feature patches with shape (n_patches, n_bands, H, W)
    y : numpy.ndarray
        Labels with shape (n_patches,)
    batch_size : int, default=32
        Batch size for DataLoaders
    val_size : float, default=0.15
        Proportion of data for validation
    test_size : float, default=0.15
        Proportion of data for testing
    random_state : int, default=42
        Random seed for reproducible splits
    num_workers : int, default=4
        Number of worker processes for data loading
    verbose : bool, default=False
        Print split information

    Returns
    -------
    train_loader : DataLoader
        Training data loader with augmentation
    val_loader : DataLoader
        Validation data loader without augmentation
    test_loader : DataLoader
        Test data loader without augmentation

    Examples
    --------
    >>> train_loader, val_loader, test_loader = get_data_loaders(
    ...     X_patches, y_patches,
    ...     batch_size=32,
    ...     val_size=0.15,
    ...     test_size=0.15
    ... )
    >>> for batch_X, batch_y in train_loader:
    ...     # Training loop
    ...     pass
    """

    # First split: train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)  # Adjust for remaining data
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=y_trainval
    )

    if verbose:
        print(f"\n📊 Data split:")
        print(f"   Training: {len(X_train):,} patches ({len(X_train)/len(X)*100:.1f}%)")
        print(f"   Validation: {len(X_val):,} patches ({len(X_val)/len(X)*100:.1f}%)")
        print(f"   Testing: {len(X_test):,} patches ({len(X_test)/len(X)*100:.1f}%)")

    # Get transforms
    train_transform = get_augmentation_transforms('train')
    val_transform = get_augmentation_transforms('val')

    # Create datasets
    train_dataset = LandCoverPatchDataset(X_train, y_train, transform=train_transform)
    val_dataset = LandCoverPatchDataset(X_val, y_val, transform=val_transform)
    test_dataset = LandCoverPatchDataset(X_test, y_test, transform=val_transform)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    if verbose:
        print(f"\n✅ DataLoaders created:")
        print(f"   Batch size: {batch_size}")
        print(f"   Training batches: {len(train_loader)}")
        print(f"   Validation batches: {len(val_loader)}")
        print(f"   Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_class_weights(y, device='cpu'):
    """
    Calculate class weights for imbalanced data.

    Parameters
    ----------
    y : numpy.ndarray
        Label array
    device : str, default='cpu'
        Device to place weights tensor on ('cpu' or 'cuda')

    Returns
    -------
    weights : torch.Tensor
        Class weights for loss function

    Examples
    --------
    >>> weights = get_class_weights(y_train, device='cuda')
    >>> criterion = nn.CrossEntropyLoss(weight=weights)
    """

    unique, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(unique)

    # Calculate weights: inversely proportional to class frequency
    weights = n_samples / (n_classes * counts)

    # Normalize weights
    weights = weights / weights.sum() * n_classes

    # Convert to tensor
    weights = torch.tensor(weights, dtype=torch.float32, device=device)

    return weights
