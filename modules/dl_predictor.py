#!/usr/bin/env python3
"""
Deep Learning Prediction Module
================================

Handles spatial prediction using trained deep learning models (ResNet, etc.)

Functions:
    - predict_spatial: Apply model to full raster
    - predict_patches: Batch prediction on patches
    - calculate_accuracy: Accuracy metrics for predictions

Author: Claude Sonnet 4.5
Date: 2026-01-03
"""

import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import time
import warnings
warnings.filterwarnings('ignore')


def load_resnet_model(model_path, n_channels=23, n_classes=6, device='cuda'):
    """
    Load trained ResNet model.

    Parameters
    ----------
    model_path : str
        Path to saved model weights (.pth file)
    n_channels : int, default=23
        Number of input channels
    n_classes : int, default=6
        Number of output classes
    device : str, default='cuda'
        Device to load model on ('cuda' or 'cpu')

    Returns
    -------
    model : torch.nn.Module
        Loaded ResNet model in eval mode

    Examples
    --------
    >>> model = load_resnet_model('models/resnet50_best.pth')
    >>> print(f"Model loaded on {next(model.parameters()).device}")
    """

    # Create model architecture
    model = models.resnet50(weights=None)

    # Modify first conv layer for custom channels
    model.conv1 = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Modify final layer for custom classes
    model.fc = nn.Linear(model.fc.in_features, n_classes)

    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    return model


def normalize_features(features, channel_means, channel_stds, verbose=False):
    """
    Normalize features using pre-computed channel statistics.

    Parameters
    ----------
    features : numpy.ndarray
        Feature array with shape (n_channels, height, width)
    channel_means : list or array
        Mean for each channel
    channel_stds : list or array
        Std for each channel
    verbose : bool, default=False
        Print normalization progress

    Returns
    -------
    features_normalized : numpy.ndarray
        Normalized features

    Examples
    --------
    >>> features_norm = normalize_features(features, means, stds)
    """

    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    n_channels = features.shape[0]

    if verbose:
        print(f"\nNormalizing {n_channels} channels...")

    for c in range(n_channels):
        mean = channel_means[c]
        std = channel_stds[c]

        if std < 1e-10:
            std = 1.0

        features[c, :, :] = (features[c, :, :] - mean) / std

        if verbose and c % 5 == 0:
            print(f"  Channel {c:2d}: μ={mean:8.4f}, σ={std:8.4f}")

    return features


def predict_patches(model, features, labels, patch_size=32, stride=16,
                   batch_size=64, device='cuda', verbose=False):
    """
    Predict land cover using patch-based approach.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model
    features : numpy.ndarray
        Normalized features (n_channels, height, width)
    labels : numpy.ndarray
        Ground truth labels (height, width) - used for masking
    patch_size : int, default=32
        Size of patches
    stride : int, default=16
        Stride for patch extraction
    batch_size : int, default=64
        Batch size for inference
    device : str, default='cuda'
        Device for computation
    verbose : bool, default=False
        Print progress

    Returns
    -------
    predictions : numpy.ndarray
        Predicted labels (height, width)
    stats : dict
        Prediction statistics

    Examples
    --------
    >>> predictions, stats = predict_patches(model, features, labels)
    >>> print(f"Predicted {stats['n_patches']} patches in {stats['time']:.1f}s")
    """

    n_channels, height, width = features.shape
    predictions = np.full((height, width), -1, dtype=np.int8)

    # Calculate number of patches
    n_patches_h = (height - patch_size) // stride + 1
    n_patches_w = (width - patch_size) // stride + 1
    total_possible = n_patches_h * n_patches_w

    if verbose:
        print(f"\nPredicting {height}×{width} pixels...")
        print(f"Patch size: {patch_size}×{patch_size}, stride: {stride}")
        print(f"Total possible patches: {total_possible:,}")

    # Predict in batches
    start_time = time.time()
    batch_patches = []
    batch_positions = []
    patch_count = 0

    for i in range(0, height - patch_size + 1, stride):
        for j in range(0, width - patch_size + 1, stride):
            # Extract patch
            patch = features[:, i:i+patch_size, j:j+patch_size]

            # Center position
            center_i = i + patch_size // 2
            center_j = j + patch_size // 2

            # Skip if no label (background)
            if labels[center_i, center_j] == -1:
                continue

            batch_patches.append(patch)
            batch_positions.append((center_i, center_j))

            # Process batch
            if len(batch_patches) >= batch_size:
                batch_tensor = torch.FloatTensor(np.array(batch_patches)).to(device)

                with torch.no_grad():
                    outputs = model(batch_tensor)
                    _, predicted = outputs.max(1)
                    predicted = predicted.cpu().numpy()

                # Assign predictions
                for k, (ci, cj) in enumerate(batch_positions):
                    predictions[ci, cj] = predicted[k]

                patch_count += len(batch_patches)
                batch_patches = []
                batch_positions = []

                if verbose and patch_count % 10000 == 0:
                    elapsed = time.time() - start_time
                    speed = patch_count / elapsed
                    print(f"  Processed {patch_count:,} patches ({speed:.0f} patches/sec)...")

    # Process remaining patches
    if len(batch_patches) > 0:
        batch_tensor = torch.FloatTensor(np.array(batch_patches)).to(device)

        with torch.no_grad():
            outputs = model(batch_tensor)
            _, predicted = outputs.max(1)
            predicted = predicted.cpu().numpy()

        for k, (ci, cj) in enumerate(batch_positions):
            predictions[ci, cj] = predicted[k]

        patch_count += len(batch_patches)

    elapsed = time.time() - start_time

    if verbose:
        print(f"\n✅ Prediction complete!")
        print(f"   Patches: {patch_count:,}")
        print(f"   Time: {elapsed:.1f}s")
        print(f"   Speed: {patch_count/elapsed:.0f} patches/sec")

    stats = {
        'n_patches': patch_count,
        'time': elapsed,
        'speed': patch_count / elapsed
    }

    return predictions, stats


def calculate_accuracy(predictions, ground_truth, label_mapping=None, verbose=False):
    """
    Calculate prediction accuracy.

    Parameters
    ----------
    predictions : numpy.ndarray
        Predicted labels
    ground_truth : numpy.ndarray
        Ground truth labels
    label_mapping : dict, optional
        Mapping from original labels to sequential labels
        e.g., {0: 0, 1: 1, 4: 2, 5: 3, 6: 4, 7: 5}
    verbose : bool, default=False
        Print detailed statistics

    Returns
    -------
    accuracy : float
        Overall accuracy
    stats : dict
        Detailed statistics

    Examples
    --------
    >>> acc, stats = calculate_accuracy(preds, truth, label_mapping)
    >>> print(f"Accuracy: {acc*100:.2f}%")
    """

    # Apply label mapping if provided
    if label_mapping is not None:
        gt_remapped = np.copy(ground_truth)
        for old, new in label_mapping.items():
            gt_remapped[ground_truth == old] = new
    else:
        gt_remapped = ground_truth

    # Mask for valid predictions
    valid_mask = (predictions != -1) & (ground_truth != -1)
    n_valid = valid_mask.sum()

    # Calculate accuracy
    correct = (predictions[valid_mask] == gt_remapped[valid_mask]).sum()
    accuracy = correct / n_valid if n_valid > 0 else 0.0

    if verbose:
        print(f"\n📊 Accuracy: {accuracy*100:.2f}%")
        print(f"   Valid pixels: {n_valid:,}")
        print(f"   Correct: {correct:,}")
        print(f"   Incorrect: {n_valid - correct:,}")

    stats = {
        'accuracy': accuracy,
        'n_valid': n_valid,
        'n_correct': correct,
        'n_incorrect': n_valid - correct
    }

    return accuracy, stats


def predict_spatial(model, features, labels, channel_means, channel_stds,
                   patch_size=32, stride=16, batch_size=64,
                   label_mapping=None, device='cuda', verbose=False):
    """
    Complete spatial prediction pipeline.

    This is the main function that combines normalization, prediction, and accuracy.

    Parameters
    ----------
    model : torch.nn.Module or str
        Trained model or path to model
    features : numpy.ndarray
        Raw features (n_channels, height, width)
    labels : numpy.ndarray
        Ground truth labels (height, width)
    channel_means : list or array
        Mean for each channel
    channel_stds : list or array
        Std for each channel
    patch_size : int, default=32
        Patch size for prediction
    stride : int, default=16
        Stride for patch extraction
    batch_size : int, default=64
        Batch size for inference
    label_mapping : dict, optional
        Label remapping dictionary
    device : str, default='cuda'
        Device for computation
    verbose : bool, default=False
        Print progress

    Returns
    -------
    predictions : numpy.ndarray
        Predicted labels
    results : dict
        Complete results including accuracy and statistics

    Examples
    --------
    >>> preds, results = predict_spatial(
    ...     model='models/resnet50_best.pth',
    ...     features=sentinel2_features,
    ...     labels=klhk_labels,
    ...     channel_means=means,
    ...     channel_stds=stds
    ... )
    >>> print(f"Accuracy: {results['accuracy']*100:.2f}%")
    """

    # Load model if path provided
    if isinstance(model, str):
        model = load_resnet_model(model, device=device)

    # Normalize features
    if verbose:
        print("\n" + "-"*80)
        print("Step 1: Normalizing Features")
        print("-"*80)

    features_norm = normalize_features(features, channel_means, channel_stds, verbose=verbose)

    # Predict
    if verbose:
        print("\n" + "-"*80)
        print("Step 2: Predicting")
        print("-"*80)

    predictions, pred_stats = predict_patches(
        model, features_norm, labels,
        patch_size=patch_size, stride=stride, batch_size=batch_size,
        device=device, verbose=verbose
    )

    # Calculate accuracy
    if verbose:
        print("\n" + "-"*80)
        print("Step 3: Calculating Accuracy")
        print("-"*80)

    accuracy, acc_stats = calculate_accuracy(
        predictions, labels, label_mapping=label_mapping, verbose=verbose
    )

    # Combine results
    results = {
        'accuracy': accuracy,
        'predictions': predictions,
        'prediction_stats': pred_stats,
        'accuracy_stats': acc_stats
    }

    return predictions, results
