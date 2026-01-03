#!/usr/bin/env python3
"""
ResNet Training - Centralized Script
=====================================

Train ResNet50 model for land cover classification.

Usage:
    python scripts/run_resnet_training.py

Outputs:
    - models/resnet50_best.pth - Best trained model
    - results/resnet/training_history.npz - Training curves
    - results/resnet/test_results.npz - Test predictions

Author: Claude Sonnet 4.5
Date: 2026-01-03
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from modules.data_loader import load_klhk_data, load_sentinel2_tiles
from modules.feature_engineering import calculate_spectral_indices, combine_bands_and_indices
from modules.preprocessor import rasterize_klhk
from modules.data_preparation import extract_patches
from modules.deep_learning_trainer import train_resnet_model

# ============================================================================
# CONFIGURATION
# ============================================================================

# Data paths
KLHK_PATH = 'data/klhk/KLHK_PL2024_Jambi_Full_WithGeometry.geojson'
PROVINCE_TILES = [
    'data/sentinel_new_cloudfree/S2_jambi_2024_20m_AllBands-0000000000-0000000000.tif',
    'data/sentinel_new_cloudfree/S2_jambi_2024_20m_AllBands-0000000000-0000010496.tif',
    'data/sentinel_new_cloudfree/S2_jambi_2024_20m_AllBands-0000010496-0000000000.tif',
    'data/sentinel_new_cloudfree/S2_jambi_2024_20m_AllBands-0000010496-0000010496.tif'
]

# Output paths
MODEL_DIR = 'models'
RESULTS_DIR = 'results/resnet'

# Training parameters
PATCH_SIZE = 32
MAX_PATCHES = 50000
BATCH_SIZE = 16
NUM_EPOCHS = 30
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-4
RANDOM_STATE = 42

print("\n" + "="*80)
print("RESNET50 TRAINING")
print("="*80)

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n" + "-"*80)
print("STEP 1: Loading Data")
print("-"*80)

print("\nLoading KLHK...")
klhk_gdf = load_klhk_data(KLHK_PATH, verbose=False)

print("Loading Sentinel-2...")
sentinel2_bands, s2_profile = load_sentinel2_tiles(PROVINCE_TILES, verbose=False)

print("Calculating spectral indices...")
indices = calculate_spectral_indices(sentinel2_bands, verbose=False)
features = combine_bands_and_indices(sentinel2_bands, indices)

print(f"✓ Features shape: {features.shape}")

print("\nRasterizing KLHK...")
klhk_raster = rasterize_klhk(klhk_gdf, s2_profile, verbose=False)

# ============================================================================
# EXTRACT AND NORMALIZE PATCHES
# ============================================================================

print("\n" + "-"*80)
print("STEP 2: Extracting Patches")
print("-"*80)

X_patches, y_patches = extract_patches(
    features, klhk_raster,
    patch_size=PATCH_SIZE,
    stride=16,
    max_patches=MAX_PATCHES,
    random_state=RANDOM_STATE,
    verbose=True
)

# Data quality check
print("\n" + "-"*80)
print("STEP 3: Data Quality Check")
print("-"*80)

has_nan = np.isnan(X_patches).any()
has_inf = np.isinf(X_patches).any()

print(f"\nNaN values: {has_nan}")
print(f"Inf values: {has_inf}")

if has_nan or has_inf:
    print("⚠️  Replacing NaN/Inf with 0...")
    X_patches = np.nan_to_num(X_patches, nan=0.0, posinf=0.0, neginf=0.0)

# Normalize features (per-channel)
print("\n" + "-"*80)
print("STEP 4: Feature Normalization")
print("-"*80)

print("\n🔧 Normalizing each channel independently...")

n_samples, n_channels, height, width = X_patches.shape
X_flat = X_patches.reshape(n_samples, n_channels, -1)

channel_means = []
channel_stds = []

for c in range(n_channels):
    channel_data = X_flat[:, c, :].flatten()
    mean = np.mean(channel_data)
    std = np.std(channel_data)

    if std < 1e-10:
        std = 1.0

    channel_means.append(mean)
    channel_stds.append(std)

    X_patches[:, c, :, :] = (X_patches[:, c, :, :] - mean) / std

    if c % 5 == 0:
        print(f"   Channel {c:2d}: mean={mean:8.4f}, std={std:8.4f}")

print("\n✅ Features normalized!")

# Save normalization parameters for prediction
os.makedirs(RESULTS_DIR, exist_ok=True)
np.savez(
    os.path.join(RESULTS_DIR, 'normalization_params.npz'),
    means=channel_means,
    stds=channel_stds
)
print(f"✓ Saved normalization parameters: {RESULTS_DIR}/normalization_params.npz")

# ============================================================================
# TRAIN MODEL
# ============================================================================

print("\n" + "-"*80)
print("STEP 5: Training ResNet50 Model")
print("-"*80)

# Train
results = train_resnet_model(
    X_patches, y_patches,
    model_name='resnet50',
    model_dir=MODEL_DIR,
    results_dir=RESULTS_DIR,
    batch_size=BATCH_SIZE,
    num_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    random_state=RANDOM_STATE,
    verbose=True
)

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("RESNET50 TRAINING COMPLETE!")
print("="*80)

print(f"\n📊 Results:")
print(f"   Test Accuracy: {results['test_accuracy']*100:.2f}%")
print(f"   Best Val Accuracy: {results['best_val_accuracy']*100:.2f}%")
print(f"   F1 (Macro): {results['f1_macro']:.4f}")
print(f"   F1 (Weighted): {results['f1_weighted']:.4f}")

print(f"\n✅ Outputs:")
print(f"   Model: {MODEL_DIR}/resnet50_best.pth")
print(f"   Training history: {RESULTS_DIR}/training_history.npz")
print(f"   Test results: {RESULTS_DIR}/test_results.npz")
print(f"   Normalization params: {RESULTS_DIR}/normalization_params.npz")

print("\n" + "="*80)
