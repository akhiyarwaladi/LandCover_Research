#!/usr/bin/env python3
"""
Generate Full Spatial Predictions with Trained ResNet50
========================================================

This script:
1. Loads the trained ResNet50 model
2. Applies it to full province and city regions
3. Generates colorful visualizations

Author: Claude Sonnet 4.5
Date: 2026-01-03
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time

from modules.data_loader import load_klhk_data, load_sentinel2_tiles
from modules.feature_engineering import calculate_spectral_indices, combine_bands_and_indices
from modules.preprocessor import rasterize_klhk

# ============================================================================
# CONFIGURATION
# ============================================================================

KLHK_PATH = 'data/klhk/KLHK_PL2024_Jambi_Full_WithGeometry.geojson'
PROVINCE_TILES = [
    'data/sentinel_new_cloudfree/S2_jambi_2024_20m_AllBands-0000000000-0000000000.tif',
    'data/sentinel_new_cloudfree/S2_jambi_2024_20m_AllBands-0000000000-0000010496.tif',
    'data/sentinel_new_cloudfree/S2_jambi_2024_20m_AllBands-0000010496-0000000000.tif',
    'data/sentinel_new_cloudfree/S2_jambi_2024_20m_AllBands-0000010496-0000010496.tif'
]

MODEL_PATH = 'models/resnet50_fixed_best.pth'
OUTPUT_DIR = 'results/resnet_predictions'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Colorful color scheme (from user preference)
CLASS_COLORS_JAMBI = {
    0: '#0066CC',  # Water - Bright Blue
    1: '#228B22',  # Trees/Forest - Forest Green
    2: '#90EE90',  # Crops - Light Green (dominant class)
    3: '#FF8C00',  # Shrub - Dark Orange
    4: '#FF1493',  # Built - Deep Pink/Magenta
    5: '#D2691E',  # Bare Ground - Chocolate Brown
}

CLASS_NAMES = ['Water', 'Trees', 'Crops', 'Shrub', 'Built', 'Bare']

# From training script - these are the normalization parameters
CHANNEL_MEANS = [0.0379, 0.0541, 0.0375, 0.0840, 0.2183, 0.2707, 0.2671, 0.2981,
                 0.1629, 0.0773, 0.7243, 0.4698, 0.4160, -0.6342, -0.4744, -0.2425,
                 -0.2101, 0.5001, 155.6564, 0.4056, 0.6342, 0.2425, 0.5398]
CHANNEL_STDS = [0.0196, 0.0230, 0.0259, 0.0330, 0.0693, 0.0857, 0.0865, 0.0949,
                0.0630, 0.0393, 0.2036, 0.1656, 0.1368, 0.1721, 0.1739, 0.1386,
                0.1290, 0.1582, 651187.3125, 0.1487, 0.1721, 0.1386, 0.1758]

# Patch prediction parameters
PATCH_SIZE = 32
STRIDE = 16  # Use smaller stride for denser predictions
BATCH_SIZE = 64

print("\n" + "="*80)
print("RESNET50 SPATIAL PREDICTION")
print("="*80)
print(f"\nDevice: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

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
# NORMALIZE FEATURES
# ============================================================================

print("\n" + "-"*80)
print("STEP 2: Normalizing Features")
print("-"*80)

print("\n🔧 Normalizing using training statistics...")
n_channels, height, width = features.shape

# Replace NaN/Inf
features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

# Normalize each channel
for c in range(n_channels):
    mean = CHANNEL_MEANS[c]
    std = CHANNEL_STDS[c]
    if std < 1e-10:
        std = 1.0
    features[c, :, :] = (features[c, :, :] - mean) / std

print("✅ Features normalized!")

# ============================================================================
# LOAD MODEL
# ============================================================================

print("\n" + "-"*80)
print("STEP 3: Loading Trained ResNet50 Model")
print("-"*80)

# Create model architecture (must match training)
model = models.resnet50(pretrained=False)

# Modify first conv layer for 23 channels
model.conv1 = nn.Conv2d(23, 64, kernel_size=7, stride=2, padding=3, bias=False)

# Modify final layer for 6 classes
model.fc = nn.Linear(model.fc.in_features, 6)

# Load trained weights
model.load_state_dict(torch.load(MODEL_PATH))
model = model.to(DEVICE)
model.eval()

print(f"✅ Model loaded from: {MODEL_PATH}")

# ============================================================================
# PREDICT FULL PROVINCE
# ============================================================================

print("\n" + "-"*80)
print("STEP 4: Predicting Full Province")
print("-"*80)

print(f"\nPredicting {height}x{width} = {height*width:,} pixels...")
print(f"Using patch size: {PATCH_SIZE}x{PATCH_SIZE}, stride: {STRIDE}")

# Initialize prediction array
predictions = np.full((height, width), -1, dtype=np.int8)

# Calculate number of patches
n_patches_h = (height - PATCH_SIZE) // STRIDE + 1
n_patches_w = (width - PATCH_SIZE) // STRIDE + 1
total_patches = n_patches_h * n_patches_w

print(f"Total patches: {total_patches:,}")

# Predict in batches
start_time = time.time()
batch_patches = []
batch_positions = []
patch_count = 0

for i in range(0, height - PATCH_SIZE + 1, STRIDE):
    for j in range(0, width - PATCH_SIZE + 1, STRIDE):
        # Extract patch
        patch = features[:, i:i+PATCH_SIZE, j:j+PATCH_SIZE]

        # Center position
        center_i = i + PATCH_SIZE // 2
        center_j = j + PATCH_SIZE // 2

        # Skip if no label (background)
        if klhk_raster[center_i, center_j] == -1:
            continue

        batch_patches.append(patch)
        batch_positions.append((center_i, center_j))

        # Process batch
        if len(batch_patches) >= BATCH_SIZE:
            batch_tensor = torch.FloatTensor(np.array(batch_patches)).to(DEVICE)

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

            if patch_count % 10000 == 0:
                elapsed = time.time() - start_time
                print(f"  Processed {patch_count:,} patches in {elapsed:.1f}s...")

# Process remaining patches
if len(batch_patches) > 0:
    batch_tensor = torch.FloatTensor(np.array(batch_patches)).to(DEVICE)

    with torch.no_grad():
        outputs = model(batch_tensor)
        _, predicted = outputs.max(1)
        predicted = predicted.cpu().numpy()

    for k, (ci, cj) in enumerate(batch_positions):
        predictions[ci, cj] = predicted[k]

    patch_count += len(batch_patches)

elapsed = time.time() - start_time
print(f"\n✅ Prediction complete!")
print(f"   Total patches: {patch_count:,}")
print(f"   Time: {elapsed:.1f}s")
print(f"   Speed: {patch_count/elapsed:.0f} patches/sec")

# ============================================================================
# CALCULATE ACCURACY
# ============================================================================

print("\n" + "-"*80)
print("STEP 5: Calculating Accuracy")
print("-"*80)

# Mask for valid predictions
valid_mask = (predictions != -1) & (klhk_raster != -1)
n_valid = valid_mask.sum()

# Map KLHK classes to sequential [0,1,2,3,4,5]
# Original labels: [0, 1, 4, 5, 6, 7]
# Remapped: [0, 1, 2, 3, 4, 5]
label_mapping = {0: 0, 1: 1, 4: 2, 5: 3, 6: 4, 7: 5}
klhk_remapped = np.copy(klhk_raster)
for old, new in label_mapping.items():
    klhk_remapped[klhk_raster == old] = new

# Calculate accuracy
correct = (predictions[valid_mask] == klhk_remapped[valid_mask]).sum()
accuracy = correct / n_valid

print(f"\nProvince Accuracy: {accuracy*100:.2f}%")
print(f"Valid pixels: {n_valid:,}")
print(f"Correct: {correct:,}")

# ============================================================================
# SAVE PREDICTIONS
# ============================================================================

print("\n" + "-"*80)
print("STEP 6: Saving Results")
print("-"*80)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save as numpy array
np.save(os.path.join(OUTPUT_DIR, 'province_predictions.npy'), predictions)
print(f"✓ Saved: {OUTPUT_DIR}/province_predictions.npy")

# Save as GeoTIFF
with rasterio.open(
    os.path.join(OUTPUT_DIR, 'province_predictions.tif'),
    'w',
    driver='GTiff',
    height=height,
    width=width,
    count=1,
    dtype=predictions.dtype,
    crs=s2_profile['crs'],
    transform=s2_profile['transform'],
    compress='lzw'
) as dst:
    dst.write(predictions, 1)

print(f"✓ Saved: {OUTPUT_DIR}/province_predictions.tif")

# ============================================================================
# VISUALIZE
# ============================================================================

print("\n" + "-"*80)
print("STEP 7: Creating Visualizations")
print("-"*80)

# Create colormap
colors = [CLASS_COLORS_JAMBI[i] for i in range(6)]
cmap = ListedColormap(colors)

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(20, 10))

# Plot ground truth
ax = axes[0]
klhk_plot = np.copy(klhk_remapped).astype(float)
klhk_plot[klhk_raster == -1] = np.nan
im1 = ax.imshow(klhk_plot, cmap=cmap, vmin=0, vmax=5, interpolation='nearest')
ax.set_title('Ground Truth (KLHK 2024)', fontsize=16, fontweight='bold')
ax.axis('off')

# Plot predictions
ax = axes[1]
pred_plot = np.copy(predictions).astype(float)
pred_plot[predictions == -1] = np.nan
im2 = ax.imshow(pred_plot, cmap=cmap, vmin=0, vmax=5, interpolation='nearest')
ax.set_title(f'ResNet50 Predictions (Acc: {accuracy*100:.2f}%)', fontsize=16, fontweight='bold')
ax.axis('off')

# Add colorbar
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=CLASS_COLORS_JAMBI[i], label=CLASS_NAMES[i])
                   for i in range(6)]
fig.legend(handles=legend_elements, loc='lower center', ncol=6,
           fontsize=12, frameon=False, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'province_comparison.png'),
            dpi=300, bbox_inches='tight')
print(f"✓ Saved: {OUTPUT_DIR}/province_comparison.png")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("RESNET50 PREDICTION COMPLETE!")
print("="*80)
print(f"\n📊 Results:")
print(f"   Province Accuracy: {accuracy*100:.2f}%")
print(f"   Valid Pixels: {n_valid:,}")
print(f"   Prediction Time: {elapsed:.1f}s")
print(f"\n✅ Outputs saved to: {OUTPUT_DIR}/")
print("\n" + "="*80)
