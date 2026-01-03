#!/usr/bin/env python3
"""
ResNet Spatial Prediction - Centralized Script
===============================================

Apply trained ResNet50 model to full province.

Usage:
    python scripts/run_resnet_prediction.py

Outputs:
    - results/resnet/predictions.npy - Prediction array
    - results/resnet/predictions.tif - GeoTIFF format

Author: Claude Sonnet 4.5
Date: 2026-01-03
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import rasterio

from modules.data_loader import load_klhk_data, load_sentinel2_tiles
from modules.feature_engineering import calculate_spectral_indices, combine_bands_and_indices
from modules.preprocessor import rasterize_klhk
from modules.dl_predictor import predict_spatial

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

# Model paths
MODEL_PATH = 'models/resnet50_best.pth'
NORM_PARAMS_PATH = 'results/resnet/normalization_params.npz'

# Output path
RESULTS_DIR = 'results/resnet'

# Prediction parameters
PATCH_SIZE = 32
STRIDE = 16
BATCH_SIZE = 64
DEVICE = 'cuda'  # or 'cpu'

# Label mapping (KLHK labels to sequential)
LABEL_MAPPING = {0: 0, 1: 1, 4: 2, 5: 3, 6: 4, 7: 5}

print("\n" + "="*80)
print("RESNET50 SPATIAL PREDICTION")
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
# LOAD NORMALIZATION PARAMETERS
# ============================================================================

print("\n" + "-"*80)
print("STEP 2: Loading Normalization Parameters")
print("-"*80)

norm_params = np.load(NORM_PARAMS_PATH)
channel_means = norm_params['means']
channel_stds = norm_params['stds']

print(f"✓ Loaded normalization parameters")
print(f"   Channels: {len(channel_means)}")

# ============================================================================
# PREDICT
# ============================================================================

print("\n" + "-"*80)
print("STEP 3: Predicting")
print("-"*80)

predictions, results = predict_spatial(
    model=MODEL_PATH,
    features=features,
    labels=klhk_raster,
    channel_means=channel_means,
    channel_stds=channel_stds,
    patch_size=PATCH_SIZE,
    stride=STRIDE,
    batch_size=BATCH_SIZE,
    label_mapping=LABEL_MAPPING,
    device=DEVICE,
    verbose=True
)

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "-"*80)
print("STEP 4: Saving Results")
print("-"*80)

os.makedirs(RESULTS_DIR, exist_ok=True)

# Save as NumPy array
pred_path_npy = os.path.join(RESULTS_DIR, 'predictions.npy')
np.save(pred_path_npy, predictions)
print(f"✓ Saved: {pred_path_npy}")

# Save as GeoTIFF
pred_path_tif = os.path.join(RESULTS_DIR, 'predictions.tif')
height, width = predictions.shape

with rasterio.open(
    pred_path_tif,
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

print(f"✓ Saved: {pred_path_tif}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("RESNET50 PREDICTION COMPLETE!")
print("="*80)

print(f"\n📊 Results:")
print(f"   Accuracy: {results['accuracy']*100:.2f}%")
print(f"   Valid Pixels: {results['accuracy_stats']['n_valid']:,}")
print(f"   Prediction Time: {results['prediction_stats']['time']:.1f}s")
print(f"   Speed: {results['prediction_stats']['speed']:.0f} patches/sec")

print(f"\n✅ Outputs:")
print(f"   Predictions (NumPy): {pred_path_npy}")
print(f"   Predictions (GeoTIFF): {pred_path_tif}")

print("\n" + "="*80)
