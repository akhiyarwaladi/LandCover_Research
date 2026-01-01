#!/usr/bin/env python3
"""
STEP 2: Data Preprocessing
===========================

Loads raw data, calculates features, prepares training samples,
and saves preprocessed data to disk for classification.

Inputs:
- KLHK ground truth (GeoJSON)
- Sentinel-2 imagery (4 tiles)

Outputs:
- data/preprocessed/features.npy (23 features, shape: 23×H×W)
- data/preprocessed/labels.npy (class labels, shape: H×W)
- data/preprocessed/profile.pkl (raster metadata)
- data/preprocessed/train_test_data.npz (X_train, X_test, y_train, y_test)

Usage:
    python scripts/2_preprocess_data.py
"""

import sys
import os
import pickle
import numpy as np

# Add modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.data_loader import load_klhk_data, load_sentinel2_tiles
from modules.feature_engineering import calculate_spectral_indices, combine_bands_and_indices
from modules.preprocessor import rasterize_klhk, prepare_training_data, split_train_test


# Configuration
KLHK_PATH = 'data/klhk/KLHK_PL2024_Jambi_Full_WithGeometry.geojson'
SENTINEL2_TILES = [
    'data/sentinel/S2_jambi_2024_20m_AllBands-0000000000-0000000000.tif',
    'data/sentinel/S2_jambi_2024_20m_AllBands-0000000000-0000010496.tif',
    'data/sentinel/S2_jambi_2024_20m_AllBands-0000010496-0000000000.tif',
    'data/sentinel/S2_jambi_2024_20m_AllBands-0000010496-0000010496.tif'
]

OUTPUT_DIR = 'data/preprocessed'
SAMPLE_SIZE = 100000
TEST_SIZE = 0.2
RANDOM_STATE = 42


def main():
    print("=" * 70)
    print("STEP 2: DATA PREPROCESSING")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  KLHK Path: {KLHK_PATH}")
    print(f"  Sentinel-2 Tiles: {len(SENTINEL2_TILES)} tiles")
    print(f"  Sample Size: {SAMPLE_SIZE:,}")
    print(f"  Test Split: {TEST_SIZE}")
    print(f"  Output Directory: {OUTPUT_DIR}")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------------
    # Load KLHK Reference Data
    # ------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Loading KLHK Reference Data...")
    print("-" * 70)

    klhk_gdf = load_klhk_data(KLHK_PATH, verbose=True)

    # ------------------------------------------------------------------------
    # Load Sentinel-2 Imagery
    # ------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Loading Sentinel-2 Imagery...")
    print("-" * 70)

    sentinel2_bands, s2_profile = load_sentinel2_tiles(SENTINEL2_TILES, verbose=True)

    # ------------------------------------------------------------------------
    # Calculate Spectral Indices
    # ------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Calculating Spectral Indices...")
    print("-" * 70)

    indices = calculate_spectral_indices(sentinel2_bands, verbose=True)

    # Combine bands and indices
    print("\nCombining bands and indices...")
    features = combine_bands_and_indices(sentinel2_bands, indices)
    print(f"  Total features: {features.shape[0]}")
    print(f"  Shape: {features.shape}")

    # ------------------------------------------------------------------------
    # Rasterize KLHK Ground Truth
    # ------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Rasterizing KLHK Ground Truth...")
    print("-" * 70)

    klhk_raster = rasterize_klhk(klhk_gdf, s2_profile, verbose=True)

    # ------------------------------------------------------------------------
    # Save Full Features and Labels
    # ------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Saving Full Spatial Data...")
    print("-" * 70)

    features_path = f'{OUTPUT_DIR}/features.npy'
    labels_path = f'{OUTPUT_DIR}/labels.npy'
    profile_path = f'{OUTPUT_DIR}/profile.pkl'

    np.save(features_path, features)
    np.save(labels_path, klhk_raster)

    with open(profile_path, 'wb') as f:
        pickle.dump(s2_profile, f)

    print(f"  Saved: {features_path} ({features.nbytes / (1024**2):.2f} MB)")
    print(f"  Saved: {labels_path} ({klhk_raster.nbytes / (1024**2):.2f} MB)")
    print(f"  Saved: {profile_path}")

    # ------------------------------------------------------------------------
    # Extract Training Samples
    # ------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Extracting Training Samples...")
    print("-" * 70)

    X, y = prepare_training_data(
        features,
        klhk_raster,
        sample_size=SAMPLE_SIZE,
        random_state=RANDOM_STATE,
        verbose=True
    )

    print(f"\nFinal training data:")
    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Label vector shape: {y.shape}")

    # ------------------------------------------------------------------------
    # Split Train/Test
    # ------------------------------------------------------------------------
    print(f"\nSplitting into train/test sets (test_size={TEST_SIZE})...")
    X_train, X_test, y_train, y_test = split_train_test(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    print(f"  Training set: {len(y_train):,} samples")
    print(f"  Test set: {len(y_test):,} samples")

    # ------------------------------------------------------------------------
    # Save Train/Test Data
    # ------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Saving Train/Test Data...")
    print("-" * 70)

    train_test_path = f'{OUTPUT_DIR}/train_test_data.npz'
    np.savez_compressed(
        train_test_path,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test
    )

    print(f"  Saved: {train_test_path}")
    print(f"  Size: {os.path.getsize(train_test_path) / (1024**2):.2f} MB")

    # ------------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE!")
    print("=" * 70)
    print(f"\nPreprocessed data saved to: {OUTPUT_DIR}/")
    print(f"\nFiles created:")
    print(f"  1. features.npy - Full spatial features (23 bands)")
    print(f"  2. labels.npy - Full spatial labels")
    print(f"  3. profile.pkl - Raster metadata")
    print(f"  4. train_test_data.npz - Training/testing samples")
    print(f"\nYou can now proceed to classification:")
    print(f"  python scripts/3_run_classification.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
