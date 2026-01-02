#!/usr/bin/env python3
"""
Full Spatial Classification - Province & City
==============================================

Generates complete classification maps for:
1. Province (NEW cloud-free data, 211M pixels)
2. City (Expanded area, full 10m data extent)

Uses trained Random Forest model to predict all pixels.

Output:
- GeoTIFF classification maps
- Publication-quality comparison figures
- Accuracy assessment

Author: Claude Sonnet 4.5
Date: 2026-01-02
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import rasterio
from rasterio.merge import merge
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Import project modules
from modules.data_loader import load_klhk_data, load_sentinel2_tiles
from modules.feature_engineering import calculate_spectral_indices, combine_bands_and_indices
from modules.preprocessor import rasterize_klhk, prepare_training_data

# Import visualization
from scripts.generate_publication_figure import create_publication_figure, create_simple_comparison

# ============================================================================
# CONFIGURATION
# ============================================================================

# Data paths - NEW CLOUD-FREE PROVINCE DATA
PROVINCE_TILES = [
    'data/sentinel_new_cloudfree/S2_jambi_2024_20m_AllBands-0000000000-0000000000.tif',
    'data/sentinel_new_cloudfree/S2_jambi_2024_20m_AllBands-0000000000-0000010496.tif',
    'data/sentinel_new_cloudfree/S2_jambi_2024_20m_AllBands-0000010496-0000000000.tif',
    'data/sentinel_new_cloudfree/S2_jambi_2024_20m_AllBands-0000010496-0000010496.tif'
]

CITY_10M_PATH = 'data/sentinel_city/sentinel_city_10m_2024dry_p25.tif'
KLHK_PATH = 'data/klhk/KLHK_PL2024_Jambi_Full_WithGeometry.geojson'

# Output directory
OUTPUT_DIR = 'results/classification_maps'

# Training parameters
TRAINING_SAMPLE_SIZE = 150000  # Increased for better model
RANDOM_STATE = 42

# Model parameters (Best from previous results: Random Forest)
MODEL_PARAMS = {
    'n_estimators': 200,
    'max_depth': 25,
    'class_weight': 'balanced',
    'n_jobs': -1,
    'random_state': RANDOM_STATE,
    'verbose': 1
}

# ============================================================================
# TRAINING
# ============================================================================

def train_classification_model(verbose=True):
    """
    Train Random Forest model on sampled data.

    Returns:
        pipeline: Trained sklearn pipeline
        feature_names: List of feature names
    """

    if verbose:
        print("\n" + "="*80)
        print("TRAINING CLASSIFICATION MODEL")
        print("="*80)

    # Load data
    print("\n1. Loading Sentinel-2 data (province)...")
    sentinel2_bands, s2_profile = load_sentinel2_tiles(PROVINCE_TILES, verbose=verbose)

    print("\n2. Calculating spectral indices...")
    indices = calculate_spectral_indices(sentinel2_bands, verbose=verbose)

    print("\n3. Combining features...")
    features = combine_bands_and_indices(sentinel2_bands, indices)
    print(f"   Total features: {features.shape[0]}")

    print("\n4. Loading KLHK ground truth...")
    klhk_gdf = load_klhk_data(KLHK_PATH, verbose=verbose)

    print("\n5. Rasterizing KLHK...")
    klhk_raster = rasterize_klhk(klhk_gdf, s2_profile, verbose=verbose)

    print("\n6. Preparing training data...")
    X, y = prepare_training_data(features, klhk_raster,
                                 sample_size=TRAINING_SAMPLE_SIZE,
                                 random_state=RANDOM_STATE,
                                 verbose=verbose)

    print(f"\n7. Training Random Forest model...")
    print(f"   Training samples: {len(X):,}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Classes: {len(np.unique(y))}")

    # Create pipeline
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(**MODEL_PARAMS))
    ])

    # Train
    start_time = time.time()
    pipeline.fit(X, y)
    training_time = time.time() - start_time

    print(f"   ✅ Training complete in {training_time:.2f} seconds")

    # Feature names
    from modules.feature_engineering import get_all_feature_names
    feature_names = get_all_feature_names()

    return pipeline, feature_names, s2_profile

# ============================================================================
# PREDICTION
# ============================================================================

def predict_full_map(pipeline, data_path_or_list, output_path,
                    is_tiles=False, verbose=True):
    """
    Predict classification for full spatial extent.

    Args:
        pipeline: Trained sklearn pipeline
        data_path_or_list: Single file path or list of tile paths
        output_path: Path to save prediction GeoTIFF
        is_tiles: Whether input is tiles (need mosaic)
        verbose: Print progress

    Returns:
        prediction_map: 2D array of predicted classes
        profile: Raster profile
    """

    if verbose:
        print("\n" + "="*80)
        print("FULL SPATIAL PREDICTION")
        print("="*80)

    # Load data
    if is_tiles:
        print("\nLoading and mosaicking tiles...")
        datasets = []
        for path in data_path_or_list:
            if os.path.exists(path):
                datasets.append(rasterio.open(path))

        data, transform = merge(datasets)
        profile = datasets[0].profile.copy()
        profile.update({
            'transform': transform,
            'height': data.shape[1],
            'width': data.shape[2]
        })

        for ds in datasets:
            ds.close()
    else:
        print(f"\nLoading: {os.path.basename(data_path_or_list)}")
        with rasterio.open(data_path_or_list) as src:
            data = src.read()
            profile = src.profile.copy()

    print(f"Data shape: {data.shape}")
    print(f"Dimensions: {data.shape[1]} × {data.shape[2]} pixels")

    # Calculate features
    print("\nCalculating spectral indices...")
    sentinel2_bands = data
    indices = calculate_spectral_indices(sentinel2_bands, verbose=False)
    features = combine_bands_and_indices(sentinel2_bands, indices)

    print(f"Total features: {features.shape[0]}")

    # Prepare for prediction
    height, width = features.shape[1], features.shape[2]
    n_features = features.shape[0]

    # Reshape for prediction
    features_flat = features.reshape(n_features, -1).T  # (n_pixels, n_features)

    # Find valid pixels
    valid_mask = ~np.isnan(features_flat).any(axis=1)
    n_valid = valid_mask.sum()
    n_total = len(valid_mask)

    print(f"\nPixel statistics:")
    print(f"  Total pixels: {n_total:,}")
    print(f"  Valid pixels: {n_valid:,} ({n_valid/n_total*100:.1f}%)")
    print(f"  NaN pixels: {n_total - n_valid:,} ({(n_total-n_valid)/n_total*100:.1f}%)")

    # Predict in batches to avoid memory error
    print(f"\nPredicting {n_valid:,} valid pixels...")
    print("Using batch processing to avoid memory issues...")

    prediction_flat = np.full(n_total, -1, dtype=np.int16)

    # Batch size: 5 million pixels per batch
    batch_size = 5_000_000
    valid_indices = np.where(valid_mask)[0]
    n_batches = (n_valid + batch_size - 1) // batch_size

    print(f"Processing in {n_batches} batches of up to {batch_size:,} pixels each")

    start_time = time.time()
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_valid)

        # Get indices for this batch
        batch_indices = valid_indices[start_idx:end_idx]

        # Predict batch
        prediction_flat[batch_indices] = pipeline.predict(features_flat[batch_indices])

        # Progress update
        elapsed = time.time() - start_time
        progress = end_idx / n_valid * 100
        print(f"  Batch {i+1}/{n_batches}: {end_idx:,}/{n_valid:,} pixels ({progress:.1f}%) - {elapsed:.1f}s elapsed")

    pred_time = time.time() - start_time

    print(f"✅ Prediction complete in {pred_time:.2f} seconds")
    print(f"   Speed: {n_valid/pred_time:.0f} pixels/second")

    # Reshape to 2D
    prediction_map = prediction_flat.reshape(height, width)

    # Save as GeoTIFF
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    profile.update({
        'dtype': 'int16',
        'count': 1,
        'compress': 'lzw',
        'nodata': -1
    })

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(prediction_map, 1)

    file_size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"\n✅ Saved: {output_path}")
    print(f"   Size: {file_size_mb:.1f} MB")

    return prediction_map, profile

# ============================================================================
# COMPARISON VISUALIZATION
# ============================================================================

def create_comparison_figure(ground_truth_raster, prediction_map,
                            title, output_path, verbose=True):
    """
    Create publication figure comparing ground truth vs prediction.

    Args:
        ground_truth_raster: Ground truth labels (2D array)
        prediction_map: Predicted labels (2D array)
        title: Figure title
        output_path: Path to save figure
    """

    if verbose:
        print("\n" + "="*80)
        print("CREATING COMPARISON FIGURE")
        print("="*80)

    # Ensure same shape
    if ground_truth_raster.shape != prediction_map.shape:
        print(f"Warning: Shape mismatch!")
        print(f"  Ground truth: {ground_truth_raster.shape}")
        print(f"  Prediction: {prediction_map.shape}")
        return

    # Create figure
    create_publication_figure(
        ground_truth_raster,
        prediction_map,
        title=title,
        output_path=output_path,
        dpi=300
    )

# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main():
    """Run full spatial classification workflow."""

    print("\n" + "="*80)
    print("FULL SPATIAL CLASSIFICATION - PROVINCE & CITY")
    print("="*80)
    print("\nUsing NEW cloud-free data:")
    print("  • Province: 20m resolution (4 tiles)")
    print("  • City: 10m resolution (expanded extent)")
    print("\n" + "="*80)

    # ========================================================================
    # STEP 1: TRAIN MODEL
    # ========================================================================

    print("\n" + "="*80)
    print("STEP 1: TRAIN CLASSIFICATION MODEL")
    print("="*80)

    pipeline, feature_names, province_profile = train_classification_model(verbose=True)

    # ========================================================================
    # STEP 2: PREDICT PROVINCE
    # ========================================================================

    print("\n" + "="*80)
    print("STEP 2: CLASSIFY FULL PROVINCE")
    print("="*80)
    print("\nClassifying entire Jambi Province...")
    print("  • Area: ~49,224 km²")
    print("  • Pixels: ~211 million at 20m")
    print("  • Expected time: 5-10 minutes")

    province_pred_path = os.path.join(OUTPUT_DIR, 'classification_province_20m.tif')

    province_prediction, province_profile = predict_full_map(
        pipeline,
        PROVINCE_TILES,
        province_pred_path,
        is_tiles=True,
        verbose=True
    )

    # Load province ground truth
    print("\nLoading province ground truth for comparison...")
    klhk_gdf = load_klhk_data(KLHK_PATH, verbose=False)
    province_gt = rasterize_klhk(klhk_gdf, province_profile, verbose=False)

    # Create province comparison figure
    province_fig_path = os.path.join(OUTPUT_DIR, 'comparison_province_20m.png')
    create_comparison_figure(
        province_gt,
        province_prediction,
        title='Land Cover Classification - Jambi Province (20m Resolution)',
        output_path=province_fig_path,
        verbose=True
    )

    # ========================================================================
    # STEP 3: PREDICT CITY
    # ========================================================================

    print("\n" + "="*80)
    print("STEP 3: CLASSIFY CITY (EXPANDED EXTENT)")
    print("="*80)
    print("\nClassifying Jambi City...")
    print("  • Area: ~285 km² (full downloaded extent)")
    print("  • Pixels: ~2.9 million at 10m")
    print("  • Expected time: 30-60 seconds")

    city_pred_path = os.path.join(OUTPUT_DIR, 'classification_city_10m.tif')

    city_prediction, city_profile = predict_full_map(
        pipeline,
        CITY_10M_PATH,
        city_pred_path,
        is_tiles=False,
        verbose=True
    )

    # Load city ground truth
    print("\nLoading city ground truth for comparison...")
    city_gt = rasterize_klhk(klhk_gdf, city_profile, verbose=False)

    # Create city comparison figure
    city_fig_path = os.path.join(OUTPUT_DIR, 'comparison_city_10m.png')
    create_comparison_figure(
        city_gt,
        city_prediction,
        title='Land Cover Classification - Jambi City (10m Resolution)',
        output_path=city_fig_path,
        verbose=True
    )

    # ========================================================================
    # SUMMARY
    # ========================================================================

    print("\n" + "="*80)
    print("CLASSIFICATION COMPLETE!")
    print("="*80)

    print(f"\n📊 Output Files:")
    print(f"\n1. Province Classification:")
    print(f"   Map: {province_pred_path}")
    print(f"   Figure: {province_fig_path}")

    print(f"\n2. City Classification:")
    print(f"   Map: {city_pred_path}")
    print(f"   Figure: {city_fig_path}")

    print(f"\n✅ All classification maps and figures ready!")
    print(f"✅ Publication-quality figures at 300 DPI")
    print(f"\n" + "="*80)

if __name__ == '__main__':
    main()
