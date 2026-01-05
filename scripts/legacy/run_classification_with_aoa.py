#!/usr/bin/env python3
"""
Land Cover Classification with Area of Applicability (AOA)
===========================================================

This script extends the standard classification workflow with AOA analysis
to assess the reliability of spatial predictions using the method from
Meyer & Pebesma (2021).

Workflow:
1. Load KLHK reference data
2. Load and mosaic Sentinel-2 imagery
3. Calculate spectral indices
4. Rasterize KLHK ground truth
5. Extract training samples
6. Train Random Forest classifier (best model)
7. Calculate Area of Applicability (AOA)
8. Generate full classification map
9. Visualize results with AOA overlay

Reference:
Meyer, H., & Pebesma, E. (2021). Predicting into unknown space?
Estimating the area of applicability of spatial prediction models.
Methods in Ecology and Evolution, 12, 1620-1633.
https://doi.org/10.1111/2041-210X.13650

Usage:
    python scripts/run_classification_with_aoa.py
"""

import sys
import os
import numpy as np
import rasterio

# Add modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.data_loader import load_klhk_data, load_sentinel2_tiles, get_sentinel2_band_names
from modules.feature_engineering import (
    calculate_spectral_indices,
    combine_bands_and_indices,
    get_all_feature_names
)
from modules.preprocessor import (
    rasterize_klhk,
    prepare_training_data,
    split_train_test
)
from modules.model_trainer import train_all_models, get_best_model
from modules.visualizer import (
    generate_all_plots,
    export_results_to_csv,
    plot_aoa_map,
    plot_di_distribution,
    plot_classification_with_aoa,
    plot_aoa_statistics
)
from modules.aoa_calculator import (
    calculate_aoa_map,
    get_feature_importance_weights
)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Input data paths
KLHK_PATH = 'data/klhk/KLHK_PL2024_Jambi_Full_WithGeometry.geojson'
SENTINEL2_TILES = [
    'data/sentinel/S2_jambi_2024_20m_AllBands-0000000000-0000000000.tif',
    'data/sentinel/S2_jambi_2024_20m_AllBands-0000000000-0000010496.tif',
    'data/sentinel/S2_jambi_2024_20m_AllBands-0000010496-0000000000.tif',
    'data/sentinel/S2_jambi_2024_20m_AllBands-0000010496-0000010496.tif'
]

# Output directory
OUTPUT_DIR = 'results_aoa'

# Sampling configuration
SAMPLE_SIZE = 100000  # Limit training samples (set None for all)
TEST_SIZE = 0.2       # Proportion for test set
RANDOM_STATE = 42     # Random seed for reproducibility

# AOA configuration
AOA_CV_FOLDS = 10           # Cross-validation folds for AOA threshold
AOA_REMOVE_OUTLIERS = True  # Remove outliers when calculating threshold
AOA_PERCENTILE = 0.95       # Percentile for outlier removal

# Model training configuration
INCLUDE_SLOW_MODELS = False  # Only use Random Forest (best model)

# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main():
    """Main execution workflow with AOA analysis."""

    print("=" * 70)
    print("LAND COVER CLASSIFICATION WITH AREA OF APPLICABILITY (AOA)")
    print("=" * 70)
    print("\nReference: Meyer & Pebesma (2021)")
    print("https://doi.org/10.1111/2041-210X.13650")
    print("\nUsing:")
    print(f"  - KLHK Reference: {KLHK_PATH}")
    print(f"  - Sentinel-2 Tiles: {len(SENTINEL2_TILES)} tiles")
    print(f"  - Sample Size: {SAMPLE_SIZE:,}" if SAMPLE_SIZE else "  - Sample Size: All")
    print(f"  - Output Directory: {OUTPUT_DIR}")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------------
    # STEP 1: Load KLHK Reference Data
    # ------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 1: Loading KLHK Reference Data")
    print("-" * 70)

    klhk_gdf = load_klhk_data(KLHK_PATH, verbose=True)

    # ------------------------------------------------------------------------
    # STEP 2: Load Sentinel-2 Imagery
    # ------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 2: Loading Sentinel-2 Imagery")
    print("-" * 70)

    sentinel2_bands, s2_profile = load_sentinel2_tiles(SENTINEL2_TILES, verbose=True)

    # ------------------------------------------------------------------------
    # STEP 3: Calculate Spectral Indices
    # ------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 3: Calculating Spectral Indices")
    print("-" * 70)

    indices = calculate_spectral_indices(sentinel2_bands, verbose=True)

    # Combine bands and indices
    print("\nCombining bands and indices...")
    features = combine_bands_and_indices(sentinel2_bands, indices)
    print(f"  Total features: {features.shape[0]}")
    print(f"  Shape: {features.shape}")

    # ------------------------------------------------------------------------
    # STEP 4: Rasterize KLHK Ground Truth
    # ------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 4: Rasterizing KLHK Ground Truth")
    print("-" * 70)

    klhk_raster = rasterize_klhk(klhk_gdf, s2_profile, verbose=True)

    # ------------------------------------------------------------------------
    # STEP 5: Extract Training Samples
    # ------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 5: Extracting Training Samples")
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

    # Split into train/test
    print(f"\nSplitting into train/test sets (test_size={TEST_SIZE})...")
    X_train, X_test, y_train, y_test = split_train_test(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    print(f"  Training set: {len(y_train):,} samples")
    print(f"  Test set: {len(y_test):,} samples")

    # ------------------------------------------------------------------------
    # STEP 6: Train Random Forest Classifier
    # ------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 6: Training Random Forest Classifier")
    print("-" * 70)

    results = train_all_models(
        X_train, y_train,
        X_test, y_test,
        include_slow=INCLUDE_SLOW_MODELS,
        verbose=True
    )

    # Get best model
    best_name, best_result = get_best_model(results)
    best_pipeline = best_result['pipeline']

    print(f"\nBest Model: {best_name}")
    print(f"  Accuracy: {best_result['accuracy']:.4f}")
    print(f"  F1 (macro): {best_result['f1_macro']:.4f}")

    # ------------------------------------------------------------------------
    # STEP 7: Calculate Area of Applicability (AOA)
    # ------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 7: Calculating Area of Applicability (AOA)")
    print("-" * 70)

    # Extract feature importance from best model
    feature_names = get_all_feature_names()
    feature_weights = get_feature_importance_weights(best_pipeline, feature_names)

    print(f"\nTop 5 Most Important Features:")
    importance_sorted = sorted(zip(feature_names, feature_weights), key=lambda x: x[1], reverse=True)
    for i, (name, weight) in enumerate(importance_sorted[:5], 1):
        print(f"  {i}. {name}: {weight:.4f}")

    # Calculate AOA map for full prediction area
    aoa_map, di_map, aoa_threshold = calculate_aoa_map(
        features,
        X_train,
        feature_weights=feature_weights,
        cv_folds=AOA_CV_FOLDS,
        remove_outliers=AOA_REMOVE_OUTLIERS,
        percentile=AOA_PERCENTILE,
        verbose=True
    )

    # ------------------------------------------------------------------------
    # STEP 8: Generate Full Classification Map
    # ------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 8: Generating Full Classification Map")
    print("-" * 70)

    # Reshape features for prediction
    n_features, height, width = features.shape
    features_2d = features.reshape(n_features, -1).T

    print(f"Predicting {features_2d.shape[0]:,} pixels...")
    predictions = best_pipeline.predict(features_2d)
    classification_map = predictions.reshape(height, width)

    print(f"Classification map shape: {classification_map.shape}")

    # Save classification map as GeoTIFF
    classification_path = f'{OUTPUT_DIR}/classification_map.tif'
    with rasterio.open(
        classification_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=classification_map.dtype,
        crs=s2_profile['crs'],
        transform=s2_profile['transform']
    ) as dst:
        dst.write(classification_map, 1)
    print(f"Saved: {classification_path}")

    # Save AOA map as GeoTIFF
    aoa_path = f'{OUTPUT_DIR}/aoa_map.tif'
    with rasterio.open(
        aoa_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=aoa_map.dtype,
        crs=s2_profile['crs'],
        transform=s2_profile['transform']
    ) as dst:
        dst.write(aoa_map, 1)
    print(f"Saved: {aoa_path}")

    # Save DI map as GeoTIFF
    di_path = f'{OUTPUT_DIR}/di_map.tif'
    with rasterio.open(
        di_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=rasterio.float32,
        crs=s2_profile['crs'],
        transform=s2_profile['transform']
    ) as dst:
        dst.write(di_map.astype(np.float32), 1)
    print(f"Saved: {di_path}")

    # ------------------------------------------------------------------------
    # STEP 9: Generate Visualizations
    # ------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 9: Generating Visualizations")
    print("-" * 70)

    # Standard classification visualizations
    plot_paths = generate_all_plots(results, feature_names, OUTPUT_DIR, verbose=True)

    # Export results CSV
    summary_df = export_results_to_csv(
        results,
        f'{OUTPUT_DIR}/classification_results.csv',
        verbose=True
    )

    # AOA-specific visualizations
    print("\nGenerating AOA visualizations...")

    # Calculate DI for test set for distribution plot
    from modules.aoa_calculator import calculate_dissimilarity_index

    # Use a subset of pixels for DI distribution (sample 10000 pixels)
    n_pixels = features_2d.shape[0]
    sample_size_viz = min(10000, n_pixels)
    sample_idx = np.random.choice(n_pixels, sample_size_viz, replace=False)
    X_sample = features_2d[sample_idx]

    DI_sample, _, DI_train_cv = calculate_dissimilarity_index(
        X_train,
        X_sample,
        feature_weights=feature_weights,
        cv_folds=AOA_CV_FOLDS,
        remove_outliers=AOA_REMOVE_OUTLIERS,
        percentile=AOA_PERCENTILE,
        verbose=False
    )

    # Plot AOA map
    plot_aoa_map(aoa_map, di_map, aoa_threshold, OUTPUT_DIR, verbose=True)

    # Plot DI distribution
    plot_di_distribution(DI_train_cv, DI_sample, aoa_threshold, OUTPUT_DIR, verbose=True)

    # Plot classification with AOA overlay
    plot_classification_with_aoa(classification_map, aoa_map, OUTPUT_DIR, verbose=True)

    # Plot AOA statistics by class
    plot_aoa_statistics(aoa_map, classification_map, OUTPUT_DIR, verbose=True)

    # ------------------------------------------------------------------------
    # STEP 10: Results Summary
    # ------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\nClassification Performance ({best_name}):")
    print(f"  Accuracy: {best_result['accuracy']:.4f}")
    print(f"  F1 (macro): {best_result['f1_macro']:.4f}")
    print(f"  F1 (weighted): {best_result['f1_weighted']:.4f}")

    print(f"\nArea of Applicability:")
    pct_inside = (aoa_map.sum() / aoa_map.size) * 100
    print(f"  Total pixels: {aoa_map.size:,}")
    print(f"  Inside AOA: {aoa_map.sum():,} ({pct_inside:.2f}%)")
    print(f"  Outside AOA: {(aoa_map == 0).sum():,} ({100-pct_inside:.2f}%)")
    print(f"  DI Threshold: {aoa_threshold:.4f}")

    # Calculate AOA by class
    print(f"\nAOA Coverage by Land Cover Class:")
    from modules.data_loader import CLASS_NAMES
    classes = sorted(np.unique(classification_map))
    for cls in classes:
        mask = (classification_map == cls)
        total_pixels = mask.sum()
        inside_aoa = (mask & (aoa_map == 1)).sum()
        pct = (inside_aoa / total_pixels) * 100 if total_pixels > 0 else 0
        class_name = CLASS_NAMES.get(cls, f'Class {cls}')
        print(f"  {class_name}: {pct:.1f}% inside AOA")

    # ------------------------------------------------------------------------
    # COMPLETION
    # ------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CLASSIFICATION WITH AOA COMPLETE!")
    print("=" * 70)
    print(f"\nAll results saved to: {OUTPUT_DIR}/")
    print(f"\nKey outputs:")
    print(f"  - Classification map: {OUTPUT_DIR}/classification_map.tif")
    print(f"  - AOA map: {OUTPUT_DIR}/aoa_map.tif")
    print(f"  - DI map: {OUTPUT_DIR}/di_map.tif")
    print(f"  - Visualizations: {OUTPUT_DIR}/*.png")
    print(f"  - Results CSV: {OUTPUT_DIR}/classification_results.csv")

    print(f"\nAOA Method Reference:")
    print(f"  Meyer, H., & Pebesma, E. (2021). Predicting into unknown space?")
    print(f"  Estimating the area of applicability of spatial prediction models.")
    print(f"  Methods in Ecology and Evolution, 12, 1620-1633.")
    print(f"  https://doi.org/10.1111/2041-210X.13650")

    return results, aoa_map, di_map, classification_map


if __name__ == "__main__":
    results, aoa_map, di_map, classification_map = main()
