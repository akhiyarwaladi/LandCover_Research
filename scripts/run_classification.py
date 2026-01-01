#!/usr/bin/env python3
"""
Land Cover Classification - Main Orchestrator
==============================================

This script orchestrates the entire land cover classification workflow using
modular components.

Workflow:
1. Load KLHK reference data
2. Load and mosaic Sentinel-2 imagery
3. Calculate spectral indices
4. Rasterize KLHK ground truth
5. Extract training samples
6. Train multiple classifiers
7. Evaluate and compare results
8. Generate visualizations and reports

Usage:
    python scripts/run_classification.py

Configuration is done via constants at the top of the script.
"""

import sys
import os

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
from modules.visualizer import generate_all_plots, export_results_to_csv


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
OUTPUT_DIR = 'results'

# Sampling configuration
SAMPLE_SIZE = 100000  # Limit training samples (set None for all)
TEST_SIZE = 0.2       # Proportion for test set
RANDOM_STATE = 42     # Random seed for reproducibility

# Model training configuration
INCLUDE_SLOW_MODELS = False  # Include computationally expensive models (XGBoost has class label issue)

# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main():
    """Main execution workflow."""

    print("=" * 70)
    print("LAND COVER CLASSIFICATION - MODULAR WORKFLOW")
    print("=" * 70)
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
    # STEP 6: Train Classifiers
    # ------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 6: Training Classifiers")
    print("-" * 70)

    results = train_all_models(
        X_train, y_train,
        X_test, y_test,
        include_slow=INCLUDE_SLOW_MODELS,
        verbose=True
    )

    # ------------------------------------------------------------------------
    # STEP 7: Results Summary
    # ------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    # Export to CSV
    summary_df = export_results_to_csv(
        results,
        f'{OUTPUT_DIR}/classification_results.csv',
        verbose=True
    )

    print("\n" + summary_df.to_string(index=False))

    # Best classifier
    best_name, best_result = get_best_model(results)
    print(f"\n🏆 Best Classifier: {best_name}")
    print(f"   Accuracy: {best_result['accuracy']:.4f}")
    print(f"   F1 (macro): {best_result['f1_macro']:.4f}")
    print(f"   F1 (weighted): {best_result['f1_weighted']:.4f}")

    # Detailed report for best classifier
    print(f"\nClassification Report ({best_name}):")
    print(best_result['report'])

    # ------------------------------------------------------------------------
    # STEP 8: Generate Visualizations
    # ------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 8: Generating Visualizations")
    print("-" * 70)

    feature_names = get_all_feature_names()
    plot_paths = generate_all_plots(results, feature_names, OUTPUT_DIR, verbose=True)

    print(f"\nGenerated {len(plot_paths)} visualizations")

    # ------------------------------------------------------------------------
    # COMPLETION
    # ------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CLASSIFICATION COMPLETE!")
    print("=" * 70)
    print(f"\n✅ Results saved to: {OUTPUT_DIR}/")
    print(f"✅ Best model: {best_name} (F1={best_result['f1_macro']:.4f})")
    print(f"✅ Total training time: {sum(r['training_time'] for r in results.values()):.2f}s")

    return results


if __name__ == "__main__":
    results = main()
