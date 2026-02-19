"""
Train Random Forest for Change Detection

Approach C: Pixel-based Random Forest using stacked temporal features
(T1 features + T2 features + change features = 56 features).

Usage:
    python scripts/train_rf_change_detection.py

Input:
    data/sentinel/{year}/S2_jambi_{year}_20m_AllBands*.tif
    data/change_labels/annual/change_{year}.tif

Output:
    results/models/rf_change/
        rf_model.joblib
        test_results.npz
        feature_importance.npz
"""

import os
import sys
import numpy as np
import joblib
import rasterio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.data_loader import (
    load_sentinel2_tiles, find_sentinel2_tiles, get_consecutive_year_pairs
)
from modules.feature_engineering import (
    calculate_spectral_indices, combine_bands_and_indices,
    get_stacked_feature_names
)
from modules.preprocessor import (
    prepare_pixel_training_data, split_train_test,
    create_bitemporal_change_label
)
from modules.model_trainer import (
    train_change_model, get_change_classifiers,
    get_feature_importance, get_best_model
)


# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
SENTINEL_DIR = os.path.join(DATA_DIR, 'sentinel')
HANSEN_DIR = os.path.join(DATA_DIR, 'hansen')
LABELS_DIR = os.path.join(DATA_DIR, 'change_labels', 'annual')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

SAMPLE_SIZE = 200000
BALANCE_RATIO = 3.0
RANDOM_STATE = 42


def load_year_features(year, verbose=True):
    """Load and compute features for a single year."""
    tiles = find_sentinel2_tiles(SENTINEL_DIR, year)

    if not tiles:
        if verbose:
            print(f"  No tiles for {year}")
        return None, None

    bands, profile = load_sentinel2_tiles(tiles, verbose=verbose)
    indices = calculate_spectral_indices(bands, verbose=verbose)
    features = combine_bands_and_indices(bands, indices)

    return features, profile


def load_change_label(year):
    """Load change label for a year."""
    path = os.path.join(LABELS_DIR, f'change_{year}.tif')
    if os.path.exists(path):
        with rasterio.open(path) as src:
            return src.read(1)
    return None


def main():
    """Train Random Forest change detection model."""
    print("=" * 60)
    print("RANDOM FOREST CHANGE DETECTION TRAINING")
    print("=" * 60)

    save_dir = os.path.join(RESULTS_DIR, 'models', 'rf_change')
    os.makedirs(save_dir, exist_ok=True)

    year_pairs = get_consecutive_year_pairs()
    feature_names = get_stacked_feature_names()

    print(f"\n  Year pairs: {len(year_pairs)}")
    print(f"  Features per pixel: {len(feature_names)}")
    print(f"  Max samples per pair: {SAMPLE_SIZE:,}")

    # Collect training data from all year pairs
    all_X = []
    all_y = []

    for year1, year2 in year_pairs:
        print(f"\n--- Processing {year1} -> {year2} ---")

        t1_features, _ = load_year_features(year1, verbose=False)
        t2_features, _ = load_year_features(year2, verbose=False)

        if t1_features is None or t2_features is None:
            print(f"  SKIPPED: Missing Sentinel-2 data")
            continue

        label = load_change_label(year2)

        if label is None:
            # Try Hansen directly
            lossyear_path = os.path.join(HANSEN_DIR, 'Hansen_lossyear_Jambi.tif')
            treecover_path = os.path.join(HANSEN_DIR, 'Hansen_treecover2000_Jambi.tif')
            if os.path.exists(lossyear_path):
                with rasterio.open(lossyear_path) as src:
                    lossyear = src.read(1)
                treecover = None
                if os.path.exists(treecover_path):
                    with rasterio.open(treecover_path) as src:
                        treecover = src.read(1)
                label = create_bitemporal_change_label(
                    lossyear, year1, year2, treecover2000=treecover
                )
            else:
                print(f"  SKIPPED: No change labels")
                continue

        # Ensure shapes match
        min_h = min(t1_features.shape[1], t2_features.shape[1], label.shape[0])
        min_w = min(t1_features.shape[2], t2_features.shape[2], label.shape[1])
        t1_crop = t1_features[:, :min_h, :min_w]
        t2_crop = t2_features[:, :min_h, :min_w]
        label_crop = label[:min_h, :min_w]

        # Prepare pixel samples
        X, y = prepare_pixel_training_data(
            t1_crop, t2_crop, label_crop,
            sample_size=SAMPLE_SIZE // len(year_pairs),
            balance_ratio=BALANCE_RATIO,
            random_state=RANDOM_STATE
        )

        if len(y) > 0:
            all_X.append(X)
            all_y.append(y)
            print(f"  Samples: {len(y):,} ({np.sum(y == 1):,} change)")

    if not all_X:
        print("\nERROR: No training data collected!")
        print("Ensure Sentinel-2 data and change labels are available")
        return

    # Combine all data
    X = np.vstack(all_X)
    y = np.concatenate(all_y)

    print(f"\n{'='*50}")
    print(f"TOTAL TRAINING DATA")
    print(f"{'='*50}")
    print(f"  Samples: {len(y):,}")
    print(f"  Change: {np.sum(y == 1):,}")
    print(f"  No-change: {np.sum(y == 0):,}")
    print(f"  Features: {X.shape[1]}")

    # Split
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.2)

    # Train Random Forest
    classifiers = get_change_classifiers()
    results = {}

    for name, pipeline in classifiers.items():
        result = train_change_model(
            pipeline, X_train, y_train, X_test, y_test, name
        )
        results[name] = result

    # Get best model
    best_name, best_result = get_best_model(results)
    print(f"\n  Best model: {best_name}")
    print(f"  Accuracy: {best_result['accuracy']:.4f}")
    print(f"  F1 (macro): {best_result['f1_macro']:.4f}")

    # Save best model
    joblib.dump(best_result['pipeline'], os.path.join(save_dir, 'rf_model.joblib'))

    # Save test results
    np.savez(
        os.path.join(save_dir, 'test_results.npz'),
        predictions=best_result['y_pred'],
        targets=best_result['y_test'],
        accuracy=best_result['accuracy'],
        f1_macro=best_result['f1_macro'],
        f1_weighted=best_result['f1_weighted'],
        f1_change=best_result['f1_change'],
        kappa=best_result['kappa'],
        confusion_matrix=best_result['confusion_matrix'],
    )

    # Feature importance
    importance = get_feature_importance(
        best_result['pipeline'], feature_names=feature_names, top_n=20
    )

    if importance:
        np.savez(
            os.path.join(save_dir, 'feature_importance.npz'),
            names=[x[0] for x in importance],
            values=[x[1] for x in importance],
        )

    # Save all results for comparison
    summary = {}
    for name, result in results.items():
        summary[name] = {
            'accuracy': result['accuracy'],
            'f1_macro': result['f1_macro'],
            'f1_change': result['f1_change'],
            'kappa': result['kappa'],
            'training_time': result['training_time'],
        }

    import json
    with open(os.path.join(save_dir, 'all_results.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("RF CHANGE DETECTION TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Best model: {best_name}")
    print(f"  Saved to: {save_dir}")


if __name__ == '__main__':
    main()
