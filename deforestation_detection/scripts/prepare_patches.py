"""
Prepare Training Patches for Siamese Network

Extracts bi-temporal patch pairs from consecutive-year Sentinel-2 composites
with corresponding Hansen-derived change labels.

Usage:
    python scripts/prepare_patches.py

Input:
    data/sentinel/{year}/S2_jambi_{year}_20m_AllBands*.tif
    data/change_labels/annual/change_{year}.tif

Output:
    data/patches/patches_{year1}_{year2}.npz (per consecutive year pair)
    data/patches/patches_2018_2024.npz (cumulative)
"""

import os
import sys
import numpy as np
import rasterio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.data_loader import (
    load_sentinel2_tiles, find_sentinel2_tiles, get_consecutive_year_pairs
)
from modules.feature_engineering import calculate_spectral_indices, combine_bands_and_indices
from modules.preprocessor import (
    extract_bitemporal_patches, create_bitemporal_change_label
)


# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
SENTINEL_DIR = os.path.join(DATA_DIR, 'sentinel')
HANSEN_DIR = os.path.join(DATA_DIR, 'hansen')
LABELS_DIR = os.path.join(DATA_DIR, 'change_labels', 'annual')
PATCHES_DIR = os.path.join(DATA_DIR, 'patches')

PATCH_SIZE = 32
STRIDE = 16
BALANCE_RATIO = 3.0
MAX_PATCHES_PER_PAIR = 50000
RANDOM_STATE = 42


def load_year_features(year, verbose=True):
    """
    Load Sentinel-2 data for a year and compute full feature stack.

    Args:
        year: Integer year
        verbose: Print info

    Returns:
        (23, H, W) feature stack or None if data not found
    """
    tiles = find_sentinel2_tiles(SENTINEL_DIR, year)

    if not tiles:
        if verbose:
            print(f"  WARNING: No Sentinel-2 tiles found for {year}")
        return None

    bands, profile = load_sentinel2_tiles(tiles, verbose=verbose)
    indices = calculate_spectral_indices(bands, verbose=verbose)
    features = combine_bands_and_indices(bands, indices)

    if verbose:
        print(f"  Feature stack shape: {features.shape}")

    return features


def load_change_label(year):
    """Load annual change label for a year."""
    label_path = os.path.join(LABELS_DIR, f'change_{year}.tif')

    if not os.path.exists(label_path):
        return None

    with rasterio.open(label_path) as src:
        return src.read(1)


def main():
    """Extract training patches for all consecutive year pairs."""
    print("=" * 60)
    print("PREPARE TRAINING PATCHES")
    print("=" * 60)
    print(f"  Patch size: {PATCH_SIZE}x{PATCH_SIZE}")
    print(f"  Stride: {STRIDE}")
    print(f"  Balance ratio: {BALANCE_RATIO}:1 (no-change:change)")
    print(f"  Max patches per pair: {MAX_PATCHES_PER_PAIR:,}")

    os.makedirs(PATCHES_DIR, exist_ok=True)

    # Process consecutive year pairs
    year_pairs = get_consecutive_year_pairs()
    print(f"\n  Year pairs to process: {len(year_pairs)}")

    # Cache loaded features to avoid reloading
    features_cache = {}
    total_patches = 0

    for year1, year2 in year_pairs:
        print(f"\n{'='*50}")
        print(f"PROCESSING {year1} -> {year2}")
        print(f"{'='*50}")

        # Load features (with caching)
        if year1 not in features_cache:
            print(f"\nLoading {year1} features...")
            features_cache[year1] = load_year_features(year1)

        if year2 not in features_cache:
            print(f"\nLoading {year2} features...")
            features_cache[year2] = load_year_features(year2)

        t1_features = features_cache[year1]
        t2_features = features_cache[year2]

        if t1_features is None or t2_features is None:
            print(f"  SKIPPED: Missing data for {year1}-{year2}")
            continue

        # Load change label
        label = load_change_label(year2)
        if label is None:
            # Try loading from Hansen directly
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
                print(f"  SKIPPED: No change labels for {year2}")
                continue

        # Ensure shapes match
        min_h = min(t1_features.shape[1], t2_features.shape[1], label.shape[0])
        min_w = min(t1_features.shape[2], t2_features.shape[2], label.shape[1])
        t1_features_crop = t1_features[:, :min_h, :min_w]
        t2_features_crop = t2_features[:, :min_h, :min_w]
        label_crop = label[:min_h, :min_w]

        # Extract patches
        patches = extract_bitemporal_patches(
            t1_features_crop, t2_features_crop, label_crop,
            patch_size=PATCH_SIZE, stride=STRIDE,
            max_patches=MAX_PATCHES_PER_PAIR,
            balance_ratio=BALANCE_RATIO,
            random_state=RANDOM_STATE
        )

        if len(patches['labels']) == 0:
            print(f"  SKIPPED: No valid patches for {year1}-{year2}")
            continue

        # Save patches
        output_path = os.path.join(PATCHES_DIR, f'patches_{year1}_{year2}.npz')
        np.savez_compressed(
            output_path,
            patches_t1=patches['patches_t1'],
            patches_t2=patches['patches_t2'],
            labels=patches['labels'],
            year1=year1,
            year2=year2,
        )

        n_patches = len(patches['labels'])
        total_patches += n_patches
        print(f"  Saved: {output_path} ({n_patches:,} patches)")

        # Free memory for oldest year
        oldest = min(features_cache.keys())
        if oldest < year1:
            del features_cache[oldest]

    # Summary
    print("\n" + "=" * 60)
    print("PATCH EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"  Total patches extracted: {total_patches:,}")
    print(f"  Output directory: {PATCHES_DIR}")
    print(f"  Patch shape: ({PATCH_SIZE}, {PATCH_SIZE}) at 20m = "
          f"{PATCH_SIZE * 20}m footprint")


if __name__ == '__main__':
    main()
