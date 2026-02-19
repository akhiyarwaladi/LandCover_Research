"""
Prepare Change Labels from Hansen GFC Data

Converts Hansen GFC lossyear into annual and cumulative change labels
aligned to the Sentinel-2 grid (20m resolution).

Usage:
    python scripts/prepare_change_labels.py

Input:
    data/hansen/Hansen_treecover2000_Jambi.tif
    data/hansen/Hansen_lossyear_Jambi.tif
    data/sentinel/2024/S2_jambi_2024_20m_AllBands*.tif (reference grid)

Output:
    data/change_labels/annual/change_{year}.tif (per year)
    data/change_labels/cumulative/cumulative_{year}.tif (cumulative from 2018)
"""

import os
import sys
import numpy as np
import rasterio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.data_loader import (
    load_hansen_gfc, resample_hansen_to_sentinel,
    load_sentinel2_tiles, find_sentinel2_tiles
)
from modules.preprocessor import (
    create_annual_change_labels, create_cumulative_change_labels
)


# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
HANSEN_DIR = os.path.join(DATA_DIR, 'hansen')
SENTINEL_DIR = os.path.join(DATA_DIR, 'sentinel')
ANNUAL_DIR = os.path.join(DATA_DIR, 'change_labels', 'annual')
CUMULATIVE_DIR = os.path.join(DATA_DIR, 'change_labels', 'cumulative')

TREECOVER_THRESHOLD = 30


def save_label_raster(data, profile, output_path, verbose=True):
    """Save a label array as GeoTIFF."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    out_profile = profile.copy()
    out_profile.update({
        'count': 1,
        'dtype': 'uint8',
        'nodata': 255,
    })

    with rasterio.open(output_path, 'w', **out_profile) as dst:
        dst.write(data.astype(np.uint8), 1)

    if verbose:
        print(f"  Saved: {output_path}")


def main():
    """Prepare annual and cumulative change labels."""
    print("=" * 60)
    print("PREPARE CHANGE LABELS FROM HANSEN GFC")
    print("=" * 60)

    os.makedirs(ANNUAL_DIR, exist_ok=True)
    os.makedirs(CUMULATIVE_DIR, exist_ok=True)

    # Load Hansen data
    treecover_path = os.path.join(HANSEN_DIR, 'Hansen_treecover2000_Jambi.tif')
    lossyear_path = os.path.join(HANSEN_DIR, 'Hansen_lossyear_Jambi.tif')

    if not os.path.exists(treecover_path) or not os.path.exists(lossyear_path):
        print("\nERROR: Hansen GFC data not found!")
        print(f"  Expected: {treecover_path}")
        print(f"  Expected: {lossyear_path}")
        print("\nRun 'python scripts/download_hansen_gfc.py' first,")
        print("then download from Google Drive to data/hansen/")
        return

    hansen = load_hansen_gfc(treecover_path, lossyear_path)

    # Load reference Sentinel-2 grid (use any year available)
    print("\nFinding reference Sentinel-2 grid...")
    sentinel_profile = None

    for year in range(2024, 2017, -1):
        tiles = find_sentinel2_tiles(SENTINEL_DIR, year)
        if tiles:
            _, sentinel_profile = load_sentinel2_tiles(tiles[:1], verbose=False)
            print(f"  Using {year} Sentinel-2 as reference grid")
            break

    # Also try parent project data
    if sentinel_profile is None:
        parent_sentinel = os.path.join(BASE_DIR, '..', 'data', 'sentinel_new_cloudfree')
        if os.path.exists(parent_sentinel):
            import glob
            parent_tiles = sorted(glob.glob(os.path.join(parent_sentinel, '*.tif')))
            if parent_tiles:
                _, sentinel_profile = load_sentinel2_tiles(parent_tiles[:1], verbose=False)
                print(f"  Using parent project Sentinel-2 as reference grid")

    if sentinel_profile is None:
        print("\nWARNING: No Sentinel-2 data found for reference grid!")
        print("Using Hansen native grid (30m). Labels will need resampling later.")
        sentinel_profile = hansen['profile']

    # Resample Hansen to Sentinel-2 grid if needed
    if (sentinel_profile['height'] != hansen['treecover2000'].shape[0] or
            sentinel_profile['width'] != hansen['treecover2000'].shape[1]):
        print("\nResampling Hansen data to Sentinel-2 grid...")
        treecover = resample_hansen_to_sentinel(
            hansen['treecover2000'], hansen['profile'], sentinel_profile
        )
        lossyear = resample_hansen_to_sentinel(
            hansen['lossyear'], hansen['profile'], sentinel_profile
        )
    else:
        treecover = hansen['treecover2000']
        lossyear = hansen['lossyear']

    # Create annual change labels
    print("\n--- Annual Change Labels ---")
    annual_labels = create_annual_change_labels(
        lossyear, year_range=(18, 24),
        treecover2000=treecover, treecover_threshold=TREECOVER_THRESHOLD
    )

    for year, label in annual_labels.items():
        output_path = os.path.join(ANNUAL_DIR, f'change_{year}.tif')
        save_label_raster(label, sentinel_profile, output_path)

    # Create cumulative change labels
    print("\n--- Cumulative Change Labels ---")
    cumulative_labels = create_cumulative_change_labels(
        lossyear, base_year_val=18, end_year_val=24,
        treecover2000=treecover, treecover_threshold=TREECOVER_THRESHOLD
    )

    for year, label in cumulative_labels.items():
        output_path = os.path.join(CUMULATIVE_DIR, f'cumulative_{year}.tif')
        save_label_raster(label, sentinel_profile, output_path)

    # Summary
    print("\n" + "=" * 60)
    print("CHANGE LABEL PREPARATION COMPLETE")
    print("=" * 60)
    total_annual = sum(np.sum(l) for l in annual_labels.values())
    total_area = total_annual * 0.04  # 20m pixel
    print(f"  Total deforestation pixels (2018-2024): {total_annual:,}")
    print(f"  Total deforestation area: {total_area:,.0f} ha")
    print(f"  Annual labels: {ANNUAL_DIR}")
    print(f"  Cumulative labels: {CUMULATIVE_DIR}")


if __name__ == '__main__':
    main()
