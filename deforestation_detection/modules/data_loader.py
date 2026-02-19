"""
Data Loader Module
==================

Handles loading of multi-temporal Sentinel-2 imagery, Hansen GFC data,
and ForestNet driver labels for deforestation detection.
"""

import os
import glob
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling


# Hansen GFC lossyear to calendar year mapping
LOSSYEAR_TO_YEAR = {yr: 2000 + yr for yr in range(1, 25)}

# Deforestation driver classes (ForestNet)
DRIVER_CLASSES = {
    0: 'Oil Palm Plantation',
    1: 'Timber Plantation',
    2: 'Smallholder Agriculture',
    3: 'Grassland/Shrub',
}

# Change detection classes
CHANGE_CLASSES = {
    0: 'No Change',
    1: 'Deforestation',
}

# KLHK forest classes (reuse from parent project)
FOREST_KLHK_CODES = [2001, 2002, 2003, 2004, 2005, 2006, 2007, 20071]


def load_sentinel2_tiles(tile_paths, verbose=True):
    """
    Load and mosaic multiple Sentinel-2 GeoTIFF tiles.

    Args:
        tile_paths: List of paths to Sentinel-2 tiles
        verbose: Print loading information

    Returns:
        tuple: (mosaicked data array, raster profile)
    """
    if verbose:
        print(f"Loading {len(tile_paths)} Sentinel-2 tiles...")

    src_files = [rasterio.open(path) for path in tile_paths]
    mosaic, mosaic_transform = merge(src_files)

    profile = src_files[0].profile.copy()
    profile.update({
        'height': mosaic.shape[1],
        'width': mosaic.shape[2],
        'transform': mosaic_transform,
        'count': mosaic.shape[0]
    })

    for src in src_files:
        src.close()

    if verbose:
        print(f"  Mosaic shape: {mosaic.shape}")
        print(f"  CRS: {profile['crs']}")

    return mosaic, profile


def load_multitemporal_sentinel2(year_paths_dict, verbose=True):
    """
    Load multi-temporal Sentinel-2 composites for multiple years.

    Args:
        year_paths_dict: Dict mapping year (int) to list of tile paths
            e.g., {2018: ['path/to/2018_tile1.tif', ...], 2019: [...], ...}
        verbose: Print loading information

    Returns:
        dict: year -> (data_array, profile)
    """
    if verbose:
        print("=" * 60)
        print("LOADING MULTI-TEMPORAL SENTINEL-2 DATA")
        print("=" * 60)

    yearly_data = {}

    for year in sorted(year_paths_dict.keys()):
        paths = year_paths_dict[year]
        if verbose:
            print(f"\n--- Year {year} ---")

        if len(paths) == 1:
            with rasterio.open(paths[0]) as src:
                data = src.read()
                profile = src.profile.copy()
            if verbose:
                print(f"  Single tile shape: {data.shape}")
        else:
            data, profile = load_sentinel2_tiles(paths, verbose=verbose)

        yearly_data[year] = (data, profile)

    if verbose:
        print(f"\nLoaded {len(yearly_data)} annual composites")

    return yearly_data


def find_sentinel2_tiles(data_dir, year, pattern='S2_jambi_{year}_20m_AllBands*.tif'):
    """
    Find Sentinel-2 tile files for a given year.

    Args:
        data_dir: Base data directory (e.g., 'data/sentinel/')
        year: Year to find tiles for
        pattern: Glob pattern with {year} placeholder

    Returns:
        list: Sorted list of tile paths
    """
    year_dir = os.path.join(data_dir, str(year))
    search_pattern = os.path.join(year_dir, pattern.format(year=year))
    tiles = sorted(glob.glob(search_pattern))

    if not tiles:
        # Try without year subdirectory
        search_pattern = os.path.join(data_dir, pattern.format(year=year))
        tiles = sorted(glob.glob(search_pattern))

    return tiles


def load_hansen_gfc(treecover_path, lossyear_path, gain_path=None, verbose=True):
    """
    Load Hansen Global Forest Change data.

    Args:
        treecover_path: Path to treecover2000 GeoTIFF
        lossyear_path: Path to lossyear GeoTIFF
        gain_path: Optional path to gain GeoTIFF
        verbose: Print loading information

    Returns:
        dict with keys: 'treecover2000', 'lossyear', 'gain' (optional), 'profile'
    """
    if verbose:
        print("Loading Hansen GFC data...")

    result = {}

    with rasterio.open(treecover_path) as src:
        result['treecover2000'] = src.read(1)
        result['profile'] = src.profile.copy()
        if verbose:
            print(f"  Tree cover 2000 shape: {result['treecover2000'].shape}")
            print(f"  CRS: {src.crs}")

    with rasterio.open(lossyear_path) as src:
        result['lossyear'] = src.read(1)
        if verbose:
            print(f"  Loss year shape: {result['lossyear'].shape}")
            unique_years = np.unique(result['lossyear'][result['lossyear'] > 0])
            print(f"  Loss years present: {[2000 + y for y in unique_years]}")

    if gain_path and os.path.exists(gain_path):
        with rasterio.open(gain_path) as src:
            result['gain'] = src.read(1)
            if verbose:
                print(f"  Forest gain pixels: {np.sum(result['gain'] > 0):,}")

    return result


def resample_hansen_to_sentinel(hansen_data, hansen_profile, sentinel_profile, verbose=True):
    """
    Resample Hansen GFC data (30m) to match Sentinel-2 grid (20m).

    Args:
        hansen_data: 2D numpy array of Hansen data
        hansen_profile: Rasterio profile for Hansen data
        sentinel_profile: Rasterio profile for Sentinel-2 data (target grid)
        verbose: Print progress

    Returns:
        numpy array resampled to Sentinel-2 grid
    """
    if verbose:
        print("Resampling Hansen data to Sentinel-2 grid...")
        print(f"  From: {hansen_data.shape} (30m)")
        print(f"  To: ({sentinel_profile['height']}, {sentinel_profile['width']}) (20m)")

    dst_shape = (sentinel_profile['height'], sentinel_profile['width'])
    dst_data = np.zeros(dst_shape, dtype=hansen_data.dtype)

    reproject(
        source=hansen_data,
        destination=dst_data,
        src_transform=hansen_profile['transform'],
        src_crs=hansen_profile['crs'],
        dst_transform=sentinel_profile['transform'],
        dst_crs=sentinel_profile['crs'],
        resampling=Resampling.nearest
    )

    if verbose:
        print(f"  Resampled shape: {dst_data.shape}")

    return dst_data


def load_forestnet_labels(forestnet_dir, verbose=True):
    """
    Load ForestNet deforestation driver labels.

    Args:
        forestnet_dir: Path to ForestNet dataset directory
        verbose: Print loading information

    Returns:
        dict with driver labels and metadata
    """
    import json

    if verbose:
        print("Loading ForestNet driver labels...")

    labels_file = os.path.join(forestnet_dir, 'forestnet_labels.json')
    if not os.path.exists(labels_file):
        labels_file = os.path.join(forestnet_dir, 'labels.csv')

    result = {'samples': [], 'drivers': DRIVER_CLASSES}

    if labels_file.endswith('.json') and os.path.exists(labels_file):
        with open(labels_file, 'r') as f:
            result['samples'] = json.load(f)
    elif labels_file.endswith('.csv') and os.path.exists(labels_file):
        import pandas as pd
        df = pd.read_csv(labels_file)
        result['samples'] = df.to_dict('records')
    else:
        if verbose:
            print(f"  WARNING: ForestNet labels not found at {forestnet_dir}")
        return result

    if verbose:
        print(f"  Loaded {len(result['samples'])} labeled samples")

    return result


def get_sentinel2_band_names():
    """Get standard Sentinel-2 band names."""
    return [
        'B2_Blue', 'B3_Green', 'B4_Red',
        'B5_RedEdge1', 'B6_RedEdge2', 'B7_RedEdge3',
        'B8_NIR', 'B8A_RedEdge4', 'B11_SWIR1', 'B12_SWIR2'
    ]


def get_study_years():
    """Get the study period years."""
    return list(range(2018, 2025))


def get_consecutive_year_pairs():
    """Get consecutive year pairs for change detection."""
    years = get_study_years()
    return [(years[i], years[i + 1]) for i in range(len(years) - 1)]
