"""
Data Loader Module
==================

Handles loading of KLHK reference data and Sentinel-2 imagery.
"""

import geopandas as gpd
import rasterio
from rasterio.merge import merge
import numpy as np


# KLHK code to simplified class mapping
KLHK_TO_SIMPLIFIED = {
    # Forest classes -> 1 (Trees)
    2001: 1, 2002: 1, 2003: 1, 2004: 1, 2005: 1, 2006: 1, 2007: 1,  20071: 1,
    # Agriculture -> 4 (Crops)
    2010: 4, 20051: 4, 20091: 4, 20092: 4, 20093: 4,
    # Built-up -> 6 (Built area)
    2012: 6, 20121: 6, 20122: 6,
    # Bare/Open -> 7 (Bare ground)
    2014: 7, 20141: 7,
    # Water -> 0
    5001: 0, 20094: 0,
    # Shrub -> 5
    2500: 5, 20041: 5,
    # Grass/Savanna -> 2
    3000: 2,
    # Cloud -> ignore
    50011: -1,
}

CLASS_NAMES = {
    0: 'Water',
    1: 'Trees/Forest',
    2: 'Grass/Savanna',
    4: 'Crops/Agriculture',
    5: 'Shrub/Scrub',
    6: 'Built Area',
    7: 'Bare Ground',
}


def load_klhk_data(geojson_path, verbose=True):
    """
    Load KLHK land cover reference data.

    Args:
        geojson_path: Path to KLHK GeoJSON file
        verbose: Print loading information

    Returns:
        GeoDataFrame with land cover polygons and simplified classes
    """
    if verbose:
        print(f"Loading KLHK data from {geojson_path}...")

    gdf = gpd.read_file(geojson_path)

    if verbose:
        print(f"  Total polygons: {len(gdf):,}")
        print(f"  CRS: {gdf.crs}")

    # Find land cover code column
    code_col = None
    for col in ['ID Penutupan Lahan Tahun 2024', 'PL2024_ID', 'PL2024', 'KELAS']:
        if col in gdf.columns:
            code_col = col
            break

    if code_col is None:
        raise ValueError(f"Land cover code column not found. Available: {list(gdf.columns)}")

    # Map to simplified classes
    gdf['klhk_code'] = gdf[code_col].astype(int)
    gdf['class_simplified'] = gdf['klhk_code'].map(KLHK_TO_SIMPLIFIED)

    # Remove cloud/unknown classes
    gdf = gdf[gdf['class_simplified'] >= 0]

    if verbose:
        print(f"\n  Class distribution:")
        for cls in sorted(gdf['class_simplified'].unique()):
            count = (gdf['class_simplified'] == cls).sum()
            name = CLASS_NAMES.get(cls, 'Unknown')
            print(f"    {cls}: {name} - {count:,} polygons")

    return gdf


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

    # Open all tiles
    src_files = [rasterio.open(path) for path in tile_paths]

    # Mosaic tiles
    mosaic, mosaic_transform = merge(src_files)

    # Get profile from first tile
    profile = src_files[0].profile.copy()
    profile.update({
        'height': mosaic.shape[1],
        'width': mosaic.shape[2],
        'transform': mosaic_transform,
        'count': mosaic.shape[0]
    })

    # Close files
    for src in src_files:
        src.close()

    if verbose:
        print(f"  Mosaic shape: {mosaic.shape}")
        print(f"  Bands: {mosaic.shape[0]}")
        print(f"  CRS: {profile['crs']}")

    return mosaic, profile


def get_sentinel2_band_names():
    """Get standard Sentinel-2 band names."""
    return [
        'B2_Blue', 'B3_Green', 'B4_Red',
        'B5_RedEdge1', 'B6_RedEdge2', 'B7_RedEdge3',
        'B8_NIR', 'B8A_RedEdge4', 'B11_SWIR1', 'B12_SWIR2'
    ]
