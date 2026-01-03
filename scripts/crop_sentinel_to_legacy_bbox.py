#!/usr/bin/env python3
"""
Crop Existing Sentinel-2 Data to Legacy Bounding Box
=====================================================

Instead of downloading from GEE again, this script crops the existing
Sentinel-2 data (20m, 10 bands) to the legacy bounding box area.

Author: Claude Sonnet 4.5
Date: 2026-01-03
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import box, mapping
import numpy as np
from datetime import datetime

print("="*80)
print("CROP SENTINEL-2 TO LEGACY BOUNDING BOX")
print("="*80)
print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# Configuration
# ============================================================================

# Legacy bounding box (from GEE script)
LEGACY_BBOX = {
    'min_lon': 103.4486,
    'min_lat': -1.8337,
    'max_lon': 103.7566,
    'max_lat': -1.4089
}

# Input/output paths
INPUT_TILE = 'data/sentinel/S2_jambi_2024_20m_AllBands-0000000000-0000010496.tif'
OUTPUT_FILE = 'data/sentinel/S2_jambi_city_legacy_bbox_20m.tif'

# Create output directory if needed
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# ============================================================================
# Crop to Bounding Box
# ============================================================================

print(f"\nInput file: {INPUT_TILE}")
print(f"Output file: {OUTPUT_FILE}")
print(f"\nLegacy BBox:")
print(f"  Lon: {LEGACY_BBOX['min_lon']} to {LEGACY_BBOX['max_lon']}")
print(f"  Lat: {LEGACY_BBOX['min_lat']} to {LEGACY_BBOX['max_lat']}")

# Create bbox geometry
bbox_geom = box(
    LEGACY_BBOX['min_lon'],
    LEGACY_BBOX['min_lat'],
    LEGACY_BBOX['max_lon'],
    LEGACY_BBOX['max_lat']
)

print(f"\nProcessing...")

with rasterio.open(INPUT_TILE) as src:
    # Print original info
    print(f"\nOriginal tile:")
    print(f"  Shape: {src.shape[0]} × {src.shape[1]} pixels")
    print(f"  Bands: {src.count}")
    print(f"  CRS: {src.crs}")
    print(f"  Resolution: ~{src.res[0]*111:.1f} m")

    # Crop to bbox
    bbox_geom_list = [mapping(bbox_geom)]
    out_image, out_transform = mask(src, bbox_geom_list, crop=True, all_touched=True)

    # Update metadata
    out_meta = src.meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform,
        "compress": "lzw",  # Compress to save space
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256
    })

    # Write output
    print(f"\nCropped result:")
    print(f"  Shape: {out_image.shape[1]} × {out_image.shape[2]} pixels")
    print(f"  Bands: {out_image.shape[0]}")

    # Calculate size
    total_pixels = out_image.shape[0] * out_image.shape[1] * out_image.shape[2]
    size_mb = total_pixels * 4 / (1024**2)

    print(f"  Total pixels: {total_pixels:,}")
    print(f"  Estimated size: {size_mb:.1f} MB")

    # Calculate coverage
    pixel_size_deg = src.res[0]
    pixel_size_km = pixel_size_deg * 111
    width_km = out_image.shape[2] * pixel_size_km
    height_km = out_image.shape[1] * pixel_size_km

    print(f"\n  Coverage:")
    print(f"    Width: {width_km:.1f} km")
    print(f"    Height: {height_km:.1f} km")
    print(f"    Area: {width_km * height_km:.1f} km²")

    print(f"\nWriting to file...")
    with rasterio.open(OUTPUT_FILE, "w", **out_meta) as dest:
        dest.write(out_image)

    print(f"✓ Done!")

# ============================================================================
# Verification
# ============================================================================

print(f"\n{'='*80}")
print("VERIFICATION")
print("="*80)

with rasterio.open(OUTPUT_FILE) as src:
    print(f"Output file created successfully:")
    print(f"  Path: {OUTPUT_FILE}")
    print(f"  Shape: {src.shape[0]} × {src.shape[1]} pixels")
    print(f"  Bands: {src.count}")
    print(f"  Bounds: {src.bounds}")
    print(f"  CRS: {src.crs}")

    # File size
    file_size_mb = os.path.getsize(OUTPUT_FILE) / (1024**2)
    print(f"  File size: {file_size_mb:.1f} MB")

    # Check for NaN/Inf
    sample = src.read(1, window=((0, 100), (0, 100)))
    has_nan = np.isnan(sample).any()
    has_inf = np.isinf(sample).any()

    print(f"\n  Data quality:")
    print(f"    Has NaN: {'⚠️ YES' if has_nan else '✓ NO'}")
    print(f"    Has Inf: {'⚠️ YES' if has_inf else '✓ NO'}")

    # Sample values
    print(f"    Sample values (band 1): min={sample.min():.4f}, max={sample.max():.4f}")

print(f"\n{'='*80}")
print("COMPLETE!")
print("="*80)
print(f"✓ Cropped Sentinel-2 data to legacy bbox")
print(f"✓ Output: {OUTPUT_FILE}")
print(f"✓ Ready to use for training!")
print(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
