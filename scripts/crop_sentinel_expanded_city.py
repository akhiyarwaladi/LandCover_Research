#!/usr/bin/env python3
"""
Crop Existing Sentinel-2 Data to Expanded Jambi City Area
==========================================================

Creates circular buffer around city center (expansion factor 1.8x)
instead of rectangular legacy bbox.

Author: Claude Sonnet 4.5
Date: 2026-01-03
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import rasterio
from rasterio.mask import mask
from shapely.geometry import Point, mapping
import numpy as np
from datetime import datetime

print("="*80)
print("CROP SENTINEL-2 TO EXPANDED CITY AREA")
print("="*80)
print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# Configuration
# ============================================================================

# Jambi City center
CITY_CENTER = {
    'lon': 103.6167,
    'lat': -1.6000
}

# Expansion parameters (from GEE script)
LEGACY_SIZE_KM = {
    'width': 34,   # ~34 km
    'height': 47   # ~47 km
}

EXPANSION_FACTOR = 1.8
target_size_km = (LEGACY_SIZE_KM['width'] * LEGACY_SIZE_KM['height']) ** 0.5  # geometric mean ≈ 40 km
buffer_distance_km = (target_size_km / 2) * EXPANSION_FACTOR
buffer_distance_deg = buffer_distance_km / 111.0  # Convert km to degrees (approximate)

print(f"\nExpansion parameters:")
print(f"  City center: ({CITY_CENTER['lon']}, {CITY_CENTER['lat']})")
print(f"  Target size: {target_size_km:.1f} km (geometric mean)")
print(f"  Expansion factor: {EXPANSION_FACTOR}")
print(f"  Buffer radius: {buffer_distance_km:.1f} km ({buffer_distance_deg:.4f}°)")

# Input/output paths - need to mosaic all tiles first
INPUT_TILES = [
    'data/sentinel/S2_jambi_2024_20m_AllBands-0000000000-0000000000.tif',
    'data/sentinel/S2_jambi_2024_20m_AllBands-0000000000-0000010496.tif',
    'data/sentinel/S2_jambi_2024_20m_AllBands-0000010496-0000000000.tif',
    'data/sentinel/S2_jambi_2024_20m_AllBands-0000010496-0000010496.tif'
]
OUTPUT_FILE = 'data/sentinel/S2_jambi_city_expanded_20m.tif'

# Create output directory if needed
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# ============================================================================
# Step 1: Create Circular Buffer Geometry
# ============================================================================

print(f"\nCreating circular buffer...")

# Create point geometry for city center
city_point = Point(CITY_CENTER['lon'], CITY_CENTER['lat'])

# Create circular buffer (in degrees)
circle_buffer = city_point.buffer(buffer_distance_deg)

print(f"✓ Circular buffer created")
print(f"  Buffer bounds: {circle_buffer.bounds}")

# ============================================================================
# Step 2: Find Which Tile Contains the Area
# ============================================================================

print(f"\nChecking which tiles intersect with expanded area...")

from shapely.geometry import box

tiles_to_use = []
for tile_path in INPUT_TILES:
    with rasterio.open(tile_path) as src:
        tile_bounds = src.bounds
        tile_box = box(tile_bounds.left, tile_bounds.bottom,
                       tile_bounds.right, tile_bounds.top)

        if tile_box.intersects(circle_buffer):
            tiles_to_use.append(tile_path)
            print(f"  ✓ {os.path.basename(tile_path)}")

print(f"\n{len(tiles_to_use)} tile(s) will be used")

# ============================================================================
# Step 3: Mosaic Tiles if Needed
# ============================================================================

if len(tiles_to_use) == 1:
    print(f"\nUsing single tile...")
    INPUT_TILE = tiles_to_use[0]
else:
    print(f"\nMosaicking {len(tiles_to_use)} tiles...")
    from rasterio.merge import merge

    src_files = [rasterio.open(path) for path in tiles_to_use]
    mosaic, mosaic_transform = merge(src_files)

    # Save temporary mosaic
    temp_mosaic_path = 'data/sentinel/temp_mosaic_expanded.tif'

    profile = src_files[0].profile.copy()
    profile.update({
        'height': mosaic.shape[1],
        'width': mosaic.shape[2],
        'transform': mosaic_transform,
        'count': mosaic.shape[0],
        'compress': 'lzw'
    })

    with rasterio.open(temp_mosaic_path, 'w', **profile) as dst:
        dst.write(mosaic)

    for src in src_files:
        src.close()

    INPUT_TILE = temp_mosaic_path
    print(f"✓ Mosaic created: {mosaic.shape}")

# ============================================================================
# Step 4: Crop to Circular Buffer
# ============================================================================

print(f"\nCropping to circular buffer...")

with rasterio.open(INPUT_TILE) as src:
    # Print original info
    print(f"\nInput tile:")
    print(f"  Shape: {src.shape[0]} × {src.shape[1]} pixels")
    print(f"  Bands: {src.count}")
    print(f"  CRS: {src.crs}")
    print(f"  Resolution: ~{src.res[0]*111:.1f} m")

    # Crop to circular buffer
    circle_geom_list = [mapping(circle_buffer)]
    out_image, out_transform = mask(src, circle_geom_list, crop=True, all_touched=True)

    # Update metadata
    out_meta = src.meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform,
        "compress": "lzw",
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
    print(f"    Circular area: ~{3.14159 * buffer_distance_km**2:.1f} km²")

    print(f"\nWriting to file...")
    with rasterio.open(OUTPUT_FILE, "w", **out_meta) as dest:
        dest.write(out_image)

    print(f"✓ Done!")

# Clean up temporary mosaic if created
if len(tiles_to_use) > 1 and os.path.exists(temp_mosaic_path):
    os.remove(temp_mosaic_path)
    print(f"✓ Cleaned up temporary mosaic")

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
    sample = src.read(1, window=((0, min(100, src.height)), (0, min(100, src.width))))
    has_nan = np.isnan(sample).any()
    has_inf = np.isinf(sample).any()

    print(f"\n  Data quality:")
    print(f"    Has NaN: {'⚠️ YES' if has_nan else '✓ NO'}")
    print(f"    Has Inf: {'⚠️ YES' if has_inf else '✓ NO'}")

    # Sample values
    valid_sample = sample[sample > 0]
    if len(valid_sample) > 0:
        print(f"    Sample values (band 1): min={valid_sample.min():.4f}, max={valid_sample.max():.4f}")

print(f"\n{'='*80}")
print("COMPLETE!")
print("="*80)
print(f"✓ Cropped Sentinel-2 to expanded city area")
print(f"✓ Output: {OUTPUT_FILE}")
print(f"✓ Ready for visualization and training!")
print(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
