#!/usr/bin/env python3
"""
Crop Sentinel-2 to Legacy Bbox with Curved Edges
=================================================

Takes the legacy bounding box and adds curves to the edges by:
1. Converting bbox to polygon
2. Adding a small buffer (creates rounded corners)
3. Cropping Sentinel-2 to this curved area

This gives: city-focused area + beautiful curves!

Author: Claude Sonnet 4.5
Date: 2026-01-03
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import rasterio
from rasterio.mask import mask
from shapely.geometry import box, mapping
import numpy as np
from datetime import datetime

print("="*80)
print("CROP SENTINEL-2: LEGACY BBOX WITH CURVED EDGES")
print("="*80)
print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# Configuration
# ============================================================================

# Legacy bounding box (original rectangular area)
LEGACY_BBOX = {
    'min_lon': 103.4486,
    'min_lat': -1.8337,
    'max_lon': 103.7566,
    'max_lat': -1.4089
}

# Buffer distance to create curves (in degrees)
# Adjust this to control how much expansion:
#   - 0.02° ≈ 2.2 km (small expansion)
#   - 0.03° ≈ 3.3 km (moderate expansion)
#   - 0.04° ≈ 4.4 km (good expansion) ← RECOMMENDED FOR LARGER AREA
#   - 0.05° ≈ 5.5 km (strong expansion)
#   - 0.06° ≈ 6.6 km (very large expansion)
BUFFER_DISTANCE_DEG = 0.045  # 👈 ADJUST THIS (~5 km expansion)

# Input/output paths
INPUT_TILE = 'data/sentinel/S2_jambi_2024_20m_AllBands-0000000000-0000010496.tif'
OUTPUT_FILE = 'data/sentinel/S2_jambi_city_legacy_curved_20m.tif'

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

print(f"\nConfiguration:")
print(f"  Legacy bbox: {LEGACY_BBOX['min_lon']:.4f}, {LEGACY_BBOX['min_lat']:.4f} to")
print(f"              {LEGACY_BBOX['max_lon']:.4f}, {LEGACY_BBOX['max_lat']:.4f}")
print(f"  Buffer: {BUFFER_DISTANCE_DEG:.4f}° (~{BUFFER_DISTANCE_DEG * 111:.1f} km)")

# ============================================================================
# Create Curved Geometry
# ============================================================================

print(f"\n" + "-"*80)
print("CREATING CURVED BOUNDARY")
print("-"*80)

# Create original rectangular bbox
original_bbox = box(
    LEGACY_BBOX['min_lon'],
    LEGACY_BBOX['min_lat'],
    LEGACY_BBOX['max_lon'],
    LEGACY_BBOX['max_lat']
)

print(f"Original bbox area: {original_bbox.area * (111**2):.0f} km²")

# Add buffer to create curves (positive buffer = expand)
# The buffer creates smooth curves instead of sharp corners!
curved_bbox = original_bbox.buffer(BUFFER_DISTANCE_DEG)

print(f"Curved bbox area: {curved_bbox.area * (111**2):.0f} km²")
print(f"Area increase: {((curved_bbox.area - original_bbox.area) / original_bbox.area * 100):.1f}%")

# Get bounds
curved_bounds = curved_bbox.bounds
print(f"\nCurved bounds:")
print(f"  Lon: {curved_bounds[0]:.4f} to {curved_bounds[2]:.4f}")
print(f"  Lat: {curved_bounds[1]:.4f} to {curved_bounds[3]:.4f}")

# ============================================================================
# Crop Sentinel-2 to Curved Boundary
# ============================================================================

print(f"\n" + "-"*80)
print("CROPPING SENTINEL-2")
print("-"*80)

print(f"\nInput: {INPUT_TILE}")

with rasterio.open(INPUT_TILE) as src:
    print(f"\nOriginal tile:")
    print(f"  Shape: {src.shape[0]} × {src.shape[1]} pixels")
    print(f"  Bands: {src.count}")
    print(f"  CRS: {src.crs}")

    # Crop to curved boundary
    curved_geom_list = [mapping(curved_bbox)]
    out_image, out_transform = mask(src, curved_geom_list, crop=True, all_touched=True)

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
    print(f"    Curved area: ~{curved_bbox.area * (111**2):.0f} km²")

    print(f"\nWriting to file...")
    with rasterio.open(OUTPUT_FILE, "w", **out_meta) as dest:
        dest.write(out_image)

    print(f"✓ Done!")

# Save curved boundary as GeoJSON for reference
import geopandas as gpd
from shapely.geometry import Polygon

boundary_output = 'data/jambi_legacy_curved_boundary.geojson'
gdf = gpd.GeoDataFrame({'geometry': [curved_bbox]}, crs='EPSG:4326')
gdf.to_file(boundary_output, driver='GeoJSON')
print(f"✓ Saved curved boundary: {boundary_output}")

# ============================================================================
# Verification
# ============================================================================

print(f"\n{'='*80}")
print("VERIFICATION")
print("="*80)

with rasterio.open(OUTPUT_FILE) as src:
    print(f"✓ Output file created successfully:")
    print(f"  Path: {OUTPUT_FILE}")
    print(f"  Shape: {src.shape[0]} × {src.shape[1]} pixels")
    print(f"  Bands: {src.count}")
    print(f"  Bounds: {src.bounds}")

    file_size_mb = os.path.getsize(OUTPUT_FILE) / (1024**2)
    print(f"  File size: {file_size_mb:.1f} MB")

    # Check data quality
    sample = src.read(1, window=((0, 100), (0, 100)))
    has_nan = np.isnan(sample).any()
    has_inf = np.isinf(sample).any()

    print(f"\n  Data quality:")
    print(f"    Has NaN: {'⚠️ YES' if has_nan else '✓ NO'}")
    print(f"    Has Inf: {'⚠️ YES' if has_inf else '✓ NO'}")

print(f"\n{'='*80}")
print("COMPLETE!")
print("="*80)
print(f"✓ Created legacy bbox with BEAUTIFUL CURVED EDGES!")
print(f"✓ Same city focus as original legacy bbox")
print(f"✓ Slightly expanded with smooth curves")
print(f"✓ Output: {OUTPUT_FILE}")
print(f"✓ Boundary: {boundary_output}")
print(f"\n💡 TIP: Adjust BUFFER_DISTANCE_DEG to control curve strength")
print(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
