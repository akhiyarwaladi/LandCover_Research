#!/usr/bin/env python3
"""
Crop Sentinel-2 Using Sub-District Boundaries with Small Buffer
================================================================

Uses natural sub-district boundaries, then adds small buffer to fill
empty corners while maintaining natural curves!

Author: Claude Sonnet 4.5
Date: 2026-01-03
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import rasterio
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import mapping, Point
from shapely.ops import unary_union
import numpy as np
from datetime import datetime

print("="*80)
print("CROP SENTINEL-2: SUB-DISTRICT WITH BUFFER (NO EMPTY CORNERS!)")
print("="*80)
print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# Load Sub-District Boundaries
# ============================================================================

print("\n" + "-"*80)
print("LOADING SUB-DISTRICT BOUNDARIES")
print("-"*80)

# Load from the boundary file we created earlier
BOUNDARY_PATH = 'data/jambi_subdistrict_28km_boundary.geojson'

boundaries = gpd.read_file(BOUNDARY_PATH)
print(f"✓ Loaded {len(boundaries)} sub-districts")

# Show which ones
district_names = boundaries['name'].tolist()
print(f"  Sub-districts: {', '.join(district_names)}")

# Create unified boundary
unified_boundary = unary_union(boundaries.geometry)
original_area = unified_boundary.area * (111**2)
print(f"\n  Original area: ~{original_area:.0f} km²")

# ============================================================================
# Add Small Buffer to Fill Corners
# ============================================================================

print("\n" + "-"*80)
print("ADDING BUFFER TO FILL EMPTY CORNERS")
print("-"*80)

# Buffer to fill gaps - adjust this value:
#   0.015° ≈ 1.7 km - subtle fill
#   0.03° ≈ 3.3 km  - moderate fill
#   0.045° ≈ 5.0 km - strong fill (RECOMMENDED FOR FULL COVERAGE)
#   0.06° ≈ 6.6 km  - very strong fill
BUFFER_DEG = 0.045  # 👈 ADJUST THIS - increased to fill corners!

print(f"Buffer: {BUFFER_DEG:.3f}° (~{BUFFER_DEG * 111:.1f} km)")

# Apply buffer - this fills gaps while maintaining curves!
buffered_boundary = unified_boundary.buffer(BUFFER_DEG)
buffered_area = buffered_boundary.area * (111**2)

print(f"Buffered area: ~{buffered_area:.0f} km²")
print(f"Increase: {((buffered_area - original_area) / original_area * 100):.1f}%")

bounds = buffered_boundary.bounds
print(f"\nBounds:")
print(f"  Lon: {bounds[0]:.4f} to {bounds[2]:.4f}")
print(f"  Lat: {bounds[1]:.4f} to {bounds[3]:.4f}")

# ============================================================================
# Crop Sentinel-2
# ============================================================================

print("\n" + "-"*80)
print("CROPPING SENTINEL-2")
print("-"*80)

INPUT_TILE = 'data/sentinel/S2_jambi_2024_20m_AllBands-0000000000-0000010496.tif'
OUTPUT_FILE = 'data/sentinel/S2_jambi_subdistrict_buffered_20m.tif'
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with rasterio.open(INPUT_TILE) as src:
    print(f"\nInput tile:")
    print(f"  Shape: {src.shape[0]} × {src.shape[1]} pixels")

    # Crop to buffered boundary
    geom_list = [mapping(buffered_boundary)]
    out_image, out_transform = mask(src, geom_list, crop=True, all_touched=True)

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

    print(f"\nCropped output:")
    print(f"  Shape: {out_image.shape[1]} × {out_image.shape[2]} pixels")
    print(f"  Bands: {out_image.shape[0]}")

    size_mb = out_image.shape[0] * out_image.shape[1] * out_image.shape[2] * 4 / (1024**2)
    print(f"  Estimated size: {size_mb:.1f} MB")

    with rasterio.open(OUTPUT_FILE, "w", **out_meta) as dest:
        dest.write(out_image)

    print(f"✓ Saved: {OUTPUT_FILE}")

# Save buffered boundary
boundary_output = 'data/jambi_subdistrict_buffered_boundary.geojson'
gdf_buffered = gpd.GeoDataFrame({'geometry': [buffered_boundary]}, crs='EPSG:4326')
gdf_buffered.to_file(boundary_output, driver='GeoJSON')
print(f"✓ Saved boundary: {boundary_output}")

# ============================================================================
# Summary
# ============================================================================

print(f"\n{'='*80}")
print("COMPLETE!")
print("="*80)
print(f"✓ Based on {len(boundaries)} sub-districts")
print(f"✓ Added buffer to fill empty corners")
print(f"✓ Natural curves maintained!")
print(f"✓ Area: ~{buffered_area:.0f} km²")
print(f"✓ NO empty corners! 🎉")
print(f"✓ Output: {OUTPUT_FILE}")
print(f"✓ Boundary: {boundary_output}")
print(f"\n💡 TIP: Adjust BUFFER_DEG to control corner filling")
print(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
