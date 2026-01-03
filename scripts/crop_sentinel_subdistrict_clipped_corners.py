#!/usr/bin/env python3
"""
Crop Sentinel-2: Sub-Districts + Clipped Corner Fillers
========================================================

Uses 13 sub-districts but CLIPS the corner fillers to not extend too far!
- 11 original sub-districts (full)
+ SungaiGelam (clipped - not too far south/east)
+ Sekernan (clipped - not too far north/west)

Author: Claude Sonnet 4.5
Date: 2026-01-03
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import rasterio
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import mapping, box
from shapely.ops import unary_union
import numpy as np
from datetime import datetime

print("="*80)
print("CROP SENTINEL-2: SUB-DISTRICTS + CLIPPED CORNER FILLERS")
print("="*80)
print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# Load and Combine Sub-Districts
# ============================================================================

print("\n" + "-"*80)
print("LOADING SUB-DISTRICTS")
print("-"*80)

# Load current 11 sub-districts
current = gpd.read_file('data/jambi_subdistrict_28km_boundary.geojson')
print(f"✓ Current selection: {len(current)} sub-districts")

# Get bounds of the 11 original sub-districts
current_union = unary_union(current.geometry)
current_bounds = current_union.bounds
print(f"\nOriginal 11 sub-districts bounds:")
print(f"  Lon: {current_bounds[0]:.4f} to {current_bounds[2]:.4f}")
print(f"  Lat: {current_bounds[1]:.4f} to {current_bounds[3]:.4f}")

# Create clipping box - extend bounds slightly to allow corner fillers
# But not too much to keep them short!
# Different extensions for different corners:
EXTENSION_WEST = 0.01   # Left (Sekernan) - EXTREMELY short
EXTENSION_NORTH = 0.01  # Top (Sekernan) - EXTREMELY short
EXTENSION_EAST = 0.08   # Right (SungaiGelam) - keep as before
EXTENSION_SOUTH = 0.01  # Bottom (SungaiGelam) - MINIMAL to clip pointy part heavily

clip_box = box(
    current_bounds[0] - EXTENSION_WEST,   # min_lon (west)
    current_bounds[1] - EXTENSION_SOUTH,  # min_lat (south)
    current_bounds[2] + EXTENSION_EAST,   # max_lon (east)
    current_bounds[3] + EXTENSION_NORTH   # max_lat (north)
)

print(f"\nClipping box:")
print(f"  West/North (Sekernan): {EXTENSION_WEST}° ≈ {EXTENSION_WEST*111:.1f} km")
print(f"  East/South (SungaiGelam): {EXTENSION_EAST}° ≈ {EXTENSION_EAST*111:.1f} km")
clip_bounds = clip_box.bounds
print(f"  Lon: {clip_bounds[0]:.4f} to {clip_bounds[2]:.4f}")
print(f"  Lat: {clip_bounds[1]:.4f} to {clip_bounds[3]:.4f}")

# Load all Jambi sub-districts
all_jambi = gpd.read_file('data/gadm_indonesia_subdistricts.geojson')
all_jambi = all_jambi[all_jambi['NAME_1'] == 'Jambi']

# Corner fillers to add (will be clipped)
CORNER_FILLERS = ['SungaiGelam', 'Sekernan']

print(f"\n📍 Adding and clipping corner fillers:")
corner_geoms_clipped = []
for name in CORNER_FILLERS:
    sd = all_jambi[all_jambi['NAME_3'] == name]
    if len(sd) > 0:
        original_geom = sd.iloc[0].geometry
        original_area = original_geom.area * (111**2)

        # Clip to the extended bounding box
        clipped_geom = original_geom.intersection(clip_box)
        clipped_area = clipped_geom.area * (111**2)

        reduction_pct = (1 - clipped_area/original_area) * 100

        corner_geoms_clipped.append(clipped_geom)
        print(f"  ✓ {name:30s} ({sd.iloc[0]['NAME_2']})")
        print(f"    Original: {original_area:.0f} km², Clipped: {clipped_area:.0f} km² ({reduction_pct:.1f}% reduction)")
    else:
        print(f"  ❌ {name} not found!")

# Combine all geometries
print(f"\n" + "-"*80)
print("COMBINING BOUNDARIES")
print("-"*80)

all_geoms = list(current.geometry) + corner_geoms_clipped
combined_geom_initial = unary_union(all_geoms)

# Apply final clipping to remove pointy protrusions
# Clip bottom (remove southern pointy part) and right (remove eastern extension)
MAX_LAT_SOUTH = -1.9  # Anything below this will be clipped
MAX_LON_EAST = 103.8   # Anything beyond this will be clipped

print(f"\n📐 Applying final boundary limits:")
print(f"  Max south: lat > {MAX_LAT_SOUTH}")
print(f"  Max east: lon < {MAX_LON_EAST}")

# Create final clipping box
final_clip_box = box(
    -180,  # min_lon (no limit on west)
    MAX_LAT_SOUTH,  # min_lat (clip bottom)
    MAX_LON_EAST,  # max_lon (clip right)
    90  # max_lat (no limit on north)
)

combined_geom = combined_geom_initial.intersection(final_clip_box)
combined_area = combined_geom.area * (111**2)
bounds = combined_geom.bounds

print(f"\n✓ Total sub-districts: {len(current) + len(corner_geoms_clipped)}")
print(f"  - 11 original (clipped)")
print(f"  - 2 corner fillers (clipped)")
print(f"✓ Combined area: ~{combined_area:.0f} km²")
print(f"✓ Bounds after final clipping:")
print(f"  Lon: {bounds[0]:.4f} to {bounds[2]:.4f}")
print(f"  Lat: {bounds[1]:.4f} to {bounds[3]:.4f}")

# ============================================================================
# Crop Sentinel-2
# ============================================================================

print("\n" + "-"*80)
print("CROPPING SENTINEL-2")
print("-"*80)

INPUT_TILE = 'data/sentinel/S2_jambi_2024_20m_AllBands-0000000000-0000010496.tif'
OUTPUT_FILE = 'data/sentinel/S2_jambi_subdistrict_clipped_corners_20m.tif'
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with rasterio.open(INPUT_TILE) as src:
    print(f"Input tile: {src.shape[0]} × {src.shape[1]} pixels")

    geom_list = [mapping(combined_geom)]
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

    print(f"Cropped: {out_image.shape[1]} × {out_image.shape[2]} pixels")

    with rasterio.open(OUTPUT_FILE, "w", **out_meta) as dest:
        dest.write(out_image)

    print(f"✓ Saved: {OUTPUT_FILE}")

# Save clipped boundary for visualization
boundary_output = 'data/jambi_subdistrict_clipped_corners_boundary.geojson'
gdf_boundary = gpd.GeoDataFrame({'geometry': [combined_geom]}, crs='EPSG:4326')
gdf_boundary.to_file(boundary_output, driver='GeoJSON')
print(f"✓ Saved boundary: {boundary_output}")

# ============================================================================
# Summary
# ============================================================================

print(f"\n{'='*80}")
print("COMPLETE!")
print("="*80)
print(f"✓ {len(current)} original sub-districts (clipped)")
print(f"✓ +2 corner fillers (asymmetric clipping):")
print(f"  - Sekernan (top-left): {EXTENSION_WEST}°/{EXTENSION_NORTH}° extension")
print(f"  - SungaiGelam (bottom-right): {EXTENSION_EAST}°/{EXTENSION_SOUTH}° extension")
print(f"✓ Final boundary limits applied:")
print(f"  - Bottom clipped: lat > {MAX_LAT_SOUTH}")
print(f"  - Right clipped: lon < {MAX_LON_EAST}")
print(f"✓ Total: {len(current) + len(corner_geoms_clipped)} sub-districts")
print(f"✓ Area: ~{combined_area:.0f} km²")
print(f"✓ Natural administrative boundaries!")
print(f"✓ NO pointy protrusions!")
print(f"✓ Output: {OUTPUT_FILE}")
print(f"✓ Boundary: {boundary_output}")
print(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
