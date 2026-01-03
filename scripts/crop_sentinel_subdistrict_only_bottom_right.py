#!/usr/bin/env python3
"""
Crop Sentinel-2: Sub-Districts + Bottom-Right Corner Only
==========================================================

Adds only SungaiGelam to fill bottom-right corner!
- Current 11 sub-districts
+ SungaiGelam (bottom-right corner)

Author: Claude Sonnet 4.5
Date: 2026-01-03
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import rasterio
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import mapping
from shapely.ops import unary_union
import numpy as np
from datetime import datetime

print("="*80)
print("CROP SENTINEL-2: SUB-DISTRICTS + BOTTOM-RIGHT CORNER ONLY")
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

# Load all Jambi sub-districts
all_jambi = gpd.read_file('data/gadm_indonesia_subdistricts.geojson')
all_jambi = all_jambi[all_jambi['NAME_1'] == 'Jambi']

# Only add SungaiGelam for bottom-right corner
CORNER_FILLER = 'SungaiGelam'

print(f"\n📍 Adding bottom-right corner filler:")
corner_subdistrict = None
sd = all_jambi[all_jambi['NAME_3'] == CORNER_FILLER]
if len(sd) > 0:
    corner_subdistrict = sd.iloc[0]
    print(f"  ✓ {CORNER_FILLER:30s} ({sd.iloc[0]['NAME_2']})")
else:
    print(f"  ❌ {CORNER_FILLER} not found!")

# Combine all geometries
print(f"\n" + "-"*80)
print("COMBINING BOUNDARIES")
print("-"*80)

all_geoms = list(current.geometry)
if corner_subdistrict is not None:
    all_geoms.append(corner_subdistrict.geometry)

combined_geom = unary_union(all_geoms)
combined_area = combined_geom.area * (111**2)
bounds = combined_geom.bounds

total_count = len(current) + (1 if corner_subdistrict is not None else 0)
print(f"\n✓ Total sub-districts: {total_count}")
print(f"✓ Combined area: ~{combined_area:.0f} km²")
print(f"✓ Bounds:")
print(f"  Lon: {bounds[0]:.4f} to {bounds[2]:.4f}")
print(f"  Lat: {bounds[1]:.4f} to {bounds[3]:.4f}")

# ============================================================================
# Crop Sentinel-2
# ============================================================================

print("\n" + "-"*80)
print("CROPPING SENTINEL-2")
print("-"*80)

INPUT_TILE = 'data/sentinel/S2_jambi_2024_20m_AllBands-0000000000-0000010496.tif'
OUTPUT_FILE = 'data/sentinel/S2_jambi_subdistrict_bottom_right_only_20m.tif'
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

# Save boundary
boundary_output = 'data/jambi_subdistrict_bottom_right_only_boundary.geojson'

# Create GeoDataFrame with all sub-districts
all_sd_data = []
for idx, row in current.iterrows():
    all_sd_data.append(row)
if corner_subdistrict is not None:
    all_sd_data.append({
        'name': corner_subdistrict['NAME_3'],
        'district': corner_subdistrict['NAME_2'],
        'geometry': corner_subdistrict.geometry
    })

gdf_combined = gpd.GeoDataFrame(all_sd_data, crs='EPSG:4326')
gdf_combined.to_file(boundary_output, driver='GeoJSON')
print(f"✓ Saved boundary: {boundary_output}")

# ============================================================================
# Summary
# ============================================================================

print(f"\n{'='*80}")
print("COMPLETE!")
print("="*80)
print(f"✓ {len(current)} original sub-districts")
print(f"✓ +1 corner filler: SungaiGelam (bottom-right only)")
print(f"✓ Total: {total_count} sub-districts")
print(f"✓ Area: ~{combined_area:.0f} km²")
print(f"✓ Natural administrative boundaries!")
print(f"✓ Top-left: Original boundary (not filled)")
print(f"✓ Bottom-right: Filled with SungaiGelam!")
print(f"✓ Output: {OUTPUT_FILE}")
print(f"✓ Boundary: {boundary_output}")
print(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
