#!/usr/bin/env python3
"""
Crop Sentinel-2 Using Sub-District (Kecamatan) Administrative Boundaries
=========================================================================

Uses NATURAL administrative boundaries at kecamatan (sub-district) level
to get city-focused area with irregular/natural curves like real admin maps!

Author: Claude Sonnet 4.5
Date: 2026-01-03
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import rasterio
from rasterio.mask import mask
from rasterio.merge import merge
import geopandas as gpd
from shapely.geometry import mapping, box, Point
from shapely.ops import unary_union
import numpy as np
from datetime import datetime
import requests

print("="*80)
print("CROP SENTINEL-2 USING SUB-DISTRICT BOUNDARIES (KECAMATAN)")
print("="*80)
print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# Step 1: Download Sub-District Level Boundaries (Level 3)
# ============================================================================

print("\n" + "-"*80)
print("DOWNLOADING SUB-DISTRICT BOUNDARIES (KECAMATAN)")
print("-"*80)

# GADM Level 3 = Kecamatan (sub-districts)
gadm_url = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_IDN_3.json"
boundary_file = "data/gadm_indonesia_subdistricts.geojson"

if not os.path.exists(boundary_file):
    print(f"Downloading sub-district boundaries from GADM...")
    print(f"  This is a large file (~20-30 MB), may take 1-2 minutes...")

    response = requests.get(gadm_url, timeout=300)

    if response.status_code == 200:
        with open(boundary_file, 'w') as f:
            f.write(response.text)
        print(f"✓ Downloaded: {len(response.content)/1024/1024:.1f} MB")
    else:
        print(f"❌ Download failed: {response.status_code}")
        sys.exit(1)
else:
    print(f"✓ Using cached file: {boundary_file}")

# Load sub-district boundaries
print(f"\nLoading sub-district boundaries...")
admin = gpd.read_file(boundary_file)
print(f"✓ Loaded {len(admin):,} sub-districts (all Indonesia)")

# Filter to Jambi Province
jambi_subdistricts = admin[admin['NAME_1'] == 'Jambi'].copy()
print(f"✓ Filtered to Jambi Province: {len(jambi_subdistricts)} sub-districts (kecamatan)")

# ============================================================================
# Step 2: Select Sub-Districts Near City Center
# ============================================================================

print(f"\n" + "-"*80)
print("SELECTING SUB-DISTRICTS AROUND JAMBI CITY")
print("-"*80)

# Jambi City center
city_center = Point(103.6167, -1.6000)

# Calculate distance from each sub-district to city center
print(f"\nCalculating distances from city center...")

subdistrict_distances = []
for idx, row in jambi_subdistricts.iterrows():
    centroid = row.geometry.centroid
    distance_deg = city_center.distance(centroid)
    distance_km = distance_deg * 111

    subdistrict_distances.append({
        'name': row['NAME_3'],
        'district': row['NAME_2'],
        'distance_km': distance_km,
        'geometry': row.geometry,
        'idx': idx
    })

# Sort by distance
subdistrict_distances.sort(key=lambda x: x['distance_km'])

# Show closest sub-districts
print(f"\n📍 Closest sub-districts to city center:")
for i, sd in enumerate(subdistrict_distances[:15], 1):
    print(f"  {i:2d}. {sd['name']:30s} ({sd['district']:15s}) - {sd['distance_km']:5.1f} km")

# ============================================================================
# Step 3: Select Strategy
# ============================================================================

print(f"\n" + "-"*80)
print("SELECTION STRATEGY")
print("-"*80)

# Strategy: Select sub-districts within X km of city center
# Adjust RADIUS_KM to control area size:
#   - 15 km = small city area
#   - 22 km = moderate city + suburbs
#   - 28 km = larger area (RECOMMENDED)
#   - 35 km = very large area

RADIUS_KM = 28  # 👈 ADJUST THIS

# First filter by distance
candidates = [sd for sd in subdistrict_distances if sd['distance_km'] <= RADIUS_KM]

# OPTION: Filter out sub-districts that are too far EAST from legacy bbox
# Legacy bbox center: ~103.6 lon
LEGACY_BBOX_CENTER_LON = 103.6026  # Center of legacy bbox
MAX_DISTANCE_EAST = 0.18  # Maximum degrees east from legacy center (~20 km) 👈 ADJUST THIS

# Filter: keep only sub-districts not too far east
selected_subdistricts = []
for sd in candidates:
    centroid = sd['geometry'].centroid
    distance_east = centroid.x - LEGACY_BBOX_CENTER_LON

    # Keep if not too far east
    if distance_east <= MAX_DISTANCE_EAST:
        selected_subdistricts.append(sd)
    else:
        print(f"  ❌ Excluding (too far east): {sd['name']} - {distance_east:.3f}° east")

print(f"\n✓ Selection: All sub-districts within {RADIUS_KM} km of city center")
print(f"✓ Selected: {len(selected_subdistricts)} sub-districts")

print(f"\nSelected sub-districts:")
for sd in selected_subdistricts:
    print(f"  - {sd['name']:30s} ({sd['district']:15s}) - {sd['distance_km']:5.1f} km")

# Get unified geometry with NATURAL administrative curves
selected_geometries = [sd['geometry'] for sd in selected_subdistricts]
unified_geometry = unary_union(selected_geometries)

# Calculate area
area_km2 = unified_geometry.area * (111**2)
bounds = unified_geometry.bounds

print(f"\n✓ Unified region:")
print(f"  Area: ~{area_km2:.0f} km²")
print(f"  Bounds: {bounds}")

# ============================================================================
# Step 4: Crop Sentinel-2
# ============================================================================

print(f"\n" + "-"*80)
print("CROPPING SENTINEL-2")
print("-"*80)

INPUT_TILES = [
    'data/sentinel/S2_jambi_2024_20m_AllBands-0000000000-0000000000.tif',
    'data/sentinel/S2_jambi_2024_20m_AllBands-0000000000-0000010496.tif',
    'data/sentinel/S2_jambi_2024_20m_AllBands-0000010496-0000000000.tif',
    'data/sentinel/S2_jambi_2024_20m_AllBands-0000010496-0000010496.tif'
]

OUTPUT_FILE = f'data/sentinel/S2_jambi_subdistrict_{RADIUS_KM}km_20m.tif'
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# Check which tiles intersect
region_box = box(*bounds)
tiles_to_use = []

for tile_path in INPUT_TILES:
    with rasterio.open(tile_path) as src:
        tile_bounds = src.bounds
        tile_box = box(tile_bounds.left, tile_bounds.bottom,
                      tile_bounds.right, tile_bounds.top)

        if tile_box.intersects(region_box):
            tiles_to_use.append(tile_path)
            print(f"  ✓ {os.path.basename(tile_path)}")

print(f"\n{len(tiles_to_use)} tile(s) will be used")

# Mosaic if needed
if len(tiles_to_use) == 1:
    mosaic_path = tiles_to_use[0]
    should_cleanup = False
else:
    print(f"\nMosaicking {len(tiles_to_use)} tiles...")
    src_files = [rasterio.open(path) for path in tiles_to_use]
    mosaic_data, mosaic_transform = merge(src_files)

    mosaic_path = 'data/sentinel/temp_mosaic_subdist.tif'

    profile = src_files[0].profile.copy()
    profile.update({
        'height': mosaic_data.shape[1],
        'width': mosaic_data.shape[2],
        'transform': mosaic_transform,
        'count': mosaic_data.shape[0],
        'compress': 'lzw'
    })

    with rasterio.open(mosaic_path, 'w', **profile) as dst:
        dst.write(mosaic_data)

    for src in src_files:
        src.close()

    should_cleanup = True
    print(f"✓ Mosaic created")

# Crop to sub-district boundaries
with rasterio.open(mosaic_path) as src:
    print(f"\nCropping to natural administrative boundaries...")

    geom_list = [mapping(unified_geometry)]
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

    print(f"  Shape: {out_image.shape[1]} × {out_image.shape[2]} pixels")
    print(f"  Bands: {out_image.shape[0]}")

    size_mb = out_image.shape[0] * out_image.shape[1] * out_image.shape[2] * 4 / (1024**2)
    print(f"  Estimated size: {size_mb:.1f} MB")

    with rasterio.open(OUTPUT_FILE, "w", **out_meta) as dest:
        dest.write(out_image)

    print(f"✓ Saved: {OUTPUT_FILE}")

# Cleanup
if should_cleanup and os.path.exists(mosaic_path):
    os.remove(mosaic_path)

# Save boundaries
boundary_output = f'data/jambi_subdistrict_{RADIUS_KM}km_boundary.geojson'
gdf = gpd.GeoDataFrame(selected_subdistricts)
gdf.to_file(boundary_output, driver='GeoJSON')
print(f"✓ Saved boundaries: {boundary_output}")

# ============================================================================
# Summary
# ============================================================================

print(f"\n{'='*80}")
print("COMPLETE!")
print("="*80)
print(f"✓ Used NATURAL sub-district (kecamatan) boundaries!")
print(f"✓ Selected {len(selected_subdistricts)} sub-districts within {RADIUS_KM} km")
print(f"✓ Area: ~{area_km2:.0f} km²")
print(f"✓ IRREGULAR administrative curves (like real maps!)")
print(f"✓ Output: {OUTPUT_FILE}")
print(f"✓ Boundary: {boundary_output}")
print(f"\n💡 TIP: Adjust RADIUS_KM to control area size")
print(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
