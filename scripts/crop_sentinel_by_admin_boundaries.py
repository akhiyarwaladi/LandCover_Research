#!/usr/bin/env python3
"""
Crop Existing Sentinel-2 Using Administrative Boundaries
=========================================================

Downloads district-level administrative boundaries and crops existing
Sentinel-2 data to get natural curved boundaries!

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
from shapely.geometry import mapping, box
import numpy as np
from datetime import datetime
import requests
import json

print("="*80)
print("CROP SENTINEL-2 USING ADMINISTRATIVE BOUNDARIES")
print("="*80)
print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# Step 1: Download Administrative Boundaries from GADM
# ============================================================================

print("\n" + "-"*80)
print("DOWNLOADING ADMINISTRATIVE BOUNDARIES")
print("-"*80)

# GADM provides free administrative boundaries
# Level 2 = Kabupaten/Kota (districts)
gadm_url = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_IDN_2.json"
boundary_file = "data/gadm_jambi_districts.geojson"

if not os.path.exists(boundary_file):
    print(f"Downloading from GADM...")
    print(f"  URL: {gadm_url}")
    print(f"  This may take a minute...")

    response = requests.get(gadm_url, timeout=300)

    if response.status_code == 200:
        # Save to file
        with open(boundary_file, 'w') as f:
            f.write(response.text)
        print(f"✓ Downloaded: {len(response.content)/1024/1024:.1f} MB")
    else:
        print(f"❌ Download failed: {response.status_code}")
        print(f"\nPlease manually download from:")
        print(f"  https://gadm.org/download_country.html")
        print(f"  Select: Indonesia -> Level 2 -> GeoJSON")
        sys.exit(1)
else:
    print(f"✓ Using cached file: {boundary_file}")

# Load administrative boundaries
print(f"\nLoading administrative boundaries...")
admin = gpd.read_file(boundary_file)
print(f"✓ Loaded {len(admin):,} administrative units (all Indonesia)")

# Filter to Jambi Province only
jambi_districts = admin[admin['NAME_1'] == 'Jambi'].copy()
print(f"✓ Filtered to Jambi Province: {len(jambi_districts)} districts")

# Show available districts
print(f"\nAvailable districts in Jambi:")
for i, row in jambi_districts.iterrows():
    district_name = row['NAME_2']
    print(f"  - {district_name}")

# ============================================================================
# Step 2: Select Districts
# ============================================================================

print(f"\n" + "-"*80)
print("DISTRICT SELECTION")
print("-"*80)

# Define options (names match GADM - no spaces!)
DISTRICT_OPTIONS = {
    'option1': ['Jambi'],  # Just city
    'option2': ['Jambi', 'MuaroJambi'],  # City + 1 adjacent (RECOMMENDED)
    'option3': ['Jambi', 'MuaroJambi', 'BatangHari'],  # City + 2 adjacent
}

print(f"\nAvailable options:")
print(f"  Option 1: Jambi City only (smallest)")
print(f"  Option 2: Jambi + Muaro Jambi (moderate - RECOMMENDED)")
print(f"  Option 3: Jambi + Muaro Jambi + Batanghari (larger)")

# SELECT YOUR OPTION HERE:
selected_option = 'option2'  # 👈 CHANGE THIS: 'option1', 'option2', or 'option3'

selected_districts = DISTRICT_OPTIONS[selected_option]
print(f"\n✓ Selected: {', '.join(selected_districts)}")

# Filter to selected districts
region_gdf = jambi_districts[jambi_districts['NAME_2'].isin(selected_districts)].copy()

if len(region_gdf) == 0:
    print(f"❌ No districts found! Check district names.")
    print(f"Available: {jambi_districts['NAME_2'].tolist()}")
    sys.exit(1)

print(f"✓ Found {len(region_gdf)} district(s)")

# Merge geometries to get unified region
from shapely.ops import unary_union
unified_geometry = unary_union(region_gdf.geometry)

# Calculate area
area_km2 = unified_geometry.area * (111**2)  # Rough approximation
bounds = unified_geometry.bounds

print(f"\nRegion statistics:")
print(f"  Districts: {', '.join(selected_districts)}")
print(f"  Area: ~{area_km2:.0f} km²")
print(f"  Bounds: {bounds}")

# ============================================================================
# Step 3: Load and Mosaic Sentinel-2 Tiles
# ============================================================================

print(f"\n" + "-"*80)
print("LOADING SENTINEL-2 DATA")
print("-"*80)

INPUT_TILES = [
    'data/sentinel/S2_jambi_2024_20m_AllBands-0000000000-0000000000.tif',
    'data/sentinel/S2_jambi_2024_20m_AllBands-0000000000-0000010496.tif',
    'data/sentinel/S2_jambi_2024_20m_AllBands-0000010496-0000000000.tif',
    'data/sentinel/S2_jambi_2024_20m_AllBands-0000010496-0000010496.tif'
]

# Check which tiles intersect with region
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
    print(f"\nUsing single tile...")
    mosaic_path = tiles_to_use[0]
    should_cleanup = False
else:
    print(f"\nMosaicking {len(tiles_to_use)} tiles...")
    src_files = [rasterio.open(path) for path in tiles_to_use]
    mosaic_data, mosaic_transform = merge(src_files)

    # Save temporary mosaic
    mosaic_path = 'data/sentinel/temp_mosaic_admin.tif'

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
    print(f"✓ Mosaic created: {mosaic_data.shape}")

# ============================================================================
# Step 4: Crop to Administrative Boundaries
# ============================================================================

print(f"\n" + "-"*80)
print("CROPPING TO ADMINISTRATIVE BOUNDARIES")
print("-"*80)

OUTPUT_FILE = f'data/sentinel/S2_jambi_admin_{selected_option}_20m.tif'
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with rasterio.open(mosaic_path) as src:
    print(f"\nInput:")
    print(f"  Shape: {src.shape[0]} × {src.shape[1]} pixels")
    print(f"  Bands: {src.count}")
    print(f"  CRS: {src.crs}")

    # Crop to admin boundaries
    geom_list = [mapping(unified_geometry)]
    out_image, out_transform = mask(src, geom_list, crop=True, all_touched=True)

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

    print(f"\nCropped output:")
    print(f"  Shape: {out_image.shape[1]} × {out_image.shape[2]} pixels")
    print(f"  Bands: {out_image.shape[0]}")

    # Calculate size
    total_pixels = out_image.shape[0] * out_image.shape[1] * out_image.shape[2]
    size_mb = total_pixels * 4 / (1024**2)
    print(f"  Estimated size: {size_mb:.1f} MB")

    # Write output
    print(f"\nWriting to file...")
    with rasterio.open(OUTPUT_FILE, "w", **out_meta) as dest:
        dest.write(out_image)

    print(f"✓ Done!")

# Cleanup temporary mosaic
if should_cleanup and os.path.exists(mosaic_path):
    os.remove(mosaic_path)
    print(f"✓ Cleaned up temporary mosaic")

# ============================================================================
# Step 5: Save Administrative Boundary
# ============================================================================

boundary_output = f'data/jambi_admin_{selected_option}_boundary.geojson'
region_gdf.to_file(boundary_output, driver='GeoJSON')
print(f"✓ Saved boundary: {boundary_output}")

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

print(f"\n{'='*80}")
print("COMPLETE!")
print("="*80)
print(f"✓ Cropped to administrative boundaries: {', '.join(selected_districts)}")
print(f"✓ Natural curved boundaries (not circular or rectangular!)")
print(f"✓ Output: {OUTPUT_FILE}")
print(f"✓ Boundary: {boundary_output}")
print(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
