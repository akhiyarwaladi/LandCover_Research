#!/usr/bin/env python3
"""
Crop Sentinel-2 Using Natural Boundary from KLHK Polygons
==========================================================

Creates a natural irregular boundary by:
1. Finding KLHK polygons near city center
2. Creating a concave hull of those polygons
3. This gives NATURAL curves (not geometric circles/ovals)!

Author: Claude Sonnet 4.5
Date: 2026-01-03
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import rasterio
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import Point, box, mapping
from shapely.ops import unary_union
import numpy as np
from datetime import datetime

print("="*80)
print("CREATE NATURAL BOUNDARY FROM KLHK POLYGONS")
print("="*80)
print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# Configuration
# ============================================================================

# Legacy bbox center (Jambi City)
CITY_CENTER = Point(103.6026, -1.6213)  # Center of legacy bbox

# Distance to include polygons (in degrees)
# Adjust this to control area size:
#   - 0.15° ≈ 17 km radius → ~900 km² (smaller, focused)
#   - 0.20° ≈ 22 km radius → ~1,500 km² (similar to legacy) ← RECOMMENDED
#   - 0.25° ≈ 28 km radius → ~2,500 km² (larger)
RADIUS_DEG = 0.20  # 👈 ADJUST THIS

print(f"\nConfiguration:")
print(f"  City center: {CITY_CENTER.x:.4f}, {CITY_CENTER.y:.4f}")
print(f"  Radius: {RADIUS_DEG:.4f}° (~{RADIUS_DEG * 111:.1f} km)")
print(f"  Expected area: ~{3.14159 * (RADIUS_DEG * 111)**2:.0f} km²")

# Paths
KLHK_PATH = 'data/klhk/KLHK_PL2024_Jambi_Full_WithGeometry.geojson'
INPUT_TILE = 'data/sentinel/S2_jambi_2024_20m_AllBands-0000000000-0000010496.tif'
OUTPUT_FILE = 'data/sentinel/S2_jambi_city_natural_boundary_20m.tif'
BOUNDARY_OUTPUT = 'data/jambi_natural_boundary.geojson'

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# ============================================================================
# Step 1: Load KLHK and Find Nearby Polygons
# ============================================================================

print(f"\n" + "-"*80)
print("LOADING KLHK DATA")
print("-"*80)

klhk = gpd.read_file(KLHK_PATH)
print(f"✓ Loaded {len(klhk):,} KLHK polygons")

# Create search radius circle
search_circle = CITY_CENTER.buffer(RADIUS_DEG)

# Find polygons that intersect with search circle
nearby_polygons = klhk[klhk.intersects(search_circle)].copy()
print(f"✓ Found {len(nearby_polygons):,} polygons within {RADIUS_DEG * 111:.1f} km")

# ============================================================================
# Step 2: Create Natural Boundary (Concave Hull)
# ============================================================================

print(f"\n" + "-"*80)
print("CREATING NATURAL BOUNDARY")
print("-"*80)

# Option 1: Use union of all nearby polygons (creates most natural boundary)
print(f"Method: Union of nearby KLHK polygons...")

# Merge all nearby polygons into one geometry
natural_boundary = unary_union(nearby_polygons.geometry)

# If needed, simplify slightly to reduce complexity
# (tolerance in degrees - smaller = more detail)
natural_boundary = natural_boundary.simplify(0.001, preserve_topology=True)

# Calculate area
area_km2 = natural_boundary.area * (111**2)
bounds = natural_boundary.bounds

print(f"✓ Natural boundary created!")
print(f"  Area: ~{area_km2:.0f} km²")
print(f"  Bounds: {bounds}")
print(f"  Geometry type: {natural_boundary.geom_type}")

# If we got a MultiPolygon, take the largest polygon
if natural_boundary.geom_type == 'MultiPolygon':
    print(f"  Note: Got MultiPolygon, extracting largest component...")
    # Get the largest polygon
    natural_boundary = max(natural_boundary.geoms, key=lambda p: p.area)
    area_km2 = natural_boundary.area * (111**2)
    print(f"  Largest component area: ~{area_km2:.0f} km²")

# Save boundary for visualization
boundary_gdf = gpd.GeoDataFrame({'geometry': [natural_boundary]}, crs='EPSG:4326')
boundary_gdf.to_file(BOUNDARY_OUTPUT, driver='GeoJSON')
print(f"✓ Saved boundary: {BOUNDARY_OUTPUT}")

# ============================================================================
# Step 3: Crop Sentinel-2 to Natural Boundary
# ============================================================================

print(f"\n" + "-"*80)
print("CROPPING SENTINEL-2")
print("-"*80)

print(f"\nInput: {INPUT_TILE}")

with rasterio.open(INPUT_TILE) as src:
    print(f"\nOriginal tile:")
    print(f"  Shape: {src.shape[0]} × {src.shape[1]} pixels")
    print(f"  Bands: {src.count}")

    # Crop to natural boundary
    natural_geom_list = [mapping(natural_boundary)]
    out_image, out_transform = mask(src, natural_geom_list, crop=True, all_touched=True)

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
    print(f"  Natural boundary area: ~{area_km2:.0f} km²")

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
print(f"✓ Created NATURAL boundary from KLHK polygons!")
print(f"✓ Irregular curves (NOT geometric circles or ovals)")
print(f"✓ Based on actual land cover patterns")
print(f"✓ City-focused, size: ~{area_km2:.0f} km²")
print(f"✓ Output: {OUTPUT_FILE}")
print(f"✓ Boundary: {BOUNDARY_OUTPUT}")
print(f"\n💡 TIP: Adjust RADIUS_DEG to control area size")
print(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
