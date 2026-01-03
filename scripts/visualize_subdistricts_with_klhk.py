#!/usr/bin/env python3
"""
Visualize Sub-District Boundaries with KLHK Ground Truth
=========================================================

Shows NATURAL administrative boundaries (kecamatan level) with
irregular curves like real administrative maps!

Author: Claude Sonnet 4.5
Date: 2026-01-03
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd

from modules.data_loader import load_klhk_data, CLASS_NAMES

print("="*80)
print("VISUALIZE SUB-DISTRICT BOUNDARIES + KLHK")
print("="*80)

# ============================================================================
# Configuration
# ============================================================================

RADIUS_KM = 28  # Match the radius used in crop script

SENTINEL2_PATH = f'data/sentinel/S2_jambi_subdistrict_{RADIUS_KM}km_20m.tif'
BOUNDARY_PATH = f'data/jambi_subdistrict_{RADIUS_KM}km_boundary.geojson'
KLHK_PATH = 'data/klhk/KLHK_PL2024_Jambi_Full_WithGeometry.geojson'
OUTPUT_PATH = f'results/sentinel_klhk_overlay_subdistrict_{RADIUS_KM}km.png'

# Legacy bounding box (untuk dibandingkan)
LEGACY_BBOX = {
    'min_lon': 103.4486,
    'min_lat': -1.8337,
    'max_lon': 103.7566,
    'max_lat': -1.4089
}

# Class colors
CLASS_COLORS = {
    0: '#0000FF',  # Water
    1: '#006400',  # Trees/Forest
    2: '#90EE90',  # Grass/Savanna
    4: '#FFD700',  # Crops/Agriculture
    5: '#D2B48C',  # Shrub/Scrub
    6: '#FF0000',  # Built Area
    7: '#8B4513',  # Bare Ground
}

# ============================================================================
# Load Data
# ============================================================================

print("\n" + "-"*80)
print("LOADING SENTINEL-2 DATA")
print("-"*80)

with rasterio.open(SENTINEL2_PATH) as src:
    red = src.read(3)
    green = src.read(2)
    blue = src.read(1)
    bounds = src.bounds

    print(f"✓ Loaded: {SENTINEL2_PATH}")
    print(f"  Shape: {red.shape[0]} × {red.shape[1]} pixels")

def normalize_band(band):
    valid = band[band > 0]
    if len(valid) == 0:
        return np.zeros_like(band)
    p2 = np.percentile(valid, 2)
    p98 = np.percentile(valid, 98)
    return np.clip((band - p2) / (p98 - p2), 0, 1)

rgb = np.dstack([normalize_band(red), normalize_band(green), normalize_band(blue)])
print("✓ RGB normalized")

print("\n" + "-"*80)
print("LOADING SUB-DISTRICT BOUNDARIES")
print("-"*80)

boundaries = gpd.read_file(BOUNDARY_PATH)
print(f"✓ Loaded {len(boundaries)} sub-districts")

district_names = boundaries['name'].tolist()
print(f"  Sub-districts: {', '.join(district_names[:5])}...")

area_km2 = boundaries.unary_union.area * (111**2)
print(f"  Total area: ~{area_km2:.0f} km²")

print("\n" + "-"*80)
print("LOADING KLHK GROUND TRUTH")
print("-"*80)

klhk_full = load_klhk_data(KLHK_PATH, verbose=False)
print(f"✓ Loaded KLHK: {len(klhk_full):,} total polygons")

from shapely.ops import unary_union
unified_boundary = unary_union(boundaries.geometry)

klhk_clipped = klhk_full[klhk_full.intersects(unified_boundary)].copy()
klhk_clipped = gpd.clip(klhk_clipped, unified_boundary)

print(f"✓ Clipped: {len(klhk_clipped):,} polygons")

print(f"\n  Class distribution:")
class_counts = klhk_clipped['class_simplified'].value_counts().sort_index()
for cls, count in class_counts.items():
    name = CLASS_NAMES.get(cls, 'Unknown')
    print(f"    {cls}: {name:20s} - {count:,} polygons")

# ============================================================================
# Visualization
# ============================================================================

print("\n" + "-"*80)
print("CREATING VISUALIZATION")
print("-"*80)

fig, axes = plt.subplots(1, 3, figsize=(21, 7))

# Panel 1: RGB + Boundaries
ax1 = axes[0]
ax1.imshow(rgb, extent=[bounds.left, bounds.right, bounds.bottom, bounds.top])
boundaries.boundary.plot(ax=ax1, color='lime', linewidth=1.5, label='Sub-District Boundaries')

# Add legacy bbox as red rectangle
from matplotlib.patches import Rectangle
legacy_rect = Rectangle(
    (LEGACY_BBOX['min_lon'], LEGACY_BBOX['min_lat']),
    LEGACY_BBOX['max_lon'] - LEGACY_BBOX['min_lon'],
    LEGACY_BBOX['max_lat'] - LEGACY_BBOX['min_lat'],
    linewidth=2.5, edgecolor='red', facecolor='none', label='Legacy BBox'
)
ax1.add_patch(legacy_rect)

ax1.set_title(f'Sentinel-2 RGB Composite\n({len(boundaries)} Sub-Districts, Natural Admin Curves)',
              fontsize=12, fontweight='bold')
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Panel 2: KLHK Ground Truth
ax2 = axes[1]
ax2.set_xlim(bounds.left, bounds.right)
ax2.set_ylim(bounds.bottom, bounds.top)
ax2.set_aspect('equal')

for cls in sorted(klhk_clipped['class_simplified'].unique()):
    subset = klhk_clipped[klhk_clipped['class_simplified'] == cls]
    color = CLASS_COLORS.get(cls, '#808080')
    subset.plot(ax=ax2, color=color, edgecolor='none', alpha=0.8)

boundaries.boundary.plot(ax=ax2, color='lime', linewidth=1.5)

# Add legacy bbox
legacy_rect2 = Rectangle(
    (LEGACY_BBOX['min_lon'], LEGACY_BBOX['min_lat']),
    LEGACY_BBOX['max_lon'] - LEGACY_BBOX['min_lon'],
    LEGACY_BBOX['max_lat'] - LEGACY_BBOX['min_lat'],
    linewidth=2.5, edgecolor='red', facecolor='none'
)
ax2.add_patch(legacy_rect2)

ax2.set_title('KLHK Ground Truth\n(Land Cover Classes)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Longitude')
ax2.set_ylabel('Latitude')
ax2.grid(True, alpha=0.3)

# Panel 3: Overlay
ax3 = axes[2]
ax3.imshow(rgb, extent=[bounds.left, bounds.right, bounds.bottom, bounds.top], alpha=0.7)

for cls in sorted(klhk_clipped['class_simplified'].unique()):
    subset = klhk_clipped[klhk_clipped['class_simplified'] == cls]
    color = CLASS_COLORS.get(cls, '#808080')
    subset.plot(ax=ax3, color=color, edgecolor='black', linewidth=0.3, alpha=0.5)

boundaries.boundary.plot(ax=ax3, color='lime', linewidth=1.5)

# Add legacy bbox
legacy_rect3 = Rectangle(
    (LEGACY_BBOX['min_lon'], LEGACY_BBOX['min_lat']),
    LEGACY_BBOX['max_lon'] - LEGACY_BBOX['min_lon'],
    LEGACY_BBOX['max_lat'] - LEGACY_BBOX['min_lat'],
    linewidth=2.5, edgecolor='red', facecolor='none'
)
ax3.add_patch(legacy_rect3)

ax3.set_title('RGB + Ground Truth Overlay\n(NATURAL Administrative Curves!)',
              fontsize=12, fontweight='bold')
ax3.set_xlabel('Longitude')
ax3.set_ylabel('Latitude')
ax3.grid(True, alpha=0.3)

# Legend
legend_patches = []
for cls in sorted(klhk_clipped['class_simplified'].unique()):
    name = CLASS_NAMES.get(cls, 'Unknown')
    color = CLASS_COLORS.get(cls, '#808080')
    count = (klhk_clipped['class_simplified'] == cls).sum()
    legend_patches.append(mpatches.Patch(color=color, label=f"{name} ({count:,})"))

legend_patches.append(mpatches.Patch(facecolor='none', edgecolor='lime', linewidth=1.5,
                                     label='Sub-District Boundaries'))
legend_patches.append(mpatches.Patch(facecolor='none', edgecolor='red', linewidth=2.5,
                                     label='Legacy BBox'))

fig.legend(handles=legend_patches, loc='lower center', ncol=4,
           frameon=True, fontsize=9, bbox_to_anchor=(0.5, -0.05))

plt.tight_layout()
plt.subplots_adjust(bottom=0.13)

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {OUTPUT_PATH}")

plt.close()

print("\n" + "="*80)
print("COMPLETE!")
print("="*80)
print(f"✓ NATURAL administrative boundaries (irregular curves!)")
print(f"✓ {len(boundaries)} sub-districts")
print(f"✓ Area: ~{area_km2:.0f} km²")
print(f"✓ KLHK polygons: {len(klhk_clipped):,}")
print(f"✓ Output: {OUTPUT_PATH}")
print("="*80)
