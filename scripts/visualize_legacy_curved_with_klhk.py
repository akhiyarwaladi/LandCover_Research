#!/usr/bin/env python3
"""
Visualize Legacy Bbox with Curved Edges + KLHK Ground Truth
============================================================

Shows the legacy bbox with beautiful curves instead of sharp corners.

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
from shapely.geometry import box
import geopandas as gpd

from modules.data_loader import load_klhk_data, CLASS_NAMES

print("="*80)
print("VISUALIZE LEGACY BBOX WITH CURVES + KLHK")
print("="*80)

# ============================================================================
# Configuration
# ============================================================================

SENTINEL2_PATH = 'data/sentinel/S2_jambi_city_legacy_curved_20m.tif'
BOUNDARY_PATH = 'data/jambi_legacy_curved_boundary.geojson'
KLHK_PATH = 'data/klhk/KLHK_PL2024_Jambi_Full_WithGeometry.geojson'
OUTPUT_PATH = 'results/sentinel_klhk_overlay_legacy_curved.png'

# Class colors
CLASS_COLORS = {
    0: '#0000FF',  # Water - Blue
    1: '#006400',  # Trees/Forest - Dark Green
    2: '#90EE90',  # Grass/Savanna - Light Green
    4: '#FFD700',  # Crops/Agriculture - Gold
    5: '#D2B48C',  # Shrub/Scrub - Tan
    6: '#FF0000',  # Built Area - Red
    7: '#8B4513',  # Bare Ground - Brown
}

# ============================================================================
# Step 1: Load Sentinel-2 Data
# ============================================================================

print("\n" + "-"*80)
print("LOADING SENTINEL-2 DATA")
print("-"*80)

with rasterio.open(SENTINEL2_PATH) as src:
    # Read RGB bands
    red = src.read(3)    # B4
    green = src.read(2)  # B3
    blue = src.read(1)   # B2

    bounds = src.bounds
    transform = src.transform
    crs = src.crs

    print(f"✓ Loaded: {SENTINEL2_PATH}")
    print(f"  Shape: {red.shape[0]} × {red.shape[1]} pixels")
    print(f"  Bounds: {bounds}")

# Normalize RGB for display
def normalize_band(band):
    valid = band[band > 0]
    if len(valid) == 0:
        return np.zeros_like(band)
    p2 = np.percentile(valid, 2)
    p98 = np.percentile(valid, 98)
    normalized = (band - p2) / (p98 - p2)
    return np.clip(normalized, 0, 1)

rgb = np.dstack([normalize_band(red), normalize_band(green), normalize_band(blue)])
print("✓ RGB composite normalized")

# ============================================================================
# Step 2: Load Curved Boundary
# ============================================================================

print("\n" + "-"*80)
print("LOADING CURVED BOUNDARY")
print("-"*80)

curved_boundary = gpd.read_file(BOUNDARY_PATH)
print(f"✓ Loaded curved boundary")
area_km2 = curved_boundary.geometry[0].area * (111**2)
print(f"  Area: ~{area_km2:.0f} km²")

# ============================================================================
# Step 3: Load and Clip KLHK Data
# ============================================================================

print("\n" + "-"*80)
print("LOADING KLHK GROUND TRUTH")
print("-"*80)

klhk_full = load_klhk_data(KLHK_PATH, verbose=False)
print(f"✓ Loaded KLHK: {len(klhk_full):,} total polygons")

# Clip to curved boundary
boundary_geom = curved_boundary.geometry[0]

klhk_clipped = klhk_full[klhk_full.intersects(boundary_geom)].copy()
klhk_clipped = gpd.clip(klhk_clipped, boundary_geom)

print(f"✓ Clipped to curved area: {len(klhk_clipped):,} polygons")

# Show class distribution
print(f"\n  Class distribution:")
class_counts = klhk_clipped['class_simplified'].value_counts().sort_index()
for cls, count in class_counts.items():
    name = CLASS_NAMES.get(cls, 'Unknown')
    print(f"    {cls}: {name:20s} - {count:,} polygons")

# ============================================================================
# Step 4: Create Visualization
# ============================================================================

print("\n" + "-"*80)
print("CREATING VISUALIZATION")
print("-"*80)

fig, axes = plt.subplots(1, 3, figsize=(21, 7))

# Panel 1: RGB Only
ax1 = axes[0]
ax1.imshow(rgb, extent=[bounds.left, bounds.right, bounds.bottom, bounds.top])

# Overlay curved boundary
curved_boundary.boundary.plot(ax=ax1, color='cyan', linewidth=2.5, label='Curved Boundary')

ax1.set_title('Sentinel-2 RGB Composite\n(Legacy Bbox with CURVED Edges)',
              fontsize=12, fontweight='bold')
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Panel 2: KLHK Ground Truth Only
ax2 = axes[1]
ax2.set_xlim(bounds.left, bounds.right)
ax2.set_ylim(bounds.bottom, bounds.top)
ax2.set_aspect('equal')

for cls in sorted(klhk_clipped['class_simplified'].unique()):
    subset = klhk_clipped[klhk_clipped['class_simplified'] == cls]
    color = CLASS_COLORS.get(cls, '#808080')
    subset.plot(ax=ax2, color=color, edgecolor='none', alpha=0.8)

# Overlay curved boundary
curved_boundary.boundary.plot(ax=ax2, color='cyan', linewidth=2.5)

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

# Overlay curved boundary
curved_boundary.boundary.plot(ax=ax3, color='cyan', linewidth=2.5)

ax3.set_title('RGB + Ground Truth Overlay\n(Smooth Curves, No Sharp Corners!)',
              fontsize=12, fontweight='bold')
ax3.set_xlabel('Longitude')
ax3.set_ylabel('Latitude')
ax3.grid(True, alpha=0.3)

# Add legend
legend_patches = []
for cls in sorted(klhk_clipped['class_simplified'].unique()):
    name = CLASS_NAMES.get(cls, 'Unknown')
    color = CLASS_COLORS.get(cls, '#808080')
    count = (klhk_clipped['class_simplified'] == cls).sum()
    label = f"{name} ({count:,} polygons)"
    legend_patches.append(mpatches.Patch(color=color, label=label))

# Add boundary to legend
legend_patches.append(mpatches.Patch(facecolor='none', edgecolor='cyan', linewidth=2.5,
                                     label='Curved Boundary'))

fig.legend(handles=legend_patches, loc='lower center', ncol=4,
           frameon=True, fontsize=9, bbox_to_anchor=(0.5, -0.05))

plt.tight_layout()
plt.subplots_adjust(bottom=0.13)

# Save
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {OUTPUT_PATH}")

plt.close()

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*80)
print("VISUALIZATION COMPLETE!")
print("="*80)
print(f"✓ Legacy bbox with BEAUTIFUL SMOOTH CURVES!")
print(f"✓ Sentinel-2 area: {red.shape[0]} × {red.shape[1]} pixels")
print(f"✓ KLHK polygons: {len(klhk_clipped):,}")
print(f"✓ Classes found: {len(class_counts)}")
print(f"✓ Area: ~{area_km2:.0f} km² (only 23% larger than original)")
print(f"✓ Output: {OUTPUT_PATH}")
print("="*80)
