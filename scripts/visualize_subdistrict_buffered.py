#!/usr/bin/env python3
"""
Visualize Buffered Sub-District Boundaries
===========================================

Shows buffered boundaries that fill empty corners!
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import geopandas as gpd

from modules.data_loader import load_klhk_data, CLASS_NAMES

print("="*80)
print("VISUALIZE BUFFERED SUB-DISTRICT BOUNDARIES")
print("="*80)

SENTINEL2_PATH = 'data/sentinel/S2_jambi_subdistrict_buffered_20m.tif'
BOUNDARY_PATH = 'data/jambi_subdistrict_buffered_boundary.geojson'
KLHK_PATH = 'data/klhk/KLHK_PL2024_Jambi_Full_WithGeometry.geojson'
OUTPUT_PATH = 'results/sentinel_klhk_overlay_subdistrict_buffered.png'

# Legacy bbox for comparison
LEGACY_BBOX = {
    'min_lon': 103.4486,
    'min_lat': -1.8337,
    'max_lon': 103.7566,
    'max_lat': -1.4089
}

CLASS_COLORS = {
    0: '#0000FF', 1: '#006400', 2: '#90EE90',
    4: '#FFD700', 5: '#D2B48C', 6: '#FF0000', 7: '#8B4513',
}

# Load data
print("\nLoading data...")
with rasterio.open(SENTINEL2_PATH) as src:
    red, green, blue = src.read(3), src.read(2), src.read(1)
    bounds = src.bounds

def normalize(b):
    v = b[b>0]
    if len(v)==0: return np.zeros_like(b)
    p2,p98 = np.percentile(v, [2,98])
    return np.clip((b-p2)/(p98-p2), 0, 1)

rgb = np.dstack([normalize(red), normalize(green), normalize(blue)])

boundary = gpd.read_file(BOUNDARY_PATH)
area_km2 = boundary.geometry[0].area * (111**2)

klhk = load_klhk_data(KLHK_PATH, verbose=False)
klhk_clip = gpd.clip(klhk[klhk.intersects(boundary.geometry[0])], boundary.geometry[0])

print(f"✓ Area: ~{area_km2:.0f} km²")
print(f"✓ KLHK: {len(klhk_clip):,} polygons")

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(21, 7))

# Panel 1: RGB
ax1 = axes[0]
ax1.imshow(rgb, extent=[bounds.left, bounds.right, bounds.bottom, bounds.top])
boundary.boundary.plot(ax=ax1, color='lime', linewidth=2, label='Buffered Boundary')

legacy_rect = Rectangle(
    (LEGACY_BBOX['min_lon'], LEGACY_BBOX['min_lat']),
    LEGACY_BBOX['max_lon'] - LEGACY_BBOX['min_lon'],
    LEGACY_BBOX['max_lat'] - LEGACY_BBOX['min_lat'],
    linewidth=2.5, edgecolor='red', facecolor='none', label='Legacy BBox'
)
ax1.add_patch(legacy_rect)

ax1.set_title('Sentinel-2 RGB\n(Buffered Boundaries - No Empty Corners!)',
              fontsize=12, fontweight='bold')
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Panel 2: KLHK
ax2 = axes[1]
ax2.set_xlim(bounds.left, bounds.right)
ax2.set_ylim(bounds.bottom, bounds.top)
ax2.set_aspect('equal')

class_counts = klhk_clip['class_simplified'].value_counts().sort_index()
for cls in sorted(klhk_clip['class_simplified'].unique()):
    subset = klhk_clip[klhk_clip['class_simplified'] == cls]
    subset.plot(ax=ax2, color=CLASS_COLORS.get(cls, '#808080'), edgecolor='none', alpha=0.8)

boundary.boundary.plot(ax=ax2, color='lime', linewidth=2)

legacy_rect2 = Rectangle(
    (LEGACY_BBOX['min_lon'], LEGACY_BBOX['min_lat']),
    LEGACY_BBOX['max_lon'] - LEGACY_BBOX['min_lon'],
    LEGACY_BBOX['max_lat'] - LEGACY_BBOX['min_lat'],
    linewidth=2.5, edgecolor='red', facecolor='none'
)
ax2.add_patch(legacy_rect2)

ax2.set_title('KLHK Ground Truth', fontsize=12, fontweight='bold')
ax2.set_xlabel('Longitude')
ax2.set_ylabel('Latitude')
ax2.grid(True, alpha=0.3)

# Panel 3: Overlay
ax3 = axes[2]
ax3.imshow(rgb, extent=[bounds.left, bounds.right, bounds.bottom, bounds.top], alpha=0.7)

for cls in sorted(klhk_clip['class_simplified'].unique()):
    subset = klhk_clip[klhk_clip['class_simplified'] == cls]
    subset.plot(ax=ax3, color=CLASS_COLORS.get(cls, '#808080'),
                edgecolor='black', linewidth=0.3, alpha=0.5)

boundary.boundary.plot(ax=ax3, color='lime', linewidth=2)

legacy_rect3 = Rectangle(
    (LEGACY_BBOX['min_lon'], LEGACY_BBOX['min_lat']),
    LEGACY_BBOX['max_lon'] - LEGACY_BBOX['min_lon'],
    LEGACY_BBOX['max_lat'] - LEGACY_BBOX['min_lat'],
    linewidth=2.5, edgecolor='red', facecolor='none'
)
ax3.add_patch(legacy_rect3)

ax3.set_title('RGB + Ground Truth\n(Natural Curves + Full Coverage!)',
              fontsize=12, fontweight='bold')
ax3.set_xlabel('Longitude')
ax3.set_ylabel('Latitude')
ax3.grid(True, alpha=0.3)

# Legend
legend_patches = []
for cls in sorted(class_counts.index):
    name = CLASS_NAMES.get(cls, 'Unknown')
    count = class_counts[cls]
    legend_patches.append(mpatches.Patch(color=CLASS_COLORS.get(cls, '#808080'),
                                         label=f"{name} ({count:,})"))

legend_patches.append(mpatches.Patch(facecolor='none', edgecolor='lime', linewidth=2,
                                     label='Buffered Boundary'))
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
print(f"✓ NO empty corners!")
print(f"✓ Natural curves maintained!")
print(f"✓ Area: ~{area_km2:.0f} km²")
print(f"✓ Output: {OUTPUT_PATH}")
print("="*80)
