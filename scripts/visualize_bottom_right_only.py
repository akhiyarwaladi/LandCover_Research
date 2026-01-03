#!/usr/bin/env python3
"""
Visualize Bottom-Right Corner Only Sub-Districts
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import geopandas as gpd
from shapely.ops import unary_union

from modules.data_loader import load_klhk_data, CLASS_NAMES

print("Visualizing bottom-right corner only sub-districts...")

SENTINEL2_PATH = 'data/sentinel/S2_jambi_subdistrict_bottom_right_only_20m.tif'
KLHK_PATH = 'data/klhk/KLHK_PL2024_Jambi_Full_WithGeometry.geojson'
OUTPUT_PATH = 'results/sentinel_klhk_overlay_subdistrict_bottom_right_only.png'

# Legacy bbox
LEGACY_BBOX = {'min_lon': 103.4486, 'min_lat': -1.8337,
               'max_lon': 103.7566, 'max_lat': -1.4089}

CLASS_COLORS = {0: '#0000FF', 1: '#006400', 2: '#90EE90',
                4: '#FFD700', 5: '#D2B48C', 6: '#FF0000', 7: '#8B4513'}

# Load Sentinel-2
with rasterio.open(SENTINEL2_PATH) as src:
    red, green, blue = src.read(3), src.read(2), src.read(1)
    bounds = src.bounds

def norm(b):
    v = b[b>0]
    if len(v)==0: return np.zeros_like(b)
    p2,p98 = np.percentile(v, [2,98])
    return np.clip((b-p2)/(p98-p2), 0, 1)

rgb = np.dstack([norm(red), norm(green), norm(blue)])

# Load boundaries - reconstruct from current + SungaiGelam only
current = gpd.read_file('data/jambi_subdistrict_28km_boundary.geojson')
all_jambi = gpd.read_file('data/gadm_indonesia_subdistricts.geojson')
all_jambi = all_jambi[all_jambi['NAME_1'] == 'Jambi']

# Only add SungaiGelam
corner_name = 'SungaiGelam'
corner_geom = None
sd = all_jambi[all_jambi['NAME_3'] == corner_name]
if len(sd) > 0:
    corner_geom = sd.iloc[0].geometry

all_geoms = list(current.geometry)
if corner_geom is not None:
    all_geoms.append(corner_geom)

combined_boundary = unary_union(all_geoms)

# Load KLHK
klhk = load_klhk_data(KLHK_PATH, verbose=False)
klhk_clip = gpd.clip(klhk[klhk.intersects(combined_boundary)], combined_boundary)

area_km2 = combined_boundary.area * (111**2)
print(f"✓ Area: ~{area_km2:.0f} km²")
print(f"✓ KLHK: {len(klhk_clip):,} polygons")
print(f"✓ 12 sub-districts (11 + SungaiGelam)")

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(21, 7))

# Panel 1
ax1 = axes[0]
ax1.imshow(rgb, extent=[bounds.left, bounds.right, bounds.bottom, bounds.top])
gpd.GeoSeries([combined_boundary]).boundary.plot(ax=ax1, color='lime', linewidth=2, label='12 Sub-Districts')
legacy_rect = Rectangle((LEGACY_BBOX['min_lon'], LEGACY_BBOX['min_lat']),
                        LEGACY_BBOX['max_lon'] - LEGACY_BBOX['min_lon'],
                        LEGACY_BBOX['max_lat'] - LEGACY_BBOX['min_lat'],
                        linewidth=2.5, edgecolor='red', facecolor='none', label='Legacy BBox')
ax1.add_patch(legacy_rect)
ax1.set_title('Sentinel-2 RGB\\n(12 Sub-Districts - Bottom-Right Only!)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Panel 2
ax2 = axes[1]
ax2.set_xlim(bounds.left, bounds.right)
ax2.set_ylim(bounds.bottom, bounds.top)
ax2.set_aspect('equal')

class_counts = klhk_clip['class_simplified'].value_counts().sort_index()
for cls in sorted(klhk_clip['class_simplified'].unique()):
    subset = klhk_clip[klhk_clip['class_simplified'] == cls]
    subset.plot(ax=ax2, color=CLASS_COLORS.get(cls, '#808080'), edgecolor='none', alpha=0.8)

gpd.GeoSeries([combined_boundary]).boundary.plot(ax=ax2, color='lime', linewidth=2)
legacy_rect2 = Rectangle((LEGACY_BBOX['min_lon'], LEGACY_BBOX['min_lat']),
                         LEGACY_BBOX['max_lon'] - LEGACY_BBOX['min_lon'],
                         LEGACY_BBOX['max_lat'] - LEGACY_BBOX['min_lat'],
                         linewidth=2.5, edgecolor='red', facecolor='none')
ax2.add_patch(legacy_rect2)
ax2.set_title('KLHK Ground Truth', fontsize=12, fontweight='bold')
ax2.set_xlabel('Longitude')
ax2.set_ylabel('Latitude')
ax2.grid(True, alpha=0.3)

# Panel 3
ax3 = axes[2]
ax3.imshow(rgb, extent=[bounds.left, bounds.right, bounds.bottom, bounds.top], alpha=0.7)
for cls in sorted(klhk_clip['class_simplified'].unique()):
    subset = klhk_clip[klhk_clip['class_simplified'] == cls]
    subset.plot(ax=ax3, color=CLASS_COLORS.get(cls, '#808080'), edgecolor='black', linewidth=0.3, alpha=0.5)

gpd.GeoSeries([combined_boundary]).boundary.plot(ax=ax3, color='lime', linewidth=2)
legacy_rect3 = Rectangle((LEGACY_BBOX['min_lon'], LEGACY_BBOX['min_lat']),
                         LEGACY_BBOX['max_lon'] - LEGACY_BBOX['min_lon'],
                         LEGACY_BBOX['max_lat'] - LEGACY_BBOX['min_lat'],
                         linewidth=2.5, edgecolor='red', facecolor='none')
ax3.add_patch(legacy_rect3)
ax3.set_title('RGB + Ground Truth\\n(Bottom-Right Corner Filled)', fontsize=12, fontweight='bold')
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
                                     label='12 Sub-Districts'))
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

print("\\n✅ COMPLETE!")
print(f"✓ Natural administrative boundaries")
print(f"✓ Bottom-right filled with SungaiGelam!")
print(f"✓ Top-left kept original (not filled)")
print(f"✓ Area: ~{area_km2:.0f} km²")
