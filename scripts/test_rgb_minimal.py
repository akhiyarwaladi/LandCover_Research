"""
Minimal RGB test - NO title, NO layout settings
Just raw RGB display to isolate the black box issue
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.data_loader import load_sentinel2_tiles
from scripts.generate_qualitative_FINAL import create_city_boundary, crop_raster_to_boundary
import geopandas as gpd

# Load data
print("Loading Sentinel-2...")
SENTINEL2_TILES = [
    'data/sentinel/S2_jambi_2024_20m_AllBands-0000000000-0000000000.tif',
    'data/sentinel/S2_jambi_2024_20m_AllBands-0000000000-0000010496.tif',
    'data/sentinel/S2_jambi_2024_20m_AllBands-0000010496-0000000000.tif',
    'data/sentinel/S2_jambi_2024_20m_AllBands-0000010496-0000010496.tif'
]
sentinel2_bands, s2_profile = load_sentinel2_tiles(SENTINEL2_TILES, verbose=False)

# Create boundary
JAMBI_CITY_CENTER = (-1.609972, 103.607254)
JAMBI_CITY_EXTENT = 0.15
city_boundary = create_city_boundary(JAMBI_CITY_CENTER[0], JAMBI_CITY_CENTER[1], JAMBI_CITY_EXTENT)

# Extract RGB
red = sentinel2_bands[2]    # B4
green = sentinel2_bands[1]  # B3
blue = sentinel2_bands[0]   # B2
rgb = np.stack([red, green, blue], axis=0)

# Crop
rgb_cropped, _ = crop_raster_to_boundary(rgb, s2_profile, city_boundary)
print(f"Cropped shape: {rgb_cropped.shape}")

# Check for NoData in different regions
print(f"\nData analysis:")
print(f"  Min values: R={rgb_cropped[0].min()}, G={rgb_cropped[1].min()}, B={rgb_cropped[2].min()}")
print(f"  Max values: R={rgb_cropped[0].max()}, G={rgb_cropped[1].max()}, B={rgb_cropped[2].max()}")

# Check top-left corner (first 100x100 pixels)
corner = rgb_cropped[:, :100, :100]
print(f"  Top-left corner (100x100):")
print(f"    R: min={corner[0].min()}, max={corner[0].max()}, mean={corner[0].mean():.2f}")
print(f"    NoData pixels: {np.sum(np.all(corner <= 0, axis=0))}")

# Normalize
rgb_display = np.ones_like(rgb_cropped, dtype=np.float32)
for i in range(3):
    band = rgb_cropped[i]
    valid = band[band > 0]
    if len(valid) > 0:
        p2, p98 = np.percentile(valid, [2, 98])
        band_norm = np.clip((band - p2) / (p98 - p2), 0, 1)
        rgb_display[i] = band_norm

rgb_display = np.transpose(rgb_display, (1, 2, 0))

# NoData to white
nodata_mask = np.all(rgb_cropped <= 0, axis=0)
rgb_display[nodata_mask] = [1.0, 1.0, 1.0]

print(f"  NoData pixels total: {nodata_mask.sum()} ({100*nodata_mask.sum()/nodata_mask.size:.2f}%)")

# MINIMAL PLOT - No title, no axis, no extras
fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
ax.imshow(rgb_display)
ax.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Full image, no margins
plt.savefig('results/TEST_minimal_rgb.png', dpi=150, bbox_inches='tight', pad_inches=0)
plt.close()

print("\n✅ Saved: results/TEST_minimal_rgb.png")
print("   This has NO title, NO layout - just raw RGB")
print("   If black box still appears, it's in the DATA itself!")
