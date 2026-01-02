"""Generate RGB using NEW Sentinel-2 data (Tiles 1 & 2 only)"""
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from modules.data_loader import load_sentinel2_tiles
import geopandas as gpd

print("Loading NEW Sentinel-2 data (Tiles 1 & 2 ONLY - skip broken tiles)...")

# Use ONLY the good tiles!
GOOD_TILES = [
    'data/sentinel_new/S2_jambi_2024_20m_AllBands-0000000000-0000000000.tif',  # Tile 1: 59.5% valid ✅
    'data/sentinel_new/S2_jambi_2024_20m_AllBands-0000000000-0000010496.tif',  # Tile 2: 52.5% valid ✅
    # Skip Tile 3 (20.7% valid) and Tile 4 (0% valid)
]

sentinel2_bands, s2_profile = load_sentinel2_tiles(GOOD_TILES, verbose=True)

# Extract RGB
red = sentinel2_bands[2]
green = sentinel2_bands[1]
blue = sentinel2_bands[0]
rgb = np.stack([red, green, blue], axis=0)

print(f"\nRGB shape: {rgb.shape}")

# Check NaN
nodata_mask = np.any(np.isnan(rgb), axis=0) | np.any(rgb <= 0, axis=0)
valid_pct = 100 * (~nodata_mask).sum() / nodata_mask.size
print(f"Valid pixels: {valid_pct:.1f}%")
print(f"NaN pixels: {100 - valid_pct:.1f}%")

# Normalize
rgb_display = np.zeros((rgb.shape[1], rgb.shape[2], 3), dtype=np.float32)

for i in range(3):
    band = rgb[i]
    valid_pixels = band[~nodata_mask]
    valid_pixels = valid_pixels[~np.isnan(valid_pixels)]

    if len(valid_pixels) > 0:
        p2, p98 = np.nanpercentile(valid_pixels, [2, 98])
        band_norm = np.clip((band - p2) / (p98 - p2), 0, 1)
        rgb_display[:, :, i] = band_norm

rgb_display[nodata_mask] = [1.0, 1.0, 1.0]
rgb_display = np.nan_to_num(rgb_display, nan=1.0)

# Crop to valid region
print("\nCropping to valid data...")
valid_mask = ~nodata_mask
rows = np.any(valid_mask, axis=1)
cols = np.any(valid_mask, axis=0)
if np.any(rows) and np.any(cols):
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rgb_display = rgb_display[rmin:rmax+1, cmin:cmax+1]
    print(f"Cropped size: {rgb_display.shape}")

# Save
print("\nGenerating visualization...")
fig, ax = plt.subplots(figsize=(14, 10), dpi=300)
fig.patch.set_facecolor('white')
ax.patch.set_facecolor('white')
ax.imshow(rgb_display)
ax.axis('off')
title_obj = ax.set_title('Sentinel-2 RGB - Jambi Province\n(NEW Dry Season Data - Jun-Sept 2024)',
                         fontsize=14, fontweight='bold', pad=15)
title_obj.set_bbox(dict(facecolor='none', edgecolor='none'))

plt.tight_layout(pad=0.5)
plt.savefig('results/RGB_NEW_DATA_final.png', dpi=300,
            bbox_inches='tight', pad_inches=0.2,
            facecolor='white', edgecolor='none')
plt.close()

print("\n" + "="*80)
print("✅ DONE!")
print("="*80)
print(f"\n✅ Saved: results/RGB_NEW_DATA_final.png")
print(f"   Using Tiles 1 & 2 only (good coverage)")
print(f"   Valid pixels: {valid_pct:.1f}%")
print(f"\n🎯 This should have MUCH less white/NaN areas!")
