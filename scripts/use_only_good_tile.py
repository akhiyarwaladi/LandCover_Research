"""Use ONLY Tile 2 (the one with actual data) - skip the broken tiles"""
import numpy as np
import matplotlib.pyplot as plt
import rasterio

print("Loading ONLY Tile 2 (the good tile with actual data)...")

# Use ONLY Tile 2 - it has 42% valid data (best of all tiles)
GOOD_TILE = 'data/sentinel/S2_jambi_2024_20m_AllBands-0000000000-0000010496.tif'

with rasterio.open(GOOD_TILE) as src:
    print(f"Tile shape: {src.shape}")
    print(f"Tile bounds: {src.bounds}")

    # Read RGB bands (1=Blue, 2=Green, 3=Red in 0-indexed)
    blue = src.read(1)
    green = src.read(2)
    red = src.read(3)

    rgb = np.stack([red, green, blue], axis=0)
    print(f"RGB shape: {rgb.shape}")

# Check NaN
nan_mask = np.any(np.isnan(rgb), axis=0)
print(f"NaN pixels: {nan_mask.sum()} ({100*nan_mask.sum()/nan_mask.size:.1f}%)")

# Normalize
nodata_mask = np.any(np.isnan(rgb), axis=0) | np.any(rgb <= 0, axis=0)
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

# Crop to valid data only
print("\nCropping to valid data bounds...")
valid_mask = ~nodata_mask
rows = np.any(valid_mask, axis=1)
cols = np.any(valid_mask, axis=0)
rmin, rmax = np.where(rows)[0][[0, -1]]
cmin, cmax = np.where(cols)[0][[0, -1]]

rgb_display = rgb_display[rmin:rmax+1, cmin:cmax+1]
print(f"Cropped size: {rgb_display.shape}")

# Save
print("\nGenerating visualization...")
fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
fig.patch.set_facecolor('white')
ax.patch.set_facecolor('white')
ax.imshow(rgb_display)
ax.axis('off')
title_obj = ax.set_title('Sentinel-2 RGB - Jambi (Tile 2 Only - Good Data)',
                         fontsize=14, fontweight='bold', pad=15)
title_obj.set_bbox(dict(facecolor='none', edgecolor='none'))

plt.tight_layout(pad=0.5)
plt.savefig('results/RGB_TILE2_ONLY.png', dpi=300,
            bbox_inches='tight', pad_inches=0.2,
            facecolor='white', edgecolor='none')
plt.close()

print("\n✅ Saved: results/RGB_TILE2_ONLY.png")
print("   Using ONLY the good tile - NO broken tiles with NaN!")
print("\n⚠️  This shows only PART of Jambi Province (the part with satellite coverage)")
print("    The other 3 tiles are mostly empty - they may need to be re-downloaded from GEE")
