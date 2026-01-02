"""Generate RGB cropped to VALID DATA only (no white borders)"""
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from modules.data_loader import load_sentinel2_tiles

print("Loading Sentinel-2...")
TILES = [
    'data/sentinel/S2_jambi_2024_20m_AllBands-0000000000-0000000000.tif',
    'data/sentinel/S2_jambi_2024_20m_AllBands-0000000000-0000010496.tif',
    'data/sentinel/S2_jambi_2024_20m_AllBands-0000010496-0000000000.tif',
    'data/sentinel/S2_jambi_2024_20m_AllBands-0000010496-0000010496.tif'
]

sentinel2_bands, s2_profile = load_sentinel2_tiles(TILES, verbose=False)

# Extract RGB
red = sentinel2_bands[2]
green = sentinel2_bands[1]
blue = sentinel2_bands[0]
rgb = np.stack([red, green, blue], axis=0)

# Find valid data bounds
print("\nFinding valid data bounds...")
valid_mask = ~(np.any(np.isnan(rgb), axis=0) | np.any(rgb <= 0, axis=0))
rows = np.any(valid_mask, axis=1)
cols = np.any(valid_mask, axis=0)
rmin, rmax = np.where(rows)[0][[0, -1]]
cmin, cmax = np.where(cols)[0][[0, -1]]

print(f"  Full mosaic: {rgb.shape[1]} x {rgb.shape[2]}")
print(f"  Valid region: {rmax-rmin+1} x {cmax-cmin+1}")
print(f"  Cropping: rows {rmin}:{rmax+1}, cols {cmin}:{cmax+1}")

# Crop to valid region
rgb_cropped = rgb[:, rmin:rmax+1, cmin:cmax+1]

# Normalize
nodata_mask = np.any(np.isnan(rgb_cropped), axis=0) | np.any(rgb_cropped <= 0, axis=0)
rgb_display = np.zeros((rgb_cropped.shape[1], rgb_cropped.shape[2], 3), dtype=np.float32)

for i in range(3):
    band = rgb_cropped[i]
    valid_pixels = band[~nodata_mask]
    valid_pixels = valid_pixels[~np.isnan(valid_pixels)]

    if len(valid_pixels) > 0:
        p2, p98 = np.nanpercentile(valid_pixels, [2, 98])
        band_norm = np.clip((band - p2) / (p98 - p2), 0, 1)
        rgb_display[:, :, i] = band_norm

rgb_display[nodata_mask] = [1.0, 1.0, 1.0]
rgb_display = np.nan_to_num(rgb_display, nan=1.0)

# Save
print("\nGenerating visualization...")
fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
fig.patch.set_facecolor('white')
ax.patch.set_facecolor('white')
ax.imshow(rgb_display)
ax.axis('off')
title_obj = ax.set_title('Sentinel-2 RGB - Jambi Province (Valid Data Only)',
                         fontsize=14, fontweight='bold', pad=15)
title_obj.set_bbox(dict(facecolor='none', edgecolor='none'))

plt.tight_layout(pad=0.5)
plt.savefig('results/FINAL_province_rgb_clean.png', dpi=300,
            bbox_inches='tight', pad_inches=0.2,
            facecolor='white', edgecolor='none')
plt.close()

print("\n✅ Saved: results/FINAL_province_rgb_clean.png")
print("   Cropped to valid data - NO white borders!")
