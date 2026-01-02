"""Debug: Check actual RGB data values"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from modules.data_loader import load_sentinel2_tiles

print("Loading Sentinel-2...")
SENTINEL2_TILES = [
    'data/sentinel/S2_jambi_2024_20m_AllBands-0000000000-0000000000.tif',
    'data/sentinel/S2_jambi_2024_20m_AllBands-0000000000-0000010496.tif',
    'data/sentinel/S2_jambi_2024_20m_AllBands-0000010496-0000000000.tif',
    'data/sentinel/S2_jambi_2024_20m_AllBands-0000010496-0000010496.tif'
]

sentinel2_bands, s2_profile = load_sentinel2_tiles(SENTINEL2_TILES, verbose=False)

print(f"\nData shape: {sentinel2_bands.shape}")
print(f"Data dtype: {sentinel2_bands.dtype}")

# Check each RGB band
for i, name in enumerate(['Blue (B2)', 'Green (B3)', 'Red (B4)']):
    band = sentinel2_bands[i]
    print(f"\n{name}:")
    print(f"  Min: {np.nanmin(band)}")
    print(f"  Max: {np.nanmax(band)}")
    print(f"  Mean: {np.nanmean(band):.4f}")
    print(f"  Zeros: {np.sum(band == 0)} ({100*np.sum(band == 0)/band.size:.2f}%)")
    print(f"  NaN: {np.sum(np.isnan(band))}")
    print(f"  Valid (>0): {np.sum(band > 0)} ({100*np.sum(band > 0)/band.size:.2f}%)")

# Create simple visualization WITHOUT any normalization
red = sentinel2_bands[2]
green = sentinel2_bands[1]
blue = sentinel2_bands[0]

# Just stack and save raw values (scaled 0-255)
rgb_raw = np.stack([red, green, blue], axis=-1)
print(f"\nRGB raw shape: {rgb_raw.shape}")
print(f"RGB raw min: {np.nanmin(rgb_raw)}, max: {np.nanmax(rgb_raw)}")

# Simple save using matplotlib
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 10))
# Scale to 0-1 using global min/max (simple linear stretch)
rgb_simple = np.clip(rgb_raw * 5, 0, 1)  # Multiply by 5 for brightness
ax.imshow(rgb_simple)
ax.axis('off')
plt.savefig('results/DEBUG_rgb_simple.png', dpi=100, bbox_inches='tight')
plt.close()

print("\n✅ Saved: results/DEBUG_rgb_simple.png (simple linear stretch)")
print("If this is also black, the data itself has issues!")
