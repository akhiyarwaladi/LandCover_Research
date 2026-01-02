"""
Test province boundary to find the black box source
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.data_loader import load_sentinel2_tiles, load_klhk_data
from scripts.generate_qualitative_FINAL import crop_raster_to_boundary
import geopandas as gpd

# Load data
print("Loading data...")
KLHK_PATH = 'data/klhk/KLHK_PL2024_Jambi_Full_WithGeometry.geojson'
SENTINEL2_TILES = [
    'data/sentinel/S2_jambi_2024_20m_AllBands-0000000000-0000000000.tif',
    'data/sentinel/S2_jambi_2024_20m_AllBands-0000000000-0000010496.tif',
    'data/sentinel/S2_jambi_2024_20m_AllBands-0000010496-0000000000.tif',
    'data/sentinel/S2_jambi_2024_20m_AllBands-0000010496-0000010496.tif'
]

sentinel2_bands, s2_profile = load_sentinel2_tiles(SENTINEL2_TILES, verbose=False)
province_boundary = gpd.read_file(KLHK_PATH).dissolve()

print(f"Province bounds: {province_boundary.total_bounds}")

# Extract RGB
red = sentinel2_bands[2]
green = sentinel2_bands[1]
blue = sentinel2_bands[0]
rgb = np.stack([red, green, blue], axis=0)

# Crop to province
print("\nCropping to province boundary...")
rgb_cropped, crop_profile = crop_raster_to_boundary(rgb, s2_profile, province_boundary)
print(f"Cropped shape: {rgb_cropped.shape}")

# Analyze different regions
regions = {
    'Top-left (0:500, 0:500)': rgb_cropped[:, 0:500, 0:500],
    'Top-right (0:500, -500:)': rgb_cropped[:, 0:500, -500:],
    'Center': rgb_cropped[:,
                          rgb_cropped.shape[1]//2-250:rgb_cropped.shape[1]//2+250,
                          rgb_cropped.shape[2]//2-250:rgb_cropped.shape[2]//2+250],
}

print("\n" + "="*80)
print("REGION ANALYSIS:")
print("="*80)

for name, region in regions.items():
    nodata = np.all(region <= 0, axis=0)
    valid_pixels = region[0][region[0] > 0]

    print(f"\n{name}:")
    print(f"  Shape: {region.shape}")
    print(f"  NoData pixels: {nodata.sum()} ({100*nodata.sum()/nodata.size:.2f}%)")
    if len(valid_pixels) > 0:
        print(f"  Valid data range: {valid_pixels.min():.4f} - {valid_pixels.max():.4f}")
    else:
        print(f"  ⚠️  NO VALID DATA - ALL NODATA!")

# Create visualization showing NoData distribution
print("\n" + "="*80)
print("Creating NoData mask visualization...")
print("="*80)

nodata_mask = np.all(rgb_cropped <= 0, axis=0)
print(f"\nTotal NoData: {nodata_mask.sum()} pixels ({100*nodata_mask.sum()/nodata_mask.size:.2f}%)")

# Create a visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Left: NoData mask (white=data, black=nodata)
axes[0].imshow(~nodata_mask, cmap='gray')
axes[0].set_title('Data Coverage (White=Data, Black=NoData)', fontsize=14, fontweight='bold')
axes[0].axis('off')

# Right: RGB with NoData as red for visibility
rgb_display = np.ones_like(rgb_cropped, dtype=np.float32)
for i in range(3):
    band = rgb_cropped[i]
    valid = band[band > 0]
    if len(valid) > 0:
        p2, p98 = np.percentile(valid, [2, 98])
        band_norm = np.clip((band - p2) / (p98 - p2), 0, 1)
        rgb_display[i] = band_norm

rgb_display = np.transpose(rgb_display, (1, 2, 0))
rgb_display[nodata_mask] = [1.0, 0.0, 0.0]  # RED for NoData (for diagnosis)

axes[1].imshow(rgb_display)
axes[1].set_title('RGB with NoData as RED', fontsize=14, fontweight='bold')
axes[1].axis('off')

plt.tight_layout()
plt.savefig('results/TEST_province_nodata_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n✅ Saved: results/TEST_province_nodata_analysis.png")
print("   Left: Data coverage map (black areas = NoData)")
print("   Right: RGB with NoData shown in RED")
