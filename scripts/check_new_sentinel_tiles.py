"""Check new Sentinel-2 tiles for NaN content"""
import rasterio
import numpy as np
import matplotlib.pyplot as plt

NEW_TILES = [
    'data/sentinel_new/S2_jambi_2024_20m_AllBands-0000000000-0000000000.tif',
    'data/sentinel_new/S2_jambi_2024_20m_AllBands-0000000000-0000010496.tif',
    'data/sentinel_new/S2_jambi_2024_20m_AllBands-0000010496-0000000000.tif',
    'data/sentinel_new/S2_jambi_2024_20m_AllBands-0000010496-0000010496.tif'
]

print("="*80)
print("CHECKING NEW SENTINEL-2 TILES (DRY SEASON DATA)")
print("="*80)

tile_data = []
total_valid = 0
total_pixels = 0

for i, tile_path in enumerate(NEW_TILES):
    print(f"\n{'='*80}")
    print(f"Tile {i+1}: {tile_path.split('/')[-1]}")
    print(f"{'='*80}")

    with rasterio.open(tile_path) as src:
        # Read first band
        band1 = src.read(1)

        file_size_mb = tile_path.replace('data/sentinel_new/', '')

        nan_count = np.sum(np.isnan(band1))
        zero_count = np.sum(band1 == 0)
        valid_count = np.sum(band1 > 0)
        total_count = band1.size

        nan_pct = 100 * nan_count / total_count
        valid_pct = 100 * valid_count / total_count

        print(f"  Shape: {band1.shape}")
        print(f"  Total pixels: {total_count:,}")
        print(f"  Valid (>0): {valid_count:,} ({valid_pct:.2f}%)")
        print(f"  NaN: {nan_count:,} ({nan_pct:.2f}%)")
        print(f"  Zero: {zero_count:,} ({100*zero_count/total_count:.2f}%)")

        if valid_count > 0:
            print(f"  Data range: {np.nanmin(band1[band1 > 0]):.4f} - {np.nanmax(band1):.4f}")

        # Read RGB for preview
        red = src.read(3)
        green = src.read(2)
        blue = src.read(1)

        tile_data.append({
            'name': f'Tile {i+1}',
            'rgb': np.stack([red, green, blue], axis=-1),
            'valid_pct': valid_pct,
            'nan_pct': nan_pct
        })

        total_valid += valid_count
        total_pixels += total_count

# Overall statistics
print("\n" + "="*80)
print("OVERALL STATISTICS")
print("="*80)
print(f"Total pixels (all tiles): {total_pixels:,}")
print(f"Total valid pixels: {total_valid:,} ({100*total_valid/total_pixels:.2f}%)")
print(f"Total NaN pixels: {total_pixels - total_valid:,} ({100*(total_pixels-total_valid)/total_pixels:.2f}%)")

# Create visualization
print("\n" + "="*80)
print("Creating visualization...")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 14))
axes = axes.flatten()

for idx, tile in enumerate(tile_data):
    rgb = tile['rgb']

    # Downsample
    h, w = rgb.shape[0], rgb.shape[1]
    step_h = max(1, h // 300)
    step_w = max(1, w // 300)
    rgb_small = rgb[::step_h, ::step_w, :]

    # Replace NaN with white
    rgb_small = np.nan_to_num(rgb_small, nan=1.0)

    # Normalize
    rgb_display = np.clip(rgb_small * 5, 0, 1)

    axes[idx].imshow(rgb_display)
    axes[idx].set_title(f"{tile['name']}\nValid: {tile['valid_pct']:.1f}%, NaN: {tile['nan_pct']:.1f}%",
                       fontsize=12, fontweight='bold')
    axes[idx].axis('off')

plt.suptitle('NEW Sentinel-2 Tiles (Dry Season Jun-Sept 2024)',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('results/NEW_sentinel_tiles_check.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n✅ Saved: results/NEW_sentinel_tiles_check.png")

# Comparison with old data
print("\n" + "="*80)
print("COMPARISON: OLD vs NEW")
print("="*80)
print("OLD data (2024 full year, threshold 0.60):")
print("  Tile 1: 64% NaN")
print("  Tile 2: 58% NaN")
print("  Tile 3: 78.5% NaN")
print("  Tile 4: 100% NaN ⚠️")
print("  Overall: ~63% NaN")
print("\nNEW data (dry season, threshold 0.50):")
print(f"  Overall: {100*(total_pixels-total_valid)/total_pixels:.1f}% NaN")
print(f"  Valid coverage: {100*total_valid/total_pixels:.1f}% ✅")
