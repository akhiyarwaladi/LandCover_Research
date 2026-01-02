"""Check each Sentinel-2 tile for NaN values"""
import rasterio
import numpy as np
import matplotlib.pyplot as plt

TILES = [
    'data/sentinel/S2_jambi_2024_20m_AllBands-0000000000-0000000000.tif',
    'data/sentinel/S2_jambi_2024_20m_AllBands-0000000000-0000010496.tif',
    'data/sentinel/S2_jambi_2024_20m_AllBands-0000010496-0000000000.tif',
    'data/sentinel/S2_jambi_2024_20m_AllBands-0000010496-0000010496.tif'
]

print("="*80)
print("CHECKING INDIVIDUAL SENTINEL-2 TILES")
print("="*80)

tile_data = []
for i, tile_path in enumerate(TILES):
    print(f"\nTile {i+1}: {tile_path.split('/')[-1]}")

    with rasterio.open(tile_path) as src:
        # Read first band (Blue)
        band1 = src.read(1)

        print(f"  Shape: {band1.shape}")
        print(f"  Min: {np.nanmin(band1):.4f}, Max: {np.nanmax(band1):.4f}")
        print(f"  NaN pixels: {np.sum(np.isnan(band1))} ({100*np.sum(np.isnan(band1))/band1.size:.2f}%)")
        print(f"  Zero pixels: {np.sum(band1 == 0)} ({100*np.sum(band1 == 0)/band1.size:.2f}%)")
        print(f"  Valid (>0): {np.sum(band1 > 0)} ({100*np.sum(band1 > 0)/band1.size:.2f}%)")

        # Read RGB bands for visualization
        red = src.read(3)
        green = src.read(2)
        blue = src.read(1)

        tile_data.append({
            'name': f'Tile {i+1}',
            'rgb': np.stack([red, green, blue], axis=-1),
            'nan_pct': 100*np.sum(np.isnan(band1))/band1.size
        })

# Create small visualization (downsample to 200x200 per tile)
print("\n" + "="*80)
print("Creating small diagnostic visualization...")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()

for idx, tile in enumerate(tile_data):
    rgb = tile['rgb']

    # Downsample to 200x200
    h, w = rgb.shape[0], rgb.shape[1]
    step_h = max(1, h // 200)
    step_w = max(1, w // 200)
    rgb_small = rgb[::step_h, ::step_w, :]

    # Replace NaN with 0 for display
    rgb_small = np.nan_to_num(rgb_small, nan=0.0)

    # Simple stretch
    rgb_display = np.clip(rgb_small * 5, 0, 1)

    axes[idx].imshow(rgb_display)
    axes[idx].set_title(f"{tile['name']} ({tile['nan_pct']:.1f}% NaN)",
                       fontsize=12, fontweight='bold')
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig('results/DIAGNOSTIC_sentinel_tiles.png', dpi=100, bbox_inches='tight')
plt.close()

print("\n✅ Saved: results/DIAGNOSTIC_sentinel_tiles.png")
print("   Low-res view of each tile - check for gaps/missing data")
