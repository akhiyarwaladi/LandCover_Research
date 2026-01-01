import numpy as np
import sys
sys.path.insert(0, '.')
from modules.data_loader import load_klhk_data, load_sentinel2_tiles, CLASS_NAMES
from modules.preprocessor import rasterize_klhk

print('Loading data...')
klhk_gdf = load_klhk_data('data/klhk/KLHK_PL2024_Jambi_Full_WithGeometry.geojson', verbose=False)
s2_bands, s2_profile = load_sentinel2_tiles([
    'data/sentinel/S2_jambi_2024_20m_AllBands-0000000000-0000000000.tif',
    'data/sentinel/S2_jambi_2024_20m_AllBands-0000000000-0000010496.tif',
    'data/sentinel/S2_jambi_2024_20m_AllBands-0000010496-0000000000.tif',
    'data/sentinel/S2_jambi_2024_20m_AllBands-0000010496-0000010496.tif'
], verbose=False)

print('Rasterizing ground truth...')
ground_truth = rasterize_klhk(klhk_gdf, s2_profile, verbose=False)

print('\n' + '='*80)
print('GROUND TRUTH PIXEL DISTRIBUTION')
print('='*80)

unique, counts = np.unique(ground_truth, return_counts=True)
total_valid = (ground_truth >= 0).sum()

results = []
for cls, count in zip(unique, counts):
    if cls >= 0:
        name = CLASS_NAMES.get(cls, f'Class {cls}')
        pct = (count / total_valid) * 100
        results.append((pct, name, cls, count))

# Sort by percentage descending
results.sort(reverse=True)

print(f'\nTotal valid pixels: {total_valid:,}\n')
for pct, name, cls, count in results:
    print(f'{name:20s} ({cls}): {count:12,} pixels ({pct:6.2f}%)')
