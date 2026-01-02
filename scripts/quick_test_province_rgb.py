"""Quick test: Regenerate ONLY province RGB with fix"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.data_loader import load_sentinel2_tiles
from scripts.generate_qualitative_FINAL import create_rgb_composite
import geopandas as gpd

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

print("Generating province RGB with FIXED normalization...")
create_rgb_composite(
    sentinel2_bands, s2_profile, province_boundary,
    'results/TEST_province_rgb_FIXED.png',
    'Sentinel-2 RGB - Jambi Province (NoData Fix Applied)'
)

print("\n✅ Saved: results/TEST_province_rgb_FIXED.png")
print("   NoData areas should now be WHITE, not black!")
