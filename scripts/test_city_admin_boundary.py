"""Test city visualization with actual administrative boundary"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.data_loader import load_sentinel2_tiles
from scripts.generate_qualitative_FINAL import create_rgb_composite
import geopandas as gpd

print("Loading data...")
SENTINEL2_TILES = [
    'data/sentinel/S2_jambi_2024_20m_AllBands-0000000000-0000000000.tif',
    'data/sentinel/S2_jambi_2024_20m_AllBands-0000000000-0000010496.tif',
    'data/sentinel/S2_jambi_2024_20m_AllBands-0000010496-0000000000.tif',
    'data/sentinel/S2_jambi_2024_20m_AllBands-0000010496-0000010496.tif'
]

sentinel2_bands, s2_profile = load_sentinel2_tiles(SENTINEL2_TILES, verbose=False)

# Load actual city boundary
city_boundary = gpd.read_file('data/klhk/Jambi_City_Boundary.geojson')
print(f"City boundary loaded: {city_boundary.total_bounds}")

print("\nGenerating city RGB with ADMINISTRATIVE boundary...")
create_rgb_composite(
    sentinel2_bands, s2_profile, city_boundary,
    'results/TEST_city_admin_boundary.png',
    'Sentinel-2 RGB - Jambi City (Administrative Boundary)',
    skip_crop=False
)

print("\n✅ Saved: results/TEST_city_admin_boundary.png")
print("   Uses actual 'Kota Jambi' administrative boundary (not rectangle!)")
