"""
Get BOTH province and city boundaries from the SAME source (GeoBoundaries)
For consistency and proper alignment
"""
import geopandas as gpd
import matplotlib.pyplot as plt
import requests

print("="*80)
print("UNIFIED ADMINISTRATIVE BOUNDARIES")
print("Source: GeoBoundaries (consistent, official)")
print("="*80)

# ============================================================================
# 1. PROVINCE LEVEL (ADM1)
# ============================================================================
print("\n[1/2] Loading Jambi Province (ADM1 - Province level)...")

url_adm1 = "https://www.geoboundaries.org/api/current/gbOpen/IDN/ADM1/"
response = requests.get(url_adm1)

if response.status_code == 200:
    data = response.json()
    download_url = data['gjDownloadURL']

    print(f"  Downloading from: {download_url[:60]}...")
    gdf_provinces = gpd.read_file(download_url)

    print(f"  Total provinces: {len(gdf_provinces)}")

    # Find Jambi province
    name_col = 'shapeName'
    jambi_province = gdf_provinces[gdf_provinces[name_col].str.lower().str.contains('jambi', na=False)]

    if len(jambi_province) > 0:
        print(f"\n  ✅ Found Jambi Province!")
        print(f"     Name: {jambi_province.iloc[0][name_col]}")
        print(f"     Bounds: {jambi_province.total_bounds}")

        # Save
        jambi_province.to_file('data/klhk/Jambi_Province_Boundary_GeoBoundaries.geojson',
                               driver='GeoJSON')
        print(f"  ✅ Saved: data/klhk/Jambi_Province_Boundary_GeoBoundaries.geojson")
    else:
        print(f"  ❌ Jambi Province not found!")
        print(f"     Available provinces: {gdf_provinces[name_col].tolist()}")
else:
    print(f"  ❌ Failed to access GeoBoundaries ADM1: {response.status_code}")
    jambi_province = None

# ============================================================================
# 2. CITY LEVEL (ADM2)
# ============================================================================
print("\n[2/2] Loading Jambi City (ADM2 - City/Regency level)...")

url_adm2 = "https://www.geoboundaries.org/api/current/gbOpen/IDN/ADM2/"
response = requests.get(url_adm2)

if response.status_code == 200:
    data = response.json()
    download_url = data['gjDownloadURL']

    print(f"  Downloading from: {download_url[:60]}...")
    gdf_cities = gpd.read_file(download_url)

    print(f"  Total cities/regencies: {len(gdf_cities)}")

    # Find Kota Jambi
    name_col = 'shapeName'
    jambi_city = gdf_cities[gdf_cities[name_col].str.lower().str.contains('kota jambi', na=False)]

    if len(jambi_city) == 0:
        # Try without "kota"
        jambi_city = gdf_cities[gdf_cities[name_col].str.lower() == 'jambi']

    if len(jambi_city) > 0:
        # If multiple matches, take the one with "Kota" (city, not regency)
        if len(jambi_city) > 1:
            kota_matches = jambi_city[jambi_city[name_col].str.contains('Kota', na=False)]
            if len(kota_matches) > 0:
                jambi_city = kota_matches.iloc[0:1]
            else:
                jambi_city = jambi_city.iloc[0:1]

        print(f"\n  ✅ Found Jambi City!")
        print(f"     Name: {jambi_city.iloc[0][name_col]}")
        print(f"     Bounds: {jambi_city.total_bounds}")

        # Already saved in previous script, but save again with consistent naming
        jambi_city.to_file('data/klhk/Jambi_City_Boundary_GeoBoundaries.geojson',
                          driver='GeoJSON')
        print(f"  ✅ Saved: data/klhk/Jambi_City_Boundary_GeoBoundaries.geojson")
    else:
        print(f"  ❌ Jambi City not found!")
        jambi_city = None
else:
    print(f"  ❌ Failed to access GeoBoundaries ADM2: {response.status_code}")
    jambi_city = None

# ============================================================================
# 3. VISUALIZE BOTH TOGETHER
# ============================================================================
if jambi_province is not None and jambi_city is not None:
    print("\n" + "="*80)
    print("CREATING COMPARISON VISUALIZATION")
    print("="*80)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Left: Province
    jambi_province.boundary.plot(ax=axes[0], color='blue', linewidth=3)
    axes[0].set_title('Jambi Province\n(GeoBoundaries ADM1)',
                     fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    axes[0].grid(True, alpha=0.3)

    # Right: City within Province
    jambi_province.boundary.plot(ax=axes[1], color='blue', linewidth=2,
                                 label='Province', alpha=0.5)
    jambi_city.boundary.plot(ax=axes[1], color='red', linewidth=3,
                            label='City')
    jambi_city.plot(ax=axes[1], facecolor='pink', alpha=0.3)
    axes[1].set_title('Jambi City within Province\n(Both from GeoBoundaries)',
                     fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    axes[1].legend(fontsize=12)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/unified_admin_boundaries.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("✅ Saved: results/unified_admin_boundaries.png")

    # Statistics
    print("\n" + "="*80)
    print("BOUNDARY STATISTICS")
    print("="*80)
    print(f"\nProvince (ADM1):")
    print(f"  Bounds: {jambi_province.total_bounds}")
    print(f"  Area: {jambi_province.to_crs('EPSG:32748').area.iloc[0] / 1e6:.2f} km²")

    print(f"\nCity (ADM2):")
    print(f"  Bounds: {jambi_city.total_bounds}")
    print(f"  Area: {jambi_city.to_crs('EPSG:32748').area.iloc[0] / 1e6:.2f} km²")
    print(f"  % of Province: {100 * jambi_city.to_crs('EPSG:32748').area.iloc[0] / jambi_province.to_crs('EPSG:32748').area.iloc[0]:.2f}%")

print("\n" + "="*80)
print("DONE!")
print("="*80)
print("\n✅ Both boundaries from SAME source: GeoBoundaries")
print("   - Province: data/klhk/Jambi_Province_Boundary_GeoBoundaries.geojson")
print("   - City: data/klhk/Jambi_City_Boundary_GeoBoundaries.geojson")
print("\n   Use these for consistency in your visualizations!")
