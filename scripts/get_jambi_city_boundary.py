"""
Get Jambi City administrative boundary from GeoBoundaries
"""
import geopandas as gpd
import matplotlib.pyplot as plt

print("="*80)
print("LOADING JAMBI CITY ADMINISTRATIVE BOUNDARY")
print("="*80)

# Option 1: Try GeoBoundaries (ADM2 level - city/regency)
print("\nTrying GeoBoundaries (ADM2 - City/Regency level)...")

try:
    # GeoBoundaries Indonesia ADM2 (cities/regencies)
    url = "https://www.geoboundaries.org/api/current/gbOpen/IDN/ADM2/"

    import requests
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        download_url = data['gjDownloadURL']

        print(f"  Downloading from: {download_url[:60]}...")
        gdf = gpd.read_file(download_url)

        print(f"  Total admin units: {len(gdf)}")
        print(f"  Columns: {list(gdf.columns)}")

        # Try to find Jambi City
        # Common name variations: "Jambi", "Kota Jambi", "Jambi City"
        jambi_keywords = ['jambi', 'kota jambi', 'jambi city']

        # Search in common name columns
        name_cols = [col for col in gdf.columns if 'name' in col.lower() or 'nama' in col.lower()]
        print(f"\n  Name columns found: {name_cols}")

        jambi_city = None
        for col in name_cols:
            for keyword in jambi_keywords:
                matches = gdf[gdf[col].str.lower().str.contains(keyword, na=False)]
                if len(matches) > 0:
                    print(f"\n  ✅ Found in column '{col}':")
                    print(matches[name_cols].to_string())
                    jambi_city = matches.iloc[0:1]  # Take first match
                    break
            if jambi_city is not None:
                break

        if jambi_city is not None:
            # Save city boundary
            output_file = 'data/klhk/Jambi_City_Boundary.geojson'
            jambi_city.to_file(output_file, driver='GeoJSON')
            print(f"\n✅ Saved: {output_file}")
            print(f"   Bounds: {jambi_city.total_bounds}")

            # Visualize
            fig, ax = plt.subplots(figsize=(10, 10))
            jambi_city.boundary.plot(ax=ax, color='red', linewidth=3, label='Jambi City Boundary')
            ax.set_title('Jambi City Administrative Boundary', fontsize=14, fontweight='bold')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.savefig('results/jambi_city_boundary.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✅ Saved: results/jambi_city_boundary.png")

        else:
            print("\n❌ Jambi City not found in GeoBoundaries ADM2")
            print("   Available cities (sample):")
            for col in name_cols[:1]:  # Show sample from first name column
                print(gdf[col].head(20).to_string())

    else:
        print(f"  ❌ Failed to access GeoBoundaries API: {response.status_code}")

except Exception as e:
    print(f"  ❌ Error: {e}")

# Option 2: Fallback - filter KLHK polygons near city center
print("\n" + "="*80)
print("FALLBACK: Creating boundary from KLHK polygons near city center")
print("="*80)

try:
    from shapely.geometry import Point, box
    import numpy as np

    # Load KLHK data
    klhk_gdf = gpd.read_file('data/klhk/KLHK_PL2024_Jambi_Full_WithGeometry.geojson')

    # Jambi City center
    city_center = Point(103.607254, -1.609972)

    # Create buffer around center (0.15 degrees ≈ 15-17 km)
    city_buffer = city_center.buffer(0.15)

    # Filter polygons that intersect with buffer
    city_polygons = klhk_gdf[klhk_gdf.intersects(city_buffer)]

    print(f"\n  KLHK polygons near city center: {len(city_polygons)}")

    # Dissolve to single boundary
    city_boundary_klhk = city_polygons.dissolve()

    # Save
    output_file = 'data/klhk/Jambi_City_KLHK_Boundary.geojson'
    city_boundary_klhk.to_file(output_file, driver='GeoJSON')
    print(f"✅ Saved: {output_file}")
    print(f"   Bounds: {city_boundary_klhk.total_bounds}")

    # Visualize
    fig, ax = plt.subplots(figsize=(10, 10))
    city_boundary_klhk.boundary.plot(ax=ax, color='blue', linewidth=2, label='City Boundary (KLHK polygons)')
    city_polygons.plot(ax=ax, facecolor='lightblue', edgecolor='darkblue', alpha=0.5)
    ax.plot(city_center.x, city_center.y, 'r*', markersize=20, label='City Center')
    ax.set_title('Jambi City Boundary from KLHK Polygons', fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig('results/jambi_city_klhk_boundary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: results/jambi_city_klhk_boundary.png")

except Exception as e:
    print(f"  ❌ Error: {e}")

print("\n" + "="*80)
print("DONE! Check the saved GeoJSON files and boundary visualizations")
print("="*80)
