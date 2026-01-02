"""
Diagnose why province has NoData but city doesn't
"""
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box

# Load province boundary
KLHK_PATH = 'data/klhk/KLHK_PL2024_Jambi_Full_WithGeometry.geojson'
province_gdf = gpd.read_file(KLHK_PATH)
province_dissolved = province_gdf.dissolve()

# Get bounds
bounds = province_dissolved.total_bounds
minx, miny, maxx, maxy = bounds

# Create bounding box (what crop function uses)
bbox = box(minx, miny, maxx, maxy)
bbox_gdf = gpd.GeoDataFrame({'geometry': [bbox]}, crs=province_dissolved.crs)

# Create city boundary for comparison
JAMBI_CITY_CENTER = (-1.609972, 103.607254)
JAMBI_CITY_EXTENT = 0.15
city_minx = JAMBI_CITY_CENTER[1] - JAMBI_CITY_EXTENT
city_maxx = JAMBI_CITY_CENTER[1] + JAMBI_CITY_EXTENT
city_miny = JAMBI_CITY_CENTER[0] - JAMBI_CITY_EXTENT
city_maxy = JAMBI_CITY_CENTER[0] + JAMBI_CITY_EXTENT
city_box = box(city_minx, city_miny, city_maxx, city_maxy)
city_gdf = gpd.GeoDataFrame({'geometry': [city_box]}, crs='EPSG:4326')

print("="*80)
print("BOUNDARY ANALYSIS")
print("="*80)

print(f"\nProvince:")
print(f"  Bounds: {bounds}")
print(f"  Geometry type: {province_dissolved.geometry.iloc[0].geom_type}")
print(f"  Number of parts: {len(province_dissolved.geometry.iloc[0].geoms) if province_dissolved.geometry.iloc[0].geom_type == 'MultiPolygon' else 1}")

# Calculate coverage ratio
province_area = province_dissolved.geometry.area.iloc[0]
bbox_area = bbox.area
coverage_ratio = province_area / bbox_area

print(f"\n  Province area: {province_area:.6f}")
print(f"  Bounding box area: {bbox_area:.6f}")
print(f"  Coverage ratio: {coverage_ratio:.2%}")
print(f"  → {(1-coverage_ratio)*100:.2f}% of bbox is OUTSIDE province (NoData)")

print(f"\nCity:")
print(f"  Bounds: [{city_minx:.6f}, {city_miny:.6f}, {city_maxx:.6f}, {city_maxy:.6f}]")
print(f"  Geometry type: Rectangle (perfect box)")
print(f"  Coverage ratio: 100% (no NoData)")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Province
axes[0].set_title('Province: Irregular Shape\n→ BBox creates NoData areas',
                  fontsize=14, fontweight='bold')
bbox_gdf.boundary.plot(ax=axes[0], color='red', linewidth=3, label='Crop BBox (red)')
province_dissolved.boundary.plot(ax=axes[0], color='green', linewidth=2, label='Actual Province (green)')
axes[0].legend(fontsize=12)
axes[0].set_xlabel('Longitude')
axes[0].set_ylabel('Latitude')
axes[0].grid(True, alpha=0.3)

# City
axes[1].set_title('City: Perfect Rectangle\n→ No NoData',
                  fontsize=14, fontweight='bold', color='green')
city_gdf.boundary.plot(ax=axes[1], color='blue', linewidth=3, label='City BBox (perfect fit)')
axes[1].legend(fontsize=12)
axes[1].set_xlabel('Longitude')
axes[1].set_ylabel('Latitude')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/DIAGNOSIS_boundary_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n" + "="*80)
print("SOLUTION OPTIONS:")
print("="*80)
print("1. ACCEPT NoData as white (accurate representation)")
print("2. Don't crop province - use FULL satellite mosaic")
print("3. Create custom tighter boundary (manually adjust bbox)")
print("\n✅ Saved: results/DIAGNOSIS_boundary_comparison.png")
print("   Shows why province has NoData but city doesn't")
