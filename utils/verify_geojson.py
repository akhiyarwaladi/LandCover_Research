#!/usr/bin/env python3
"""
Verify the converted GeoJSON file has geometry and proper structure
"""

import geopandas as gpd
import json

# Read GeoJSON
print("Loading GeoJSON...")
gdf = gpd.read_file('data/klhk/KLHK_PL2024_Jambi_batch1.geojson')

print('✅ GeoJSON loaded successfully!')
print(f'Total features: {len(gdf)}')
print(f'CRS: {gdf.crs}')
print(f'Geometry type: {gdf.geometry.type.unique()}')
print(f'\nColumns: {list(gdf.columns)}')

# Check for NULL geometry
null_geom = gdf.geometry.isna().sum()
print(f'\nNull geometries: {null_geom}')

# Show sample
print(f'\n--- SAMPLE FEATURE ---')
sample = gdf.iloc[0]
print(f"Name: {sample['Name']}")
print(f"\nDescription:\n{sample['description'][:400]}")

# Check geometry
print(f'\n--- GEOMETRY CHECK ---')
print(f'First geometry type: {sample.geometry.geom_type}')
print(f'First geometry bounds: {sample.geometry.bounds}')
print(f'First geometry area: {sample.geometry.area:.8f} sq degrees')

# Count by geometry type
print(f'\n--- GEOMETRY TYPES ---')
print(gdf.geometry.geom_type.value_counts())

# Get total area
total_area = gdf.geometry.area.sum()
print(f'\nTotal area: {total_area:.2f} sq degrees')

print('\n✅ SUCCESS! File has valid geometry!')
