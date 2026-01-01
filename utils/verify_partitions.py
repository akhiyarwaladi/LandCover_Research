#!/usr/bin/env python3
import geopandas as gpd

p1 = gpd.read_file('data/klhk/partitions/partition_001_oid_182_68865_clean.geojson')
p2 = gpd.read_file('data/klhk/partitions/partition_002_oid_68866_69865_clean.geojson')
p3 = gpd.read_file('data/klhk/partitions/partition_003_oid_69866_70865_clean.geojson')

print(f'Partition 1: {len(p1)} features, OBJECTID {p1["OBJECTID"].min()}-{p1["OBJECTID"].max()}')
print(f'Partition 2: {len(p2)} features, OBJECTID {p2["OBJECTID"].min()}-{p2["OBJECTID"].max()}')
print(f'Partition 3: {len(p3)} features, OBJECTID {p3["OBJECTID"].min()}-{p3["OBJECTID"].max()}')

common_1_2 = set(p1['OBJECTID']) & set(p2['OBJECTID'])
common_1_3 = set(p1['OBJECTID']) & set(p3['OBJECTID'])
common_2_3 = set(p2['OBJECTID']) & set(p3['OBJECTID'])

print(f'\nOverlap check:')
print(f'  P1 & P2: {len(common_1_2)} common IDs (should be 0)')
print(f'  P1 & P3: {len(common_1_3)} common IDs (should be 0)')
print(f'  P2 & P3: {len(common_2_3)} common IDs (should be 0)')

if len(common_1_2) == 0 and len(common_1_3) == 0 and len(common_2_3) == 0:
    print('\n✅ SUCCESS! All partitions have unique data!')
else:
    print('\n❌ WARNING: Partitions have overlapping data')
