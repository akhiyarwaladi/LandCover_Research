#!/usr/bin/env python3
import geopandas as gpd

# Check batch 3 and batch 10
batch3 = gpd.read_file('data/klhk/batches/batch_003_offset_2000_clean.geojson')
batch10 = gpd.read_file('data/klhk/batches/batch_010_offset_9000_clean.geojson')
batch20 = gpd.read_file('data/klhk/batches/batch_020_offset_19000_clean.geojson')

print('Batch 3 (offset 2000):')
print(f'  Features: {len(batch3)}')
print(f'  OBJECTID range: {batch3["OBJECTID"].min()} - {batch3["OBJECTID"].max()}')
print(f'  Sample IDs: {sorted(batch3["OBJECTID"].astype(int).unique())[:10]}')

print('\nBatch 10 (offset 9000):')
print(f'  Features: {len(batch10)}')
print(f'  OBJECTID range: {batch10["OBJECTID"].min()} - {batch10["OBJECTID"].max()}')
print(f'  Sample IDs: {sorted(batch10["OBJECTID"].astype(int).unique())[:10]}')

print('\nBatch 20 (offset 19000):')
print(f'  Features: {len(batch20)}')
print(f'  OBJECTID range: {batch20["OBJECTID"].min()} - {batch20["OBJECTID"].max()}')
print(f'  Sample IDs: {sorted(batch20["OBJECTID"].astype(int).unique())[:10]}')

# Check if they're identical
common_3_10 = set(batch3['OBJECTID']) & set(batch10['OBJECTID'])
common_3_20 = set(batch3['OBJECTID']) & set(batch20['OBJECTID'])
common_10_20 = set(batch10['OBJECTID']) & set(batch20['OBJECTID'])

print(f'\nCommon OBJECTIDs between batches:')
print(f'  Batch 3 & 10: {len(common_3_10)} / 1000')
print(f'  Batch 3 & 20: {len(common_3_20)} / 1000')
print(f'  Batch 10 & 20: {len(common_10_20)} / 1000')

if len(common_3_10) == 1000:
    print('\n❌ ALL BATCHES ARE IDENTICAL!')
    print('  The resultOffset parameter does NOT work with f=kmz format')
else:
    print('\n✅ Batches contain different data')
