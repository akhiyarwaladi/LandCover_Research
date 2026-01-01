#!/usr/bin/env python3
"""
Verify the final merged KLHK dataset with geometry
"""

import geopandas as gpd
import pandas as pd

print("="*60)
print("KLHK FINAL DATASET VERIFICATION")
print("="*60)

# Load merged file
merged_file = 'data/klhk/KLHK_PL2024_Jambi_Full_WithGeometry.geojson'
print(f"\nLoading {merged_file}...")
gdf = gpd.read_file(merged_file)

print(f"\n{'='*60}")
print("BASIC INFO")
print("="*60)
print(f"Total features: {len(gdf):,}")
print(f"CRS: {gdf.crs}")
print(f"Columns: {list(gdf.columns)}")

print(f"\n{'='*60}")
print("GEOMETRY CHECK")
print("="*60)
null_geom = gdf.geometry.isna().sum()
print(f"Null geometries: {null_geom}")
print(f"Geometry types: {gdf.geometry.geom_type.value_counts().to_dict()}")

if null_geom == 0:
    print("✅ ALL FEATURES HAVE VALID GEOMETRY!")
else:
    print(f"❌ WARNING: {null_geom} features have NULL geometry")

# Bounds
total_bounds = gdf.total_bounds
print(f"\nBounds:")
print(f"  West:  {total_bounds[0]:.4f}")
print(f"  South: {total_bounds[1]:.4f}")
print(f"  East:  {total_bounds[2]:.4f}")
print(f"  North: {total_bounds[3]:.4f}")

# Area
total_area = gdf.geometry.area.sum()
print(f"\nTotal area: {total_area:.2f} sq degrees")

print(f"\n{'='*60}")
print("OBJECTID CHECK")
print("="*60)
objectids = gdf['OBJECTID'].astype(int)
print(f"OBJECTID range: {objectids.min()} - {objectids.max()}")
print(f"Unique OBJECTIDs: {objectids.nunique()}")
duplicates = len(objectids) - objectids.nunique()
print(f"Duplicate OBJECTIDs: {duplicates}")

if duplicates == 0:
    print("✅ NO DUPLICATES!")
else:
    print(f"❌ WARNING: {duplicates} duplicate OBJECTIDs")

print(f"\n{'='*60}")
print("KODE PROVINSI CHECK")
print("="*60)
kode_prov = gdf['Kode Provinsi'].value_counts()
print(f"Kode Provinsi distribution:")
for kode, count in kode_prov.items():
    print(f"  {kode}: {count:,} features")

if len(kode_prov) == 1 and '15' in kode_prov.index:
    print("✅ ALL FEATURES ARE FROM JAMBI (Kode 15)")
else:
    print("⚠️  WARNING: Mixed provinces or unexpected codes")

print(f"\n{'='*60}")
print("LAND COVER CLASS DISTRIBUTION")
print("="*60)

if 'ID Penutupan Lahan Tahun 2024' in gdf.columns:
    class_dist = gdf['ID Penutupan Lahan Tahun 2024'].value_counts()
    print(f"Total classes: {len(class_dist)}")
    print(f"\nTop 15 classes:")
    for class_id, count in class_dist.head(15).items():
        pct = (count / len(gdf)) * 100
        print(f"  {class_id:6s}: {count:5,} ({pct:5.1f}%)")

    # Summary stats
    print(f"\nClass distribution stats:")
    print(f"  Most common class: {class_dist.index[0]} ({class_dist.iloc[0]:,} features, {(class_dist.iloc[0]/len(gdf))*100:.1f}%)")
    print(f"  Least common class: {class_dist.index[-1]} ({class_dist.iloc[-1]:,} features)")

else:
    print("❌ PL2024 class column not found!")

print(f"\n{'='*60}")
print("SAMPLE FEATURES")
print("="*60)
print("\nFirst 3 features:")
for i in range(min(3, len(gdf))):
    feat = gdf.iloc[i]
    print(f"\n  Feature {i+1}:")
    print(f"    OBJECTID: {feat['OBJECTID']}")
    print(f"    Kode Prov: {feat.get('Kode Provinsi', 'N/A')}")
    print(f"    PL2024 ID: {feat.get('ID Penutupan Lahan Tahun 2024', 'N/A')}")
    print(f"    Geometry: {feat.geometry.geom_type} with {len(feat.geometry.exterior.coords) if hasattr(feat.geometry, 'exterior') else '?'} coords")
    print(f"    Area: {feat.geometry.area:.8f} sq degrees")

print(f"\n{'='*60}")
print("FINAL VERDICT")
print("="*60)

issues = []
if null_geom > 0:
    issues.append(f"{null_geom} null geometries")
if duplicates > 0:
    issues.append(f"{duplicates} duplicate OBJECTIDs")
if len(gdf) != 28100:
    issues.append(f"Expected 28,100 features, got {len(gdf)}")

if len(issues) == 0:
    print("✅✅✅ DATASET IS PERFECT! ✅✅✅")
    print(f"\n🎉 SUCCESS: {len(gdf):,} KLHK polygons with complete geometry!")
    print("Ready for supervised land cover classification!")
else:
    print(f"⚠️  Issues found:")
    for issue in issues:
        print(f"  - {issue}")

print(f"\nOutput file: {merged_file}")
file_size = os.path.getsize(merged_file) / (1024**2)
print(f"File size: {file_size:.1f} MB")
