#!/usr/bin/env python3
"""
Find Sub-Districts in Empty Corners
====================================

Analyzes which sub-districts can fill the top-left and bottom-right
empty corners naturally!

Author: Claude Sonnet 4.5
Date: 2026-01-03
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import geopandas as gpd
from shapely.geometry import box, Point
from shapely.ops import unary_union

print("="*80)
print("FINDING SUB-DISTRICTS FOR EMPTY CORNERS")
print("="*80)

# Load all Jambi sub-districts
admin = gpd.read_file('data/gadm_indonesia_subdistricts.geojson')
jambi_all = admin[admin['NAME_1'] == 'Jambi'].copy()

print(f"\n✓ Total Jambi sub-districts: {len(jambi_all)}")

# Load current selection
current_boundary = gpd.read_file('data/jambi_subdistrict_28km_boundary.geojson')
current_geom = unary_union(current_boundary.geometry)
current_bounds = current_geom.bounds

print(f"\nCurrent selection: {len(current_boundary)} sub-districts")
print(f"Current bounds:")
print(f"  Lon: {current_bounds[0]:.4f} to {current_bounds[2]:.4f}")
print(f"  Lat: {current_bounds[1]:.4f} to {current_bounds[3]:.4f}")

# Define corner areas to check
# Top-left corner
top_left_box = box(
    current_bounds[0] - 0.15,  # Extend west
    current_bounds[3] - 0.15,  # From top
    current_bounds[0] + 0.15,  # To east
    current_bounds[3] + 0.15   # Extend north
)

# Bottom-right corner
bottom_right_box = box(
    current_bounds[2] - 0.15,  # From west
    current_bounds[1] - 0.15,  # Extend south
    current_bounds[2] + 0.15,  # Extend east
    current_bounds[1] + 0.15   # To north
)

print(f"\n" + "-"*80)
print("SEARCHING FOR CORNER SUB-DISTRICTS")
print("-"*80)

# Find sub-districts in top-left corner
print(f"\n🔍 TOP-LEFT CORNER:")
top_left_candidates = []
for idx, row in jambi_all.iterrows():
    if row.geometry.intersects(top_left_box):
        # Check if not already in current selection
        name = row['NAME_3']
        if name not in current_boundary['name'].values:
            centroid = row.geometry.centroid
            top_left_candidates.append({
                'name': name,
                'district': row['NAME_2'],
                'geometry': row.geometry,
                'centroid_lon': centroid.x,
                'centroid_lat': centroid.y
            })

if top_left_candidates:
    print(f"  Found {len(top_left_candidates)} candidate(s):")
    for c in top_left_candidates:
        print(f"    - {c['name']:30s} ({c['district']:15s}) at ({c['centroid_lon']:.3f}, {c['centroid_lat']:.3f})")
else:
    print(f"  ❌ No candidates found")

# Find sub-districts in bottom-right corner
print(f"\n🔍 BOTTOM-RIGHT CORNER:")
bottom_right_candidates = []
for idx, row in jambi_all.iterrows():
    if row.geometry.intersects(bottom_right_box):
        name = row['NAME_3']
        if name not in current_boundary['name'].values:
            centroid = row.geometry.centroid
            bottom_right_candidates.append({
                'name': name,
                'district': row['NAME_2'],
                'geometry': row.geometry,
                'centroid_lon': centroid.x,
                'centroid_lat': centroid.y
            })

if bottom_right_candidates:
    print(f"  Found {len(bottom_right_candidates)} candidate(s):")
    for c in bottom_right_candidates:
        print(f"    - {c['name']:30s} ({c['district']:15s}) at ({c['centroid_lon']:.3f}, {c['centroid_lat']:.3f})")
else:
    print(f"  ❌ No candidates found")

# Recommendations
print(f"\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

all_recommendations = []

if top_left_candidates:
    print(f"\n📍 For TOP-LEFT corner, consider adding:")
    for c in top_left_candidates[:3]:  # Top 3
        print(f"   ✓ {c['name']:30s} ({c['district']})")
        all_recommendations.append(c['name'])

if bottom_right_candidates:
    print(f"\n📍 For BOTTOM-RIGHT corner, consider adding:")
    for c in bottom_right_candidates[:3]:  # Top 3
        print(f"   ✓ {c['name']:30s} ({c['district']})")
        all_recommendations.append(c['name'])

if all_recommendations:
    print(f"\n💡 SUGGESTED SUB-DISTRICTS TO ADD:")
    for name in all_recommendations:
        print(f"   • {name}")

    # Test combined area
    print(f"\n" + "-"*80)
    print("TESTING COMBINED SELECTION")
    print("-"*80)

    # Get geometries for recommended sub-districts
    recommended_geoms = []
    for idx, row in jambi_all.iterrows():
        if row['NAME_3'] in all_recommendations:
            recommended_geoms.append(row.geometry)

    # Combine with current
    all_geoms = list(current_boundary.geometry) + recommended_geoms
    combined_geom = unary_union(all_geoms)
    combined_area = combined_geom.area * (111**2)
    combined_bounds = combined_geom.bounds

    current_area = current_geom.area * (111**2)

    print(f"\nCurrent area: ~{current_area:.0f} km²")
    print(f"After adding {len(all_recommendations)} sub-districts: ~{combined_area:.0f} km²")
    print(f"Increase: {((combined_area - current_area) / current_area * 100):.1f}%")

    print(f"\nNew bounds:")
    print(f"  Lon: {combined_bounds[0]:.4f} to {combined_bounds[2]:.4f}")
    print(f"  Lat: {combined_bounds[1]:.4f} to {combined_bounds[3]:.4f}")

else:
    print(f"\n⚠️  No additional sub-districts found for corners")
    print(f"   Consider using buffer method instead")

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE!")
print("="*80)
