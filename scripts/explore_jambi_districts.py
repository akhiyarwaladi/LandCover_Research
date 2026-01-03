#!/usr/bin/env python3
"""
Explore Jambi Districts (Kabupaten/Kota) for Administrative Cropping
=====================================================================

Find district-level administrative boundaries around Jambi City.

Author: Claude Sonnet 4.5
Date: 2026-01-03
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point

print("="*80)
print("EXPLORING JAMBI DISTRICTS")
print("="*80)

# ============================================================================
# Load Level 2 Administrative Boundaries (Kabupaten/Kota)
# ============================================================================

print("\nLoading district-level boundaries from KLHK data...")

# Load KLHK data which has administrative info
klhk_path = 'data/klhk/KLHK_PL2024_Jambi_Full_WithGeometry.geojson'
klhk = gpd.read_file(klhk_path)

print(f"✓ Loaded {len(klhk):,} KLHK polygons")

# Check what administrative columns are available
print(f"\nAvailable columns:")
admin_cols = [col for col in klhk.columns if any(x in col.upper() for x in ['KAB', 'KOTA', 'NAMA', 'ADM', 'DAERAH', 'WILAYAH'])]
for col in admin_cols:
    print(f"  - {col}")

# Find unique districts
if 'Nama Kabupaten/Kota' in klhk.columns:
    districts = klhk['Nama Kabupaten/Kota'].unique()
    print(f"\n✓ Found {len(districts)} districts in Jambi Province:")
    for i, district in enumerate(sorted(districts), 1):
        count = (klhk['Nama Kabupaten/Kota'] == district).sum()
        print(f"  {i}. {district:30s} - {count:,} polygons")

    # Jambi City location
    jambi_city_center = Point(103.6167, -1.6000)

    # Calculate distance from each district to city center
    print(f"\n📍 Distances from Jambi City center (103.6167, -1.6000):")

    district_info = []
    for district in sorted(districts):
        subset = klhk[klhk['Nama Kabupaten/Kota'] == district]

        # Get district centroid
        district_geom = subset.unary_union
        centroid = district_geom.centroid

        # Calculate distance in degrees (approximate)
        dist_deg = jambi_city_center.distance(centroid)
        dist_km = dist_deg * 111  # Approximate conversion

        # Get bounds
        bounds = district_geom.bounds
        area_km2 = district_geom.area * (111**2)  # Rough approximation

        district_info.append({
            'name': district,
            'polygons': len(subset),
            'distance_km': dist_km,
            'area_km2': area_km2,
            'geometry': district_geom
        })

        print(f"  {district:30s} - {dist_km:5.1f} km away, ~{area_km2:6.0f} km²")

    # ========================================================================
    # Suggestion: Districts near city
    # ========================================================================

    print(f"\n{'='*80}")
    print("SUGGESTIONS FOR ADMINISTRATIVE CROPPING")
    print("="*80)

    # Sort by distance
    district_info.sort(key=lambda x: x['distance_km'])

    print(f"\n🎯 Closest districts to Jambi City:")
    for i, info in enumerate(district_info[:5], 1):
        print(f"  {i}. {info['name']:30s} - {info['distance_km']:5.1f} km, {info['area_km2']:6.0f} km²")

    # Recommendations
    print(f"\n💡 Recommended combinations:")

    # Option 1: Just city
    city_districts = [d for d in district_info if 'KOTA' in d['name'].upper() or d['distance_km'] < 5]
    if city_districts:
        print(f"\n  Option 1: City only")
        for d in city_districts:
            print(f"    - {d['name']} (~{d['area_km2']:.0f} km²)")

    # Option 2: City + closest district
    print(f"\n  Option 2: City + 1 adjacent district (moderate size)")
    closest_2 = district_info[:2]
    total_area = sum(d['area_km2'] for d in closest_2)
    print(f"    Districts: {', '.join(d['name'] for d in closest_2)}")
    print(f"    Total area: ~{total_area:.0f} km²")

    # Option 3: City + 2 adjacent districts
    print(f"\n  Option 3: City + 2 adjacent districts (larger)")
    closest_3 = district_info[:3]
    total_area = sum(d['area_km2'] for d in closest_3)
    print(f"    Districts: {', '.join(d['name'] for d in closest_3)}")
    print(f"    Total area: ~{total_area:.0f} km²")

    # ========================================================================
    # Visualize districts
    # ========================================================================

    print(f"\n{'='*80}")
    print("CREATING DISTRICT MAP")
    print("="*80)

    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot all districts
    for info in district_info:
        color = '#FF0000' if info['distance_km'] < 50 else '#CCCCCC'
        alpha = 0.6 if info['distance_km'] < 50 else 0.3

        gpd.GeoSeries([info['geometry']]).plot(
            ax=ax,
            color=color,
            edgecolor='black',
            linewidth=1.5,
            alpha=alpha
        )

        # Add district name
        centroid = info['geometry'].centroid
        ax.text(centroid.x, centroid.y, info['name'],
                ha='center', va='center', fontsize=8, fontweight='bold')

    # Mark city center
    ax.plot(jambi_city_center.x, jambi_city_center.y, 'r*', markersize=20, label='Jambi City Center')

    ax.set_title('Jambi Province Districts (Kabupaten/Kota)\nRed = Within 50km of city center',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/jambi_districts_map.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved district map: results/jambi_districts_map.png")

    plt.close()

else:
    print(f"\n⚠️  District column not found. Available columns:")
    print(klhk.columns.tolist())

print(f"\n{'='*80}")
print("EXPLORATION COMPLETE")
print("="*80)
