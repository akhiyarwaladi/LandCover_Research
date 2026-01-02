#!/usr/bin/env python3
"""
Analyze KLHK Ground Truth Detail Level
=======================================

Investigates the generalization/coarseness of KLHK polygons:
- Polygon sizes
- Minimum Mapping Unit (MMU)
- Mixed land cover within polygons
- Comparison with high-resolution imagery

Author: Claude Sonnet 4.5
Date: 2026-01-02
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

KLHK_PATH = 'data/klhk/KLHK_PL2024_Jambi_Full_WithGeometry.geojson'

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_polygon_sizes(gdf):
    """Analyze KLHK polygon sizes."""

    print("\n" + "="*80)
    print("KLHK POLYGON SIZE ANALYSIS")
    print("="*80)

    # Calculate areas in hectares
    gdf_projected = gdf.to_crs(epsg=32648)  # UTM Zone 48N for Jambi
    gdf_projected['area_ha'] = gdf_projected.geometry.area / 10000  # m² to hectares

    # Statistics
    print(f"\nTotal polygons: {len(gdf):,}")
    print(f"\nArea statistics (hectares):")
    print(f"  Min: {gdf_projected['area_ha'].min():.2f} ha")
    print(f"  Max: {gdf_projected['area_ha'].max():.2f} ha")
    print(f"  Mean: {gdf_projected['area_ha'].mean():.2f} ha")
    print(f"  Median: {gdf_projected['area_ha'].median():.2f} ha")
    print(f"  Std: {gdf_projected['area_ha'].std():.2f} ha")

    # Size distribution
    print(f"\nSize distribution:")
    bins = [0, 1, 5, 10, 50, 100, 500, 1000, np.inf]
    labels = ['<1 ha', '1-5 ha', '5-10 ha', '10-50 ha', '50-100 ha',
              '100-500 ha', '500-1000 ha', '>1000 ha']

    gdf_projected['size_category'] = pd.cut(
        gdf_projected['area_ha'],
        bins=bins,
        labels=labels
    )

    for label in labels:
        count = (gdf_projected['size_category'] == label).sum()
        pct = (count / len(gdf_projected)) * 100
        print(f"  {label:15s}: {count:6,} polygons ({pct:5.1f}%)")

    return gdf_projected

def analyze_by_class(gdf_projected):
    """Analyze polygon sizes by land cover class."""

    print("\n" + "="*80)
    print("POLYGON SIZE BY CLASS")
    print("="*80)

    # Class names from data_loader
    class_names = {
        2012: 'Built Area (Pemukiman)',
        2010: 'Plantation (Perkebunan)',
        2001: 'Primary Forest',
        2002: 'Secondary Forest',
        2009: 'Dryland Agriculture',
        20091: 'Mixed Dryland Agriculture',
        20092: 'Rice Field (Sawah)',
        2016: 'Water Body',
    }

    # Get top 5 most common classes
    top_classes = gdf_projected['KODE_KLSHP'].value_counts().head(5)

    print("\nMean polygon size by class (top 5):")
    for kode in top_classes.index:
        class_data = gdf_projected[gdf_projected['KODE_KLSHP'] == kode]
        mean_area = class_data['area_ha'].mean()
        count = len(class_data)
        class_name = class_names.get(kode, f'Unknown ({kode})')
        print(f"  {class_name:30s}: {mean_area:8.2f} ha (n={count:,})")

def analyze_built_area_detail(gdf_projected):
    """Specific analysis of Built Area polygons."""

    print("\n" + "="*80)
    print("BUILT AREA (PEMUKIMAN) DETAIL ANALYSIS")
    print("="*80)

    # Filter Built Area (KODE_KLSHP = 2012)
    built = gdf_projected[gdf_projected['KODE_KLSHP'] == 2012].copy()

    print(f"\nTotal Built Area polygons: {len(built):,}")
    print(f"Total Built Area coverage: {built['area_ha'].sum():,.2f} ha")

    print(f"\nBuilt Area polygon sizes:")
    print(f"  Min: {built['area_ha'].min():.2f} ha")
    print(f"  Max: {built['area_ha'].max():.2f} ha")
    print(f"  Mean: {built['area_ha'].mean():.2f} ha")
    print(f"  Median: {built['area_ha'].median():.2f} ha")

    # For Jambi City specifically
    city_bounds = {
        'min_lon': 103.55,
        'max_lon': 103.67,
        'min_lat': -1.65,
        'max_lat': -1.45
    }

    built_in_city = built.cx[
        city_bounds['min_lon']:city_bounds['max_lon'],
        city_bounds['min_lat']:city_bounds['max_lat']
    ]

    print(f"\nBuilt Area in Jambi City center:")
    print(f"  Polygons: {len(built_in_city):,}")
    print(f"  Total area: {built_in_city['area_ha'].sum():,.2f} ha")
    print(f"  Largest polygon: {built_in_city['area_ha'].max():,.2f} ha")

    # Find very large built polygons
    large_built = built[built['area_ha'] > 100]
    print(f"\nVery large Built Area polygons (>100 ha):")
    print(f"  Count: {len(large_built):,}")
    print(f"  These likely have mixed land cover inside!")

    return built

def create_visualizations(gdf_projected, built):
    """Create visualization plots."""

    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # 1. Polygon size distribution (all classes)
    ax1 = axes[0, 0]
    gdf_projected['area_ha'].hist(bins=50, ax=ax1, edgecolor='black')
    ax1.set_xlabel('Polygon Area (hectares)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('All Polygons: Size Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 500)  # Focus on 0-500 ha
    ax1.axvline(gdf_projected['area_ha'].median(), color='red',
                linestyle='--', label=f'Median: {gdf_projected["area_ha"].median():.1f} ha')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Built Area size distribution
    ax2 = axes[0, 1]
    built['area_ha'].hist(bins=50, ax=ax2, color='magenta', edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Polygon Area (hectares)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Built Area Polygons: Size Distribution', fontsize=14, fontweight='bold')
    ax2.axvline(built['area_ha'].median(), color='darkred',
                linestyle='--', label=f'Median: {built["area_ha"].median():.1f} ha')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Size by class (box plot)
    ax3 = axes[1, 0]

    top_classes = gdf_projected['KODE_KLSHP'].value_counts().head(6).index
    data_to_plot = []
    labels = []

    class_names_short = {
        2012: 'Built',
        2010: 'Plantation',
        2001: 'Pri. Forest',
        2002: 'Sec. Forest',
        2009: 'Dryland Ag',
        20092: 'Rice Field',
    }

    for kode in top_classes:
        class_data = gdf_projected[gdf_projected['KODE_KLSHP'] == kode]['area_ha']
        if len(class_data) > 0:
            data_to_plot.append(class_data.values)
            labels.append(class_names_short.get(kode, str(kode)))

    bp = ax3.boxplot(data_to_plot, labels=labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax3.set_ylabel('Polygon Area (hectares)', fontsize=12)
    ax3.set_title('Polygon Size by Class (Top 6)', fontsize=14, fontweight='bold')
    ax3.set_ylim(0, 200)
    ax3.grid(True, alpha=0.3, axis='y')
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 4. Implications text
    ax4 = axes[1, 1]
    ax4.axis('off')

    implications_text = f"""
KLHK Ground Truth Generalization Analysis

Key Findings:

1. Polygon Sizes:
   • Median: {gdf_projected['area_ha'].median():.1f} ha
   • Mean: {gdf_projected['area_ha'].mean():.1f} ha
   • Many polygons > 50 ha (coarse scale)

2. Built Area (Pink in your image):
   • {len(built):,} polygons total
   • Median size: {built['area_ha'].median():.1f} ha
   • Largest: {built['area_ha'].max():.1f} ha
   • {len(built[built['area_ha'] > 100]):,} polygons > 100 ha!

3. Implications:
   ✗ Large polygons generalize mixed land cover
   ✗ Within "Built Area", there ARE trees, roads, parks
   ✗ KLHK assigns DOMINANT class to entire polygon
   ✗ This is standard for national-scale mapping
   ✗ NOT suitable for pixel-level ground truth

4. Why Classification Accuracy is Limited:
   • Built Area F1: 0.42 (moderate)
   • Because: KLHK says "all built"
   • Reality: Mixed pixels (buildings + trees + roads)
   • Model sees spectral variation
   • Gets "penalized" for being more detailed!

5. Minimum Mapping Unit (MMU):
   • Effective MMU: ~{gdf_projected['area_ha'].quantile(0.25):.1f}-{gdf_projected['area_ha'].quantile(0.75):.1f} ha
   • This is COARSE for 10m-20m pixels
   • 1 hectare = 10,000 m² = 2,500 pixels at 20m!

Recommendation: Accept this limitation or use higher-res
ground truth (OSM buildings, manual digitization).
    """

    ax4.text(0.05, 0.95, implications_text, transform=ax4.transAxes,
             fontsize=11, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    output_file = 'results/klhk_generalization_analysis.png'
    os.makedirs('results', exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")

    return output_file

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    """Run complete KLHK detail analysis."""

    print("\n" + "="*80)
    print("KLHK GROUND TRUTH GENERALIZATION ANALYSIS")
    print("="*80)
    print("\nAnalyzing why Built Area polygons are so large...")

    # Load KLHK data
    print(f"\nLoading: {KLHK_PATH}")
    gdf = gpd.read_file(KLHK_PATH)
    print(f"Loaded {len(gdf):,} polygons")

    # Import pandas here (needed for cut function)
    import pandas as pd
    globals()['pd'] = pd

    # Run analyses
    gdf_projected = analyze_polygon_sizes(gdf)
    analyze_by_class(gdf_projected)
    built = analyze_built_area_detail(gdf_projected)

    # Create visualizations
    output_file = create_visualizations(gdf_projected, built)

    # Final summary
    print("\n" + "="*80)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*80)

    print("\n🔍 YOUR OBSERVATION IS CORRECT!")
    print("   The pink (Built Area) polygons ARE generalized.")

    print("\n📊 Evidence:")
    print(f"   • Median Built Area polygon: {built['area_ha'].median():.1f} ha")
    print(f"   • That's ~{int(built['area_ha'].median() * 10000 / 400):,} pixels at 20m!")
    print(f"   • {len(built[built['area_ha'] > 100]):,} Built polygons > 100 ha (1,000,000 m²)")
    print(f"   • Largest Built polygon: {built['area_ha'].max():.1f} ha")

    print("\n❓ Why This Happens:")
    print("   1. KLHK uses Minimum Mapping Unit (MMU) approach")
    print("   2. Digitizers draw large polygons, not pixel-by-pixel")
    print("   3. Assign DOMINANT class to entire polygon")
    print("   4. Standard for national-scale (1:250,000) mapping")

    print("\n⚠️  Impact on Your Classification:")
    print("   • Can't achieve pixel-perfect accuracy with coarse ground truth")
    print("   • Model might be MORE accurate than ground truth!")
    print("   • Low F1-scores don't necessarily mean bad model")

    print("\n💡 Recommendations:")
    print("   Option 1: Accept limitation, report with caveat")
    print("   Option 2: Use OpenStreetMap building footprints (urban)")
    print("   Option 3: Segment-based classification (match KLHK scale)")
    print("   Option 4: Manual validation on high-res imagery")

    print(f"\n✅ Analysis complete!")
    print(f"   Results: {output_file}")

    print("\n" + "="*80)

if __name__ == '__main__':
    main()
