"""
Analyze KLHK Ground Truth Classes
==================================

Investigates the original KLHK class schema and how it maps to simplified classes.

Purpose:
- Count all unique KLHK original class codes
- Show distribution of original classes
- Compare original vs simplified class schema
- Identify any unmapped classes
- Generate comprehensive class mapping report

Author: Claude Sonnet 4.5
Date: 2026-01-01
"""

import os
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Import the current mapping from data_loader
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from modules.data_loader import KLHK_TO_SIMPLIFIED, CLASS_NAMES

# KLHK original class names (based on Indonesian land cover classification)
KLHK_ORIGINAL_NAMES = {
    # Forest classes
    2001: 'Hutan Lahan Kering Primer (Primary Dryland Forest)',
    2002: 'Hutan Lahan Kering Sekunder (Secondary Dryland Forest)',
    2003: 'Hutan Rawa Primer (Primary Swamp Forest)',
    2004: 'Hutan Rawa Sekunder (Secondary Swamp Forest)',
    2005: 'Hutan Mangrove Primer (Primary Mangrove Forest)',
    2006: 'Hutan Mangrove Sekunder (Secondary Mangrove Forest)',
    2007: 'Hutan Tanaman (Plantation Forest)',
    20071: 'Hutan Tanaman Industri (Industrial Plantation Forest)',

    # Agriculture
    2010: 'Perkebunan (Plantation)',
    20051: 'Tambak (Fishpond)',
    20091: 'Pertanian Lahan Kering Campur (Mixed Dryland Agriculture)',
    20092: 'Sawah (Paddy Field)',
    20093: 'Ladang (Shifting Cultivation)',

    # Built-up
    2012: 'Pemukiman (Settlement)',
    20121: 'Lahan Terbangun (Built-up Land)',
    20122: 'Jaringan Jalan (Road Network)',

    # Bare/Open
    2014: 'Tanah Terbuka (Bare Land)',
    20141: 'Pertambangan (Mining)',

    # Water
    5001: 'Tubuh Air (Water Body)',
    20094: 'Rawa (Swamp)',

    # Shrub/Scrub
    2500: 'Savanna/Padang Rumput (Savanna/Grassland)',
    20041: 'Belukar Rawa (Swamp Shrub)',

    # Grass
    3000: 'Padang Rumput (Grassland)',

    # Cloud
    50011: 'Awan (Cloud)',
}


def analyze_klhk_classes(geojson_path, output_dir='results/klhk_analysis'):
    """
    Analyze KLHK ground truth class distribution.

    Args:
        geojson_path: Path to KLHK GeoJSON file
        output_dir: Directory to save analysis outputs
    """
    print("=" * 70)
    print("KLHK GROUND TRUTH CLASS ANALYSIS")
    print("=" * 70)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print(f"\n📂 Loading data from: {geojson_path}")
    gdf = gpd.read_file(geojson_path)
    print(f"   Total polygons: {len(gdf):,}")
    print(f"   CRS: {gdf.crs}")

    # Find the land cover code column
    code_col = None
    for col in ['ID Penutupan Lahan Tahun 2024', 'PL2024_ID', 'PL2024', 'KELAS']:
        if col in gdf.columns:
            code_col = col
            break

    if code_col is None:
        print(f"\n❌ ERROR: Land cover code column not found!")
        print(f"   Available columns: {list(gdf.columns)}")
        return

    print(f"   Using column: '{code_col}'")

    # Extract class codes
    gdf['klhk_code'] = gdf[code_col].astype(int)

    # Count unique classes
    unique_codes = sorted(gdf['klhk_code'].unique())
    print(f"\n📊 Found {len(unique_codes)} unique KLHK class codes")

    # Analyze each original class
    class_stats = []
    unmapped_codes = []

    print("\n" + "=" * 70)
    print("ORIGINAL KLHK CLASSES")
    print("=" * 70)
    print(f"{'Code':<8} {'Name':<50} {'Count':>10}")
    print("-" * 70)

    for code in unique_codes:
        count = (gdf['klhk_code'] == code).sum()
        name = KLHK_ORIGINAL_NAMES.get(code, 'Unknown Class')
        simplified = KLHK_TO_SIMPLIFIED.get(code, None)
        simplified_name = CLASS_NAMES.get(simplified, 'Not Mapped') if simplified is not None else 'Not Mapped'

        class_stats.append({
            'Original_Code': code,
            'Original_Name': name,
            'Polygon_Count': count,
            'Simplified_Class': simplified if simplified is not None else -999,
            'Simplified_Name': simplified_name
        })

        if simplified is None:
            unmapped_codes.append(code)

        print(f"{code:<8} {name:<50} {count:>10,}")

    # Create DataFrame
    df_stats = pd.DataFrame(class_stats)

    # Analyze simplified classes
    print("\n" + "=" * 70)
    print("SIMPLIFIED CLASSES (Current Mapping)")
    print("=" * 70)

    simplified_counts = df_stats[df_stats['Simplified_Class'] >= 0].groupby(
        ['Simplified_Class', 'Simplified_Name']
    )['Polygon_Count'].sum().reset_index()

    print(f"{'Class':<5} {'Name':<25} {'Polygons':>12} {'Original Codes':<30}")
    print("-" * 70)

    for _, row in simplified_counts.iterrows():
        cls = int(row['Simplified_Class'])
        name = row['Simplified_Name']
        count = int(row['Polygon_Count'])

        # Find original codes that map to this class
        orig_codes = df_stats[df_stats['Simplified_Class'] == cls]['Original_Code'].tolist()
        codes_str = ', '.join(map(str, orig_codes))

        print(f"{cls:<5} {name:<25} {count:>12,} {codes_str:<30}")

    # Check for unmapped codes
    if unmapped_codes:
        print("\n⚠️  WARNING: Found unmapped classes!")
        print(f"   Codes not in KLHK_TO_SIMPLIFIED: {unmapped_codes}")
    else:
        print("\n✅ All KLHK codes are mapped to simplified classes")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"Total original KLHK codes: {len(unique_codes)}")
    print(f"Total simplified classes: {len(simplified_counts)}")
    print(f"Reduction ratio: {len(unique_codes)}/{len(simplified_counts)} = {len(unique_codes)/len(simplified_counts):.1f}x")
    print(f"Total polygons: {len(gdf):,}")
    print(f"Unmapped polygons: {len(gdf[~gdf['klhk_code'].isin(KLHK_TO_SIMPLIFIED.keys())]):,}")

    # Save detailed statistics to CSV
    csv_path = os.path.join(output_dir, 'klhk_class_mapping.csv')
    df_stats.to_csv(csv_path, index=False)
    print(f"\n💾 Saved class mapping to: {csv_path}")

    # Create visualizations
    print("\n📊 Generating visualizations...")

    # Figure 1: Original class distribution
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Original classes bar chart
    ax = axes[0]
    df_sorted = df_stats.sort_values('Polygon_Count', ascending=False)
    colors = plt.cm.tab20(np.linspace(0, 1, len(df_sorted)))

    bars = ax.bar(range(len(df_sorted)), df_sorted['Polygon_Count'], color=colors)
    ax.set_xticks(range(len(df_sorted)))
    ax.set_xticklabels([f"{row['Original_Code']}" for _, row in df_sorted.iterrows()],
                       rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Number of Polygons', fontsize=11, fontweight='bold')
    ax.set_title('KLHK Original Class Distribution (All Codes)', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        ax.text(i, row['Polygon_Count'], f"{row['Polygon_Count']:,}",
                ha='center', va='bottom', fontsize=7)

    # Simplified classes bar chart
    ax = axes[1]
    simp_sorted = simplified_counts.sort_values('Polygon_Count', ascending=False)
    colors_simp = ['#0173B2', '#029E73', '#CC78BC', '#DE8F05', '#CA9161', '#949494', '#ECE133']

    bars = ax.bar(range(len(simp_sorted)), simp_sorted['Polygon_Count'],
                  color=colors_simp[:len(simp_sorted)])
    ax.set_xticks(range(len(simp_sorted)))
    ax.set_xticklabels([f"Class {int(row['Simplified_Class'])}\n{row['Simplified_Name']}"
                        for _, row in simp_sorted.iterrows()],
                       fontsize=9, fontweight='bold')
    ax.set_ylabel('Number of Polygons', fontsize=11, fontweight='bold')
    ax.set_title('Simplified Class Distribution (6-7 Classes)', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, (_, row) in enumerate(simp_sorted.iterrows()):
        ax.text(i, row['Polygon_Count'], f"{row['Polygon_Count']:,}",
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'class_distribution.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"   Saved: {fig_path}")
    plt.close()

    # Figure 2: Mapping diagram
    fig, ax = plt.subplots(figsize=(14, 10))

    # Prepare data for Sankey-style diagram
    y_positions_orig = np.linspace(0, 100, len(unique_codes))
    y_positions_simp = np.linspace(10, 90, len(simplified_counts))

    # Draw original classes on left
    for i, code in enumerate(unique_codes):
        row = df_stats[df_stats['Original_Code'] == code].iloc[0]
        count = row['Polygon_Count']
        name = row['Original_Name'][:40]  # Truncate

        # Draw box
        box_height = 3
        rect = plt.Rectangle((0, y_positions_orig[i] - box_height/2), 15, box_height,
                            facecolor='lightblue', edgecolor='black', linewidth=0.5)
        ax.add_patch(rect)

        # Add text
        ax.text(7.5, y_positions_orig[i], f"{code}\n{count:,}",
               ha='center', va='center', fontsize=6, fontweight='bold')

    # Draw simplified classes on right
    for i, (_, row) in enumerate(simplified_counts.iterrows()):
        cls = int(row['Simplified_Class'])
        name = row['Simplified_Name']
        count = int(row['Polygon_Count'])

        # Draw box
        box_height = 5
        rect = plt.Rectangle((85, y_positions_simp[i] - box_height/2), 15, box_height,
                            facecolor=colors_simp[i % len(colors_simp)],
                            edgecolor='black', linewidth=1)
        ax.add_patch(rect)

        # Add text
        ax.text(92.5, y_positions_simp[i], f"{name}\n{count:,}",
               ha='center', va='center', fontsize=8, fontweight='bold')

    # Draw connections
    for i, code in enumerate(unique_codes):
        simplified = KLHK_TO_SIMPLIFIED.get(code, None)
        if simplified is not None and simplified >= 0:
            # Find position of simplified class
            simp_idx = list(simplified_counts['Simplified_Class']).index(simplified)

            # Draw line
            ax.plot([15, 85], [y_positions_orig[i], y_positions_simp[simp_idx]],
                   'gray', alpha=0.3, linewidth=0.5)

    # Labels
    ax.text(7.5, 105, 'ORIGINAL KLHK CODES\n(24 codes)',
           ha='center', fontsize=12, fontweight='bold')
    ax.text(92.5, 105, 'SIMPLIFIED CLASSES\n(6-7 classes)',
           ha='center', fontsize=12, fontweight='bold')

    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 110)
    ax.axis('off')
    ax.set_title('KLHK Class Mapping: Original → Simplified',
                fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'class_mapping_diagram.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"   Saved: {fig_path}")
    plt.close()

    # Create detailed mapping table
    print("\n📄 Creating detailed mapping table...")

    # Group by simplified class
    mapping_details = []
    for simp_cls in sorted(simplified_counts['Simplified_Class'].unique()):
        if simp_cls < 0:
            continue

        simp_name = CLASS_NAMES.get(simp_cls, 'Unknown')
        orig_classes = df_stats[df_stats['Simplified_Class'] == simp_cls]
        total_count = orig_classes['Polygon_Count'].sum()

        mapping_details.append({
            'Simplified_Class': simp_cls,
            'Simplified_Name': simp_name,
            'Total_Polygons': total_count,
            'Original_Codes': len(orig_classes),
            'KLHK_Codes': ', '.join(map(str, orig_classes['Original_Code'].tolist())),
            'Original_Names': ' | '.join(orig_classes['Original_Name'].str[:30].tolist())
        })

    df_mapping = pd.DataFrame(mapping_details)
    mapping_csv = os.path.join(output_dir, 'simplified_class_mapping.csv')
    df_mapping.to_csv(mapping_csv, index=False)
    print(f"   Saved: {mapping_csv}")

    # Final summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"\n📂 Results saved to: {output_dir}/")
    print(f"   - klhk_class_mapping.csv (detailed class statistics)")
    print(f"   - simplified_class_mapping.csv (simplified mapping)")
    print(f"   - class_distribution.png (distribution charts)")
    print(f"   - class_mapping_diagram.png (mapping visualization)")

    print("\n📋 Key Findings:")
    print(f"   • Original KLHK classes: {len(unique_codes)} codes")
    print(f"   • Simplified classes: {len(simplified_counts)} classes")
    print(f"   • Class reduction: {len(unique_codes)//len(simplified_counts)}x simplification")
    print(f"   • Largest class: {simplified_counts.iloc[0]['Simplified_Name']} "
          f"({int(simplified_counts.iloc[0]['Polygon_Count']):,} polygons)")

    return df_stats, df_mapping


if __name__ == '__main__':
    # Path to KLHK data
    klhk_path = 'data/klhk/KLHK_PL2024_Jambi_Full_WithGeometry.geojson'

    if not os.path.exists(klhk_path):
        print(f"❌ ERROR: KLHK data not found at {klhk_path}")
        print("   Please ensure data file exists.")
    else:
        # Run analysis
        df_stats, df_mapping = analyze_klhk_classes(klhk_path)
