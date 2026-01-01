"""
Analyze KLHK Ground Truth Classes (Simplified Version - No GeoPandas Required)
===============================================================================

Investigates the original KLHK class schema using JSON parsing.

Author: Claude Sonnet 4.5
Date: 2026-01-01
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Import the current mapping from data_loader
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Define mappings directly (from modules/data_loader.py)
KLHK_TO_SIMPLIFIED = {
    # Forest classes -> 1 (Trees)
    2001: 1, 2002: 1, 2003: 1, 2004: 1, 2005: 1, 2006: 1, 2007: 1,  20071: 1,
    # Agriculture -> 4 (Crops)
    2010: 4, 20051: 4, 20091: 4, 20092: 4, 20093: 4,
    # Built-up -> 6 (Built area)
    2012: 6, 20121: 6, 20122: 6,
    # Bare/Open -> 7 (Bare ground)
    2014: 7, 20141: 7,
    # Water -> 0
    5001: 0, 20094: 0,
    # Shrub -> 5
    2500: 5, 20041: 5,
    # Grass/Savanna -> 2
    3000: 2,
    # Cloud -> ignore
    50011: -1,
}

CLASS_NAMES = {
    0: 'Water',
    1: 'Trees/Forest',
    2: 'Grass/Savanna',
    4: 'Crops/Agriculture',
    5: 'Shrub/Scrub',
    6: 'Built Area',
    7: 'Bare Ground',
}

# KLHK original class names
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
    Analyze KLHK ground truth class distribution using JSON parsing.

    Args:
        geojson_path: Path to KLHK GeoJSON file
        output_dir: Directory to save analysis outputs
    """
    print("=" * 70)
    print("KLHK GROUND TRUTH CLASS ANALYSIS")
    print("=" * 70)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load GeoJSON
    print(f"\n📂 Loading data from: {geojson_path}")
    with open(geojson_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    features = data['features']
    print(f"   Total polygons: {len(features):,}")

    # Find the land cover code column
    possible_cols = ['ID Penutupan Lahan Tahun 2024', 'PL2024_ID', 'PL2024', 'KELAS']
    code_col = None
    sample_props = features[0]['properties']

    for col in possible_cols:
        if col in sample_props:
            code_col = col
            break

    if code_col is None:
        print(f"\n❌ ERROR: Land cover code column not found!")
        print(f"   Available columns: {list(sample_props.keys())}")
        return

    print(f"   Using column: '{code_col}'")

    # Extract all class codes
    class_codes = []
    for feature in features:
        try:
            code = int(feature['properties'][code_col])
            class_codes.append(code)
        except (KeyError, ValueError, TypeError):
            continue

    # Count occurrences
    code_counter = Counter(class_codes)
    unique_codes = sorted(code_counter.keys())

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
        count = code_counter[code]
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

    print(f"{'Class':<5} {'Name':<25} {'Polygons':>12} {'% of Total':>10} {'Original Codes':<30}")
    print("-" * 70)

    total_polygons = simplified_counts['Polygon_Count'].sum()

    for _, row in simplified_counts.iterrows():
        cls = int(row['Simplified_Class'])
        name = row['Simplified_Name']
        count = int(row['Polygon_Count'])
        pct = 100.0 * count / total_polygons

        # Find original codes that map to this class
        orig_codes = df_stats[df_stats['Simplified_Class'] == cls]['Original_Code'].tolist()
        codes_str = ', '.join(map(str, orig_codes))

        print(f"{cls:<5} {name:<25} {count:>12,} {pct:>9.1f}% {codes_str:<30}")

    # Check for unmapped codes
    if unmapped_codes:
        print("\n⚠️  WARNING: Found unmapped classes!")
        print(f"   Codes not in KLHK_TO_SIMPLIFIED: {unmapped_codes}")
        unmapped_count = sum(code_counter[c] for c in unmapped_codes)
        print(f"   Total unmapped polygons: {unmapped_count:,}")
    else:
        print("\n✅ All KLHK codes are mapped to simplified classes")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"Total original KLHK codes: {len(unique_codes)}")
    print(f"Total simplified classes: {len(simplified_counts)}")
    print(f"Reduction ratio: {len(unique_codes)}/{len(simplified_counts)} = {len(unique_codes)/len(simplified_counts):.1f}x")
    print(f"Total polygons: {len(features):,}")
    print(f"Unmapped polygons: {sum(code_counter.get(c, 0) for c in unmapped_codes):,}")

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
    ax.set_title(f'KLHK Original Class Distribution ({len(unique_codes)} Codes)',
                fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels for top classes
    for i, (_, row) in enumerate(df_sorted.head(10).iterrows()):
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
    ax.set_title(f'Simplified Class Distribution ({len(simplified_counts)} Classes)',
                fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, (_, row) in enumerate(simp_sorted.iterrows()):
        count = int(row['Polygon_Count'])
        pct = 100.0 * count / total_polygons
        ax.text(i, row['Polygon_Count'], f"{count:,}\n({pct:.1f}%)",
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'class_distribution.png')
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
            'Percentage': 100.0 * total_count / total_polygons,
            'Original_Codes_Count': len(orig_classes),
            'KLHK_Codes': ', '.join(map(str, orig_classes['Original_Code'].tolist())),
            'Original_Names': ' | '.join(orig_classes['Original_Name'].str[:40].tolist())
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

    print("\n📋 Key Findings:")
    print(f"   • Original KLHK classes: {len(unique_codes)} codes")
    print(f"   • Simplified classes: {len(simplified_counts)} classes")
    print(f"   • Class reduction: {len(unique_codes)/len(simplified_counts):.1f}x simplification")
    print(f"   • Largest class: {simplified_counts.iloc[0]['Simplified_Name']} "
          f"({int(simplified_counts.iloc[0]['Polygon_Count']):,} polygons, "
          f"{100*simplified_counts.iloc[0]['Polygon_Count']/total_polygons:.1f}%)")
    print(f"   • Smallest class: {simplified_counts.iloc[-1]['Simplified_Name']} "
          f"({int(simplified_counts.iloc[-1]['Polygon_Count']):,} polygons, "
          f"{100*simplified_counts.iloc[-1]['Polygon_Count']/total_polygons:.1f}%)")

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
