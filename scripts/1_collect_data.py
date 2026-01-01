#!/usr/bin/env python3
"""
STEP 1: Data Collection
========================

Verifies that all required data files exist for classification.

Required Data:
- KLHK ground truth: data/klhk/KLHK_PL2024_Jambi_Full_WithGeometry.geojson
- Sentinel-2 imagery: data/sentinel/S2_jambi_2024_*.tif (4 tiles)

Usage:
    python scripts/1_collect_data.py

Outputs:
- Data verification report
- Instructions if data is missing
"""

import os
import sys
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if file exists and return size."""
    if os.path.exists(filepath):
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"  ✅ {description}")
        print(f"     Path: {filepath}")
        print(f"     Size: {size_mb:.2f} MB")
        return True
    else:
        print(f"  ❌ {description}")
        print(f"     Expected path: {filepath}")
        return False


def main():
    print("=" * 70)
    print("STEP 1: DATA COLLECTION VERIFICATION")
    print("=" * 70)

    # Required files
    required_files = {
        'KLHK Ground Truth': 'data/klhk/KLHK_PL2024_Jambi_Full_WithGeometry.geojson',
        'Sentinel-2 Tile 1': 'data/sentinel/S2_jambi_2024_20m_AllBands-0000000000-0000000000.tif',
        'Sentinel-2 Tile 2': 'data/sentinel/S2_jambi_2024_20m_AllBands-0000000000-0000010496.tif',
        'Sentinel-2 Tile 3': 'data/sentinel/S2_jambi_2024_20m_AllBands-0000010496-0000000000.tif',
        'Sentinel-2 Tile 4': 'data/sentinel/S2_jambi_2024_20m_AllBands-0000010496-0000010496.tif',
    }

    print("\nChecking required data files...")
    print("-" * 70)

    all_exist = True
    for description, filepath in required_files.items():
        exists = check_file_exists(filepath, description)
        if not exists:
            all_exist = False
        print()

    print("=" * 70)
    if all_exist:
        print("✅ ALL DATA FILES FOUND!")
        print("=" * 70)
        print("\nYou can proceed to the next step:")
        print("  python scripts/2_preprocess_data.py")
        return 0
    else:
        print("❌ SOME DATA FILES ARE MISSING!")
        print("=" * 70)
        print("\nTo download missing data:")
        print()
        print("1. KLHK Ground Truth:")
        print("   python scripts/download_klhk_kmz_partitioned.py")
        print()
        print("2. Sentinel-2 Imagery:")
        print("   python scripts/download_sentinel2.py")
        print()
        print("See CLAUDE.md for detailed instructions.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
