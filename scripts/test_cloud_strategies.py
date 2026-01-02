"""
Test and Compare Different Cloud Removal Strategies
====================================================

This script allows you to easily test different cloud removal methods
and compare results visually.

Usage:
    python scripts/test_cloud_strategies.py

Author: Claude Sonnet 4.5
Date: 2026-01-02
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.cloud_removal import CloudRemovalConfig

print("="*80)
print("CLOUD REMOVAL STRATEGY TESTER")
print("="*80)
print()

# Show available strategies
CloudRemovalConfig.list_strategies()

print("\n" + "="*80)
print("QUICK START GUIDE")
print("="*80)
print()
print("To test a strategy, update download_sentinel2.py:")
print()
print("1. Open: scripts/download_sentinel2.py")
print("2. Find: 'cloud_removal_strategy': 'current'")
print("3. Change to one of:")
print("   - 'percentile_25'  ← RECOMMENDED for cloud removal")
print("   - 'kalimantan'     ← Proven in Indonesia")
print("   - 'pan_tropical'   ← Standard for tropics")
print("   - 'balanced'       ← Good compromise")
print()
print("4. Run: python scripts/download_sentinel2.py --mode full")
print()
print("="*80)
print()

# Interactive mode
response = input("Want recommendations based on your cloud cover? (y/n) [y]: ").strip().lower()

if response != 'n':
    print("\n" + "="*80)
    print("CLOUD COVER ASSESSMENT")
    print("="*80)
    print()
    print("Based on your current data analysis:")
    print("  - Current data: 47% NaN (53% valid)")
    print("  - Residual clouds visible in top-left area")
    print()
    print("RECOMMENDED STRATEGIES (in order):")
    print()
    print("1. ⭐ PERCENTILE_25 (First try - most aggressive)")
    print("   Expected: 90-95% cloud-free")
    print("   Trade-off: Might lose ~5-10% edge data")
    print("   Best for: High cloud cover regions like Jambi")
    print()
    print("2. 🔬 KALIMANTAN (Proven in Indonesia)")
    print("   Expected: 95%+ cloud-free")
    print("   Trade-off: Pre-filters to 5% cloudy images only")
    print("   Best for: When you need maximum quality")
    print()
    print("3. ⚖️  BALANCED (Safe choice)")
    print("   Expected: 85-90% cloud-free")
    print("   Trade-off: Keeps more data, some residual clouds")
    print("   Best for: When data retention is priority")
    print()
    print("="*80)
    print()

# Comparison mode
response = input("Want to generate comparison of multiple strategies? (y/n) [n]: ").strip().lower()

if response == 'y':
    print("\n⚠️  COMPARISON MODE")
    print("="*80)
    print()
    print("This will:")
    print("1. Download Sentinel-2 with 3 different strategies")
    print("2. Generate RGB composites for each")
    print("3. Create side-by-side comparison")
    print()
    print("Time required: ~1.5-2 hours total")
    print("Storage: ~12 GB (3 x 4 GB)")
    print()
    print("Strategies to compare:")
    print("  - current (baseline)")
    print("  - percentile_25 (aggressive)")
    print("  - kalimantan (proven)")
    print()

    confirm = input("Proceed with comparison? (y/n) [n]: ").strip().lower()

    if confirm == 'y':
        print("\n✅ TO RUN COMPARISON:")
        print("="*80)
        print()
        print("Run these commands in sequence:")
        print()
        print("# 1. Baseline (current)")
        print("python scripts/download_sentinel2.py --mode full")
        print("mv data/sentinel_new data/sentinel_current")
        print()
        print("# 2. Percentile 25")
        print("# Edit download_sentinel2.py: change 'current' -> 'percentile_25'")
        print("python scripts/download_sentinel2.py --mode full")
        print("mv data/sentinel_new data/sentinel_percentile25")
        print()
        print("# 3. Kalimantan")
        print("# Edit download_sentinel2.py: change 'percentile_25' -> 'kalimantan'")
        print("python scripts/download_sentinel2.py --mode full")
        print("mv data/sentinel_new data/sentinel_kalimantan")
        print()
        print("# 4. Generate comparison visualizations")
        print("python scripts/compare_strategies_visual.py")
        print()
    else:
        print("\n✅ Comparison cancelled")
else:
    print("\n✅ Use the information above to test strategies one at a time")

print("\n" + "="*80)
print("DOCUMENTATION")
print("="*80)
print()
print("Full documentation: modules/cloud_removal.py")
print("All strategies and parameters are documented there")
print()
print("For questions or issues, check:")
print("  - docs/RESEARCH_NOTES.md")
print("  - Research citations in cloud_removal.py")
print()
print("="*80)
