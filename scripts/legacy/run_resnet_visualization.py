#!/usr/bin/env python3
"""
ResNet Visualization - Centralized Script
==========================================

Generate all visualizations for ResNet results.
Supports ALL ResNet architectures (18, 34, 50, 101, 152).

Usage:
    python scripts/run_resnet_visualization.py                    # ResNet50 (default)
    python scripts/run_resnet_visualization.py --variant resnet18 # Specific variant
    python scripts/run_resnet_visualization.py --all              # All variants

Outputs:
    - results/{variant}/visualizations/training_curves.png
    - results/{variant}/visualizations/confusion_matrix.png
    - results/{variant}/visualizations/model_comparison.png
    - results/{variant}/visualizations/spatial_predictions.png

Author: Claude Sonnet 4.5
Date: 2026-01-03
Updated: 2026-01-03 (Added multi-architecture support)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import argparse

from modules.data_loader import load_klhk_data, load_sentinel2_tiles
from modules.preprocessor import rasterize_klhk
from modules.dl_visualizer import generate_all_visualizations

# ============================================================================
# PARSE ARGUMENTS
# ============================================================================

parser = argparse.ArgumentParser(description='Generate ResNet visualizations')
parser.add_argument('--variant', type=str, default='resnet50',
                   help='ResNet variant (resnet18/34/50/101/152)')
parser.add_argument('--all', action='store_true',
                   help='Generate for all available variants')
args = parser.parse_args()

# ============================================================================
# CONFIGURATION
# ============================================================================

# Data paths (same for all variants)
KLHK_PATH = 'data/klhk/KLHK_PL2024_Jambi_Full_WithGeometry.geojson'
PROVINCE_TILES = [
    'data/sentinel/S2_jambi_2024_20m_AllBands-0000000000-0000000000.tif',
    'data/sentinel/S2_jambi_2024_20m_AllBands-0000000000-0000010496.tif',
    'data/sentinel/S2_jambi_2024_20m_AllBands-0000010496-0000000000.tif',
    'data/sentinel/S2_jambi_2024_20m_AllBands-0000010496-0000010496.tif'
]

# Baseline results (Random Forest - same for all comparisons)
BASELINE_RESULTS = {
    'accuracy': 0.7495,
    'f1_macro': 0.542,
    'f1_weighted': 0.744
}

# Label mapping and class info (same for all)
LABEL_MAPPING = {0: 0, 1: 1, 4: 2, 5: 3, 6: 4, 7: 5}
CLASS_NAMES = ['Water', 'Trees', 'Crops', 'Shrub', 'Built', 'Bare']
CLASS_COLORS = {
    0: '#0066CC',  # Water - Bright Blue
    1: '#228B22',  # Trees/Forest - Forest Green
    2: '#90EE90',  # Crops - Light Green
    3: '#FF8C00',  # Shrub - Dark Orange
    4: '#FF1493',  # Built - Deep Pink/Magenta
    5: '#D2691E',  # Bare Ground - Chocolate Brown
}

# Determine which variants to process
if args.all:
    VARIANTS = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
else:
    VARIANTS = [args.variant]

print("\n" + "="*80)
print("RESNET VISUALIZATION (MULTI-ARCHITECTURE)")
print("="*80)
print(f"Variants to process: {', '.join(VARIANTS)}")

# ============================================================================
# LOAD GROUND TRUTH (once, shared by all)
# ============================================================================

print("\n" + "-"*80)
print("Loading Ground Truth (shared by all variants)")
print("-"*80)

print("\nLoading KLHK...")
klhk_gdf = load_klhk_data(KLHK_PATH, verbose=False)

print("Loading Sentinel-2...")
_, s2_profile = load_sentinel2_tiles(PROVINCE_TILES, verbose=False)

print("Rasterizing KLHK...")
klhk_raster = rasterize_klhk(klhk_gdf, s2_profile, verbose=False)

print("✓ Ground truth loaded")

# ============================================================================
# GENERATE VISUALIZATIONS FOR EACH VARIANT
# ============================================================================

for i, variant in enumerate(VARIANTS, 1):
    print("\n" + "="*80)
    print(f"VARIANT {i}/{len(VARIANTS)}: {variant.upper()}")
    print("="*80)

    # Paths for this variant
    training_history = f'results/{variant}/training_history.npz'
    test_results = f'results/{variant}/test_results.npz'
    predictions = f'results/{variant}/predictions.npy'
    output_dir = f'results/{variant}/visualizations'

    # Check if results exist
    if not os.path.exists(training_history):
        print(f"⚠️  Training history not found: {training_history}")
        print(f"   Skipping {variant}...")
        continue

    if not os.path.exists(test_results):
        print(f"⚠️  Test results not found: {test_results}")
        print(f"   Skipping {variant}...")
        continue

    if not os.path.exists(predictions):
        print(f"⚠️  Predictions not found: {predictions}")
        print(f"   Skipping {variant}...")
        continue

    # Generate visualizations
    print(f"\n📊 Generating visualizations for {variant}...")
    print(f"   Output: {output_dir}/")

    generate_all_visualizations(
        training_history_path=training_history,
        test_results_path=test_results,
        predictions_path=predictions,
        ground_truth=klhk_raster,
        output_dir=output_dir,
        baseline_results=BASELINE_RESULTS,
        label_mapping=LABEL_MAPPING,
        class_names=CLASS_NAMES,
        class_colors=CLASS_COLORS,
        verbose=False  # Less verbose for batch processing
    )

    print(f"✓ {variant} visualizations complete!")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("ALL VISUALIZATIONS COMPLETE!")
print("="*80)

print("\n📁 Generated visualizations:")
for variant in VARIANTS:
    output_dir = f'results/{variant}/visualizations'
    if os.path.exists(output_dir):
        print(f"\n✓ {variant}:")
        print(f"   {output_dir}/training_curves.png")
        print(f"   {output_dir}/confusion_matrix.png")
        print(f"   {output_dir}/model_comparison.png")
        print(f"   {output_dir}/spatial_predictions.png")

print("\n✨ All files are SEPARATE and ready for Microsoft Word!")
print("✨ Re-run anytime: python scripts/run_resnet_visualization.py")
print("✨ Works from saved models - no retraining needed!")

print("\n" + "="*80)
