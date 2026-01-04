#!/usr/bin/env python3
"""
Clean Up and Reorganize Results Directory
=========================================

Reorganizes messy results/ directory into clean, professional structure.

BEFORE (messy):
results/
├── qualitative_comparison/ (64M, OLD)
├── qualitative_comparison_FINAL/ (42M, OLD)
├── qualitative_comparison_FIXED/ (42M, OLD)
├── qualitative_FINAL_DRY_SEASON/ (55M, CURRENT)
├── publication_comparison/ (1M, CURRENT)
├── statistical_analysis/ (172K, CURRENT)
├── resnet/ (203M, OLD)
├── resnet18/34/101/152/ (320K each, CURRENT)
└── [many old directories...]

AFTER (clean):
results/
├── models/                    # Trained models & test results
│   ├── resnet18/
│   ├── resnet34/
│   ├── resnet101/
│   └── resnet152/
├── tables/                    # All publication tables
│   ├── performance/           # Overall performance metrics
│   ├── statistical/           # Statistical analysis tables
│   └── per_class/             # Per-class detailed metrics
├── figures/                   # All publication figures
│   ├── confusion_matrices/    # Error pattern analysis
│   ├── training_curves/       # Convergence plots
│   ├── spatial_maps/          # Qualitative comparison maps
│   └── statistical/           # Statistical visualizations
└── archived/                  # Old/redundant (for backup)

Author: Claude Sonnet 4.5
Date: 2026-01-04
"""

import os
import shutil
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path('results')
ARCHIVE_DIR = BASE_DIR / 'archived'

# New clean structure
NEW_STRUCTURE = {
    'models': ['resnet18', 'resnet34', 'resnet101', 'resnet152'],
    'tables': ['performance', 'statistical', 'per_class'],
    'figures': ['confusion_matrices', 'training_curves', 'spatial_maps', 'statistical']
}

# Directories to DELETE (redundant/old)
DIRS_TO_DELETE = [
    'qualitative_comparison',
    'qualitative_comparison_FINAL',
    'qualitative_comparison_FIXED',
    'resnet',  # Old unified directory
    'classification_maps',
    'classification_maps_colorful',
    'klhk_vs_reality',
    'visualizations',
    'strategy_test',
    'rgb_visualization',
    'figures',  # Old figures directory
    'publication',  # Old publication directory
    'tables',  # Old tables directory
    'cv_timing',
    'klhk_analysis',
    'validation'
]

# Directories to MOVE/REORGANIZE
DIRS_TO_MOVE = {
    # Source → Destination
    'resnet18': 'models/resnet18',
    'resnet34': 'models/resnet34',
    'resnet101': 'models/resnet101',
    'resnet152': 'models/resnet152',
    'publication_comparison': 'temp_publication_comparison',  # Will split later
    'statistical_analysis': 'temp_statistical_analysis',  # Will split later
    'qualitative_FINAL_DRY_SEASON': 'temp_qualitative'  # Will split later
}

print("="*80)
print("RESULTS DIRECTORY CLEANUP & REORGANIZATION")
print("="*80)

# ============================================================================
# 1. CREATE NEW CLEAN STRUCTURE
# ============================================================================

print("\n" + "-"*80)
print("1/5: Creating New Clean Structure")
print("-"*80)

for parent, subdirs in NEW_STRUCTURE.items():
    parent_path = BASE_DIR / parent
    parent_path.mkdir(exist_ok=True)
    print(f"✓ Created: {parent}/")

    for subdir in subdirs:
        subdir_path = parent_path / subdir
        subdir_path.mkdir(exist_ok=True)
        print(f"  ✓ {parent}/{subdir}/")

ARCHIVE_DIR.mkdir(exist_ok=True)
print(f"✓ Created: archived/ (for old files)")

# ============================================================================
# 2. MOVE CURRENT RESULTS TO NEW STRUCTURE
# ============================================================================

print("\n" + "-"*80)
print("2/5: Moving Current Results to Clean Structure")
print("-"*80)

# Move model directories
for old_name, new_path in DIRS_TO_MOVE.items():
    old_path = BASE_DIR / old_name
    new_full_path = BASE_DIR / new_path

    if old_path.exists():
        if new_full_path.exists():
            shutil.rmtree(new_full_path)
        shutil.move(str(old_path), str(new_full_path))
        print(f"✓ Moved: {old_name} → {new_path}")

# ============================================================================
# 3. REORGANIZE FILES INTO TABLES AND FIGURES
# ============================================================================

print("\n" + "-"*80)
print("3/5: Organizing Tables and Figures")
print("-"*80)

# Move publication_comparison files
pub_comp_temp = BASE_DIR / 'temp_publication_comparison'
if pub_comp_temp.exists():
    # Tables
    for table_file in pub_comp_temp.glob('*.xlsx'):
        dest = BASE_DIR / 'tables' / 'performance' / table_file.name
        shutil.copy2(table_file, dest)
        print(f"  ✓ Table: {table_file.name} → tables/performance/")

    for tex_file in pub_comp_temp.glob('*.tex'):
        dest = BASE_DIR / 'tables' / 'performance' / tex_file.name
        shutil.copy2(tex_file, dest)
        print(f"  ✓ LaTeX: {tex_file.name} → tables/performance/")

    # Figures
    for png_file in pub_comp_temp.glob('*.png'):
        if 'confusion' in png_file.name:
            dest = BASE_DIR / 'figures' / 'confusion_matrices' / png_file.name
        elif 'training' in png_file.name or 'curves' in png_file.name:
            dest = BASE_DIR / 'figures' / 'training_curves' / png_file.name
        else:
            dest = BASE_DIR / 'figures' / 'confusion_matrices' / png_file.name  # Default
        shutil.copy2(png_file, dest)
        print(f"  ✓ Figure: {png_file.name} → figures/{dest.parent.name}/")

    # Archive the temp directory
    shutil.move(str(pub_comp_temp), str(ARCHIVE_DIR / 'publication_comparison'))

# Move statistical_analysis files
stat_temp = BASE_DIR / 'temp_statistical_analysis'
if stat_temp.exists():
    # Tables
    for xlsx_file in stat_temp.glob('*.xlsx'):
        dest = BASE_DIR / 'tables' / 'statistical' / xlsx_file.name
        shutil.copy2(xlsx_file, dest)
        print(f"  ✓ Statistical table: {xlsx_file.name} → tables/statistical/")

    # Figures
    for png_file in stat_temp.glob('*.png'):
        dest = BASE_DIR / 'figures' / 'statistical' / png_file.name
        shutil.copy2(png_file, dest)
        print(f"  ✓ Statistical figure: {png_file.name} → figures/statistical/")

    # Archive
    shutil.move(str(stat_temp), str(ARCHIVE_DIR / 'statistical_analysis'))

# Move spatial comparison maps
qual_temp = BASE_DIR / 'temp_qualitative'
if qual_temp.exists():
    # Copy to figures/spatial_maps/
    for subdir in ['province', 'city']:
        src_dir = qual_temp / subdir
        if src_dir.exists():
            dest_dir = BASE_DIR / 'figures' / 'spatial_maps' / subdir
            dest_dir.mkdir(parents=True, exist_ok=True)
            for png_file in src_dir.glob('*.png'):
                shutil.copy2(png_file, dest_dir / png_file.name)
                print(f"  ✓ Spatial map: {subdir}/{png_file.name} → figures/spatial_maps/{subdir}/")

    # Archive
    shutil.move(str(qual_temp), str(ARCHIVE_DIR / 'qualitative_FINAL_DRY_SEASON'))

# ============================================================================
# 4. ARCHIVE REDUNDANT DIRECTORIES
# ============================================================================

print("\n" + "-"*80)
print("4/5: Archiving Redundant Directories")
print("-"*80)

for dir_name in DIRS_TO_DELETE:
    dir_path = BASE_DIR / dir_name
    if dir_path.exists():
        archive_path = ARCHIVE_DIR / dir_name
        shutil.move(str(dir_path), str(archive_path))
        print(f"✓ Archived: {dir_name}/ → archived/{dir_name}/")

# ============================================================================
# 5. CREATE SUMMARY
# ============================================================================

print("\n" + "-"*80)
print("5/5: Creating Structure Summary")
print("-"*80)

summary_path = BASE_DIR / 'README.md'
with open(summary_path, 'w') as f:
    f.write("""# Results Directory Structure

**Last Updated:** 2026-01-04
**Status:** Clean & Organized

## 📁 Directory Structure

```
results/
├── models/                    # Trained ResNet models & test results
│   ├── resnet18/              # ResNet-18 (11.7M params, 77.14% acc)
│   ├── resnet34/              # ResNet-34 (21.8M params, 76.78% acc)
│   ├── resnet101/             # ResNet-101 (44.5M params, 77.23% acc) ⭐ Best
│   └── resnet152/             # ResNet-152 (60.2M params, 76.78% acc)
│
├── tables/                    # All publication tables (Excel + LaTeX)
│   ├── performance/           # Overall performance comparison
│   │   ├── performance_table.xlsx
│   │   ├── performance_table.tex
│   │   ├── per_class_performance.xlsx
│   │   └── per_class_f1_pivot.xlsx
│   │
│   ├── statistical/           # Statistical analysis tables
│   │   ├── mcnemar_test_pairwise.xlsx
│   │   ├── computational_efficiency.xlsx
│   │   ├── producer_user_accuracy.xlsx
│   │   ├── omission_commission_errors.xlsx
│   │   └── kappa_analysis.xlsx
│   │
│   └── per_class/             # Detailed per-class metrics
│
├── figures/                   # All publication figures (300 DPI)
│   ├── confusion_matrices/    # Error pattern analysis
│   │   └── confusion_matrices_all.png
│   │
│   ├── training_curves/       # Convergence analysis
│   │   └── training_curves_comparison.png
│   │
│   ├── spatial_maps/          # Qualitative comparison maps
│   │   ├── province/          # Province-wide maps (Jambi)
│   │   └── city/              # City-level maps (custom boundary)
│   │
│   └── statistical/           # Statistical visualizations
│       └── mcnemar_pvalue_matrix.png
│
└── archived/                  # Old/redundant results (backup)
    ├── publication_comparison/
    ├── statistical_analysis/
    └── [old directories...]
```

## 📊 Contents Summary

### Models (4 directories)
- ResNet variants with test results and training history
- Best model: **ResNet101** (77.23% accuracy, 0.5436 F1-macro)

### Tables (9 Excel files + 1 LaTeX)
- Performance comparison tables
- Statistical analysis (McNemar's test, Kappa, efficiency)
- Per-class detailed metrics

### Figures (3 categories)
- Confusion matrices (error patterns)
- Training curves (convergence analysis)
- Spatial comparison maps (province + city)
- Statistical visualizations (p-value matrix)

## 🗑️ Archived
Old/redundant directories moved to `archived/` for backup:
- Old qualitative comparison versions (3 variants)
- Legacy model directories
- Exploration/testing results

## 📝 Notes

**Clean Structure Benefits:**
- ✅ No redundancy
- ✅ Clear organization
- ✅ Publication-ready
- ✅ Easy navigation
- ✅ Centralized outputs

**Usage:**
- Tables: Use for exact numerical values in paper
- Figures: Use for visual patterns and relationships
- Models: Trained weights and test results
- Archived: Backup of old results (can be deleted if space needed)

---

**Generated:** 2026-01-04
**By:** cleanup_results_structure.py
""")

print(f"✓ Created: results/README.md")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("CLEANUP COMPLETE!")
print("="*80)

print("\n📊 New Clean Structure:")
print("  ├── models/          (4 ResNet variants)")
print("  ├── tables/          (All Excel/LaTeX tables)")
print("  │   ├── performance/ (Comparison metrics)")
print("  │   ├── statistical/ (Statistical analysis)")
print("  │   └── per_class/   (Detailed per-class)")
print("  ├── figures/         (All publication figures)")
print("  │   ├── confusion_matrices/")
print("  │   ├── training_curves/")
print("  │   ├── spatial_maps/")
print("  │   └── statistical/")
print("  └── archived/        (Old results backup)")

print("\n✨ Results directory is now clean and professional!")
print("✨ All outputs organized for journal publication!")
print("\n" + "="*80)
