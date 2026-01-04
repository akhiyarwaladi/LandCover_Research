# Results Directory Structure

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
