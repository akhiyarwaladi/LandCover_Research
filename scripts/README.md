# Production Scripts Index

**Total Scripts:** 13 (down from 51 - cleaned 74% redundancy)

This directory contains all production-ready scripts for the Land Cover Research project. All scripts follow standardized `snake_case` naming with clear action verbs.

---

## 📥 Data Download & Preparation (3 scripts)

### 1. `download_klhk_kmz_partitioned.py`
**Purpose:** Download KLHK ground truth data using KMZ format (29 partitions)
**Output:** `data/klhk/KLHK_PL2024_Jambi_Full_WithGeometry.geojson`
**Usage:**
```bash
python scripts/download_klhk_kmz_partitioned.py
```

### 2. `download_sentinel2.py`
**Purpose:** Download Sentinel-2 satellite imagery via Google Earth Engine
**Output:** `data/sentinel_new_cloudfree/*.tif` (4 tiles)
**Usage:**
```bash
python scripts/download_sentinel2.py --mode full
```

### 3. `parse_klhk_kmz.py`
**Purpose:** Parse KMZ files to GeoJSON format
**Output:** GeoJSON files with full geometry
**Usage:**
```bash
python scripts/parse_klhk_kmz.py <input.kmz> <output.geojson>
```

---

## 🗺️ Spatial Preprocessing (1 script)

### 4. `crop_sentinel_custom_boundary.py`
**Purpose:** Crop Sentinel-2 data to custom administrative boundary (13 sub-districts with clipped corners)
**Output:** Custom boundary GeoJSON
**Features:**
- 13 sub-districts (11 original + SungaiGelam + Sekernan)
- Clipped corners (lat > -1.9, lon < 103.8)
- Area: ~2,252 km²
**Usage:**
```bash
python scripts/crop_sentinel_custom_boundary.py
```

---

## 🤖 Machine Learning Workflows (2 scripts)

### 5. `run_classification.py`
**Purpose:** Run complete Random Forest classification pipeline
**Output:** `results/` - Classification results, visualizations, metrics
**Features:**
- 7 classifiers (Random Forest, Extra Trees, LightGBM, etc.)
- 23 features (10 bands + 13 spectral indices)
- 100,000 training samples
**Usage:**
```bash
python scripts/run_classification.py
```

### 6. `run_classification_with_aoa.py`
**Purpose:** Classification with Area of Applicability (AOA) analysis
**Output:** `results/` - Results + AOA maps
**Note:** Computationally intensive for large areas
**Usage:**
```bash
python scripts/run_classification_with_aoa.py
```

---

## 🧠 Deep Learning Workflows (4 scripts)

### 7. `train_all_resnet_variants.py`
**Purpose:** Train all 4 ResNet variants (18, 34, 101, 152)
**Output:** `results/{variant}/` - Models, history, test results
**Features:**
- Parallel training support
- Automatic checkpointing
- Comprehensive logging
**Usage:**
```bash
python scripts/train_all_resnet_variants.py
```

### 8. `run_resnet_training.py`
**Purpose:** Train single ResNet variant
**Output:** `results/{variant}/` - Model weights, training history
**Usage:**
```bash
python scripts/run_resnet_training.py --variant resnet50
```

### 9. `run_resnet_prediction.py`
**Purpose:** Generate spatial predictions using trained ResNet model
**Output:** `results/{variant}/predictions.npy` - Full spatial classification
**Usage:**
```bash
python scripts/run_resnet_prediction.py --variant resnet101
```

### 10. `run_resnet_visualization.py`
**Purpose:** Generate comprehensive visualizations for ResNet results
**Output:** `results/{variant}/visualizations/` - Training curves, confusion matrices, spatial maps
**Usage:**
```bash
# Single variant
python scripts/run_resnet_visualization.py --variant resnet101

# All variants
python scripts/run_resnet_visualization.py --all
```

---

## 📊 Publication Outputs (3 scripts)

### 11. `generate_qualitative_comparison.py`
**Purpose:** Generate spatial comparison maps (ground truth vs predictions)
**Output:** `results/qualitative_FINAL_DRY_SEASON/` - Province & city maps
**Features:**
- Province-wide AND city-level maps
- Custom administrative boundary support
- Jambi-optimized color scheme
- Publication-ready (300 DPI)
**Usage:**
```bash
python scripts/generate_qualitative_comparison.py
```

### 12. `generate_publication_comparison.py`
**Purpose:** Generate publication-quality comparison tables and charts
**Output:** `results/publication_comparison/` - Excel tables, PNG charts
**Features:**
- Beautiful Excel formatting with auto-adjusted columns
- Confusion matrices for all models
- Per-class F1 comparison
- Training curves comparison
- LaTeX tables
**Usage:**
```bash
python scripts/generate_publication_comparison.py
```

### 13. `generate_per_class_f1_chart.py`
**Purpose:** Generate grouped bar chart for per-class F1-scores
**Output:** `results/per_class_f1_comparison.png`
**Features:**
- Colorblind-friendly palette
- Journal-standard formatting
- Publication-ready (300 DPI)
**Usage:**
```bash
python scripts/generate_per_class_f1_chart.py
```

---

## 📋 Script Naming Convention

All scripts follow standardized naming:

**Action Verbs:**
- `download_` - Data acquisition
- `parse_` - Data parsing/conversion
- `crop_` - Spatial preprocessing
- `run_` - Execute workflows
- `train_` - Model training
- `generate_` - Output generation

**Format:** `{action}_{subject}_{modifier}.py`

**Examples:**
- ✅ `download_sentinel2.py` - Clear, concise
- ✅ `run_classification.py` - Standard format
- ✅ `generate_qualitative_comparison.py` - Descriptive
- ❌ `test_something.py` - Test scripts removed
- ❌ `quick_check.py` - Debug scripts removed
- ❌ `generate_FINAL.py` - Caps removed

---

## 🗂️ Directory Structure

```
scripts/
├── README.md                             # This file
├── download_klhk_kmz_partitioned.py     # KLHK download
├── download_sentinel2.py                 # Sentinel-2 download
├── parse_klhk_kmz.py                    # KMZ parser
├── crop_sentinel_custom_boundary.py      # Boundary cropping
├── run_classification.py                 # ML classification
├── run_classification_with_aoa.py        # ML with AOA
├── train_all_resnet_variants.py         # Train all ResNets
├── run_resnet_training.py                # Train single ResNet
├── run_resnet_prediction.py              # ResNet prediction
├── run_resnet_visualization.py           # ResNet visualizations
├── generate_qualitative_comparison.py    # Spatial comparison maps
├── generate_publication_comparison.py    # Publication tables/charts
└── generate_per_class_f1_chart.py       # Per-class F1 chart
```

---

## 🔄 Typical Workflows

### Complete ML Workflow
```bash
# 1. Download data
python scripts/download_klhk_kmz_partitioned.py
python scripts/download_sentinel2.py

# 2. Run classification
python scripts/run_classification.py

# 3. Generate visualizations
python scripts/generate_qualitative_comparison.py
python scripts/generate_publication_comparison.py
```

### Complete Deep Learning Workflow
```bash
# 1. Train all variants
python scripts/train_all_resnet_variants.py

# 2. Generate predictions
python scripts/run_resnet_prediction.py --variant resnet101

# 3. Visualize results
python scripts/run_resnet_visualization.py --all

# 4. Publication outputs
python scripts/generate_publication_comparison.py
python scripts/generate_qualitative_comparison.py
```

---

## ✨ Recent Cleanup

**Date:** 2026-01-04
**Deleted:** 38 redundant scripts (74% reduction)
**Standardized:** 3 script names

**Removed Categories:**
- Old workflow scripts (1_, 2_, 3_)
- Test/debug scripts (test_*, check_*, quick_*, diagnose_*)
- Exploration scripts (explore_*, find_*)
- Legacy cropping variants (9 different approaches → 1 final)
- Redundant table generators (3 scripts → 1)
- Redundant visualization scripts (18 scripts → 3)

**Result:** Clean, maintainable, production-ready codebase!

---

**Last Updated:** 2026-01-04
**Maintained By:** Claude Sonnet 4.5
