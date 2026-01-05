# Scripts Directory

**Total Scripts:** 12 active scripts (+ 6 legacy scripts in `legacy/`)

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

## 🧠 Deep Learning Training (1 script)

### 5. `train_models.py` ⭐ **MAIN TRAINING SCRIPT**
**Purpose:** Train all deep learning models for land cover classification
**Output:** `models/{model_name}_best.pth` + `results/models/{model_name}/`
**Models:**
- ResNet-50 (baseline, 23.5M params)
- EfficientNet-B3 (efficient, 10.7M params)
- ConvNeXt-Tiny (modern CNN, 28.6M params)
- DenseNet-121 (lightweight, 7.0M params)
- Swin-Tiny (transformer, 28.3M params)

**Features:**
- 23 multispectral features (10 bands + 13 indices)
- 100,000 training patches (32×32)
- Automatic checkpointing (best validation accuracy)
- Comprehensive evaluation metrics

**Usage:**
```bash
python scripts/train_models.py
```

---

## 📊 Training Monitoring (3 scripts)

### 6. `monitor_training.py`
**Purpose:** Real-time monitoring of training progress
**Output:** Console display with epoch, accuracy, loss, ETA
**Usage:**
```bash
python scripts/monitor_training.py
```

### 7. `watch_training.py`
**Purpose:** Continuous auto-refresh monitoring
**Output:** Live terminal updates
**Usage:**
```bash
python scripts/watch_training.py
```

### 8. `plot_training_progress.py`
**Purpose:** Generate training progress visualizations
**Output:** `results/models/{model_name}/training_curves.png`
**Features:**
- Loss curves (train & validation)
- Accuracy curves (train & validation)
- Per-epoch timing analysis
**Usage:**
```bash
python scripts/plot_training_progress.py
```

---

## 📊 Publication Outputs (4 scripts)

### 9. `generate_publication_comparison.py`
**Purpose:** Generate publication-quality comparison tables and charts
**Output:** `results/publication_comparison/` - Excel tables, PNG charts, LaTeX tables
**Features:**
- Model performance comparison (accuracy, F1, parameters, FLOPs)
- Confusion matrices for all models
- Per-class F1-score comparison
- Training time analysis
- LaTeX tables for journal submission
**Usage:**
```bash
python scripts/generate_publication_comparison.py
```

### 10. `generate_statistical_analysis.py`
**Purpose:** Perform statistical significance testing between models
**Output:** `results/statistical_analysis/` - Test results, confidence intervals
**Features:**
- McNemar's test for paired model comparison
- Confidence intervals (95%)
- Effect size calculation (Cohen's h)
- Pairwise significance matrix
**Usage:**
```bash
python scripts/generate_statistical_analysis.py
```

### 11. `generate_per_class_f1_chart.py`
**Purpose:** Generate grouped bar chart for per-class F1-scores
**Output:** `results/per_class_f1_comparison.png`
**Features:**
- Per-class performance comparison across all models
- Colorblind-friendly palette
- Journal-standard formatting (300 DPI)
**Usage:**
```bash
python scripts/generate_per_class_f1_chart.py
```

### 12. `generate_qualitative_comparison.py`
**Purpose:** Generate spatial comparison maps (ground truth vs predictions)
**Output:** `results/qualitative_comparison/` - Province & city-level maps
**Features:**
- Visual comparison of model predictions vs ground truth
- Custom administrative boundary support
- Jambi-optimized land cover color scheme
- Publication-ready (300 DPI)
**Usage:**
```bash
python scripts/generate_qualitative_comparison.py
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
│
├── 📥 Data Download & Preparation
│   ├── download_klhk_kmz_partitioned.py     # KLHK download
│   ├── download_sentinel2.py                 # Sentinel-2 download
│   ├── parse_klhk_kmz.py                    # KMZ parser
│   └── crop_sentinel_custom_boundary.py      # Boundary cropping
│
├── 🧠 Deep Learning Training
│   └── train_models.py                       # ⭐ Main training script
│
├── 📊 Training Monitoring
│   ├── monitor_training.py                   # Real-time monitoring
│   ├── watch_training.py                     # Auto-refresh monitoring
│   └── plot_training_progress.py             # Training visualizations
│
├── 📊 Publication Outputs
│   ├── generate_publication_comparison.py    # Comparison tables/charts
│   ├── generate_statistical_analysis.py      # Statistical testing
│   ├── generate_per_class_f1_chart.py       # Per-class F1 chart
│   └── generate_qualitative_comparison.py    # Spatial comparison maps
│
└── legacy/                                   # Old scripts (archived)
    ├── run_classification.py                 # Old Random Forest pipeline
    ├── run_classification_with_aoa.py        # RF with AOA
    ├── run_resnet_training.py                # Early ResNet experiment
    ├── run_resnet_prediction.py              # Early ResNet prediction
    ├── run_resnet_visualization.py           # Early ResNet visualization
    └── cleanup_results_structure.py          # Old cleanup script
```

---

## 🔄 Typical Workflows

### Complete Deep Learning Workflow (Recommended)
```bash
# 1. Download data (one-time setup)
python scripts/download_klhk_kmz_partitioned.py
python scripts/download_sentinel2.py

# 2. Train all models (~4-5 hours total)
python scripts/train_models.py

# 3. Monitor training (in another terminal)
python scripts/monitor_training.py
# or
python scripts/watch_training.py

# 4. Generate publication outputs (after training completes)
python scripts/generate_publication_comparison.py
python scripts/generate_statistical_analysis.py
python scripts/generate_per_class_f1_chart.py
python scripts/generate_qualitative_comparison.py

# 5. Plot training curves
python scripts/plot_training_progress.py
```

### Quick Monitoring During Training
```bash
# Check current status
python scripts/monitor_training.py

# Or watch continuously
python scripts/watch_training.py
```

---

## ✨ Recent Changes

**Date:** 2026-01-05
**Changes:**
- Organized legacy scripts into `legacy/` subdirectory
- Updated documentation to reflect deep learning approach
- Cleaned up temporary log files
- Focused on production-ready scripts only

**Legacy Scripts Moved:**
- `run_classification.py` → `legacy/` (old Random Forest approach)
- `run_classification_with_aoa.py` → `legacy/`
- `run_resnet_training.py` → `legacy/` (early ResNet experiment)
- `run_resnet_prediction.py` → `legacy/`
- `run_resnet_visualization.py` → `legacy/`
- `cleanup_results_structure.py` → `legacy/`

**Result:** Clean, organized, production-ready codebase with clear separation between active and legacy scripts!

---

## 📋 Output Locations

- **Trained models:** `models/{model_name}_best.pth`
- **Training history:** `results/models/{model_name}/training_history.npz`
- **Evaluation results:** `results/models/{model_name}/evaluation_results.json`
- **Confusion matrices:** `results/models/{model_name}/confusion_matrix.png`
- **Publication outputs:** `results/publication_comparison/`, `results/statistical_analysis/`
- **Qualitative maps:** `results/qualitative_comparison/`

---

**Last Updated:** 2026-01-05
**Maintained By:** Claude Sonnet 4.5
