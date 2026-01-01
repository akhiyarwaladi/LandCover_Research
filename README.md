# Land Cover Classification - Provinsi Jambi 2024

Supervised land cover classification menggunakan Sentinel-2 imagery dan KLHK (Kementerian Lingkungan Hidup dan Kehutanan) reference data.

## 🎯 Project Overview

**Objective:** Klasifikasi tutupan lahan Provinsi Jambi menggunakan:
- **Input:** Sentinel-2 satellite imagery (10 bands, 20m resolution)
- **Ground Truth:** KLHK PL2024 official land cover data
- **Method:** Supervised machine learning classification
- **Output:** Land cover map dengan 7 classes + accuracy metrics

**Status:** ✅ **COMPLETED** - Achieving 74.95% accuracy with Random Forest

---

## ⚡ Quick Start - 3-Step Workflow

```bash
# 1. Verify data exists
python scripts/1_collect_data.py

# 2. Preprocess data (~2 minutes)
python scripts/2_preprocess_data.py

# 3. Train models & generate results (~15 seconds)
python scripts/3_run_classification.py
```

**See:** `WORKFLOW.md` for detailed workflow documentation

---

## 📊 Results Summary

| Metric | Value |
|--------|-------|
| **Best Classifier** | Random Forest |
| **Accuracy** | 74.95% |
| **F1-Score (macro)** | 0.5421 |
| **Training Samples** | 100,000 pixels |
| **Total Valid Pixels** | 76.4 million |
| **Coverage** | 58.3% of Jambi Province |
| **Training Time** | 4.15 seconds |

**Land Cover Classes:**
- Water (79% F1-score)
- Trees/Forest (74% F1-score)
- Crops/Agriculture (78% F1-score)
- Shrub/Scrub (37% F1-score)
- Built Area (42% F1-score)
- Bare Ground (15% F1-score)

---

## 🗂️ Repository Structure

```
LandCover_Research/
│
├── 📄 README.md                    # This file - Project overview
├── 📄 CLAUDE.md                    # Comprehensive project documentation
├── 📄 STATUS.md                    # Quick project status
├── 📄 FINAL_SUMMARY.md            # Completion summary & results
├── 📄 environment.yml              # Conda environment specification
│
├── 📁 docs/                        # 📚 Documentation
│   ├── ALTERNATIVE_APPROACH.md     # Unsupervised alternatives
│   ├── KLHK_DATA_ISSUE.md         # Geometry access issue analysis
│   ├── KLHK_MANUAL_DOWNLOAD.md    # Manual download guide
│   ├── GET_KLHK_GEOMETRY.md       # Guide to obtain KLHK geometry
│   └── RESEARCH_NOTES.md          # Research notes and findings
│
├── 📁 modules/                     # 🧩 Modular Components
│   ├── README.md                   # Module documentation
│   ├── __init__.py                 # Package initialization
│   ├── data_loader.py              # KLHK & Sentinel-2 loading
│   ├── feature_engineering.py      # Spectral indices calculation
│   ├── preprocessor.py             # Rasterization & data prep
│   ├── model_trainer.py            # ML training & evaluation
│   └── visualizer.py               # Plots & reports generation
│
├── 📁 scripts/                     # 🚀 Production Scripts
│   ├── run_classification.py              # ⭐ Main orchestrator
│   ├── download_klhk_kmz_partitioned.py  # KLHK data download
│   ├── download_sentinel2.py              # Sentinel-2 download
│   └── parse_klhk_kmz.py                 # KMZ file parser
│
├── 📁 utils/                       # 🔧 Utility Scripts
│   ├── README.md                   # Utils documentation
│   ├── verify_final_dataset.py     # Dataset verification
│   ├── verify_geojson.py           # GeoJSON structure check
│   ├── verify_partitions.py        # Partition uniqueness check
│   └── compare_batches.py          # Batch comparison
│
├── 📁 tests/                       # 🧪 Tests & Legacy Code
│   ├── README.md                   # Tests documentation
│   ├── debug_geometry.py           # Geometry issue investigation
│   ├── test_geojson_vs_kmz.py     # Format comparison test
│   └── [other test & legacy files]
│
├── 📁 data/                        # 💾 Data Directory
│   ├── klhk/                       # KLHK reference data
│   │   ├── KLHK_PL2024_Jambi_Full_WithGeometry.geojson  # ⭐ 28,100 polygons
│   │   └── partitions/             # Download partitions
│   └── sentinel/                   # Sentinel-2 imagery (2.7 GB)
│       └── S2_jambi_2024_20m_AllBands-*.tif
│
└── 📁 results/                     # 📈 Classification Results
    ├── classification_results.csv
    ├── classifier_comparison.png
    ├── confusion_matrix_random_forest.png
    └── feature_importance_*.png
```

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate landcover_jambi
```

### 2. Download Data (Optional - Already Downloaded)

```bash
# Download KLHK data with geometry
python scripts/download_klhk_kmz_partitioned.py

# Download Sentinel-2 imagery
python scripts/download_sentinel2.py
```

### 3. Run Classification

```bash
# Run complete classification workflow
python scripts/run_classification.py
```

**Output:** Results saved to `results/` directory

---

## 📦 Data

### KLHK Reference Data
- **File:** `data/klhk/KLHK_PL2024_Jambi_Full_WithGeometry.geojson`
- **Records:** 28,100 polygons
- **Classes:** 20 KLHK classes → simplified to 7 classes
- **Coverage:** Provinsi Jambi complete
- **Format:** GeoJSON with full geometry
- **CRS:** EPSG:4326

**How We Got It:**
- KLHK REST API blocks geometry for `f=geojson`
- ✅ Solution: Use `f=kmz` export format
- ✅ Method: Partitioned download using WHERE clauses on OBJECTID ranges

### Sentinel-2 Imagery
- **Files:** 4 tiles (total 2.7 GB)
- **Bands:** B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12 (10 bands)
- **Resolution:** 20 meters
- **Source:** Google Earth Engine
- **Year:** 2024
- **CRS:** EPSG:4326

---

## 🧩 Modular Architecture

### Production Workflow

```python
from modules.data_loader import load_klhk_data, load_sentinel2_tiles
from modules.feature_engineering import calculate_spectral_indices
from modules.preprocessor import rasterize_klhk, prepare_training_data
from modules.model_trainer import train_all_models
from modules.visualizer import generate_all_plots

# 1. Load data
klhk = load_klhk_data('data/klhk/KLHK_PL2024_Jambi_Full_WithGeometry.geojson')
s2_bands, profile = load_sentinel2_tiles([...])

# 2. Calculate features
indices = calculate_spectral_indices(s2_bands)
features = combine_bands_and_indices(s2_bands, indices)

# 3. Prepare training data
labels = rasterize_klhk(klhk, profile)
X, y = prepare_training_data(features, labels)

# 4. Train models
results = train_all_models(X_train, y_train, X_test, y_test)

# 5. Visualize
generate_all_plots(results, feature_names)
```

**See:** `modules/README.md` for detailed module documentation

---

## 📊 Features

### Sentinel-2 Bands (10)
- B2 (Blue), B3 (Green), B4 (Red)
- B5, B6, B7 (Red Edge)
- B8 (NIR), B8A (Red Edge 4)
- B11 (SWIR1), B12 (SWIR2)

### Spectral Indices (13)
- **Vegetation:** NDVI, EVI, SAVI, MSAVI, GNDVI
- **Water:** NDWI, MNDWI
- **Built-up:** NDBI, BSI
- **Red Edge:** NDRE, CIRE
- **Moisture:** NDMI, NBR

**Total:** 23 features per pixel

---

## 🤖 Models

Trained and evaluated 7 classifiers:

1. **Random Forest** ⭐ - Best performer (74.95% accuracy)
2. **Extra Trees** - Fast and accurate (73.47%)
3. **LightGBM** - Gradient boosting (70.51%)
4. **SGD Classifier** - Fast linear model (68.45%)
5. **Decision Tree** - Interpretable (63.63%)
6. **Logistic Regression** - Baseline (55.77%)
7. **Naive Bayes** - Very fast (49.16%)

---

## 📖 Documentation

### Primary Docs
- **`README.md`** (this file) - Quick start & overview
- **`CLAUDE.md`** - Comprehensive project documentation
- **`FINAL_SUMMARY.md`** - Complete results & achievements

### Guides
- **`docs/GET_KLHK_GEOMETRY.md`** - How to obtain KLHK data with geometry
- **`docs/KLHK_DATA_ISSUE.md`** - Technical analysis of geometry restriction
- **`docs/KLHK_MANUAL_DOWNLOAD.md`** - Alternative download methods
- **`docs/ALTERNATIVE_APPROACH.md`** - Unsupervised alternatives

### Module Docs
- **`modules/README.md`** - Module architecture & usage
- **`utils/README.md`** - Utility scripts documentation
- **`tests/README.md`** - Test scripts & legacy code

---

## 🔧 Configuration

Edit `scripts/run_classification.py` to configure:

```python
# Input paths
KLHK_PATH = 'data/klhk/KLHK_PL2024_Jambi_Full_WithGeometry.geojson'
SENTINEL2_TILES = [...]  # List of tile paths

# Sampling
SAMPLE_SIZE = 100000  # or None for all data
TEST_SIZE = 0.2       # 20% test split

# Training
INCLUDE_SLOW_MODELS = False  # Include XGBoost, etc.
```

---

## 📈 Results

### Classification Metrics

All results saved in `results/`:

- **CSV:** `classification_results.csv` - Metrics table
- **Plots:**
  - `classifier_comparison.png` - Performance comparison
  - `confusion_matrix_random_forest.png` - Confusion matrix
  - `feature_importance_*.png` - Feature importance (4 models)

### View Results

```bash
# Open results directory
cd results/

# View CSV
cat classification_results.csv

# View plots (requires image viewer)
open *.png
```

---

## 🛠️ Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Ensure running from project root
cd /path/to/LandCover_Research
python scripts/run_classification.py
```

**2. Memory Issues**
```python
# Reduce sample size in run_classification.py
SAMPLE_SIZE = 50000  # Instead of 100000
```

**3. Missing Dependencies**
```bash
conda install -c conda-forge lightgbm xgboost
```

**4. KLHK Geometry Still NULL**
- ✅ Use `f=kmz` format, not `f=geojson`
- ✅ Use partitioned download script
- See: `docs/GET_KLHK_GEOMETRY.md`

---

## 📚 References

### Data Sources
- **KLHK:** https://geoportal.menlhk.go.id/
- **Sentinel-2:** Google Earth Engine
- **Admin Boundaries:** Indonesia Geospasial

### Methods
- Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.
- Standard spectral indices formulations
- Supervised classification best practices

---

## ✅ Status

**Current Status:** ✅ **PRODUCTION READY**

- [x] Data acquisition complete (28,100 KLHK polygons + 2.7GB Sentinel-2)
- [x] Feature engineering implemented (23 features)
- [x] Model training successful (7 classifiers)
- [x] Evaluation metrics computed
- [x] Visualizations generated
- [x] Modular architecture created
- [x] Comprehensive documentation
- [x] Repository cleaned and organized

**Next Steps:**
- Fine-tune hyperparameters
- Implement spatial cross-validation
- Generate full land cover map
- Extend to other provinces

---

**Author:** Research Project
**Date:** January 2026
**Version:** 1.0.0

🎉 **Project Complete & Ready for Use!**
