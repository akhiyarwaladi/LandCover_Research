# Land Cover Classification - Complete Workflow

**3-Step Modular Workflow for Reproducible Land Cover Classification**

---

## ⚡ Quick Start (3 Commands)

```bash
# Step 1: Verify data exists
python scripts/1_collect_data.py

# Step 2: Preprocess data (create training samples)
python scripts/2_preprocess_data.py

# Step 3: Train models and generate results
python scripts/3_run_classification.py
```

**Total Runtime:** ~3 minutes (Step 1: <1s, Step 2: ~2min, Step 3: ~15s)

---

## 📋 Detailed Workflow

### STEP 1: Data Collection Verification
```bash
python scripts/1_collect_data.py
```

**What it does:**
- Checks if KLHK ground truth exists (28,100 polygons)
- Checks if Sentinel-2 imagery exists (4 tiles, 2.7 GB)
- Reports file sizes and status

**Output:**
- ✅ or ❌ status for each required file
- Instructions if data is missing

**Runtime:** <1 second

---

### STEP 2: Data Preprocessing
```bash
python scripts/2_preprocess_data.py
```

**What it does:**
1. Loads KLHK reference data (28,100 polygons)
2. Loads and mosaics Sentinel-2 imagery (4 tiles)
3. Calculates 13 spectral indices (NDVI, EVI, NDWI, etc.)
4. Combines to 23 total features
5. Rasterizes KLHK ground truth
6. Extracts 100,000 training samples
7. Splits into train (80k) / test (20k)
8. Saves preprocessed data to disk

**Outputs:**
```
data/preprocessed/
├── features.npy           # Full spatial features (23 bands, ~18.5 GB)
├── labels.npy             # Full spatial labels (~400 MB)
├── profile.pkl            # Raster metadata
└── train_test_data.npz    # Training/test samples (7 MB)
```

**Runtime:** ~2 minutes

**Configuration:** Edit `scripts/2_preprocess_data.py`
```python
SAMPLE_SIZE = 100000  # Number of training samples
TEST_SIZE = 0.2       # Test set proportion
RANDOM_STATE = 42     # Random seed
```

---

### STEP 3: Classification
```bash
python scripts/3_run_classification.py
```

**What it does:**
1. Loads preprocessed train/test data (7 MB)
2. Trains 7 different classifiers
3. Evaluates performance metrics
4. Generates visualizations
5. Exports results to CSV

**Outputs:**
```
results/
├── classification_results.csv          # Performance metrics table
├── classifier_comparison.png           # Accuracy/F1 comparison
├── confusion_matrix_random_forest.png  # Confusion matrix
├── feature_importance_random_forest.png
├── feature_importance_extra_trees.png
├── feature_importance_lightgbm.png
└── feature_importance_decision_tree.png
```

**Runtime:** ~15 seconds

**Best Results:**
- **Model:** Random Forest
- **Accuracy:** 74.95%
- **F1-Score (macro):** 0.542
- **F1-Score (weighted):** 0.744

---

## 🎯 Expected Results

```
         Classifier  Accuracy  F1 (Macro)  F1 (Weighted)
      Random Forest   74.95%      0.542         0.744
        Extra Trees   73.47%      0.539         0.732
           LightGBM   70.51%      0.519         0.720
                SGD   68.45%      0.417         0.691
      Decision Tree   63.63%      0.428         0.650
Logistic Regression   55.77%      0.392         0.613
        Naive Bayes   49.16%      0.337         0.458
```

---

## 🔧 Customization

### Change Sample Size
Edit `scripts/2_preprocess_data.py`:
```python
SAMPLE_SIZE = 200000  # Increase for potentially better accuracy
# or
SAMPLE_SIZE = None    # Use all available data (~76M samples, very slow)
```

Then re-run Steps 2 and 3.

### Change Test Split
Edit `scripts/2_preprocess_data.py`:
```python
TEST_SIZE = 0.3  # 30% test set instead of 20%
```

### Enable Slow Models
Edit `scripts/3_run_classification.py`:
```python
INCLUDE_SLOW_MODELS = True  # Include XGBoost (currently disabled due to class label issue)
```

---

## 📁 Data Flow

```
┌─────────────────────────────────────────┐
│  RAW DATA (data/)                       │
├─────────────────────────────────────────┤
│  • KLHK GeoJSON (123 MB)               │
│  • Sentinel-2 TIFFs (2.7 GB)           │
└─────────────────────────────────────────┘
                  ↓
        ┌──────────────────┐
        │  STEP 1: VERIFY  │
        └──────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│  PREPROCESSING (Step 2)                 │
├─────────────────────────────────────────┤
│  • Load & mosaic imagery               │
│  • Calculate indices                    │
│  • Rasterize ground truth              │
│  • Extract samples                      │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│  PREPROCESSED DATA (data/preprocessed/) │
├─────────────────────────────────────────┤
│  • features.npy (~18.5 GB)             │
│  • labels.npy (~400 MB)                │
│  • train_test_data.npz (7 MB) ⭐       │
└─────────────────────────────────────────┘
                  ↓
        ┌──────────────────┐
        │  STEP 3: TRAIN   │
        └──────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│  RESULTS (results/)                     │
├─────────────────────────────────────────┤
│  • CSV metrics                          │
│  • Visualizations (6 plots)            │
└─────────────────────────────────────────┘
```

---

## ⚠️ Troubleshooting

### Error: "Preprocessed data not found"
```
Solution: Run Step 2 first
  python scripts/2_preprocess_data.py
```

### Error: "Data files missing"
```
Solution: Download data
  python scripts/download_klhk_kmz_partitioned.py
  python scripts/download_sentinel2.py
```

### Warning: "lbfgs failed to converge"
```
This is normal for Logistic Regression.
The model still completes with reasonable accuracy (55%).
No action needed.
```

### Low Memory
```
Solution: Reduce sample size in 2_preprocess_data.py
  SAMPLE_SIZE = 50000  # Instead of 100000
```

---

## 🔄 Re-running the Workflow

**To re-run classification with different parameters:**
```bash
# Option 1: Only re-run Step 3 (uses existing preprocessed data)
python scripts/3_run_classification.py

# Option 2: Re-run both Steps 2 and 3 (new sampling)
python scripts/2_preprocess_data.py
python scripts/3_run_classification.py
```

**To start completely fresh:**
```bash
# Delete preprocessed data
rm -rf data/preprocessed/

# Delete results
rm -rf results/

# Run all steps
python scripts/1_collect_data.py
python scripts/2_preprocess_data.py
python scripts/3_run_classification.py
```

---

## 📊 Performance Benchmarks

**Hardware Used:** (Your system specs)
- CPU: (varies)
- RAM: Requires ~20 GB for full preprocessing

**Timing:**
| Step | Duration | Bottleneck |
|------|----------|------------|
| 1. Verify | <1 second | I/O |
| 2. Preprocess | ~2 minutes | Feature calculation |
| 3. Classify | ~15 seconds | Model training |
| **Total** | **~3 minutes** | - |

---

## 🎯 Next Steps

After successful classification:

1. **View Results:**
   ```bash
   cd results/
   ls *.png *.csv
   ```

2. **Analyze Confusion Matrix:**
   - Check which classes are confused
   - Identify areas for improvement

3. **Feature Importance:**
   - See which spectral bands/indices matter most
   - Consider feature selection

4. **Hyperparameter Tuning:**
   - Grid search for Random Forest
   - Potentially improve accuracy by 2-5%

5. **Full Spatial Prediction:**
   - Apply best model to entire province
   - Generate classification map

---

**For more details, see:**
- `README.md` - Project overview
- `CLAUDE.md` - Complete technical documentation
- `modules/README.md` - Module API documentation
