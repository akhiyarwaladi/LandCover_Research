# Land Cover Classification Modules

Arsitektur modular untuk supervised land cover classification menggunakan Sentinel-2 dan data referensi KLHK.

## Struktur Modular

```
modules/
├── __init__.py              # Package initialization
├── data_loader.py           # Loading KLHK dan Sentinel-2 data
├── feature_engineering.py   # Kalkulasi spectral indices
├── preprocessor.py          # Rasterization dan data preparation
├── model_trainer.py         # Training dan evaluation model
└── visualizer.py            # Generasi plots dan reports
```

## Module Descriptions

### 1. `data_loader.py`
**Fungsi:** Loading data KLHK dan Sentinel-2

**Key Functions:**
- `load_klhk_data(geojson_path)` - Load KLHK reference polygons
- `load_sentinel2_tiles(tile_paths)` - Load dan mosaic Sentinel-2 tiles
- `get_sentinel2_band_names()` - Get standard band names

**Input:**
- KLHK GeoJSON file dengan geometry
- Sentinel-2 GeoTIFF tiles (multiple files)

**Output:**
- GeoDataFrame dengan KLHK polygons + simplified classes
- Mosaicked Sentinel-2 array + raster profile

### 2. `feature_engineering.py`
**Fungsi:** Kalkulasi spectral indices dari Sentinel-2 bands

**Key Functions:**
- `calculate_spectral_indices(sentinel2_data)` - Calculate 13 spectral indices
- `combine_bands_and_indices(bands, indices)` - Combine bands + indices
- `get_index_names()` - Get index names
- `get_all_feature_names()` - Get all feature names (bands + indices)

**Indices Calculated:**
- Vegetation: NDVI, EVI, SAVI, MSAVI, GNDVI
- Water: NDWI, MNDWI
- Built-up: NDBI, BSI
- Red Edge: NDRE, CIRE
- Moisture: NDMI, NBR

**Input:**
- Sentinel-2 bands array (10 bands)

**Output:**
- Spectral indices array (13 indices)
- Combined features (23 total features)

### 3. `preprocessor.py`
**Fungsi:** Rasterisasi KLHK dan persiapan training data

**Key Functions:**
- `rasterize_klhk(gdf, reference_profile)` - Rasterize KLHK vector ke raster grid
- `prepare_training_data(features, labels)` - Extract training samples
- `split_train_test(X, y)` - Split data dengan stratification

**Input:**
- KLHK GeoDataFrame
- Sentinel-2 profile (untuk grid matching)
- Feature data + label raster

**Output:**
- Rasterized KLHK labels
- Training samples (X, y)
- Train/test splits

### 4. `model_trainer.py`
**Fungsi:** Training dan evaluasi multiple classifiers

**Key Functions:**
- `get_classifiers(include_slow)` - Get configured classifiers
- `train_single_model(...)` - Train satu model
- `train_all_models(...)` - Train semua models
- `get_best_model(results)` - Get best performing model

**Classifiers:**
- Random Forest ⭐
- Extra Trees
- Logistic Regression
- Decision Tree
- Naive Bayes
- SGD Classifier
- LightGBM (if available) ⭐
- XGBoost (if available)

**Input:**
- X_train, y_train, X_test, y_test

**Output:**
- Results dictionary dengan metrics untuk semua models
- Best model identifier

### 5. `visualizer.py`
**Fungsi:** Generasi visualizations dan reports

**Key Functions:**
- `plot_classifier_comparison(results)` - Bar chart performance comparison
- `plot_confusion_matrix(y_test, y_pred)` - Confusion matrix heatmap
- `plot_feature_importance(pipeline, feature_names)` - Feature importance plot
- `export_results_to_csv(results)` - Export metrics ke CSV
- `generate_all_plots(results, feature_names)` - Generate semua plots

**Outputs:**
- `classifier_comparison.png` - Performance bar charts
- `confusion_matrix_*.png` - Confusion matrix untuk best model
- `feature_importance_*.png` - Feature importance untuk tree models
- `classification_results.csv` - Summary metrics table

## Usage

### Basic Usage (via main script):

```bash
python scripts/run_classification.py
```

### Advanced Usage (import modules):

```python
from modules.data_loader import load_klhk_data, load_sentinel2_tiles
from modules.feature_engineering import calculate_spectral_indices
from modules.preprocessor import rasterize_klhk, prepare_training_data
from modules.model_trainer import train_all_models
from modules.visualizer import generate_all_plots

# Load data
klhk = load_klhk_data('data/klhk/...')
s2_bands, profile = load_sentinel2_tiles(['tile1.tif', 'tile2.tif'])

# Calculate features
indices = calculate_spectral_indices(s2_bands)
features = combine_bands_and_indices(s2_bands, indices)

# Prepare training data
labels = rasterize_klhk(klhk, profile)
X, y = prepare_training_data(features, labels)

# Train models
results = train_all_models(X_train, y_train, X_test, y_test)

# Generate visualizations
generate_all_plots(results, feature_names)
```

## Configuration

Edit constants di `scripts/run_classification.py`:

```python
# Input paths
KLHK_PATH = 'path/to/klhk.geojson'
SENTINEL2_TILES = ['tile1.tif', 'tile2.tif', ...]

# Sampling
SAMPLE_SIZE = 100000  # or None untuk semua data
TEST_SIZE = 0.2

# Model training
INCLUDE_SLOW_MODELS = True  # Include XGBoost, etc.
```

## Data Requirements

### KLHK Data:
- Format: GeoJSON with geometry
- Required columns: Land cover code (PL2024_ID, etc.)
- CRS: Any (akan otomatis diproyeksikan)

### Sentinel-2 Data:
- Format: GeoTIFF (10 bands)
- Bands: B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12
- Resolution: 20m recommended
- Multiple tiles supported (will be mosaicked)

## Extensibility

### Add New Spectral Indices:

Edit `feature_engineering.py`:

```python
def calculate_spectral_indices(data):
    # Existing indices...

    # Add new index
    my_index = (band1 - band2) / (band1 + band2 + epsilon)

    # Add to stack
    indices = np.stack([
        ndvi, evi, ..., my_index  # Add here
    ])

    return indices

def get_index_names():
    return [
        'NDVI', 'EVI', ..., 'MyIndex'  # Add name here
    ]
```

### Add New Classifier:

Edit `model_trainer.py`:

```python
def get_classifiers(include_slow=True):
    classifiers = {}

    # Existing classifiers...

    # Add new classifier
    classifiers['My Classifier'] = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler()),
        ('classifier', MyClassifier(...))
    ])

    return classifiers
```

### Add New Visualization:

Edit `visualizer.py`:

```python
def plot_my_visualization(results, save_dir):
    # Create plot
    fig, ax = plt.subplots()
    # ... plotting code ...

    # Save
    path = f'{save_dir}/my_plot.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

    return path

# Add to generate_all_plots()
def generate_all_plots(results, ...):
    # Existing plots...

    # Add new plot
    path = plot_my_visualization(results, save_dir)
    saved_plots.append(path)
```

## Benefits of Modular Structure

✅ **Maintainability** - Easy to update individual components
✅ **Testability** - Each module can be tested independently
✅ **Reusability** - Modules can be reused across projects
✅ **Clarity** - Clear separation of concerns
✅ **Extensibility** - Easy to add new features/models
✅ **Documentation** - Each module is self-contained

## Troubleshooting

### Import Errors:
Ensure you're running from project root:
```bash
cd /path/to/LandCover_Research
python scripts/run_classification.py
```

### Memory Issues:
Reduce `SAMPLE_SIZE`:
```python
SAMPLE_SIZE = 50000  # Instead of 100000
```

### Missing Dependencies:
Install via conda:
```bash
conda install -c conda-forge lightgbm xgboost
```

---

**Author:** Land Cover Research Project
**Date:** January 2026
