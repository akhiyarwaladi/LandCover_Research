# Deep Learning Modules

Modular components for deep learning land cover classification.

## Module Structure

### Core Modules

1. **`dl_predictor.py`** - Spatial Prediction
   - `load_resnet_model()` - Load trained ResNet model
   - `normalize_features()` - Normalize features using training stats
   - `predict_patches()` - Batch prediction on patches
   - `calculate_accuracy()` - Calculate prediction accuracy
   - `predict_spatial()` - Complete prediction pipeline

2. **`dl_visualizer.py`** - Visualization
   - `plot_training_curves()` - Training loss and accuracy plots
   - `plot_confusion_matrix()` - Confusion matrix heatmap
   - `plot_model_comparison()` - Compare multiple models
   - `plot_spatial_predictions()` - Spatial prediction maps
   - `generate_all_visualizations()` - Complete visualization pipeline

3. **`data_preparation.py`** - Data Preparation
   - `extract_patches()` - Extract image patches from raster
   - `LandCoverPatchDataset` - PyTorch dataset class
   - `get_data_loaders()` - Create train/val/test loaders

4. **`deep_learning_trainer.py`** - Model Training (existing)
   - `train_resnet_model()` - Train ResNet with proper configuration
   - Handles normalization, data splits, training loop

## Centralized Run Scripts

### Training
```bash
python scripts/run_resnet_training.py
```
**Outputs:**
- `models/resnet50_best.pth` - Best trained model
- `results/resnet/training_history.npz` - Training curves
- `results/resnet/test_results.npz` - Test predictions
- `results/resnet/normalization_params.npz` - Feature normalization parameters

### Spatial Prediction
```bash
python scripts/run_resnet_prediction.py
```
**Outputs:**
- `results/resnet/predictions.npy` - Prediction array (NumPy)
- `results/resnet/predictions.tif` - Prediction raster (GeoTIFF)

### Visualization
```bash
python scripts/run_resnet_visualization.py
```
**Outputs:**
- `results/resnet/visualizations/training_curves.png`
- `results/resnet/visualizations/confusion_matrix.png`
- `results/resnet/visualizations/model_comparison.png`
- `results/resnet/visualizations/spatial_predictions.png`

## Output Directory Structure

```
results/
└── resnet/
    ├── training_history.npz           # Training curves data
    ├── test_results.npz                # Test set predictions
    ├── normalization_params.npz        # Feature normalization stats
    ├── predictions.npy                 # Spatial predictions (NumPy)
    ├── predictions.tif                 # Spatial predictions (GeoTIFF)
    └── visualizations/
        ├── training_curves.png         # Loss and accuracy plots
        ├── confusion_matrix.png        # Confusion matrix heatmap
        ├── model_comparison.png        # ResNet vs Random Forest
        └── spatial_predictions.png     # Spatial prediction maps

models/
└── resnet50_best.pth                   # Best trained model (91 MB)
```

## Usage Examples

### Example 1: Complete Pipeline

```python
# 1. Train model
from modules.deep_learning_trainer import train_resnet_model

results = train_resnet_model(
    X_patches, y_patches,
    model_name='resnet50',
    model_dir='models',
    results_dir='results/resnet'
)

# 2. Predict on new data
from modules.dl_predictor import predict_spatial

predictions, results = predict_spatial(
    model='models/resnet50_best.pth',
    features=sentinel2_features,
    labels=klhk_labels,
    channel_means=means,
    channel_stds=stds
)

# 3. Visualize results
from modules.dl_visualizer import generate_all_visualizations

generate_all_visualizations(
    training_history_path='results/resnet/training_history.npz',
    test_results_path='results/resnet/test_results.npz',
    predictions_path='results/resnet/predictions.npy',
    ground_truth=klhk_raster,
    output_dir='results/resnet/visualizations'
)
```

### Example 2: Custom Prediction

```python
from modules.dl_predictor import load_resnet_model, normalize_features, predict_patches

# Load model
model = load_resnet_model('models/resnet50_best.pth', device='cuda')

# Normalize features
features_norm = normalize_features(features, channel_means, channel_stds)

# Predict
predictions, stats = predict_patches(
    model, features_norm, labels,
    patch_size=32, stride=16, batch_size=64
)

print(f"Accuracy: {stats['accuracy']*100:.2f}%")
print(f"Speed: {stats['speed']:.0f} patches/sec")
```

### Example 3: Custom Visualization

```python
from modules.dl_visualizer import plot_training_curves, plot_confusion_matrix

# Plot training curves
import numpy as np
history = np.load('results/resnet/training_history.npz')

fig = plot_training_curves(
    history,
    save_path='my_curves.png',
    baseline_acc=74.95,  # Random Forest baseline
    best_epoch=6
)

# Plot confusion matrix
test_data = np.load('results/resnet/test_results.npz')
y_true = test_data['targets']
y_pred = test_data['predictions']

fig = plot_confusion_matrix(
    y_true, y_pred,
    class_names=['Water', 'Trees', 'Crops', 'Shrub', 'Built', 'Bare'],
    save_path='my_cm.png',
    accuracy=0.80
)
```

## Configuration

All scripts use consistent configuration:

- **Patch size:** 32×32 pixels
- **Stride:** 16 pixels (50% overlap)
- **Batch size:** 16 (training), 64 (inference)
- **Learning rate:** 0.0001
- **Epochs:** 30
- **Device:** CUDA if available, else CPU

## Color Scheme

Bright, colorful Jambi-optimized palette:

```python
CLASS_COLORS = {
    0: '#0066CC',  # Water - Bright Blue
    1: '#228B22',  # Trees/Forest - Forest Green
    2: '#90EE90',  # Crops - Light Green
    3: '#FF8C00',  # Shrub - Dark Orange
    4: '#FF1493',  # Built - Deep Pink/Magenta
    5: '#D2691E',  # Bare Ground - Chocolate Brown
}
```

## Legacy Scripts

Old scripts have been moved to `scripts/legacy/`:
- `run_resnet_classification_FIXED.py`
- `generate_resnet_predictions.py`
- `visualize_resnet_results.py`

These are kept for reference but should not be used. Use the new modular scripts instead.

## Best Practices

1. **Always use centralized scripts** for standard workflows
2. **Import modules** for custom analysis
3. **Save normalization parameters** from training for prediction
4. **Use consistent naming** for output files
5. **Organize results** in clean directory structure

## Troubleshooting

### DLL Errors (Windows)
If you get DLL errors with torch/pillow/pyproj:
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
pip install --upgrade pillow pyproj rasterio pyogrio
```

### Out of Memory
Reduce batch size or patch size:
```python
BATCH_SIZE = 8  # Instead of 16
PATCH_SIZE = 16  # Instead of 32
```

### Slow Prediction
Use GPU if available:
```python
DEVICE = 'cuda'  # Instead of 'cpu'
```

---

**Author:** Claude Sonnet 4.5
**Date:** 2026-01-03
**Version:** 1.0
