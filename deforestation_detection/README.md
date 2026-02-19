# Multi-Temporal Deforestation Detection - Jambi Province, Indonesia

Deep learning change detection for deforestation monitoring using Sentinel-2 annual composites (2018-2024).

## Overview

This project compares **3 change detection approaches** for mapping deforestation in Jambi Province, Sumatra:

| Approach | Method | Key Feature |
|----------|--------|-------------|
| **A. PCC-ResNet** | Post-Classification Comparison | Classify each year, compare maps |
| **B. Siamese CNN** | Siamese ResNet-50 | Learn change directly from image pairs |
| **C. Random Forest** | Stacked temporal features | Fast baseline with feature importance |

## Data

- **Sentinel-2**: 7 annual dry-season composites (2018-2024), 10 bands, 20m resolution
- **Hansen GFC**: Global Forest Change v1.12 — annual deforestation labels (2001-2024)
- **ForestNet**: Stanford deforestation driver dataset (Oil Palm, Timber, Smallholder, Grassland)

## Quick Start

```bash
# 1. Activate environment (reuse from parent project)
conda activate landcover_jambi

# 2. Download data (run GEE scripts first, then Python scripts)
python scripts/download_sentinel2_multitemporal.py
python scripts/download_hansen_gfc.py
python scripts/download_forestnet.py

# 3. Prepare labels and patches
python scripts/prepare_change_labels.py
python scripts/prepare_patches.py

# 4. Train all 3 approaches
python scripts/train_all_approaches.py

# 5. Generate publication outputs
python scripts/generate_publication_comparison.py
python scripts/generate_statistical_analysis.py
```

## Project Structure

```
deforestation_detection/
├── modules/                    # Reusable components
│   ├── data_loader.py          # Load multi-temporal S2, Hansen, ForestNet
│   ├── feature_engineering.py  # Spectral indices + temporal change features
│   ├── preprocessor.py         # Patch extraction, change label creation
│   ├── change_detector.py      # Post-classification comparison logic
│   ├── siamese_network.py      # Siamese CNN architecture
│   ├── model_trainer.py        # ML training (RF baseline)
│   ├── deep_learning_trainer.py # DL training loop
│   └── visualizer.py           # Publication plots
├── scripts/                    # Production scripts
├── gee_scripts/                # Google Earth Engine JavaScript
├── results/                    # Centralized outputs
│   ├── models/                 # Trained model weights
│   ├── tables/                 # Excel/LaTeX tables
│   └── figures/                # PNG figures (300 DPI)
├── data/                       # Downloaded data
│   ├── sentinel/               # Annual composites (2018-2024)
│   ├── hansen/                 # Hansen GFC layers
│   ├── forestnet/              # ForestNet driver labels
│   └── change_labels/          # Derived annual change labels
└── docs/                       # Documentation
```

## Multi-Temporal Analysis

The 7-year time series enables:
- **Annual deforestation rates** (ha/year, trend analysis)
- **Cumulative forest loss** (2018-2024 total)
- **Hotspot identification** (persistent deforestation areas)
- **Event correlation** (El Nino 2019/2023, COVID 2020-2021)
- **Driver trend analysis** (oil palm vs smallholder over time)

## Results

Results will be stored in `results/` following the same standardized structure as the parent project.

## Environment

Reuses the `landcover_jambi` conda environment from the parent project. No additional dependencies required.

## Related

- Parent project: `../` (Land Cover Classification with ResNet + Random Forest)
- KLHK ground truth: `../data/klhk/`
- Sentinel-2 2024 data: `../data/sentinel_new_cloudfree/`
