# Land Cover Classification Research - Jambi Province

Research project for land cover classification in Jambi Province, Indonesia using satellite imagery and machine learning.

## Project Overview

This research focuses on analyzing land cover in Jambi Province using:
- **Sentinel-2** multispectral imagery (10-20m resolution)
- **KLHK Official Data** as ground truth reference (28,100 polygons)
- **Cloud Score+** for advanced cloud masking
- Multiple **spectral indices** for feature extraction
- Multiple **ML classifiers** comparison

### Key Decision: Using KLHK Instead of Dynamic World

**Why KLHK instead of Dynamic World?**

Dynamic World is itself a machine learning product (Neural Network by Google), making it unsuitable as "ground truth" - this would be circular reasoning. KLHK (Ministry of Environment and Forestry) data is official Indonesian government land cover mapping, providing real ground truth for validation.

## Repository Structure

```
land_cover/
├── data/
│   └── klhk/                           # KLHK reference data
│       └── KLHK_PL2024_Jambi_Full.geojson  # 28,100 polygons
├── docs/
│   └── RESEARCH_NOTES.md               # Comprehensive research documentation
├── gee_scripts/
│   ├── g_earth_engine_improved.js      # Main GEE script
│   ├── verification_boundaries.js      # Boundary verification helper
│   └── legacy/                         # Old script versions
├── notebooks/
│   └── archive/                        # Original notebooks (reference)
├── references/                         # Research papers
├── results/                            # Output files (generated)
├── scripts/
│   ├── download_klhk.py                # KLHK data downloader
│   ├── download_sentinel2.py           # Sentinel-2 via EE Python API
│   ├── land_cover_classification_klhk.py  # Main classification script
│   └── legacy/                         # Old script versions
├── .gitignore
└── README.md
```

## Getting Started

### 1. Install Dependencies

```bash
pip install earthengine-api geopandas rasterio scikit-learn lightgbm matplotlib seaborn
```

### 2. Authenticate Earth Engine

```bash
earthengine authenticate
```

### 3. Download KLHK Reference Data (if not already downloaded)

```bash
python scripts/download_klhk.py
```

This downloads 28,100 land cover polygons for Jambi Province from KLHK.

### 4. Download Sentinel-2 Imagery

**Option A: Via Python (recommended)**
```bash
# Download small sample for testing
python scripts/download_sentinel2.py --mode sample --scale 20

# Download full province (exports to Google Drive)
python scripts/download_sentinel2.py --mode full
```

**Option B: Via GEE Console**
1. Open [GEE Code Editor](https://code.earthengine.google.com/)
2. Copy `gee_scripts/g_earth_engine_improved.js`
3. Run and check Google Drive for exports

### 5. Run Classification

```bash
python scripts/land_cover_classification_klhk.py
```

## Data Sources

| Data | Description | Resolution | Source |
|------|-------------|------------|--------|
| Sentinel-2 SR Harmonized | Surface reflectance imagery | 10-20m | [GEE Catalog](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED) |
| Cloud Score+ | Cloud masking QA | 10m | [GEE Catalog](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_CLOUD_SCORE_PLUS_V1_S2_HARMONIZED) |
| KLHK PL2024 | Official land cover | Vector | [KLHK Geoportal](https://geoportal.menlhk.go.id/) |
| FAO GAUL 2015 | Administrative boundaries | Vector | [GEE Catalog](https://developers.google.com/earth-engine/datasets/catalog/FAO_GAUL_2015_level1) |

## KLHK Land Cover Classes

Jambi 2024 Distribution (28,100 polygons):

| Class | Name | Count | % |
|-------|------|-------|---|
| 2007 | Hutan Tanaman | 7,448 | 26.5% |
| 2002 | Hutan Lahan Kering Sekunder | 5,347 | 19.0% |
| 2010 | Perkebunan | 3,829 | 13.6% |
| 2014 | Tanah Terbuka | 3,746 | 13.3% |
| 20092 | Sawah | 2,776 | 9.9% |
| 2012 | Pemukiman | 2,054 | 7.3% |
| 2500 | Semak Belukar | 1,188 | 4.2% |
| 20091 | Pertanian Lahan Kering Campur | 738 | 2.6% |
| Other | Various | 974 | 3.6% |

### Simplified 9-Class Mapping

For comparison with global products:

| Code | Class | KLHK Classes Mapped |
|------|-------|---------------------|
| 0 | Water | Tubuh Air, Rawa |
| 1 | Trees/Forest | All Hutan classes |
| 2 | Grass/Savanna | Savana |
| 3 | Flooded Vegetation | - |
| 4 | Crops/Agriculture | Perkebunan, Sawah, Pertanian |
| 5 | Shrub/Scrub | Semak Belukar |
| 6 | Built Area | Pemukiman |
| 7 | Bare Ground | Tanah Terbuka |

## Features Used (23 total)

### Sentinel-2 Bands (10)
B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12

### Spectral Indices (13)
| Index | Full Name | Purpose |
|-------|-----------|---------|
| NDVI | Normalized Difference Vegetation Index | Vegetation health |
| EVI | Enhanced Vegetation Index | Dense vegetation |
| SAVI | Soil Adjusted Vegetation Index | Sparse vegetation |
| NDWI | Normalized Difference Water Index | Water bodies |
| MNDWI | Modified NDWI | Better water detection |
| NDBI | Normalized Difference Built-up Index | Urban areas |
| BSI | Bare Soil Index | Bare ground |
| NDRE | Normalized Difference Red Edge | Crop health |
| CIRE | Chlorophyll Index Red Edge | Chlorophyll content |
| MSAVI | Modified SAVI | Exposed soil |
| GNDVI | Green NDVI | Greenness |
| NDMI | Normalized Difference Moisture Index | Vegetation moisture |
| NBR | Normalized Burn Ratio | Burned areas |

## Classifiers Compared

| Classifier | Type |
|------------|------|
| Random Forest | Ensemble |
| Extra Trees | Ensemble |
| LightGBM | Gradient Boosting |
| XGBoost | Gradient Boosting |
| Decision Tree | Tree-based |
| Logistic Regression | Linear |
| SGD | Linear |
| Naive Bayes | Probabilistic |

## Documentation

See `docs/RESEARCH_NOTES.md` for comprehensive documentation including:
- Available reference datasets for Indonesia
- State-of-the-art classification methods
- Research novelty options
- Handling temporal gap in ground truth
- Detailed methodology framework

## License

This project is for research purposes.

**Data Licenses:**
- KLHK Data: Indonesian Government Open Data
- FAO GAUL: Non-commercial use only
- Sentinel-2: Copernicus terms

## Author

Research by Akhiyar Waladi

## Changelog

### 2024-12-31
- Switched from Dynamic World to KLHK as ground truth reference
- Downloaded complete KLHK data (28,100 polygons vs 1,000 sample)
- Created comprehensive classification script with multiple classifiers
- Added Sentinel-2 Python download script
- Updated research notes with current project status
- Reorganized folder structure
