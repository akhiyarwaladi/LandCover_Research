# Research Notes: Land Cover Classification - Jambi Province

> **Project**: Land Cover Classification Research
> **Location**: Jambi Province, Indonesia
> **Last Updated**: December 2024

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current Approach Analysis](#2-current-approach-analysis)
3. [Available Reference Datasets](#3-available-reference-datasets)
4. [State-of-the-Art Methods](#4-state-of-the-art-methods)
5. [Research Novelty Options](#5-research-novelty-options)
6. [Proposed Research Framework](#6-proposed-research-framework)
7. [References](#7-references)

---

## 1. Executive Summary

### Problem Statement

Penelitian klasifikasi tutupan lahan saat ini menggunakan **Dynamic World** sebagai ground truth. Namun, Dynamic World sendiri merupakan hasil **Neural Network** yang di-training oleh Google, sehingga:

- Bukan ground truth sebenarnya (circular reasoning)
- Tidak ada validasi lapangan
- Novelty penelitian rendah (hanya mereplikasi hasil Google)

### Key Findings

| Temuan | Implikasi |
|--------|-----------|
| Dynamic World = hasil ML | Tidak valid sebagai ground truth |
| ESA WorldCover akurasi 83.8% | Lebih akurat dari Dynamic World (73.4%) |
| Ada Indonesia National LC Dataset | **Ground truth asli dari Indonesia tersedia!** |
| Transformer-based methods trending | Peluang novelty dengan attention mechanism |

### Recommendation

Gunakan **Indonesia National Land Cover Reference Dataset** sebagai ground truth, kombinasikan dengan metode **Transformer/Attention-based** untuk novelty penelitian.

---

## 2. Current Approach Analysis

### 2.1 Existing Script Overview

**File**: `land_cover_classification.py`

```
Input Features (23 total):
├── Sentinel-2 Bands (10): B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12
└── Spectral Indices (13): NDVI, EVI, SAVI, NDWI, MNDWI, NDBI, BSI,
                           NDRE, CIRE, MSAVI, GNDVI, NDMI, NBRI

Labels: Dynamic World (9 classes)
```

### 2.2 Classifiers Used

| Classifier | Type | Status | Notes |
|------------|------|--------|-------|
| Random Forest | Ensemble | Active | n_estimators=200, max_depth=25 |
| Extra Trees | Ensemble | Active | Similar config to RF |
| LightGBM | Gradient Boosting | Active | n_estimators=100 |
| Decision Tree | Tree-based | Active | max_depth=15 |
| Logistic Regression | Linear | Active | multinomial |
| Naive Bayes | Probabilistic | Active | Gaussian |
| SGD | Linear | Active | modified_huber loss |
| KNN | Instance-based | Disabled | Commented out (slow) |

### 2.3 Issues Identified

1. **Ground Truth Problem**: Dynamic World adalah hasil ML, bukan referensi sebenarnya
2. **Validation Method**: Menggunakan random split, bukan spatial block CV
3. **No Independent Validation**: Tidak ada perbandingan dengan produk lain
4. **Traditional ML Only**: Belum menggunakan deep learning

---

## 3. Available Reference Datasets

### 3.1 Indonesia-Specific Datasets

#### A. Indonesia National Land Cover Reference Dataset (RECOMMENDED)

| Attribute | Value |
|-----------|-------|
| **Publication** | Scientific Data (Nature), 2022 |
| **DOI** | https://doi.org/10.1038/s41597-022-01689-5 |
| **Coverage** | Seluruh Indonesia |
| **Classes** | 7 generic + 17 detailed |
| **Method** | Crowdsourcing + Expert validation |
| **Resolution** | 100x100m chips from VHR imagery |
| **Download** | [Figshare](https://figshare.com/articles/dataset/20278341) |
| **License** | CC BY 4.0 |

**Keunggulan**:
- Ground truth asli dari Indonesia
- Dikumpulkan oleh citizen scientists dan expert lokal
- Sudah di-validate dengan quality assessment
- Tersedia raw dan filtered dataset

#### B. KLHK Official Land Cover Data

| Attribute | Value |
|-----------|-------|
| **Source** | Direktorat IPSDH, KLHK |
| **Scale** | 1:250,000 |
| **Access** | [SIGAP KLHK](https://sigap.menlhk.go.id/) |
| **Standard** | SNI 7645-1:2014 (FAO LCCS compatible) |

**Classes (23 classes)**:
- Hutan primer (lahan kering, rawa, mangrove)
- Hutan sekunder (lahan kering, rawa, mangrove)
- Hutan tanaman
- Perkebunan, pertanian, sawah
- Semak belukar, savanna
- Permukiman, area terbuka
- Badan air, dll

### 3.2 Global Datasets with Indonesia Coverage

#### A. GLanCE Training Dataset

| Attribute | Value |
|-----------|-------|
| **Publication** | Scientific Data (Nature), 2023 |
| **DOI** | https://doi.org/10.1038/s41597-023-02798-5 |
| **Samples** | ~2 million globally |
| **Period** | 1984-2020 |
| **Resolution** | 30m (Landsat-based) |
| **Indonesia Coverage** | Partial (lower density in Asia) |
| **Access** | GEE Community Catalog, Radiant MLHub |
| **License** | CC BY 4.0 |

#### B. Globe230k Benchmark Dataset

| Attribute | Value |
|-----------|-------|
| **Publication** | Journal of Remote Sensing, 2023 |
| **DOI** | https://doi.org/10.34133/remotesensing.0078 |
| **Samples** | 232,819 images (512x512) |
| **Resolution** | 1m |
| **Classes** | 10 (cropland, forest, grass, shrub, wetland, water, tundra, impervious, bareland, ice/snow) |
| **Indonesia Coverage** | Random global sampling |
| **Download** | [Zenodo](https://zenodo.org/records/8429200) |

### 3.3 Global Land Cover Products (for Comparison)

| Product | Resolution | Accuracy | Provider | GEE Asset |
|---------|------------|----------|----------|-----------|
| ESA WorldCover | 10m | 83.8% | ESA | `ESA/WorldCover/v200` |
| Dynamic World | 10m | 73.4% | Google/WRI | `GOOGLE/DYNAMICWORLD/V1` |
| ESRI LULC | 10m | ~75% | ESRI/Impact Observatory | Community |
| Copernicus GLC | 100m | ~80% | Copernicus | `COPERNICUS/Landcover/100m` |

**Source**: [Comparative validation study, Remote Sensing of Environment 2024](https://www.sciencedirect.com/science/article/pii/S0034425724003341)

### 3.4 Datasets NOT Suitable for Indonesia

| Dataset | Reason |
|---------|--------|
| LUCAS | Europe only |
| NLCD | USA only |
| CORINE | Europe only |

---

## 4. State-of-the-Art Methods

### 4.1 Deep Learning Architectures (2024-2025)

#### A. Transformer-Based Models

| Model | Description | Accuracy | Reference |
|-------|-------------|----------|-----------|
| **Vision Transformer (ViT)** | Patch-based attention | 96.92% | Forests 2024 |
| **Swin Transformer** | Hierarchical with shifted windows | SOTA | IEEE TGRS |
| **HDAM-Net** | ViT + Multiscale attention | 99.42% | Forests 2024 |

#### B. CNN-Transformer Hybrids

| Model | Description | Key Feature |
|-------|-------------|-------------|
| **DE-UNet** | Dual encoder (CNN + Swin) | Global + Local features |
| **IRUNet** | InceptionResNetV2 + UNet | Multi-scale fusion + TTA |
| **TransUNet** | Transformer + UNet | Medical-to-RS transfer |

#### C. Foundation Models

| Model | Training Data | Provider |
|-------|---------------|----------|
| **Prithvi** | HLS (Landsat-Sentinel) | NASA-IBM |
| **SkySense** | Multi-modal | Research |
| **SatMAE** | Self-supervised | Meta |

### 4.2 Traditional ML (Baseline)

| Method | Typical Accuracy | Speed |
|--------|------------------|-------|
| Random Forest | 85-95% | Fast |
| XGBoost/LightGBM | 85-95% | Fast |
| SVM | 80-90% | Medium |

### 4.3 Key Methodological Considerations

#### Validation Best Practices

From [MDPI Sustainability 2024](https://www.mdpi.com/2071-1050/17/22/10324):

> **67% of studies (2020-2025)** still use random validation, causing **accuracy overestimation up to 30%** for deep learning models.

**Recommended validation methods**:
1. Spatial block cross-validation (min 1-5 km separation)
2. Leave-one-location-out validation
3. Temporal validation (train on year X, test on year Y)

---

## 5. Research Novelty Options

### 5.1 Novelty Matrix

| Novelty Aspect | Difficulty | Impact | Description |
|----------------|------------|--------|-------------|
| **Multi-source Fusion** | High | Very High | Sentinel-1 (SAR) + Sentinel-2 (Optical) |
| **Attention Mechanism** | Medium | High | Transformer/ViT for spatial context |
| **Local Ground Truth** | Medium | High | Use Indonesia National LC Dataset |
| **Comparative Study** | Low | Medium | Compare with DW, WorldCover, ESRI |
| **Tropical Focus** | Low | Medium | Specific to Jambi forest characteristics |
| **Temporal Analysis** | Medium | High | Multi-year change detection |
| **Explainability** | Medium | High | Attention visualization, SHAP |

### 5.2 Recommended Novelty Combinations

#### Option A: Deep Learning + Local Ground Truth
```
Novelty: Transformer-based classification dengan ground truth Indonesia
Input: Sentinel-2 bands + indices
Method: Vision Transformer atau Swin Transformer
Validation: Indonesia National LC Dataset
Comparison: Dynamic World, ESA WorldCover
```

#### Option B: Multi-Source Fusion
```
Novelty: Optical-SAR fusion untuk hutan tropis Jambi
Input: Sentinel-1 (VV, VH) + Sentinel-2 + DEM
Method: Late fusion atau Feature-level fusion
Advantage: SAR tidak terpengaruh awan (penting untuk tropis!)
```

#### Option C: Attention + Explainability
```
Novelty: Explainable AI untuk klasifikasi tutupan lahan
Input: Sentinel-2 bands + indices
Method: Attention-based CNN dengan visualization
Output: Classification + Attention maps showing important regions
```

---

## 6. Proposed Research Framework

### 6.1 Research Questions

1. Bagaimana performa Transformer-based model dibandingkan traditional ML untuk klasifikasi tutupan lahan di Jambi?
2. Apakah penambahan data SAR (Sentinel-1) meningkatkan akurasi klasifikasi di wilayah tropis berawan?
3. Bagaimana hasil klasifikasi dibandingkan dengan produk global (Dynamic World, ESA WorldCover)?

### 6.2 Methodology Framework

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     RESEARCH METHODOLOGY FRAMEWORK                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐  │
│  │   DATA SOURCES   │    │   PROCESSING     │    │   GROUND TRUTH   │  │
│  ├──────────────────┤    ├──────────────────┤    ├──────────────────┤  │
│  │ • Sentinel-2     │───▶│ • Cloud masking  │    │ • Indonesia      │  │
│  │   (10 bands)     │    │   (Cloud Score+) │    │   National LC    │  │
│  │ • Sentinel-1     │───▶│ • Compositing    │    │   Dataset        │  │
│  │   (VV, VH)       │    │ • Indices calc   │    │ • KLHK official  │  │
│  │ • DEM (SRTM)     │───▶│ • Normalization  │    │ • Visual interp  │  │
│  └──────────────────┘    └────────┬─────────┘    └────────┬─────────┘  │
│                                   │                       │             │
│                                   ▼                       ▼             │
│                    ┌──────────────────────────────────────────┐        │
│                    │           FEATURE ENGINEERING            │        │
│                    ├──────────────────────────────────────────┤        │
│                    │ • 10 Sentinel-2 bands                    │        │
│                    │ • 13 Spectral indices (NDVI, EVI, etc)   │        │
│                    │ • 2 SAR bands (VV, VH) [optional]        │        │
│                    │ • DEM + Slope + Aspect [optional]        │        │
│                    └─────────────────┬────────────────────────┘        │
│                                      │                                  │
│                                      ▼                                  │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                      CLASSIFICATION METHODS                        │ │
│  ├───────────────────────────────────────────────────────────────────┤ │
│  │                                                                    │ │
│  │   BASELINE (Traditional ML)    │    PROPOSED (Deep Learning)      │ │
│  │   ─────────────────────────    │    ────────────────────────      │ │
│  │   • Random Forest              │    • Vision Transformer (ViT)    │ │
│  │   • LightGBM                   │    • Swin Transformer            │ │
│  │   • XGBoost                    │    • CNN-Transformer Hybrid      │ │
│  │                                │    • Attention U-Net             │ │
│  │                                │                                   │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                      │                                  │
│                                      ▼                                  │
│                    ┌──────────────────────────────────────────┐        │
│                    │             VALIDATION                   │        │
│                    ├──────────────────────────────────────────┤        │
│                    │ • Spatial Block Cross-Validation         │        │
│                    │ • Independent test set (Indonesia LC)    │        │
│                    │ • Comparison with DW & WorldCover         │        │
│                    └─────────────────┬────────────────────────┘        │
│                                      │                                  │
│                                      ▼                                  │
│                    ┌──────────────────────────────────────────┐        │
│                    │              OUTPUTS                     │        │
│                    ├──────────────────────────────────────────┤        │
│                    │ • Land cover map 10m resolution          │        │
│                    │ • Accuracy metrics per class             │        │
│                    │ • Attention/importance visualization     │        │
│                    │ • Comparative analysis report            │        │
│                    └──────────────────────────────────────────┘        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.3 Land Cover Classes

**Proposed 9-class scheme** (compatible with Dynamic World for comparison):

| Code | Class | Indonesian | Description |
|------|-------|------------|-------------|
| 0 | Water | Air | Sungai, danau, kolam |
| 1 | Trees | Hutan | Hutan primer & sekunder |
| 2 | Grass | Rumput | Padang rumput, savana |
| 3 | Flooded Vegetation | Vegetasi Tergenang | Rawa, mangrove |
| 4 | Crops | Pertanian | Sawah, ladang |
| 5 | Shrub and Scrub | Semak | Semak belukar |
| 6 | Built | Terbangun | Permukiman, infrastruktur |
| 7 | Bare | Lahan Terbuka | Tanah kosong, tambang |
| 8 | Snow and Ice | Salju/Es | Tidak relevan untuk Jambi |

### 6.4 Evaluation Metrics

| Metric | Purpose |
|--------|---------|
| Overall Accuracy (OA) | General performance |
| Kappa Coefficient | Agreement beyond chance |
| F1-Score (per class) | Balance precision-recall |
| Producer's Accuracy | Omission error |
| User's Accuracy | Commission error |
| IoU (for DL) | Segmentation quality |

---

## 7. References

### 7.1 Datasets

1. Hadi, et al. (2022). "A national-scale land cover reference dataset from local crowdsourcing initiatives in Indonesia." *Scientific Data*, 9, 574. https://doi.org/10.1038/s41597-022-01689-5

2. Stanimirova, R., et al. (2023). "A global land cover training dataset from 1984 to 2020." *Scientific Data*, 10, 879. https://doi.org/10.1038/s41597-023-02798-5

3. Li, et al. (2023). "Globe230k: A Benchmark Dense-Pixel Annotation Dataset for Global Land Cover Mapping." *Journal of Remote Sensing*. https://doi.org/10.34133/remotesensing.0078

### 7.2 Methods

4. Vali, A., et al. (2024). "Comparative validation of recent 10 m-resolution global land cover maps." *Remote Sensing of Environment*. https://doi.org/10.1016/j.rse.2024.114316

5. Singh, R., et al. (2024). "Transformer-based land use and land cover classification with explainability using satellite imagery." *Scientific Reports*, 14. https://doi.org/10.1038/s41598-024-67186-4

6. MDPI (2024). "Sentinel-2 Land Cover Classification: State-of-the-Art Methods and the Reality of Operational Deployment." *Sustainability*, 17(22). https://doi.org/10.3390/su172210324

### 7.3 Products

7. Brown, C.F., et al. (2022). "Dynamic World, Near real-time global 10 m land use land cover mapping." *Scientific Data*, 9, 251. https://doi.org/10.1038/s41597-022-01307-4

8. Zanaga, D., et al. (2022). "ESA WorldCover 10 m 2021 v200." https://doi.org/10.5281/zenodo.7254221

### 7.4 Training Data Review

9. Moreira, et al. (2024). "Training data in satellite image classification for land cover mapping: a review." *European Journal of Remote Sensing*. https://doi.org/10.1080/22797254.2024.2341414

---

## Appendix A: Data Access Links

| Resource | URL |
|----------|-----|
| Indonesia National LC Dataset | https://figshare.com/articles/dataset/20278341 |
| GLanCE on GEE | `projects/sat-io/open-datasets/GLANCE/GLANCE_TRAINING` |
| Globe230k | https://zenodo.org/records/8429200 |
| KLHK SIGAP | https://sigap.menlhk.go.id/ |
| ESA WorldCover | https://esa-worldcover.org/en/data-access |
| Dynamic World | https://dynamicworld.app/ |

## Appendix B: GEE Asset Paths

```javascript
// Sentinel-2
var S2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED');

// Sentinel-1
var S1 = ee.ImageCollection('COPERNICUS/S1_GRD');

// Cloud Score+
var csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED');

// Dynamic World
var DW = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1');

// ESA WorldCover
var WC = ee.ImageCollection('ESA/WorldCover/v200');

// GLanCE Training
var GLanCE = ee.FeatureCollection('projects/sat-io/open-datasets/GLANCE/GLANCE_TRAINING');

// Administrative Boundaries
var GAUL = ee.FeatureCollection('FAO/GAUL/2015/level1');
var geoBoundaries = ee.FeatureCollection('WM/geoLab/geoBoundaries/600/ADM1');
```

---

## 8. Handling Temporal Gap in Ground Truth Data

### 8.1 The Problem

Dataset ground truth yang tersedia (Indonesia National LC 2022) tidak selalu sesuai dengan periode citra yang ingin diklasifikasi (misalnya 2024). Ini adalah **masalah umum** dalam penelitian land cover.

### 8.2 Solusi yang Digunakan dalam Jurnal Bereputasi

#### A. Sample Migration Method (Recommended)

Dari [Remote Sensing MDPI 2024](https://www.mdpi.com/2072-4292/16/9/1566):

> Metode migrasi sampel dari tahun referensi ke tahun target mencapai **akurasi 91-98%** tergantung kombinasi data.

**Cara kerja**:
1. Gunakan ground truth dari tahun tersedia (misal 2022)
2. Filter sampel yang **stabil** (tidak berubah) menggunakan time-series analysis
3. Migrasi sampel stabil ke tahun target
4. Validasi dengan change detection

```
Akurasi Sample Migration:
├── Sentinel-2 only: 96.82%
├── Sentinel-1 only: 87.68%
└── S1 + S2 combined: 98.25%  ← BEST
```

#### B. Visual Interpretation (Create Your Own)

Menggunakan tools seperti **Collect Earth** (FAO) atau manual di Google Earth:

| Tool | Developer | Features |
|------|-----------|----------|
| [Collect Earth](http://www.openforis.org/tools/collect-earth.html) | FAO | Free, integrates GEE, systematic sampling |
| Google Earth Pro | Google | Free, historical imagery, manual |
| QGIS + Google/Bing | Open Source | Free, full control |

**Protocol untuk visual interpretation**:
1. Stratified random sampling (min 50 points per class)
2. Interpretasi oleh 2-3 interpreter (reduce bias)
3. Gunakan VHR imagery (Google Earth, Bing)
4. Cross-check dengan time-series NDVI

#### C. Transfer Learning / Domain Adaptation

Dari [ScienceDirect 2020](https://www.sciencedirect.com/science/article/abs/pii/S0924271620300101):

> Model yang di-training pada tahun X dapat di-adaptasi untuk tahun Y menggunakan **domain adaptation techniques**.

**Metode yang digunakan**:
- Adversarial learning (SpADANN)
- Self-training dengan pseudo-labels
- Feature alignment

#### D. Use Stable Samples Only

Pendekatan konservatif:
1. Ambil ground truth dari 2022
2. Filter hanya lokasi yang **tidak berubah** (stable)
3. Deteksi perubahan menggunakan:
   - NDVI time-series anomaly
   - BFAST algorithm
   - LandTrendr

### 8.3 Recommended Approach for This Research

```
┌─────────────────────────────────────────────────────────────────┐
│  RECOMMENDED WORKFLOW: Handling Temporal Gap                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Step 1: DOWNLOAD EXISTING DATA                                 │
│  ├── Indonesia National LC Dataset (2022)                       │
│  └── KLHK Tutupan Lahan (latest available)                     │
│                                                                 │
│  Step 2: FILTER STABLE SAMPLES                                  │
│  ├── Calculate NDVI time-series 2022-2024                       │
│  ├── Identify pixels with low variance (stable)                 │
│  └── Remove samples in areas with detected change               │
│                                                                 │
│  Step 3: AUGMENT WITH VISUAL INTERPRETATION                     │
│  ├── Use Google Earth Pro (2024 imagery)                        │
│  ├── Add samples for under-represented classes                  │
│  └── Multiple interpreters for quality control                  │
│                                                                 │
│  Step 4: VALIDATION                                             │
│  ├── Independent test set from visual interpretation            │
│  ├── Compare with ESA WorldCover 2021                           │
│  └── Compare with Dynamic World 2024                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 8.4 Important Considerations

| Issue | Solution |
|-------|----------|
| Land cover change 2022→2024 | Use sample migration + change detection filter |
| Interpreter bias | Multiple interpreters + consensus |
| Class imbalance | Stratified sampling |
| Spatial autocorrelation | Spatial block cross-validation |

### 8.5 Alternative: Focus on Comparison Study

Jika membuat ground truth sendiri terlalu time-consuming, **alternatif yang valid** untuk jurnal:

> **Comparative Study**: Bandingkan performa berbagai produk global (Dynamic World vs ESA WorldCover vs ESRI LULC) untuk wilayah Jambi, menggunakan **independent reference data** dari visual interpretation (sample kecil tapi rigorous).

Ini tetap memiliki **novelty** karena:
- Belum ada studi perbandingan spesifik untuk Jambi/Sumatra
- Fokus pada akurasi untuk hutan tropis Indonesia
- Dapat memberikan rekomendasi produk mana yang terbaik untuk Indonesia

---

## 9. Practical Workflow Options

### Option A: Full Research (High Effort, High Novelty)

```
Timeline: 3-6 months
- Create ground truth via visual interpretation
- Implement Transformer-based classifier
- Multi-source fusion (S1 + S2)
- Full validation protocol
```

### Option B: Comparative Study (Medium Effort, Medium Novelty)

```
Timeline: 1-3 months
- Use existing products (DW, WorldCover, ESRI)
- Create small validation dataset (~500 points)
- Statistical comparison
- Recommendations for Indonesia
```

### Option C: Method Focus (Medium Effort, High Novelty)

```
Timeline: 2-4 months
- Use existing ground truth + sample migration
- Focus on novel classification method
- Attention/Transformer architecture
- Explainability analysis
```

---

## 10. Current Project Status

### 10.1 Data Downloaded

| Data | Status | Location | Records |
|------|--------|----------|---------|
| KLHK PL2024 Jambi | ✅ Complete | `data/klhk/KLHK_PL2024_Jambi_Full.geojson` | 28,100 polygons |
| Sentinel-2 Imagery | ⏳ Ready to download | Use `scripts/download_sentinel2.py` | - |

### 10.2 KLHK Data Distribution (Jambi 2024)

| Land Cover Class | Count | Percentage |
|-----------------|-------|------------|
| Hutan Tanaman | 7,448 | 26.5% |
| Hutan Lahan Kering Sekunder | 5,347 | 19.0% |
| Perkebunan | 3,829 | 13.6% |
| Tanah Terbuka | 3,746 | 13.3% |
| Sawah | 2,776 | 9.9% |
| Pemukiman | 2,054 | 7.3% |
| Semak Belukar | 1,188 | 4.2% |
| Pertanian Lahan Kering Campur | 738 | 2.6% |
| Other classes | 974 | 3.6% |
| **Total** | **28,100** | **100%** |

### 10.3 Available Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `scripts/download_klhk.py` | Download KLHK land cover data | `python scripts/download_klhk.py` |
| `scripts/download_sentinel2.py` | Download Sentinel-2 via Earth Engine API | `python scripts/download_sentinel2.py --mode sample` |
| `scripts/land_cover_classification_klhk.py` | Main classification script | `python scripts/land_cover_classification_klhk.py` |
| `gee_scripts/g_earth_engine_improved.js` | GEE Console script (alternative) | Copy to GEE Code Editor |

### 10.4 Project Folder Structure

```
land_cover/
├── data/
│   └── klhk/                    # KLHK reference data
│       └── KLHK_PL2024_Jambi_Full.geojson  (28,100 polygons)
├── docs/
│   └── RESEARCH_NOTES.md        # This document
├── gee_scripts/
│   ├── g_earth_engine_improved.js   # Main GEE script
│   ├── verification_boundaries.js    # Boundary verification
│   └── legacy/                       # Old script versions
├── notebooks/
│   └── archive/                 # Original notebooks (reference)
├── references/                  # Research papers
├── results/                     # Output files (generated)
└── scripts/
    ├── download_klhk.py         # KLHK data downloader
    ├── download_sentinel2.py    # Sentinel-2 downloader
    ├── land_cover_classification_klhk.py  # Main classifier
    └── legacy/                  # Old script versions
```

### 10.5 Next Steps

1. **Download Sentinel-2 Data**:
   ```bash
   # First authenticate with Earth Engine
   earthengine authenticate

   # Download sample area for testing
   python scripts/download_sentinel2.py --mode sample --scale 20

   # Download full Jambi province (exports to Google Drive)
   python scripts/download_sentinel2.py --mode full
   ```

2. **Run Classification**:
   ```bash
   # After Sentinel-2 data is ready
   python scripts/land_cover_classification_klhk.py
   ```

3. **Review Results**:
   - Check `results/` folder for outputs
   - Review confusion matrix and feature importance plots
   - Compare classifier performance

---

*Document generated: December 2024*
*Last Updated: December 2024*
*Project: LandCover_Research*
