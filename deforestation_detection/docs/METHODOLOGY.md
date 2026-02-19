# Methodology: Multi-Temporal Deforestation Detection

## 1. Study Area

Jambi Province, Sumatra, Indonesia — a major deforestation hotspot due to oil palm expansion, timber plantations, and smallholder agriculture.

## 2. Data Acquisition

### 2.1 Sentinel-2 Multi-Temporal Composites

- **Source**: Google Earth Engine (COPERNICUS/S2_SR_HARMONIZED)
- **Period**: 7 annual composites (2018-2024)
- **Season**: June-October (dry season) for consistent phenology and minimal cloud cover
- **Cloud masking**: Cloud Score+ (threshold 0.60)
- **Composite**: Median pixel selection
- **Bands**: B2, B3, B4, B5, B6, B7, B8A, B11, B12 (10 bands at 20m)
- **Resolution**: 20 meters, EPSG:4326

### 2.2 Hansen Global Forest Change (GFC)

- **Source**: UMD/hansen/global_forest_change_2024_v1_12
- **Layers**: treecover2000, lossyear (1-24 = 2001-2024), gain
- **Native resolution**: 30m (resampled to 20m via nearest-neighbor)
- **Tree cover threshold**: 30% (standard Hansen definition)
- **Annual labels**: lossyear values 18-24 correspond to deforestation in 2018-2024

### 2.3 ForestNet Driver Labels

- **Source**: Stanford ForestNet dataset
- **Samples**: 2,756 labeled Indonesian deforestation events
- **Drivers**: Oil Palm Plantation, Timber Plantation, Smallholder Agriculture, Grassland/Shrub
- **Use**: Driver attribution for detected deforestation areas

## 3. Feature Engineering

### 3.1 Per-Year Features (23 per year)
- 10 spectral bands (B2-B12)
- 13 spectral indices: NDVI, EVI, SAVI, MSAVI, GNDVI, NDWI, MNDWI, NDBI, BSI, NDRE, CIRE, NDMI, NBR

### 3.2 Temporal Change Features (10 per year-pair)
- dNDVI, dEVI, dNBR, dNDMI, dNDWI, dNDBI, dSAVI, dNDRE, dMSAVI, dBSI
- Computed as: feature_t2 - feature_t1

### 3.3 Stacked Features (56 per year-pair, for RF)
- T1 features (23) + T2 features (23) + Change features (10) = 56

## 4. Change Detection Approaches

### 4.1 Approach A: Post-Classification Comparison (PCC)

1. Classify each annual composite independently using ResNet-101
2. Map classes: Forest (class 1) vs Non-forest (classes 0, 4, 5, 6, 7)
3. Detect change: Forest@T1 AND Non-forest@T2 = Deforestation
4. Generate transition matrices for each consecutive year pair

**Advantages**: Simple, interpretable, reuses parent project architecture
**Disadvantages**: Error propagation (classification errors compound)

### 4.2 Approach B: Siamese CNN (Main Novelty)

1. Extract paired patches (32x32 at 20m = 640m footprint)
2. Feed both patches through shared ResNet-50 backbone
3. Fusion: concatenation + absolute difference [f1; f2; |f1-f2|]
4. Binary classification head: Change vs No-change
5. Training with Focal Loss to handle class imbalance (~95% no-change)

**Architecture**: SiameseResNet with shared weights
**Advantages**: Learns change directly, no intermediate classification
**Disadvantages**: More complex, requires paired training data

### 4.3 Approach C: Random Forest Baseline

1. Stack features from both years (56 features per pixel)
2. Train Random Forest on Hansen-derived labels
3. Apply to each consecutive year pair

**Advantages**: Fast, interpretable (feature importance), no GPU needed
**Disadvantages**: No spatial context (pixel-based)

## 5. Training Protocol

- **Labels**: Hansen GFC lossyear (annual deforestation ground truth)
- **Split**: Stratified 80/20 train/test
- **Balancing**: Undersample no-change class to 3:1 ratio
- **Epochs** (DL): 50 with early stopping (patience 10)
- **Optimizer**: Adam (lr=1e-4, weight_decay=1e-5)
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=5)

## 6. Evaluation

### 6.1 Per Year-Pair Metrics
- Overall Accuracy, F1-Macro, F1-Weighted, Kappa
- Precision/Recall for change and no-change classes

### 6.2 Statistical Tests
- McNemar's test: pairwise comparison of all 3 approaches
- Cohen's Kappa: inter-rater agreement

### 6.3 Temporal Analysis
- Annual deforestation area (ha, km2)
- Annual deforestation rate (% of remaining forest)
- Cumulative forest loss (2018-2024)
- Trend analysis (linear regression on annual rates)

## 7. References

- Hansen, M.C., et al. (2013). High-Resolution Global Maps of 21st-Century Forest Cover Change. Science, 342(6160), 850-853.
- He, K., et al. (2016). Deep Residual Learning for Image Recognition. CVPR 2016.
- Daudt, R.C., et al. (2018). Fully Convolutional Siamese Networks for Change Detection. ICIP 2018.
- Lin, T.Y., et al. (2017). Focal Loss for Dense Object Detection. ICCV 2017.
- Meyer, H., & Pebesma, E. (2021). Predicting into unknown space? Methods in Ecology and Evolution.
- Irvin, J., et al. (2020). ForestNet: Classifying Drivers of Deforestation in Indonesia. arXiv:2011.05479.
