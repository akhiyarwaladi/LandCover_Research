# Land Cover Classification Research - Jambi Province

Research project for land cover classification in Jambi Province, Indonesia using satellite imagery and machine learning.

## Project Overview

This research focuses on analyzing land cover changes in Jambi Province using:
- **Sentinel-2** multispectral imagery (10-20m resolution)
- **Dynamic World** near-real-time land cover dataset
- **Cloud Score+** for advanced cloud masking
- Various **spectral indices** for feature extraction

## Repository Structure

```
land_cover/
├── gee_scripts/                    # Google Earth Engine scripts
│   ├── g_earth_engine_improved.js  # Main analysis script (RECOMMENDED)
│   ├── verification_boundaries.js  # Boundary verification helper
│   └── legacy/                     # Old script versions
│       ├── g_earth_engine_v1.js
│       └── g_earth_engine_v2.js
├── references/                     # Reference papers and materials
├── Land Cover Classification.ipynb # Jupyter notebook for ML classification
├── Land Cover Classification1.ipynb
├── land_cover_classification.py    # Python classification script
├── results_plots.zip               # Previous analysis results
└── README.md
```

## Getting Started

### 1. Google Earth Engine Setup

1. Sign up for [Google Earth Engine](https://earthengine.google.com/)
2. Open [GEE Code Editor](https://code.earthengine.google.com/)
3. **First**, run `verification_boundaries.js` to verify boundary field names
4. Then run `g_earth_engine_improved.js` for main analysis

### 2. Boundary Verification

Before running the main script, verify the boundary data:

```javascript
// Run verification_boundaries.js first to see:
// - Available province names in FAO GAUL and geoBoundaries
// - Exact spelling of "Jambi" in each dataset
// - Visual comparison of boundaries
```

### 3. Configuration

Edit the CONFIG section in `g_earth_engine_improved.js`:

```javascript
var CONFIG = {
  startDate: '2024-01-01',
  endDate: '2024-12-31',
  maxCloudPercent: 20,
  cloudScoreThreshold: 0.60,
  exportScale: 10,
  exportFolder: 'GEE_Exports',
  regionName: 'jambi',
  yearLabel: '2024'
};

// Choose boundary source
var USE_BOUNDARY_SOURCE = 'GAUL';  // Options: 'GAUL', 'GEOBOUNDARIES', 'CUSTOM', 'BBOX'
```

## Datasets Used

| Dataset | Description | Resolution | Source |
|---------|-------------|------------|--------|
| Sentinel-2 SR Harmonized | Surface reflectance imagery | 10-20m | [GEE Catalog](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED) |
| Cloud Score+ | Cloud masking QA | 10m | [GEE Catalog](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_CLOUD_SCORE_PLUS_V1_S2_HARMONIZED) |
| Dynamic World V1 | Land cover classification | 10m | [GEE Catalog](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1) |
| FAO GAUL 2015 | Administrative boundaries | Vector | [GEE Catalog](https://developers.google.com/earth-engine/datasets/catalog/FAO_GAUL_2015_level1) |

## Spectral Indices Calculated

| Index | Full Name | Purpose |
|-------|-----------|---------|
| NDVI | Normalized Difference Vegetation Index | Vegetation health |
| EVI | Enhanced Vegetation Index | Dense vegetation (tropics) |
| NDWI | Normalized Difference Water Index | Water bodies |
| NDMI | Normalized Difference Moisture Index | Vegetation moisture |
| MNDWI | Modified NDWI | Better water detection |
| NDBI | Normalized Difference Built-up Index | Urban/built-up areas |
| SAVI | Soil Adjusted Vegetation Index | Sparse vegetation |
| NBR | Normalized Burn Ratio | Burned areas |

## Dynamic World Classes

| Code | Class | Color |
|------|-------|-------|
| 0 | Water | #419BDF |
| 1 | Trees | #397D49 |
| 2 | Grass | #88B053 |
| 3 | Flooded Vegetation | #7A87C6 |
| 4 | Crops | #E49635 |
| 5 | Shrub and Scrub | #DFC35A |
| 6 | Built | #C4281B |
| 7 | Bare | #A59B8F |
| 8 | Snow and Ice | #B39FE1 |

## Output Files

The script exports 6 GeoTIFF files to Google Drive:

1. `S2_jambi_2024_10m_RGBNIR` - Sentinel-2 bands B2, B3, B4, B8 (10m)
2. `S2_jambi_2024_20m_RedEdgeSWIR` - Red Edge + SWIR bands (20m)
3. `DW_jambi_2024_classification` - Dynamic World land cover classes
4. `DW_jambi_2024_probabilities` - Class probability layers
5. `Indices_jambi_2024_all` - All spectral indices
6. `QC_jambi_2024_obsCount` - Observation count per pixel

## Key Improvements (v2024)

Compared to previous versions:

- [x] Updated to `S2_SR_HARMONIZED` (recommended since 2024)
- [x] Cloud Score+ integration (best cloud masking method)
- [x] Proper administrative boundary options (GAUL, geoBoundaries, custom)
- [x] 8 spectral indices for comprehensive analysis
- [x] Quality control metrics (observation count)
- [x] Separated 10m and 20m band exports
- [x] Configurable parameters

## References

See `references/` folder for related papers:
- Land cover change analysis methodologies
- SAR-based mapping approaches
- Indonesian land cover studies

## License

This project is for research purposes.

**Data Licenses:**
- FAO GAUL: Non-commercial use only
- geoBoundaries: CC BY 4.0
- Sentinel-2/Dynamic World: Copernicus/Google terms

## Author

Research by Akhiyar Waladi

## Changelog

### 2024-12-31
- Initial repository setup
- Added improved GEE script with Cloud Score+
- Created boundary verification script
- Organized folder structure
