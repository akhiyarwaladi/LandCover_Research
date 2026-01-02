# Repository Naming Standards

**Consistent File Naming Across All Outputs**

Last Updated: 2026-01-02

---

## 🎯 Naming Convention

### Standard Format:
```
{category}_{region}_{resolution}_{timeframe}_{descriptor}.{ext}
```

### Components:

1. **category** - Type of output
2. **region** - Geographic scope
3. **resolution** - Spatial resolution
4. **timeframe** - Temporal coverage
5. **descriptor** - Specific variant/method
6. **ext** - File extension

---

## 📁 Category Types

| Category | Description | Examples |
|----------|-------------|----------|
| `sentinel` | Raw satellite data | sentinel_province_20m_2024dry_p25.tif |
| `rgb` | RGB visualizations | rgb_city_10m_2024dry_natural.png |
| `classification` | Classified maps | classification_province_20m_2024_rf.tif |
| `results` | Analysis results | results_province_20m_2024_comparison.csv |
| `metrics` | Performance metrics | metrics_province_20m_2024_rf.csv |
| `test` | Test outputs | test_cloud_province_20m_percentile25.tif |
| `boundary` | Geographic boundaries | boundary_province_geoboundaries.geojson |

---

## 🌍 Region Codes

| Code | Description | Area |
|------|-------------|------|
| `province` | Full Jambi Province | 49,224 km² |
| `city` | Jambi City (Kota Jambi) | 172 km² |
| `region_{name}` | Custom region | Variable |
| `sample_{desc}` | Sample/test area | Variable |

---

## 📏 Resolution Codes

| Code | Description |
|------|-------------|
| `10m` | 10-meter resolution |
| `20m` | 20-meter resolution |
| `mixed` | Multiple resolutions |

---

## 📅 Timeframe Codes

| Code | Description | Dates |
|------|-------------|-------|
| `2024` | Full year | 2024-01-01 to 2024-12-31 |
| `2024dry` | Dry season | 2024-06-01 to 2024-09-30 |
| `2024wet` | Wet season | 2024-11-01 to 2025-03-31 |
| `2024Q1` | Quarter | 2024-01-01 to 2024-03-31 |

---

## 🏷️ Descriptor Codes

### Cloud Removal Strategies:
| Code | Strategy |
|------|----------|
| `p25` | percentile_25 |
| `p30` | percentile_30 |
| `median` | median |
| `kalimantan` | kalimantan |
| `balanced` | balanced |

### Classification Models:
| Code | Model |
|------|-------|
| `rf` | Random Forest |
| `et` | Extra Trees |
| `lgbm` | LightGBM |
| `dt` | Decision Tree |
| `lr` | Logistic Regression |

### Visualization Types:
| Code | Type |
|------|------|
| `natural` | Natural color RGB |
| `falsecolor` | False color composite |
| `ndvi` | NDVI visualization |
| `comparison` | Side-by-side comparison |

---

## 📋 Examples by Type

### 1. Sentinel-2 Downloads

**Pattern:** `sentinel_{region}_{resolution}_{timeframe}_{strategy}`

```
sentinel_province_20m_2024dry_p25.tif
sentinel_province_20m_2024dry_p25-tile1.tif
sentinel_province_20m_2024dry_p25-tile2.tif

sentinel_city_10m_2024dry_p25.tif
sentinel_city_20m_2024dry_median.tif
```

### 2. RGB Visualizations

**Pattern:** `rgb_{region}_{resolution}_{timeframe}_{type}`

```
rgb_province_20m_2024dry_natural.png
rgb_province_20m_2024dry_falsecolor.png
rgb_city_10m_2024dry_natural.png
```

### 3. Classification Outputs

**Pattern:** `classification_{region}_{resolution}_{timeframe}_{model}`

```
classification_province_20m_2024dry_rf.tif
classification_city_10m_2024dry_rf.tif
```

### 4. Results & Metrics

**Pattern:** `results_{region}_{resolution}_{timeframe}_{descriptor}`

```
results_province_20m_2024dry_comparison.csv
results_province_20m_2024dry_confusion_matrix.png
results_province_20m_2024dry_feature_importance.png
```

### 5. Test Outputs

**Pattern:** `test_{type}_{region}_{resolution}_{descriptor}`

```
test_cloud_sample_20m_p25.tif
test_cloud_sample_20m_median.tif
test_cloud_comparison.png
```

### 6. Boundaries

**Pattern:** `boundary_{region}_{source}`

```
boundary_province_geoboundaries.geojson
boundary_city_geoboundaries.geojson
boundary_province_klhk.geojson
```

---

## 🗂️ Directory Structure

```
LandCover_Research/
├── data/
│   ├── sentinel/
│   │   ├── sentinel_province_20m_2024dry_p25-tile1.tif
│   │   ├── sentinel_province_20m_2024dry_p25-tile2.tif
│   │   └── ...
│   ├── sentinel_city/
│   │   ├── sentinel_city_10m_2024dry_p25.tif
│   │   └── sentinel_city_20m_2024dry_p25.tif
│   ├── boundaries/
│   │   ├── boundary_province_geoboundaries.geojson
│   │   └── boundary_city_geoboundaries.geojson
│   └── klhk/
│       └── klhk_province_2024_full.geojson
│
├── results/
│   ├── rgb/
│   │   ├── rgb_province_20m_2024dry_natural.png
│   │   ├── rgb_city_10m_2024dry_natural.png
│   │   └── rgb_comparison_province_city.png
│   ├── classification/
│   │   ├── classification_province_20m_2024dry_rf.tif
│   │   └── classification_city_10m_2024dry_rf.tif
│   ├── metrics/
│   │   ├── metrics_province_20m_2024dry_rf.csv
│   │   ├── results_province_20m_2024dry_confusion_matrix.png
│   │   └── results_province_20m_2024dry_feature_importance.png
│   └── test/
│       ├── test_cloud_sample_20m_p25.tif
│       └── test_cloud_comparison.png
│
└── docs/
    └── NAMING_STANDARDS.md
```

---

## 🔧 Implementation in Scripts

### Update download_sentinel2_flexible.py:

```python
# OLD naming:
description = f"S2_{preset['output_suffix']}_{year}_{scale}m"

# NEW naming:
region_code = 'city' if preset['region_type'] == 'city' else 'province'
strategy_code = get_strategy_code(strategy_name)  # p25, median, etc.
description = f"sentinel_{region_code}_{scale}m_{year}dry_{strategy_code}"
```

### Update generate_qualitative_FINAL.py:

```python
# OLD naming:
output_file = f'qualitative_FINAL_DRY_SEASON/province/RGB_{i}.png'

# NEW naming:
output_file = f'results/rgb/rgb_province_20m_2024dry_natural_{i}.png'
```

### Update run_classification.py:

```python
# OLD naming:
output_file = 'results/classification_results.csv'

# NEW naming:
output_file = 'results/metrics/metrics_province_20m_2024dry_comparison.csv'
```

---

## 📊 Comparison: Old vs New

### Old (Inconsistent):
```
❌ S2_jambi_2024_20m_AllBands-0000000000-0000000000.tif
❌ qualitative_FINAL_DRY_SEASON/province/RGB_1.png
❌ classification_results.csv
❌ confusion_matrix_random_forest.png
❌ test_percentile_25.tif
```

### New (Standardized):
```
✅ sentinel_province_20m_2024dry_p25-tile1.tif
✅ rgb_province_20m_2024dry_natural_1.png
✅ metrics_province_20m_2024dry_comparison.csv
✅ results_province_20m_2024dry_confusion_rf.png
✅ test_cloud_province_20m_p25.tif
```

---

## ✅ Benefits

1. **Consistency** - Same pattern everywhere
2. **Sortable** - Files naturally sort by type → region → resolution
3. **Self-describing** - Filename tells you everything
4. **No collisions** - Unique names prevent overwrites
5. **Easy filtering** - `ls rgb_city_*` finds all city RGB images
6. **Scalable** - Works for new regions, resolutions, methods

---

## 🔄 Migration Plan

### Phase 1: Update Scripts (Priority)
1. ✅ `download_sentinel2_flexible.py` - Use new naming
2. ⏳ `generate_qualitative_FINAL.py` - Update output names
3. ⏳ `run_classification.py` - Update result names
4. ⏳ `test_cloud_strategies_quick.py` - Update test names

### Phase 2: Rename Existing Files
```bash
# Old test files → New names
mv results/strategy_test/test_current.tif \
   results/test/test_cloud_sample_20m_median.tif

mv results/strategy_test/test_percentile_25.tif \
   results/test/test_cloud_sample_20m_p25.tif
```

### Phase 3: Update Documentation
- Update all guides with new naming examples
- Update README with naming standards link

---

## 📖 Quick Reference

### Common Patterns:

**Sentinel-2 Data:**
```
sentinel_{region}_{res}_{time}_{strategy}[-tile{N}].tif
```

**Visualizations:**
```
rgb_{region}_{res}_{time}_{type}[_{variant}].png
```

**Classification:**
```
classification_{region}_{res}_{time}_{model}.tif
```

**Results:**
```
results_{region}_{res}_{time}_{analysis}.{csv|png}
metrics_{region}_{res}_{time}_{model}.csv
```

**Tests:**
```
test_{type}_{region}_{res}_{descriptor}.tif
```

---

## 🎯 Helper Functions

### Create Standardized Name:

```python
def create_standard_name(category, region, resolution, timeframe, descriptor, ext):
    """Create standardized filename."""

    parts = [category, region, resolution, timeframe, descriptor]
    filename = '_'.join(parts) + '.' + ext

    return filename

# Example:
name = create_standard_name(
    category='sentinel',
    region='province',
    resolution='20m',
    timeframe='2024dry',
    descriptor='p25',
    ext='tif'
)
# Returns: 'sentinel_province_20m_2024dry_p25.tif'
```

### Parse Standard Name:

```python
def parse_standard_name(filename):
    """Parse standardized filename into components."""

    name, ext = filename.rsplit('.', 1)
    parts = name.split('_')

    return {
        'category': parts[0],
        'region': parts[1],
        'resolution': parts[2],
        'timeframe': parts[3],
        'descriptor': parts[4] if len(parts) > 4 else None,
        'extension': ext
    }

# Example:
info = parse_standard_name('sentinel_province_20m_2024dry_p25.tif')
# Returns: {
#     'category': 'sentinel',
#     'region': 'province',
#     'resolution': '20m',
#     'timeframe': '2024dry',
#     'descriptor': 'p25',
#     'extension': 'tif'
# }
```

---

## 📝 Update Checklist

**Scripts to Update:**
- [ ] download_sentinel2_flexible.py
- [ ] download_sentinel2.py (legacy)
- [ ] generate_qualitative_FINAL.py
- [ ] generate_rgb_new_data.py
- [ ] run_classification.py
- [ ] test_cloud_strategies_quick.py

**Files to Rename:**
- [ ] Existing sentinel data
- [ ] Existing RGB outputs
- [ ] Existing classification results
- [ ] Existing test outputs

**Documentation to Update:**
- [ ] README.md
- [ ] FLEXIBLE_DOWNLOAD_GUIDE.md
- [ ] CLOUD_REMOVAL_GUIDE.md
- [ ] All usage examples

---

**STATUS: Standard Defined** ✅

**Next: Implement in scripts** ⏳

---

*Clean, Consistent, Professional Naming*
*2026-01-02*
