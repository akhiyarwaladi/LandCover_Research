# Flexible Sentinel-2 Download Guide

**Easy Download: Any Region, Any Resolution**

Last Updated: 2026-01-02

---

## 🎯 Quick Start

### Super Simple - 4 Presets Ready:

```bash
# 1. CITY at 10m (RECOMMENDED for urban detail)
python scripts/download_sentinel2_flexible.py --preset city_10m

# 2. City at 20m (all bands, smaller file)
python scripts/download_sentinel2_flexible.py --preset city_20m

# 3. Province at 20m (standard, all bands)
python scripts/download_sentinel2_flexible.py --preset province_20m

# 4. Province at 10m (high detail RGB, large file)
python scripts/download_sentinel2_flexible.py --preset province_10m
```

**That's it! One line, no code editing needed.**

---

## 📊 Preset Comparison

| Preset | Region | Resolution | Bands | File Size | Download Time | Use Case |
|--------|--------|------------|-------|-----------|---------------|----------|
| **city_10m** ⭐ | Jambi City | 10m | 4 | ~40 MB | ~5 min | **Urban detail, buildings** |
| city_20m | Jambi City | 20m | 10 | ~10 MB | ~3 min | City classification |
| province_20m | Full Province | 20m | 10 | ~2.7 GB | ~30 min | Province classification |
| province_10m | Full Province | 10m | 4 | ~4 GB | ~40 min | High-res visualization |

---

## 🔍 Resolution Details

### 10m Resolution (High Detail)
**Native Sentinel-2 Bands:**
- B2 - Blue (490 nm)
- B3 - Green (560 nm)
- B4 - Red (665 nm)
- B8 - NIR (842 nm)

**Total: 4 bands**

**What you can see:**
- ✅ Individual buildings
- ✅ Roads and streets
- ✅ Small water bodies
- ✅ Tree canopy detail
- ✅ Urban features

**Best for:**
- Urban mapping
- Building detection
- High-detail visualization
- RGB imagery

**File size: 4× larger than 20m** (more pixels)

### 20m Resolution (More Bands)
**Native Sentinel-2 Bands:**
- B2, B3, B4 (RGB) - 10m resampled to 20m
- B5, B6, B7, B8A (Red Edge) - Native 20m
- B8 (NIR) - 10m resampled to 20m
- B11, B12 (SWIR) - Native 20m

**Total: 10 bands**

**What you get:**
- ✅ Red Edge bands (vegetation health)
- ✅ SWIR bands (water, soil, minerals)
- ✅ More spectral indices possible
- ✅ Better classification

**Best for:**
- Land cover classification
- Vegetation analysis
- Multi-spectral analysis
- NDVI, EVI, NDWI, etc.

**File size: Smaller** (fewer pixels, but more bands)

---

## 🎯 Which Preset to Choose?

### For Jambi CITY Analysis:

**Option 1: city_10m** (RECOMMENDED)
```bash
python scripts/download_sentinel2_flexible.py --preset city_10m
```
- ✅ See buildings clearly
- ✅ Urban detail (10m = ~30 feet)
- ✅ Fast download (~40 MB)
- ✅ Perfect for city mapping
- ❌ Only 4 bands (RGB + NIR)

**Option 2: city_20m**
```bash
python scripts/download_sentinel2_flexible.py --preset city_20m
```
- ✅ All 10 bands (SWIR, Red Edge)
- ✅ Better for classification
- ✅ Very fast (~10 MB)
- ❌ Less spatial detail

**My Recommendation:** Start with **city_10m** for visualization, then get **city_20m** if you need classification with all bands.

### For PROVINCE Analysis:

**Option 1: province_20m** (STANDARD)
```bash
python scripts/download_sentinel2_flexible.py --preset province_20m
```
- ✅ All 10 bands
- ✅ Good resolution
- ✅ Reasonable file size (2.7 GB)
- ✅ Best for classification

**Option 2: province_10m** (only if needed)
```bash
python scripts/download_sentinel2_flexible.py --preset province_10m
```
- ✅ High detail RGB
- ❌ Large file (4 GB)
- ❌ Only 4 bands
- ❌ Longer processing

**My Recommendation:** Use **province_20m** (already running!) - it's the best balance.

---

## 🚀 Advanced Usage

### Custom Strategy:

```bash
# City at 10m with Kalimantan strategy
python scripts/download_sentinel2_flexible.py --preset city_10m --strategy kalimantan

# Province at 20m with balanced strategy
python scripts/download_sentinel2_flexible.py --preset province_20m --strategy balanced
```

### Available Strategies:
- `percentile_25` - Best for cloud removal (default, 99.1% tested)
- `kalimantan` - Indonesia proven (strict)
- `balanced` - Compromise
- `pan_tropical` - Standard tropics
- `current` - Baseline
- `conservative` - Max data retention

### List All Options:

```bash
python scripts/download_sentinel2_flexible.py --list
```

---

## 📁 Output Structure

### City Downloads → `GEE_Exports_City/`

```
Google Drive/GEE_Exports_City/
├── S2_city_10m_2024_10m.tif       (10m resolution, 4 bands, ~40 MB)
└── S2_city_20m_2024_20m.tif       (20m resolution, 10 bands, ~10 MB)
```

### Province Downloads → `GEE_Exports/`

```
Google Drive/GEE_Exports/
├── S2_province_20m_2024_20m-0000000000-0000000000.tif
├── S2_province_20m_2024_20m-0000000000-0000010496.tif
├── S2_province_20m_2024_20m-0000010496-0000000000.tif
└── S2_province_20m_2024_20m-0000010496-0000010496.tif
```

**Save locally to:**
- City: `data/sentinel_city/`
- Province: `data/sentinel/`

---

## ⏱️ Expected Timeline

### City at 10m (~40 MB):
```
1. Processing: ~5 minutes
2. Export to Drive: ~2 minutes
3. Download: <1 minute
Total: ~10 minutes
```

### City at 20m (~10 MB):
```
1. Processing: ~3 minutes
2. Export to Drive: ~1 minute
3. Download: <1 minute
Total: ~5 minutes
```

### Province at 20m (~2.7 GB):
```
1. Processing: ~20 minutes
2. Export to Drive: ~30 minutes
3. Download: ~5 minutes
Total: ~55 minutes
```

### Province at 10m (~4 GB):
```
1. Processing: ~30 minutes
2. Export to Drive: ~40 minutes
3. Download: ~10 minutes
Total: ~80 minutes
```

---

## 📊 File Size Comparison

**Why is city 10m larger than city 20m?**

- **10m**: 4 bands, but 4× more pixels → ~40 MB
- **20m**: 10 bands, but 4× fewer pixels → ~10 MB

**Math:**
```
City area: 172 km²

At 10m: 172,000,000 m² ÷ 100 m²/pixel = 1,720,000 pixels
  → 1.72M pixels × 4 bands = 6.88M values

At 20m: 172,000,000 m² ÷ 400 m²/pixel = 430,000 pixels
  → 0.43M pixels × 10 bands = 4.3M values

10m has more total data values!
```

---

## 🔧 How the System Works

### Clean Design Features:

**1. Preset-Based Configuration**
- No code editing needed
- All settings in one place
- Easy to understand

**2. Automatic Band Selection**
- 10m presets → only native 10m bands (B2, B3, B4, B8)
- 20m presets → all bands (B2-B12)
- No manual band configuration

**3. Automatic Boundary Loading**
- Province → loads from GeoBoundaries ADM1
- City → loads from GeoBoundaries ADM2
- Same data source, consistent

**4. Integrated Cloud Removal**
- Uses same strategy system
- Tested percentile_25 by default
- Easy to override

**5. Smart Output Naming**
- Includes region, resolution, year
- Separate folders for city/province
- No file conflicts

---

## 🎓 Technical Details

### Why Not Upsample 20m to 10m?

**BAD (don't do):**
```python
# Upsampling 20m bands to 10m - NO NEW DETAIL!
composite.resample('bilinear').reproject(scale=10)
```

**GOOD (what we do):**
```python
# Use native 10m bands at 10m
# Use native 20m bands at 20m
```

**Reason:** Upsampling doesn't add real detail, just interpolates pixels. Better to use native resolution for each band.

### Band Selection Logic:

```python
if scale == 10:
    bands = ['B2', 'B3', 'B4', 'B8']  # Only native 10m
else:  # scale == 20
    bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
```

---

## ✅ Current Running Tasks

**As of now (2026-01-02):**

```
1. ✅ Province at 20m (percentile_25) - RUNNING
   → S2_jambi_2024_20m_AllBands
   → Folder: GEE_Exports
   → Size: ~2.7 GB
   → ETA: ~30 minutes

2. ✅ City at 10m (percentile_25) - RUNNING
   → S2_city_10m_2024_10m
   → Folder: GEE_Exports_City
   → Size: ~40 MB
   → ETA: ~5 minutes
```

**Check status:**
```bash
python scripts/check_task_status.py
```

---

## 🎯 Recommended Workflow

### Step 1: Download City at 10m (DONE - running now)
```bash
python scripts/download_sentinel2_flexible.py --preset city_10m
```
**Why:** Fast, high detail, perfect for urban analysis

### Step 2: Download Province at 20m (DONE - running now)
```bash
python scripts/download_sentinel2_flexible.py --preset province_20m
```
**Why:** Full spectral bands, good for classification

### Step 3: (Optional) City at 20m if needed
```bash
python scripts/download_sentinel2_flexible.py --preset city_20m
```
**Why:** Get all bands for city classification

### Step 4: Compare Results
- City 10m → High detail RGB/NIR visualization
- Province 20m → Province-wide classification
- City 20m → City classification with full bands

---

## 📝 Examples

### Example 1: Quick City Analysis
```bash
# Download city at 10m
python scripts/download_sentinel2_flexible.py --preset city_10m

# Wait ~5 min, download from Drive

# Generate RGB visualization
python scripts/generate_rgb_city.py

# See buildings clearly at 10m!
```

### Example 2: Full Province Classification
```bash
# Download province at 20m (already running!)
# Wait ~30 min, download from Drive

# Run classification
python scripts/run_classification.py

# Get accuracy with all bands
```

### Example 3: Both City and Province
```bash
# City detail
python scripts/download_sentinel2_flexible.py --preset city_10m

# Province overview
python scripts/download_sentinel2_flexible.py --preset province_20m

# Compare scales and detail levels
```

---

## 🆚 Old vs New System

### OLD Way (download_sentinel2.py):
```python
# Edit code every time
CONFIG = {
    'region_name': 'jambi',  # Edit here
    'scale': 20,              # Edit here
    # ...
}

# Run
python scripts/download_sentinel2.py --mode full

# Hard to switch regions/resolutions
```

### NEW Way (download_sentinel2_flexible.py):
```bash
# Just change preset - NO code editing!
python scripts/download_sentinel2_flexible.py --preset city_10m
python scripts/download_sentinel2_flexible.py --preset province_20m

# Clean, simple, flexible
```

---

## 🧹 Clean Codebase

**New Files:**
```
scripts/
└── download_sentinel2_flexible.py  ✅ New flexible system

docs/
└── FLEXIBLE_DOWNLOAD_GUIDE.md     ✅ This guide
```

**Old Files (keep for compatibility):**
```
scripts/
├── download_sentinel2.py           ⚠️  Old single-purpose (keep for now)
└── ...
```

**Recommendation:** Use `download_sentinel2_flexible.py` for all new downloads.

---

## 📞 Support

**List presets:**
```bash
python scripts/download_sentinel2_flexible.py --list
```

**Check task status:**
```bash
python scripts/check_task_status.py
```

**Help:**
```bash
python scripts/download_sentinel2_flexible.py --help
```

---

## 🏆 Summary

**What We Built:**
- ✅ 4 presets (city/province × 10m/20m)
- ✅ One-line usage
- ✅ No code editing
- ✅ Clean, maintainable
- ✅ Integrated with cloud removal
- ✅ Smart band selection
- ✅ Automatic boundary loading

**What You Get:**
- 🎯 City at 10m: High detail urban mapping
- 🎯 City at 20m: Full spectral city analysis
- 🎯 Province at 20m: Standard classification
- 🎯 Province at 10m: High-res visualization

**Time Saved:**
- No more config editing
- No more boundary setup
- No more band selection
- Just one command!

---

**READY TO USE** ✅

**Currently Running:**
1. City 10m → ~5 min
2. Province 20m → ~30 min

**Download from Google Drive when done!**

---

*Clean, Flexible, Easy to Use*
*Built: 2026-01-02*
*Author: Claude Sonnet 4.5*
