# Tests & Debug Scripts

Test scripts, debug tools, dan legacy code untuk development dan troubleshooting.

## Test Scripts

### Geometry Investigation

#### `debug_geometry.py`
Deep investigation kenapa KLHK API returns NULL geometry.

**Purpose:**
- Test different `returnGeometry` parameter values
- Compare ESRI JSON vs GeoJSON formats
- Check service capabilities and restrictions
- Document root cause of geometry issue

**Findings:**
- Server-side restriction blocks geometry untuk public queries
- Recommendation: Use KMZ export atau contact KLHK

---

#### `test_geojson_vs_kmz.py`
Comparison test: GeoJSON vs KMZ format geometry access.

**Result:**
- ❌ `f=geojson`: Returns NULL geometry
- ✅ `f=kmz`: Returns full geometry

**Conclusion:** Only KMZ export bypasses restriction.

---

#### `test_geometry.py`
Quick geometry access test script.

---

### API Exploration

#### `test_esri_json.py`
Test ESRI JSON format response.

#### `check_service.py`
Check ArcGIS REST service capabilities.

#### `try_featureserver.py`
Attempt to use FeatureServer endpoint instead of MapServer.

#### `find_download_option.py`
Search for alternative download options.

#### `download_with_gdal.py`
Attempt download using GDAL/OGR tools.

---

## Legacy Code

### `legacy/`
Contains old classification scripts before modular refactoring.

**Files:**
- `land_cover_classification.py` - Original monolithic classification script

**Note:** Kept for reference, use `scripts/run_classification.py` instead.

---

### `satellite/`
Old Earth Engine satellite download modules.

**Files:**
- `__init__.py`, `auth.py`, `boundaries.py`, `config.py`
- `export.py`, `indices.py`, `landsat.py`, `sentinel2.py`

**Note:** Superseded by `scripts/download_sentinel2.py`.

---

## Old Download Scripts

### `download_klhk_old.py`
Original KLHK download script without pagination (returns NULL geometry).

**Issue:** No pagination support, geometry always NULL.

**Replaced by:** `scripts/download_klhk_kmz_partitioned.py`

---

### `download_klhk_kmz_batch.py`
First attempt at batch KMZ download using `resultOffset`.

**Issue:** `resultOffset` doesn't work with KMZ format - all batches returned same data.

**Replaced by:** `scripts/download_klhk_kmz_partitioned.py` (uses WHERE clause partitioning)

---

### `download_satellite.py`
Old satellite download script.

**Replaced by:** `scripts/download_sentinel2.py`

---

### `try_old_klhk_data.py`
Test script untuk coba download older KLHK years (2019-2023).

**Result:** All years return NULL geometry.

---

### `land_cover_classification_klhk_old.py`
Original monolithic classification script before modular refactoring.

**Replaced by:**
- Modular architecture in `modules/`
- Main orchestrator: `scripts/run_classification.py`

---

## Usage

These scripts are for:
- ✅ **Reference** - Understanding how we solved problems
- ✅ **Debugging** - Troubleshooting similar issues
- ✅ **Learning** - See evolution of solution
- ❌ **NOT for production use** - Use scripts/ instead

---

## Running Test Scripts

```bash
# Activate environment
conda activate landcover_jambi

# Run any test script
python tests/test_geojson_vs_kmz.py
python tests/debug_geometry.py
```

**Note:** Test scripts may require manual configuration of file paths.

---

## Directory Structure

```
tests/
├── README.md                              # This file
│
├── Geometry Tests
│   ├── debug_geometry.py                  # Deep geometry investigation
│   ├── test_geojson_vs_kmz.py            # Format comparison
│   ├── test_geometry.py                   # Quick geometry test
│   └── test_esri_json.py                  # ESRI JSON format test
│
├── API Exploration
│   ├── check_service.py                   # Service capabilities
│   ├── try_featureserver.py              # FeatureServer endpoint test
│   ├── find_download_option.py           # Alternative download methods
│   └── download_with_gdal.py             # GDAL download attempt
│
├── Legacy Downloads
│   ├── download_klhk_old.py              # Original (no pagination)
│   ├── download_klhk_kmz_batch.py        # Batch with resultOffset (failed)
│   ├── download_satellite.py             # Old satellite download
│   └── try_old_klhk_data.py              # Test older years
│
├── Legacy Classification
│   └── land_cover_classification_klhk_old.py  # Monolithic script
│
└── Legacy Modules
    ├── legacy/                            # Old classification modules
    │   └── land_cover_classification.py
    └── satellite/                         # Old Earth Engine modules
        ├── auth.py, boundaries.py, config.py
        ├── export.py, indices.py
        └── landsat.py, sentinel2.py
```

---

**Maintenance:** Keep for historical reference and debugging, but do not modify.
**Production code:** Located in `scripts/` and `modules/`.
