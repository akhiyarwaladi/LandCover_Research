# Utilities

Utility scripts untuk verification dan data quality checks.

## Scripts

### `verify_final_dataset.py`
Comprehensive verification untuk final merged KLHK dataset.

**Usage:**
```bash
python utils/verify_final_dataset.py
```

**Checks:**
- Total features count
- Geometry validity (NULL check)
- OBJECTID uniqueness
- Province code consistency
- Class distribution
- Sample feature inspection

---

### `verify_geojson.py`
Quick GeoJSON structure verification.

**Usage:**
```bash
python utils/verify_geojson.py
```

**Checks:**
- File loading success
- CRS information
- Geometry types
- Column structure

---

### `verify_partitions.py`
Verify partition files have unique data (no overlaps).

**Usage:**
```bash
python utils/verify_partitions.py
```

**Checks:**
- OBJECTID uniqueness across partitions
- No duplicate features
- Partition overlap detection

---

### `compare_batches.py`
Compare different batch download results.

**Usage:**
```bash
python utils/compare_batches.py
```

**Purpose:**
- Validate batch download consistency
- Check for data duplication
- Compare OBJECTID ranges

---

## When to Use

**During Data Download:**
- Use `verify_partitions.py` to ensure batch downloads are unique

**After Merging:**
- Use `verify_final_dataset.py` for comprehensive quality check

**Debugging:**
- Use `compare_batches.py` to diagnose download issues
- Use `verify_geojson.py` for quick file structure checks

---

**Note:** These are utility scripts for data validation, not part of the main classification pipeline.
