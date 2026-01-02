# Quick Start: Cloud Removal System

**Centralized Cloud Removal - Ready to Use!**

Last Updated: 2026-01-02

---

## ✅ System is Ready!

The centralized cloud removal system has been fully integrated. You can now switch between 6 different research-based strategies by changing **ONE LINE** of code.

---

## 🚀 How to Use

### Step 1: Choose Your Strategy

Open `scripts/download_sentinel2.py` and find this line (around line 53):

```python
'cloud_removal_strategy': 'current',  # Change this to test different strategies!
```

### Step 2: Change Strategy

Replace `'current'` with one of these options:

```python
# Recommended for Jambi (high cloud cover)
'cloud_removal_strategy': 'percentile_25',  # 90-95% cloud-free

# Proven in Indonesia
'cloud_removal_strategy': 'kalimantan',    # 95%+ cloud-free (strict filtering)

# Standard for tropics
'cloud_removal_strategy': 'pan_tropical',   # 85-90% cloud-free

# Balanced approach
'cloud_removal_strategy': 'balanced',       # 85-90% cloud-free

# Keep more data
'cloud_removal_strategy': 'conservative',   # 80-85% cloud-free (accepts some clouds)
```

### Step 3: Run the Download

```bash
python scripts/download_sentinel2.py --mode full
```

That's it! The script will:
- Load your chosen strategy
- Display strategy details
- Apply optimal cloud filtering
- Create composite using the best method
- Export to Google Drive

---

## 📊 Strategy Comparison

| Strategy | Cloud-Free | Data Loss | Best For |
|----------|-----------|-----------|----------|
| **percentile_25** ⭐ | 90-95% | 5-10% edges | High cloud cover (Jambi) |
| **kalimantan** 🔬 | 95%+ | 10-20% | Maximum quality needed |
| **balanced** ⚖️ | 85-90% | <5% | Safe middle ground |
| **pan_tropical** 🌍 | 85-90% | Minimal | Standard approach |
| **conservative** 📈 | 80-85% | None | Data retention priority |
| **current** 📌 | 53% | None | Baseline (dry season median) |

---

## 🎯 Recommended Test Plan

### Option 1: Quick Test (1 strategy)

```python
# In download_sentinel2.py:
'cloud_removal_strategy': 'percentile_25',
```

```bash
python scripts/download_sentinel2.py --mode full
# Wait 20-30 minutes for export
# Download from Google Drive to: data/sentinel_percentile25/
# Generate visualizations
python scripts/generate_qualitative_FINAL.py
```

### Option 2: Full Comparison (3 strategies)

```bash
# Strategy 1: Current (baseline)
# Keep: 'cloud_removal_strategy': 'current'
python scripts/download_sentinel2.py --mode full
# Download to: data/sentinel_current/

# Strategy 2: Percentile 25 (recommended)
# Change to: 'cloud_removal_strategy': 'percentile_25'
python scripts/download_sentinel2.py --mode full
# Download to: data/sentinel_percentile25/

# Strategy 3: Kalimantan (Indonesia proven)
# Change to: 'cloud_removal_strategy': 'kalimantan'
python scripts/download_sentinel2.py --mode full
# Download to: data/sentinel_kalimantan/

# Compare all three visually
python scripts/compare_strategies_visual.py  # (create this script if needed)
```

---

## 🧪 Test Current Strategy

To see what strategy is configured without downloading:

```bash
python scripts/test_cloud_strategies.py
```

This will show:
- All available strategies
- Current configuration
- Recommendations based on your data
- Expected results for each method

---

## 📖 Full Documentation

For detailed information:
- **Full guide**: `docs/CLOUD_REMOVAL_GUIDE.md`
- **Module code**: `modules/cloud_removal.py`
- **Research citations**: Included in module docstrings

---

## 🔧 Advanced: Custom Strategy

Want to create your own? Edit `modules/cloud_removal.py`:

```python
# Add to CloudRemovalConfig.STRATEGIES dictionary:
'my_custom': {
    'name': 'My Custom Strategy',
    'description': 'Custom parameters for specific needs',
    'cloud_score_threshold': 0.58,
    'max_cloud_percent': 35,
    'composite_method': 'percentile_30',
    'pre_filter_percent': None,
    'post_processing': True,
    'source': 'Custom - based on testing'
}
```

Then use it:
```python
'cloud_removal_strategy': 'my_custom',
```

---

## ✅ What Changed?

**Before:**
- Cloud parameters hardcoded in multiple places
- Difficult to test different approaches
- No centralized research-based methods

**After:**
- ✅ All strategies centralized in `modules/cloud_removal.py`
- ✅ Change strategy with **one line** edit
- ✅ 6 research-based methods ready to use
- ✅ Full documentation and citations
- ✅ Easy to add new strategies

---

## 📞 Need Help?

1. Run the test script: `python scripts/test_cloud_strategies.py`
2. Check full guide: `docs/CLOUD_REMOVAL_GUIDE.md`
3. See module source: `modules/cloud_removal.py`

---

**Ready to remove those clouds? Just change one line and run!** 🚀
