# Cloud Removal Strategy Guide

**Centralized System for Sentinel-2 Cloud Removal**

Last Updated: 2026-01-02

---

## 📋 Quick Start

### 1. **List Available Strategies**

```bash
python scripts/test_cloud_strategies.py
```

### 2. **Choose a Strategy**

Edit `scripts/download_sentinel2.py`:

```python
'cloud_removal_strategy': 'percentile_25',  # Change this line!
```

### 3. **Download with New Strategy**

```bash
python scripts/download_sentinel2.py --mode full
```

---

## 🎯 Available Strategies

### **CURRENT** (Baseline)
- **What**: Dry season median composite
- **Cloud Score+**: 0.50
- **Max Cloud**: 40%
- **When to use**: Current baseline for comparison

### **PERCENTILE_25** ⭐ (Recommended!)
- **What**: Takes 25th percentile (darkest pixels)
- **Cloud Score+**: 0.55
- **Max Cloud**: 50%
- **Expected**: 90-95% cloud-free
- **Trade-off**: May lose 5-10% edge data
- **Source**: Corbane et al. 2015 - Global cloud-free composites
- **When to use**: High cloud cover (like Jambi)

### **KALIMANTAN** (Indonesia Proven)
- **What**: Pre-filter 5% + median
- **Cloud Score+**: 0.60
- **Max Cloud**: 5% (strict pre-filter)
- **Expected**: 95%+ cloud-free
- **Trade-off**: Fewer images available
- **Source**: Central Kalimantan study - 99.1% accuracy
- **When to use**: Maximum quality needed

### **PAN_TROPICAL** (Standard)
- **What**: Annual median (full year)
- **Cloud Score+**: 0.60
- **Max Cloud**: 60%
- **Expected**: 85-90% cloud-free
- **Source**: Simonetti et al. 2021 - Pan-tropical dataset
- **When to use**: Need full annual coverage

### **BALANCED** (Compromise)
- **What**: Percentile 30 (less aggressive)
- **Cloud Score+**: 0.55
- **Max Cloud**: 40%
- **Expected**: 85-90% cloud-free
- **When to use**: Balance cloud removal & data retention

### **CONSERVATIVE** (Data Priority)
- **What**: Median with high threshold
- **Cloud Score+**: 0.65
- **Max Cloud**: 50%
- **Expected**: 80-85% cloud-free
- **When to use**: Data retention more important than cloud-free

---

## 🔧 How It Works

### Architecture

```
scripts/download_sentinel2.py
    ↓ uses
modules/cloud_removal.py
    ↓ contains
CloudRemovalConfig (6 strategies)
    ↓ applies to
Sentinel-2 imagery → Cloud-free composite
```

### Centralized Configuration

All cloud removal logic is in `modules/cloud_removal.py`:

```python
from modules.cloud_removal import CloudRemovalConfig

# Get strategy config
config = CloudRemovalConfig.get_strategy('percentile_25')

# Apply to imagery
composite = process_sentinel2_with_strategy(
    region, start_date, end_date,
    strategy='percentile_25'
)
```

### Easy to Modify

Want to tweak a strategy?

1. Open `modules/cloud_removal.py`
2. Edit the strategy in `CloudRemovalConfig.STRATEGIES`
3. All scripts automatically use updated config!

---

## 📊 Comparison Workflow

### Test Multiple Strategies

```bash
# 1. Baseline
python scripts/download_sentinel2.py --mode full
mv data/sentinel_new data/sentinel_current

# 2. Edit config: 'current' → 'percentile_25'
python scripts/download_sentinel2.py --mode full
mv data/sentinel_new data/sentinel_percentile25

# 3. Compare visually
python scripts/generate_rgb_new_data.py --input data/sentinel_current
python scripts/generate_rgb_new_data.py --input data/sentinel_percentile25
```

### Metrics to Compare

1. **Visual inspection**: Residual clouds visible?
2. **% Valid pixels**: How much data retained?
3. **Classification accuracy**: Does cleaner data = better classification?

---

## 🎓 Research Citations

### Percentile 25
- **Corbane et al. (2015)**: [A global cloud free pixel-based image composite from Sentinel-2](https://pmc.ncbi.nlm.nih.gov/articles/PMC7262415/)
- Best for high cloud cover tropical regions
- Visual quality rated superior by trained analysts

### Kalimantan Method
- **Forest Disturbance Monitoring (2024)**: [Central Kalimantan Study](https://pmc.ncbi.nlm.nih.gov/articles/PMC10817504/)
- 99.1% producer accuracy for forest damage detection
- Proven in Indonesia specifically

### Pan-Tropical
- **Simonetti et al. (2021)**: [Pan-tropical Sentinel-2 composites](https://pmc.ncbi.nlm.nih.gov/articles/PMC8545689/)
- Successfully processed entire tropics (2015-2020)
- Works even in >80% cloud cover regions

### Cloud Score+
- **Google Earth Engine**: [Cloud Score+ Documentation](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_CLOUD_SCORE_PLUS_V1_S2_HARMONIZED)
- 89.4% clear pixel success rate in tropics (Hawaii study)
- Optimal threshold: 0.50-0.65 for tropical regions

---

## 🚀 Advanced: Custom Strategy

### Create Your Own Strategy

Edit `modules/cloud_removal.py`:

```python
'my_custom': {
    'name': 'My Custom Strategy',
    'description': 'Optimized for my specific needs',
    'cloud_score_threshold': 0.58,  # Your value
    'max_cloud_percent': 35,        # Your value
    'composite_method': 'percentile_30',  # Your choice
    'pre_filter_percent': None,
    'post_processing': True,
    'source': 'Custom - based on local testing'
}
```

Then use it:

```python
'cloud_removal_strategy': 'my_custom'
```

---

## 📈 Expected Results

### For Jambi Province

| Strategy | Cloud-Free % | Data Retention | Processing Time |
|----------|--------------|----------------|-----------------|
| Current | 53% | High | Fast |
| Percentile 25 | 90-95% | Medium | Fast |
| Kalimantan | 95%+ | Low-Medium | Fast |
| Pan-Tropical | 85-90% | High | Medium |

**Recommendation**: Start with **percentile_25**, then adjust based on results.

---

## ❓ FAQ

### Q: Which strategy should I use?
**A**: For Jambi (high cloud cover), use **percentile_25** first.

### Q: Will I lose data?
**A**: Percentile 25 might lose 5-10% at edges, but removes 90%+ clouds. Net benefit!

### Q: Can I combine strategies?
**A**: Yes! Modify `modules/cloud_removal.py` to create hybrid approaches.

### Q: How do I validate results?
**A**:
1. Visual inspection (look for residual clouds)
2. Check % valid pixels retained
3. Run classification - does accuracy improve?

### Q: Processing taking too long?
**A**: Use `pre_filter_percent: 5` to limit images processed. Trade data quantity for speed.

---

## 🔄 Maintenance

### Adding New Research

1. Find new method in paper
2. Add to `CloudRemovalConfig.STRATEGIES` in `modules/cloud_removal.py`
3. Document source and expected results
4. Test and compare with existing strategies

### Version Control

```bash
git add modules/cloud_removal.py
git commit -m "Add new cloud removal strategy: [name]"
```

All strategies versioned together!

---

## 📞 Support

**Issues**: Check existing strategies and their sources in `modules/cloud_removal.py`

**Documentation**: Full technical details in module docstrings

**Research**: Citations and links provided for each strategy
