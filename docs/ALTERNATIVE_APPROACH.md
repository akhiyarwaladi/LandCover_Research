# Alternative Classification Approach - Tanpa KLHK Geometry

**Problem:** KLHK data tidak ada geometry, server restrict access
**Solution:** Hybrid unsupervised + validation approach

---

## 🎯 Approach: Unsupervised Classification + KLHK Validation

### Konsep:

1. **Lakukan unsupervised classification** (K-means, ISODATA)
   - Input: Sentinel-2 bands + spectral indices
   - Output: Clusters (misal 9-15 clusters)
   - No ground truth needed!

2. **Interpretasi clusters** berdasarkan:
   - Spectral signatures
   - Visual inspection dengan RGB composite
   - NDVI/EVI patterns
   - Lokasi geografis

3. **Validasi menggunakan KLHK attributes**:
   - Kita punya distribusi kelas KLHK (28,100 records):
     - 26.5% Hutan Tanaman
     - 19.0% Hutan Lahan Kering
     - 13.6% Perkebunan
     - dll
   - Compare cluster distribution vs KLHK distribution
   - Match clusters to KLHK classes based on statistics

4. **Semi-supervised refinement** (optional):
   - Sample-based labeling
   - Active learning
   - Transfer learning

---

## 📊 Workflow Detail

### Step 1: Unsupervised Classification

```python
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

# Load Sentinel-2
# Calculate indices (NDVI, EVI, etc.)
# Stack all features

# K-means clustering
n_clusters = 9  # Match KLHK major classes
kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(features)

# Save cluster map
```

**Advantages:**
- ✅ No ground truth geometry needed
- ✅ Works with current data
- ✅ Fast processing
- ✅ Good for exploratory analysis

### Step 2: Cluster Interpretation

**Manual interpretation based on:**

| Cluster | NDVI Range | NDWI | Visual | Likely Class |
|---------|-----------|------|---------|--------------|
| 0 | -0.2-0.1 | 0.3-0.8 | Blue | Water |
| 1 | 0.6-0.9 | <0 | Dark green | Dense forest |
| 2 | 0.4-0.6 | <0 | Light green | Plantation |
| 3 | 0.2-0.4 | <0 | Yellow | Grassland |
| 4 | 0.3-0.5 | <0 | Brown-green | Agriculture |
| ... | ... | ... | ... | ... |

### Step 3: Statistical Validation

```python
# Compare cluster distribution with KLHK class distribution
klhk_dist = {
    'Hutan Tanaman': 26.5,
    'Hutan Lahan Kering': 19.0,
    'Perkebunan': 13.6,
    'Tanah Terbuka': 13.3,
    # ...
}

# Calculate cluster areas
cluster_dist = calculate_cluster_distribution(cluster_map)

# Match clusters to KLHK classes
# Based on spectral similarity + distribution matching
```

### Step 4: Accuracy Assessment (Without Full Ground Truth)

**Methods:**

1. **Visual Assessment:**
   - High-res imagery comparison (Google Earth)
   - Expert interpretation
   - Field photos (if available)

2. **Consistency Metrics:**
   - Cluster compactness
   - Silhouette score
   - Davies-Bouldin index

3. **Cross-validation with Known Areas:**
   - Sample urban areas (should be Built class)
   - Sample water bodies (should be Water)
   - Sample known plantations

4. **KLHK Statistics Comparison:**
   ```python
   # Chi-square test
   # Compare expected (KLHK) vs observed (our clusters) distributions
   from scipy.stats import chisquare

   statistic, pvalue = chisquare(cluster_counts, klhk_counts)
   print(f"Distribution match p-value: {pvalue}")
   ```

---

## 🛠️ Implementation Script

### Create: `scripts/unsupervised_classification.py`

```python
#!/usr/bin/env python3
"""
Unsupervised Land Cover Classification for Jambi Province
Uses K-means clustering + KLHK distribution validation
"""

import numpy as np
import rasterio
from rasterio.windows import Window
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import json

def load_sentinel_tiles(tile_paths):
    """Load and mosaic Sentinel-2 tiles"""
    # Implementation
    pass

def calculate_indices(bands):
    """Calculate spectral indices"""
    indices = {}

    # NDVI
    indices['ndvi'] = (bands['B8'] - bands['B4']) / (bands['B8'] + bands['B4'] + 1e-10)

    # EVI
    indices['evi'] = 2.5 * ((bands['B8'] - bands['B4']) /
                            (bands['B8'] + 6*bands['B4'] - 7.5*bands['B2'] + 1))

    # NDWI
    indices['ndwi'] = (bands['B3'] - bands['B8']) / (bands['B3'] + bands['B8'] + 1e-10)

    # Add more indices...

    return indices

def kmeans_classification(features, n_clusters=9, sample_size=100000):
    """
    Perform K-means clustering on features

    Args:
        features: stacked feature array (n_pixels, n_features)
        n_clusters: number of clusters (should match major KLHK classes)
        sample_size: subsample for faster processing
    """
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # K-means
    print(f"Running K-means with {n_clusters} clusters...")
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=42,
        batch_size=10000,
        max_iter=100,
        verbose=1
    )

    labels = kmeans.fit_predict(features_scaled)

    return labels, kmeans, scaler

def interpret_clusters(cluster_map, features, n_clusters):
    """
    Analyze cluster characteristics
    """
    results = {}

    for cluster_id in range(n_clusters):
        mask = cluster_map == cluster_id
        cluster_features = features[mask]

        results[cluster_id] = {
            'count': mask.sum(),
            'percentage': (mask.sum() / mask.size) * 100,
            'mean_features': cluster_features.mean(axis=0),
            'std_features': cluster_features.std(axis=0)
        }

    return results

def compare_with_klhk(cluster_distribution, klhk_distribution):
    """
    Compare cluster distribution with KLHK reference
    """
    from scipy.stats import chisquare

    # Normalize distributions
    cluster_pct = np.array([v['percentage'] for v in cluster_distribution.values()])
    klhk_pct = np.array(list(klhk_distribution.values()))

    # Chi-square test
    statistic, pvalue = chisquare(cluster_pct, klhk_pct)

    print(f"\nDistribution Comparison:")
    print(f"Chi-square statistic: {statistic:.4f}")
    print(f"P-value: {pvalue:.4f}")

    if pvalue > 0.05:
        print("✅ Cluster distribution MATCHES KLHK distribution")
    else:
        print("⚠️  Cluster distribution DIFFERS from KLHK")

    return statistic, pvalue

# Main execution
if __name__ == "__main__":
    # Load data
    # Calculate features
    # Run K-means
    # Interpret clusters
    # Validate with KLHK stats
    # Save results
    pass
```

---

## 📈 Expected Results

### Outputs:

1. **Cluster Map (GeoTIFF)**
   - 9-15 classes
   - Same extent as Sentinel-2

2. **Cluster Statistics (JSON)**
   ```json
   {
     "cluster_0": {
       "likely_class": "Water",
       "area_km2": 150.5,
       "percentage": 2.1,
       "mean_ndvi": -0.15,
       "mean_ndwi": 0.65
     },
     "cluster_1": {
       "likely_class": "Dense Forest",
       "area_km2": 3200.8,
       "percentage": 28.5,
       "mean_ndvi": 0.75,
       "mean_ndwi": -0.20
     }
   }
   ```

3. **Visualization (PNG)**
   - Cluster map with colors
   - Comparison chart: Cluster % vs KLHK %
   - Feature space plots (PCA)

4. **Validation Report (TXT)**
   - Distribution comparison
   - Chi-square test results
   - Cluster interpretations

---

## ✅ Advantages of This Approach

1. **Works with Current Data** ✅
   - No KLHK geometry needed
   - Uses Sentinel-2 we already have

2. **Scientifically Valid** ✅
   - Unsupervised is accepted method
   - KLHK stats provide validation
   - Reproducible

3. **Fast** ✅
   - Can run immediately
   - No waiting for data access

4. **Flexible** ✅
   - Can refine later with ground truth
   - Can add supervised step if KLHK geometry obtained

---

## 🔄 Future Enhancement (When KLHK Geometry Available)

```python
# Step 1: Already have unsupervised clusters
clusters = load_cluster_map()

# Step 2: Get KLHK geometry (when available)
klhk_geom = gpd.read_file('klhk_with_geometry.geojson')

# Step 3: Extract training samples
training_samples = extract_samples(sentinel2, klhk_geom, n_samples=5000)

# Step 4: Supervised classification
rf = RandomForestClassifier()
rf.fit(training_samples['features'], training_samples['labels'])

# Step 5: Hybrid approach
# Use unsupervised for unknown areas
# Use supervised for areas with KLHK coverage
# Combine results
```

---

## 🎯 RECOMMENDATION

**START WITH UNSUPERVISED NOW:**
1. Run K-means classification on Sentinel-2
2. Interpret clusters using spectral signatures
3. Validate distribution vs KLHK statistics
4. Publish with caveats about lack of pixel-level ground truth

**PARALLEL:** Contact KLHK for geometry access
- If granted → upgrade to supervised
- If denied → unsupervised is still valid research

---

## 📚 References

Unsupervised classification is widely accepted in remote sensing:

- Duda & Hart (1973) - Pattern Classification
- Richards (2013) - Remote Sensing Digital Image Analysis
- Congalton & Green (2019) - Assessing Accuracy of Remotely Sensed Data

**Key Point:** When ground truth is unavailable/limited, unsupervised + expert interpretation is standard practice.

---

**Ready to implement this approach?**
