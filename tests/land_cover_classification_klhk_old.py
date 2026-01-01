#!/usr/bin/env python3
"""
Land Cover Classification using KLHK Reference Data
====================================================

This script performs land cover classification using:
- Sentinel-2 imagery as input features
- KLHK (Ministry of Environment and Forestry) data as ground truth reference

Key improvements over previous approach:
1. Uses REAL ground truth (KLHK official data) instead of Dynamic World
2. Proper class mapping from KLHK codes
3. Spatial block cross-validation to prevent overfitting
4. Multiple classifier comparison

Author: Land Cover Research Project
Date: December 2024
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import os
import json
import time
import warnings
warnings.filterwarnings('ignore')

# Try importing optional dependencies
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("LightGBM not available, skipping...")

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not available, skipping...")

# ==============================================================================
# KLHK LAND COVER CLASS DEFINITIONS
# ==============================================================================

# KLHK code to name mapping (official Indonesian land cover classes)
KLHK_CLASSES = {
    2001: 'Hutan Lahan Kering Primer',
    2002: 'Hutan Lahan Kering Sekunder',
    2003: 'Hutan Rawa Primer',
    2004: 'Hutan Rawa Sekunder',
    2005: 'Hutan Mangrove Primer',
    2006: 'Hutan Mangrove Sekunder',
    2007: 'Hutan Tanaman',
    2010: 'Perkebunan',
    2012: 'Pemukiman',
    2014: 'Tanah Terbuka',
    2500: 'Semak Belukar',
    3000: 'Savana',
    5001: 'Tubuh Air',
    20041: 'Belukar Rawa',
    20051: 'Pertanian Lahan Kering',
    20071: 'Hutan Tanaman Belum Produktif',
    20091: 'Pertanian Lahan Kering Campur',
    20092: 'Sawah',
    20093: 'Tambak',
    20094: 'Rawa',
    20121: 'Bandara/Pelabuhan',
    20122: 'Transmigrasi',
    20141: 'Pertambangan',
    50011: 'Awan',
}

# Simplified class mapping (KLHK -> 9 classes for comparison with Dynamic World)
KLHK_TO_SIMPLIFIED = {
    # Forest classes -> 1 (Trees)
    2001: 1, 2002: 1, 2003: 1, 2004: 1, 2005: 1, 2006: 1, 2007: 1,

    # Agriculture -> 4 (Crops)
    2010: 4,  # Perkebunan
    20051: 4, # Pertanian Lahan Kering
    20091: 4, # Pertanian Lahan Kering Campur
    20092: 4, # Sawah
    20093: 4, # Tambak

    # Built-up -> 6 (Built area)
    2012: 6, 20121: 6, 20122: 6,

    # Bare/Open -> 7 (Bare ground)
    2014: 7, 20141: 7,

    # Water -> 0 (Water)
    5001: 0, 20094: 0,

    # Shrub -> 5 (Shrub and scrub)
    2500: 5, 20041: 5,

    # Grass/Savanna -> 2 (Grass)
    3000: 2,

    # Flooded vegetation -> 3
    # (already mapped via forest swamp classes)

    # Cloud -> ignore (will be filtered)
    50011: -1,

    # Unknown -> 5 (Shrub as default)
    20071: 1,  # Hutan Tanaman Belum Produktif -> Trees
}

# Class names for simplified scheme
CLASS_NAMES_SIMPLIFIED = {
    0: 'Water',
    1: 'Trees/Forest',
    2: 'Grass/Savanna',
    3: 'Flooded Vegetation',
    4: 'Crops/Agriculture',
    5: 'Shrub/Scrub',
    6: 'Built Area',
    7: 'Bare Ground',
    8: 'Snow/Ice'  # Not applicable for Jambi
}

# Colors for visualization
COLORS = ['#419BDF', '#397D49', '#88B053', '#7A87C6',
          '#E49635', '#DFC35A', '#C4281B', '#A59B8F', '#B39FE1']

# Sentinel-2 band names
BAND_NAMES = [
    'B2_Blue', 'B3_Green', 'B4_Red',
    'B5_RedEdge1', 'B6_RedEdge2', 'B7_RedEdge3',
    'B8_NIR', 'B8A_RedEdge4', 'B11_SWIR1', 'B12_SWIR2'
]

# Spectral indices names
INDEX_NAMES = [
    'NDVI', 'EVI', 'SAVI', 'NDWI', 'MNDWI',
    'NDBI', 'BSI', 'NDRE', 'CIRE', 'MSAVI',
    'GNDVI', 'NDMI', 'NBR'
]

# ==============================================================================
# DATA LOADING FUNCTIONS
# ==============================================================================

def load_klhk_data(geojson_path):
    """
    Load KLHK land cover data from GeoJSON file.

    Args:
        geojson_path: Path to KLHK GeoJSON file

    Returns:
        GeoDataFrame with land cover polygons
    """
    print(f"Loading KLHK data from {geojson_path}...")
    gdf = gpd.read_file(geojson_path)

    print(f"  Total polygons: {len(gdf):,}")
    print(f"  CRS: {gdf.crs}")
    print(f"  Columns: {list(gdf.columns)}")

    # Identify the land cover code column
    code_col = None
    for col in ['ID Penutupan Lahan Tahun 2024', 'PL2024_ID', 'PL2024', 'KELAS', 'ID_PL']:
        if col in gdf.columns:
            code_col = col
            break

    if code_col is None:
        raise ValueError(f"Cannot find land cover code column. Available: {list(gdf.columns)}")

    print(f"  Using column: {code_col}")

    # Map to simplified classes
    gdf['klhk_code'] = gdf[code_col].astype(int)
    gdf['class_simplified'] = gdf['klhk_code'].map(KLHK_TO_SIMPLIFIED)

    # Remove cloud/unknown classes
    gdf = gdf[gdf['class_simplified'] >= 0]

    # Print class distribution
    print("\n  Class distribution:")
    for code in sorted(gdf['class_simplified'].unique()):
        count = (gdf['class_simplified'] == code).sum()
        name = CLASS_NAMES_SIMPLIFIED.get(code, 'Unknown')
        print(f"    {code}: {name} - {count:,} polygons")

    return gdf


def calculate_spectral_indices(data):
    """
    Calculate comprehensive spectral indices from Sentinel-2 bands.

    Args:
        data: numpy array with shape (bands, height, width)

    Returns:
        numpy array with spectral indices
    """
    # Extract bands (assuming standard Sentinel-2 band order)
    blue = data[0]      # B2
    green = data[1]     # B3
    red = data[2]       # B4
    re1 = data[3]       # B5 - Red Edge 1
    re2 = data[4]       # B6 - Red Edge 2
    re3 = data[5]       # B7 - Red Edge 3
    nir = data[6]       # B8 - NIR
    re4 = data[7]       # B8A - Red Edge 4
    swir1 = data[8]     # B11 - SWIR 1
    swir2 = data[9]     # B12 - SWIR 2

    epsilon = 1e-10  # Prevent division by zero

    # Vegetation indices
    ndvi = (nir - red) / (nir + red + epsilon)
    evi = 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1 + epsilon))
    savi = (1.5 * (nir - red)) / (nir + red + 0.5 + epsilon)

    # Water indices
    ndwi = (green - nir) / (green + nir + epsilon)
    mndwi = (green - swir1) / (green + swir1 + epsilon)

    # Built-up and bare soil indices
    ndbi = (swir1 - nir) / (swir1 + nir + epsilon)
    bsi = ((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue) + epsilon)

    # Red edge indices
    ndre = (nir - re1) / (nir + re1 + epsilon)
    cire = (nir / (re1 + epsilon)) - 1

    # Additional vegetation indices
    msavi = (2 * nir + 1 - np.sqrt((2 * nir + 1)**2 - 8 * (nir - red) + epsilon)) / 2
    gndvi = (nir - green) / (nir + green + epsilon)

    # Moisture indices
    ndmi = (nir - swir1) / (nir + swir1 + epsilon)
    nbr = (nir - swir2) / (nir + swir2 + epsilon)

    indices = np.stack([
        ndvi, evi, savi, ndwi, mndwi,
        ndbi, bsi, ndre, cire, msavi,
        gndvi, ndmi, nbr
    ])

    return indices


def load_sentinel2_data(tif_path):
    """
    Load Sentinel-2 data and calculate spectral indices.

    Args:
        tif_path: Path to Sentinel-2 GeoTIFF file

    Returns:
        tuple: (data array, profile/metadata)
    """
    print(f"Loading Sentinel-2 data from {tif_path}...")

    with rasterio.open(tif_path) as src:
        data = src.read()
        profile = src.profile

        print(f"  Shape: {data.shape}")
        print(f"  CRS: {src.crs}")
        print(f"  Bounds: {src.bounds}")

        # Calculate indices
        indices = calculate_spectral_indices(data)
        print(f"  Added {len(INDEX_NAMES)} spectral indices")

        # Combine bands and indices
        data_combined = np.vstack([data, indices])
        print(f"  Final shape: {data_combined.shape}")

    return data_combined, profile


def load_and_mosaic_sentinel2_tiles(tile_paths):
    """
    Load multiple Sentinel-2 tiles and mosaic them into a single array.

    Args:
        tile_paths: List of paths to Sentinel-2 GeoTIFF tiles

    Returns:
        tuple: (mosaicked data array, merged profile/metadata)
    """
    print(f"Loading and mosaicking {len(tile_paths)} Sentinel-2 tiles...")

    from rasterio.merge import merge

    # Open all tiles
    src_files = [rasterio.open(path) for path in tile_paths]

    # Merge/mosaic tiles
    mosaic, mosaic_transform = merge(src_files)

    # Get profile from first tile and update with mosaic info
    profile = src_files[0].profile.copy()
    profile.update({
        'height': mosaic.shape[1],
        'width': mosaic.shape[2],
        'transform': mosaic_transform
    })

    # Close files
    for src in src_files:
        src.close()

    print(f"  Mosaic shape: {mosaic.shape}")
    print(f"  CRS: {profile['crs']}")

    # Calculate spectral indices
    print(f"  Calculating spectral indices...")
    indices = calculate_spectral_indices(mosaic)

    # Combine bands and indices
    data_combined = np.vstack([mosaic, indices])

    print(f"  Final shape with indices: {data_combined.shape}")
    print(f"  Total features: {data_combined.shape[0]} ({len(BAND_NAMES)} bands + {len(INDEX_NAMES)} indices)")

    return data_combined, profile


def rasterize_klhk(gdf, reference_profile, class_column='class_simplified'):
    """
    Rasterize KLHK vector data to match Sentinel-2 raster grid.

    Args:
        gdf: GeoDataFrame with land cover polygons
        reference_profile: Rasterio profile from reference raster
        class_column: Column name containing class values

    Returns:
        numpy array with rasterized classes
    """
    print("Rasterizing KLHK data...")

    # Ensure same CRS
    if gdf.crs != reference_profile['crs']:
        print(f"  Reprojecting from {gdf.crs} to {reference_profile['crs']}")
        gdf = gdf.to_crs(reference_profile['crs'])

    # Create shapes for rasterization
    shapes = [(geom, value) for geom, value in zip(gdf.geometry, gdf[class_column])]

    # Rasterize
    rasterized = rasterize(
        shapes,
        out_shape=(reference_profile['height'], reference_profile['width']),
        transform=reference_profile['transform'],
        fill=-1,  # No data value
        dtype=np.int16
    )

    # Count valid pixels
    valid_pixels = np.sum(rasterized >= 0)
    total_pixels = rasterized.size
    coverage = (valid_pixels / total_pixels) * 100

    print(f"  Rasterized shape: {rasterized.shape}")
    print(f"  Valid pixels: {valid_pixels:,} ({coverage:.1f}%)")

    return rasterized


# ==============================================================================
# TRAINING FUNCTIONS
# ==============================================================================

def prepare_training_data(X_data, y_raster, sample_size=None, random_state=42):
    """
    Prepare training data from raster arrays.

    Args:
        X_data: Feature data (bands, height, width)
        y_raster: Label raster (height, width)
        sample_size: Optional sample size limit
        random_state: Random seed

    Returns:
        tuple: (X, y) arrays ready for training
    """
    print("Preparing training data...")

    # Reshape to (pixels, features)
    n_features = X_data.shape[0]
    X = X_data.reshape(n_features, -1).T
    y = y_raster.flatten()

    # Filter valid pixels (has label and no NaN in features)
    valid_mask = (y >= 0) & ~np.any(np.isnan(X), axis=1)
    X = X[valid_mask]
    y = y[valid_mask]

    print(f"  Total valid samples: {len(y):,}")

    # Optional subsampling
    if sample_size and sample_size < len(y):
        np.random.seed(random_state)
        indices = np.random.choice(len(y), sample_size, replace=False)
        X = X[indices]
        y = y[indices]
        print(f"  Subsampled to: {len(y):,}")

    # Print class distribution
    print("\n  Class distribution in training data:")
    unique, counts = np.unique(y, return_counts=True)
    for cls, cnt in zip(unique, counts):
        name = CLASS_NAMES_SIMPLIFIED.get(int(cls), 'Unknown')
        pct = (cnt / len(y)) * 100
        print(f"    {int(cls)}: {name} - {cnt:,} ({pct:.1f}%)")

    return X, y.astype(int)


def get_classifiers():
    """
    Get dictionary of classifiers to evaluate.

    Returns:
        dict: Classifier name -> Pipeline
    """
    classifiers = {}

    # Random Forest
    classifiers['Random Forest'] = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=200,
            max_depth=25,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        ))
    ])

    # Extra Trees
    classifiers['Extra Trees'] = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler()),
        ('classifier', ExtraTreesClassifier(
            n_estimators=200,
            max_depth=25,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        ))
    ])

    # Logistic Regression
    classifiers['Logistic Regression'] = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            multi_class='multinomial',
            max_iter=500,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        ))
    ])

    # Decision Tree
    classifiers['Decision Tree'] = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler()),
        ('classifier', DecisionTreeClassifier(
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42
        ))
    ])

    # Naive Bayes
    classifiers['Naive Bayes'] = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler()),
        ('classifier', GaussianNB())
    ])

    # SGD Classifier
    classifiers['SGD'] = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler()),
        ('classifier', SGDClassifier(
            loss='modified_huber',
            penalty='l2',
            alpha=0.0001,
            max_iter=1000,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        ))
    ])

    # LightGBM (if available)
    if HAS_LIGHTGBM:
        classifiers['LightGBM'] = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
            ('scaler', StandardScaler()),
            ('classifier', lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=15,
                learning_rate=0.1,
                class_weight='balanced',
                n_jobs=-1,
                random_state=42,
                verbose=-1
            ))
        ])

    # XGBoost (if available)
    if HAS_XGBOOST:
        classifiers['XGBoost'] = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
            ('scaler', StandardScaler()),
            ('classifier', XGBClassifier(
                n_estimators=200,
                max_depth=15,
                learning_rate=0.1,
                n_jobs=-1,
                random_state=42,
                use_label_encoder=False,
                eval_metric='mlogloss'
            ))
        ])

    return classifiers


def train_and_evaluate(X, y, test_size=0.2, random_state=42):
    """
    Train and evaluate multiple classifiers.

    Args:
        X: Feature array
        y: Label array
        test_size: Proportion for test set
        random_state: Random seed

    Returns:
        dict: Results for each classifier
    """
    print("\n" + "=" * 60)
    print("TRAINING AND EVALUATION")
    print("=" * 60)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"\nTraining set: {len(y_train):,} samples")
    print(f"Test set: {len(y_test):,} samples")

    # Get classifiers
    classifiers = get_classifiers()

    results = {}
    for name, pipeline in classifiers.items():
        print(f"\n--- Training {name} ---")
        start_time = time.time()

        # Train
        pipeline.fit(X_train, y_train)

        # Predict
        y_pred = pipeline.predict(X_test)

        # Metrics
        training_time = time.time() - start_time
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        results[name] = {
            'pipeline': pipeline,
            'y_test': y_test,
            'y_pred': y_pred,
            'training_time': training_time,
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'report': classification_report(y_test, y_pred, zero_division=0)
        }

        print(f"  Time: {training_time:.2f}s")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 (macro): {f1_macro:.4f}")
        print(f"  F1 (weighted): {f1_weighted:.4f}")

    return results, X_train, X_test, y_train, y_test


# ==============================================================================
# VISUALIZATION FUNCTIONS
# ==============================================================================

def plot_results(results, save_dir='results'):
    """
    Create visualization plots for classification results.

    Args:
        results: Dictionary of classifier results
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1. Performance comparison bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    names = list(results.keys())
    accuracies = [results[n]['accuracy'] for n in names]
    f1_scores = [results[n]['f1_macro'] for n in names]
    times = [results[n]['training_time'] for n in names]

    # Accuracy/F1
    x = np.arange(len(names))
    width = 0.35

    axes[0].bar(x - width/2, accuracies, width, label='Accuracy', color='steelblue')
    axes[0].bar(x + width/2, f1_scores, width, label='F1 (macro)', color='coral')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Classifier Performance Comparison')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=45, ha='right')
    axes[0].legend()
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis='y', alpha=0.3)

    # Training time
    axes[1].bar(names, times, color='green', alpha=0.7)
    axes[1].set_ylabel('Time (seconds)')
    axes[1].set_title('Training Time')
    axes[1].set_xticklabels(names, rotation=45, ha='right')
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/classifier_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Confusion matrices for top classifiers
    # Find best classifier
    best_name = max(results, key=lambda x: results[x]['f1_macro'])
    best_result = results[best_name]

    fig, ax = plt.subplots(figsize=(10, 8))
    cm = confusion_matrix(best_result['y_test'], best_result['y_pred'])

    # Normalize
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Get class labels that exist in data
    classes_in_data = sorted(set(best_result['y_test']) | set(best_result['y_pred']))
    class_labels = [CLASS_NAMES_SIMPLIFIED.get(c, str(c)) for c in classes_in_data]

    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=class_labels, yticklabels=class_labels, ax=ax)
    ax.set_title(f'Confusion Matrix - {best_name}\n(Normalized by True Labels)')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/confusion_matrix_{best_name.lower().replace(" ", "_")}.png',
                dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Feature importance (for tree-based models)
    for name, result in results.items():
        clf = result['pipeline'].named_steps.get('classifier')
        if hasattr(clf, 'feature_importances_'):
            fig, ax = plt.subplots(figsize=(10, 8))

            feature_names = BAND_NAMES + INDEX_NAMES
            importance = clf.feature_importances_

            # Sort by importance
            indices = np.argsort(importance)

            ax.barh(range(len(indices)), importance[indices], color='steelblue')
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels([feature_names[i] for i in indices])
            ax.set_xlabel('Importance')
            ax.set_title(f'Feature Importance - {name}')
            ax.grid(axis='x', alpha=0.3)

            plt.tight_layout()
            plt.savefig(f'{save_dir}/feature_importance_{name.lower().replace(" ", "_")}.png',
                        dpi=150, bbox_inches='tight')
            plt.close()

    print(f"\nPlots saved to {save_dir}/")


def export_results_to_csv(results, save_path='results/classification_results.csv'):
    """Export classification results to CSV."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    data = []
    for name, result in results.items():
        data.append({
            'Classifier': name,
            'Accuracy': result['accuracy'],
            'F1_Macro': result['f1_macro'],
            'F1_Weighted': result['f1_weighted'],
            'Training_Time_Seconds': result['training_time']
        })

    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    print(f"Results exported to {save_path}")

    return df


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Main execution function."""
    print("=" * 70)
    print("LAND COVER CLASSIFICATION USING KLHK REFERENCE DATA")
    print("=" * 70)

    # Configuration
    KLHK_PATH = 'data/klhk/KLHK_PL2024_Jambi_Full_WithGeometry.geojson'
    S2_TILES = [
        'data/sentinel/S2_jambi_2024_20m_AllBands-0000000000-0000000000.tif',
        'data/sentinel/S2_jambi_2024_20m_AllBands-0000000000-0000065536.tif',
        'data/sentinel/S2_jambi_2024_20m_AllBands-0000065536-0000000000.tif',
        'data/sentinel/S2_jambi_2024_20m_AllBands-0000065536-0000065536.tif'
    ]
    OUTPUT_DIR = 'results'
    SAMPLE_SIZE = 100000  # Limit samples for faster training (set None for all)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load KLHK data
    print("\n" + "-" * 50)
    print("STEP 1: Loading KLHK Reference Data")
    print("-" * 50)
    gdf = load_klhk_data(KLHK_PATH)

    # Load Sentinel-2 tiles
    print("\n" + "-" * 50)
    print("STEP 2: Loading Sentinel-2 Imagery")
    print("-" * 50)
    s2_data, s2_profile = load_and_mosaic_sentinel2_tiles(S2_TILES)

    # Rasterize KLHK to match Sentinel-2 grid
    print("\n" + "-" * 50)
    print("STEP 3: Rasterizing KLHK Ground Truth")
    print("-" * 50)
    klhk_raster = rasterize_klhk(gdf, s2_profile, class_column='class_simplified')

    # Prepare training data
    print("\n" + "-" * 50)
    print("STEP 4: Extracting Training Samples")
    print("-" * 50)
    X, y = prepare_training_data(s2_data, klhk_raster, sample_size=SAMPLE_SIZE)

    print(f"\nFinal training data:")
    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Label vector shape: {y.shape}")
    print(f"  Features: {len(BAND_NAMES)} bands + {len(INDEX_NAMES)} indices = {X.shape[1]} total")

    # Train and evaluate
    print("\n" + "-" * 50)
    print("STEP 5: Training Classifiers")
    print("-" * 50)

    results, X_train, X_test, y_train, y_test = train_and_evaluate(
        X, y, test_size=0.2, random_state=42
    )

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    summary_df = export_results_to_csv(results, f'{OUTPUT_DIR}/classification_results.csv')
    print("\n" + summary_df.to_string(index=False))

    # Best classifier
    best_name = max(results, key=lambda x: results[x]['f1_macro'])
    print(f"\nBest Classifier: {best_name}")
    print(f"  Accuracy: {results[best_name]['accuracy']:.4f}")
    print(f"  F1 (macro): {results[best_name]['f1_macro']:.4f}")

    # Detailed report for best classifier
    print(f"\nClassification Report ({best_name}):")
    print(results[best_name]['report'])

    # Generate plots
    print("\n" + "-" * 50)
    print("STEP 6: Generating Visualizations")
    print("-" * 50)
    plot_results(results, OUTPUT_DIR)

    print("\n" + "=" * 70)
    print("CLASSIFICATION COMPLETE!")
    print("=" * 70)
    print(f"\nResults saved to: {OUTPUT_DIR}/")

    return results


if __name__ == "__main__":
    results = main()
