"""
Preprocessor Module
===================

Handles rasterization of KLHK data and preparation of training samples.
"""

import numpy as np
from rasterio.features import rasterize


def rasterize_klhk(gdf, reference_profile, class_column='class_simplified', verbose=True):
    """
    Rasterize KLHK vector data to match Sentinel-2 raster grid.

    Args:
        gdf: GeoDataFrame with land cover polygons
        reference_profile: Rasterio profile from reference raster
        class_column: Column name containing class values
        verbose: Print progress information

    Returns:
        numpy array with rasterized classes
    """
    if verbose:
        print("Rasterizing KLHK data...")

    # Ensure same CRS
    if gdf.crs != reference_profile['crs']:
        if verbose:
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

    # Calculate coverage
    valid_pixels = np.sum(rasterized >= 0)
    total_pixels = rasterized.size
    coverage = (valid_pixels / total_pixels) * 100

    if verbose:
        print(f"  Rasterized shape: {rasterized.shape}")
        print(f"  Valid pixels: {valid_pixels:,} ({coverage:.1f}%)")

    return rasterized


def prepare_training_data(feature_data, label_raster, sample_size=None, random_state=42, verbose=True):
    """
    Extract training samples from raster data.

    Args:
        feature_data: Feature data (bands/indices, height, width)
        label_raster: Label raster (height, width)
        sample_size: Optional sample size limit
        random_state: Random seed for sampling
        verbose: Print progress information

    Returns:
        tuple: (X, y) arrays ready for training
    """
    if verbose:
        print("Preparing training data...")

    # Reshape to (pixels, features)
    n_features = feature_data.shape[0]
    X = feature_data.reshape(n_features, -1).T
    y = label_raster.flatten()

    # Filter valid pixels
    valid_mask = (y >= 0) & ~np.any(np.isnan(X), axis=1) & ~np.any(np.isinf(X), axis=1)
    X = X[valid_mask]
    y = y[valid_mask]

    if verbose:
        print(f"  Total valid samples: {len(y):,}")

    # Optional subsampling
    if sample_size and sample_size < len(y):
        np.random.seed(random_state)
        indices = np.random.choice(len(y), sample_size, replace=False)
        X = X[indices]
        y = y[indices]
        if verbose:
            print(f"  Subsampled to: {len(y):,}")

    # Print class distribution
    if verbose:
        from .data_loader import CLASS_NAMES
        print("\n  Class distribution in training data:")
        unique, counts = np.unique(y, return_counts=True)
        for cls, cnt in zip(unique, counts):
            name = CLASS_NAMES.get(int(cls), 'Unknown')
            pct = (cnt / len(y)) * 100
            print(f"    {int(cls)}: {name} - {cnt:,} ({pct:.1f}%)")

    return X, y.astype(int)


def split_train_test(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and test sets with stratification.

    Args:
        X: Feature array
        y: Label array
        test_size: Proportion for test set
        random_state: Random seed

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test
