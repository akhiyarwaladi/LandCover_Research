"""
Feature Engineering Module
===========================

Calculates spectral indices and temporal change features
for multi-temporal deforestation detection.
"""

import numpy as np


def calculate_spectral_indices(sentinel2_data, verbose=True):
    """
    Calculate spectral indices from Sentinel-2 bands.

    Args:
        sentinel2_data: numpy array (bands, height, width)
                       Bands: [B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12]
        verbose: Print calculation info

    Returns:
        numpy array (13, height, width) of spectral indices
    """
    if verbose:
        print("Calculating spectral indices...")

    blue = sentinel2_data[0]
    green = sentinel2_data[1]
    red = sentinel2_data[2]
    re1 = sentinel2_data[3]
    re2 = sentinel2_data[4]
    re3 = sentinel2_data[5]
    nir = sentinel2_data[6]
    re4 = sentinel2_data[7]
    swir1 = sentinel2_data[8]
    swir2 = sentinel2_data[9]

    epsilon = 1e-10

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

    if verbose:
        print(f"  Calculated {indices.shape[0]} spectral indices")

    return indices


def combine_bands_and_indices(bands, indices):
    """
    Combine Sentinel-2 bands with spectral indices.

    Args:
        bands: (10, H, W) Sentinel-2 bands
        indices: (13, H, W) Spectral indices

    Returns:
        (23, H, W) Combined feature stack
    """
    return np.vstack([bands, indices])


def calculate_change_features(t1_features, t2_features, verbose=True):
    """
    Calculate temporal change features between two time periods.

    Computes difference (t2 - t1) for key spectral indices, which captures
    vegetation loss (negative dNDVI), burn signals (negative dNBR), etc.

    Args:
        t1_features: (23, H, W) features at time 1
        t2_features: (23, H, W) features at time 2
        verbose: Print calculation info

    Returns:
        (10, H, W) change features
    """
    if verbose:
        print("Calculating temporal change features...")

    # Index positions in the 23-feature stack (bands 0-9, indices 10-22)
    # Indices: NDVI=10, EVI=11, SAVI=12, NDWI=13, MNDWI=14,
    #          NDBI=15, BSI=16, NDRE=17, CIRE=18, MSAVI=19,
    #          GNDVI=20, NDMI=21, NBR=22
    change_indices = {
        'dNDVI': 10,   # Vegetation loss indicator
        'dEVI': 11,    # Enhanced vegetation change
        'dSAVI': 12,   # Soil-adjusted vegetation change
        'dNDWI': 13,   # Water change
        'dNDBI': 15,   # Built-up change
        'dBSI': 16,    # Bare soil change
        'dNDRE': 17,   # Red edge vegetation change
        'dMSAVI': 19,  # Modified soil-adjusted change
        'dNDMI': 21,   # Moisture change
        'dNBR': 22,    # Burn ratio change
    }

    change_features = []
    for name, idx in change_indices.items():
        diff = t2_features[idx] - t1_features[idx]
        change_features.append(diff)

    result = np.stack(change_features)

    if verbose:
        print(f"  Calculated {result.shape[0]} change features")
        print(f"  Features: {list(change_indices.keys())}")

    return result


def create_stacked_features(t1_features, t2_features, verbose=True):
    """
    Create stacked feature set for RF-based change detection.

    Stacks: T1 features (23) + T2 features (23) + Change features (10) = 56

    Args:
        t1_features: (23, H, W) features at time 1
        t2_features: (23, H, W) features at time 2
        verbose: Print info

    Returns:
        (56, H, W) stacked feature array
    """
    if verbose:
        print("Creating stacked features for change detection...")

    change = calculate_change_features(t1_features, t2_features, verbose=False)
    stacked = np.vstack([t1_features, t2_features, change])

    if verbose:
        print(f"  T1 features: {t1_features.shape[0]}")
        print(f"  T2 features: {t2_features.shape[0]}")
        print(f"  Change features: {change.shape[0]}")
        print(f"  Total stacked: {stacked.shape[0]}")

    return stacked


def get_index_names():
    """Get names of calculated spectral indices."""
    return [
        'NDVI', 'EVI', 'SAVI', 'NDWI', 'MNDWI',
        'NDBI', 'BSI', 'NDRE', 'CIRE', 'MSAVI',
        'GNDVI', 'NDMI', 'NBR'
    ]


def get_change_feature_names():
    """Get names of temporal change features."""
    return [
        'dNDVI', 'dEVI', 'dSAVI', 'dNDWI', 'dNDBI',
        'dBSI', 'dNDRE', 'dMSAVI', 'dNDMI', 'dNBR'
    ]


def get_all_feature_names():
    """Get names of all per-year features (bands + indices)."""
    from .data_loader import get_sentinel2_band_names
    return get_sentinel2_band_names() + get_index_names()


def get_stacked_feature_names():
    """Get names of all stacked features for RF change detection."""
    base_names = get_all_feature_names()
    t1_names = [f'{n}_T1' for n in base_names]
    t2_names = [f'{n}_T2' for n in base_names]
    change_names = get_change_feature_names()
    return t1_names + t2_names + change_names
