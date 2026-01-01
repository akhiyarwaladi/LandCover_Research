"""
Feature Engineering Module
===========================

Calculates spectral indices from Sentinel-2 bands.
"""

import numpy as np


def calculate_spectral_indices(sentinel2_data, verbose=True):
    """
    Calculate comprehensive spectral indices from Sentinel-2 bands.

    Args:
        sentinel2_data: numpy array with shape (bands, height, width)
                       Bands: [B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12]
        verbose: Print calculation info

    Returns:
        numpy array with spectral indices (indices, height, width)
    """
    if verbose:
        print("Calculating spectral indices...")

    # Extract bands
    blue = sentinel2_data[0]      # B2
    green = sentinel2_data[1]     # B3
    red = sentinel2_data[2]       # B4
    re1 = sentinel2_data[3]       # B5 - Red Edge 1
    re2 = sentinel2_data[4]       # B6 - Red Edge 2
    re3 = sentinel2_data[5]       # B7 - Red Edge 3
    nir = sentinel2_data[6]       # B8 - NIR
    re4 = sentinel2_data[7]       # B8A - Red Edge 4
    swir1 = sentinel2_data[8]     # B11 - SWIR 1
    swir2 = sentinel2_data[9]     # B12 - SWIR 2

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

    if verbose:
        print(f"  Calculated {indices.shape[0]} spectral indices")
        print(f"  Shape: {indices.shape}")

    return indices


def combine_bands_and_indices(bands, indices):
    """
    Combine Sentinel-2 bands with calculated indices.

    Args:
        bands: Sentinel-2 bands array
        indices: Spectral indices array

    Returns:
        Combined features array
    """
    combined = np.vstack([bands, indices])
    return combined


def get_index_names():
    """Get names of calculated spectral indices."""
    return [
        'NDVI', 'EVI', 'SAVI', 'NDWI', 'MNDWI',
        'NDBI', 'BSI', 'NDRE', 'CIRE', 'MSAVI',
        'GNDVI', 'NDMI', 'NBR'
    ]


def get_all_feature_names():
    """Get names of all features (bands + indices)."""
    from .data_loader import get_sentinel2_band_names

    band_names = get_sentinel2_band_names()
    index_names = get_index_names()

    return band_names + index_names
