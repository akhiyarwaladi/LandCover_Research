"""
Spectral Indices Calculation
============================

Calculate various spectral indices from satellite imagery.
"""

import ee


def calculate_indices(image, satellite='sentinel2'):
    """
    Calculate all spectral indices for an image.

    Args:
        image: ee.Image - Satellite composite
        satellite: str - 'sentinel2' or 'landsat'

    Returns:
        ee.Image: Image with all indices as bands
    """
    print(f"[..] Calculating spectral indices for {satellite}...")

    if satellite == 'sentinel2':
        indices = _calculate_indices_s2(image)
    elif satellite == 'landsat':
        indices = _calculate_indices_landsat(image)
    else:
        raise ValueError(f"Unknown satellite: {satellite}")

    print(f"[OK] Calculated 8 spectral indices")
    return indices


def _calculate_indices_s2(image):
    """
    Calculate indices for Sentinel-2 imagery.

    Band mapping:
        B2=Blue, B3=Green, B4=Red, B8=NIR, B11=SWIR1, B12=SWIR2
    """
    # NDVI - Normalized Difference Vegetation Index
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')

    # EVI - Enhanced Vegetation Index
    evi = image.expression(
        '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
        {
            'NIR': image.select('B8'),
            'RED': image.select('B4'),
            'BLUE': image.select('B2')
        }
    ).rename('EVI')

    # NDWI - Normalized Difference Water Index
    ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')

    # NDMI - Normalized Difference Moisture Index
    ndmi = image.normalizedDifference(['B8', 'B11']).rename('NDMI')

    # MNDWI - Modified NDWI
    mndwi = image.normalizedDifference(['B3', 'B11']).rename('MNDWI')

    # NDBI - Normalized Difference Built-up Index
    ndbi = image.normalizedDifference(['B11', 'B8']).rename('NDBI')

    # SAVI - Soil Adjusted Vegetation Index
    savi = image.expression(
        '((NIR - RED) / (NIR + RED + 0.5)) * 1.5',
        {
            'NIR': image.select('B8'),
            'RED': image.select('B4')
        }
    ).rename('SAVI')

    # NBR - Normalized Burn Ratio
    nbr = image.normalizedDifference(['B8', 'B12']).rename('NBR')

    return ee.Image.cat([ndvi, evi, ndwi, ndmi, mndwi, ndbi, savi, nbr])


def _calculate_indices_landsat(image):
    """
    Calculate indices for Landsat imagery.

    Band mapping (renamed):
        Blue, Green, Red, NIR, SWIR1, SWIR2
    """
    # NDVI
    ndvi = image.normalizedDifference(['NIR', 'Red']).rename('NDVI')

    # EVI
    evi = image.expression(
        '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
        {
            'NIR': image.select('NIR'),
            'RED': image.select('Red'),
            'BLUE': image.select('Blue')
        }
    ).rename('EVI')

    # NDWI
    ndwi = image.normalizedDifference(['Green', 'NIR']).rename('NDWI')

    # NDMI
    ndmi = image.normalizedDifference(['NIR', 'SWIR1']).rename('NDMI')

    # MNDWI
    mndwi = image.normalizedDifference(['Green', 'SWIR1']).rename('MNDWI')

    # NDBI
    ndbi = image.normalizedDifference(['SWIR1', 'NIR']).rename('NDBI')

    # SAVI
    savi = image.expression(
        '((NIR - RED) / (NIR + RED + 0.5)) * 1.5',
        {
            'NIR': image.select('NIR'),
            'RED': image.select('Red')
        }
    ).rename('SAVI')

    # NBR
    nbr = image.normalizedDifference(['NIR', 'SWIR2']).rename('NBR')

    return ee.Image.cat([ndvi, evi, ndwi, ndmi, mndwi, ndbi, savi, nbr])


def get_indices_names():
    """Get list of index names."""
    return ['NDVI', 'EVI', 'NDWI', 'NDMI', 'MNDWI', 'NDBI', 'SAVI', 'NBR']


def calculate_single_index(image, index_name, satellite='sentinel2'):
    """
    Calculate a single spectral index.

    Args:
        image: ee.Image
        index_name: str - Name of index (NDVI, EVI, etc.)
        satellite: str - 'sentinel2' or 'landsat'

    Returns:
        ee.Image: Single band with index values
    """
    if satellite == 'sentinel2':
        band_map = {
            'BLUE': 'B2', 'GREEN': 'B3', 'RED': 'B4',
            'NIR': 'B8', 'SWIR1': 'B11', 'SWIR2': 'B12'
        }
    else:
        band_map = {
            'BLUE': 'Blue', 'GREEN': 'Green', 'RED': 'Red',
            'NIR': 'NIR', 'SWIR1': 'SWIR1', 'SWIR2': 'SWIR2'
        }

    if index_name == 'NDVI':
        return image.normalizedDifference([band_map['NIR'], band_map['RED']]).rename('NDVI')
    elif index_name == 'NDWI':
        return image.normalizedDifference([band_map['GREEN'], band_map['NIR']]).rename('NDWI')
    elif index_name == 'NDMI':
        return image.normalizedDifference([band_map['NIR'], band_map['SWIR1']]).rename('NDMI')
    elif index_name == 'NDBI':
        return image.normalizedDifference([band_map['SWIR1'], band_map['NIR']]).rename('NDBI')
    elif index_name == 'NBR':
        return image.normalizedDifference([band_map['NIR'], band_map['SWIR2']]).rename('NBR')
    else:
        raise ValueError(f"Unknown index: {index_name}")
