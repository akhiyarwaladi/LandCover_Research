"""
Landsat Data Processing
=======================

Functions for loading and processing Landsat 8/9 imagery.
"""

import ee
from .config import CONFIG, LANDSAT_CONFIG


def get_landsat_collection(region, start_date=None, end_date=None, sensors='both'):
    """
    Load Landsat 8/9 Collection 2 Level-2 imagery.

    Args:
        region: ee.Geometry - Study area
        start_date: str - Start date (YYYY-MM-DD)
        end_date: str - End date (YYYY-MM-DD)
        sensors: str - 'L8', 'L9', or 'both'

    Returns:
        ee.ImageCollection: Cloud-masked collection
    """
    if start_date is None:
        start_date = CONFIG['start_date']
    if end_date is None:
        end_date = CONFIG['end_date']

    print(f"\n[..] Loading Landsat data: {start_date} to {end_date}")

    collections = []

    # Landsat 8
    if sensors in ['L8', 'both']:
        l8 = (ee.ImageCollection(LANDSAT_CONFIG['collection_l8'])
              .filterDate(start_date, end_date)
              .filterBounds(region)
              .filter(ee.Filter.lt('CLOUD_COVER', CONFIG['max_cloud_percent']))
              .map(_mask_clouds_landsat)
              .map(_scale_landsat))
        collections.append(l8)
        print(f"     Landsat 8 images: {l8.size().getInfo()}")

    # Landsat 9
    if sensors in ['L9', 'both']:
        l9 = (ee.ImageCollection(LANDSAT_CONFIG['collection_l9'])
              .filterDate(start_date, end_date)
              .filterBounds(region)
              .filter(ee.Filter.lt('CLOUD_COVER', CONFIG['max_cloud_percent']))
              .map(_mask_clouds_landsat)
              .map(_scale_landsat))
        collections.append(l9)
        print(f"     Landsat 9 images: {l9.size().getInfo()}")

    # Merge collections
    if len(collections) == 2:
        merged = collections[0].merge(collections[1])
    else:
        merged = collections[0]

    total = merged.size().getInfo()
    print(f"[OK] Total Landsat images: {total}")

    return merged


def _mask_clouds_landsat(image):
    """
    Mask clouds using QA_PIXEL band for Landsat Collection 2.

    Args:
        image: ee.Image - Landsat image

    Returns:
        ee.Image: Cloud-masked image
    """
    # QA_PIXEL band bit flags
    # Bit 3: Cloud
    # Bit 4: Cloud Shadow
    # Bit 5: Snow

    qa = image.select('QA_PIXEL')

    # Create mask for clear pixels
    cloud_bit = 1 << 3
    shadow_bit = 1 << 4
    snow_bit = 1 << 5

    mask = (qa.bitwiseAnd(cloud_bit).eq(0)
            .And(qa.bitwiseAnd(shadow_bit).eq(0))
            .And(qa.bitwiseAnd(snow_bit).eq(0)))

    return image.updateMask(mask)


def _scale_landsat(image):
    """
    Apply scale factors and rename bands for Landsat Collection 2 Level-2.

    Args:
        image: ee.Image - Landsat image

    Returns:
        ee.Image: Scaled and renamed image
    """
    # Scale optical bands
    optical = (image.select(LANDSAT_CONFIG['optical_bands'])
               .multiply(LANDSAT_CONFIG['scale_factor'])
               .add(LANDSAT_CONFIG['offset']))

    # Rename bands to common names
    optical = optical.rename(LANDSAT_CONFIG['renamed_bands'])

    return optical.copyProperties(image, ['system:time_start'])


def create_landsat_composite(collection, region, method=None):
    """
    Create composite from Landsat collection.

    Args:
        collection: ee.ImageCollection - Input collection
        region: ee.Geometry - Clip region
        method: str - 'median', 'mean', or 'mosaic'

    Returns:
        ee.Image: Composite image
    """
    if method is None:
        method = CONFIG['composite_method']

    print(f"[..] Creating Landsat {method} composite...")

    if method == 'median':
        composite = collection.median()
    elif method == 'mean':
        composite = collection.mean()
    elif method == 'mosaic':
        composite = collection.mosaic()
    else:
        print(f"[WARNING] Unknown method '{method}', using median")
        composite = collection.median()

    composite = composite.clip(region)
    print(f"[OK] Landsat composite created ({method})")

    return composite


def get_landsat_bands():
    """Get list of Landsat band names (renamed)."""
    return LANDSAT_CONFIG['renamed_bands']


def get_landsat_info():
    """Get Landsat dataset information."""
    return {
        'name': 'Landsat 8/9 Collection 2 Level-2',
        'collection_l8': LANDSAT_CONFIG['collection_l8'],
        'collection_l9': LANDSAT_CONFIG['collection_l9'],
        'native_resolution': LANDSAT_CONFIG['native_resolution'],
        'bands': LANDSAT_CONFIG['renamed_bands'],
        'provider': 'USGS',
        'temporal_coverage': 'L8: 2013-present, L9: 2021-present',
    }
