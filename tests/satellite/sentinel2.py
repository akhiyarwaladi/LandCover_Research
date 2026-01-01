"""
Sentinel-2 Data Processing
==========================

Functions for loading and processing Sentinel-2 imagery.
"""

import ee
from .config import CONFIG, SENTINEL2_CONFIG


def get_sentinel2_collection(region, start_date=None, end_date=None):
    """
    Load Sentinel-2 Surface Reflectance collection with Cloud Score+ masking.

    Args:
        region: ee.Geometry - Study area
        start_date: str - Start date (YYYY-MM-DD)
        end_date: str - End date (YYYY-MM-DD)

    Returns:
        ee.ImageCollection: Cloud-masked collection
    """
    if start_date is None:
        start_date = CONFIG['start_date']
    if end_date is None:
        end_date = CONFIG['end_date']

    print(f"\n[..] Loading Sentinel-2 SR: {start_date} to {end_date}")

    # Load S2 SR Harmonized
    s2 = (ee.ImageCollection(SENTINEL2_CONFIG['collection'])
          .filterDate(start_date, end_date)
          .filterBounds(region)
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CONFIG['max_cloud_percent'])))

    # Load Cloud Score+
    cs_plus = (ee.ImageCollection(SENTINEL2_CONFIG['cloud_score_collection'])
               .filterDate(start_date, end_date)
               .filterBounds(region))

    # Apply Cloud Score+ masking
    def apply_cs_mask(image):
        return _mask_clouds_csplus(image, cs_plus)

    s2_masked = s2.map(apply_cs_mask)

    count = s2.size().getInfo()
    print(f"[OK] Sentinel-2 images found: {count}")

    return s2_masked


def _mask_clouds_csplus(image, cs_collection):
    """
    Mask clouds using Cloud Score+ (best method for 2024+).

    Args:
        image: ee.Image - Sentinel-2 image
        cs_collection: ee.ImageCollection - Cloud Score+ collection

    Returns:
        ee.Image: Cloud-masked image
    """
    # Get matching Cloud Score+ image
    cs_image = cs_collection.filter(
        ee.Filter.eq('system:index', image.get('system:index'))
    ).first()

    # Use cs_cdf band
    cs = cs_image.select('cs_cdf')

    # Apply threshold
    clear_mask = cs.gte(CONFIG['cloud_score_threshold'])

    # Select bands and scale to reflectance
    return (image
            .updateMask(clear_mask)
            .select(SENTINEL2_CONFIG['all_bands'])
            .divide(SENTINEL2_CONFIG['scale_factor'])
            .copyProperties(image, ['system:time_start']))


def _mask_clouds_scl(image):
    """
    Mask clouds using SCL band (fallback method).

    Args:
        image: ee.Image - Sentinel-2 SR image

    Returns:
        ee.Image: Cloud-masked image
    """
    scl = image.select('SCL')

    # Keep: vegetation (4), bare soil (5), water (6), unclassified (7)
    clear_mask = scl.gte(4).And(scl.lte(7))

    # Cloud probability
    cloud_prob = image.select('MSK_CLDPRB')
    prob_mask = cloud_prob.lt(40)

    final_mask = clear_mask.And(prob_mask)

    return (image
            .updateMask(final_mask)
            .select(SENTINEL2_CONFIG['all_bands'])
            .divide(SENTINEL2_CONFIG['scale_factor'])
            .copyProperties(image, ['system:time_start']))


def create_sentinel2_composite(collection, region, method=None):
    """
    Create composite from Sentinel-2 collection.

    Args:
        collection: ee.ImageCollection - Input collection
        region: ee.Geometry - Clip region
        method: str - 'median', 'mean', or 'mosaic'

    Returns:
        ee.Image: Composite image
    """
    if method is None:
        method = CONFIG['composite_method']

    print(f"[..] Creating Sentinel-2 {method} composite...")

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
    print(f"[OK] Sentinel-2 composite created ({method})")

    return composite


def get_sentinel2_bands():
    """Get list of Sentinel-2 band names."""
    return SENTINEL2_CONFIG['all_bands']


def get_sentinel2_info():
    """Get Sentinel-2 dataset information."""
    return {
        'name': 'Sentinel-2 Surface Reflectance Harmonized',
        'collection': SENTINEL2_CONFIG['collection'],
        'native_resolution': SENTINEL2_CONFIG['native_resolution'],
        'bands_10m': SENTINEL2_CONFIG['bands_10m'],
        'bands_20m': SENTINEL2_CONFIG['bands_20m'],
        'provider': 'ESA/Copernicus',
        'temporal_coverage': '2015-present',
    }
