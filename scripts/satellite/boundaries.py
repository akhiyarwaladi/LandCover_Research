"""
Boundary/Region Handling
========================

Functions to get administrative boundaries from various sources.
"""

import ee
from .config import CONFIG, BOUNDARY_SOURCES, JAMBI_BBOX


def get_boundary(source=None, province_name=None, country=None):
    """
    Get administrative boundary geometry.

    Args:
        source: str - 'GAUL', 'GEOBOUNDARIES', or 'BBOX'
        province_name: str - Province name to filter
        country: str - Country name or ISO code

    Returns:
        ee.Geometry: Province boundary
    """
    if source is None:
        source = CONFIG['boundary_source']
    if province_name is None:
        province_name = CONFIG['province_name']
    if country is None:
        country = CONFIG['country']

    if source == 'GAUL':
        return _get_gaul_boundary(province_name, country)
    elif source == 'GEOBOUNDARIES':
        return _get_geoboundaries(province_name)
    elif source == 'BBOX':
        return _get_bbox()
    else:
        print(f"[WARNING] Unknown source '{source}', using BBOX fallback")
        return _get_bbox()


def _get_gaul_boundary(province_name, country):
    """Get boundary from FAO GAUL 2015."""
    config = BOUNDARY_SOURCES['GAUL']

    print(f"[..] Loading FAO GAUL 2015 boundary for: {province_name}")

    collection = (ee.FeatureCollection(config['collection'])
                  .filter(ee.Filter.eq(config['country_field'], country))
                  .filter(ee.Filter.eq(config['province_field'], province_name)))

    count = collection.size().getInfo()
    if count == 0:
        print(f"[WARNING] No boundary found for {province_name}, using BBOX")
        return _get_bbox()

    geometry = collection.geometry()
    area = geometry.area().divide(1e6).getInfo()
    print(f"[OK] GAUL boundary loaded: {area:,.0f} km² (year: {config['year']})")

    return geometry


def _get_geoboundaries(province_name):
    """Get boundary from geoBoundaries v6.0 (2023)."""
    config = BOUNDARY_SOURCES['GEOBOUNDARIES']

    print(f"[..] Loading geoBoundaries 2023 for: {province_name}")

    collection = (ee.FeatureCollection(config['collection'])
                  .filter(ee.Filter.eq(config['country_field'], config['country_code']))
                  .filter(ee.Filter.eq(config['province_field'], province_name)))

    count = collection.size().getInfo()
    if count == 0:
        print(f"[WARNING] No boundary found for {province_name}, using BBOX")
        return _get_bbox()

    geometry = collection.geometry()
    area = geometry.area().divide(1e6).getInfo()
    print(f"[OK] geoBoundaries loaded: {area:,.0f} km² (year: {config['year']})")

    return geometry


def _get_bbox():
    """Get bounding box fallback."""
    print(f"[WARNING] Using bounding box - not recommended for final analysis!")
    print(f"          Bounds: {JAMBI_BBOX}")
    return ee.Geometry.Rectangle(JAMBI_BBOX)


def get_boundary_info(geometry):
    """
    Get information about a geometry.

    Args:
        geometry: ee.Geometry

    Returns:
        dict: Area in km², bounds, etc.
    """
    area_km2 = geometry.area().divide(1e6).getInfo()
    bounds = geometry.bounds().getInfo()

    return {
        'area_km2': area_km2,
        'bounds': bounds,
    }
