"""
Satellite Data Download Module
==============================

Modular package for downloading satellite imagery from Google Earth Engine.

Supported satellites:
- Sentinel-2 (10-20m resolution)
- Landsat 8/9 (30m resolution)

Usage:
    from satellite import download
    download('sentinel2', region='Jambi', year=2024, scale=20)
"""

from .config import CONFIG
from .auth import initialize_ee
from .boundaries import get_boundary
from .sentinel2 import get_sentinel2_collection, create_sentinel2_composite
from .landsat import get_landsat_collection, create_landsat_composite
from .indices import calculate_indices
from .export import export_to_drive, check_export_status

__version__ = '1.0.0'
__all__ = [
    'CONFIG',
    'initialize_ee',
    'get_boundary',
    'get_sentinel2_collection',
    'create_sentinel2_composite',
    'get_landsat_collection',
    'create_landsat_composite',
    'calculate_indices',
    'export_to_drive',
    'check_export_status',
]
