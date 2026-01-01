"""
Configuration Settings
======================

Central configuration for satellite data download.
"""

# ==============================================================================
# MAIN CONFIGURATION
# ==============================================================================

CONFIG = {
    # ----- Earth Engine Project -----
    'project_id': 'ee-akhiyarwaladi',

    # ----- Study Area -----
    'region_name': 'jambi',
    'province_name': 'Jambi',
    'country': 'Indonesia',
    'boundary_source': 'GEOBOUNDARIES',  # Options: 'GAUL', 'GEOBOUNDARIES', 'BBOX'

    # ----- Time Period -----
    'start_date': '2024-01-01',
    'end_date': '2024-12-31',

    # ----- Cloud Filtering -----
    'max_cloud_percent': 20,
    'cloud_score_threshold': 0.60,  # For Cloud Score+ (0.5-0.65 recommended)

    # ----- Export Settings -----
    'scale': 20,  # Default resolution in meters
    'crs': 'EPSG:4326',
    'export_folder': 'GEE_Exports',
    'output_dir': 'data/satellite',

    # ----- Composite Method -----
    'composite_method': 'median',  # Options: 'median', 'mean', 'mosaic'
}

# ==============================================================================
# SATELLITE-SPECIFIC CONFIGURATIONS
# ==============================================================================

SENTINEL2_CONFIG = {
    'collection': 'COPERNICUS/S2_SR_HARMONIZED',
    'cloud_score_collection': 'GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED',
    'bands_10m': ['B2', 'B3', 'B4', 'B8'],
    'bands_20m': ['B5', 'B6', 'B7', 'B8A', 'B11', 'B12'],
    'all_bands': ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'],
    'scale_factor': 10000,  # DN to reflectance
    'native_resolution': 10,
}

LANDSAT_CONFIG = {
    'collection_l8': 'LANDSAT/LC08/C02/T1_L2',
    'collection_l9': 'LANDSAT/LC09/C02/T1_L2',
    'optical_bands': ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'],
    'thermal_bands': ['ST_B10'],
    'renamed_bands': ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2'],
    'scale_factor': 0.0000275,
    'offset': -0.2,
    'native_resolution': 30,
}

# ==============================================================================
# BOUNDARY CONFIGURATIONS
# ==============================================================================

BOUNDARY_SOURCES = {
    'GAUL': {
        'collection': 'FAO/GAUL/2015/level1',
        'country_field': 'ADM0_NAME',
        'province_field': 'ADM1_NAME',
        'year': 2015,
        'license': 'Non-commercial',
    },
    'GEOBOUNDARIES': {
        'collection': 'WM/geoLab/geoBoundaries/600/ADM1',
        'country_field': 'shapeGroup',
        'province_field': 'shapeName',
        'country_code': 'IDN',  # ISO3 for Indonesia
        'year': 2023,
        'license': 'CC BY 4.0',
    },
}

# Jambi bounding box (fallback)
JAMBI_BBOX = [102.5, -2.6, 104.6, -0.8]  # [west, south, east, north]

# ==============================================================================
# SPECTRAL INDICES DEFINITIONS
# ==============================================================================

INDICES_DEFINITIONS = {
    'NDVI': {
        'name': 'Normalized Difference Vegetation Index',
        'formula': '(NIR - RED) / (NIR + RED)',
        'bands': {'NIR': 'B8', 'RED': 'B4'},  # Sentinel-2
        'bands_landsat': {'NIR': 'NIR', 'RED': 'Red'},
    },
    'EVI': {
        'name': 'Enhanced Vegetation Index',
        'formula': '2.5 * (NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1)',
    },
    'NDWI': {
        'name': 'Normalized Difference Water Index',
        'formula': '(GREEN - NIR) / (GREEN + NIR)',
    },
    'NDMI': {
        'name': 'Normalized Difference Moisture Index',
        'formula': '(NIR - SWIR1) / (NIR + SWIR1)',
    },
    'MNDWI': {
        'name': 'Modified NDWI',
        'formula': '(GREEN - SWIR1) / (GREEN + SWIR1)',
    },
    'NDBI': {
        'name': 'Normalized Difference Built-up Index',
        'formula': '(SWIR1 - NIR) / (SWIR1 + NIR)',
    },
    'SAVI': {
        'name': 'Soil Adjusted Vegetation Index',
        'formula': '1.5 * (NIR - RED) / (NIR + RED + 0.5)',
    },
    'NBR': {
        'name': 'Normalized Burn Ratio',
        'formula': '(NIR - SWIR2) / (NIR + SWIR2)',
    },
}
