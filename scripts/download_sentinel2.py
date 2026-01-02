#!/usr/bin/env python3
"""
Download Sentinel-2 + Dynamic World Data using Earth Engine Python API
======================================================================

This script mirrors the functionality of g_earth_engine_improved.js
Downloads data directly from Google Earth Engine without needing the GEE Console.

Features:
- Sentinel-2 SR Harmonized imagery
- Cloud Score+ cloud masking (best practice 2024)
- Dynamic World land cover classification
- 8 spectral indices
- Multiple export options

Requirements:
    pip install earthengine-api

First-time setup:
    earthengine authenticate

Author: Land Cover Research Project
Date: December 2024
"""

import ee
import os
import sys
import time
import requests
from datetime import datetime

# Add project root to path for module imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from modules.cloud_removal import CloudRemovalConfig

# ==============================================================================
# CONFIGURATION - Same as JS version
# ==============================================================================

CONFIG = {
    # Study area
    'region_name': 'jambi',
    'province_name': 'Jambi',
    'boundary_source': 'GEOBOUNDARIES',  # Options: 'GAUL', 'GEOBOUNDARIES', 'BBOX' (GEOBOUNDARIES = 2023, GAUL = 2015)

    # Time period - DRY SEASON for Indonesia (clearest imagery!)
    'start_date': '2024-06-01',  # Dry season: June-September
    'end_date': '2024-09-30',

    # Cloud removal strategy (see modules/cloud_removal.py for options)
    # Options: 'current', 'pan_tropical', 'percentile_25', 'kalimantan', 'balanced', 'conservative'
    'cloud_removal_strategy': 'percentile_25',  # TESTED: 99.1% cloud-free!

    # Manual override (will be overridden by strategy if strategy is set)
    'max_cloud_percent': 40,  # Moderate threshold
    'cloud_score_threshold': 0.50,  # Balanced masking for dry season

    # Export settings
    'scale': 20,  # Default 20m (smaller file size, good enough for classification)
    'output_dir': 'data/sentinel',
    'export_folder': 'GEE_Exports',
    'crs': 'EPSG:4326',

    # Earth Engine Project
    'project_id': 'ee-akhiyarwaladi',
}

# Jambi bounding box (fallback)
JAMBI_BOUNDS = [102.5, -2.6, 104.6, -0.8]  # [west, south, east, north]

# Dynamic World class names and colors
DW_CLASSES = {
    0: {'name': 'Water', 'color': '#419BDF'},
    1: {'name': 'Trees', 'color': '#397D49'},
    2: {'name': 'Grass', 'color': '#88B053'},
    3: {'name': 'Flooded Vegetation', 'color': '#7A87C6'},
    4: {'name': 'Crops', 'color': '#E49635'},
    5: {'name': 'Shrub and Scrub', 'color': '#DFC35A'},
    6: {'name': 'Built', 'color': '#C4281B'},
    7: {'name': 'Bare', 'color': '#A59B8F'},
    8: {'name': 'Snow and Ice', 'color': '#B39FE1'},
}

# ==============================================================================
# EARTH ENGINE FUNCTIONS
# ==============================================================================

def initialize_ee(project_id=None):
    """Initialize Earth Engine with authentication."""
    if project_id is None:
        project_id = CONFIG['project_id']

    try:
        ee.Initialize(project=project_id)
        print(f"Earth Engine initialized successfully! (Project: {project_id})")
    except Exception as e:
        print(f"Earth Engine not authenticated. Starting authentication...")
        print("Browser will open for Google authentication.\n")

        # Authenticate - will open browser
        ee.Authenticate()

        # Initialize after authentication
        ee.Initialize(project=project_id)
        print(f"Earth Engine initialized successfully! (Project: {project_id})")


def get_boundary(source='GAUL', province_name='Jambi'):
    """
    Get province boundary from various sources.

    Args:
        source: 'GAUL', 'GEOBOUNDARIES', or 'BBOX'
        province_name: Name of province

    Returns:
        ee.Geometry: Province boundary
    """
    if source == 'GAUL':
        # FAO GAUL 2015 (recommended)
        print(f"Using FAO GAUL boundary for: {province_name}")
        collection = (ee.FeatureCollection('FAO/GAUL/2015/level1')
                     .filter(ee.Filter.eq('ADM0_NAME', 'Indonesia'))
                     .filter(ee.Filter.eq('ADM1_NAME', province_name)))
        return collection.geometry()

    elif source == 'GEOBOUNDARIES':
        # geoBoundaries v6.0
        print(f"Using geoBoundaries for: {province_name}")
        collection = (ee.FeatureCollection('WM/geoLab/geoBoundaries/600/ADM1')
                     .filter(ee.Filter.eq('shapeGroup', 'IDN'))
                     .filter(ee.Filter.eq('shapeName', province_name)))
        return collection.geometry()

    else:
        # Bounding box fallback
        print("WARNING: Using bounding box - not recommended for final analysis!")
        return ee.Geometry.Rectangle(JAMBI_BOUNDS)


def mask_clouds_csplus(image, cs_collection, threshold):
    """
    Mask clouds using Cloud Score+ (best method for 2024+).

    Args:
        image: ee.Image - Sentinel-2 image
        cs_collection: ee.ImageCollection - Cloud Score+ collection
        threshold: float - Cloud score threshold

    Returns:
        ee.Image: Cloud-masked image
    """
    # Get matching Cloud Score+ image
    cs_image = cs_collection.filter(
        ee.Filter.eq('system:index', image.get('system:index'))
    ).first()

    # Use cs_cdf band (cumulative distribution function - more robust)
    cs = cs_image.select('cs_cdf')

    # Apply threshold
    clear_mask = cs.gte(threshold)

    return (image
            .updateMask(clear_mask)
            .select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'])
            .divide(10000)
            .copyProperties(image, ['system:time_start']))


def mask_clouds_scl(image):
    """
    Mask clouds using SCL band (fallback method).

    Args:
        image: ee.Image - Sentinel-2 SR image

    Returns:
        ee.Image: Cloud-masked image
    """
    scl = image.select('SCL')

    # Keep only: vegetation (4), bare soil (5), water (6), unclassified (7)
    clear_mask = scl.gte(4).And(scl.lte(7))

    # Also use cloud probability
    cloud_prob = image.select('MSK_CLDPRB')
    prob_mask = cloud_prob.lt(40)

    final_mask = clear_mask.And(prob_mask)

    return (image
            .updateMask(final_mask)
            .select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'])
            .divide(10000)
            .copyProperties(image, ['system:time_start']))


def apply_cloud_removal_strategy():
    """
    Apply cloud removal strategy configuration to CONFIG.
    Updates CONFIG with strategy-specific parameters.
    """
    if 'cloud_removal_strategy' in CONFIG and CONFIG['cloud_removal_strategy']:
        strategy_name = CONFIG['cloud_removal_strategy']
        strategy_config = CloudRemovalConfig.get_strategy(strategy_name)

        # Update CONFIG with strategy parameters
        CONFIG['max_cloud_percent'] = strategy_config['max_cloud_percent']
        CONFIG['cloud_score_threshold'] = strategy_config['cloud_score_threshold']
        CONFIG['_composite_method'] = strategy_config['composite_method']
        CONFIG['_pre_filter_percent'] = strategy_config.get('pre_filter_percent')

        print(f"\n{'='*80}")
        print(f"CLOUD REMOVAL STRATEGY: {strategy_config['name']}")
        print(f"{'='*80}")
        print(f"  Description: {strategy_config['description']}")
        print(f"  Cloud Score+ Threshold: {strategy_config['cloud_score_threshold']}")
        print(f"  Max Cloud %: {strategy_config['max_cloud_percent']}")
        print(f"  Composite Method: {strategy_config['composite_method']}")
        if strategy_config.get('pre_filter_percent'):
            print(f"  Pre-filter: ≤{strategy_config['pre_filter_percent']}% cloudy images only")
        print(f"  Source: {strategy_config['source']}")
        print(f"{'='*80}\n")
    else:
        # Use manual configuration
        CONFIG['_composite_method'] = 'median'
        CONFIG['_pre_filter_percent'] = None


def create_composite_from_collection(collection, region):
    """
    Create composite using strategy-defined method.

    Args:
        collection: ee.ImageCollection - Filtered and cloud-masked collection
        region: ee.Geometry - Region to clip to

    Returns:
        ee.Image - Composite image
    """
    method = CONFIG.get('_composite_method', 'median')

    if method == 'median':
        composite = collection.median()
    elif method == 'percentile_25':
        composite = collection.reduce(ee.Reducer.percentile([25]))
        # Rename bands (percentile reducer adds _p25 suffix)
        original_bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
        new_bands = [b + '_p25' for b in original_bands]
        composite = composite.select(new_bands, original_bands)
    elif method == 'percentile_30':
        composite = collection.reduce(ee.Reducer.percentile([30]))
        # Rename bands
        original_bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
        new_bands = [b + '_p30' for b in original_bands]
        composite = composite.select(new_bands, original_bands)
    elif method == 'min':
        composite = collection.reduce(ee.Reducer.min())
        # Rename bands
        original_bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
        new_bands = [b + '_min' for b in original_bands]
        composite = composite.select(new_bands, original_bands)
    else:
        print(f"Warning: Unknown composite method '{method}', using median")
        composite = collection.median()

    return composite.clip(region)


def calculate_indices(composite):
    """
    Calculate spectral indices (same as JS version).

    Args:
        composite: ee.Image - Sentinel-2 composite

    Returns:
        ee.Image: Image with 8 spectral indices
    """
    # NDVI - Normalized Difference Vegetation Index
    ndvi = composite.normalizedDifference(['B8', 'B4']).rename('NDVI')

    # EVI - Enhanced Vegetation Index
    evi = composite.expression(
        '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
        {
            'NIR': composite.select('B8'),
            'RED': composite.select('B4'),
            'BLUE': composite.select('B2')
        }
    ).rename('EVI')

    # NDWI - Normalized Difference Water Index
    ndwi = composite.normalizedDifference(['B3', 'B8']).rename('NDWI')

    # NDMI - Normalized Difference Moisture Index
    ndmi = composite.normalizedDifference(['B8', 'B11']).rename('NDMI')

    # MNDWI - Modified NDWI
    mndwi = composite.normalizedDifference(['B3', 'B11']).rename('MNDWI')

    # NDBI - Normalized Difference Built-up Index
    ndbi = composite.normalizedDifference(['B11', 'B8']).rename('NDBI')

    # SAVI - Soil Adjusted Vegetation Index
    savi = composite.expression(
        '((NIR - RED) / (NIR + RED + 0.5)) * 1.5',
        {
            'NIR': composite.select('B8'),
            'RED': composite.select('B4')
        }
    ).rename('SAVI')

    # NBR - Normalized Burn Ratio
    nbr = composite.normalizedDifference(['B8', 'B12']).rename('NBR')

    return ee.Image.cat([ndvi, evi, ndwi, ndmi, mndwi, ndbi, savi, nbr])


def get_sentinel2_collection(region, start_date, end_date):
    """
    Get Sentinel-2 collection with Cloud Score+ masking.

    Args:
        region: ee.Geometry - Study area
        start_date: str - Start date
        end_date: str - End date

    Returns:
        ee.ImageCollection: Cloud-masked collection
    """
    print(f"\nLoading Sentinel-2 data: {start_date} to {end_date}")

    # Load S2 SR Harmonized
    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterDate(start_date, end_date)
          .filterBounds(region)
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CONFIG['max_cloud_percent'])))

    # Load Cloud Score+
    cs_plus = (ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
               .filterDate(start_date, end_date)
               .filterBounds(region))

    # Apply Cloud Score+ masking
    def apply_cs_mask(image):
        return mask_clouds_csplus(image, cs_plus, CONFIG['cloud_score_threshold'])

    s2_masked = s2.map(apply_cs_mask)

    count = s2.size().getInfo()
    print(f"  Total images found: {count}")

    return s2_masked


def get_dynamic_world(region, start_date, end_date):
    """
    Get Dynamic World classification.

    Args:
        region: ee.Geometry - Study area
        start_date: str - Start date
        end_date: str - End date

    Returns:
        tuple: (label composite, probability composite)
    """
    print(f"\nLoading Dynamic World data...")

    dw = (ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
          .filterDate(start_date, end_date)
          .filterBounds(region))

    # Mode composite for classification
    dw_label = dw.select('label').mode()

    # Apply smoothing to reduce salt-and-pepper noise
    dw_smoothed = dw_label.focal_mode(
        kernel=ee.Kernel.circle(radius=1),
        iterations=2
    )

    # Mean composite for probabilities
    prob_bands = ['water', 'trees', 'grass', 'flooded_vegetation',
                  'crops', 'shrub_and_scrub', 'built', 'bare', 'snow_and_ice']
    dw_prob = dw.select(prob_bands).mean()

    count = dw.size().getInfo()
    print(f"  Dynamic World images: {count}")

    return dw_smoothed, dw_prob


def export_to_drive(image, description, folder, region, scale, as_type='float'):
    """
    Export image to Google Drive.

    Args:
        image: ee.Image - Image to export
        description: str - Export task description
        folder: str - Google Drive folder
        region: ee.Geometry - Export region
        scale: int - Resolution in meters
        as_type: str - 'float', 'byte', or 'int16'

    Returns:
        ee.batch.Task
    """
    if as_type == 'float':
        image = image.toFloat()
    elif as_type == 'byte':
        image = image.toByte()
    elif as_type == 'int16':
        image = image.toInt16()

    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=folder,
        region=region,
        scale=scale,
        crs=CONFIG['crs'],
        maxPixels=1e13,
        fileFormat='GeoTIFF'
    )

    task.start()
    print(f"  Started: {description} ({scale}m)")

    return task


# ==============================================================================
# MAIN FUNCTIONS
# ==============================================================================

def download_full_dataset(include_dw=False, include_indices=False, include_qc=False):
    """
    Download dataset to Google Drive.

    Args:
        include_dw: bool - Include Dynamic World classification & probabilities
        include_indices: bool - Include spectral indices (can be calculated in Python)
        include_qc: bool - Include observation count QC layer

    Default: Only Sentinel-2 bands (indices calculated in classification script)
    """
    print("=" * 60)
    print("SENTINEL-2 DATA DOWNLOAD")
    print("=" * 60)

    # Apply cloud removal strategy
    apply_cloud_removal_strategy()

    # Initialize
    initialize_ee()

    # Get boundary
    region = get_boundary(
        source=CONFIG['boundary_source'],
        province_name=CONFIG['province_name']
    )

    # Calculate area
    area_km2 = region.area().divide(1e6).getInfo()
    print(f"  Area: {area_km2:,.0f} km²")

    # Get Sentinel-2 collection
    s2_collection = get_sentinel2_collection(
        region,
        CONFIG['start_date'],
        CONFIG['end_date']
    )

    # Create composite using strategy
    print("\nCreating Sentinel-2 composite...")
    s2_composite = create_composite_from_collection(s2_collection, region)

    # Export to Drive
    print("\n" + "-" * 40)
    print("STARTING EXPORTS TO GOOGLE DRIVE")
    print(f"Folder: {CONFIG['export_folder']}")
    print("-" * 40)

    year = CONFIG['start_date'][:4]
    name = CONFIG['region_name']
    folder = CONFIG['export_folder']

    tasks = []

    # 1. S2 ALL bands (ALWAYS EXPORTED)
    scale = CONFIG['scale']
    all_bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
    tasks.append(export_to_drive(
        s2_composite.select(all_bands),
        f'S2_{name}_{year}_{scale}m_AllBands',
        folder, region, scale, 'float'
    ))

    # 2. Dynamic World (OPTIONAL)
    if include_dw:
        print("\nLoading Dynamic World data...")
        dw_label, dw_prob = get_dynamic_world(
            region,
            CONFIG['start_date'],
            CONFIG['end_date']
        )
        dw_label = dw_label.clip(region)
        dw_prob = dw_prob.clip(region)

        tasks.append(export_to_drive(
            dw_label,
            f'DW_{name}_{year}_classification',
            folder, region, 10, 'byte'
        ))
        tasks.append(export_to_drive(
            dw_prob,
            f'DW_{name}_{year}_probabilities',
            folder, region, 10, 'float'
        ))

    # 3. Spectral Indices (OPTIONAL - can be calculated in Python)
    if include_indices:
        print("Calculating spectral indices...")
        indices = calculate_indices(s2_composite)
        tasks.append(export_to_drive(
            indices,
            f'Indices_{name}_{year}_all',
            folder, region, 10, 'float'
        ))

    # 4. Observation Count QC (OPTIONAL)
    if include_qc:
        obs_count = s2_collection.select('B4').count().rename('observation_count').clip(region)
        tasks.append(export_to_drive(
            obs_count,
            f'QC_{name}_{year}_obsCount',
            folder, region, 10, 'int16'
        ))

    print("\n" + "=" * 60)
    print("EXPORT TASKS STARTED")
    print("=" * 60)
    print(f"\nTotal tasks: {len(tasks)}")
    print(f"Check Google Drive folder: {folder}")
    print("Monitor progress: https://code.earthengine.google.com/tasks")

    return tasks


def download_sample(bounds=None, scale=20):
    """
    Download smaller sample area for testing.

    Args:
        bounds: list - [west, south, east, north] or None for default
        scale: int - Resolution in meters
    """
    print("=" * 60)
    print("SAMPLE DOWNLOAD (Testing)")
    print("=" * 60)

    # Apply cloud removal strategy
    apply_cloud_removal_strategy()

    initialize_ee()

    # Smaller sample area
    if bounds is None:
        bounds = [103.5, -1.8, 103.8, -1.5]  # ~30x30 km area

    region = ee.Geometry.Rectangle(bounds)
    print(f"Sample bounds: {bounds}")

    # Get data
    s2_collection = get_sentinel2_collection(
        region,
        CONFIG['start_date'],
        CONFIG['end_date']
    )

    # Create composite using strategy
    print("\nCreating composite...")
    s2_composite = create_composite_from_collection(s2_collection, region)
    indices = calculate_indices(s2_composite)

    # Get Dynamic World
    dw_label, dw_prob = get_dynamic_world(
        region,
        CONFIG['start_date'],
        CONFIG['end_date']
    )
    dw_label = dw_label.clip(region)

    # Combine S2 bands + indices + DW
    combined = s2_composite.addBands(indices).addBands(dw_label.rename('DW_label'))

    # Try direct download first
    print("\nAttempting direct download...")
    output_file = f"{CONFIG['output_dir']}/S2_sample_{CONFIG['start_date'][:4]}.tif"

    try:
        url = combined.getDownloadURL({
            'scale': scale,
            'crs': CONFIG['crs'],
            'region': region,
            'format': 'GEO_TIFF'
        })

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        response = requests.get(url, stream=True)
        with open(output_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        file_size = os.path.getsize(output_file) / (1024 * 1024)
        print(f"\nDownload complete: {output_file} ({file_size:.2f} MB)")
        return output_file

    except Exception as e:
        print(f"\nDirect download failed: {e}")
        print("Falling back to Drive export...")

        task = export_to_drive(
            combined.toFloat(),
            'S2_sample_combined',
            CONFIG['export_folder'],
            region,
            scale,
            'float'
        )
        return task


def check_export_status():
    """Check status of Earth Engine export tasks."""
    initialize_ee()

    tasks = ee.batch.Task.list()

    print("\n" + "=" * 60)
    print("EXPORT TASK STATUS")
    print("=" * 60)

    for task in tasks[:10]:
        status = task.status()
        state = status['state']
        desc = status['description']

        # Color coding for status
        if state == 'COMPLETED':
            symbol = '[OK]'
        elif state == 'RUNNING':
            symbol = '[..]'
        elif state == 'FAILED':
            symbol = '[XX]'
        else:
            symbol = '[--]'

        print(f"{symbol} {desc}: {state}")

        if 'error_message' in status:
            print(f"    Error: {status['error_message']}")


# ==============================================================================
# COMMAND LINE INTERFACE
# ==============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Download Sentinel-2 data from Earth Engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: Only Sentinel-2 bands (recommended)
  python download_sentinel2.py --mode full

  # Include Dynamic World for comparison
  python download_sentinel2.py --mode full --include-dw

  # Include everything
  python download_sentinel2.py --mode full --include-dw --include-indices --include-qc

  # Small sample for testing
  python download_sentinel2.py --mode sample

  # Check export status
  python download_sentinel2.py --mode status
        """
    )
    parser.add_argument(
        '--mode',
        choices=['full', 'sample', 'status'],
        default='sample',
        help='Download mode: full (entire Jambi), sample (small test), status (check tasks)'
    )
    parser.add_argument(
        '--scale',
        type=int,
        default=10,
        help='Resolution in meters (default: 10)'
    )
    parser.add_argument(
        '--include-dw',
        action='store_true',
        help='Include Dynamic World classification & probabilities'
    )
    parser.add_argument(
        '--include-indices',
        action='store_true',
        help='Include spectral indices (optional - can be calculated in Python)'
    )
    parser.add_argument(
        '--include-qc',
        action='store_true',
        help='Include observation count QC layer'
    )

    args = parser.parse_args()

    if args.mode == 'full':
        download_full_dataset(
            include_dw=args.include_dw,
            include_indices=args.include_indices,
            include_qc=args.include_qc
        )
    elif args.mode == 'sample':
        download_sample(scale=args.scale)
    elif args.mode == 'status':
        check_export_status()
