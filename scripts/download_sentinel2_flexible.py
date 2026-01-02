#!/usr/bin/env python3
"""
Flexible Sentinel-2 Download - City or Province, 10m or 20m
============================================================

Clean preset system for easy switching between:
- Region: Province or City
- Resolution: 10m (high detail) or 20m (more bands)
- Strategy: Any cloud removal strategy

Usage:
    # Jambi City at 10m (high detail, fast)
    python download_sentinel2_flexible.py --preset city_10m

    # Jambi Province at 20m (all bands)
    python download_sentinel2_flexible.py --preset province_20m

    # Custom strategy
    python download_sentinel2_flexible.py --preset city_10m --strategy kalimantan

Author: Claude Sonnet 4.5
Date: 2026-01-02
"""

import ee
import os
import sys
import argparse
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from modules.cloud_removal import CloudRemovalConfig
from modules.naming_standards import create_sentinel_name

# ============================================================================
# PRESET CONFIGURATIONS - Easy to Switch!
# ============================================================================

CONFIG_PRESETS = {
    # High detail for city - 10m resolution (4 bands)
    'city_10m': {
        'name': 'Jambi City - 10m Resolution (High Detail)',
        'region_type': 'city',
        'region_name': 'Kota Jambi',
        'scale': 10,
        'bands': ['B2', 'B3', 'B4', 'B8'],  # Native 10m bands only
        'band_names': ['Blue', 'Green', 'Red', 'NIR'],
        'output_suffix': 'city_10m',
        'export_folder': 'GEE_Exports_City',
        'expected_size': '~40 MB',
        'use_case': 'Urban analysis, building detection, high detail mapping'
    },

    # All bands for city - 10m resolution (10 bands, 6 resampled)
    'city_10m_allbands': {
        'name': 'Jambi City - 10m Resolution (All Bands)',
        'region_type': 'city',
        'region_name': 'Kota Jambi',
        'scale': 10,
        'bands': ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'],
        'band_names': ['Blue', 'Green', 'Red', 'RedEdge1', 'RedEdge2', 'RedEdge3',
                      'NIR', 'RedEdge4', 'SWIR1', 'SWIR2'],
        'output_suffix': 'city_10m_allbands',
        'export_folder': 'GEE_Exports_City',
        'expected_size': '~55 MB',
        'use_case': 'City classification with all bands at highest spatial detail'
    },

    # All bands for city - 20m resolution (10 bands)
    'city_20m': {
        'name': 'Jambi City - 20m Resolution (All Bands)',
        'region_type': 'city',
        'region_name': 'Kota Jambi',
        'scale': 20,
        'bands': ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'],
        'band_names': ['Blue', 'Green', 'Red', 'RedEdge1', 'RedEdge2', 'RedEdge3',
                      'NIR', 'RedEdge4', 'SWIR1', 'SWIR2'],
        'output_suffix': 'city_20m',
        'export_folder': 'GEE_Exports_City',
        'expected_size': '~10 MB',
        'use_case': 'Land cover classification with full spectral bands'
    },

    # Province at 20m - standard (10 bands)
    'province_20m': {
        'name': 'Jambi Province - 20m Resolution (All Bands)',
        'region_type': 'province',
        'region_name': 'Jambi',
        'scale': 20,
        'bands': ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'],
        'band_names': ['Blue', 'Green', 'Red', 'RedEdge1', 'RedEdge2', 'RedEdge3',
                      'NIR', 'RedEdge4', 'SWIR1', 'SWIR2'],
        'output_suffix': 'province_20m',
        'export_folder': 'GEE_Exports',
        'expected_size': '~2.7 GB',
        'use_case': 'Province-wide land cover classification'
    },

    # Province at 10m - high detail RGB/NIR only (4 bands)
    'province_10m': {
        'name': 'Jambi Province - 10m Resolution (RGB+NIR only)',
        'region_type': 'province',
        'region_name': 'Jambi',
        'scale': 10,
        'bands': ['B2', 'B3', 'B4', 'B8'],
        'band_names': ['Blue', 'Green', 'Red', 'NIR'],
        'output_suffix': 'province_10m',
        'export_folder': 'GEE_Exports',
        'expected_size': '~4 GB',
        'use_case': 'High resolution RGB visualization (large file!)'
    },
}

# Default settings
DEFAULT_CONFIG = {
    'start_date': '2024-06-01',  # Dry season
    'end_date': '2024-09-30',
    'cloud_removal_strategy': 'percentile_25',  # Tested best
    'crs': 'EPSG:4326',
    'project_id': 'ee-akhiyarwaladi',
}

# ============================================================================
# BOUNDARY FUNCTIONS
# ============================================================================

def get_province_boundary(province_name):
    """Get province boundary from GeoBoundaries."""

    # GeoBoundaries Indonesia ADM1 (provinces)
    url = "https://www.geoboundaries.org/api/current/gbOpen/IDN/ADM1/"

    import requests
    response = requests.get(url)
    data = response.json()

    import geopandas as gpd
    gdf = gpd.read_file(data['gjDownloadURL'])

    # Find province
    province = gdf[gdf['shapeName'].str.contains(province_name, case=False, na=False)]

    if len(province) == 0:
        raise ValueError(f"Province '{province_name}' not found")

    # Convert to Earth Engine geometry
    import json
    geojson = json.loads(province.to_json())
    coords = geojson['features'][0]['geometry']['coordinates']

    return ee.Geometry.Polygon(coords)

def get_city_boundary(city_name):
    """Get city boundary from GeoBoundaries."""

    # GeoBoundaries Indonesia ADM2 (cities/regencies)
    url = "https://www.geoboundaries.org/api/current/gbOpen/IDN/ADM2/"

    import requests
    response = requests.get(url)
    data = response.json()

    import geopandas as gpd
    gdf = gpd.read_file(data['gjDownloadURL'])

    # Find city
    city = gdf[gdf['shapeName'].str.contains(city_name, case=False, na=False)]

    if len(city) == 0:
        raise ValueError(f"City '{city_name}' not found")

    # Convert to Earth Engine geometry
    import json
    geojson = json.loads(city.to_json())
    coords = geojson['features'][0]['geometry']['coordinates']

    return ee.Geometry.Polygon(coords)

def get_boundary(region_type, region_name):
    """Get boundary based on type (province or city)."""

    print(f"\nLoading boundary: {region_name} ({region_type})")

    if region_type == 'province':
        return get_province_boundary(region_name)
    elif region_type == 'city':
        return get_city_boundary(region_name)
    else:
        raise ValueError(f"Unknown region_type: {region_type}")

# ============================================================================
# CLOUD REMOVAL & COMPOSITE
# ============================================================================

def apply_cloud_removal_strategy(strategy_name):
    """Apply cloud removal strategy and update DEFAULT_CONFIG."""

    strategy_config = CloudRemovalConfig.get_strategy(strategy_name)

    # Update config
    DEFAULT_CONFIG['cloud_score_threshold'] = strategy_config['cloud_score_threshold']
    DEFAULT_CONFIG['max_cloud_percent'] = strategy_config['max_cloud_percent']
    DEFAULT_CONFIG['_composite_method'] = strategy_config['composite_method']
    DEFAULT_CONFIG['_pre_filter_percent'] = strategy_config.get('pre_filter_percent')

    print(f"\n{'='*80}")
    print(f"CLOUD REMOVAL STRATEGY: {strategy_config['name']}")
    print(f"{'='*80}")
    print(f"  Description: {strategy_config['description']}")
    print(f"  Cloud Score+: {strategy_config['cloud_score_threshold']}")
    print(f"  Max Cloud %: {strategy_config['max_cloud_percent']}")
    print(f"  Composite: {strategy_config['composite_method']}")
    if strategy_config.get('pre_filter_percent'):
        print(f"  Pre-filter: ≤{strategy_config['pre_filter_percent']}%")
    print(f"  Source: {strategy_config['source']}")
    print(f"{'='*80}\n")

def mask_clouds_csplus(image, cs_collection, threshold, bands):
    """Mask clouds using Cloud Score+."""

    cs_image = cs_collection.filter(
        ee.Filter.eq('system:index', image.get('system:index'))
    ).first()

    cs = cs_image.select('cs_cdf')
    clear_mask = cs.gte(threshold)

    return (image
            .updateMask(clear_mask)
            .select(bands)
            .divide(10000)
            .copyProperties(image, ['system:time_start']))

def create_composite_from_collection(collection, region):
    """Create composite using strategy-defined method."""

    method = DEFAULT_CONFIG.get('_composite_method', 'median')

    if method == 'median':
        composite = collection.median()
    elif method == 'percentile_25':
        composite = collection.reduce(ee.Reducer.percentile([25]))
        # Rename bands (remove _p25 suffix)
        old_names = composite.bandNames()
        new_names = old_names.map(lambda name: ee.String(name).replace('_p25', ''))
        composite = composite.rename(new_names)
    elif method == 'percentile_30':
        composite = collection.reduce(ee.Reducer.percentile([30]))
        old_names = composite.bandNames()
        new_names = old_names.map(lambda name: ee.String(name).replace('_p30', ''))
        composite = composite.rename(new_names)
    else:
        print(f"Warning: Unknown method '{method}', using median")
        composite = collection.median()

    return composite.clip(region)

# ============================================================================
# MAIN DOWNLOAD FUNCTION
# ============================================================================

def download_sentinel2(preset_name, strategy_name=None):
    """
    Download Sentinel-2 with specified preset.

    Args:
        preset_name: Name of preset ('city_10m', 'province_20m', etc.)
        strategy_name: Cloud removal strategy (default: percentile_25)
    """

    # Get preset config
    if preset_name not in CONFIG_PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}\n"
                        f"Available: {list(CONFIG_PRESETS.keys())}")

    preset = CONFIG_PRESETS[preset_name]

    # Use default strategy if not specified
    if strategy_name is None:
        strategy_name = DEFAULT_CONFIG['cloud_removal_strategy']

    # Print configuration
    print("\n" + "="*80)
    print("SENTINEL-2 FLEXIBLE DOWNLOAD")
    print("="*80)
    print(f"\nPreset: {preset_name}")
    print(f"  Name: {preset['name']}")
    print(f"  Region: {preset['region_name']} ({preset['region_type']})")
    print(f"  Resolution: {preset['scale']}m")
    print(f"  Bands: {len(preset['bands'])} ({', '.join(preset['band_names'])})")
    print(f"  Expected Size: {preset['expected_size']}")
    print(f"  Use Case: {preset['use_case']}")

    # Apply cloud removal strategy
    apply_cloud_removal_strategy(strategy_name)

    # Initialize Earth Engine
    try:
        ee.Initialize(project=DEFAULT_CONFIG['project_id'])
        print(f"✅ Earth Engine initialized")
    except:
        print("Authenticating Earth Engine...")
        ee.Authenticate()
        ee.Initialize(project=DEFAULT_CONFIG['project_id'])

    # Get boundary
    region = get_boundary(preset['region_type'], preset['region_name'])
    area_km2 = region.area().divide(1e6).getInfo()
    print(f"  Area: {area_km2:,.1f} km²")

    # Load Sentinel-2
    print(f"\nLoading Sentinel-2: {DEFAULT_CONFIG['start_date']} to {DEFAULT_CONFIG['end_date']}")

    max_cloud = DEFAULT_CONFIG.get('_pre_filter_percent') or DEFAULT_CONFIG['max_cloud_percent']
    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterDate(DEFAULT_CONFIG['start_date'], DEFAULT_CONFIG['end_date'])
          .filterBounds(region)
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud)))

    count = s2.size().getInfo()
    print(f"  Images found: {count}")

    # Apply cloud masking
    cs_plus = (ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
               .filterDate(DEFAULT_CONFIG['start_date'], DEFAULT_CONFIG['end_date'])
               .filterBounds(region))

    threshold = DEFAULT_CONFIG['cloud_score_threshold']
    bands = preset['bands']

    s2_masked = s2.map(lambda img: mask_clouds_csplus(img, cs_plus, threshold, bands))

    # Create composite
    print("\nCreating composite...")
    composite = create_composite_from_collection(s2_masked, region)

    # Export
    print("\n" + "-"*80)
    print("STARTING EXPORT TO GOOGLE DRIVE")
    print("-"*80)

    # Use standardized naming
    timeframe = f"{DEFAULT_CONFIG['start_date'][:4]}dry"  # 2024dry
    description = create_sentinel_name(
        region=preset['region_type'],
        resolution=preset['scale'],
        timeframe=timeframe,
        strategy=strategy_name
    )

    task = ee.batch.Export.image.toDrive(
        image=composite,
        description=description,
        folder=preset['export_folder'],
        region=region,
        scale=preset['scale'],
        crs=DEFAULT_CONFIG['crs'],
        maxPixels=1e13,
        fileFormat='GeoTIFF'
    )

    task.start()

    print(f"✅ Export started: {description}")
    print(f"   Folder: {preset['export_folder']}")
    print(f"   Resolution: {preset['scale']}m")
    print(f"   Bands: {len(bands)}")
    print(f"   Expected size: {preset['expected_size']}")

    print("\n" + "="*80)
    print("EXPORT TASK STARTED")
    print("="*80)
    print(f"\nCheck progress:")
    print(f"  1. Google Drive: {preset['export_folder']}/")
    print(f"  2. GEE Console: https://code.earthengine.google.com/tasks")
    print(f"  3. Run: python scripts/check_task_status.py")

    print(f"\nEstimated time:")
    if area_km2 < 500:
        print(f"  Processing: ~5-10 minutes (small area)")
    else:
        print(f"  Processing: ~20-30 minutes (large area)")

    print("\n" + "="*80)

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Main entry point with argument parsing."""

    parser = argparse.ArgumentParser(
        description='Flexible Sentinel-2 Download - City or Province, 10m or 20m',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Jambi City at 10m (recommended for city)
  python download_sentinel2_flexible.py --preset city_10m

  # Jambi City at 20m (all bands)
  python download_sentinel2_flexible.py --preset city_20m

  # Full Province at 20m (standard)
  python download_sentinel2_flexible.py --preset province_20m

  # Custom strategy
  python download_sentinel2_flexible.py --preset city_10m --strategy kalimantan

Available Presets:
  city_10m           - Jambi City, 10m resolution, 4 bands (~40 MB)
  city_10m_allbands  - Jambi City, 10m resolution, 10 bands (~55 MB)
  city_20m           - Jambi City, 20m resolution, 10 bands (~10 MB)
  province_20m       - Jambi Province, 20m, 10 bands (~2.7 GB)
  province_10m       - Jambi Province, 10m, 4 bands (~4 GB)

Available Strategies:
  percentile_25 - Best for cloud removal (default, 99.1% tested)
  kalimantan    - Indonesia proven (strict filtering)
  balanced      - Compromise approach
  pan_tropical  - Standard for tropics
  current       - Baseline method
  conservative  - Maximum data retention
        """
    )

    parser.add_argument(
        '--preset',
        type=str,
        default='city_10m',
        choices=list(CONFIG_PRESETS.keys()),
        help='Download preset (default: city_10m)'
    )

    parser.add_argument(
        '--strategy',
        type=str,
        default=None,
        help='Cloud removal strategy (default: percentile_25)'
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='List available presets and exit'
    )

    args = parser.parse_args()

    # List presets if requested
    if args.list:
        print("\n" + "="*80)
        print("AVAILABLE PRESETS")
        print("="*80)
        for name, preset in CONFIG_PRESETS.items():
            print(f"\n{name}:")
            print(f"  Name: {preset['name']}")
            print(f"  Region: {preset['region_name']} ({preset['region_type']})")
            print(f"  Resolution: {preset['scale']}m")
            print(f"  Bands: {len(preset['bands'])}")
            print(f"  Size: {preset['expected_size']}")
            print(f"  Use: {preset['use_case']}")
        print("\n" + "="*80)
        return

    # Run download
    download_sentinel2(args.preset, args.strategy)

if __name__ == '__main__':
    main()
