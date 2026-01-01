#!/usr/bin/env python3
"""
Satellite Data Download - Main Script
======================================

Centralized script for downloading satellite imagery from Google Earth Engine.

Supported satellites:
- Sentinel-2 (10-20m resolution)
- Landsat 8/9 (30m resolution)

Usage:
    # Download Sentinel-2 (default)
    python download_satellite.py

    # Download Landsat
    python download_satellite.py --satellite landsat

    # Custom options
    python download_satellite.py --satellite sentinel2 --scale 10 --year 2024

    # Check export status
    python download_satellite.py --status

Author: Land Cover Research Project
Date: January 2025
"""

import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from satellite import (
    CONFIG,
    initialize_ee,
    get_boundary,
    get_sentinel2_collection,
    create_sentinel2_composite,
    get_landsat_collection,
    create_landsat_composite,
    calculate_indices,
    export_to_drive,
    check_export_status,
)


def download_satellite_data(
    satellite='sentinel2',
    scale=None,
    year=None,
    include_indices=False,
    region_name=None,
    boundary_source=None
):
    """
    Main function to download satellite data.

    Args:
        satellite: str - 'sentinel2' or 'landsat'
        scale: int - Resolution in meters (default: 20 for S2, 30 for Landsat)
        year: int - Year to download
        include_indices: bool - Include spectral indices in export
        region_name: str - Region name for filename
        boundary_source: str - 'GAUL', 'GEOBOUNDARIES', or 'BBOX'

    Returns:
        list: Export tasks
    """
    # Set defaults
    if scale is None:
        scale = 20 if satellite == 'sentinel2' else 30
    if year is None:
        year = int(CONFIG['start_date'][:4])
    if region_name is None:
        region_name = CONFIG['region_name']
    if boundary_source is None:
        boundary_source = CONFIG['boundary_source']

    # Header
    print("=" * 60)
    print(f"SATELLITE DATA DOWNLOAD")
    print(f"Satellite: {satellite.upper()}")
    print(f"Resolution: {scale}m")
    print(f"Year: {year}")
    print("=" * 60)

    # Initialize Earth Engine
    if not initialize_ee():
        print("[ERROR] Failed to initialize Earth Engine")
        return []

    # Get boundary
    region = get_boundary(source=boundary_source)
    area_km2 = region.area().divide(1e6).getInfo()
    print(f"[OK] Region: {region_name} ({area_km2:,.0f} km²)")

    # Set date range
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    # Get collection and create composite
    if satellite == 'sentinel2':
        collection = get_sentinel2_collection(region, start_date, end_date)
        composite = create_sentinel2_composite(collection, region)
        bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
    elif satellite == 'landsat':
        collection = get_landsat_collection(region, start_date, end_date)
        composite = create_landsat_composite(collection, region)
        bands = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']
    else:
        print(f"[ERROR] Unknown satellite: {satellite}")
        return []

    # Export
    print("\n" + "-" * 40)
    print("STARTING EXPORTS")
    print("-" * 40)

    tasks = []
    sat_code = 'S2' if satellite == 'sentinel2' else 'L89'

    # 1. Export bands
    filename = f"{sat_code}_{region_name}_{year}_{scale}m_AllBands"
    tasks.append(export_to_drive(
        composite.select(bands),
        filename,
        region,
        scale=scale,
        as_type='float'
    ))

    # 2. Export indices (optional)
    if include_indices:
        indices = calculate_indices(composite, satellite)
        filename_idx = f"{sat_code}_{region_name}_{year}_{scale}m_Indices"
        tasks.append(export_to_drive(
            indices,
            filename_idx,
            region,
            scale=scale,
            as_type='float'
        ))

    # Summary
    print("\n" + "=" * 60)
    print("EXPORT SUMMARY")
    print("=" * 60)
    print(f"Total tasks: {len(tasks)}")
    print(f"Output folder: Google Drive/{CONFIG['export_folder']}/")
    print(f"Monitor: https://code.earthengine.google.com/tasks")

    # Estimate file size
    pixels = (area_km2 * 1e6) / (scale * scale)
    bands_count = len(bands) + (8 if include_indices else 0)
    size_mb = (pixels * bands_count * 4) / (1024 * 1024) * 0.3  # ~30% compression
    print(f"\nEstimated file size: ~{size_mb:,.0f} MB")

    return tasks


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Download satellite data from Google Earth Engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download Sentinel-2 at 20m (default)
  python download_satellite.py

  # Download Landsat at 30m
  python download_satellite.py --satellite landsat

  # Download Sentinel-2 at 10m with indices
  python download_satellite.py --satellite sentinel2 --scale 10 --indices

  # Custom year
  python download_satellite.py --year 2023

  # Check export status
  python download_satellite.py --status

  # Show satellite info
  python download_satellite.py --info
        """
    )

    parser.add_argument(
        '--satellite', '-s',
        choices=['sentinel2', 'landsat'],
        default='sentinel2',
        help='Satellite to download (default: sentinel2)'
    )
    parser.add_argument(
        '--scale', '-r',
        type=int,
        help='Resolution in meters (default: 20 for S2, 30 for Landsat)'
    )
    parser.add_argument(
        '--year', '-y',
        type=int,
        help='Year to download (default: from config)'
    )
    parser.add_argument(
        '--indices', '-i',
        action='store_true',
        help='Include spectral indices (NDVI, EVI, etc.)'
    )
    parser.add_argument(
        '--boundary', '-b',
        choices=['GAUL', 'GEOBOUNDARIES', 'BBOX'],
        help='Boundary source (default: GEOBOUNDARIES)'
    )
    parser.add_argument(
        '--status',
        action='store_true',
        help='Check export task status'
    )
    parser.add_argument(
        '--info',
        action='store_true',
        help='Show satellite information'
    )

    args = parser.parse_args()

    # Check status mode
    if args.status:
        initialize_ee()
        check_export_status()
        return

    # Info mode
    if args.info:
        print("\n=== SATELLITE INFORMATION ===\n")
        print("SENTINEL-2:")
        print("  Resolution: 10m (RGBNIR), 20m (RedEdge, SWIR)")
        print("  Bands: B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12")
        print("  Revisit: 5 days")
        print("  Provider: ESA/Copernicus")
        print()
        print("LANDSAT 8/9:")
        print("  Resolution: 30m (optical)")
        print("  Bands: Blue, Green, Red, NIR, SWIR1, SWIR2")
        print("  Revisit: 16 days (8 days combined)")
        print("  Provider: USGS")
        return

    # Download mode
    download_satellite_data(
        satellite=args.satellite,
        scale=args.scale,
        year=args.year,
        include_indices=args.indices,
        boundary_source=args.boundary
    )


if __name__ == "__main__":
    main()
