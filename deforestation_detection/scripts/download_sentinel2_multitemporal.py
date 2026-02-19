"""
Download Multi-Temporal Sentinel-2 Annual Composites via GEE Python API

Downloads 7 annual dry-season (June-October) median composites for 2018-2024.
Uses Cloud Score+ or SCL-based cloud masking.

Usage:
    python scripts/download_sentinel2_multitemporal.py

Output:
    data/sentinel/{year}/S2_jambi_{year}_20m_AllBands.tif (per year)

Prerequisites:
    - earthengine-api authenticated (earthengine authenticate)
    - Google Drive storage available for intermediate exports
"""

import os
import sys
import ee

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# Configuration
# ============================================================

YEARS = list(range(2018, 2025))
DRY_SEASON_START = '-06-01'
DRY_SEASON_END = '-11-01'
CLOUD_THRESHOLD = 0.60
SCALE = 20
CRS = 'EPSG:4326'
BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8A', 'B11', 'B12']

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'data', 'sentinel')
GEE_FOLDER = 'GEE_Deforestation_Jambi'
PROJECT_ID = 'ee-akhiyarwaladi'


def initialize_gee():
    """Initialize Google Earth Engine."""
    try:
        ee.Initialize(project=PROJECT_ID)
        print(f"GEE initialized successfully (Project: {PROJECT_ID})")
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=PROJECT_ID)
        print(f"GEE authenticated and initialized (Project: {PROJECT_ID})")


def get_jambi_boundary():
    """Get Jambi Province boundary from FAO GAUL."""
    jambi = ee.FeatureCollection('FAO/GAUL/2015/level1') \
        .filter(ee.Filter.eq('ADM1_NAME', 'Jambi'))
    return jambi.geometry()


def mask_clouds_scl(image):
    """Mask clouds using Scene Classification Layer."""
    scl = image.select('SCL')
    mask = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(11))
    return image.updateMask(mask)


def create_annual_composite(year, boundary):
    """
    Create dry-season median composite for a given year.

    Args:
        year: Integer year
        boundary: ee.Geometry for clipping

    Returns:
        ee.Image composite
    """
    start_date = f'{year}{DRY_SEASON_START}'
    end_date = f'{year}{DRY_SEASON_END}'

    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterBounds(boundary)
          .filterDate(start_date, end_date)
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50)))

    count = s2.size().getInfo()
    print(f"  Year {year}: {count} images found")

    # Apply cloud masking (SCL-based for all years for consistency)
    s2_masked = s2.map(mask_clouds_scl)

    # Median composite
    composite = (s2_masked
                 .select(BANDS)
                 .median()
                 .clip(boundary)
                 .toFloat())

    return composite


def export_composite(composite, year, boundary):
    """
    Export composite to Google Drive.

    Args:
        composite: ee.Image
        year: Integer year
        boundary: ee.Geometry
    """
    file_name = f'S2_jambi_{year}_20m_AllBands'

    task = ee.batch.Export.image.toDrive(
        image=composite,
        description=file_name,
        folder=GEE_FOLDER,
        fileNamePrefix=file_name,
        region=boundary,
        scale=SCALE,
        crs=CRS,
        maxPixels=1e13,
        fileFormat='GeoTIFF'
    )

    task.start()
    print(f"  Export started: {file_name}")
    return task


def main():
    """Download multi-temporal Sentinel-2 composites."""
    print("=" * 60)
    print("MULTI-TEMPORAL SENTINEL-2 DOWNLOAD")
    print("=" * 60)
    print(f"Years: {YEARS[0]}-{YEARS[-1]}")
    print(f"Season: June-October (dry season)")
    print(f"Resolution: {SCALE}m")
    print(f"Bands: {len(BANDS)}")
    print()

    # Create output directories
    for year in YEARS:
        os.makedirs(os.path.join(OUTPUT_DIR, str(year)), exist_ok=True)

    # Initialize GEE
    initialize_gee()
    boundary = get_jambi_boundary()

    # Create and export composites
    tasks = {}
    for year in YEARS:
        print(f"\nProcessing {year}...")
        composite = create_annual_composite(year, boundary)
        task = export_composite(composite, year, boundary)
        tasks[year] = task

    print("\n" + "=" * 60)
    print("ALL EXPORT TASKS STARTED")
    print("=" * 60)
    print(f"\nExports will appear in Google Drive: {GEE_FOLDER}/")
    print("After download, move files to data/sentinel/{year}/ directories")
    print("\nTo check task status, run:")
    print("  python -c \"import ee; ee.Initialize(); "
          "[print(t.status()) for t in ee.batch.Task.list()[:7]]\"")


if __name__ == '__main__':
    main()
