"""
Download Hansen Global Forest Change Data via GEE Python API

Downloads treecover2000, lossyear, and gain layers for Jambi Province
from Hansen et al. (2013) Global Forest Change dataset v1.12.

Usage:
    python scripts/download_hansen_gfc.py

Output:
    data/hansen/Hansen_treecover2000_Jambi.tif
    data/hansen/Hansen_lossyear_Jambi.tif
    data/hansen/Hansen_gain_Jambi.tif

Reference:
    Hansen, M.C., et al. (2013). High-Resolution Global Maps of
    21st-Century Forest Cover Change. Science, 342(6160), 850-853.
"""

import os
import sys
import ee

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Configuration
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'data', 'hansen')
GEE_FOLDER = 'GEE_Deforestation_Jambi'
SCALE = 30  # Native resolution
CRS = 'EPSG:4326'
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
    """Get Jambi Province boundary."""
    jambi = ee.FeatureCollection('FAO/GAUL/2015/level1') \
        .filter(ee.Filter.eq('ADM1_NAME', 'Jambi'))
    return jambi.geometry()


def main():
    """Download Hansen GFC data for Jambi Province."""
    print("=" * 60)
    print("HANSEN GLOBAL FOREST CHANGE DOWNLOAD")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    initialize_gee()
    boundary = get_jambi_boundary()

    # Load Hansen GFC
    gfc = ee.Image('UMD/hansen/global_forest_change_2024_v1_12')

    # Extract layers
    layers = {
        'treecover2000': gfc.select('treecover2000'),
        'lossyear': gfc.select('lossyear'),
        'gain': gfc.select('gain'),
    }

    # Compute statistics
    print("\nComputing forest statistics...")

    treecover = layers['treecover2000'].clip(boundary)
    forest_mask = treecover.gte(30)
    forest_area = forest_mask.multiply(ee.Image.pixelArea()).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=boundary,
        scale=30,
        maxPixels=1e13
    )
    forest_m2 = forest_area.get('treecover2000').getInfo()
    print(f"  Forest area in 2000 (>30% cover): {forest_m2 / 1e6:,.0f} km2")

    # Loss statistics for study period
    print("\n  Annual forest loss (study period):")
    lossyear = layers['lossyear'].clip(boundary)
    for yr_val in range(18, 25):
        year = 2000 + yr_val
        year_loss = lossyear.eq(yr_val)
        loss_area = year_loss.multiply(ee.Image.pixelArea()).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=boundary,
            scale=30,
            maxPixels=1e13
        )
        loss_m2 = loss_area.get('lossyear').getInfo()
        print(f"    {year}: {loss_m2 / 1e6:,.1f} km2 ({loss_m2 / 1e4:,.0f} ha)")

    # Export all layers
    print("\nStarting exports...")
    tasks = {}

    for name, layer in layers.items():
        clipped = layer.clip(boundary).toFloat()
        file_name = f'Hansen_{name}_Jambi'

        task = ee.batch.Export.image.toDrive(
            image=clipped,
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
        tasks[name] = task
        print(f"  Export started: {file_name}")

    print("\n" + "=" * 60)
    print("ALL EXPORT TASKS STARTED")
    print("=" * 60)
    print(f"\nExports will appear in Google Drive: {GEE_FOLDER}/")
    print("After download, move files to data/hansen/ directory")
    print(f"\nExpected files:")
    for name in layers:
        print(f"  data/hansen/Hansen_{name}_Jambi.tif")


if __name__ == '__main__':
    main()
