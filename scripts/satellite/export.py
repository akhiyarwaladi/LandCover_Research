"""
Export Functions
================

Functions for exporting images to Google Drive.
"""

import ee
from .config import CONFIG


def export_to_drive(image, description, region, scale=None, folder=None, as_type='float'):
    """
    Export image to Google Drive.

    Args:
        image: ee.Image - Image to export
        description: str - Export task description (filename)
        region: ee.Geometry - Export region
        scale: int - Resolution in meters
        folder: str - Google Drive folder name
        as_type: str - 'float', 'byte', or 'int16'

    Returns:
        ee.batch.Task: Export task
    """
    if scale is None:
        scale = CONFIG['scale']
    if folder is None:
        folder = CONFIG['export_folder']

    # Convert data type
    if as_type == 'float':
        image = image.toFloat()
    elif as_type == 'byte':
        image = image.toByte()
    elif as_type == 'int16':
        image = image.toInt16()

    # Create export task
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
    print(f"[EXPORT] {description} ({scale}m) -> Drive/{folder}/")

    return task


def export_to_asset(image, description, asset_id, region, scale=None):
    """
    Export image to Earth Engine Asset.

    Args:
        image: ee.Image - Image to export
        description: str - Export task description
        asset_id: str - Full asset path
        region: ee.Geometry - Export region
        scale: int - Resolution in meters

    Returns:
        ee.batch.Task: Export task
    """
    if scale is None:
        scale = CONFIG['scale']

    task = ee.batch.Export.image.toAsset(
        image=image.toFloat(),
        description=description,
        assetId=asset_id,
        region=region,
        scale=scale,
        crs=CONFIG['crs'],
        maxPixels=1e13
    )

    task.start()
    print(f"[EXPORT] {description} ({scale}m) -> Asset: {asset_id}")

    return task


def check_export_status(limit=10):
    """
    Check status of Earth Engine export tasks.

    Args:
        limit: int - Number of recent tasks to show

    Returns:
        list: Task status information
    """
    tasks = ee.batch.Task.list()

    print("\n" + "=" * 60)
    print("EXPORT TASK STATUS")
    print("=" * 60)

    results = []
    for task in tasks[:limit]:
        status = task.status()
        state = status['state']
        desc = status['description']

        # Status symbol
        symbols = {
            'COMPLETED': '[OK]',
            'RUNNING': '[..]',
            'FAILED': '[XX]',
            'CANCELLED': '[--]',
            'READY': '[  ]',
        }
        symbol = symbols.get(state, '[??]')

        print(f"{symbol} {desc}: {state}")

        if 'error_message' in status:
            print(f"     Error: {status['error_message']}")

        results.append({
            'description': desc,
            'state': state,
            'error': status.get('error_message')
        })

    return results


def cancel_task(task_id):
    """
    Cancel a running export task.

    Args:
        task_id: str - Task ID to cancel
    """
    tasks = ee.batch.Task.list()
    for task in tasks:
        if task.id == task_id:
            task.cancel()
            print(f"[CANCELLED] Task {task_id}")
            return True

    print(f"[ERROR] Task {task_id} not found")
    return False


def cancel_all_running():
    """Cancel all running export tasks."""
    tasks = ee.batch.Task.list()
    cancelled = 0

    for task in tasks:
        if task.status()['state'] == 'RUNNING':
            task.cancel()
            cancelled += 1
            print(f"[CANCELLED] {task.status()['description']}")

    print(f"\nTotal cancelled: {cancelled} tasks")
    return cancelled
