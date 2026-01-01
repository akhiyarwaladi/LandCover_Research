"""
Earth Engine Authentication
===========================

Handle Google Earth Engine authentication and initialization.
"""

import ee
from .config import CONFIG


def initialize_ee(project_id=None):
    """
    Initialize Earth Engine with authentication.

    Args:
        project_id: str - GEE project ID (default from CONFIG)

    Returns:
        bool: True if successful
    """
    if project_id is None:
        project_id = CONFIG['project_id']

    try:
        ee.Initialize(project=project_id)
        print(f"[OK] Earth Engine initialized (Project: {project_id})")
        return True
    except Exception as e:
        print(f"[..] Earth Engine not authenticated. Starting authentication...")
        print("     Browser will open for Google authentication.\n")

        try:
            # Authenticate - will open browser
            ee.Authenticate()

            # Initialize after authentication
            ee.Initialize(project=project_id)
            print(f"[OK] Earth Engine initialized (Project: {project_id})")
            return True
        except Exception as auth_error:
            print(f"[ERROR] Authentication failed: {auth_error}")
            return False


def check_ee_connection():
    """
    Check if Earth Engine connection is active.

    Returns:
        bool: True if connected
    """
    try:
        # Simple test - get server timestamp
        ee.Date('2024-01-01').getInfo()
        return True
    except:
        return False
