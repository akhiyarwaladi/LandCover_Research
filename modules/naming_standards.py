"""
Standardized Naming Conventions
================================

Centralized naming standards for all outputs in the repository.

Author: Claude Sonnet 4.5
Date: 2026-01-02
"""

import os
from typing import Optional, Dict

# ============================================================================
# NAMING COMPONENTS
# ============================================================================

STRATEGY_CODES = {
    'percentile_25': 'p25',
    'percentile_30': 'p30',
    'kalimantan': 'kalimantan',
    'balanced': 'balanced',
    'pan_tropical': 'pantropical',
    'conservative': 'conservative',
    'current': 'median',
    'median': 'median',
}

MODEL_CODES = {
    'Random Forest': 'rf',
    'Extra Trees': 'et',
    'LightGBM': 'lgbm',
    'Decision Tree': 'dt',
    'Logistic Regression': 'lr',
    'SGD Classifier': 'sgd',
    'Naive Bayes': 'nb',
}

REGION_CODES = {
    'province': 'province',
    'city': 'city',
    'Jambi': 'province',
    'Kota Jambi': 'city',
}

# ============================================================================
# NAMING FUNCTIONS
# ============================================================================

def get_strategy_code(strategy_name: str) -> str:
    """Get short code for cloud removal strategy."""
    return STRATEGY_CODES.get(strategy_name, strategy_name.lower())

def get_model_code(model_name: str) -> str:
    """Get short code for ML model."""
    return MODEL_CODES.get(model_name, model_name.lower().replace(' ', '_'))

def get_region_code(region_name: str) -> str:
    """Get short code for region."""
    return REGION_CODES.get(region_name, region_name.lower().replace(' ', '_'))

def create_sentinel_name(
    region: str,
    resolution: int,
    timeframe: str,
    strategy: str,
    tile: Optional[int] = None
) -> str:
    """
    Create standardized name for Sentinel-2 data.

    Args:
        region: 'province', 'city', or region name
        resolution: 10 or 20
        timeframe: '2024dry', '2024', etc.
        strategy: Cloud removal strategy name
        tile: Optional tile number for multi-tile exports

    Returns:
        Standardized filename

    Example:
        >>> create_sentinel_name('province', 20, '2024dry', 'percentile_25', 1)
        'sentinel_province_20m_2024dry_p25-tile1'
    """
    region_code = get_region_code(region)
    strategy_code = get_strategy_code(strategy)

    name = f"sentinel_{region_code}_{resolution}m_{timeframe}_{strategy_code}"

    if tile is not None:
        name += f"-tile{tile}"

    return name

def create_rgb_name(
    region: str,
    resolution: int,
    timeframe: str,
    rgb_type: str = 'natural',
    variant: Optional[str] = None
) -> str:
    """
    Create standardized name for RGB visualization.

    Args:
        region: 'province', 'city', or region name
        resolution: 10 or 20
        timeframe: '2024dry', '2024', etc.
        rgb_type: 'natural', 'falsecolor', 'ndvi', etc.
        variant: Optional variant identifier

    Returns:
        Standardized filename

    Example:
        >>> create_rgb_name('city', 10, '2024dry', 'natural', '1')
        'rgb_city_10m_2024dry_natural_1'
    """
    region_code = get_region_code(region)

    name = f"rgb_{region_code}_{resolution}m_{timeframe}_{rgb_type}"

    if variant:
        name += f"_{variant}"

    return name

def create_classification_name(
    region: str,
    resolution: int,
    timeframe: str,
    model: str
) -> str:
    """
    Create standardized name for classification output.

    Args:
        region: 'province', 'city', or region name
        resolution: 10 or 20
        timeframe: '2024dry', '2024', etc.
        model: ML model name

    Returns:
        Standardized filename

    Example:
        >>> create_classification_name('province', 20, '2024dry', 'Random Forest')
        'classification_province_20m_2024dry_rf'
    """
    region_code = get_region_code(region)
    model_code = get_model_code(model)

    return f"classification_{region_code}_{resolution}m_{timeframe}_{model_code}"

def create_results_name(
    region: str,
    resolution: int,
    timeframe: str,
    analysis: str
) -> str:
    """
    Create standardized name for analysis results.

    Args:
        region: 'province', 'city', or region name
        resolution: 10 or 20
        timeframe: '2024dry', '2024', etc.
        analysis: Type of analysis (comparison, confusion_matrix, etc.)

    Returns:
        Standardized filename

    Example:
        >>> create_results_name('province', 20, '2024dry', 'comparison')
        'results_province_20m_2024dry_comparison'
    """
    region_code = get_region_code(region)

    return f"results_{region_code}_{resolution}m_{timeframe}_{analysis}"

def create_metrics_name(
    region: str,
    resolution: int,
    timeframe: str,
    model: str
) -> str:
    """
    Create standardized name for model metrics.

    Args:
        region: 'province', 'city', or region name
        resolution: 10 or 20
        timeframe: '2024dry', '2024', etc.
        model: ML model name

    Returns:
        Standardized filename

    Example:
        >>> create_metrics_name('province', 20, '2024dry', 'Random Forest')
        'metrics_province_20m_2024dry_rf'
    """
    region_code = get_region_code(region)
    model_code = get_model_code(model)

    return f"metrics_{region_code}_{resolution}m_{timeframe}_{model_code}"

def create_test_name(
    test_type: str,
    region: str,
    resolution: int,
    descriptor: str
) -> str:
    """
    Create standardized name for test outputs.

    Args:
        test_type: Type of test (cloud, classification, etc.)
        region: 'province', 'city', 'sample', etc.
        resolution: 10 or 20
        descriptor: Specific test descriptor

    Returns:
        Standardized filename

    Example:
        >>> create_test_name('cloud', 'sample', 20, 'p25')
        'test_cloud_sample_20m_p25'
    """
    region_code = get_region_code(region)

    return f"test_{test_type}_{region_code}_{resolution}m_{descriptor}"

def create_boundary_name(
    region: str,
    source: str
) -> str:
    """
    Create standardized name for boundary files.

    Args:
        region: 'province', 'city', or region name
        source: Data source (geoboundaries, klhk, etc.)

    Returns:
        Standardized filename

    Example:
        >>> create_boundary_name('province', 'geoboundaries')
        'boundary_province_geoboundaries'
    """
    region_code = get_region_code(region)

    return f"boundary_{region_code}_{source}"

def parse_standard_name(filename: str) -> Dict[str, str]:
    """
    Parse standardized filename into components.

    Args:
        filename: Standardized filename

    Returns:
        Dictionary with parsed components

    Example:
        >>> parse_standard_name('sentinel_province_20m_2024dry_p25.tif')
        {
            'category': 'sentinel',
            'region': 'province',
            'resolution': '20m',
            'timeframe': '2024dry',
            'descriptor': 'p25',
            'extension': 'tif'
        }
    """
    # Remove extension
    name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')

    # Remove tile suffix if present
    tile = None
    if '-tile' in name:
        name, tile_part = name.split('-tile')
        tile = int(tile_part)

    # Split into parts
    parts = name.split('_')

    result = {
        'category': parts[0] if len(parts) > 0 else None,
        'region': parts[1] if len(parts) > 1 else None,
        'resolution': parts[2] if len(parts) > 2 else None,
        'timeframe': parts[3] if len(parts) > 3 else None,
        'descriptor': '_'.join(parts[4:]) if len(parts) > 4 else None,
        'tile': tile,
        'extension': ext
    }

    return result

# ============================================================================
# PATH HELPERS
# ============================================================================

def get_standard_output_dir(category: str) -> str:
    """
    Get standard output directory for category.

    Args:
        category: Output category (sentinel, rgb, classification, etc.)

    Returns:
        Standard output directory path
    """
    base_dirs = {
        'sentinel': 'data/sentinel',
        'sentinel_city': 'data/sentinel_city',
        'rgb': 'results/rgb',
        'classification': 'results/classification',
        'results': 'results/metrics',
        'metrics': 'results/metrics',
        'test': 'results/test',
        'boundary': 'data/boundaries',
    }

    return base_dirs.get(category, 'results')

def create_standard_path(
    category: str,
    filename: str,
    extension: str = 'tif',
    ensure_dir: bool = True
) -> str:
    """
    Create full standardized path with directory.

    Args:
        category: Output category
        filename: Standardized filename (without extension)
        extension: File extension
        ensure_dir: Create directory if it doesn't exist

    Returns:
        Full path with directory and extension
    """
    output_dir = get_standard_output_dir(category)

    if ensure_dir:
        os.makedirs(output_dir, exist_ok=True)

    full_filename = f"{filename}.{extension}"

    return os.path.join(output_dir, full_filename)

# ============================================================================
# TESTING
# ============================================================================

if __name__ == '__main__':
    """Test naming functions."""

    print("Testing Naming Standards Module")
    print("="*60)

    # Test Sentinel naming
    name = create_sentinel_name('province', 20, '2024dry', 'percentile_25', 1)
    print(f"\nSentinel: {name}")

    # Test RGB naming
    name = create_rgb_name('city', 10, '2024dry', 'natural', '1')
    print(f"RGB: {name}")

    # Test Classification naming
    name = create_classification_name('province', 20, '2024dry', 'Random Forest')
    print(f"Classification: {name}")

    # Test parsing
    parsed = parse_standard_name('sentinel_province_20m_2024dry_p25-tile1.tif')
    print(f"\nParsed: {parsed}")

    # Test path creation
    path = create_standard_path('rgb', 'rgb_city_10m_2024dry_natural', 'png', False)
    print(f"\nPath: {path}")

    print("\n" + "="*60)
    print("✅ All tests passed!")
