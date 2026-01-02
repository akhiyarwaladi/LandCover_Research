#!/usr/bin/env python3
"""
Generate Visualizations - Clean & Standardized
==============================================

Main visualization script for the land cover research project.

Features:
- RGB natural color composites
- KLHK ground truth overlays
- Cloud coverage analysis
- Multiple resolution support (10m, 20m)
- Standardized output naming

Usage:
    # Generate all visualizations
    python generate_visualizations.py --all

    # Province RGB only
    python generate_visualizations.py --rgb-province

    # City 10m with KLHK overlay
    python generate_visualizations.py --city-klhk

    # List available options
    python generate_visualizations.py --list

Author: Claude Sonnet 4.5
Date: 2026-01-02
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
from rasterio.merge import merge
from rasterio.windows import from_bounds
from matplotlib.patches import Patch

# Import standardized naming
from modules.naming_standards import create_rgb_name

# ============================================================================
# CONFIGURATION
# ============================================================================

# Data paths
DATA_CONFIG = {
    'klhk': 'data/klhk/KLHK_PL2024_Jambi_Full_WithGeometry.geojson',
    'province_20m': [
        'data/sentinel_new_cloudfree/S2_jambi_2024_20m_AllBands-0000000000-0000000000.tif',
        'data/sentinel_new_cloudfree/S2_jambi_2024_20m_AllBands-0000000000-0000010496.tif',
        'data/sentinel_new_cloudfree/S2_jambi_2024_20m_AllBands-0000010496-0000000000.tif',
        'data/sentinel_new_cloudfree/S2_jambi_2024_20m_AllBands-0000010496-0000010496.tif'
    ],
    'city_10m': 'data/sentinel_city/sentinel_city_10m_2024dry_p25.tif'
}

# Output directory
OUTPUT_DIR = 'results/visualizations'

# Region bounds
REGIONS = {
    'city': {
        'min_lon': 103.545,
        'max_lon': 103.660,
        'min_lat': -1.680,
        'max_lat': -1.560
    }
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_sentinel_rgb(tile_paths_or_single, verbose=True):
    """
    Load Sentinel-2 data and extract RGB.

    Args:
        tile_paths_or_single: List of tile paths or single file path
        verbose: Print progress messages

    Returns:
        rgb_normalized: Normalized RGB array (H, W, 3)
        profile: Raster profile
    """
    if verbose:
        print("\n" + "="*80)
        print("LOADING SENTINEL-2 DATA")
        print("="*80)

    # Handle single file or multiple tiles
    if isinstance(tile_paths_or_single, str):
        # Single file
        if verbose:
            print(f"\nLoading: {os.path.basename(tile_paths_or_single)}")

        with rasterio.open(tile_paths_or_single) as src:
            data = src.read()
            profile = src.profile.copy()

            if verbose:
                print(f"  Bands: {src.count}")
                print(f"  Dimensions: {src.width} x {src.height} pixels")
    else:
        # Multiple tiles - mosaic
        if verbose:
            print(f"\nLoading {len(tile_paths_or_single)} tiles:")

        datasets = []
        for path in tile_paths_or_single:
            if os.path.exists(path):
                if verbose:
                    print(f"  • {os.path.basename(path)}")
                datasets.append(rasterio.open(path))

        if not datasets:
            raise FileNotFoundError("No valid tiles found!")

        # Mosaic
        if verbose:
            print("\nMosaicking...")
        data, transform = merge(datasets)

        profile = datasets[0].profile.copy()
        profile.update({
            'transform': transform,
            'height': data.shape[1],
            'width': data.shape[2]
        })

        # Close datasets
        for ds in datasets:
            ds.close()

        if verbose:
            print(f"  Mosaic: {data.shape[0]} bands, {data.shape[1]} x {data.shape[2]} pixels")

    # Extract RGB (B4=Red, B3=Green, B2=Blue)
    blue = data[0]   # B2
    green = data[1]  # B3
    red = data[2]    # B4

    rgb = np.dstack([red, green, blue])

    # Calculate coverage
    total_pixels = rgb.shape[0] * rgb.shape[1]
    valid_pixels = np.sum(~np.isnan(red))
    valid_pct = (valid_pixels / total_pixels) * 100

    if verbose:
        print(f"\nData coverage: {valid_pct:.1f}% valid pixels")

    # Normalize for display (2-98 percentile stretch)
    vmin = np.nanpercentile(rgb, 2)
    vmax = np.nanpercentile(rgb, 98)
    rgb_normalized = np.clip((rgb - vmin) / (vmax - vmin), 0, 1)
    rgb_normalized = np.nan_to_num(rgb_normalized, nan=0)

    return rgb_normalized, profile

def load_klhk_built_area(verbose=True):
    """
    Load KLHK Built Area polygons.

    Returns:
        GeoDataFrame with Built Area polygons and area_ha column
    """
    if verbose:
        print("\n" + "="*80)
        print("LOADING KLHK GROUND TRUTH")
        print("="*80)

    gdf = gpd.read_file(DATA_CONFIG['klhk'])

    if verbose:
        print(f"\nTotal KLHK polygons: {len(gdf):,}")

    # Filter Built Area (code '2012')
    built = gdf[gdf['ID Penutupan Lahan Tahun 2024'] == '2012'].copy()

    # Calculate areas
    built_utm = built.to_crs(epsg=32648)
    built_utm['area_ha'] = built_utm.geometry.area / 10000
    built['area_ha'] = built_utm['area_ha']

    if verbose:
        print(f"Built Area polygons: {len(built):,}")
        print(f"  Median size: {built['area_ha'].median():.2f} ha")
        print(f"  Largest: {built['area_ha'].max():.2f} ha")

    return built

def assess_cloud_coverage(rgb, verbose=True):
    """
    Assess cloud coverage using brightness analysis.

    Returns:
        cloud_pct: Percentage of potential cloud pixels
    """
    brightness = np.mean(rgb, axis=2)
    valid_mask = brightness > 0

    if np.sum(valid_mask) == 0:
        return 0.0

    bright_threshold = np.percentile(brightness[valid_mask], 99)
    potential_clouds = np.sum(brightness > bright_threshold)
    cloud_pct = (potential_clouds / np.sum(valid_mask)) * 100

    if verbose:
        print(f"\nCloud assessment:")
        print(f"  Bright pixels (>99th percentile): {cloud_pct:.2f}%")
        if cloud_pct < 1.0:
            print(f"  ✅ EXCELLENT - Cloud-free!")
        elif cloud_pct < 5.0:
            print(f"  ✅ GOOD - Minimal clouds")
        else:
            print(f"  ⚠️  WARNING - Possible clouds")

    return cloud_pct

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def visualize_province_rgb(output_dir=OUTPUT_DIR, verbose=True):
    """Generate province RGB visualization."""

    print("\n" + "="*80)
    print("PROVINCE RGB VISUALIZATION (20m)")
    print("="*80)

    # Load data
    rgb, profile = load_sentinel_rgb(DATA_CONFIG['province_20m'], verbose=verbose)

    # Assess clouds
    cloud_pct = assess_cloud_coverage(rgb, verbose=verbose)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Full view
    print("\nCreating full province view...")
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))

    ax.imshow(rgb)
    ax.set_title('Jambi Province - RGB Natural Color (20m Resolution)\n'
                 'Percentile 25 Cloud Removal Strategy',
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Pixel X', fontsize=12)
    ax.set_ylabel('Pixel Y', fontsize=12)
    ax.grid(True, alpha=0.3)

    # Info text
    valid_pct = (np.sum(rgb[:,:,0] > 0) / (rgb.shape[0] * rgb.shape[1])) * 100
    info_text = f"Coverage: {valid_pct:.1f}% | Cloud-free: {100-cloud_pct:.1f}%"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Use standardized naming
    output_name = create_rgb_name('province', 20, '2024dry', 'natural')
    output_path = os.path.join(output_dir, f'{output_name}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")

    plt.close()

    return output_path

def visualize_city_rgb(output_dir=OUTPUT_DIR, verbose=True):
    """Generate city RGB visualization at 10m."""

    print("\n" + "="*80)
    print("CITY RGB VISUALIZATION (10m)")
    print("="*80)

    # Load data
    rgb, profile = load_sentinel_rgb(DATA_CONFIG['city_10m'], verbose=verbose)

    # Assess clouds
    cloud_pct = assess_cloud_coverage(rgb, verbose=verbose)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # City view
    print("\nCreating city view...")
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))

    ax.imshow(rgb)
    ax.set_title('Jambi City - RGB Natural Color (10m Resolution)\n'
                 'High Detail - Can See Individual Buildings',
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Pixel X', fontsize=12)
    ax.set_ylabel('Pixel Y', fontsize=12)
    ax.grid(True, alpha=0.3)

    # Info text
    valid_pct = (np.sum(rgb[:,:,0] > 0) / (rgb.shape[0] * rgb.shape[1])) * 100
    info_text = f"Coverage: {valid_pct:.1f}% | Resolution: 10m | Bands: 10"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    plt.tight_layout()

    # Use standardized naming
    output_name = create_rgb_name('city', 10, '2024dry', 'natural')
    output_path = os.path.join(output_dir, f'{output_name}.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")

    plt.close()

    return output_path

def visualize_klhk_overlay_city(output_dir=OUTPUT_DIR, verbose=True):
    """Generate KLHK Built Area overlay on city 10m data."""

    print("\n" + "="*80)
    print("KLHK OVERLAY VISUALIZATION (City 10m)")
    print("="*80)

    # Load data
    rgb, profile = load_sentinel_rgb(DATA_CONFIG['city_10m'], verbose=verbose)
    built = load_klhk_built_area(verbose=verbose)

    # Clip to city bounds
    bounds = REGIONS['city']
    built_city = built.cx[bounds['min_lon']:bounds['max_lon'],
                          bounds['min_lat']:bounds['max_lat']]

    if verbose:
        print(f"\nBuilt Area in city: {len(built_city):,} polygons")
        print(f"Total area: {built_city['area_ha'].sum():.2f} ha")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Three-panel comparison
    print("\nCreating three-panel overlay...")
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # Panel 1: RGB only
    axes[0].imshow(rgb)
    axes[0].set_title('Reality: 10m Satellite RGB\n(What Actually Exists)',
                      fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Pixel X')
    axes[0].set_ylabel('Pixel Y')
    axes[0].grid(True, alpha=0.3)

    # Panel 2: KLHK polygons only
    axes[1].set_xlim(bounds['min_lon'], bounds['max_lon'])
    axes[1].set_ylim(bounds['min_lat'], bounds['max_lat'])
    if len(built_city) > 0:
        built_city.plot(ax=axes[1], color='magenta', edgecolor='black',
                        linewidth=1, alpha=0.7)
    axes[1].set_title('KLHK Ground Truth\n(Generalized "Built Area" Labels)',
                      fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Overlay
    extent = [bounds['min_lon'], bounds['max_lon'],
              bounds['min_lat'], bounds['max_lat']]
    axes[2].imshow(rgb, extent=extent, aspect='auto')
    if len(built_city) > 0:
        built_city.plot(ax=axes[2], color='magenta', edgecolor='yellow',
                        linewidth=2, alpha=0.35)
    axes[2].set_xlim(bounds['min_lon'], bounds['max_lon'])
    axes[2].set_ylim(bounds['min_lat'], bounds['max_lat'])
    axes[2].set_title('Overlay: See Trees/Water INSIDE Pink!\n(Proves Generalization Issue)',
                      fontsize=14, fontweight='bold', color='red')
    axes[2].set_xlabel('Longitude')
    axes[2].set_ylabel('Latitude')
    axes[2].grid(True, alpha=0.3, color='white', linewidth=1.5)

    # Legend
    legend_elements = [
        Patch(facecolor='magenta', edgecolor='yellow', alpha=0.35,
              label='KLHK "Built Area" (generalized)')
    ]
    axes[2].legend(handles=legend_elements, loc='upper right', fontsize=11)

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'klhk_overlay_city_10m.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")

    plt.close()

    return output_path

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Main entry point with argument parsing."""

    parser = argparse.ArgumentParser(
        description='Generate visualizations for land cover research',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all visualizations
  python generate_visualizations.py --all

  # Province RGB only
  python generate_visualizations.py --rgb-province

  # City RGB only
  python generate_visualizations.py --rgb-city

  # City KLHK overlay
  python generate_visualizations.py --city-klhk

  # Multiple options
  python generate_visualizations.py --rgb-province --city-klhk

  # List available options
  python generate_visualizations.py --list
        """
    )

    parser.add_argument('--all', action='store_true',
                        help='Generate all visualizations')
    parser.add_argument('--rgb-province', action='store_true',
                        help='Province RGB (20m)')
    parser.add_argument('--rgb-city', action='store_true',
                        help='City RGB (10m)')
    parser.add_argument('--city-klhk', action='store_true',
                        help='City KLHK overlay (10m)')
    parser.add_argument('--list', action='store_true',
                        help='List available visualizations')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR,
                        help=f'Output directory (default: {OUTPUT_DIR})')

    args = parser.parse_args()

    # List mode
    if args.list:
        print("\n" + "="*80)
        print("AVAILABLE VISUALIZATIONS")
        print("="*80)
        print("\n1. --rgb-province")
        print("   Province RGB natural color (20m resolution)")
        print("   Shows full Jambi Province with cloud-free imagery")
        print("\n2. --rgb-city")
        print("   City RGB natural color (10m resolution)")
        print("   High detail view of Jambi City")
        print("\n3. --city-klhk")
        print("   KLHK overlay on city 10m data")
        print("   Three-panel comparison showing generalization issue")
        print("\n4. --all")
        print("   Generate all of the above")
        print("\n" + "="*80)
        return

    # Default to all if no options specified
    if not any([args.all, args.rgb_province, args.rgb_city, args.city_klhk]):
        args.all = True

    # Generate visualizations
    output_dir = args.output_dir

    print("\n" + "="*80)
    print("GENERATE VISUALIZATIONS - STANDARDIZED")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")

    results = []

    if args.all or args.rgb_province:
        path = visualize_province_rgb(output_dir=output_dir)
        results.append(('Province RGB', path))

    if args.all or args.rgb_city:
        path = visualize_city_rgb(output_dir=output_dir)
        results.append(('City RGB', path))

    if args.all or args.city_klhk:
        path = visualize_klhk_overlay_city(output_dir=output_dir)
        results.append(('City KLHK Overlay', path))

    # Summary
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nGenerated {len(results)} visualizations:")
    for name, path in results:
        print(f"  ✅ {name}: {path}")

    print("\n" + "="*80)

if __name__ == '__main__':
    main()
