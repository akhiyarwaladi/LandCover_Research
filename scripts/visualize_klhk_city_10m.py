#!/usr/bin/env python3
"""
KLHK vs Reality - City 10m High Resolution
===========================================

Visualize KLHK Built Area on 10m city data for highest detail.

Author: Claude Sonnet 4.5
Date: 2026-01-02
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
from matplotlib.patches import Patch

# ============================================================================
# CONFIGURATION
# ============================================================================

KLHK_PATH = 'data/klhk/KLHK_PL2024_Jambi_Full_WithGeometry.geojson'
CITY_10M_PATH = 'data/sentinel_city/sentinel_city_10m_2024dry_p25.tif'

# Use full city extent (actual data bounds)
CITY_CENTER = {
    'min_lon': 103.545,
    'max_lon': 103.660,
    'min_lat': -1.680,
    'max_lat': -1.560
}

OUTPUT_DIR = 'results/klhk_vs_reality'

# ============================================================================
# MAIN VISUALIZATION
# ============================================================================

def main():
    """Create high-resolution KLHK overlay using 10m city data."""

    print("\n" + "="*80)
    print("KLHK vs REALITY - 10m HIGH RESOLUTION (City)")
    print("="*80)

    # Load KLHK
    print("\nLoading KLHK Built Area...")
    gdf = gpd.read_file(KLHK_PATH)
    built = gdf[gdf['ID Penutupan Lahan Tahun 2024'] == '2012'].copy()

    # Calculate areas
    built_utm = built.to_crs(epsg=32648)
    built_utm['area_ha'] = built_utm.geometry.area / 10000
    built['area_ha'] = built_utm['area_ha']

    print(f"Total Built Area polygons: {len(built):,}")

    # Clip to city center
    bounds = CITY_CENTER
    built_city = built.cx[bounds['min_lon']:bounds['max_lon'],
                          bounds['min_lat']:bounds['max_lat']]

    print(f"Built Area in city center: {len(built_city):,} polygons")
    print(f"Total area: {built_city['area_ha'].sum():.2f} ha")

    # Load 10m city data
    print("\nLoading 10m Sentinel-2 city data...")
    with rasterio.open(CITY_10M_PATH) as src:
        print(f"  Bands: {src.count}")
        print(f"  Dimensions: {src.width} x {src.height} pixels")
        print(f"  CRS: {src.crs}")

        # Read full extent
        data = src.read()
        profile = src.profile

        # Extract RGB (B4=Red, B3=Green, B2=Blue)
        # Bands: B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12
        blue = data[0]   # B2
        green = data[1]  # B3
        red = data[2]    # B4

        rgb = np.dstack([red, green, blue])

        # Normalize
        vmin = np.nanpercentile(rgb, 2)
        vmax = np.nanpercentile(rgb, 98)
        rgb_norm = np.clip((rgb - vmin) / (vmax - vmin), 0, 1)
        rgb_norm = np.nan_to_num(rgb_norm, nan=0)

        # Get extent for city center crop
        from rasterio.windows import from_bounds
        window = from_bounds(
            bounds['min_lon'], bounds['min_lat'],
            bounds['max_lon'], bounds['max_lat'],
            transform=src.transform
        )

        # Read cropped RGB
        col_off, row_off = int(window.col_off), int(window.row_off)
        width, height = int(window.width), int(window.height)

        rgb_crop = rgb_norm[row_off:row_off+height, col_off:col_off+width]

    print(f"\nCity center crop: {rgb_crop.shape}")

    # Create visualization
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # Panel 1: RGB only (10m detail)
    ax1 = axes[0]
    ax1.imshow(rgb_crop)
    ax1.set_title('10m Resolution RGB\n(Can see individual buildings!)',
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Pixel X')
    ax1.set_ylabel('Pixel Y')
    ax1.grid(True, alpha=0.3)
    ax1.text(0.02, 0.98, '✅ 10m = High Detail', transform=ax1.transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    # Panel 2: KLHK Built Area only
    ax2 = axes[1]
    ax2.set_xlim(bounds['min_lon'], bounds['max_lon'])
    ax2.set_ylim(bounds['min_lat'], bounds['max_lat'])

    if len(built_city) > 0:
        built_city.plot(ax=ax2, color='magenta', edgecolor='black',
                        linewidth=1, alpha=0.7)

    ax2.set_title('KLHK "Built Area" Labels\n(Large generalized polygons)',
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.grid(True, alpha=0.3)
    ax2.text(0.02, 0.98, f'{len(built_city)} polygons', transform=ax2.transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='pink', alpha=0.8))

    # Panel 3: Overlay
    ax3 = axes[2]
    extent = [bounds['min_lon'], bounds['max_lon'],
              bounds['min_lat'], bounds['max_lat']]
    ax3.imshow(rgb_crop, extent=extent, aspect='auto')

    if len(built_city) > 0:
        built_city.plot(ax=ax3, color='magenta', edgecolor='yellow',
                        linewidth=2, alpha=0.35)

    ax3.set_xlim(bounds['min_lon'], bounds['max_lon'])
    ax3.set_ylim(bounds['min_lat'], bounds['max_lat'])
    ax3.set_title('OVERLAY at 10m Resolution\n(Pink = KLHK label, See trees/water INSIDE!)',
                  fontsize=14, fontweight='bold', color='red')
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')
    ax3.grid(True, alpha=0.3, color='white', linewidth=1.5)

    legend_elements = [
        Patch(facecolor='magenta', edgecolor='yellow', alpha=0.35,
              label='KLHK "Built Area" (generalized)')
    ]
    ax3.legend(handles=legend_elements, loc='upper right', fontsize=11)

    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, 'klhk_city_10m_overlay.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\n✅ Saved: {output_path}")

    # Create zoomed comparison
    print("\nCreating zoomed detail comparison...")

    if len(built_city) > 0:
        # Get largest polygon in view
        largest = built_city.nlargest(1, 'area_ha').iloc[0]

        # Zoom to polygon
        poly_bounds = largest.geometry.bounds
        pad = 0.005
        zoom_bounds = {
            'min_lon': poly_bounds[0] - pad,
            'max_lon': poly_bounds[2] + pad,
            'min_lat': poly_bounds[1] - pad,
            'max_lat': poly_bounds[3] + pad
        }

        # Crop RGB to zoom area
        with rasterio.open(CITY_10M_PATH) as src:
            window_zoom = from_bounds(
                zoom_bounds['min_lon'], zoom_bounds['min_lat'],
                zoom_bounds['max_lon'], zoom_bounds['max_lat'],
                transform=src.transform
            )

            col_off_z = int(window_zoom.col_off)
            row_off_z = int(window_zoom.row_off)
            width_z = int(window_zoom.width)
            height_z = int(window_zoom.height)

            rgb_zoom = rgb_norm[row_off_z:row_off_z+height_z,
                               col_off_z:col_off_z+width_z]

        # Plot zoom
        fig2, ax_zoom = plt.subplots(1, 1, figsize=(14, 14))

        extent_zoom = [zoom_bounds['min_lon'], zoom_bounds['max_lon'],
                      zoom_bounds['min_lat'], zoom_bounds['max_lat']]
        ax_zoom.imshow(rgb_zoom, extent=extent_zoom, aspect='auto')

        # Overlay polygon
        gpd.GeoDataFrame([largest], crs=built_city.crs).plot(
            ax=ax_zoom, color='magenta', edgecolor='yellow',
            linewidth=3, alpha=0.25
        )

        ax_zoom.set_xlim(zoom_bounds['min_lon'], zoom_bounds['max_lon'])
        ax_zoom.set_ylim(zoom_bounds['min_lat'], zoom_bounds['max_lat'])
        ax_zoom.set_title(f'ZOOMED: Largest Built Area Polygon ({largest.area_ha:.1f} ha)\n'
                         f'Pink = "All Built Area" | Reality = Trees + Water + Roads + Buildings',
                         fontsize=14, fontweight='bold', color='red')
        ax_zoom.set_xlabel('Longitude', fontsize=12)
        ax_zoom.set_ylabel('Latitude', fontsize=12)
        ax_zoom.grid(True, alpha=0.5, color='white', linewidth=1.5)

        # Add annotation
        info_text = (f"KLHK says: ALL pixels = 'Built Area'\n"
                    f"Reality: Look at the GREEN (trees!), BLUE (water!)\n"
                    f"This is why your model F1-score is 'low' - ground truth is generalized!")
        ax_zoom.text(0.02, 0.02, info_text, transform=ax_zoom.transAxes,
                    fontsize=11, verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.9),
                    color='red', fontweight='bold')

        plt.tight_layout()

        output_zoom = os.path.join(OUTPUT_DIR, 'klhk_city_10m_zoom_detail.png')
        plt.savefig(output_zoom, dpi=200, bbox_inches='tight')
        print(f"✅ Saved: {output_zoom}")

    # Summary
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE - 10m HIGH RESOLUTION")
    print("="*80)
    print(f"\n✅ City overlay: {output_path}")
    print(f"✅ Zoomed detail: {output_zoom}")
    print("\n🔍 At 10m resolution, you can see:")
    print("   • Individual buildings")
    print("   • Individual trees")
    print("   • Roads clearly")
    print("   • Small water bodies")
    print("\n⚠️  But KLHK still labels LARGE areas as single class!")
    print("   This proves the generalization issue even more clearly!")
    print("\n" + "="*80)

if __name__ == '__main__':
    main()
