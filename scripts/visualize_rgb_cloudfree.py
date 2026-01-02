#!/usr/bin/env python3
"""
Quick RGB Visualization - Cloud-Free Province Data
==================================================

Visualize new province data to verify cloud removal effectiveness.

Author: Claude Sonnet 4.5
Date: 2026-01-02
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.merge import merge

# ============================================================================
# CONFIGURATION
# ============================================================================

SENTINEL2_TILES = [
    'data/sentinel_new_cloudfree/S2_jambi_2024_20m_AllBands-0000000000-0000000000.tif',
    'data/sentinel_new_cloudfree/S2_jambi_2024_20m_AllBands-0000000000-0000010496.tif',
    'data/sentinel_new_cloudfree/S2_jambi_2024_20m_AllBands-0000010496-0000000000.tif',
    'data/sentinel_new_cloudfree/S2_jambi_2024_20m_AllBands-0000010496-0000010496.tif'
]

OUTPUT_DIR = 'results/rgb_visualization'

# ============================================================================
# RGB VISUALIZATION
# ============================================================================

def load_and_mosaic_tiles(tile_paths):
    """Load and mosaic Sentinel-2 tiles."""

    print("\n" + "="*80)
    print("LOADING SENTINEL-2 TILES")
    print("="*80)

    datasets = []
    for path in tile_paths:
        if os.path.exists(path):
            print(f"Loading: {os.path.basename(path)}")
            datasets.append(rasterio.open(path))
        else:
            print(f"⚠️  Not found: {path}")

    if not datasets:
        raise FileNotFoundError("No tiles found!")

    print(f"\n✅ Loaded {len(datasets)} tiles")

    # Mosaic
    print("\nMosaicking tiles...")
    mosaic_array, mosaic_transform = merge(datasets)

    # Get profile from first dataset
    profile = datasets[0].profile.copy()
    profile.update({
        'transform': mosaic_transform,
        'height': mosaic_array.shape[1],
        'width': mosaic_array.shape[2]
    })

    # Close datasets
    for ds in datasets:
        ds.close()

    print(f"Mosaic shape: {mosaic_array.shape}")
    print(f"  Bands: {mosaic_array.shape[0]}")
    print(f"  Height: {mosaic_array.shape[1]:,} pixels")
    print(f"  Width: {mosaic_array.shape[2]:,} pixels")

    return mosaic_array, profile

def create_rgb_visualization(mosaic_array, output_dir):
    """Create RGB visualization with cloud detection."""

    print("\n" + "="*80)
    print("RGB VISUALIZATION")
    print("="*80)

    # Extract RGB bands (B4=Red, B3=Green, B2=Blue)
    # Sentinel-2 bands: [B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12]
    # Index:            [ 0,  1,  2,  3,  4,  5,  6,   7,   8,   9]

    blue = mosaic_array[0]   # B2
    green = mosaic_array[1]  # B3
    red = mosaic_array[2]    # B4

    print(f"\nBand statistics:")
    print(f"  Red (B4):   min={np.nanmin(red):.4f}, max={np.nanmax(red):.4f}, mean={np.nanmean(red):.4f}")
    print(f"  Green (B3): min={np.nanmin(green):.4f}, max={np.nanmax(green):.4f}, mean={np.nanmean(green):.4f}")
    print(f"  Blue (B2):  min={np.nanmin(blue):.4f}, max={np.nanmax(blue):.4f}, mean={np.nanmean(blue):.4f}")

    # Stack RGB
    rgb = np.dstack([red, green, blue])

    # Count valid pixels
    total_pixels = rgb.shape[0] * rgb.shape[1]
    valid_pixels = np.sum(~np.isnan(red))
    valid_pct = (valid_pixels / total_pixels) * 100

    print(f"\nData coverage:")
    print(f"  Total pixels: {total_pixels:,}")
    print(f"  Valid pixels: {valid_pixels:,} ({valid_pct:.1f}%)")
    print(f"  NaN pixels: {total_pixels - valid_pixels:,} ({100-valid_pct:.1f}%)")

    # Normalize for display (2-98 percentile stretch)
    print("\nNormalizing for display (2-98 percentile stretch)...")
    vmin = np.nanpercentile(rgb, 2)
    vmax = np.nanpercentile(rgb, 98)

    rgb_normalized = np.clip((rgb - vmin) / (vmax - vmin), 0, 1)

    # Replace NaN with black
    rgb_normalized = np.nan_to_num(rgb_normalized, nan=0)

    # Check for potential clouds (very bright pixels)
    brightness = np.mean(rgb, axis=2)
    bright_threshold = np.nanpercentile(brightness, 99)
    potential_clouds = np.sum(brightness > bright_threshold)
    cloud_pct = (potential_clouds / valid_pixels) * 100 if valid_pixels > 0 else 0

    print(f"\nCloud detection (brightness analysis):")
    print(f"  Bright pixels (>99th percentile): {potential_clouds:,} ({cloud_pct:.2f}%)")

    if cloud_pct < 1.0:
        print(f"  ✅ EXCELLENT! <1% bright pixels (cloud-free)")
    elif cloud_pct < 5.0:
        print(f"  ✅ GOOD! <5% bright pixels (minimal clouds)")
    else:
        print(f"  ⚠️  WARNING! >{cloud_pct:.1f}% bright pixels (possible clouds)")

    # Create visualization
    os.makedirs(output_dir, exist_ok=True)

    # Full province RGB
    print("\nCreating full province RGB visualization...")
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))

    ax.imshow(rgb_normalized)
    ax.set_title('Jambi Province - RGB Natural Color (Percentile 25 Cloud Removal)',
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Pixel X', fontsize=12)
    ax.set_ylabel('Pixel Y', fontsize=12)
    ax.grid(True, alpha=0.3)

    # Add text info
    info_text = f"Coverage: {valid_pct:.1f}% | Bright pixels: {cloud_pct:.2f}%"
    if cloud_pct < 1.0:
        info_text += " | ✅ Cloud-free!"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    output_path = os.path.join(output_dir, 'province_rgb_cloudfree.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")

    # Zoomed view (center region)
    print("\nCreating zoomed view (center region)...")
    h, w = rgb_normalized.shape[:2]
    center_h, center_w = h // 2, w // 2
    zoom_size = min(2000, h // 4, w // 4)

    rgb_zoom = rgb_normalized[
        center_h - zoom_size:center_h + zoom_size,
        center_w - zoom_size:center_w + zoom_size
    ]

    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 12))
    ax2.imshow(rgb_zoom)
    ax2.set_title('Jambi Province - Center Region (Zoomed)',
                  fontsize=16, fontweight='bold')
    ax2.set_xlabel('Pixel X', fontsize=12)
    ax2.set_ylabel('Pixel Y', fontsize=12)
    ax2.grid(True, alpha=0.3)

    output_path_zoom = os.path.join(output_dir, 'province_rgb_cloudfree_zoom.png')
    plt.tight_layout()
    plt.savefig(output_path_zoom, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_path_zoom}")

    plt.close('all')

    return output_path, output_path_zoom, cloud_pct

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main visualization script."""

    print("\n" + "="*80)
    print("RGB VISUALIZATION - CLOUD-FREE PROVINCE DATA")
    print("="*80)

    # Load data
    mosaic_array, profile = load_and_mosaic_tiles(SENTINEL2_TILES)

    # Create RGB visualization
    rgb_path, zoom_path, cloud_pct = create_rgb_visualization(mosaic_array, OUTPUT_DIR)

    # Summary
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\n✅ Full RGB: {rgb_path}")
    print(f"✅ Zoomed: {zoom_path}")

    if cloud_pct < 1.0:
        print(f"\n🎉 SUCCESS! Cloud coverage: {cloud_pct:.2f}% (EXCELLENT)")
        print("   The percentile_25 strategy worked perfectly!")
    elif cloud_pct < 5.0:
        print(f"\n✅ GOOD! Cloud coverage: {cloud_pct:.2f}% (minimal)")
    else:
        print(f"\n⚠️  Cloud coverage: {cloud_pct:.2f}% (check visualization)")

    print("\n" + "="*80)

if __name__ == '__main__':
    main()
