#!/usr/bin/env python3
"""
Quick Cloud Strategy Testing - Small Area Comparison
====================================================

Tests multiple cloud removal strategies on a SMALL area to:
1. Avoid waiting 30 min for each full download
2. Compare strategies visually in ~5-10 minutes
3. Decide which strategy to use for full province

Author: Claude Sonnet 4.5
Date: 2026-01-02
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import ee
import numpy as np
import rasterio
from rasterio.transform import from_bounds
import matplotlib.pyplot as plt
from modules.cloud_removal import CloudRemovalConfig

# ============================================================================
# CONFIGURATION
# ============================================================================

# Small test area in Jambi (where clouds are visible in current data)
# This is the TOP-LEFT area with residual clouds
TEST_BOUNDS = [102.8, -1.2, 103.0, -1.0]  # ~20x20 km area with known clouds

# Strategies to compare
STRATEGIES_TO_TEST = [
    'current',        # Baseline
    'percentile_25',  # Recommended
    'kalimantan',     # Indonesia proven
    'balanced'        # Compromise
]

# Time period (same as main download)
START_DATE = '2024-06-01'
END_DATE = '2024-09-30'

# Output
OUTPUT_DIR = 'results/strategy_test'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# FUNCTIONS
# ============================================================================

def initialize_ee():
    """Initialize Earth Engine."""
    try:
        ee.Initialize(project='ee-akhiyarwaladi')
        print("✅ Earth Engine initialized")
    except:
        print("Authenticating Earth Engine...")
        ee.Authenticate()
        ee.Initialize(project='ee-akhiyarwaladi')

def get_collection_with_strategy(region, start_date, end_date, strategy_name):
    """Get Sentinel-2 collection with strategy parameters."""

    strategy = CloudRemovalConfig.get_strategy(strategy_name)

    print(f"\n{'='*60}")
    print(f"Strategy: {strategy['name']}")
    print(f"{'='*60}")
    print(f"  Cloud Score+: {strategy['cloud_score_threshold']}")
    print(f"  Max Cloud %: {strategy['max_cloud_percent']}")
    print(f"  Composite: {strategy['composite_method']}")
    if strategy.get('pre_filter_percent'):
        print(f"  Pre-filter: ≤{strategy['pre_filter_percent']}%")

    # Load Sentinel-2
    max_cloud = strategy.get('pre_filter_percent') or strategy['max_cloud_percent']
    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterDate(start_date, end_date)
          .filterBounds(region)
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud)))

    count = s2.size().getInfo()
    print(f"  Images found: {count}")

    # Cloud Score+ masking
    cs_plus = (ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
               .filterDate(start_date, end_date)
               .filterBounds(region))

    def mask_clouds(image):
        cs = cs_plus.filter(ee.Filter.eq('system:index', image.get('system:index'))).first()
        clear_mask = cs.select('cs_cdf').gte(strategy['cloud_score_threshold'])
        return (image
                .updateMask(clear_mask)
                .select(['B2', 'B3', 'B4', 'B8'])
                .divide(10000))

    s2_masked = s2.map(mask_clouds)

    return s2_masked, strategy

def create_composite(collection, strategy_config, region):
    """Create composite based on strategy method."""

    method = strategy_config['composite_method']

    if method == 'median':
        composite = collection.median()
    elif method == 'percentile_25':
        composite = collection.reduce(ee.Reducer.percentile([25]))
        # Rename bands
        composite = composite.select(
            ['B2_p25', 'B3_p25', 'B4_p25', 'B8_p25'],
            ['B2', 'B3', 'B4', 'B8']
        )
    elif method == 'percentile_30':
        composite = collection.reduce(ee.Reducer.percentile([30]))
        composite = composite.select(
            ['B2_p30', 'B3_p30', 'B4_p30', 'B8_p30'],
            ['B2', 'B3', 'B4', 'B8']
        )
    else:
        print(f"⚠️  Unknown method '{method}', using median")
        composite = collection.median()

    return composite.clip(region)

def download_composite_direct(composite, region, bounds):
    """Download composite directly (small area only)."""

    try:
        url = composite.getDownloadURL({
            'scale': 20,
            'region': region,
            'format': 'GEO_TIFF',
            'crs': 'EPSG:4326'
        })

        import requests
        response = requests.get(url, timeout=300)

        if response.status_code == 200:
            return response.content
        else:
            print(f"❌ Download failed: {response.status_code}")
            return None

    except Exception as e:
        print(f"❌ Download error: {e}")
        return None

def save_geotiff(data_bytes, filepath, bounds):
    """Save downloaded bytes as GeoTIFF."""

    with open(filepath, 'wb') as f:
        f.write(data_bytes)

    print(f"  ✅ Saved: {filepath}")

def calculate_valid_percentage(filepath):
    """Calculate % of valid (non-NaN) pixels."""

    with rasterio.open(filepath) as src:
        data = src.read()

        # Check for NaN or 0 values
        valid_mask = ~np.isnan(data) & (data > 0)
        valid_pixels = np.sum(np.all(valid_mask, axis=0))
        total_pixels = data.shape[1] * data.shape[2]

        valid_pct = (valid_pixels / total_pixels) * 100

    return valid_pct, total_pixels, valid_pixels

def create_rgb_preview(filepath, strategy_name):
    """Create RGB preview with statistics."""

    with rasterio.open(filepath) as src:
        # Read RGB bands (B4=Red, B3=Green, B2=Blue)
        r = src.read(3)  # B4
        g = src.read(2)  # B3
        b = src.read(1)  # B2

        rgb = np.dstack([r, g, b])

        # Calculate valid pixels
        valid_mask = ~np.isnan(rgb).any(axis=2) & (rgb > 0).any(axis=2)
        valid_pct = (np.sum(valid_mask) / valid_mask.size) * 100

        # Normalize for display (2-98 percentile stretch)
        rgb_display = np.zeros_like(rgb, dtype=np.float32)

        for i in range(3):
            band = rgb[:, :, i]
            valid_pixels = band[valid_mask]

            if len(valid_pixels) > 0:
                p2, p98 = np.nanpercentile(valid_pixels, [2, 98])
                band_norm = np.clip((band - p2) / (p98 - p2 + 1e-10), 0, 1)
                rgb_display[:, :, i] = band_norm

        # Set NaN to white
        rgb_display[~valid_mask] = [1.0, 1.0, 1.0]

    return rgb_display, valid_pct

# ============================================================================
# MAIN TESTING WORKFLOW
# ============================================================================

def main():
    """Run strategy comparison test."""

    print("\n" + "="*80)
    print("CLOUD REMOVAL STRATEGY - QUICK TEST")
    print("="*80)
    print(f"\nTest Area: {TEST_BOUNDS}")
    print(f"Area Size: ~20×20 km")
    print(f"Strategies: {len(STRATEGIES_TO_TEST)}")
    print(f"Expected Time: 5-10 minutes")
    print("\n" + "="*80)

    # Initialize
    initialize_ee()

    # Test region
    region = ee.Geometry.Rectangle(TEST_BOUNDS)

    # Store results
    results = {}

    # Test each strategy
    for i, strategy_name in enumerate(STRATEGIES_TO_TEST, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{len(STRATEGIES_TO_TEST)}] Testing: {strategy_name.upper()}")
        print(f"{'='*80}")

        try:
            # Get collection with strategy
            collection, strategy_config = get_collection_with_strategy(
                region, START_DATE, END_DATE, strategy_name
            )

            # Create composite
            print("  Creating composite...")
            composite = create_composite(collection, strategy_config, region)

            # Download
            print("  Downloading (small area, ~30 seconds)...")
            data_bytes = download_composite_direct(composite, region, TEST_BOUNDS)

            if data_bytes:
                # Save
                filepath = f"{OUTPUT_DIR}/test_{strategy_name}.tif"
                save_geotiff(data_bytes, filepath, TEST_BOUNDS)

                # Calculate stats
                valid_pct, total_px, valid_px = calculate_valid_percentage(filepath)

                print(f"  📊 Valid pixels: {valid_pct:.1f}% ({valid_px:,}/{total_px:,})")

                # Store results
                results[strategy_name] = {
                    'filepath': filepath,
                    'valid_pct': valid_pct,
                    'total_pixels': total_px,
                    'valid_pixels': valid_px,
                    'strategy': strategy_config
                }

            else:
                print(f"  ❌ Failed to download {strategy_name}")

        except Exception as e:
            print(f"  ❌ Error testing {strategy_name}: {e}")
            import traceback
            traceback.print_exc()

    # Generate comparison visualization
    if results:
        print(f"\n{'='*80}")
        print("GENERATING COMPARISON")
        print(f"{'='*80}")

        n_strategies = len(results)
        fig, axes = plt.subplots(2, n_strategies, figsize=(5*n_strategies, 10))

        if n_strategies == 1:
            axes = axes.reshape(2, 1)

        for idx, (strategy_name, result) in enumerate(results.items()):

            # RGB preview
            rgb, valid_pct = create_rgb_preview(result['filepath'], strategy_name)

            # Top row: RGB
            ax_rgb = axes[0, idx]
            ax_rgb.imshow(rgb)
            ax_rgb.set_title(f"{strategy_name.upper()}\nValid: {valid_pct:.1f}%",
                           fontsize=12, fontweight='bold')
            ax_rgb.axis('off')

            # Bottom row: Valid pixel mask
            ax_mask = axes[1, idx]
            with rasterio.open(result['filepath']) as src:
                data = src.read()
                valid_mask = ~np.isnan(data).any(axis=0) & (data > 0).any(axis=0)

            ax_mask.imshow(valid_mask, cmap='RdYlGn', vmin=0, vmax=1)
            ax_mask.set_title(f"Valid Pixel Mask\n{result['valid_pixels']:,} pixels",
                            fontsize=10)
            ax_mask.axis('off')

        plt.suptitle('Cloud Removal Strategy Comparison - Test Area\n' +
                     'Top: RGB | Bottom: Valid Pixels (Green=Valid, Red=NaN/Cloud)',
                     fontsize=14, fontweight='bold', y=0.98)

        plt.tight_layout()

        comparison_file = f"{OUTPUT_DIR}/strategy_comparison.png"
        plt.savefig(comparison_file, dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"✅ Saved comparison: {comparison_file}")

        # Print summary table
        print(f"\n{'='*80}")
        print("RESULTS SUMMARY")
        print(f"{'='*80}")
        print(f"{'Strategy':<20} {'Valid %':<12} {'Valid Pixels':<15} {'Composite Method':<20}")
        print("-"*80)

        # Sort by valid percentage
        sorted_results = sorted(results.items(), key=lambda x: x[1]['valid_pct'], reverse=True)

        for strategy_name, result in sorted_results:
            print(f"{strategy_name:<20} {result['valid_pct']:>6.1f}%     "
                  f"{result['valid_pixels']:>10,}      "
                  f"{result['strategy']['composite_method']:<20}")

        print("="*80)

        # Recommendation
        best_strategy = sorted_results[0][0]
        best_pct = sorted_results[0][1]['valid_pct']
        current_pct = results.get('current', {}).get('valid_pct', 0)

        print(f"\n🏆 BEST STRATEGY: {best_strategy.upper()}")
        print(f"   Valid pixels: {best_pct:.1f}%")

        if 'current' in results and best_strategy != 'current':
            improvement = best_pct - current_pct
            print(f"   Improvement: +{improvement:.1f}% vs current baseline")

        print(f"\n💡 RECOMMENDATION:")
        print(f"   Use '{best_strategy}' strategy for full province download")
        print(f"   Edit scripts/download_sentinel2.py line 53:")
        print(f"   'cloud_removal_strategy': '{best_strategy}',")

        print(f"\n📁 OUTPUT:")
        print(f"   Comparison: {comparison_file}")
        print(f"   Test tiles: {OUTPUT_DIR}/test_*.tif")

    else:
        print("\n❌ No results to compare")

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()
