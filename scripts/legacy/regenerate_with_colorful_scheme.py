#!/usr/bin/env python3
"""
Regenerate Classification Comparison with COLORFUL Jambi Color Scheme
======================================================================

Uses the bright, colorful scheme from previous visualizations:
- Light green for crops (dominant 57%)
- Deep pink/magenta for built areas (highly visible)
- Forest green for trees
- Blue for water
- Dark orange for shrub
- Chocolate brown for bare ground

Author: Claude Sonnet 4.5
Date: 2026-01-02
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
import seaborn as sns

# ============================================================================
# COLORFUL JAMBI COLOR SCHEME (Previous bright colors)
# ============================================================================

CLASS_COLORS_JAMBI = {
    0: '#0066CC',  # Water - Bright Blue
    1: '#228B22',  # Trees/Forest - Forest Green
    4: '#90EE90',  # Crops - Light Green (agricultural, dominant)
    5: '#FF8C00',  # Shrub - Dark Orange (high visibility)
    6: '#FF1493',  # Built - Deep Pink/Magenta (HIGHLY VISIBLE)
    7: '#D2691E',  # Bare Ground - Chocolate Brown
    -1: '#FFFFFF', # No data - White
}

CLASS_NAMES = {
    0: 'Water',
    1: 'Trees/Forest',
    4: 'Crops/Agriculture',
    5: 'Shrub/Scrub',
    6: 'Built Area',
    7: 'Bare Ground',
}

# ============================================================================
# VISUALIZATION FUNCTION
# ============================================================================

def create_colorful_comparison(ground_truth_path, prediction_path,
                               output_path, title, verbose=True):
    """
    Create publication figure with COLORFUL Jambi color scheme.

    Args:
        ground_truth_path: Path to KLHK ground truth GeoTIFF
        prediction_path: Path to prediction GeoTIFF
        output_path: Path to save figure
        title: Figure title
    """

    if verbose:
        print(f"\nCreating colorful comparison: {title}")
        print(f"  Ground truth: {ground_truth_path}")
        print(f"  Prediction: {prediction_path}")

    # Load ground truth
    with rasterio.open(ground_truth_path) as src:
        ground_truth = src.read(1)
        profile = src.profile

    # Load prediction
    with rasterio.open(prediction_path) as src:
        prediction = src.read(1)

    # Get unique classes
    unique_classes = sorted(np.unique(np.concatenate([
        ground_truth[ground_truth >= 0],
        prediction[prediction >= 0]
    ])))

    if verbose:
        print(f"  Classes found: {unique_classes}")
        print(f"  Ground truth shape: {ground_truth.shape}")
        print(f"  Prediction shape: {prediction.shape}")

    # Create colormap
    colors = []
    for val in range(-1, 8):  # -1 to 7
        if val in CLASS_COLORS_JAMBI:
            colors.append(CLASS_COLORS_JAMBI[val])
        else:
            colors.append('#FFFFFF')
    cmap = ListedColormap(colors)

    # Calculate agreement
    valid_mask = (ground_truth >= 0) & (prediction >= 0)
    agreement = np.full_like(ground_truth, -1, dtype=int)
    agreement[valid_mask] = (ground_truth[valid_mask] == prediction[valid_mask]).astype(int)

    # Calculate metrics
    total_valid = np.sum(valid_mask)
    total_correct = np.sum(ground_truth[valid_mask] == prediction[valid_mask])
    overall_accuracy = (total_correct / total_valid * 100) if total_valid > 0 else 0

    # Per-class accuracy
    per_class_f1 = {}
    for cls in unique_classes:
        if cls < 0:
            continue
        gt_mask = ground_truth == cls
        pred_mask = prediction == cls

        tp = np.sum(gt_mask & pred_mask & valid_mask)
        fp = np.sum(~gt_mask & pred_mask & valid_mask)
        fn = np.sum(gt_mask & ~pred_mask & valid_mask)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        per_class_f1[cls] = f1

    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    y_true = ground_truth[valid_mask]
    y_pred = prediction[valid_mask]
    cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # ========================================================================
    # CREATE FIGURE - 5 PANELS
    # ========================================================================

    fig = plt.figure(figsize=(24, 14), facecolor='white')
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)

    gs = GridSpec(3, 3, figure=fig, height_ratios=[0.08, 2.5, 1.2], hspace=0.35, wspace=0.3)

    # Shared legend at top
    ax_legend = fig.add_subplot(gs[0, :])
    ax_legend.axis('off')

    legend_elements = []
    for cls in unique_classes:
        if cls >= 0:
            name = CLASS_NAMES.get(cls, f'Class {cls}')
            color = CLASS_COLORS_JAMBI.get(cls, '#FFFFFF')
            legend_elements.append(mpatches.Patch(facecolor=color, edgecolor='black', label=name))

    ax_legend.legend(handles=legend_elements, loc='center', ncol=len(unique_classes),
                    fontsize=13, frameon=True, fancybox=True, shadow=True)

    # Panel 1: Ground Truth
    ax1 = fig.add_subplot(gs[1, 0])
    im1 = ax1.imshow(ground_truth, cmap=cmap, vmin=-1, vmax=7, interpolation='nearest')
    ax1.set_title('(a) Ground Truth (KLHK)\nReference Data', fontsize=14, fontweight='bold', pad=10)
    ax1.set_xlabel('Pixel X', fontsize=11)
    ax1.set_ylabel('Pixel Y', fontsize=11)
    ax1.grid(False)

    # Panel 2: Prediction
    ax2 = fig.add_subplot(gs[1, 1])
    im2 = ax2.imshow(prediction, cmap=cmap, vmin=-1, vmax=7, interpolation='nearest')
    ax2.set_title(f'(b) Prediction (Random Forest)\nOverall Accuracy: {overall_accuracy:.2f}%',
                 fontsize=14, fontweight='bold', pad=10)
    ax2.set_xlabel('Pixel X', fontsize=11)
    ax2.set_ylabel('Pixel Y', fontsize=11)
    ax2.grid(False)

    # Panel 3: Agreement Map
    ax3 = fig.add_subplot(gs[1, 2])
    agreement_colors = ['#000000', '#FF4444', '#44FF44']  # Black, Red, Green
    agreement_cmap = ListedColormap(agreement_colors)
    im3 = ax3.imshow(agreement, cmap=agreement_cmap, vmin=-1, vmax=1, interpolation='nearest')
    ax3.set_title(f'(c) Agreement Map\nCorrect: {total_correct:,} | Incorrect: {total_valid-total_correct:,}',
                 fontsize=14, fontweight='bold', pad=10, color='darkgreen')
    ax3.set_xlabel('Pixel X', fontsize=11)
    ax3.set_ylabel('Pixel Y', fontsize=11)
    ax3.grid(False)

    # Panel 4: Per-Class F1 Scores
    ax4 = fig.add_subplot(gs[2, :2])
    classes_sorted = sorted(per_class_f1.keys())
    f1_values = [per_class_f1[c] * 100 for c in classes_sorted]
    class_labels = [CLASS_NAMES.get(c, f'Class {c}') for c in classes_sorted]
    colors_bars = [CLASS_COLORS_JAMBI.get(c, '#CCCCCC') for c in classes_sorted]

    bars = ax4.barh(class_labels, f1_values, color=colors_bars, edgecolor='black', linewidth=1.5)
    ax4.axvline(overall_accuracy, color='red', linestyle='--', linewidth=2, label=f'Overall Acc: {overall_accuracy:.1f}%')
    ax4.set_xlabel('F1-Score (%)', fontsize=12, fontweight='bold')
    ax4.set_title('(d) Per-Class Accuracy', fontsize=14, fontweight='bold', pad=10)
    ax4.set_xlim(0, 100)
    ax4.grid(axis='x', alpha=0.3)
    ax4.legend(fontsize=10)

    for bar, val in zip(bars, f1_values):
        width = bar.get_width()
        ax4.text(width + 1, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=10, fontweight='bold')

    # Panel 5: Confusion Matrix
    ax5 = fig.add_subplot(gs[2, 2])
    sns.heatmap(cm_normalized * 100, annot=True, fmt='.1f', cmap='YlOrRd',
                xticklabels=[CLASS_NAMES.get(c, str(c)) for c in unique_classes],
                yticklabels=[CLASS_NAMES.get(c, str(c)) for c in unique_classes],
                cbar_kws={'label': 'Percentage (%)'},
                ax=ax5, linewidths=0.5, linecolor='gray')
    ax5.set_title('(e) Confusion Matrix (%)', fontsize=14, fontweight='bold', pad=10)
    ax5.set_xlabel('Predicted Class', fontsize=11, fontweight='bold')
    ax5.set_ylabel('True Class', fontsize=11, fontweight='bold')
    plt.setp(ax5.get_xticklabels(), rotation=45, ha='right')

    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    if verbose:
        print(f"  ✅ Saved: {output_path}")
        print(f"  Overall Accuracy: {overall_accuracy:.2f}%")
        print(f"  Per-class F1 scores:")
        for cls in classes_sorted:
            print(f"    {CLASS_NAMES.get(cls, f'Class {cls}')}: {per_class_f1[cls]*100:.2f}%")

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Regenerate comparisons with colorful scheme."""

    print("\n" + "="*80)
    print("REGENERATING WITH COLORFUL JAMBI COLOR SCHEME")
    print("="*80)

    # Input paths - using EXISTING classification maps
    classification_dir = 'results/classification_maps'
    output_dir = 'results/classification_maps_colorful'
    os.makedirs(output_dir, exist_ok=True)

    # We need ground truth GeoTIFFs - let me create them from KLHK
    print("\nStep 1: Creating ground truth GeoTIFFs...")

    from modules.data_loader import load_klhk_data, load_sentinel2_tiles
    from modules.preprocessor import rasterize_klhk

    # Province
    print("\n  Creating province ground truth...")
    KLHK_PATH = 'data/klhk/KLHK_PL2024_Jambi_Full_WithGeometry.geojson'
    PROVINCE_TILES = [
        'data/sentinel_new_cloudfree/S2_jambi_2024_20m_AllBands-0000000000-0000000000.tif',
        'data/sentinel_new_cloudfree/S2_jambi_2024_20m_AllBands-0000000000-0000010496.tif',
        'data/sentinel_new_cloudfree/S2_jambi_2024_20m_AllBands-0000010496-0000000000.tif',
        'data/sentinel_new_cloudfree/S2_jambi_2024_20m_AllBands-0000010496-0000010496.tif'
    ]

    klhk_gdf = load_klhk_data(KLHK_PATH, verbose=False)
    sentinel2_bands, s2_profile = load_sentinel2_tiles(PROVINCE_TILES, verbose=False)
    province_gt = rasterize_klhk(klhk_gdf, s2_profile, verbose=False)

    # Save province ground truth
    gt_province_path = os.path.join(output_dir, 'ground_truth_province_20m.tif')
    gt_profile = s2_profile.copy()
    gt_profile.update({'dtype': 'int16', 'count': 1, 'nodata': -1})
    with rasterio.open(gt_province_path, 'w', **gt_profile) as dst:
        dst.write(province_gt, 1)
    print(f"  ✅ Saved: {gt_province_path}")

    # City
    print("\n  Creating city ground truth...")
    CITY_10M_PATH = 'data/sentinel_city/sentinel_city_10m_2024dry_p25.tif'
    with rasterio.open(CITY_10M_PATH) as src:
        city_profile = src.profile
    city_gt = rasterize_klhk(klhk_gdf, city_profile, verbose=False)

    # Save city ground truth
    gt_city_path = os.path.join(output_dir, 'ground_truth_city_10m.tif')
    gt_city_profile = city_profile.copy()
    gt_city_profile.update({'dtype': 'int16', 'count': 1, 'nodata': -1})
    with rasterio.open(gt_city_path, 'w', **gt_city_profile) as dst:
        dst.write(city_gt, 1)
    print(f"  ✅ Saved: {gt_city_path}")

    # Step 2: Create colorful comparisons
    print("\nStep 2: Creating colorful comparison figures...")

    # Province comparison
    create_colorful_comparison(
        ground_truth_path=gt_province_path,
        prediction_path=os.path.join(classification_dir, 'classification_province_20m.tif'),
        output_path=os.path.join(output_dir, 'comparison_province_20m_COLORFUL.png'),
        title='Land Cover Classification - Jambi Province (20m Resolution)',
        verbose=True
    )

    # City comparison
    create_colorful_comparison(
        ground_truth_path=gt_city_path,
        prediction_path=os.path.join(classification_dir, 'classification_city_10m.tif'),
        output_path=os.path.join(output_dir, 'comparison_city_10m_COLORFUL.png'),
        title='Land Cover Classification - Jambi City (10m Resolution)',
        verbose=True
    )

    print("\n" + "="*80)
    print("COLORFUL REGENERATION COMPLETE!")
    print("="*80)
    print(f"\n📁 Output directory: {output_dir}")
    print("\nGenerated files:")
    print("  1. comparison_province_20m_COLORFUL.png")
    print("  2. comparison_city_10m_COLORFUL.png")
    print("\nUsing bright Jambi color scheme:")
    print("  • Light green for crops (dominant)")
    print("  • Deep pink for built areas (highly visible)")
    print("  • Forest green for trees")
    print("  • Bright blue for water")
    print("  • Dark orange for shrub")
    print("  • Chocolate brown for bare ground")
    print("\n" + "="*80)

if __name__ == '__main__':
    main()
