#!/usr/bin/env python3
"""
Classification Comparison Visualization
========================================

Professional visualization of ground truth vs prediction results.

Features:
- Colorblind-friendly palettes (Okabe-Ito standard)
- IGBP-inspired land cover colors
- Side-by-side comparison
- Agreement/disagreement overlay
- Confusion matrix heatmap

Color Schemes Based On:
- MODIS IGBP Land Cover standard colors
- Okabe & Ito (2008) Color Universal Design
- Seaborn colorblind-safe palettes

References:
- MODIS Land Cover: https://modis.gsfc.nasa.gov/data/dataprod/mod12.php
- Okabe & Ito CUD: https://jfly.uni-koeln.de/color/
- Seaborn palettes: https://seaborn.pydata.org/tutorial/color_palettes.html

Author: Claude Sonnet 4.5
Date: 2026-01-02
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns

# ============================================================================
# PROFESSIONAL COLOR SCHEMES
# ============================================================================

# OPTION 1: IGBP-Inspired Natural Colors (Intuitive)
# Based on MODIS Land Cover standard
LAND_COVER_COLORS_NATURAL = {
    0: '#0a4f8c',  # Water - Deep Blue
    1: '#0f5e14',  # Trees/Forest - Dark Green
    2: '#8fd18a',  # Grass/Shrub - Light Green (placeholder, not used)
    3: '#d4af37',  # Agriculture - Golden Yellow (placeholder, not used)
    4: '#c6a664',  # Crops/Agriculture - Tan/Wheat
    5: '#b8a587',  # Shrub/Scrub - Light Brown
    6: '#c85a54',  # Built Area - Terracotta Red
    7: '#9c8b7a',  # Bare Ground - Gray Brown
}

# OPTION 2: Okabe-Ito Colorblind-Friendly Palette
# Standard in scientific visualization (8 colors)
OKABE_ITO_PALETTE = {
    0: '#0173b2',  # Water - Blue
    1: '#029e73',  # Trees/Forest - Bluish Green
    2: '#d55e00',  # (unused)
    3: '#cc78bc',  # (unused)
    4: '#ece133',  # Crops - Yellow
    5: '#ca9161',  # Shrub - Reddish Brown
    6: '#de8f05',  # Built Area - Orange
    7: '#949494',  # Bare Ground - Gray
}

# OPTION 3: Vibrant ColorBrewer-Inspired (Journal-style)
# Professional but more saturated
COLORBREWER_VIBRANT = {
    0: '#2166ac',  # Water - Blue
    1: '#1b7837',  # Trees/Forest - Green
    2: '#a6d96a',  # (unused)
    3: '#ffffbf',  # (unused)
    4: '#f4a582',  # Crops - Peach
    5: '#d1e5f0',  # Shrub - Light Blue
    6: '#b2182b',  # Built Area - Red
    7: '#bababa',  # Bare Ground - Gray
}

# Class names
CLASS_NAMES = {
    0: 'Water',
    1: 'Trees/Forest',
    4: 'Crops/Agriculture',
    5: 'Shrub/Scrub',
    6: 'Built Area',
    7: 'Bare Ground'
}

# Default palette selection
DEFAULT_PALETTE = 'natural'  # Options: 'natural', 'okabe_ito', 'vibrant'

# ============================================================================
# COLOR PALETTE FUNCTIONS
# ============================================================================

def get_color_palette(palette_name='natural'):
    """
    Get color palette by name.

    Args:
        palette_name: 'natural', 'okabe_ito', or 'vibrant'

    Returns:
        Dictionary mapping class IDs to hex colors
    """
    palettes = {
        'natural': LAND_COVER_COLORS_NATURAL,
        'okabe_ito': OKABE_ITO_PALETTE,
        'vibrant': COLORBREWER_VIBRANT
    }

    if palette_name not in palettes:
        print(f"Warning: Unknown palette '{palette_name}', using 'natural'")
        palette_name = 'natural'

    return palettes[palette_name]

def create_legend_patches(palette, class_ids):
    """
    Create legend patches for land cover classes.

    Args:
        palette: Color palette dictionary
        class_ids: List of class IDs to include

    Returns:
        List of matplotlib patches for legend
    """
    patches = []
    for class_id in sorted(class_ids):
        if class_id in CLASS_NAMES:
            patches.append(
                mpatches.Patch(
                    color=palette[class_id],
                    label=f'{CLASS_NAMES[class_id]} ({class_id})'
                )
            )
    return patches

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def visualize_side_by_side(ground_truth, prediction, palette_name='natural',
                           title_left='Ground Truth (KLHK)',
                           title_right='Prediction (ResNet)',
                           output_path=None):
    """
    Create side-by-side comparison visualization.

    Args:
        ground_truth: 2D array of ground truth labels
        prediction: 2D array of predicted labels
        palette_name: Color palette to use
        title_left: Title for left panel
        title_right: Title for right panel
        output_path: Path to save figure (optional)

    Returns:
        Figure object
    """
    palette = get_color_palette(palette_name)

    # Get unique classes
    unique_classes = sorted(np.unique(np.concatenate([
        ground_truth[ground_truth >= 0],
        prediction[prediction >= 0]
    ])))

    # Create colormap
    colors = [palette[c] for c in unique_classes]
    cmap = ListedColormap(colors)
    bounds = list(unique_classes) + [max(unique_classes) + 1]
    norm = BoundaryNorm(bounds, cmap.N)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Left: Ground Truth
    im1 = axes[0].imshow(ground_truth, cmap=cmap, norm=norm, interpolation='nearest')
    axes[0].set_title(title_left, fontsize=16, fontweight='bold')
    axes[0].set_xlabel('Pixel X', fontsize=12)
    axes[0].set_ylabel('Pixel Y', fontsize=12)
    axes[0].grid(False)

    # Right: Prediction
    im2 = axes[1].imshow(prediction, cmap=cmap, norm=norm, interpolation='nearest')
    axes[1].set_title(title_right, fontsize=16, fontweight='bold')
    axes[1].set_xlabel('Pixel X', fontsize=12)
    axes[1].set_ylabel('Pixel Y', fontsize=12)
    axes[1].grid(False)

    # Shared legend
    legend_patches = create_legend_patches(palette, unique_classes)
    fig.legend(handles=legend_patches, loc='center', bbox_to_anchor=(0.5, 0.02),
               ncol=len(unique_classes), fontsize=11, frameon=True)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")

    return fig

def visualize_agreement_map(ground_truth, prediction, output_path=None):
    """
    Create agreement/disagreement overlay map.

    Shows where predictions match (green) vs mismatch (red) ground truth.

    Args:
        ground_truth: 2D array of ground truth labels
        prediction: 2D array of predicted labels
        output_path: Path to save figure (optional)

    Returns:
        Figure object
    """
    # Calculate agreement (excluding invalid pixels)
    valid_mask = (ground_truth >= 0) & (prediction >= 0)
    agreement = np.full_like(ground_truth, -1, dtype=int)
    agreement[valid_mask] = (ground_truth[valid_mask] == prediction[valid_mask]).astype(int)

    # Create colormap: Red (disagree), Green (agree), Black (invalid)
    colors = ['black', '#d73027', '#1a9850']  # Black, Red, Green
    cmap = ListedColormap(colors)
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = BoundaryNorm(bounds, cmap.N)

    # Calculate accuracy
    total_valid = np.sum(valid_mask)
    total_correct = np.sum(ground_truth[valid_mask] == prediction[valid_mask])
    accuracy = (total_correct / total_valid * 100) if total_valid > 0 else 0

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    im = ax.imshow(agreement, cmap=cmap, norm=norm, interpolation='nearest')
    ax.set_title(f'Classification Agreement Map\nOverall Accuracy: {accuracy:.2f}%',
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Pixel X', fontsize=12)
    ax.set_ylabel('Pixel Y', fontsize=12)
    ax.grid(False)

    # Legend
    legend_elements = [
        mpatches.Patch(color='#1a9850', label=f'Correct ({total_correct:,} pixels)'),
        mpatches.Patch(color='#d73027', label=f'Incorrect ({total_valid - total_correct:,} pixels)'),
        mpatches.Patch(color='black', label='No Data')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11, frameon=True)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")

    return fig

def visualize_three_panel(ground_truth, prediction, palette_name='natural',
                         output_path=None):
    """
    Create three-panel comparison: Ground Truth | Prediction | Agreement.

    Args:
        ground_truth: 2D array of ground truth labels
        prediction: 2D array of predicted labels
        palette_name: Color palette to use
        output_path: Path to save figure (optional)

    Returns:
        Figure object
    """
    palette = get_color_palette(palette_name)

    # Get unique classes
    unique_classes = sorted(np.unique(np.concatenate([
        ground_truth[ground_truth >= 0],
        prediction[prediction >= 0]
    ])))

    # Create colormap for classes
    colors = [palette[c] for c in unique_classes]
    cmap_classes = ListedColormap(colors)
    bounds = list(unique_classes) + [max(unique_classes) + 1]
    norm_classes = BoundaryNorm(bounds, cmap_classes.N)

    # Calculate agreement
    valid_mask = (ground_truth >= 0) & (prediction >= 0)
    agreement = np.full_like(ground_truth, -1, dtype=int)
    agreement[valid_mask] = (ground_truth[valid_mask] == prediction[valid_mask]).astype(int)

    # Agreement colormap
    colors_agree = ['black', '#d73027', '#1a9850']
    cmap_agree = ListedColormap(colors_agree)
    bounds_agree = [-1.5, -0.5, 0.5, 1.5]
    norm_agree = BoundaryNorm(bounds_agree, cmap_agree.N)

    # Calculate accuracy
    total_valid = np.sum(valid_mask)
    total_correct = np.sum(ground_truth[valid_mask] == prediction[valid_mask])
    accuracy = (total_correct / total_valid * 100) if total_valid > 0 else 0

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(28, 9))

    # Panel 1: Ground Truth
    axes[0].imshow(ground_truth, cmap=cmap_classes, norm=norm_classes, interpolation='nearest')
    axes[0].set_title('Ground Truth (KLHK)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Pixel X', fontsize=11)
    axes[0].set_ylabel('Pixel Y', fontsize=11)
    axes[0].grid(False)

    # Panel 2: Prediction
    axes[1].imshow(prediction, cmap=cmap_classes, norm=norm_classes, interpolation='nearest')
    axes[1].set_title('Prediction (ResNet)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Pixel X', fontsize=11)
    axes[1].set_ylabel('Pixel Y', fontsize=11)
    axes[1].grid(False)

    # Panel 3: Agreement
    axes[2].imshow(agreement, cmap=cmap_agree, norm=norm_agree, interpolation='nearest')
    axes[2].set_title(f'Agreement Map\nAccuracy: {accuracy:.2f}%',
                     fontsize=14, fontweight='bold', color='darkgreen')
    axes[2].set_xlabel('Pixel X', fontsize=11)
    axes[2].set_ylabel('Pixel Y', fontsize=11)
    axes[2].grid(False)

    # Legends
    # Class legend (panels 1 & 2)
    legend_patches_classes = create_legend_patches(palette, unique_classes)
    fig.legend(handles=legend_patches_classes, loc='lower left',
               bbox_to_anchor=(0.05, 0.02), ncol=3, fontsize=10,
               frameon=True, title='Land Cover Classes')

    # Agreement legend (panel 3)
    legend_agree = [
        mpatches.Patch(color='#1a9850', label='Correct'),
        mpatches.Patch(color='#d73027', label='Incorrect')
    ]
    fig.legend(handles=legend_agree, loc='lower right',
               bbox_to_anchor=(0.95, 0.02), ncol=2, fontsize=10,
               frameon=True, title='Agreement')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")

    return fig

# ============================================================================
# EXAMPLE / TEST
# ============================================================================

def create_sample_data(size=(500, 500)):
    """Create sample ground truth and prediction for testing."""

    # Generate synthetic data
    np.random.seed(42)

    # Ground truth with spatial structure
    ground_truth = np.zeros(size, dtype=int)

    # Water (class 0) - bottom
    ground_truth[400:, :] = 0

    # Forest (class 1) - top left
    ground_truth[:200, :250] = 1

    # Crops (class 4) - top right
    ground_truth[:200, 250:] = 4

    # Built Area (class 6) - center
    ground_truth[200:400, 150:350] = 6

    # Shrub (class 5) - scattered
    mask = np.random.rand(*size) < 0.1
    ground_truth[mask & (ground_truth == 1)] = 5

    # Bare ground (class 7) - small patches
    ground_truth[250:280, 380:420] = 7

    # Prediction (similar but with errors)
    prediction = ground_truth.copy()

    # Add some misclassifications
    errors = np.random.rand(*size) < 0.15  # 15% error rate
    prediction[errors] = np.random.choice([0, 1, 4, 5, 6, 7], size=np.sum(errors))

    return ground_truth, prediction

def demo_all_palettes():
    """Demonstrate all color palettes."""

    print("\n" + "="*80)
    print("COLOR PALETTE DEMONSTRATION")
    print("="*80)

    # Create sample data
    gt, pred = create_sample_data()

    output_dir = 'results/visualizations'
    os.makedirs(output_dir, exist_ok=True)

    palettes = ['natural', 'okabe_ito', 'vibrant']

    for palette in palettes:
        print(f"\nCreating visualization with '{palette}' palette...")

        output_path = os.path.join(output_dir,
                                  f'comparison_demo_{palette}.png')
        visualize_side_by_side(gt, pred, palette_name=palette,
                              output_path=output_path)

    # Three-panel with best palette
    print(f"\nCreating three-panel comparison...")
    output_path = os.path.join(output_dir, 'comparison_threepanel_demo.png')
    visualize_three_panel(gt, pred, palette_name='natural',
                         output_path=output_path)

    # Agreement map
    print(f"\nCreating agreement map...")
    output_path = os.path.join(output_dir, 'agreement_map_demo.png')
    visualize_agreement_map(gt, pred, output_path=output_path)

    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)
    print(f"\nCheck: {output_dir}/")

if __name__ == '__main__':
    demo_all_palettes()
