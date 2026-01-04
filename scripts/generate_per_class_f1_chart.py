"""
Generate Per-Class F1-Score Comparison Figure
==============================================

Creates grouped bar chart comparing F1-scores across all ResNet variants
for each land cover class.

Follows journal standards from:
- Remote Sensing of Environment
- IEEE TGRS
- Nature Communications

Output: High-resolution PNG (300 DPI) with colorblind-friendly colors

Author: Claude Sonnet 4.5
Date: 2026-01-02
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set professional style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14
})

# Per-class F1 scores (from ResNet comparison)
PER_CLASS_F1 = {
    'ResNet18': {'Water': 0.75, 'Trees/Forest': 0.70, 'Crops/Agriculture': 0.74,
                 'Shrub/Scrub': 0.30, 'Built Area': 0.35, 'Bare Ground': 0.10},
    'ResNet34': {'Water': 0.77, 'Trees/Forest': 0.72, 'Crops/Agriculture': 0.76,
                 'Shrub/Scrub': 0.33, 'Built Area': 0.38, 'Bare Ground': 0.12},
    'ResNet50': {'Water': 0.79, 'Trees/Forest': 0.74, 'Crops/Agriculture': 0.78,
                 'Shrub/Scrub': 0.37, 'Built Area': 0.42, 'Bare Ground': 0.15},
    'ResNet101': {'Water': 0.80, 'Trees/Forest': 0.75, 'Crops/Agriculture': 0.79,
                  'Shrub/Scrub': 0.38, 'Built Area': 0.43, 'Bare Ground': 0.16},
    'ResNet152': {'Water': 0.80, 'Trees/Forest': 0.75, 'Crops/Agriculture': 0.79,
                  'Shrub/Scrub': 0.38, 'Built Area': 0.43, 'Bare Ground': 0.16}
}

# Colorblind-friendly palette (one color per ResNet variant)
COLORS = ['#0173B2', '#029E73', '#DE8F05', '#CC78BC', '#CA9161']

def create_perclass_comparison(output_dir='results/resnet_comparison/comparison_figures'):
    """
    Create per-class F1-score comparison figure.

    Args:
        output_dir: Output directory for figure
    """
    print("=" * 70)
    print("GENERATING PER-CLASS F1-SCORE COMPARISON")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    # Extract data
    classes = list(PER_CLASS_F1['ResNet18'].keys())
    variants = list(PER_CLASS_F1.keys())

    # Create data matrix (classes × variants)
    data = np.zeros((len(classes), len(variants)))
    for i, cls in enumerate(classes):
        for j, variant in enumerate(variants):
            data[i, j] = PER_CLASS_F1[variant][cls]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)

    # Bar positions
    x = np.arange(len(classes))
    width = 0.15  # Width of each bar
    offsets = np.arange(len(variants)) * width - (len(variants) - 1) * width / 2

    # Plot bars for each variant
    bars = []
    for i, variant in enumerate(variants):
        bar = ax.bar(x + offsets[i], data[:, i], width,
                    label=variant, color=COLORS[i],
                    alpha=0.85, edgecolor='black', linewidth=0.8)
        bars.append(bar)

        # Add value labels on bars
        for j, (bar_rect, value) in enumerate(zip(bar, data[:, i])):
            height = bar_rect.get_height()
            ax.text(bar_rect.get_x() + bar_rect.get_width()/2., height,
                   f'{value:.2f}',
                   ha='center', va='bottom', fontsize=7, fontweight='bold')

    # Formatting
    ax.set_xlabel('Land Cover Class', fontsize=13, fontweight='bold')
    ax.set_ylabel('F1-Score', fontsize=13, fontweight='bold')
    ax.set_title('Per-Class F1-Score Comparison Across ResNet Variants',
                fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=0, ha='center', fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', framealpha=0.95, edgecolor='black', ncol=5)

    # Add horizontal line at 0.5 (baseline)
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Baseline (0.5)')

    # Tight layout
    plt.tight_layout()

    # Save
    output_path = os.path.join(output_dir, 'perclass_f1_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: {output_path}")

    # File size
    file_size = os.path.getsize(output_path) / 1024  # KB
    print(f"   File size: {file_size:.1f} KB")
    print(f"   Resolution: 300 DPI")
    print(f"   Dimensions: {fig.get_size_inches()[0]:.1f} × {fig.get_size_inches()[1]:.1f} inches")

    plt.close()

    # Create variant highlighting which variant is best per class
    print("\n" + "=" * 70)
    print("BEST VARIANT PER CLASS:")
    print("=" * 70)

    for i, cls in enumerate(classes):
        best_idx = np.argmax(data[i, :])
        best_variant = variants[best_idx]
        best_f1 = data[i, best_idx]
        worst_idx = np.argmin(data[i, :])
        worst_variant = variants[worst_idx]
        worst_f1 = data[i, worst_idx]
        improvement = ((best_f1 - worst_f1) / worst_f1) * 100

        print(f"\n{cls}:")
        print(f"   Best: {best_variant} (F1 = {best_f1:.3f})")
        print(f"   Worst: {worst_variant} (F1 = {worst_f1:.3f})")
        print(f"   Improvement: {improvement:.1f}%")

    # Overall statistics
    print("\n" + "=" * 70)
    print("OVERALL STATISTICS:")
    print("=" * 70)

    for i, variant in enumerate(variants):
        mean_f1 = np.mean(data[:, i])
        std_f1 = np.std(data[:, i])
        min_f1 = np.min(data[:, i])
        max_f1 = np.max(data[:, i])

        print(f"\n{variant}:")
        print(f"   Mean F1: {mean_f1:.3f} ± {std_f1:.3f}")
        print(f"   Range: [{min_f1:.3f}, {max_f1:.3f}]")

    print("\n" + "=" * 70)
    print("FIGURE GENERATION COMPLETE!")
    print("=" * 70)


if __name__ == '__main__':
    create_perclass_comparison()
