"""
Generate Methodology Flowchart for Publication

Creates a clear, professional flowchart showing the research pipeline:
Datasets → Preprocessing → Models → Training → Evaluation → Statistical Analysis

Output: publication/manuscript/figures/methodology_flowchart.pdf
        results/figures/methodology_flowchart.png
"""

import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_DIR

# Output paths
PUB_FIGURES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'publication', 'manuscript', 'figures')
RES_FIGURES = os.path.join(RESULTS_DIR, 'figures')
os.makedirs(PUB_FIGURES, exist_ok=True)
os.makedirs(RES_FIGURES, exist_ok=True)


def draw_box(ax, x, y, w, h, text, color='#EBF5FB', edgecolor='#2980B9',
             fontsize=8, fontweight='bold', text_color='black', linewidth=1.5,
             subtext=None, subsize=6.5):
    """Draw a rounded box with text."""
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.02",
                         facecolor=color, edgecolor=edgecolor,
                         linewidth=linewidth, zorder=2)
    ax.add_patch(box)

    if subtext:
        ax.text(x, y + 0.012, text, ha='center', va='center',
                fontsize=fontsize, fontweight=fontweight, color=text_color,
                zorder=3)
        ax.text(x, y - 0.018, subtext, ha='center', va='center',
                fontsize=subsize, color='#555555', zorder=3,
                style='italic')
    else:
        ax.text(x, y, text, ha='center', va='center',
                fontsize=fontsize, fontweight=fontweight, color=text_color,
                zorder=3)


def draw_arrow(ax, x1, y1, x2, y2, color='#2C3E50'):
    """Draw an arrow between two points."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=1.5, connectionstyle='arc3,rad=0'),
                zorder=1)


def draw_arrow_curved(ax, x1, y1, x2, y2, color='#2C3E50', rad=0.2):
    """Draw a curved arrow."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=1.2, connectionstyle=f'arc3,rad={rad}'),
                zorder=1)


def generate_flowchart():
    print("=" * 60)
    print("Generating Methodology Flowchart")
    print("=" * 60)

    fig, ax = plt.subplots(1, 1, figsize=(7.16, 8.5))  # IEEE column width
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Color scheme
    C_HEADER = '#1A5276'     # Dark blue headers
    C_DATA = '#D5F5E3'       # Green for data
    C_DATA_E = '#27AE60'
    C_PROC = '#EBF5FB'       # Light blue for processing
    C_PROC_E = '#2980B9'
    C_MODEL = '#FDEBD0'      # Orange for models
    C_MODEL_E = '#E67E22'
    C_EVAL = '#F5EEF8'       # Purple for evaluation
    C_EVAL_E = '#8E44AD'
    C_RESULT = '#FDEDEC'     # Red for results
    C_RESULT_E = '#E74C3C'

    bw = 0.38   # box width
    bh = 0.050  # box height
    bh_tall = 0.065

    # ============================================================
    # ROW 1: DATASETS (top)
    # ============================================================
    y_top = 0.95
    ax.text(0.5, y_top, 'Research Methodology Overview',
            ha='center', va='center', fontsize=11, fontweight='bold',
            color=C_HEADER)

    y1 = 0.88
    draw_box(ax, 0.27, y1, bw, bh_tall,
             'EuroSAT Dataset',
             subtext='27,000 images | 10 classes | 64x64 px\nSentinel-2 satellite (10m)',
             color=C_DATA, edgecolor=C_DATA_E, fontsize=8)

    draw_box(ax, 0.73, y1, bw, bh_tall,
             'UC Merced Dataset',
             subtext='2,100 images | 21 classes | 256x256 px\nUSGS aerial (0.3m)',
             color=C_DATA, edgecolor=C_DATA_E, fontsize=8)

    # ============================================================
    # ROW 2: PREPROCESSING
    # ============================================================
    y2 = 0.77
    draw_box(ax, 0.5, y2, 0.80, bh_tall,
             'Data Preprocessing',
             subtext='Resize to 224x224 | Stratified 80/20 split (seed=42) | RGB channels only',
             color=C_PROC, edgecolor=C_PROC_E, fontsize=8)

    draw_arrow(ax, 0.27, y1 - bh_tall/2, 0.40, y2 + bh_tall/2)
    draw_arrow(ax, 0.73, y1 - bh_tall/2, 0.60, y2 + bh_tall/2)

    # ============================================================
    # ROW 3: AUGMENTATION
    # ============================================================
    y3 = 0.67
    draw_box(ax, 0.5, y3, 0.80, bh_tall,
             'Data Augmentation',
             subtext='Random H/V flip | Rotation (+-15) | Color jitter | ImageNet normalization',
             color=C_PROC, edgecolor=C_PROC_E, fontsize=8)

    draw_arrow(ax, 0.5, y2 - bh_tall/2, 0.5, y3 + bh_tall/2)

    # ============================================================
    # ROW 4: MODELS (3 families side by side)
    # ============================================================
    y4 = 0.545
    y4_label = 0.59
    ax.text(0.5, y4_label, '8 Model Architectures (3 Families)',
            ha='center', va='center', fontsize=8.5, fontweight='bold',
            color=C_HEADER, style='italic')

    # CNN family
    draw_box(ax, 0.18, y4, 0.30, 0.07,
             'Classical CNN',
             subtext='ResNet-50 (23.5M)\nResNet-101 (42.5M)\nDenseNet-121 (7.0M)\nEfficientNet-B0 (4.0M)\nEfficientNet-B3 (10.7M)',
             color=C_MODEL, edgecolor=C_MODEL_E, fontsize=7.5, subsize=5.8)

    # Transformer family
    draw_box(ax, 0.52, y4, 0.26, 0.07,
             'Vision Transformer',
             subtext='ViT-B/16 (85.8M)\nSwin-T (27.5M)',
             color=C_MODEL, edgecolor=C_MODEL_E, fontsize=7.5, subsize=5.8)

    # Modern CNN
    draw_box(ax, 0.83, y4, 0.26, 0.07,
             'Modern CNN',
             subtext='ConvNeXt-Tiny (27.8M)',
             color=C_MODEL, edgecolor=C_MODEL_E, fontsize=7.5, subsize=5.8)

    draw_arrow(ax, 0.5, y3 - bh_tall/2, 0.5, y4_label + 0.012)

    # ============================================================
    # ROW 5: TRAINING PROTOCOL
    # ============================================================
    y5 = 0.44
    draw_box(ax, 0.5, y5, 0.80, bh_tall,
             'Uniform Training Protocol',
             subtext='ImageNet pretrained | AdamW (lr=1e-4) | ReduceLROnPlateau | Early stopping (patience=10) | 30 epochs',
             color=C_PROC, edgecolor=C_PROC_E, fontsize=8)

    # Arrows from 3 model boxes down to training
    draw_arrow(ax, 0.18, y4 - 0.07/2, 0.35, y5 + bh_tall/2)
    draw_arrow(ax, 0.52, y4 - 0.07/2, 0.50, y5 + bh_tall/2)
    draw_arrow(ax, 0.83, y4 - 0.07/2, 0.65, y5 + bh_tall/2)

    # ============================================================
    # ROW 6: EVALUATION METRICS
    # ============================================================
    y6 = 0.34
    draw_box(ax, 0.5, y6, 0.80, bh_tall,
             'Performance Evaluation',
             subtext='Overall Accuracy | F1-Macro | F1-Weighted | Cohen\'s Kappa | Per-class Precision/Recall/F1',
             color=C_EVAL, edgecolor=C_EVAL_E, fontsize=8)

    draw_arrow(ax, 0.5, y5 - bh_tall/2, 0.5, y6 + bh_tall/2)

    # ============================================================
    # ROW 7: ANALYSIS (3 boxes)
    # ============================================================
    y7 = 0.23
    y7_label = 0.275
    ax.text(0.5, y7_label, 'Comparative Analysis',
            ha='center', va='center', fontsize=8.5, fontweight='bold',
            color=C_HEADER, style='italic')

    draw_box(ax, 0.18, y7, 0.30, bh,
             'Statistical Testing',
             subtext="McNemar's test (pairwise)",
             color=C_EVAL, edgecolor=C_EVAL_E, fontsize=7.5, subsize=6)

    draw_box(ax, 0.52, y7, 0.28, bh,
             'Error Analysis',
             subtext='Confusion matrices\nMisclassified samples',
             color=C_EVAL, edgecolor=C_EVAL_E, fontsize=7.5, subsize=6)

    draw_box(ax, 0.83, y7, 0.26, bh,
             'Efficiency Analysis',
             subtext='Params vs. accuracy\nTraining time',
             color=C_EVAL, edgecolor=C_EVAL_E, fontsize=7.5, subsize=6)

    draw_arrow(ax, 0.5, y6 - bh_tall/2, 0.5, y7_label + 0.012)

    # ============================================================
    # ROW 8: RESULTS / CONCLUSIONS
    # ============================================================
    y8 = 0.12
    draw_box(ax, 0.5, y8, 0.80, 0.07,
             'Key Findings',
             subtext='Architecture matters less than training recipe | ConvNeXt-T best on EuroSAT (99.06%)\n'
                     'EfficientNet-B3 best on UC Merced (99.76%) | Most differences not statistically significant\n'
                     'EfficientNet-B0 (4.0M params) within 1% of all models',
             color=C_RESULT, edgecolor=C_RESULT_E, fontsize=8, subsize=6)

    draw_arrow(ax, 0.18, y7 - bh/2, 0.35, y8 + 0.07/2)
    draw_arrow(ax, 0.52, y7 - bh/2, 0.50, y8 + 0.07/2)
    draw_arrow(ax, 0.83, y7 - bh/2, 0.65, y8 + 0.07/2)

    # ============================================================
    # PHASE LABELS on the left side
    # ============================================================
    phase_x = -0.02
    phases = [
        (y1, 'Phase 1:\nData'),
        ((y2 + y3) / 2, 'Phase 2:\nPreprocessing'),
        ((y4_label + y5) / 2, 'Phase 3:\nModeling'),
        ((y6 + y7) / 2, 'Phase 4:\nEvaluation'),
        (y8, 'Phase 5:\nFindings'),
    ]
    for py, ptxt in phases:
        ax.text(phase_x, py, ptxt, ha='center', va='center',
                fontsize=6, fontweight='bold', color='#7F8C8D',
                rotation=0, style='italic')

    plt.tight_layout(pad=0.5)

    # Save as PDF (for LaTeX) and PNG (for preview)
    pdf_path = os.path.join(PUB_FIGURES, 'methodology_flowchart.pdf')
    png_path = os.path.join(RES_FIGURES, 'methodology_flowchart.png')

    fig.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {pdf_path}")

    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {png_path}")

    plt.close(fig)


if __name__ == '__main__':
    generate_flowchart()
    print()
    print("=" * 60)
    print("Flowchart generated successfully!")
    print("=" * 60)
