#!/usr/bin/env python3
"""
Generate Publication-Quality ResNet Comparison
===============================================

Creates comprehensive comparison visualizations and tables for all ResNet variants.

Outputs:
    - Performance comparison table (LaTeX + PNG)
    - Confusion matrices (all 4 models)
    - Per-class F1 comparison
    - Training curves comparison

Author: Claude Sonnet 4.5
Date: 2026-01-04
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import pandas as pd

# ============================================================================
# CONFIGURATION
# ============================================================================

VARIANTS = ['resnet18', 'resnet34', 'resnet101', 'resnet152']
CLASS_NAMES = ['Water', 'Trees', 'Crops', 'Shrub', 'Built', 'Bare']
OUTPUT_DIR = 'results/publication_comparison'

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n" + "="*80)
print("PUBLICATION-QUALITY RESNET COMPARISON")
print("="*80)

# ============================================================================
# LOAD ALL RESULTS
# ============================================================================

print("\n" + "-"*80)
print("Loading Results from All Variants")
print("-"*80)

all_results = {}
all_training = {}

for variant in VARIANTS:
    test_path = f'results/{variant}/test_results.npz'
    train_path = f'results/{variant}/training_history.npz'

    if os.path.exists(test_path):
        data = np.load(test_path)
        all_results[variant] = {
            'predictions': data['predictions'],
            'targets': data['targets'],
            'accuracy': float(data['accuracy']),
            'f1_macro': float(data['f1_macro']),
            'f1_weighted': float(data['f1_weighted'])
        }
        print(f"✓ {variant}: Acc={data['accuracy']:.4f}, F1={data['f1_macro']:.4f}")
    else:
        print(f"⚠️  {variant}: test_results.npz not found")

    if os.path.exists(train_path):
        all_training[variant] = np.load(train_path)

# ============================================================================
# 1. PERFORMANCE COMPARISON TABLE
# ============================================================================

print("\n" + "-"*80)
print("1/6: Generating Performance Comparison Table")
print("-"*80)

# Create DataFrame
table_data = []
for variant in VARIANTS:
    if variant in all_results:
        r = all_results[variant]
        table_data.append({
            'Model': variant.upper(),
            'Accuracy (%)': f"{r['accuracy']*100:.2f}",
            'F1-Macro': f"{r['f1_macro']:.4f}",
            'F1-Weighted': f"{r['f1_weighted']:.4f}"
        })

df = pd.DataFrame(table_data)

# Save as CSV
csv_path = os.path.join(OUTPUT_DIR, 'performance_table.csv')
df.to_csv(csv_path, index=False)
print(f"✓ Saved CSV: {csv_path}")

# Save as LaTeX
latex_path = os.path.join(OUTPUT_DIR, 'performance_table.tex')
with open(latex_path, 'w') as f:
    f.write("\\begin{table}[h]\n")
    f.write("\\centering\n")
    f.write("\\caption{Performance Comparison of ResNet Architectures}\n")
    f.write("\\label{tab:resnet_comparison}\n")
    f.write("\\begin{tabular}{lccc}\n")
    f.write("\\hline\n")
    f.write("Model & Accuracy (\\%) & F1-Macro & F1-Weighted \\\\\n")
    f.write("\\hline\n")
    for _, row in df.iterrows():
        f.write(f"{row['Model']} & {row['Accuracy (%)']} & {row['F1-Macro']} & {row['F1-Weighted']} \\\\\n")
    f.write("\\hline\n")
    f.write("\\end{tabular}\n")
    f.write("\\end{table}\n")
print(f"✓ Saved LaTeX: {latex_path}")

# Create visual table
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('tight')
ax.axis('off')

table = ax.table(cellText=df.values, colLabels=df.columns,
                cellLoc='center', loc='center',
                colWidths=[0.3, 0.25, 0.25, 0.25])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2)

# Style header
for i in range(len(df.columns)):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Highlight best values
best_acc_idx = df['Accuracy (%)'].astype(float).idxmax()
best_f1m_idx = df['F1-Macro'].astype(float).idxmax()
best_f1w_idx = df['F1-Weighted'].astype(float).idxmax()

table[(best_acc_idx+1, 1)].set_facecolor('#C6E0B4')
table[(best_f1m_idx+1, 2)].set_facecolor('#C6E0B4')
table[(best_f1w_idx+1, 3)].set_facecolor('#C6E0B4')

plt.title('ResNet Architecture Performance Comparison', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
table_img_path = os.path.join(OUTPUT_DIR, 'performance_table.png')
plt.savefig(table_img_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"✓ Saved image: {table_img_path}")

# ============================================================================
# 2. CONFUSION MATRICES (All 4 models)
# ============================================================================

print("\n" + "-"*80)
print("2/6: Generating Confusion Matrices")
print("-"*80)

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.flatten()

for idx, variant in enumerate(VARIANTS):
    if variant not in all_results:
        continue

    r = all_results[variant]
    cm = confusion_matrix(r['targets'], r['predictions'])
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    ax = axes[idx]
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=ax, cbar_kws={'label': 'Proportion'})
    ax.set_title(f'{variant.upper()} - Acc: {r["accuracy"]:.2%}, F1: {r["f1_macro"]:.4f}',
                fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=11)
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.tick_params(labelsize=10)

plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, 'confusion_matrices_all.png')
plt.savefig(cm_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"✓ Saved: {cm_path}")

# ============================================================================
# 3. PER-CLASS F1 SCORES COMPARISON
# ============================================================================

print("\n" + "-"*80)
print("3/6: Generating Per-Class F1 Comparison")
print("-"*80)

# Calculate per-class F1 for each model
per_class_f1 = {}
for variant in VARIANTS:
    if variant not in all_results:
        continue

    r = all_results[variant]
    report = classification_report(r['targets'], r['predictions'],
                                  target_names=CLASS_NAMES, output_dict=True)
    per_class_f1[variant] = [report[cls]['f1-score'] for cls in CLASS_NAMES]

# Create grouped bar chart
x = np.arange(len(CLASS_NAMES))
width = 0.2
fig, ax = plt.subplots(figsize=(14, 6))

for idx, (variant, f1_scores) in enumerate(per_class_f1.items()):
    offset = width * (idx - 1.5)
    bars = ax.bar(x + offset, f1_scores, width, label=variant.upper())

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=8)

ax.set_xlabel('Land Cover Class', fontsize=12, fontweight='bold')
ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
ax.set_title('Per-Class F1-Score Comparison Across ResNet Architectures',
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(CLASS_NAMES, fontsize=11)
ax.legend(fontsize=10, loc='upper right')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, 1.0)

plt.tight_layout()
f1_comp_path = os.path.join(OUTPUT_DIR, 'per_class_f1_comparison.png')
plt.savefig(f1_comp_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"✓ Saved: {f1_comp_path}")

# ============================================================================
# 4. OVERALL METRICS BAR CHART
# ============================================================================

print("\n" + "-"*80)
print("4/6: Generating Overall Metrics Comparison")
print("-"*80)

metrics = ['Accuracy', 'F1-Macro', 'F1-Weighted']
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(VARIANTS))
width = 0.25

for idx, metric in enumerate(metrics):
    if metric == 'Accuracy':
        values = [all_results[v]['accuracy'] for v in VARIANTS if v in all_results]
    elif metric == 'F1-Macro':
        values = [all_results[v]['f1_macro'] for v in VARIANTS if v in all_results]
    else:
        values = [all_results[v]['f1_weighted'] for v in VARIANTS if v in all_results]

    offset = width * (idx - 1)
    bars = ax.bar(x + offset, values, width, label=metric)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

ax.set_xlabel('Model Architecture', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Overall Performance Metrics Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([v.upper() for v in VARIANTS if v in all_results], fontsize=11)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, 1.0)

plt.tight_layout()
overall_path = os.path.join(OUTPUT_DIR, 'overall_metrics_comparison.png')
plt.savefig(overall_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"✓ Saved: {overall_path}")

# ============================================================================
# 5. TRAINING CURVES COMPARISON
# ============================================================================

print("\n" + "-"*80)
print("5/6: Generating Training Curves Comparison")
print("-"*80)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for idx, variant in enumerate(VARIANTS):
    if variant not in all_training:
        continue

    train_data = all_training[variant]
    epochs = range(1, len(train_data['train_loss']) + 1)

    # Loss curves
    ax1.plot(epochs, train_data['train_loss'],
            label=f'{variant.upper()} Train',
            color=colors[idx], linestyle='-', linewidth=2, alpha=0.8)
    ax1.plot(epochs, train_data['val_loss'],
            label=f'{variant.upper()} Val',
            color=colors[idx], linestyle='--', linewidth=2, alpha=0.6)

    # Accuracy curves
    ax2.plot(epochs, train_data['train_acc'],
            label=f'{variant.upper()} Train',
            color=colors[idx], linestyle='-', linewidth=2, alpha=0.8)
    ax2.plot(epochs, train_data['val_acc'],
            label=f'{variant.upper()} Val',
            color=colors[idx], linestyle='--', linewidth=2, alpha=0.6)

ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax1.set_title('Training and Validation Loss', fontsize=13, fontweight='bold')
ax1.legend(fontsize=9, loc='upper right')
ax1.grid(alpha=0.3)

ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax2.set_title('Training and Validation Accuracy', fontsize=13, fontweight='bold')
ax2.legend(fontsize=9, loc='lower right')
ax2.grid(alpha=0.3)

plt.tight_layout()
curves_path = os.path.join(OUTPUT_DIR, 'training_curves_comparison.png')
plt.savefig(curves_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"✓ Saved: {curves_path}")

# ============================================================================
# 6. PER-CLASS PERFORMANCE TABLE
# ============================================================================

print("\n" + "-"*80)
print("6/6: Generating Per-Class Performance Table")
print("-"*80)

# Create detailed per-class table
class_table_data = []
for variant in VARIANTS:
    if variant not in all_results:
        continue

    r = all_results[variant]
    report = classification_report(r['targets'], r['predictions'],
                                  target_names=CLASS_NAMES, output_dict=True)

    for cls in CLASS_NAMES:
        class_table_data.append({
            'Model': variant.upper(),
            'Class': cls,
            'Precision': f"{report[cls]['precision']:.3f}",
            'Recall': f"{report[cls]['recall']:.3f}",
            'F1-Score': f"{report[cls]['f1-score']:.3f}",
            'Support': int(report[cls]['support'])
        })

df_class = pd.DataFrame(class_table_data)

# Save as CSV
class_csv_path = os.path.join(OUTPUT_DIR, 'per_class_performance.csv')
df_class.to_csv(class_csv_path, index=False)
print(f"✓ Saved: {class_csv_path}")

# Create pivot table for better visualization
df_pivot = df_class.pivot_table(index='Class', columns='Model',
                                 values='F1-Score', aggfunc='first')

# Save pivot table
pivot_path = os.path.join(OUTPUT_DIR, 'per_class_f1_pivot.csv')
df_pivot.to_csv(pivot_path)
print(f"✓ Saved pivot table: {pivot_path}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("PUBLICATION COMPARISON COMPLETE!")
print("="*80)

print(f"\n📁 All files saved to: {OUTPUT_DIR}/")
print("\n✓ Generated files:")
print("  1. performance_table.csv - Overall metrics (CSV)")
print("  2. performance_table.tex - LaTeX table")
print("  3. performance_table.png - Visual table")
print("  4. confusion_matrices_all.png - 2x2 grid of confusion matrices")
print("  5. per_class_f1_comparison.png - Grouped bar chart")
print("  6. overall_metrics_comparison.png - Overall metrics bars")
print("  7. training_curves_comparison.png - Training/validation curves")
print("  8. per_class_performance.csv - Detailed per-class metrics")
print("  9. per_class_f1_pivot.csv - Pivot table of F1 scores")

print("\n✨ All visualizations are publication-ready (300 DPI)!")
print("✨ Tables are formatted for Microsoft Word and LaTeX!")

print("\n" + "="*80)
