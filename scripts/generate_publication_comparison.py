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
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# ============================================================================
# EXCEL FORMATTING FUNCTION
# ============================================================================

def format_excel_table(file_path, header_row=2):
    """
    Apply beautiful formatting to Excel file with auto-adjusted columns.

    Args:
        file_path: Path to Excel file
        header_row: Row number containing headers (1-indexed)
    """
    wb = load_workbook(file_path)
    ws = wb.active

    # Header styling
    header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    header_font = Font(bold=True, color="FFFFFF", size=11)
    border = Border(
        left=Side(style='thin'),
        right=Side(style='thin'),
        top=Side(style='thin'),
        bottom=Side(style='thin')
    )

    # Apply formatting to all cells
    for row in ws.iter_rows():
        for cell in row:
            cell.border = border
            cell.alignment = Alignment(horizontal='center', vertical='center')

            # Header row
            if cell.row == header_row:
                cell.fill = header_fill
                cell.font = header_font
            # Data rows
            else:
                cell.font = Font(size=10)

            # First column left-aligned and bold
            if cell.column == 1:
                cell.alignment = Alignment(horizontal='left', vertical='center')
                cell.font = Font(bold=True)

    # Auto-adjust column widths
    for column in ws.columns:
        max_length = 0
        column_letter = get_column_letter(column[0].column)
        for cell in column:
            try:
                if cell.value:
                    max_length = max(max_length, len(str(cell.value)))
            except:
                pass
        adjusted_width = min(max_length + 3, 50)
        ws.column_dimensions[column_letter].width = adjusted_width

    # Header row height
    ws.row_dimensions[header_row].height = 25

    wb.save(file_path)
    print(f"✓ Formatted Excel: {os.path.basename(file_path)}")

# ============================================================================
# CONFIGURATION
# ============================================================================

VARIANTS = ['resnet18', 'resnet34', 'resnet101', 'resnet152']
CLASS_NAMES = ['Water', 'Trees', 'Crops', 'Shrub', 'Built', 'Bare']

# New centralized paths
TABLES_DIR = 'results/tables/performance'
FIGURES_CONFUSION_DIR = 'results/figures/confusion_matrices'
FIGURES_TRAINING_DIR = 'results/figures/training_curves'

os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_CONFUSION_DIR, exist_ok=True)
os.makedirs(FIGURES_TRAINING_DIR, exist_ok=True)

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

# Save as Excel with beautiful formatting
xlsx_path = os.path.join(TABLES_DIR, 'performance_table.xlsx')
df.to_excel(xlsx_path, index=False, sheet_name='Performance Comparison')
format_excel_table(xlsx_path, header_row=1)
print(f"✓ Saved Excel: {xlsx_path}")

# Save as LaTeX
latex_path = os.path.join(TABLES_DIR, 'performance_table.tex')
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

# ============================================================================
# 2. CONFUSION MATRICES (All 4 models) - UNIQUE: Shows error patterns
# ============================================================================

print("\n" + "-"*80)
print("2/4: Generating Confusion Matrices (Error Pattern Analysis)")
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
cm_path = os.path.join(FIGURES_CONFUSION_DIR, 'confusion_matrices_all.png')
plt.savefig(cm_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"✓ Saved: {cm_path}")

# ============================================================================
# 3. TRAINING CURVES - UNIQUE: Shows convergence over time
# ============================================================================

print("\n" + "-"*80)
print("3/4: Generating Training Curves (Convergence Analysis)")
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
curves_path = os.path.join(FIGURES_TRAINING_DIR, 'training_curves_comparison.png')
plt.savefig(curves_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"✓ Saved: {curves_path}")

# ============================================================================
# 4. PER-CLASS PERFORMANCE TABLES (Excel)
# ============================================================================

print("\n" + "-"*80)
print("4/4: Generating Per-Class Performance Tables")
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

# Save as Excel with beautiful formatting
class_xlsx_path = os.path.join(TABLES_DIR, 'per_class_performance.xlsx')
df_class.to_excel(class_xlsx_path, index=False, sheet_name='Per-Class Performance')
format_excel_table(class_xlsx_path, header_row=1)
print(f"✓ Saved Excel: {class_xlsx_path}")

# Create pivot table for better visualization
df_pivot = df_class.pivot_table(index='Class', columns='Model',
                                 values='F1-Score', aggfunc='first')

# Save pivot table as Excel
pivot_path = os.path.join(TABLES_DIR, 'per_class_f1_pivot.xlsx')
df_pivot.to_excel(pivot_path, sheet_name='F1-Score Pivot')
format_excel_table(pivot_path, header_row=1)
print(f"✓ Saved Excel pivot table: {pivot_path}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("PUBLICATION COMPARISON COMPLETE!")
print("="*80)

print(f"\n📁 Files saved to centralized structure:")
print(f"  📊 Tables: {TABLES_DIR}/")
print(f"  📈 Figures: results/figures/ (confusion_matrices/, training_curves/)")

print("\n📊 TABLES (for exact numerical values):")
print("  1. performance_table.xlsx - Overall metrics (Accuracy, F1-Macro, F1-Weighted)")
print("  2. performance_table.tex - LaTeX format (for journal submission)")
print("  3. per_class_performance.xlsx - Detailed per-class Precision/Recall/F1")
print("  4. per_class_f1_pivot.xlsx - Quick lookup matrix (Class × Model)")

print("\n📈 FIGURES (for patterns & relationships):")
print("  5. confusion_matrices_all.png - ERROR PATTERNS (which classes confused)")
print("  6. training_curves_comparison.png - CONVERGENCE (learning over time)")

print("\n✨ No redundancy - Tables show exact values, Figures show patterns!")
print("✨ Excel tables: Auto-formatted, beautiful, ready for Microsoft Word!")
print("✨ Figures: Publication-ready 300 DPI, unique visual information!")

print("\n" + "="*80)
