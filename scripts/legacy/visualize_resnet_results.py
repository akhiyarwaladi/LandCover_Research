#!/usr/bin/env python3
"""
Visualize ResNet Training Results
==================================

Creates comprehensive visualizations:
1. Training curves (loss and accuracy)
2. Confusion matrix
3. ResNet vs Random Forest comparison

Author: Claude Sonnet 4.5
Date: 2026-01-03
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

# Load training history
history = np.load('results/resnet_fixed/training_history.npz')
train_loss = history['train_loss']
train_acc = history['train_acc']
val_loss = history['val_loss']
val_acc = history['val_acc']

# Load test results
test_data = np.load('results/resnet_fixed/test_results.npz')
y_true = test_data['targets']
y_pred = test_data['predictions']

# ============================================================================
# PLOT 1: Training Curves
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Loss curve
ax = axes[0]
epochs = range(1, len(train_loss) + 1)
ax.plot(epochs, train_loss, 'b-', linewidth=2, label='Training Loss')
ax.plot(epochs, val_loss, 'r-', linewidth=2, label='Validation Loss')
ax.axvline(x=6, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Best Model (Epoch 6)')
ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
ax.set_ylabel('Loss', fontsize=14, fontweight='bold')
ax.set_title('ResNet50 Training Loss', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

# Accuracy curve
ax = axes[1]
ax.plot(epochs, np.array(train_acc) * 100, 'b-', linewidth=2, label='Training Accuracy')
ax.plot(epochs, np.array(val_acc) * 100, 'r-', linewidth=2, label='Validation Accuracy')
ax.axvline(x=6, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Best Model (82.04%)')
ax.axhline(y=74.95, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='Random Forest Baseline')
ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_title('ResNet50 Training Accuracy', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_ylim([60, 100])

plt.tight_layout()
plt.savefig('results/resnet_fixed/training_curves.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/resnet_fixed/training_curves.png")
plt.close()

# ============================================================================
# PLOT 2: Confusion Matrix
# ============================================================================

from sklearn.metrics import confusion_matrix

class_names = ['Water', 'Trees', 'Crops', 'Shrub', 'Built', 'Bare']

# Calculate confusion matrix
cm = confusion_matrix(y_true, y_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Plot
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Normalized Frequency'},
            ax=ax, vmin=0, vmax=1)
ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
ax.set_title('ResNet50 Confusion Matrix\nTest Accuracy: 79.80%',
             fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig('results/resnet_fixed/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/resnet_fixed/confusion_matrix.png")
plt.close()

# ============================================================================
# PLOT 3: ResNet vs Random Forest Comparison
# ============================================================================

# ResNet per-class F1 scores
resnet_f1 = {
    'Water': 0.74,
    'Trees': 0.77,
    'Crops': 0.84,
    'Shrub': 0.31,
    'Built': 0.50,
    'Bare': 0.20
}

# Random Forest per-class F1 scores (from previous results)
rf_f1 = {
    'Water': 0.79,
    'Trees': 0.74,
    'Crops': 0.78,
    'Shrub': 0.37,
    'Built': 0.42,
    'Bare': 0.15
}

classes = ['Water', 'Trees', 'Crops', 'Shrub', 'Built', 'Bare']
resnet_scores = [resnet_f1[c] for c in classes]
rf_scores = [rf_f1[c] for c in classes]

# Create comparison plot
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Per-class comparison
ax = axes[0]
x = np.arange(len(classes))
width = 0.35

bars1 = ax.bar(x - width/2, rf_scores, width, label='Random Forest',
               color='#FF7F0E', alpha=0.8)
bars2 = ax.bar(x + width/2, resnet_scores, width, label='ResNet50',
               color='#2CA02C', alpha=0.8)

ax.set_xlabel('Land Cover Class', fontsize=14, fontweight='bold')
ax.set_ylabel('F1-Score', fontsize=14, fontweight='bold')
ax.set_title('Per-Class Performance Comparison', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(classes, rotation=45, ha='right')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 1.0])

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=9)

# Overall metrics comparison
ax = axes[1]
metrics = ['Overall\nAccuracy', 'F1\n(Weighted)', 'F1\n(Macro)']
rf_metrics = [74.95, 0.744*100, 0.542*100]  # Convert to percentage for consistency
resnet_metrics = [79.80, 0.792*100, 0.559*100]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax.bar(x - width/2, rf_metrics, width, label='Random Forest',
               color='#FF7F0E', alpha=0.8)
bars2 = ax.bar(x + width/2, resnet_metrics, width, label='ResNet50',
               color='#2CA02C', alpha=0.8)

ax.set_ylabel('Score (%)', fontsize=14, fontweight='bold')
ax.set_title('Overall Performance Comparison', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=12)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 100])

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

# Add improvement annotations
improvements = [
    resnet_metrics[0] - rf_metrics[0],
    resnet_metrics[1] - rf_metrics[1],
    resnet_metrics[2] - rf_metrics[2]
]

for i, imp in enumerate(improvements):
    if imp > 0:
        ax.text(i, max(rf_metrics[i], resnet_metrics[i]) + 5,
                f'+{imp:.1f}%',
                ha='center', fontsize=11, fontweight='bold', color='green')

plt.tight_layout()
plt.savefig('results/resnet_fixed/resnet_vs_random_forest.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/resnet_fixed/resnet_vs_random_forest.png")
plt.close()

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*80)
print("RESNET50 VISUALIZATION COMPLETE")
print("="*80)

print("\n📊 Summary Statistics:")
print(f"\nTraining:")
print(f"  Best Epoch: 6")
print(f"  Best Val Accuracy: {val_acc[5]*100:.2f}%")
print(f"  Final Train Accuracy: {train_acc[-1]*100:.2f}%")
print(f"  Final Val Accuracy: {val_acc[-1]*100:.2f}%")

print(f"\nTest Set:")
print(f"  Accuracy: 79.80%")
print(f"  F1 (Weighted): 0.792")
print(f"  F1 (Macro): 0.559")

print(f"\nImprovement over Random Forest:")
print(f"  Accuracy: +4.85%")
print(f"  F1 (Weighted): +0.048")
print(f"  F1 (Macro): +0.017")

print(f"\n✅ All visualizations saved to: results/resnet_fixed/")
print("="*80)
