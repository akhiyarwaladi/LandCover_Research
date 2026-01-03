#!/usr/bin/env python3
"""Generate Publication Figures - One Concept Per Figure"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from sklearn.metrics import confusion_matrix, classification_report

plt.style.use('seaborn-v0_8-paper')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

OUTPUT_DIR = 'results/publication/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_COLORS = {0: '#0066CC', 1: '#228B22', 2: '#90EE90', 3: '#FF8C00', 4: '#FF1493', 5: '#D2691E'}
CLASS_NAMES = ['Water', 'Trees', 'Crops', 'Shrub', 'Built', 'Bare']

print("Generating publication figures...")

# Figure 1: Training Curves
history = np.load('results/resnet/training_history.npz')
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
epochs = range(1, len(history['train_loss']) + 1)
best_epoch = np.argmax(history['val_acc']) + 1

axes[0].plot(epochs, history['train_loss'], 'b-', lw=2, label='Training')
axes[0].plot(epochs, history['val_loss'], 'r-', lw=2, label='Validation')
axes[0].axvline(best_epoch, color='g', ls='--', label=f'Best ({best_epoch})')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
axes[0].set_title('(a) Loss Curves'); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(epochs, history['train_acc']*100, 'b-', lw=2, label='Training')
axes[1].plot(epochs, history['val_acc']*100, 'r-', lw=2, label='Validation')
axes[1].axhline(74.95, color='orange', ls=':', lw=2, label='RF Baseline')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('(b) Accuracy Curves'); axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/Figure1_Training_Curves.png', dpi=300, bbox_inches='tight')
print("✓ Figure 1: Training Curves")

# Figure 2: Confusion Matrix
test_data = np.load('results/resnet/test_results.npz')
cm = confusion_matrix(test_data['targets'], test_data['predictions'])
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues', xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES, ax=ax, vmin=0, vmax=1, linewidths=0.5)
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
ax.set_title('Confusion Matrix (Accuracy: 79.80%)')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/Figure2_Confusion_Matrix.png', dpi=300, bbox_inches='tight')
print("✓ Figure 2: Confusion Matrix")

# Figure 3: Per-Class Performance
report = classification_report(test_data['targets'], test_data['predictions'], 
                               output_dict=True, zero_division=0)
resnet_f1 = [report[str(i)]['f1-score'] for i in range(6)]
rf_f1 = [0.79, 0.74, 0.78, 0.37, 0.42, 0.15]

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(6)
width = 0.35
ax.bar(x - width/2, rf_f1, width, label='Random Forest', color='#FF7F0E', alpha=0.85)
ax.bar(x + width/2, resnet_f1, width, label='ResNet50', color='#2CA02C', alpha=0.85)
ax.set_xlabel('Land Cover Class'); ax.set_ylabel('F1-Score')
ax.set_title('Per-Class Performance: ResNet50 vs Random Forest')
ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
ax.legend(); ax.grid(alpha=0.3, axis='y'); ax.set_ylim([0, 1])
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/Figure4_PerClass_Performance.png', dpi=300, bbox_inches='tight')
print("✓ Figure 4: Per-Class Performance")

print(f"\n✅ All figures saved to {OUTPUT_DIR}/")
