# ResNet Training - Saved Data Reference

**What information is saved during training and how to use it for publication tables**

---

## 📊 Data Saved by Training Script

### For Each Variant (resnet18, resnet34, resnet50, resnet101, resnet152):

#### 1. **`results/{variant}/training_history.npz`**
Contains training progress arrays:
```python
{
    'train_loss': [epoch1, epoch2, ..., epoch30],  # Training loss per epoch
    'train_acc': [epoch1, epoch2, ..., epoch30],   # Training accuracy per epoch
    'val_loss': [epoch1, epoch2, ..., epoch30],    # Validation loss per epoch
    'val_acc': [epoch1, epoch2, ..., epoch30]      # Validation accuracy per epoch
}
```

**Usage:** Training curves, convergence analysis, best epoch identification

---

#### 2. **`results/{variant}/test_results.npz`**
Contains test set evaluation:
```python
{
    'predictions': [pred1, pred2, ..., pred20000],  # Predicted labels (test set)
    'targets': [true1, true2, ..., true20000],      # True labels (test set)
    'accuracy': 0.7980,                             # Overall test accuracy (scalar)
    'f1_macro': 0.5593,                             # F1-score macro average (scalar)
    'f1_weighted': 0.7920                           # F1-score weighted average (scalar)
}
```

**Usage:** Confusion matrix, per-class metrics, overall performance metrics

---

#### 3. **`results/all_variants_summary.json`**
Combined summary for all variants:
```json
[
    {
        "variant": "resnet18",
        "acc": 0.7654,
        "f1_macro": 0.5234,
        "time_min": 14.2
    },
    {
        "variant": "resnet34",
        "acc": 0.7812,
        "f1_macro": 0.5456,
        "time_min": 18.7
    },
    ...
]
```

**Usage:** Quick comparison table, training time analysis

---

#### 4. **`models/{variant}_best.pth`**
Trained model weights (binary file)

**Usage:** Inference, generating predictions, transfer learning

---

#### 5. **`results/{variant}/predictions.npy`** (if spatial prediction ran)
Full spatial predictions:
```python
array([[0, 1, 2, ...],   # 2D array of predicted class labels
       [1, 1, 2, ...],   # Shape: (11268, 18740)
       ...])
```

**Usage:** Spatial accuracy calculation, prediction maps

---

## 📋 Publication Tables Generated

### Centralized Script: `generate_publication_tables.py`

#### Mode 1: Single Variant Detail (DEFAULT)
```bash
python scripts/generate_publication_tables.py --variant resnet50
```

**Generates:**
- `resnet50_detailed_tables.xlsx` (multi-sheet, formatted)
  - Sheet 1: Overall Performance (vs RF baseline)
  - Sheet 2: Training Configuration
  - Sheet 3: Per-Class Metrics
- CSV backups for each table

**Features:**
- ✅ Beautiful XLSX formatting
- ✅ Auto-adjusted column widths
- ✅ Professional headers (blue background, white text)
- ✅ Borders and alignment
- ✅ Title rows with merge cells

---

#### Mode 2: Multi-Architecture Comparison
```bash
python scripts/generate_publication_tables.py --all
```

**Generates:**
- `Architecture_Comparison.xlsx` (single sheet, formatted)
  - Columns: Architecture, Depth, Parameters, Accuracy, F1, Precision, Recall, Improvement
  - Rows: One per variant (ResNet18, 34, 50, 101, 152)
- CSV backup

**Features:**
- ✅ Side-by-side comparison of all architectures
- ✅ Sorted by architecture
- ✅ Improvement vs RF baseline calculated automatically

---

## 🎨 XLSX Formatting Details

### Header Row Formatting:
- **Background:** Blue (#4472C4)
- **Text:** White, Bold, 11pt
- **Alignment:** Center, Wrapped
- **Borders:** All sides, thin

### Title Row (if present):
- **Background:** Dark Blue (#366092)
- **Text:** White, Bold, 14pt
- **Height:** 30 pixels
- **Merged:** Across all columns

### Data Rows:
- **Alignment:** Center (except first column = left)
- **Borders:** All sides, thin
- **First Column:** Bold text

### Column Widths:
- **Auto-adjusted** based on content length
- Maximum width: 50 characters
- Minimum padding: 3 characters

---

## 📊 Table Contents Detail

### Table 1: Overall Performance
From `test_results.npz`:

| Metric | Random Forest | ResNet{XX} |
|--------|--------------|------------|
| Overall Accuracy (%) | 74.95 | {from test_results.accuracy} |
| Precision (Macro) | 0.5800 | {calculated from predictions/targets} |
| Recall (Macro) | 0.5400 | {calculated from predictions/targets} |
| F1-Score (Macro) | 0.5420 | {from test_results.f1_macro} |
| F1-Score (Weighted) | 0.7440 | {from test_results.f1_weighted} |
| Improvement (Accuracy) | - | +{difference}% |
| Improvement (F1-Macro) | - | +{difference}% |

---

### Table 2: Training Configuration
From `training_history.npz` and constants:

| Configuration Parameter | Value |
|------------------------|-------|
| Model Architecture | ResNet{XX} (pretrained) |
| Input Patch Size | 32 × 32 pixels |
| Number of Channels | 23 (10 bands + 13 indices) |
| Total Epochs | 30 |
| Best Epoch | {argmax(val_acc) + 1} |
| Best Validation Accuracy (%) | {max(val_acc) * 100} |
| Final Training Loss | {train_loss[-1]} |
| Final Validation Loss | {val_loss[-1]} |
| Learning Rate | 0.0001 |
| Optimizer | Adam |
| Batch Size | 16 |
| Training Samples | 80,000 |
| Test Samples | 20,000 |

---

### Table 3: Per-Class Performance
From `test_results.npz` predictions and targets:

| Class | Precision | Recall | F1-Score (ResNet) | F1-Score (RF) | Improvement | Test Samples |
|-------|-----------|--------|-------------------|---------------|-------------|--------------|
| Water | {calc} | {calc} | {calc} | 0.79 | {diff} | {support[0]} |
| Trees | {calc} | {calc} | {calc} | 0.74 | {diff} | {support[1]} |
| Crops | {calc} | {calc} | {calc} | 0.78 | {diff} | {support[2]} |
| Shrub | {calc} | {calc} | {calc} | 0.37 | {diff} | {support[3]} |
| Built | {calc} | {calc} | {calc} | 0.42 | {diff} | {support[4]} |
| Bare | {calc} | {calc} | {calc} | 0.15 | {diff} | {support[5]} |

Calculated using sklearn's `precision_recall_fscore_support()` function.

---

### Architecture Comparison Table (--all mode)
From all `test_results.npz` files:

| Architecture | Depth | Parameters (M) | Test Accuracy (%) | F1 (Macro) | F1 (Weighted) | Precision (Macro) | Recall (Macro) | Improvement vs RF (%) |
|--------------|-------|----------------|-------------------|------------|---------------|-------------------|----------------|----------------------|
| RESNET18 | 18 | 11.7 | {accuracy*100} | {f1_macro} | {f1_weighted} | {precision} | {recall} | +{improvement} |
| RESNET34 | 34 | 21.8 | {accuracy*100} | {f1_macro} | {f1_weighted} | {precision} | {recall} | +{improvement} |
| RESNET50 | 50 | 25.6 | {accuracy*100} | {f1_macro} | {f1_weighted} | {precision} | {recall} | +{improvement} |
| RESNET101 | 101 | 44.5 | {accuracy*100} | {f1_macro} | {f1_weighted} | {precision} | {recall} | +{improvement} |
| RESNET152 | 152 | 60.2 | {accuracy*100} | {f1_macro} | {f1_weighted} | {precision} | {recall} | +{improvement} |

---

## 🔧 Usage Examples

### After Training Completes:

```bash
# Generate detailed tables for ResNet50
python scripts/generate_publication_tables.py --variant resnet50

# Generate comparison table for all variants
python scripts/generate_publication_tables.py --all

# Generate detailed tables for ResNet18
python scripts/generate_publication_tables.py --variant resnet18
```

### Output Location:
```
results/publication/tables/
├── Architecture_Comparison.xlsx        (all variants)
├── Architecture_Comparison.csv         (backup)
├── resnet50_detailed_tables.xlsx       (multi-sheet)
├── resnet50_overall_performance.csv
├── resnet50_training_config.csv
├── resnet50_perclass_metrics.csv
└── ... (same for other variants)
```

---

## ✨ Key Features

### 1. **Works from Saved Models**
- No need to retrain!
- Reads from `.npz` and `.json` files
- Regenerate tables anytime

### 2. **Professional Formatting**
- Publication-ready XLSX
- Auto-adjusted widths
- Formatted headers
- Borders and alignment

### 3. **Flexible**
- Single variant detail OR
- All variants comparison
- CSV backups included

### 4. **Complete Metrics**
- Overall performance
- Per-class breakdown
- Training configuration
- Comparison to baseline

---

## 📝 For Journal Paper

### Recommended Table Usage:

**Table 1:** Architecture Comparison (`--all` mode)
- Shows trade-off between model complexity and performance
- Highlights parameter efficiency

**Table 2:** Best Model Detailed Performance (e.g., `--variant resnet101`)
- Shows winning model's complete metrics
- Includes training configuration for reproducibility

**Table 3:** Per-Class Analysis (from detailed tables)
- Shows which classes benefit most from deeper networks
- Identifies challenging classes

---

**Last Updated:** 2026-01-03
**Training Data Structure:** v1.0
**Table Generator:** v2.0 (with XLSX support)
