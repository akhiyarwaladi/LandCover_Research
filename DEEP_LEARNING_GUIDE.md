# Deep Learning Implementation Guide

**Status:** ✅ Complete and Ready to Use
**Date:** 2026-01-01
**Author:** Claude Sonnet 4.5

---

## 🎯 Overview

This implementation adds **ResNet transfer learning** to the project as a NEW classification method, different from the previous Random Forest work.

### Key Differences from Previous Work

| Aspect | Previous Work (2025) | Current Work (2026) |
|--------|---------------------|---------------------|
| **Method** | Random Forest (Traditional ML) | ResNet (Deep Learning) |
| **Input** | Individual pixels | 32x32 image patches |
| **Features** | 23 hand-crafted features | Learned features |
| **Spatial Context** | None | Local neighborhood |
| **Training Data** | 100,000 pixels | ~50,000 patches |
| **Expected Accuracy** | 74.95% | 85-90% |
| **Training Time** | 4 seconds | 30-60 minutes |

---

## 📁 New Architecture

### New Modules

```
modules/
├── data_preparation.py          # 🆕 Patch extraction & DataLoaders
└── deep_learning_trainer.py     # 🆕 ResNet/ViT/U-Net training

scripts/
└── run_resnet_classification.py # 🆕 Main deep learning script
```

### Module Descriptions

#### `modules/data_preparation.py`

**Purpose:** Prepare data for deep learning models

**Key Functions:**
- `extract_patches()` - Extract 32x32 patches from raster
- `LandCoverPatchDataset` - PyTorch Dataset class
- `get_data_loaders()` - Create train/val/test loaders
- `get_augmentation_transforms()` - Data augmentation
- `get_class_weights()` - Handle class imbalance

**Example:**
```python
from modules.data_preparation import extract_patches, get_data_loaders

# Extract patches
X_patches, y_patches = extract_patches(
    features, labels,
    patch_size=32,
    stride=16,
    max_patches=50000
)

# Create data loaders
train_loader, val_loader, test_loader = get_data_loaders(
    X_patches, y_patches,
    batch_size=32,
    val_size=0.15,
    test_size=0.15
)
```

#### `modules/deep_learning_trainer.py`

**Purpose:** Train and evaluate deep learning models

**Key Functions:**
- `get_resnet_model()` - Create ResNet with pretrained weights
- `modify_first_conv_for_multispectral()` - Support 23 channels
- `train_model()` - Training loop with validation
- `evaluate_model()` - Test set evaluation
- `save_model()` / `load_model()` - Model persistence

**Example:**
```python
from modules.deep_learning_trainer import (
    get_resnet_model,
    modify_first_conv_for_multispectral,
    train_model,
    evaluate_model
)

# Create model
model = get_resnet_model(num_classes=6, pretrained=True)
model = modify_first_conv_for_multispectral(model, in_channels=23)

# Train
history, best_state = train_model(
    model, train_loader, val_loader,
    num_epochs=20,
    learning_rate=0.001,
    device='cuda'
)

# Evaluate
results = evaluate_model(model, test_loader, device='cuda')
```

---

## 🚀 Usage

### Basic Usage

```bash
# Activate environment
conda activate landcover_jambi

# Run ResNet classification
python scripts/run_resnet_classification.py
```

### Expected Runtime

- **GPU (CUDA):** 30-60 minutes
- **CPU:** 4-6 hours (not recommended)

### Hardware Requirements

**Minimum:**
- 16 GB RAM
- 8 GB GPU memory (NVIDIA with CUDA)
- 50 GB disk space

**Recommended:**
- 32 GB RAM
- 16 GB GPU memory (RTX 3080 or better)
- 100 GB disk space

---

## 📊 Configuration

Edit constants at the top of `scripts/run_resnet_classification.py`:

```python
# Patch extraction
PATCH_SIZE = 32          # Patch size (32x32 pixels)
STRIDE = 16             # Overlap (16 = 50% overlap)
MAX_PATCHES = 50000     # Memory limit

# Training
BATCH_SIZE = 32         # Batch size
NUM_EPOCHS = 20         # Training epochs
LEARNING_RATE = 0.001   # Learning rate

# Model
MODEL_TYPE = 'resnet50'  # ResNet variant
PRETRAINED = True        # Use ImageNet weights
FREEZE_BASE = True       # Freeze conv layers
```

---

## 🔧 Dependencies

### Install PyTorch

**Option 1: Conda (Recommended)**
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

**Option 2: Pip**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Option 3: CPU-only** (not recommended)
```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

### Verify Installation

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

---

## 📈 Expected Results

Based on literature review:

| Metric | Random Forest | ResNet50 (Expected) |
|--------|---------------|---------------------|
| **Accuracy** | 74.95% | **85-90%** |
| **F1 (macro)** | 0.542 | **0.70-0.80** |
| **F1 (weighted)** | 0.744 | **0.82-0.88** |
| **Training Time** | 4 seconds | 30-60 minutes |

### Per-Class Performance Improvement

Expected improvements for minority classes:
- **Shrub/Scrub:** 0.37 → 0.55 (spatial context helps)
- **Built Area:** 0.42 → 0.65 (better feature learning)
- **Bare Ground:** 0.15 → 0.40 (transfer learning helps)

---

## 🧩 Modular Design Principles

### 1. **Separation of Concerns**

Each module has a single responsibility:
- `data_preparation.py` - Data loading only
- `deep_learning_trainer.py` - Model training only
- `run_resnet_classification.py` - Orchestration only

### 2. **Extensibility**

Easy to add new models:

```python
# In deep_learning_trainer.py - just add new function

def get_vit_model(num_classes=6, pretrained=True):
    """Create Vision Transformer model."""
    from transformers import ViTForImageClassification
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224',
        num_labels=num_classes
    )
    return model

def get_unet_model(num_classes=6):
    """Create U-Net model."""
    # Implementation here
    pass
```

Then in new script:
```python
# scripts/run_vit_classification.py
from modules.deep_learning_trainer import get_vit_model

model = get_vit_model(num_classes=6)
# Same training code as ResNet!
```

### 3. **Reusability**

All functions work independently:

```python
# Use patch extraction for other purposes
from modules.data_preparation import extract_patches
X_patches, y_patches = extract_patches(features, labels)

# Use with different model
from modules.deep_learning_trainer import train_model
history, best_state = train_model(my_custom_model, train_loader, val_loader)
```

### 4. **Consistency**

Follows same pattern as existing modules:
- Same verbose parameter
- Same return types
- Same documentation style
- Same error handling

---

## 🔬 Future Extensions

### Adding Vision Transformer (ViT)

1. Install transformers:
```bash
pip install transformers
```

2. Add function to `deep_learning_trainer.py`:
```python
def get_vit_model(num_classes=6):
    from transformers import ViTForImageClassification
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224',
        num_labels=num_classes
    )
    return model
```

3. Create `scripts/run_vit_classification.py` (copy from ResNet script)

### Adding U-Net Semantic Segmentation

1. Modify `extract_patches()` to return full patch labels:
```python
# Instead of center pixel:
y_patch = labels[i:i+patch_size, j:j+patch_size]  # Full patch
```

2. Add U-Net model:
```python
def get_unet_model(num_classes=6):
    # U-Net implementation
    pass
```

3. Modify loss function for semantic segmentation:
```python
criterion = nn.CrossEntropyLoss()  # Per-pixel loss
```

---

## 🐛 Troubleshooting

### CUDA Out of Memory

**Error:** `RuntimeError: CUDA out of memory`

**Solution 1:** Reduce batch size
```python
BATCH_SIZE = 16  # or even 8
```

**Solution 2:** Reduce max patches
```python
MAX_PATCHES = 25000  # instead of 50000
```

**Solution 3:** Reduce patch size
```python
PATCH_SIZE = 24  # instead of 32
```

### Slow Training on CPU

**Problem:** Training taking 6+ hours

**Solution:** Use GPU or reduce data
```python
MAX_PATCHES = 10000  # Smaller dataset
NUM_EPOCHS = 10      # Fewer epochs
```

### Model Not Improving

**Problem:** Validation accuracy stuck

**Solutions:**
1. Unfreeze more layers:
```python
FREEZE_BASE = False  # Train all layers
```

2. Adjust learning rate:
```python
LEARNING_RATE = 0.0001  # Lower LR
```

3. Add more augmentation:
```python
# In data_preparation.py
transforms.ColorJitter(brightness=0.2, contrast=0.2)
```

---

## ✅ Checklist for First Run

Before running ResNet classification:

- [ ] PyTorch installed and CUDA working
- [ ] GPU has 8+ GB memory
- [ ] KLHK data downloaded and in `data/klhk/`
- [ ] Sentinel-2 tiles downloaded and in `data/sentinel/`
- [ ] `results/resnet_classification/` directory will be created
- [ ] `models/` directory will be created
- [ ] Estimated time: 30-60 minutes (GPU) or 4-6 hours (CPU)

**After successful run:**

- [ ] Model saved to `models/resnet50_best.pth`
- [ ] Results show improvement over Random Forest
- [ ] Training curves show convergence
- [ ] Per-class metrics calculated

---

## 📚 References

**ResNet Transfer Learning:**
- [Deep Transfer Learning for LULC Classification](https://pmc.ncbi.nlm.nih.gov/articles/PMC8662416/)
- [Land Use Classification with ResNet](https://lgslm.medium.com/land-use-and-land-cover-classification-using-a-resnet-deep-learning-architecture-e353e7131ea4)

**PyTorch Documentation:**
- [Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [ResNet Models](https://pytorch.org/vision/stable/models.html#classification)

---

## 💡 Tips for Success

1. **Start small:** Test with `MAX_PATCHES=5000` first
2. **Monitor GPU:** Use `nvidia-smi` to watch memory usage
3. **Save frequently:** Models auto-save best version
4. **Compare carefully:** Document differences from Random Forest
5. **Visualize results:** Plot training curves and confusion matrices

---

**Questions?** Check the module docstrings for detailed documentation!

**Ready to go?** Run `python scripts/run_resnet_classification.py`! 🚀
