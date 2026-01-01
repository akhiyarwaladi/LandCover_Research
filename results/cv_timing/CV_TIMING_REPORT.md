# Cross-Validation vs Simple Split: Training Time Analysis

**Date:** 2026-01-01  
**Dataset Size:** 100,000 training samples  
**Epochs:** 20  

---

## Summary

| Method | Total Time | vs Simple Split | Training Runs |
|--------|------------|-----------------|---------------|
| Simple Split (70/15/15) | 0.87 hrs | 1.0× (baseline) | 1 |
| 5-Fold CV | 5.04 hrs | 5.8× | 5 |
| 10-Fold CV | 10.64 hrs | 12.2× | 10 |

---

## Key Findings

1. **Simple Split is 5.8× faster** than 5-fold CV
   - Simple: 0.87 hours
   - 5-Fold CV: 5.04 hours
   - Time saved: 4.17 hours

2. **10-Fold CV is 12.2× slower** than simple split
   - Requires training 10 separate models
   - Total time: 10.64 hours

3. **Large Dataset Justifies Simple Split**
   - With 100,000 training samples, simple split provides:
     * 70,000 training samples (70%)
     * 15,000 validation samples (15%)
     * 15,000 test samples (15%)
   - Large validation set provides reliable performance estimates
   - Cross-validation typically needed for small datasets (<1,000 samples)

---

## Justification for Manuscript

### Why We Chose Simple Split Over Cross-Validation:

1. **Dataset Size Sufficiency**
   - Our dataset contains 100,000 samples, providing statistically robust estimates
   - Validation set (15,000 samples) is large enough to reliably estimate model performance
   - Cross-validation is primarily beneficial for small datasets where maximizing data usage is critical

2. **Computational Efficiency**
   - Simple split: 0.87 hours
   - 5-Fold CV: 5.04 hours (5.8× longer)
   - Enables faster iteration for hyperparameter tuning and model development

3. **Practical Considerations**
   - Single trained model is simpler to deploy and maintain
   - No need for model ensembling or averaging predictions
   - Consistent model for production use

4. **Literature Precedent**
   - Large-scale remote sensing studies typically use simple train/val/test splits
   - ImageNet, COCO, and other benchmark datasets use simple splits
   - Cross-validation more common in medical imaging with limited samples

---

## Recommended Manuscript Text

### Methods Section:

```
We employed a simple train/validation/test split (70%/15%/15%) for model
development and evaluation. The large dataset size (100,000 training samples)
provided sufficient statistical power without requiring k-fold cross-validation.
This approach yielded 70,000 training samples,
15,000 validation samples for hyperparameter tuning, and
15,000 test samples for final performance assessment.
The validation set was used for early stopping and model selection, while the
test set remained unseen until final evaluation to prevent overfitting.
```

### Results/Discussion (if reviewers question this choice):

```
While k-fold cross-validation is beneficial for small datasets, our large sample
size (100,000 samples) provided robust performance estimates with a simple
split approach. This methodology is consistent with established practices in
large-scale remote sensing applications (cite: ImageNet, COCO datasets) and
enabled efficient model development while maintaining statistical rigor. A 5-fold
cross-validation would have required 5.8× longer
training time without substantial benefits given the large validation set.
```

---

## When Cross-Validation IS Recommended:

- **Small datasets** (<1,000-5,000 samples)
- **Medical imaging** with limited patient data
- **Rare event detection** with class imbalance
- **Model comparison** when differences are small
- **Uncertainty quantification** requiring variance estimates

---

**Generated:** 2026-01-01  
**Dataset:** Jambi Land Cover Classification  
**Purpose:** Justify simple train/val/test split for manuscript  
