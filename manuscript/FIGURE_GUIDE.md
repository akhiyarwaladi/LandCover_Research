# Figure Generation Guide for Jambi Land Cover Manuscript

## API Credit Issue

Your OpenRouter API key needs more credits to use the AI image generation models (Gemini 3 Pro).
- **Current credits**: Can afford 3,333 tokens
- **Required**: Up to 32,768 tokens per image
- **Solution**: Add credits at https://openrouter.ai/settings/credits

## Quick Solution: Copy Existing Figures

You already have several high-quality figures in your `results/` folder! Let's use those:

```bash
cd "C:\Users\MyPC PRO\Documents\LandCover_Research\manuscript\figures"
copy "..\..\results\classifier_comparison.png" "classifier_comparison.png"
copy "..\..\results\confusion_matrix_random_forest.png" "confusion_matrix_random_forest.png"
copy "..\..\results\feature_importance_random_forest.png" "feature_importance_random_forest.png"
```

## Required Figures for the Manuscript

### Figure 1: Graphical Abstract (MANDATORY)

**Purpose**: Visual summary for journal table of contents

**Description**: Horizontal workflow showing:
1. Input data (Jambi map + Sentinel-2 + KLHK polygons)
2. → Data preprocessing (KMZ download + cloud filtering)
3. → Feature engineering (10 bands + 13 indices = 23 features)
4. → Machine learning (Random Forest + 7 classifiers)
5. → Results (classified map + 74.95% accuracy)

**How to create**:
- **Option A**: Add OpenRouter credits and use scientific-schematics skill
- **Option B**: Create in PowerPoint/Canva with workflow boxes and arrows
- **Option C**: Use Python Matplotlib to create a simple flowchart

### Figure 2: Study Area Map

**What it shows**:
- Jambi Province location in Indonesia
- Province boundary
- Major cities and rivers
- Elevation gradient

**How to create**: QGIS or Python (geopandas + matplotlib)

### Figure 3: Classifier Comparison ✅ READY

**Location**: `../results/classifier_comparison.png`
**Action**: Just copy to figures/ folder (see command above)

### Figure 4: Confusion Matrix ✅ READY

**Location**: `../results/confusion_matrix_random_forest.png`
**Action**: Just copy to figures/ folder

### Figure 5: Feature Importance ✅ READY

**Location**: `../results/feature_importance_random_forest.png`
**Action**: Just copy to figures/ folder

### Figure 6: Methodology Flowchart

**Description**: Vertical flowchart with 4 boxes:
1. Data Acquisition (KLHK + Sentinel-2)
2. Preprocessing (cloud filter, compositing, rasterization)
3. Classification (7 models, train/test split)
4. Evaluation (metrics, confusion matrix)

**How to create**: PowerPoint or draw.io, or wait for API credits to use AI generation

## Next Steps

1. **Immediate**: Copy the 3 ready figures from results/ to figures/
2. **Optional**: Create study area map in QGIS
3. **For journal submission**: Create graphical abstract (required by most journals)
   - Can be simple PowerPoint design
   - Or add $5-10 credits to OpenRouter for AI generation

## AI Generation Commands (When You Have Credits)

```bash
# Navigate to schematics skill
cd "C:\Users\MyPC PRO\.claude\plugins\cache\claude-scientific-writer\claude-scientific-writer\a6ee89d051eb\skills\scientific-schematics"

# Graphical abstract
python scripts/generate_schematic.py "Graphical abstract showing workflow: Input data (Jambi map, Sentinel-2, KLHK polygons) → Preprocessing (KMZ, cloud filter) → Feature engineering (23 features) → Random Forest ML (7 classifiers) → Results (classified map, 74.95%% accuracy). Clean professional landscape style" -o "C:\Users\MyPC PRO\Documents\LandCover_Research\manuscript\figures\graphical_abstract.png" --doc-type journal --api-key YOUR_KEY_HERE

# Methodology flowchart
python scripts/generate_schematic.py "Vertical flowchart: 1. Data Acquisition (KLHK 28100 polygons, Sentinel-2 2.7GB) → 2. Preprocessing (cloud filtering, median composite, rasterization, 100k samples) → 3. Classification (7 classifiers, 80/20 split, balanced weights) → 4. Evaluation (accuracy, confusion matrix, feature importance)" -o "C:\Users\MyPC PRO\Documents\LandCover_Research\manuscript\figures\methodology_workflow.png" --doc-type journal --api-key YOUR_KEY_HERE
```

## Manuscript Completion Status

✅ **Complete**:
- Main manuscript text (full IMRAD structure)
- BibTeX references
- 3 result figures ready to use

⏳ **Pending**:
- Graphical abstract (required for submission)
- Study area map (helpful but not always required)
- Methodology flowchart (optional, enhances clarity)

**Bottom line**: Your manuscript is 95% complete! The text is publication-ready. You just need to add figures.
