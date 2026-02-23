# Scene Classification Project — Instructions for Claude

## Project Overview

Comparative analysis of CNN vs Transformer architectures for remote sensing scene classification.
- 8 models: ResNet-50, ResNet-101, DenseNet-121, EfficientNet-B0/B3, ViT-B/16, Swin-T, ConvNeXt-Tiny
- 2 datasets: EuroSAT (27K images, 10 classes) and UC Merced (2.1K images, 21 classes)
- Best: ConvNeXt-Tiny on EuroSAT (99.06%), EfficientNet-B3 on UC Merced (99.76%)

## Manuscript Location

`publication/manuscript/manuscript.tex` — IEEE format, LaTeX

## Figure Layout Rules

### Same-page pairs (full-width `figure*`)
These figure pairs MUST appear on the same page in the compiled PDF:
1. **Dataset samples**: EuroSAT samples + UC Merced samples (combined as `figure*` with `fig:samples`)
2. **Misclassified examples**: EuroSAT misclassified + UC Merced misclassified (combined as `figure*` with `fig:misclassified`)

Do NOT split these back into separate single-column figures.

### Single-column constraint
ALL other figures MUST stay in single-column (`\columnwidth`). Do NOT use `figure*` or `\textwidth` for individual figures. Never make a figure span the full page width unless it is one of the same-page pairs listed above.

### Heatmap design (F1-score heatmaps)
- Use horizontal colorbar at bottom (not vertical on the right) — this lets the heatmap grid fill the full column width and look visually square
- Figure dimensions: COL_W x COL_W (3.45 x 3.45 in), saved with explicit `Bbox.from_bounds` to guarantee exact square PDF
- EuroSAT (10 classes): show annotations inside cells
- UC Merced (21 classes): color-only, no annotations (too many columns)
- Colormap: `Blues` (clean, standard in Nature/IEEE, matches confusion matrices)

### Color scheme (all figures)
- Model palette: Okabe-Ito + Paul Tol (8 colorblind-safe colors) — Nature Methods standard
- Family colors: blue (CNN), vermillion (Transformer), teal green (Modern CNN) — Okabe-Ito
- Bar charts: Okabe-Ito blue + vermillion (not green/red — colorblind issue)
- Heatmaps: `Blues` for performance, `PuOr` for McNemar p-values
- All colormaps must be colorblind-safe and perceptually uniform

## Writing Style Rules

### Turnitin / Similarity
- All flagged passages from `check_turnitin/similarity_map.md` have been paraphrased
- When writing new text or editing existing text, do NOT use generic descriptions of models (ResNet, ViT, Swin, etc.) — always tie descriptions to this specific study's context
- Do NOT copy standard phrasing from survey papers (e.g., "splits an image into fixed-size patches and feeds them to a transformer encoder")
- Rewrite dataset statistics in varied sentence structures, not "X has N images across K classes at R resolution"

### AI Detection
- Do NOT use these AI-favorite words: "foundational", "complementary", "multi-faceted", "notably", "aligns with", "granularity", "paradigm shift", "leverage", "comprehensive"
- Avoid overly polished hedging like "A plausible explanation lies in..." — write more directly
- Avoid rule-of-three/four lists that feel forced
- Reduce em-dash (---) usage; use commas or periods instead when possible
- Use short punchy sentences mixed with longer ones — vary rhythm
- Write like a real researcher: direct, specific, occasionally informal ("the gap largely disappears" not "the disparity diminishes substantially")
- Do NOT over-qualify: "the results suggest that it may potentially be the case that..." — just state the finding

### Formal Language
- No question-type sentences in the manuscript (e.g., "How much does architecture matter?")
- All sentences must be declarative statements
- Keep DPI at 300 for all figures

## Key Files

- `config.py` — central config with MODELS, DATASETS, TRAINING params
- `publication/manuscript/manuscript.tex` — main LaTeX manuscript
- `publication/manuscript/generate_all_figures.py` — figure generation script
- `publication/manuscript/references.bib` — BibTeX references
- `publication/manuscript/check_turnitin/similarity_map.md` — Turnitin similarity report
- `publication/manuscript/figures/` — all PDF/PNG figures

## Compilation

```bash
cd scene_classification/publication/manuscript
pdflatex -interaction=nonstopmode manuscript.tex
pdflatex -interaction=nonstopmode manuscript.tex  # second pass for references
```

## Environment

- LaTeX: MiKTeX on Windows
- Python: conda environment with PyTorch, timm, matplotlib, seaborn
- Draw.io: `"C:/Program Files/draw.io/draw.io.exe"` for flowchart export
