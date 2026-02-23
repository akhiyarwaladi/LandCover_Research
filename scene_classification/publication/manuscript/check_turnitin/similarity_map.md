# Turnitin Similarity Map — Manuscript: CNN vs Transformer for RS Scene Classification

**Overall Similarity: 21%** | Not Cited or Quoted: 18% (35 matches) | Missing Quotations: 2% (12 matches)

> **Goal:** Paraphrase every flagged passage below to reduce similarity. Each entry shows the source number, matched source, and the flagged text.

---

## Top Sources Reference

| # | Source | Type | Match % |
|---|--------|------|---------|
| 1 | arxiv.org | Internet | 7% |
| 2 | export.arxiv.org | Internet | 2% |
| 3 | www.joig.net | Internet | 1% |
| 4 | Cork Institute of Technology | Student paper | 1% |
| 5 | downloads.hindawi.com | Internet | 1% |
| 6 | d197for5662m48.cloudfront.net | Internet | <1% |
| 7 | hal.science | Internet | <1% |
| 8 | Zhaobin Wang et al. "Hybrid cGAN..." | Publication | <1% |
| 9 | theses.liacs.nl | Internet | <1% |
| 10 | djvu.online | Internet | <1% |
| 11–31 | Various (all <1%) | Mixed | <1% each |

---

## ⚠️ Integrity Flag

**Hidden Text detected:** 70 suspect characters on 1 page — text altered to blend into white background. **Fix this first** — remove any hidden/white text from your manuscript before resubmission.

---

## ABSTRACT

### Flag A1 — Source 31 (Gong Cheng et al., <1%)
**Flagged text:**
> "remote sensing scene classification by benchmarking eight deep learning models on two standard datasets"

**Why flagged:** Common phrasing in RS literature. Rewrite to be more specific to your contribution.

### Flag A2 — Source 21 (mediatum.ub.tum.de, <1%)
**Flagged text:**
> "classical CNNs (ResNet-50, ResNet-101, DenseNet-121, EfficientNet-B0, EfficientNet-B3), vision transformers (ViT-B/16, Swin Transformer), and a modernized CNN (ConvNeXt-Tiny)"

**Why flagged:** Listing model names in this standard format matches many papers. Rewrite the sentence structure around the model listing.

### Flag A3 — Source 23 (Maheshbhai et al., <1%)
**Flagged text:**
> "Every model was trained with the same hyperparameters, the same augmentation pipeline, and the same ImageNet-pretrained initialization"

**Why flagged:** Common experimental description phrasing.

---

## I. INTRODUCTION

### Flag I1 — Source 27 (Rui Cao et al., <1%)
**Flagged text:**
> "Assigning a single land-use label to a satellite or aerial image patch is one of the oldest problems in remote sensing"

**Why flagged:** Standard opening phrasing in RS scene classification papers.

### Flag I2 — Source 13 (Christoph Koller et al., <1%)
**Flagged text:**
> "Networks such as VGGNet [5], ResNet [6], and DenseNet [7] learn features directly from pixels, and when pretrained on ImageNet [8] and fine-tuned on remote sensing data, they consistently outperform handcrafted approaches"

**Why flagged:** Standard description of CNN transfer learning pipeline.

### Flag I3 — Source 30 (www.arxiv-vanity.com, <1%)
**Flagged text:**
> "pretrained on ImageNet and fine-tuned on remote sensing data, they consistently outperform handcrafted approaches [9], [10]. Transfer learning proved especially useful because labeled satellite imagery is often scarce"

**Why flagged:** Overlaps with common transfer learning descriptions.

---

## II. RELATED WORK

### Flag R1 — Source 1 (arxiv.org, 7%) ⭐ HIGHEST PRIORITY
**Section: II-A CNN-Based Scene Classification**

**Flagged text:**
> "ResNet [6] introduced shortcut connections that let gradients flow directly through the network, and the 50- and 101-layer variants quickly became the default baselines in remote sensing. DenseNet [7] took a different route: every layer receives input from all preceding layers, which encourages feature reuse and keeps the parameter count low."

**Why flagged:** This is a very standard description that closely matches survey papers on arxiv. **Must rewrite substantially.**

### Flag R2 — Source 1 (arxiv.org)
**Section: II-A continued**

**Flagged text:**
> "The EfficientNet family [20] showed that scaling depth, width, and resolution together is more effective than scaling any one dimension alone. Nogueira et al. [9] provided early evidence that fine-tuning ImageNet-pretrained CNNs beats training from scratch for aerial scene recognition, a finding that has been replicated many times since"

**Why flagged:** Standard literature review phrasing.

### Flag R3 — Source 1 (arxiv.org)
**Section: II-B Transformer-Based Approaches**

**Flagged text:**
> "ViT [14] splits an image into fixed-size patches, embeds them, and feeds the sequence to a standard transformer encoder with self-attention. The appeal for remote sensing is that attention can capture relationships between distant image regions that local convolution filters would miss"

**Why flagged:** Almost verbatim from ViT description in survey papers. **Must rewrite.**

### Flag R4 — Source 1 (arxiv.org)
**Section: II-B continued**

**Flagged text:**
> "self-attention scales quadratically with the number of patches. Swin Transformer [15] addresses this by computing attention inside local windows that shift across layers, producing a hierarchical feature pyramid at linear cost"

**Why flagged:** Standard Swin Transformer description.

### Flag R5 — Source 1 (arxiv.org)
**Section: II-C Modernized CNNs**

**Flagged text:**
> "ConvNeXt [18] is the result of a thought experiment: starting from a plain ResNet and, one design choice at a time, adopting ideas from transformers. Larger kernels (7×7), layer normalization, GELU activations, and an inverted bottleneck layout brought a standard convolution network to the same accuracy as Swin Transformer on ImageNet"

**Why flagged:** Closely mirrors the ConvNeXt paper abstract/introduction.

---

## II-D. BENCHMARK DATASETS

### Flag D1 — Sources 15, 26 (Dilxat Muhtar et al.; users.ntua.gr)
**Flagged text:**
> "The UC Merced Land Use dataset [3] has 2,100 aerial images across 21 land-use classes at 0.3 m resolution, drawn from USGS National Map imagery"

**Why flagged:** Standard dataset description copied across many papers.

### Flag D2 — Source 15 (Dilxat Muhtar et al.)
**Flagged text:**
> "EuroSAT [22] provides 27,000 Sentinel-2 multispectral patches at 10 m resolution with 10 classes. Larger collections exist, including NWPU-RESISC45 [4] (31,500 images, 45 classes) and AID [23] (10,000 images, 30 classes)"

**Why flagged:** Listing dataset statistics in this format is extremely common.

---

## III. METHODOLOGY

### Flag M1 — Source 29 (Zhang, Fan et al., <1%)
**Flagged text:**
> "Fig. 1 provides an overview of the research methodology. The pipeline consists of five phases: data acquisition, preprocessing, model training, performance evaluation, and comparative analysis."

### Flag M2 — Sources 28, 19 (Tushar Nayak et al.; City University)
**Section: III-A-1 EuroSAT**

**Flagged text:**
> "We use EuroSAT [22], a collection of 27,000 Sentinel-2 satellite patches in 10 land-use categories. Each patch is 64×64 pixels at 10 m ground sampling distance"

**Why flagged:** Verbatim dataset description.

### Flag M3 — Sources 16, 20 (Asad Ullah Haider et al.; Supanat Jintawatsakoon et al.)
**Section: III-C Training Protocol**

**Flagged text:**
> "All images are resized to 224×224 pixels and augmented with random horizontal/vertical flips, random rotation (±15°), and color jitter (brightness 0.2, contrast 0.2, saturation 0.1), followed by ImageNet normalization"

**Why flagged:** Common training protocol description.

### Flag M4 — Sources 18, 22 (inria.hal.science; repositum.tuwien.at)
**Section: III-C Training Protocol continued**

**Flagged text:**
> "AdamW [25] with a learning rate of 10⁻⁴ and weight decay of 10⁻⁴, a ReduceLROnPlateau scheduler (patience 5, factor 0.5), and early stopping with patience 10 on validation loss"

**Why flagged:** Common optimizer setup description.

### Flag M5 — Source 25 (pdffox.com, <1%)
**Section: III-C continued**

**Flagged text:**
> "Batch size is 32 and the maximum epoch count is 30. All training runs use PyTorch 2.0 [26] on an NVIDIA GPU with CUDA"

### Flag M6 — Source 17 (Dublin Business School, <1%)
**Section: III-D Evaluation Metrics**

**Flagged text:**
> "We report overall accuracy (OA), macro-averaged F1-score (unweighted mean across classes), and Cohen's kappa (κ) [27]. Kappa measures agreement beyond what chance would produce and is standard in remote sensing accuracy assessment"

**Why flagged:** Standard metrics description.

### Flag M7 — Source 1 (arxiv.org)
**Section: III-D-2 Statistical Significance**

**Flagged text:**
> "We use McNemar's test [19], [30] with continuity correction to check whether accuracy differences between pairs of models are statistically real"

---

## IV. RESULTS

*(Mostly clean — results tables and figures are original data. Minor flags only.)*

### Flag V1 — Source 1 (arxiv.org)
**Section: IV-D Statistical Significance**

**Flagged text (Fig. 10 caption area):**
> "Green = p > 0.05 (no significant difference); red = p < 0.05 (significant difference)"

**Why flagged:** Standard statistical description.

---

## V. DISCUSSION

### Flag S1 — Source 1 (arxiv.org)
**Section: V-A Architecture Family Comparison**

**Flagged text:**
> "the accuracy gap between CNNs and transformers closes once CNNs adopt modern training practices"

**Why flagged:** Closely mirrors ConvNeXt paper claim.

### Flag S2 — Source 16 (Asad Ullah Haider et al.)
**Section: V-A continued**

**Flagged text:**
> "When images are resized to 224×224 pixels and each patch contains a single land-use type, local texture and color patterns may be sufficient for classification"

### Flag S3 — Source 1 (arxiv.org)
**Section: V-A continued**

**Flagged text:**
> "Self-attention's ability to relate distant patches may become more useful for tasks that require understanding spatial layout, such as detecting a harbor by noticing boats near a dock, rather than just recognizing a uniform texture"

---

## VI. CONCLUSION

### Flag C1 — Source 1 (arxiv.org)
**Flagged text:**
> "investing effort in training strategy and data quality is likely to matter more than switching to a fancier architecture. Future work should move to harder evaluation settings (larger datasets, cross-domain transfer, limited labels) where architectural differences may become more pronounced"

---

## REFERENCES

Reference entries themselves are flagged because they are identical across papers. **This is normal and expected — do NOT change reference formatting.** Most reviewers and Turnitin users exclude the reference list from the similarity count.

Flagged references: [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [24]

---

## 📋 PRIORITY PARAPHRASE PLAN

### 🔴 High Priority (Source 1 — arxiv.org = 7% alone)
These sections contribute the most to your similarity score:

1. **Section II-A** (CNN-Based Scene Classification) — Rewrite all model descriptions in your own words
2. **Section II-B** (Transformer-Based Approaches) — Rewrite ViT and Swin descriptions completely
3. **Section II-C** (Modernized CNNs) — Rewrite ConvNeXt description
4. **Section V-A** (Architecture Family Comparison) — Rephrase discussion points
5. **Section VI** (Conclusion) — Rephrase future work sentence

### 🟡 Medium Priority (Sources 2–5, combined ~5%)
6. **Section II-D** (Benchmark Datasets) — Rephrase dataset statistics sentences
7. **Section III-C** (Training Protocol) — Restructure the augmentation/optimizer descriptions
8. **Section III-D** (Evaluation Metrics) — Rephrase metrics definitions

### 🟢 Low Priority (<1% sources)
9. **Abstract** — Minor rephrasing of standard terms
10. **Section I** (Introduction) — Rephrase opening sentences and transfer learning description

### 🚨 MUST FIX
11. **Hidden text flag** — Remove any white/invisible text from the document immediately

---

## 💡 General Tips to Reduce Similarity

- **Related Work is your biggest problem** — don't just describe what each paper/model does; instead, compare them, critique them, connect them to YOUR research question
- **Dataset descriptions**: Instead of "X has N images across K classes at R resolution," try restructuring: "We selected X for its K-class taxonomy covering ... at R spatial resolution (N patches total)"
- **Training protocol**: Group multiple settings into a table and refer to it, rather than writing them all out in prose
- **Don't just synonym-swap** — restructure sentences entirely. Change passive to active voice, merge sentences, split sentences, change the order of information
- **Add your own analysis/interpretation** between descriptions of existing work
