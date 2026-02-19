# Remote Sensing Scene Classification Benchmark

Systematic comparison of CNN and Vision Transformer architectures on three
standard remote sensing scene classification benchmarks.

## Datasets

| Dataset | Classes | Images | Resolution | Source |
|---------|---------|--------|------------|--------|
| EuroSAT | 10 | 27,000 | 64x64 | Sentinel-2 multispectral |
| NWPU-RESISC45 | 45 | 31,500 | 256x256 | Google Earth RGB |
| AID | 30 | 10,000 | 600x600 | Google Earth RGB |

## Models

| Architecture | Type | Params (M) |
|-------------|------|------------|
| ResNet-50 | CNN | 25.6 |
| ResNet-101 | CNN | 44.5 |
| DenseNet-121 | CNN | 8.0 |
| EfficientNet-B0 | CNN | 5.3 |
| EfficientNet-B3 | CNN | 12.2 |
| ViT-B/16 | Transformer | 86.6 |
| Swin-T | Transformer | 28.3 |
| ConvNeXt-Tiny | CNN-Modern | 28.6 |

## Quick Start

```bash
conda activate landcover_jambi

# 1. Download datasets
python scripts/download_datasets.py

# 2. Run all experiments
python scripts/train_all_experiments.py

# 3. Generate publication outputs
python scripts/generate_publication_outputs.py
python scripts/generate_statistical_analysis.py
```

## References

- Helber et al. (2019). EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification. IEEE JSTARS.
- Cheng et al. (2017). Remote Sensing Image Scene Classification: Benchmark and State of the Art. Proc. IEEE.
- Xia et al. (2017). AID: A Benchmark Data Set for Performance Evaluation of Aerial Scene Classification. IEEE TGRS.
