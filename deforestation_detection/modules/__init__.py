"""
Deforestation Detection Modules
Multi-temporal change detection for Jambi Province, Indonesia

Modules:
    data_loader          - Load multi-temporal Sentinel-2, Hansen GFC, ForestNet
    feature_engineering  - Spectral indices + temporal change features
    preprocessor         - Patch extraction, change label creation
    change_detector      - Post-classification comparison logic
    siamese_network      - Siamese CNN architecture (ResNet-50 backbone)
    model_trainer        - ML training (Random Forest baseline)
    deep_learning_trainer - DL training loop (ResNet + Siamese)
    visualizer           - Publication-quality plots
"""

__version__ = '1.0.0'
