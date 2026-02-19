"""
Central Configuration for Scene Classification Benchmark
"""

import os

# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

DATASET_PATHS = {
    'eurosat': os.path.join(DATA_DIR, 'eurosat'),
    'ucmerced': os.path.join(DATA_DIR, 'ucmerced'),
    'nwpu_resisc45': os.path.join(DATA_DIR, 'nwpu_resisc45'),
    'aid': os.path.join(DATA_DIR, 'aid'),
}

# ============================================================
# Dataset Configuration
# ============================================================

DATASETS = {
    'eurosat': {
        'num_classes': 10,
        'input_size': 64,
        'channels': 3,       # Using RGB subset for fair comparison
        'train_ratio': 0.8,
        'class_names': [
            'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
            'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
            'River', 'SeaLake'
        ],
    },
    'nwpu_resisc45': {
        'num_classes': 45,
        'input_size': 256,
        'channels': 3,
        'train_ratio': 0.8,
        'class_names': [
            'airplane', 'airport', 'baseball_diamond', 'basketball_court',
            'beach', 'bridge', 'chaparral', 'church', 'circular_farmland',
            'cloud', 'commercial_area', 'dense_residential', 'desert',
            'forest', 'freeway', 'golf_course', 'ground_track_field',
            'harbor', 'industrial_area', 'intersection', 'island', 'lake',
            'meadow', 'medium_residential', 'mobile_home_park', 'mountain',
            'overpass', 'palace', 'parking_lot', 'railway', 'railway_station',
            'rectangular_farmland', 'river', 'roundabout', 'runway',
            'sea_ice', 'ship', 'snowberg', 'sparse_residential', 'stadium',
            'storage_tank', 'tennis_court', 'terrace', 'thermal_power_station',
            'wetland'
        ],
    },
    'ucmerced': {
        'num_classes': 21,
        'input_size': 256,
        'channels': 3,
        'train_ratio': 0.8,
        'class_names': [
            'agricultural', 'airplane', 'baseballdiamond', 'beach',
            'buildings', 'chaparral', 'denseresidential', 'forest',
            'freeway', 'golfcourse', 'harbor', 'intersection',
            'mediumresidential', 'mobilehomepark', 'overpass',
            'parkinglot', 'river', 'runway', 'sparseresidential',
            'storagetanks', 'tenniscourt'
        ],
    },
    'aid': {
        'num_classes': 30,
        'input_size': 600,
        'channels': 3,
        'train_ratio': 0.8,
        'class_names': [
            'Airport', 'BareLand', 'BaseballField', 'Beach', 'Bridge',
            'Center', 'Church', 'Commercial', 'DenseResidential', 'Desert',
            'Farmland', 'Forest', 'Industrial', 'Meadow', 'MediumResidential',
            'Mountain', 'Park', 'Parking', 'Playground', 'Pond', 'Port',
            'RailwayStation', 'Resort', 'River', 'School', 'SparseResidential',
            'Square', 'Stadium', 'StorageTanks', 'Viaduct'
        ],
    },
}

# ============================================================
# Model Configuration
# ============================================================

MODELS = {
    'resnet50': {'family': 'cnn', 'params_m': 25.6},
    'resnet101': {'family': 'cnn', 'params_m': 44.5},
    'densenet121': {'family': 'cnn', 'params_m': 8.0},
    'efficientnet_b0': {'family': 'cnn', 'params_m': 5.3},
    'efficientnet_b3': {'family': 'cnn', 'params_m': 12.2},
    'vit_b_16': {'family': 'transformer', 'params_m': 86.6},
    'swin_t': {'family': 'transformer', 'params_m': 28.3},
    'convnext_tiny': {'family': 'cnn_modern', 'params_m': 28.6},
}

# ============================================================
# Training Configuration
# ============================================================

TRAINING = {
    'batch_size': 32,
    'epochs': 30,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'scheduler_patience': 5,
    'scheduler_factor': 0.5,
    'early_stopping_patience': 10,
    'num_workers': 4,
    'image_size': 224,      # Resize all inputs to 224x224 for fair comparison
    'random_seed': 42,
}

# ============================================================
# Augmentation
# ============================================================

AUGMENTATION = {
    'horizontal_flip': True,
    'vertical_flip': True,
    'random_rotation': 15,
    'color_jitter_brightness': 0.2,
    'color_jitter_contrast': 0.2,
    'color_jitter_saturation': 0.1,
    'normalize_mean': [0.485, 0.456, 0.406],   # ImageNet stats
    'normalize_std': [0.229, 0.224, 0.225],
}
