"""Test all module imports and model creation."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATASETS, MODELS, TRAINING, RESULTS_DIR
print('Config OK:', len(MODELS), 'models,', len(DATASETS), 'datasets')

from modules.dataset_loader import SceneDataset, get_train_transform, get_test_transform
print('Dataset loader OK')

from modules.models import create_model, count_parameters
print('Models OK')

from modules.trainer import train_model
print('Trainer OK')

from modules.evaluator import evaluate_model, mcnemar_test
print('Evaluator OK')

from modules.visualizer import plot_confusion_matrix, plot_training_curves
print('Visualizer OK')

import torch
for name in MODELS:
    m = create_model(name, 10, pretrained=False)
    t, _ = count_parameters(m)
    print(f'  {name}: {t/1e6:.1f}M params')
    del m
print('All models created successfully')
print('ALL IMPORTS PASSED')
