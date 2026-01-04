#!/usr/bin/env python3
"""
Model Registry - Central Registry for All Deep Learning Models
================================================================

Maintains metadata for all available models including:
- Architecture family (ResNet, EfficientNet, ConvNeXt, DenseNet, Inception)
- Parameter counts and computational requirements
- Expected performance characteristics
- Training recommendations

Easy to extend - just add new entries to MODEL_REGISTRY dictionary.

Author: Claude Sonnet 4.5
Date: 2026-01-04
"""

# ============================================================================
# MODEL REGISTRY - All Available Models
# ============================================================================

MODEL_REGISTRY = {
    # ========================================================================
    # ResNet Family - Deep Residual Learning
    # ========================================================================
    'resnet18': {
        'family': 'resnet',
        'display_name': 'ResNet-18',
        'params': 11.7e6,
        'flops': 1.8e9,
        'depth': 18,
        'description': 'ResNet-18: Lightweight residual network',
        'paper': 'He et al. (2016) - Deep Residual Learning',
        'best_for': 'Fast training, baseline comparison',
        'data_efficiency': 'Good',
        'expected_acc_range': (75, 78),
        'training_time_factor': 1.0,  # Relative to ResNet50
    },

    'resnet34': {
        'family': 'resnet',
        'display_name': 'ResNet-34',
        'params': 21.8e6,
        'flops': 3.7e9,
        'depth': 34,
        'description': 'ResNet-34: Deeper residual network',
        'paper': 'He et al. (2016) - Deep Residual Learning',
        'best_for': 'Balanced depth and speed',
        'data_efficiency': 'Good',
        'expected_acc_range': (75, 78),
        'training_time_factor': 1.3,
    },

    'resnet50': {
        'family': 'resnet',
        'display_name': 'ResNet-50',
        'params': 25.6e6,
        'flops': 4.1e9,
        'depth': 50,
        'description': 'ResNet-50: Standard deep residual network (baseline)',
        'paper': 'He et al. (2016) - Deep Residual Learning',
        'best_for': 'Standard baseline, proven performance',
        'data_efficiency': 'Good',
        'expected_acc_range': (76, 79),
        'training_time_factor': 2.0,
        'notes': 'Current baseline - 77.23% accuracy achieved'
    },

    'resnet101': {
        'family': 'resnet',
        'display_name': 'ResNet-101',
        'params': 44.5e6,
        'flops': 7.8e9,
        'depth': 101,
        'description': 'ResNet-101: Very deep residual network',
        'paper': 'He et al. (2016) - Deep Residual Learning',
        'best_for': 'Maximum capacity (but diminishing returns)',
        'data_efficiency': 'Moderate',
        'expected_acc_range': (76, 79),
        'training_time_factor': 3.0,
        'notes': 'Similar performance to ResNet50 on this dataset'
    },

    'resnet152': {
        'family': 'resnet',
        'display_name': 'ResNet-152',
        'params': 60.2e6,
        'flops': 11.6e9,
        'depth': 152,
        'description': 'ResNet-152: Extremely deep residual network',
        'paper': 'He et al. (2016) - Deep Residual Learning',
        'best_for': 'Very large datasets (overkill for 100k samples)',
        'data_efficiency': 'Poor',
        'expected_acc_range': (76, 79),
        'training_time_factor': 4.0,
        'notes': 'Diminishing returns - not recommended for this dataset size'
    },

    # ========================================================================
    # EfficientNet Family - Compound Scaling
    # ========================================================================
    'efficientnet_b0': {
        'family': 'efficientnet',
        'display_name': 'EfficientNet-B0',
        'params': 5.3e6,
        'flops': 0.39e9,
        'depth': 18,
        'description': 'EfficientNet-B0: Ultra-lightweight efficient network',
        'paper': 'Tan & Le (2019) - EfficientNet',
        'best_for': 'Resource-constrained environments',
        'data_efficiency': 'Excellent',
        'expected_acc_range': (74, 77),
        'training_time_factor': 0.8,
    },

    'efficientnet_b1': {
        'family': 'efficientnet',
        'display_name': 'EfficientNet-B1',
        'params': 7.8e6,
        'flops': 0.70e9,
        'depth': 20,
        'description': 'EfficientNet-B1: Lightweight efficient network',
        'paper': 'Tan & Le (2019) - EfficientNet',
        'best_for': 'Balance of efficiency and accuracy',
        'data_efficiency': 'Excellent',
        'expected_acc_range': (75, 78),
        'training_time_factor': 1.0,
    },

    'efficientnet_b3': {
        'family': 'efficientnet',
        'display_name': 'EfficientNet-B3',
        'params': 12.0e6,
        'flops': 1.8e9,
        'depth': 26,
        'description': 'EfficientNet-B3: Compound-scaled efficient network',
        'paper': 'Tan & Le (2019) - EfficientNet',
        'best_for': '100k samples, best efficiency/accuracy trade-off',
        'data_efficiency': 'Excellent',
        'expected_acc_range': (76, 79),
        'training_time_factor': 1.5,
        'recommended': True,
        'notes': 'Best for limited data (100k samples) - proven 62% on CIFAR-100'
    },

    # ========================================================================
    # ConvNeXt Family - Modernized CNN
    # ========================================================================
    'convnext_tiny': {
        'family': 'convnext',
        'display_name': 'ConvNeXt-Tiny',
        'params': 28.6e6,
        'flops': 4.5e9,
        'depth': 28,
        'description': 'ConvNeXt-Tiny: Modern pure CNN, state-of-art for small data',
        'paper': 'Liu et al. (2022) - A ConvNet for the 2020s',
        'best_for': 'Small datasets (<1M samples), modern architecture',
        'data_efficiency': 'Excellent',
        'expected_acc_range': (78, 81),
        'training_time_factor': 2.5,
        'recommended': True,
        'notes': 'LIKELY WINNER - 49% faster than Swin, best for small data'
    },

    'convnext_small': {
        'family': 'convnext',
        'display_name': 'ConvNeXt-Small',
        'params': 50.2e6,
        'flops': 8.7e9,
        'depth': 30,
        'description': 'ConvNeXt-Small: Larger modern CNN',
        'paper': 'Liu et al. (2022) - A ConvNet for the 2020s',
        'best_for': 'When you need more capacity',
        'data_efficiency': 'Good',
        'expected_acc_range': (78, 82),
        'training_time_factor': 3.5,
    },

    # ========================================================================
    # DenseNet Family - Dense Connections
    # ========================================================================
    'densenet121': {
        'family': 'densenet',
        'display_name': 'DenseNet-121',
        'params': 8.0e6,
        'flops': 2.9e9,
        'depth': 121,
        'description': 'DenseNet-121: Dense feature reuse, lightweight',
        'paper': 'Huang et al. (2017) - Densely Connected CNNs',
        'best_for': 'Feature reuse, memory efficiency',
        'data_efficiency': 'Good',
        'expected_acc_range': (74, 77),
        'training_time_factor': 1.8,
        'recommended': True,
        'notes': 'Most parameter-efficient - only 8M params'
    },

    'densenet169': {
        'family': 'densenet',
        'display_name': 'DenseNet-169',
        'params': 14.1e6,
        'flops': 3.4e9,
        'depth': 169,
        'description': 'DenseNet-169: Deeper dense network',
        'paper': 'Huang et al. (2017) - Densely Connected CNNs',
        'best_for': 'More capacity with dense connections',
        'data_efficiency': 'Good',
        'expected_acc_range': (75, 78),
        'training_time_factor': 2.2,
    },

    # ========================================================================
    # Inception Family - Multi-scale Feature Extraction
    # ========================================================================
    'inception_v3': {
        'family': 'inception',
        'display_name': 'Inception-V3',
        'params': 23.8e6,
        'flops': 5.7e9,
        'depth': 48,
        'description': 'Inception-V3: Multi-scale parallel feature extraction',
        'paper': 'Szegedy et al. (2016) - Rethinking Inception',
        'best_for': 'Multi-scale features (water, crops, buildings)',
        'data_efficiency': 'Good',
        'expected_acc_range': (77, 80),
        'training_time_factor': 2.3,
        'recommended': True,
        'notes': 'UNIQUE - 92% on land cover, multi-scale crucial for diverse classes'
    },
}

# ============================================================================
# MODEL FAMILIES - Grouping for Analysis
# ============================================================================

MODEL_FAMILIES = {
    'resnet': {
        'name': 'ResNet Family',
        'description': 'Deep residual learning with skip connections',
        'key_innovation': 'Skip connections enable very deep networks',
        'inductive_bias': 'Strong local spatial structure',
    },
    'efficientnet': {
        'name': 'EfficientNet Family',
        'description': 'Compound scaling of depth, width, and resolution',
        'key_innovation': 'Balanced scaling across all dimensions',
        'inductive_bias': 'Efficient compound scaling',
    },
    'convnext': {
        'name': 'ConvNeXt Family',
        'description': 'Modernized pure CNN without transformers',
        'key_innovation': 'Modern CNN design principles from transformers',
        'inductive_bias': 'Depthwise convolutions, layer normalization',
    },
    'densenet': {
        'name': 'DenseNet Family',
        'description': 'Dense connections between all layers',
        'key_innovation': 'Feature reuse via dense connections',
        'inductive_bias': 'Strong feature reuse, gradient flow',
    },
    'inception': {
        'name': 'Inception Family',
        'description': 'Multi-scale parallel feature extraction',
        'key_innovation': 'Parallel convolutions at multiple scales',
        'inductive_bias': 'Multi-scale spatial features',
    },
}

# ============================================================================
# RECOMMENDED MODELS - For Different Use Cases
# ============================================================================

RECOMMENDED_MODELS = {
    'small_data': ['convnext_tiny', 'efficientnet_b3', 'densenet121'],
    'efficiency': ['efficientnet_b0', 'efficientnet_b1', 'densenet121'],
    'accuracy': ['convnext_tiny', 'inception_v3', 'resnet50'],
    'baseline': ['resnet50'],
    'deployment': ['efficientnet_b3', 'densenet121'],
    'research_comparison': ['resnet50', 'efficientnet_b3', 'convnext_tiny',
                           'densenet121', 'inception_v3'],
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_model_info(model_name):
    """Get metadata for a specific model."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not found in registry. "
                        f"Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name]


def list_models(family=None, recommended_only=False):
    """List available models, optionally filtered by family."""
    models = MODEL_REGISTRY.keys()

    if family:
        models = [m for m in models if MODEL_REGISTRY[m]['family'] == family]

    if recommended_only:
        models = [m for m in models if MODEL_REGISTRY[m].get('recommended', False)]

    return list(models)


def get_family_models(family):
    """Get all models in a specific family."""
    return [m for m in MODEL_REGISTRY.keys()
            if MODEL_REGISTRY[m]['family'] == family]


def print_model_summary(model_name):
    """Print a formatted summary of a model."""
    info = get_model_info(model_name)

    print(f"\n{'='*70}")
    print(f"{info['display_name']}")
    print(f"{'='*70}")
    print(f"Family:          {MODEL_FAMILIES[info['family']]['name']}")
    print(f"Parameters:      {info['params']/1e6:.1f}M")
    print(f"FLOPs:           {info['flops']/1e9:.1f}G")
    print(f"Depth:           {info['depth']} layers")
    print(f"Description:     {info['description']}")
    print(f"Best for:        {info['best_for']}")
    print(f"Data efficiency: {info['data_efficiency']}")
    print(f"Expected range:  {info['expected_acc_range'][0]}-{info['expected_acc_range'][1]}%")

    if info.get('recommended'):
        print(f"⭐ RECOMMENDED")

    if info.get('notes'):
        print(f"\nNotes: {info['notes']}")

    print(f"{'='*70}\n")


def compare_models(model_names):
    """Generate comparison table for multiple models."""
    import pandas as pd

    data = []
    for name in model_names:
        info = get_model_info(name)
        data.append({
            'Model': info['display_name'],
            'Family': info['family'].upper(),
            'Params (M)': f"{info['params']/1e6:.1f}",
            'FLOPs (G)': f"{info['flops']/1e9:.1f}",
            'Depth': info['depth'],
            'Expected Acc (%)': f"{info['expected_acc_range'][0]}-{info['expected_acc_range'][1]}",
            'Best For': info['best_for'],
        })

    df = pd.DataFrame(data)
    return df


if __name__ == '__main__':
    """Demo: Show all recommended models."""
    print("\n" + "="*80)
    print("MODEL REGISTRY - RECOMMENDED MODELS FOR 100K SAMPLES")
    print("="*80)

    recommended = RECOMMENDED_MODELS['research_comparison']

    print(f"\nRecommended for research comparison ({len(recommended)} models):")
    for model in recommended:
        info = MODEL_REGISTRY[model]
        print(f"\n  • {info['display_name']}")
        print(f"    {info['description']}")
        print(f"    Expected: {info['expected_acc_range'][0]}-{info['expected_acc_range'][1]}%")

    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)

    df = compare_models(recommended)
    print(df.to_string(index=False))

    print("\n" + "="*80)
