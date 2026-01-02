#!/usr/bin/env python3
"""
Comprehensive System Verification
==================================

Tests all major components after big changes to ensure nothing broke.

Tests:
1. Module imports
2. Naming standards
3. Cloud removal strategies
4. File structure
5. Download script
6. Classification pipeline (dry run)

Author: Claude Sonnet 4.5
Date: 2026-01-02
"""

import sys
import os

# Add project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import traceback

# ============================================================================
# TEST CONFIGURATION
# ============================================================================

TEST_RESULTS = {
    'passed': [],
    'failed': [],
    'warnings': []
}

def test_passed(test_name):
    """Mark test as passed."""
    TEST_RESULTS['passed'].append(test_name)
    print(f"  ✅ {test_name}")

def test_failed(test_name, error):
    """Mark test as failed."""
    TEST_RESULTS['failed'].append((test_name, error))
    print(f"  ❌ {test_name}: {error}")

def test_warning(test_name, warning):
    """Mark test warning."""
    TEST_RESULTS['warnings'].append((test_name, warning))
    print(f"  ⚠️  {test_name}: {warning}")

# ============================================================================
# TEST 1: MODULE IMPORTS
# ============================================================================

def test_module_imports():
    """Test that all modules can be imported."""

    print("\n" + "="*80)
    print("TEST 1: MODULE IMPORTS")
    print("="*80)

    modules_to_test = [
        ('modules.cloud_removal', 'CloudRemovalConfig'),
        ('modules.naming_standards', 'create_sentinel_name'),
        ('modules.data_loader', 'load_klhk_data'),
        ('modules.feature_engineering', 'calculate_spectral_indices'),
        ('modules.preprocessor', 'rasterize_klhk'),
        ('modules.model_trainer', 'get_classifiers'),
        ('modules.visualizer', 'plot_classifier_comparison'),
    ]

    for module_name, class_or_func in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[class_or_func])
            assert hasattr(module, class_or_func)
            test_passed(f"{module_name}.{class_or_func}")
        except Exception as e:
            test_failed(f"{module_name}.{class_or_func}", str(e))

# ============================================================================
# TEST 2: NAMING STANDARDS
# ============================================================================

def test_naming_standards():
    """Test naming standards module functions."""

    print("\n" + "="*80)
    print("TEST 2: NAMING STANDARDS")
    print("="*80)

    try:
        from modules.naming_standards import (
            create_sentinel_name,
            create_rgb_name,
            create_classification_name,
            parse_standard_name
        )

        # Test Sentinel naming
        name = create_sentinel_name('province', 20, '2024dry', 'percentile_25', 1)
        assert name == 'sentinel_province_20m_2024dry_p25-tile1'
        test_passed("create_sentinel_name")

        # Test RGB naming
        name = create_rgb_name('city', 10, '2024dry', 'natural')
        assert name == 'rgb_city_10m_2024dry_natural'
        test_passed("create_rgb_name")

        # Test Classification naming
        name = create_classification_name('province', 20, '2024dry', 'Random Forest')
        assert name == 'classification_province_20m_2024dry_rf'
        test_passed("create_classification_name")

        # Test parsing
        parsed = parse_standard_name('sentinel_province_20m_2024dry_p25-tile1.tif')
        assert parsed['category'] == 'sentinel'
        assert parsed['region'] == 'province'
        assert parsed['tile'] == 1
        test_passed("parse_standard_name")

    except Exception as e:
        test_failed("Naming standards", str(e))
        traceback.print_exc()

# ============================================================================
# TEST 3: CLOUD REMOVAL STRATEGIES
# ============================================================================

def test_cloud_removal_strategies():
    """Test cloud removal strategy system."""

    print("\n" + "="*80)
    print("TEST 3: CLOUD REMOVAL STRATEGIES")
    print("="*80)

    try:
        from modules.cloud_removal import CloudRemovalConfig

        strategies = ['current', 'percentile_25', 'kalimantan',
                     'balanced', 'pan_tropical', 'conservative']

        for strategy in strategies:
            try:
                config = CloudRemovalConfig.get_strategy(strategy)
                assert 'name' in config
                assert 'cloud_score_threshold' in config
                assert 'composite_method' in config
                test_passed(f"Strategy: {strategy}")
            except Exception as e:
                test_failed(f"Strategy: {strategy}", str(e))

    except Exception as e:
        test_failed("Cloud removal strategies", str(e))
        traceback.print_exc()

# ============================================================================
# TEST 4: FILE STRUCTURE
# ============================================================================

def test_file_structure():
    """Test that required directories and files exist."""

    print("\n" + "="*80)
    print("TEST 4: FILE STRUCTURE")
    print("="*80)

    required_dirs = [
        'modules',
        'scripts',
        'data',
        'results',
        'docs',
    ]

    required_files = [
        'modules/cloud_removal.py',
        'modules/naming_standards.py',
        'modules/data_loader.py',
        'modules/feature_engineering.py',
        'modules/preprocessor.py',
        'modules/model_trainer.py',
        'modules/visualizer.py',
        'scripts/download_sentinel2_flexible.py',
        'scripts/test_cloud_strategies.py',
        'scripts/check_task_status.py',
        'docs/NAMING_STANDARDS.md',
        'docs/FLEXIBLE_DOWNLOAD_GUIDE.md',
    ]

    for dir_path in required_dirs:
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            test_passed(f"Directory: {dir_path}")
        else:
            test_failed(f"Directory: {dir_path}", "Not found")

    for file_path in required_files:
        if os.path.exists(file_path) and os.path.isfile(file_path):
            test_passed(f"File: {file_path}")
        else:
            test_warning(f"File: {file_path}", "Not found")

# ============================================================================
# TEST 5: DOWNLOAD SCRIPT INTEGRATION
# ============================================================================

def test_download_script():
    """Test download script can be imported and configured."""

    print("\n" + "="*80)
    print("TEST 5: DOWNLOAD SCRIPT INTEGRATION")
    print("="*80)

    try:
        # Import download script as module
        sys.path.insert(0, 'scripts')

        # Check flexible download script
        try:
            import download_sentinel2_flexible
            test_passed("Import: download_sentinel2_flexible.py")

            # Check preset configuration
            assert hasattr(download_sentinel2_flexible, 'CONFIG_PRESETS')
            presets = download_sentinel2_flexible.CONFIG_PRESETS
            assert 'city_10m' in presets
            assert 'province_20m' in presets
            test_passed("CONFIG_PRESETS structure")

        except Exception as e:
            test_failed("download_sentinel2_flexible.py", str(e))

    except Exception as e:
        test_failed("Download script integration", str(e))
        traceback.print_exc()

# ============================================================================
# TEST 6: DATA LOADER
# ============================================================================

def test_data_loader():
    """Test data loader module (dry run - no actual data)."""

    print("\n" + "="*80)
    print("TEST 6: DATA LOADER MODULE")
    print("="*80)

    try:
        from modules.data_loader import (
            get_sentinel2_band_names,
            KLHK_TO_SIMPLIFIED,
            CLASS_NAMES
        )

        # Test band names
        bands = get_sentinel2_band_names()
        assert len(bands) == 10
        test_passed("get_sentinel2_band_names")

        # Test KLHK mapping
        assert isinstance(KLHK_TO_SIMPLIFIED, dict)
        assert len(KLHK_TO_SIMPLIFIED) > 0
        test_passed("KLHK_TO_SIMPLIFIED mapping")

        # Test class names
        assert isinstance(CLASS_NAMES, dict)
        assert 0 in CLASS_NAMES  # Water
        assert 1 in CLASS_NAMES  # Trees
        test_passed("CLASS_NAMES mapping")

    except Exception as e:
        test_failed("Data loader module", str(e))
        traceback.print_exc()

# ============================================================================
# TEST 7: FEATURE ENGINEERING
# ============================================================================

def test_feature_engineering():
    """Test feature engineering module (dry run)."""

    print("\n" + "="*80)
    print("TEST 7: FEATURE ENGINEERING MODULE")
    print("="*80)

    try:
        from modules.feature_engineering import get_all_feature_names

        # Test feature names
        features = get_all_feature_names()
        assert len(features) == 23  # 10 bands + 13 indices
        test_passed("get_all_feature_names (23 features)")

        # Check specific features
        assert 'B2_Blue' in features or 'B2' in features
        assert 'NDVI' in features
        assert 'EVI' in features
        test_passed("Feature name content check")

    except Exception as e:
        test_failed("Feature engineering module", str(e))
        traceback.print_exc()

# ============================================================================
# TEST 8: MODEL TRAINER
# ============================================================================

def test_model_trainer():
    """Test model trainer module (dry run)."""

    print("\n" + "="*80)
    print("TEST 8: MODEL TRAINER MODULE")
    print("="*80)

    try:
        from modules.model_trainer import get_classifiers

        # Test classifier creation
        classifiers = get_classifiers(include_slow=False)
        assert 'Random Forest' in classifiers
        assert 'Extra Trees' in classifiers
        test_passed("get_classifiers (fast models)")

        # Test with slow models
        classifiers_all = get_classifiers(include_slow=True)
        assert len(classifiers_all) >= len(classifiers)
        test_passed("get_classifiers (all models)")

    except Exception as e:
        test_failed("Model trainer module", str(e))
        traceback.print_exc()

# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_all_tests():
    """Run all verification tests."""

    print("\n" + "="*80)
    print("COMPREHENSIVE SYSTEM VERIFICATION")
    print("="*80)
    print("\nRunning all tests...\n")

    # Run all tests
    test_module_imports()
    test_naming_standards()
    test_cloud_removal_strategies()
    test_file_structure()
    test_download_script()
    test_data_loader()
    test_feature_engineering()
    test_model_trainer()

    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    total_tests = (len(TEST_RESULTS['passed']) +
                   len(TEST_RESULTS['failed']) +
                   len(TEST_RESULTS['warnings']))

    print(f"\nTotal Tests: {total_tests}")
    print(f"  ✅ Passed: {len(TEST_RESULTS['passed'])}")
    print(f"  ❌ Failed: {len(TEST_RESULTS['failed'])}")
    print(f"  ⚠️  Warnings: {len(TEST_RESULTS['warnings'])}")

    if TEST_RESULTS['failed']:
        print("\n" + "-"*80)
        print("FAILED TESTS:")
        print("-"*80)
        for test_name, error in TEST_RESULTS['failed']:
            print(f"  ❌ {test_name}")
            print(f"     Error: {error}")

    if TEST_RESULTS['warnings']:
        print("\n" + "-"*80)
        print("WARNINGS:")
        print("-"*80)
        for test_name, warning in TEST_RESULTS['warnings']:
            print(f"  ⚠️  {test_name}")
            print(f"     Warning: {warning}")

    print("\n" + "="*80)

    if len(TEST_RESULTS['failed']) == 0:
        print("✅ ALL CRITICAL TESTS PASSED!")
        print("="*80)
        return 0
    else:
        print("❌ SOME TESTS FAILED - CHECK ABOVE")
        print("="*80)
        return 1

if __name__ == '__main__':
    exit_code = run_all_tests()
    sys.exit(exit_code)
