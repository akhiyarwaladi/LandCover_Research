"""
Change Detector Module
======================

Post-classification comparison logic and change analysis utilities
for multi-temporal deforestation detection.
"""

import numpy as np


# Land cover class IDs from parent project
FOREST_CLASS = 1  # Trees/Forest
CROP_CLASS = 4    # Crops/Agriculture
SHRUB_CLASS = 5   # Shrub/Scrub
BUILT_CLASS = 6   # Built Area
BARE_CLASS = 7    # Bare Ground
WATER_CLASS = 0   # Water

CLASS_NAMES = {
    0: 'Water',
    1: 'Trees/Forest',
    4: 'Crops/Agriculture',
    5: 'Shrub/Scrub',
    6: 'Built Area',
    7: 'Bare Ground',
}

# Pixel area at 20m resolution
PIXEL_AREA_HA = 0.04     # hectares
PIXEL_AREA_KM2 = 0.0004  # square kilometers


def post_classification_comparison(map_t1, map_t2, verbose=True):
    """
    Detect deforestation via post-classification comparison.

    Deforestation = Forest at T1 AND Non-forest at T2.

    Args:
        map_t1: (H, W) classified land cover map at time 1
        map_t2: (H, W) classified land cover map at time 2
        verbose: Print statistics

    Returns:
        dict with keys:
            'deforestation': (H, W) binary deforestation map
            'forest_to_class': (H, W) target class where deforestation occurred
            'transition_matrix': 2D array of from-to transitions
    """
    if verbose:
        print("Running post-classification comparison...")

    forest_t1 = (map_t1 == FOREST_CLASS)
    forest_t2 = (map_t2 == FOREST_CLASS)

    # Deforestation: forest at T1, non-forest at T2
    deforestation = forest_t1 & ~forest_t2

    # What class replaced the forest?
    forest_to_class = np.where(deforestation, map_t2, -1)

    # Full transition matrix
    transition_matrix = compute_transition_matrix(map_t1, map_t2)

    if verbose:
        n_defor = np.sum(deforestation)
        area_ha = n_defor * PIXEL_AREA_HA
        print(f"  Deforestation pixels: {n_defor:,}")
        print(f"  Deforestation area: {area_ha:,.0f} ha ({area_ha / 100:,.1f} km2)")

        # Forest loss destinations
        if n_defor > 0:
            print("  Forest converted to:")
            for cls, name in CLASS_NAMES.items():
                if cls == FOREST_CLASS:
                    continue
                n = np.sum(forest_to_class == cls)
                if n > 0:
                    pct = 100 * n / n_defor
                    print(f"    {name}: {n:,} pixels ({pct:.1f}%)")

    return {
        'deforestation': deforestation.astype(np.uint8),
        'forest_to_class': forest_to_class,
        'transition_matrix': transition_matrix,
    }


def compute_transition_matrix(map_t1, map_t2, classes=None):
    """
    Compute from-to transition matrix between two classified maps.

    Args:
        map_t1: (H, W) classified map at time 1
        map_t2: (H, W) classified map at time 2
        classes: List of class IDs (default: auto-detect)

    Returns:
        dict with 'matrix' (2D array), 'classes' (list), 'class_names' (list)
    """
    if classes is None:
        classes = sorted(set(np.unique(map_t1)) | set(np.unique(map_t2)))
        classes = [c for c in classes if c >= 0]

    n = len(classes)
    matrix = np.zeros((n, n), dtype=np.int64)

    class_to_idx = {c: i for i, c in enumerate(classes)}

    for from_cls in classes:
        for to_cls in classes:
            mask = (map_t1 == from_cls) & (map_t2 == to_cls)
            matrix[class_to_idx[from_cls], class_to_idx[to_cls]] = np.sum(mask)

    class_names = [CLASS_NAMES.get(c, f'Class {c}') for c in classes]

    return {
        'matrix': matrix,
        'classes': classes,
        'class_names': class_names,
    }


def compute_annual_deforestation_stats(annual_change_maps, pixel_area_ha=PIXEL_AREA_HA,
                                        verbose=True):
    """
    Compute annual deforestation statistics from a dictionary of change maps.

    Args:
        annual_change_maps: dict year -> (H, W) binary change map
        pixel_area_ha: Area per pixel in hectares
        verbose: Print statistics

    Returns:
        dict with keys:
            'years': list of years
            'pixels': list of deforestation pixel counts
            'area_ha': list of deforestation area in hectares
            'area_km2': list of deforestation area in km2
            'rate_pct': list of deforestation rate as % of total area
            'cumulative_ha': list of cumulative deforestation
            'trend': dict with linear regression slope and r-squared
    """
    if verbose:
        print("\n" + "=" * 60)
        print("ANNUAL DEFORESTATION STATISTICS")
        print("=" * 60)

    years = sorted(annual_change_maps.keys())
    stats = {
        'years': years,
        'pixels': [],
        'area_ha': [],
        'area_km2': [],
        'rate_pct': [],
        'cumulative_ha': [],
    }

    total_pixels = annual_change_maps[years[0]].size
    total_area_ha = total_pixels * pixel_area_ha
    cumulative = 0

    for year in years:
        change_map = annual_change_maps[year]
        n_pixels = int(np.sum(change_map > 0))
        area_ha = n_pixels * pixel_area_ha
        area_km2 = area_ha / 100
        rate_pct = (n_pixels / total_pixels) * 100
        cumulative += area_ha

        stats['pixels'].append(n_pixels)
        stats['area_ha'].append(area_ha)
        stats['area_km2'].append(area_km2)
        stats['rate_pct'].append(rate_pct)
        stats['cumulative_ha'].append(cumulative)

        if verbose:
            print(f"  {year}: {area_ha:>10,.0f} ha | {area_km2:>8,.1f} km2 | "
                  f"{rate_pct:.4f}% | Cumulative: {cumulative:,.0f} ha")

    # Linear trend
    if len(years) >= 3:
        from scipy import stats as scipy_stats
        x = np.array(years, dtype=float)
        y = np.array(stats['area_ha'], dtype=float)
        slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x, y)
        stats['trend'] = {
            'slope_ha_per_year': slope,
            'intercept': intercept,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'direction': 'increasing' if slope > 0 else 'decreasing',
        }
        if verbose:
            print(f"\n  Trend: {stats['trend']['direction']} "
                  f"({slope:+.1f} ha/year, R2={r_value**2:.3f}, p={p_value:.4f})")
    else:
        stats['trend'] = None

    if verbose:
        print(f"\n  Total deforestation (study period): {cumulative:,.0f} ha "
              f"({cumulative / 100:,.1f} km2)")
        mean_annual = cumulative / len(years)
        print(f"  Mean annual deforestation: {mean_annual:,.0f} ha/year")

    return stats


def compute_forest_area_timeseries(classified_maps, verbose=True):
    """
    Compute forest area over time from annual classified maps.

    Args:
        classified_maps: dict year -> (H, W) classified land cover map
        verbose: Print statistics

    Returns:
        dict with 'years', 'forest_ha', 'forest_pct', 'loss_from_baseline'
    """
    if verbose:
        print("Computing forest area time series...")

    years = sorted(classified_maps.keys())
    stats = {
        'years': years,
        'forest_pixels': [],
        'forest_ha': [],
        'forest_pct': [],
    }

    total_pixels = classified_maps[years[0]].size
    total_area_ha = total_pixels * PIXEL_AREA_HA

    for year in years:
        forest_pixels = int(np.sum(classified_maps[year] == FOREST_CLASS))
        forest_ha = forest_pixels * PIXEL_AREA_HA
        forest_pct = (forest_pixels / total_pixels) * 100

        stats['forest_pixels'].append(forest_pixels)
        stats['forest_ha'].append(forest_ha)
        stats['forest_pct'].append(forest_pct)

        if verbose:
            print(f"  {year}: {forest_ha:>12,.0f} ha ({forest_pct:.2f}%)")

    # Loss from baseline
    baseline_ha = stats['forest_ha'][0]
    stats['loss_from_baseline'] = [baseline_ha - ha for ha in stats['forest_ha']]

    if verbose:
        total_loss = stats['loss_from_baseline'][-1]
        loss_pct = (total_loss / baseline_ha) * 100 if baseline_ha > 0 else 0
        print(f"\n  Net forest loss ({years[0]}-{years[-1]}): "
              f"{total_loss:,.0f} ha ({loss_pct:.2f}%)")

    return stats


def identify_deforestation_hotspots(annual_change_maps, min_recurrence=2, verbose=True):
    """
    Identify areas with recurring deforestation.

    Args:
        annual_change_maps: dict year -> (H, W) binary change map
        min_recurrence: Minimum years of change to flag as hotspot
        verbose: Print statistics

    Returns:
        (H, W) recurrence count array
    """
    years = sorted(annual_change_maps.keys())
    stacked = np.stack([annual_change_maps[y] for y in years], axis=0)
    recurrence = np.sum(stacked > 0, axis=0)

    if verbose:
        for threshold in range(1, len(years) + 1):
            n = np.sum(recurrence >= threshold)
            if n > 0:
                print(f"  Pixels with >= {threshold} years of change: "
                      f"{n:,} ({n * PIXEL_AREA_HA:,.0f} ha)")

    return recurrence
