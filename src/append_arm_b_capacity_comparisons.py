#!/usr/bin/env python3
"""Append Arm B 4-unit configuration summaries and paired Holm tests.

Uses already-recorded runs in capacity_sweep_arm_b_interior_results.csv.
Existing mechanistic comparison rows are preserved; any prior appended rows
for these exact configurations are replaced so this script is idempotent.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
SWEEP_FILE = ROOT / 'capacity_sweep_arm_b_interior_results.csv'
TABLE_FILE = ROOT / 'results' / 'processed' / 'arm_b_mechanistic_comparison.csv'
CONFIGS = ['NO_STATIC_MECH', 'ONLY_MECH', 'NO_TEMPORAL']
REFERENCE_CONFIGS = ['FULL_HYBRID', 'PURE_ML']
EXPECTED_SEEDS = list(range(42, 52))
EXPECTED_PATIENT_COUNTS = (715, 572, 143)


def holm_adjust(p_values):
    """Return Holm-adjusted p-values in input order."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted_sorted = np.maximum.accumulate(
        (len(p) - np.arange(len(p))) * p[order]
    )
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


def paired_p(config_rows, reference_rows):
    left = config_rows.set_index('seed')['mae'].sort_index()
    right = reference_rows.set_index('seed')['mae'].sort_index()
    paired = pd.concat([left.rename('config'), right.rename('reference')], axis=1).dropna()
    if paired.index.tolist() != EXPECTED_SEEDS:
        raise ValueError(f'Expected matching seeds {EXPECTED_SEEDS}; got {paired.index.tolist()}')
    differences = paired['config'].to_numpy() - paired['reference'].to_numpy()
    if np.allclose(differences, 0):
        return 1.0
    return float(stats.wilcoxon(differences, alternative='two-sided').pvalue)


def main():
    sweep = pd.read_csv(SWEEP_FILE)
    if not {'arm', 'algorithm', 'configuration', 'seed', 'lstm_units',
            'n_patients', 'n_train_patients', 'n_test_patients', 'mae', 'r2'} <= set(sweep.columns):
        raise ValueError('Sweep output is missing required columns.')

    frame = sweep[
        (sweep['arm'] == 'interior') &
        (sweep['algorithm'] == 'LSTM') &
        (sweep['lstm_units'] == 4) &
        (sweep['configuration'].isin(CONFIGS + REFERENCE_CONFIGS))
    ].copy()

    for config in CONFIGS + REFERENCE_CONFIGS:
        runs = frame[frame['configuration'] == config].sort_values('seed')
        if runs['seed'].astype(int).tolist() != EXPECTED_SEEDS:
            raise ValueError(f'{config}: expected ten seeds {EXPECTED_SEEDS}, found {runs.seed.tolist()}')
        counts = tuple(int(runs[column].iloc[0]) for column in
                       ['n_patients', 'n_train_patients', 'n_test_patients'])
        if counts != EXPECTED_PATIENT_COUNTS:
            raise ValueError(f'{config}: Arm B patient split mismatch: {counts}')

    tests = []
    for config in CONFIGS:
        config_rows = frame[frame['configuration'] == config]
        for reference in REFERENCE_CONFIGS:
            reference_rows = frame[frame['configuration'] == reference]
            tests.append({
                'configuration': config,
                'reference': reference,
                'p_raw': paired_p(config_rows, reference_rows),
            })
    adjusted = holm_adjust([test['p_raw'] for test in tests])
    for test, p_adjusted in zip(tests, adjusted):
        test['p_holm'] = float(p_adjusted)
    test_lookup = {(t['configuration'], t['reference']): t for t in tests}

    # Retain the existing table rows and append one summary row per requested
    # configuration. Re-running only replaces these three appended labels.
    table = pd.read_csv(TABLE_FILE)
    arm_b_reference = table.loc[
        table['cohort'].eq('Arm B interior test: population EC50'),
        'n_test_instances',
    ]
    if arm_b_reference.empty:
        raise ValueError('Could not find the existing Arm B test row for instance count.')
    n_test_instances = int(arm_b_reference.iloc[0])
    append_labels = [f'Arm B 4-unit capacity sweep: {config}' for config in CONFIGS]
    table = table[~table['cohort'].isin(append_labels)].copy()
    added_rows = []
    for config in CONFIGS:
        runs = frame[frame['configuration'] == config].sort_values('seed')
        row = {column: '' for column in table.columns}
        row.update({
            'cohort': f'Arm B 4-unit capacity sweep: {config}',
            'n_test_instances': n_test_instances,
            'n_test_patients': int(runs['n_test_patients'].iloc[0]),
            'mae': float(runs['mae'].mean()),
            'sd_run_mae_means': float(runs['mae'].std(ddof=1)),
            'r2': float(runs['r2'].mean()),
            'paired_comparison': (
                'paired Wilcoxon on run MAE by matching seed; Holm corrected '
                'across 3 configs x 2 references'
            ),
            'p_vs_full_hybrid': test_lookup[(config, 'FULL_HYBRID')]['p_raw'],
            'holm_p_vs_full_hybrid': test_lookup[(config, 'FULL_HYBRID')]['p_holm'],
            'p_vs_pure_ml': test_lookup[(config, 'PURE_ML')]['p_raw'],
            'holm_p_vs_pure_ml': test_lookup[(config, 'PURE_ML')]['p_holm'],
            'n_runs': len(runs),
        })
        added_rows.append(row)

    table = pd.concat([table, pd.DataFrame(added_rows)], ignore_index=True)
    TABLE_FILE.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLE_FILE, index=False)

    print('New Arm B comparison summaries:')
    print(pd.DataFrame(added_rows)[[
        'cohort', 'n_runs', 'mae', 'sd_run_mae_means', 'r2',
        'p_vs_full_hybrid', 'holm_p_vs_full_hybrid',
        'p_vs_pure_ml', 'holm_p_vs_pure_ml',
    ]].to_string(index=False))
    print('Holm family: all six new paired tests (3 configurations x 2 references).')
    print(f'Updated {TABLE_FILE}; existing non-appended rows were retained.')


if __name__ == '__main__':
    main()
