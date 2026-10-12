#!/usr/bin/env python3
"""Arm B calibrated Eleveld comparison and paired baseline p-values.

Reuses the stored EC50 values and capacity-sweep patient split. Does not
recalibrate EC50 and does not retrain any LSTM. E0/Emax are fitted on Arm B
training patients only for the Arm B comparison; those same Arm B-fit anchors
are applied to the full-cohort holdout's boundary-pinned patients as context.
"""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from compute_population_baseline import (
    BASE_SEED,
    fit_scale_anchors,
    predictions_with_anchors,
)

ROOT = Path(__file__).resolve().parent.parent
CACHE_FILE = ROOT / 'cached_features.pkl'
ARM_B_SWEEP = ROOT / 'capacity_sweep_arm_b_interior_results.csv'
ARM_A_SWEEP = ROOT / 'capacity_sweep_arm_a_full_results.csv'
BASELINE_CSV = ROOT / 'results' / 'processed' / 'baseline_comparison.csv'
FULL_CALIBRATION_CSV = ROOT / 'results' / 'processed' / 'baseline_scale_calibration.csv'
OUT_CSV = ROOT / 'results' / 'processed' / 'arm_b_mechanistic_comparison.csv'


def paired_wilcoxon_abs_error(y, pred_a, pred_b):
    """Two-sided paired Wilcoxon on same-instance absolute errors."""
    err_a = np.abs(np.asarray(y, dtype=float) - np.asarray(pred_a, dtype=float))
    err_b = np.abs(np.asarray(y, dtype=float) - np.asarray(pred_b, dtype=float))
    diff = err_a - err_b
    if np.allclose(diff, 0):
        return 1.0
    return float(stats.wilcoxon(diff, alternative='two-sided').pvalue)


def score_variant(name, y, pred, e0, emax, n_test_patients, comparison):
    errors = np.abs(np.asarray(y, dtype=float) - np.asarray(pred, dtype=float))
    return {
        'cohort': name,
        'n_test_instances': len(y),
        'n_test_patients': int(n_test_patients),
        'e0': float(e0),
        'emax': float(emax),
        'mae': float(mean_absolute_error(y, pred)),
        'sd_test_instance_abs_errors': float(np.std(errors, ddof=1)),
        'r2': float(r2_score(y, pred)),
        'paired_comparison': comparison,
    }


def main():
    with open(CACHE_FILE, 'rb') as f:
        df = pickle.load(f)

    ec50 = df.groupby('stay_id')['EC50_calibrated'].first()
    pinned_ids = set(ec50.index[(ec50 - 0.5).abs() <= 1e-3]) | set(
        ec50.index[(ec50 - 10.0).abs() <= 1e-3]
    )
    interior_ids = set(ec50.index) - pinned_ids
    if len(pinned_ids) != 774:
        raise ValueError(f'Expected 774 boundary-pinned patients; found {len(pinned_ids)}.')

    arm_df = df[df['stay_id'].isin(interior_ids)].reset_index(drop=True)
    arm_train_ids, arm_test_ids = train_test_split(
        arm_df['stay_id'].unique(), test_size=0.2, random_state=BASE_SEED
    )
    arm_train = arm_df[arm_df['stay_id'].isin(arm_train_ids)].reset_index(drop=True)
    arm_test = arm_df[arm_df['stay_id'].isin(arm_test_ids)].reset_index(drop=True)
    if (len(arm_train_ids), len(arm_test_ids)) != (572, 143):
        raise ValueError(
            'Arm B patient split mismatch: expected 572 train/143 test, got '
            f'{len(arm_train_ids)} train/{len(arm_test_ids)} test.'
        )

    anchors = fit_scale_anchors(arm_train)
    y_arm = arm_test['actual_sas'].to_numpy()
    ce_arm = arm_test['Ce'].to_numpy()
    ec50_arm = arm_test['EC50_calibrated'].to_numpy()
    pop_arm = predictions_with_anchors(ce_arm, 3.08, anchors['e0'], anchors['emax'])
    personalized_arm = predictions_with_anchors(
        ce_arm, ec50_arm, anchors['e0'], anchors['emax']
    )
    p_arm = paired_wilcoxon_abs_error(y_arm, pop_arm, personalized_arm)

    rows = [
        score_variant(
            'Arm B interior test: population EC50', y_arm, pop_arm,
            anchors['e0'], anchors['emax'], arm_test['stay_id'].nunique(),
            'population EC50 vs personalized EC50 (paired by test instance)',
        ),
        score_variant(
            'Arm B interior test: personalized EC50', y_arm, personalized_arm,
            anchors['e0'], anchors['emax'], arm_test['stay_id'].nunique(),
            'population EC50 vs personalized EC50 (paired by test instance)',
        ),
    ]
    rows[0]['paired_wilcoxon_p'] = p_arm
    rows[1]['paired_wilcoxon_p'] = p_arm

    # Context: score all prediction instances from the 774 boundary-pinned
    # patients. These patients are excluded from Arm B, so none contributed to
    # the Arm B anchor fit. EC50 remains the value stored in the cache.
    pinned_cohort = df[df['stay_id'].isin(pinned_ids)].reset_index(drop=True)
    if pinned_cohort['stay_id'].nunique() != 774:
        raise ValueError('Expected 774 boundary-pinned patients in context cohort.')
    y_pinned = pinned_cohort['actual_sas'].to_numpy()
    ce_pinned = pinned_cohort['Ce'].to_numpy()
    ec50_pinned = pinned_cohort['EC50_calibrated'].to_numpy()
    pop_pinned = predictions_with_anchors(
        ce_pinned, 3.08, anchors['e0'], anchors['emax']
    )
    personalized_pinned = predictions_with_anchors(
        ce_pinned, ec50_pinned, anchors['e0'], anchors['emax']
    )
    p_pinned = paired_wilcoxon_abs_error(y_pinned, pop_pinned, personalized_pinned)
    rows.extend([
        score_variant(
            'All 774 boundary-pinned patients: population EC50',
            y_pinned, pop_pinned, anchors['e0'], anchors['emax'],
            pinned_cohort['stay_id'].nunique(),
            'population EC50 vs stored personalized EC50 (paired by test instance)',
        ),
        score_variant(
            'All 774 boundary-pinned patients: personalized EC50',
            y_pinned, personalized_pinned, anchors['e0'], anchors['emax'],
            pinned_cohort['stay_id'].nunique(),
            'population EC50 vs stored personalized EC50 (paired by test instance)',
        ),
    ])
    rows[-2]['paired_wilcoxon_p'] = p_pinned
    rows[-1]['paired_wilcoxon_p'] = p_pinned

    result = pd.DataFrame(rows)
    result['e0_ci_lower'] = anchors['e0_ci_lower']
    result['e0_ci_upper'] = anchors['e0_ci_upper']
    result['emax_ci_lower'] = anchors['emax_ci_lower']
    result['emax_ci_upper'] = anchors['emax_ci_upper']
    result['n_anchor_fit_train_instances'] = anchors['n_train_instances']
    result['n_anchor_fit_train_patients'] = anchors['n_train_patients']
    result['interpretation'] = anchors['interpretation']
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_CSV, index=False)

    # The baseline table's p_vs_ours remains reserved for comparisons to the
    # full-hybrid row. Add a separately named pairwise p-value for the two
    # calibrated Eleveld variants. For PURE_ML vs FULL_HYBRID, pair matching
    # seed-level run MAEs from the full-cohort sweep.
    baseline = pd.read_csv(BASELINE_CSV)
    baseline['paired_comparison'] = ''
    baseline['paired_wilcoxon_p'] = np.nan
    calibrated_mask = baseline['method'].isin([
        'Eleveld, population EC50 (training-calibrated E0/Emax)',
        'Eleveld, personalized EC50 (same training-calibrated E0/Emax)',
    ])
    baseline.loc[calibrated_mask, 'paired_comparison'] = (
        'training-calibrated population vs personalized EC50; paired test-instance absolute errors'
    )
    full_anchors = pd.read_csv(FULL_CALIBRATION_CSV).iloc[0]
    _, full_test_ids = train_test_split(
        df['stay_id'].unique(), test_size=0.2, random_state=BASE_SEED
    )
    full_test = df[df['stay_id'].isin(full_test_ids)].reset_index(drop=True)
    y_full = full_test['actual_sas'].to_numpy()
    pred_pop_full = predictions_with_anchors(
        full_test['Ce'], 3.08, full_anchors['e0'], full_anchors['emax']
    )
    pred_personalized_full = predictions_with_anchors(
        full_test['Ce'], full_test['EC50_calibrated'],
        full_anchors['e0'], full_anchors['emax'],
    )
    p_full_mechanistic = paired_wilcoxon_abs_error(
        y_full, pred_pop_full, pred_personalized_full
    )
    baseline.loc[calibrated_mask, 'paired_wilcoxon_p'] = p_full_mechanistic

    full = pd.read_csv(ARM_A_SWEEP)
    pure = full[
        (full['algorithm'] == 'LSTM') &
        (full['configuration'] == 'PURE_ML') &
        (full['lstm_units'] == 4)
    ].set_index('seed')['mae']
    hybrid = full[
        (full['algorithm'] == 'LSTM') &
        (full['configuration'] == 'FULL_HYBRID') &
        (full['lstm_units'] == 4)
    ].set_index('seed')['mae']
    matched = pd.concat([pure.rename('pure'), hybrid.rename('hybrid')], axis=1).dropna()
    if len(matched) != 10:
        raise ValueError(f'Expected 10 paired full-cohort LSTM seeds; found {len(matched)}.')
    p_lstm = float(stats.wilcoxon(
        matched['pure'] - matched['hybrid'], alternative='two-sided'
    ).pvalue)
    pure_row = baseline['method'].eq('LSTM PURE_ML, 4 units')
    baseline.loc[pure_row, 'p_vs_ours'] = p_lstm
    baseline.loc[pure_row, 'paired_comparison'] = (
        'PURE_ML vs FULL_HYBRID; paired 4-unit full-cohort run MAEs by seed (n=10)'
    )
    baseline.loc[pure_row, 'paired_wilcoxon_p'] = p_lstm
    baseline.to_csv(BASELINE_CSV, index=False)

    print('Arm B scale anchors:')
    print(anchors)
    print(f'Arm B paired instance-level Wilcoxon p={p_arm:.12g}')
    print(f'Boundary-pinned patients={pinned_cohort.stay_id.nunique()}, '
          f'instances={len(pinned_cohort)}, paired p={p_pinned:.12g}')
    print(f'PURE_ML vs FULL_HYBRID paired seed-level Wilcoxon p={p_lstm:.12g}')
    print(result.to_string(index=False))
    print(f'Wrote {OUT_CSV}')
    print(f'Updated {BASELINE_CSV}')


if __name__ == '__main__':
    main()
