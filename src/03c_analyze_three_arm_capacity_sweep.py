#!/usr/bin/env python3
"""Summarize the three-arm LSTM capacity sweep."""

import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import spearmanr

ARM_FILES = {
    'A_full': 'capacity_sweep_arm_a_full_results.csv',
    'B_interior': 'capacity_sweep_arm_b_interior_results.csv',
    'C_matched': 'capacity_sweep_arm_c_matched_results.csv',
}
CONFIGURATIONS = [
    'FULL_HYBRID', 'NO_TEMPORAL', 'NO_STATIC_MECH', 'PURE_ML', 'ONLY_MECH'
]


def exponential_decay(units, amplitude, rate, asymptote):
    return amplitude * np.exp(-rate * units) + asymptote


def main():
    all_summaries = []
    pure_ml_rows = []
    trend_rows = []

    for arm, results_file in ARM_FILES.items():
        if not os.path.exists(results_file):
            print(f"ERROR: missing {results_file}; complete the runner first.")
            return 1

        results = pd.read_csv(results_file)
        expected_per_cell = 10
        coverage = results.groupby(['lstm_units', 'configuration']).size()
        incomplete = coverage[coverage != expected_per_cell]
        if not incomplete.empty:
            print(f"ERROR: {arm} has incomplete cells; expected 10 seeds each:")
            print(incomplete.to_string())
            return 1

        summary_rows = []
        for units in sorted(results['lstm_units'].unique()):
            size_results = results[results['lstm_units'] == units]
            baseline = size_results.loc[
                size_results['configuration'] == 'FULL_HYBRID', 'mae'
            ]
            baseline_mean = baseline.mean()
            for config in CONFIGURATIONS:
                values = size_results.loc[
                    size_results['configuration'] == config, 'mae'
                ]
                mean_mae = values.mean()
                std_mae = values.std(ddof=1)
                degradation = 0.0 if config == 'FULL_HYBRID' else (
                    (mean_mae - baseline_mean) / baseline_mean * 100.0
                )
                degradation_sd = 0.0 if config == 'FULL_HYBRID' else (
                    std_mae / baseline_mean * 100.0
                )
                summary_rows.append({
                    'arm': arm,
                    'lstm_units': int(units),
                    'configuration': config,
                    'n_runs': int(len(values)),
                    'mae_mean': mean_mae,
                    'mae_std': std_mae,
                    'r2_mean': size_results.loc[
                        size_results['configuration'] == config, 'r2'
                    ].mean(),
                    'degradation_pct': degradation,
                    'degradation_std_pct': degradation_sd,
                })

        summary = pd.DataFrame(summary_rows)
        summary_file = f"capacity_sweep_{arm.lower()}_summary.csv"
        summary.to_csv(summary_file, index=False)
        all_summaries.append(summary)

        pure = (summary[summary['configuration'] == 'PURE_ML']
                .sort_values('lstm_units'))
        for _, row in pure.iterrows():
            pure_ml_rows.append({
                'arm': arm,
                'lstm_units': int(row['lstm_units']),
                'pure_ml_mae_mean': row['mae_mean'],
                'full_hybrid_mae_mean': summary.loc[
                    (summary['lstm_units'] == row['lstm_units'])
                    & (summary['configuration'] == 'FULL_HYBRID'),
                    'mae_mean'
                ].iloc[0],
                'pure_ml_degradation_pct': row['degradation_pct'],
                'pure_ml_degradation_sd_pct': row['degradation_std_pct'],
            })

        rho, p_value = spearmanr(pure['lstm_units'], pure['degradation_pct'])
        units = pure['lstm_units'].to_numpy(dtype=float)
        degradation = pure['degradation_pct'].to_numpy(dtype=float)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                params, _ = curve_fit(
                    exponential_decay,
                    units,
                    degradation,
                    p0=(degradation[0] - degradation[-1], 0.1, degradation[-1]),
                    maxfev=20000,
                )
            fitted = exponential_decay(units, *params)
            residual_sum = float(np.sum((degradation - fitted) ** 2))
            total_sum = float(np.sum((degradation - degradation.mean()) ** 2))
            fit_r2 = 1.0 - residual_sum / total_sum if total_sum else np.nan
            fit_status = 'fit'
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            params = [np.nan, np.nan, np.nan]
            fit_r2 = np.nan
            fit_status = f'failed: {type(exc).__name__}'

        trend_rows.append({
            'arm': arm,
            'spearman_rho_units_vs_pure_ml_degradation': rho,
            'spearman_p_value': p_value,
            'fit_model': 'a * exp(-b * units) + c',
            'fit_amplitude_a': params[0],
            'fit_rate_b': params[1],
            'fit_asymptote_c': params[2],
            'fit_r_squared': fit_r2,
            'fit_status': fit_status,
        })

        print(f"\n{arm}: Pure ML degradation from Full Hybrid (% MAE change)")
        print(pure[['lstm_units', 'degradation_pct', 'degradation_std_pct']]
              .to_string(index=False, float_format=lambda value: f'{value:.3f}'))
        print(f"Spearman rho={rho:.4f}, p={p_value:.4g}; "
              f"exponential fit {fit_status}, params={params}, R²={fit_r2:.4f}")
        print(f"Saved {summary_file}")

    comparison = pd.DataFrame(pure_ml_rows)
    pivot = comparison.pivot(
        index='lstm_units', columns='arm', values='pure_ml_degradation_pct'
    )
    pivot['B_minus_C_percentage_points'] = (
        pivot['B_interior'] - pivot['C_matched']
    )
    pivot.reset_index().to_csv('three_arm_pure_ml_comparison.csv', index=False)

    trends = pd.DataFrame(trend_rows)
    trends.to_csv('three_arm_capacity_trend_stats.csv', index=False)
    pd.concat(all_summaries, ignore_index=True).to_csv(
        'three_arm_capacity_summary.csv', index=False
    )

    print("\nPure ML degradation comparison (% MAE change vs Full Hybrid):")
    print(pivot.to_string(float_format=lambda value: f'{value:.3f}'))
    print("\nPositive B-minus-C values indicate greater full-hybrid benefit in the "
          "interior arm than in the size-matched random sample.")
    print("Wrote three_arm_pure_ml_comparison.csv, "
          "three_arm_capacity_trend_stats.csv, and three_arm_capacity_summary.csv")
    return 0


if __name__ == '__main__':
    sys.exit(main())