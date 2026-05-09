#!/usr/bin/env python3
"""
03b_analyze_capacity_sweep.py
=============================

This script analyzes the LSTM capacity sweep results produced by
02b_run_capacity_sweep.py. It computes degradation percentages relative to
the FULL_HYBRID baseline at each LSTM size, so that a "size 4 NO_STATIC_MECH"
result is compared against a "size 4 FULL_HYBRID" baseline rather than against
some absolute reference, since the absolute MAE values shift with capacity.

The central question this analysis answers is whether the degradation when
mechanistic features are removed shrinks as LSTM capacity grows. If smaller
LSTMs show clear positive degradation in NO_STATIC_MECH and PURE_ML while
larger ones show flat or negative degradation, that is consistent with the
hypothesis that small models lean on mechanistic features for inductive bias,
while larger models can extract similar information from the raw inputs and
even overfit when both kinds of signal are provided.

USAGE:
    python 03b_analyze_capacity_sweep.py

REQUIREMENTS:
    capacity_sweep_results.csv in the current directory

OUTPUT:
    capacity_sweep_summary.csv - long-form summary, one row per
    (lstm_units, configuration), with mean and std for MAE and degradation.

    Console output: pretty-printed pivot tables and an explicit hypothesis
    check on whether NO_STATIC_MECH and PURE_ML degradation decrease with size.

AUTHOR: Christopher Morris
"""

import pandas as pd
import numpy as np
import os
import sys

RESULTS_FILE = 'capacity_sweep_results.csv'
SUMMARY_FILE = 'capacity_sweep_summary.csv'

print("=" * 70)
print("CAPACITY SWEEP ANALYSIS")
print("=" * 70)

if not os.path.exists(RESULTS_FILE):
    print(f"ERROR: Results file not found: {RESULTS_FILE}")
    print("Please run 'python 02b_run_capacity_sweep.py' first.")
    sys.exit(1)

results_df = pd.read_csv(RESULTS_FILE)
print(f"\nLoaded {len(results_df)} run results from {RESULTS_FILE}")

# Show coverage
print("\nResults coverage (rows = lstm_units, columns = configurations):")
coverage = (results_df.groupby(['lstm_units', 'configuration'])
            .size().unstack(fill_value=0))
print(coverage)

# Compute summary statistics per (lstm_units, configuration)
configurations = ['FULL_HYBRID', 'NO_TEMPORAL', 'NO_STATIC_MECH', 'PURE_ML', 'ONLY_MECH']
sizes = sorted(results_df['lstm_units'].unique().tolist())

summary_rows = []

for size in sizes:
    size_df = results_df[results_df['lstm_units'] == size]
    baseline_sub = size_df[size_df['configuration'] == 'FULL_HYBRID']
    if len(baseline_sub) == 0:
        print(f"\nWARNING: no FULL_HYBRID rows at lstm_units={size}, skipping size")
        continue
    baseline_mean = baseline_sub['mae'].mean()

    for config in configurations:
        sub = size_df[size_df['configuration'] == config]
        if len(sub) == 0:
            continue
        mean_mae = sub['mae'].mean()
        std_mae = sub['mae'].std() if len(sub) > 1 else 0.0
        mean_r2 = sub['r2'].mean()

        if config == 'FULL_HYBRID':
            deg_mean = 0.0
            deg_std = 0.0
        else:
            deg_mean = ((mean_mae - baseline_mean) / baseline_mean) * 100
            deg_std = (std_mae / baseline_mean) * 100

        summary_rows.append({
            'lstm_units': int(size),
            'configuration': config,
            'n_runs': int(len(sub)),
            'mae_mean': round(mean_mae, 4),
            'mae_std': round(std_mae, 4),
            'r2_mean': round(mean_r2, 4),
            'degradation_pct': round(deg_mean, 2),
            'degradation_std_pct': round(deg_std, 2),
        })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(SUMMARY_FILE, index=False)
print(f"\nSaved summary to {SUMMARY_FILE}")

# Pretty-print the degradation table (sizes as rows, configs as columns)
print("\n" + "=" * 70)
print("DEGRADATION FROM FULL_HYBRID BASELINE (% MAE change)")
print("=" * 70)

deg_pivot = summary_df.pivot(index='lstm_units', columns='configuration',
                              values='degradation_pct').reindex(columns=configurations)
std_pivot = summary_df.pivot(index='lstm_units', columns='configuration',
                              values='degradation_std_pct').reindex(columns=configurations)

# Build a string-formatted version of the table that combines mean ± std
formatted_rows = []
for size in deg_pivot.index:
    row_strs = {'lstm_units': int(size)}
    for config in configurations:
        mean = deg_pivot.loc[size, config]
        std = std_pivot.loc[size, config]
        if config == 'FULL_HYBRID':
            row_strs[config] = 'baseline'
        else:
            row_strs[config] = f"{mean:+.2f}% ± {std:.2f}%"
    formatted_rows.append(row_strs)
formatted_df = pd.DataFrame(formatted_rows).set_index('lstm_units')
print()
print(formatted_df.to_string())

# Pretty-print absolute MAE per cell as a separate view
print("\n" + "=" * 70)
print("ABSOLUTE MAE (mean ± std)")
print("=" * 70)

mae_pivot = summary_df.pivot(index='lstm_units', columns='configuration',
                              values='mae_mean').reindex(columns=configurations)
mae_std_pivot = summary_df.pivot(index='lstm_units', columns='configuration',
                                  values='mae_std').reindex(columns=configurations)

formatted_mae_rows = []
for size in mae_pivot.index:
    row = {'lstm_units': int(size)}
    for config in configurations:
        m = mae_pivot.loc[size, config]
        s = mae_std_pivot.loc[size, config]
        row[config] = f"{m:.4f} ± {s:.4f}"
    formatted_mae_rows.append(row)
formatted_mae_df = pd.DataFrame(formatted_mae_rows).set_index('lstm_units')
print()
print(formatted_mae_df.to_string())

# Hypothesis test: does removing mechanistic features hurt smaller LSTMs more
# than larger ones? We look at NO_STATIC_MECH and PURE_ML across sizes.
print("\n" + "=" * 70)
print("CAPACITY-MATCHING HYPOTHESIS CHECK")
print("=" * 70)
print()
print("Hypothesis: smaller LSTMs benefit more from mechanistic features, so")
print("removing them (NO_STATIC_MECH, PURE_ML) should hurt small models more")
print("than large ones. Look for a monotonically decreasing degradation as")
print("LSTM size grows.")
print()

for target in ['NO_STATIC_MECH', 'PURE_ML']:
    series = summary_df[summary_df['configuration'] == target].sort_values('lstm_units')
    print(f"{target}:")
    for _, row in series.iterrows():
        print(f"   units={int(row['lstm_units']):>3d}: degradation = "
              f"{row['degradation_pct']:+6.2f}% ± {row['degradation_std_pct']:5.2f}%, "
              f"MAE = {row['mae_mean']:.4f}")
    # Quick monotonic check: pairwise differences
    sorted_deg = series['degradation_pct'].tolist()
    pairwise_dec = all(sorted_deg[i] >= sorted_deg[i+1] for i in range(len(sorted_deg)-1))
    pairwise_inc = all(sorted_deg[i] <= sorted_deg[i+1] for i in range(len(sorted_deg)-1))
    if pairwise_dec:
        trend = "monotonically decreasing (consistent with hypothesis)"
    elif pairwise_inc:
        trend = "monotonically increasing (opposite of hypothesis)"
    else:
        trend = "non-monotonic (mixed evidence)"
    print(f"   Trend across sizes: {trend}")
    print()

print("=" * 70)
print("Files generated:")
print(f"   {SUMMARY_FILE}")
print()
print("Next step: run 'python 04b_plot_capacity_sweep.py' to generate figures")
