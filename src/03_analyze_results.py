#!/usr/bin/env python3
"""
analyze_results.py
==================

This script analyzes the ablation results produced by run_ablation.py and
generates summary statistics suitable for inclusion in the paper. The
analysis computes degradation percentages relative to the FULL_HYBRID
baseline for each algorithm, with proper mean and standard deviation
reporting where multiple runs are available.

The key output is a degradation table that quantifies how much each
algorithm relies on different feature groups. By comparing the
degradation patterns between XGBoost and LSTM, we can identify whether
the algorithms benefit from mechanistic features in fundamentally
different ways. This is the core finding our paper is built around.

USAGE:
    python analyze_results.py

REQUIREMENTS:
    ablation_results.csv in the current directory

OUTPUT:
    summary_statistics.csv - mean and standard deviation per configuration
    degradation_analysis.csv - degradation percentages with statistics
    Console output with the same information formatted for reading

AUTHOR: Christopher Morris
"""

import pandas as pd
import numpy as np
import os
import sys

RESULTS_FILE = 'ablation_results.csv'
SUMMARY_FILE = 'summary_statistics.csv'
DEGRADATION_FILE = 'degradation_analysis.csv'

print("=" * 70)
print("RESULTS ANALYSIS")
print("=" * 70)

# Verify input file exists
if not os.path.exists(RESULTS_FILE):
    print(f"ERROR: Results file not found: {RESULTS_FILE}")
    print("Please run 'python run_ablation.py' first.")
    sys.exit(1)

# Load results
results_df = pd.read_csv(RESULTS_FILE)
print(f"\nLoaded {len(results_df)} run results from {RESULTS_FILE}")

# Show what we have collected
print("\nResults coverage:")
coverage = results_df.groupby(['algorithm', 'configuration']).size().unstack(fill_value=0)
print(coverage)

# Compute summary statistics for each (algorithm, configuration) cell
print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)

summary_rows = []
configurations = ['FULL_HYBRID', 'NO_TEMPORAL', 'NO_STATIC_MECH', 'PURE_ML', 'ONLY_MECH']

for algo in ['XGBoost', 'LSTM']:
    print(f"\n{algo}:")
    for config in configurations:
        sub = results_df[(results_df['algorithm'] == algo) & 
                         (results_df['configuration'] == config)]
        if len(sub) == 0:
            print(f"   {config}: no data")
            continue
        
        mean_mae = sub['mae'].mean()
        std_mae = sub['mae'].std() if len(sub) > 1 else 0.0
        mean_rmse = sub['rmse'].mean()
        mean_r2 = sub['r2'].mean()
        
        summary_rows.append({
            'algorithm': algo,
            'configuration': config,
            'n_runs': len(sub),
            'mae_mean': mean_mae,
            'mae_std': std_mae,
            'rmse_mean': mean_rmse,
            'r2_mean': mean_r2,
        })
        
        print(f"   {config:18s}: MAE = {mean_mae:.4f} ± {std_mae:.4f} (n={len(sub)}), "
              f"R² = {mean_r2:.4f}")

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(SUMMARY_FILE, index=False)
print(f"\nSaved summary statistics to {SUMMARY_FILE}")

# Compute degradation analysis with proper statistics
print("\n" + "=" * 70)
print("DEGRADATION FROM FULL_HYBRID BASELINE")
print("=" * 70)

degradation_rows = []

for algo in ['XGBoost', 'LSTM']:
    algo_summary = summary_df[summary_df['algorithm'] == algo]
    if len(algo_summary) == 0:
        continue
    
    baseline_row = algo_summary[algo_summary['configuration'] == 'FULL_HYBRID']
    if len(baseline_row) == 0:
        print(f"\n{algo}: no FULL_HYBRID baseline available")
        continue
    
    baseline_mean = baseline_row['mae_mean'].values[0]
    
    print(f"\n{algo} (baseline MAE = {baseline_mean:.4f}):")
    
    for _, row in algo_summary.iterrows():
        if row['configuration'] == 'FULL_HYBRID':
            deg_mean = 0.0
            deg_std = 0.0
        else:
            deg_mean = ((row['mae_mean'] - baseline_mean) / baseline_mean) * 100
            # Approximate degradation std using error propagation. This treats
            # the baseline as a fixed reference point and propagates the
            # variance of the configuration through.
            deg_std = (row['mae_std'] / baseline_mean) * 100
        
        degradation_rows.append({
            'algorithm': algo,
            'configuration': row['configuration'],
            'mae_mean': round(row['mae_mean'], 4),
            'mae_std': round(row['mae_std'], 4),
            'degradation_pct': round(deg_mean, 2),
            'degradation_std_pct': round(deg_std, 2),
            'n_runs': int(row['n_runs']),
        })
        
        if row['configuration'] == 'FULL_HYBRID':
            print(f"   {row['configuration']:18s}: baseline (0.00%)")
        else:
            print(f"   {row['configuration']:18s}: {deg_mean:+6.2f}% ± {deg_std:5.2f}%")

degradation_df = pd.DataFrame(degradation_rows)
degradation_df.to_csv(DEGRADATION_FILE, index=False)
print(f"\nSaved degradation analysis to {DEGRADATION_FILE}")

# Compute the discovery comparison
print("\n" + "=" * 70)
print("THE COMPUTATIONAL DISCOVERY COMPARISON")
print("=" * 70)
print()
print("Our hypothesis was that LSTM would benefit more from temporal")
print("mechanistic features than XGBoost does. The relevant comparison is")
print("between the NO_TEMPORAL and NO_STATIC_MECH degradation values.")
print()

for algo in ['XGBoost', 'LSTM']:
    algo_deg = degradation_df[degradation_df['algorithm'] == algo]
    no_temp_row = algo_deg[algo_deg['configuration'] == 'NO_TEMPORAL']
    no_static_row = algo_deg[algo_deg['configuration'] == 'NO_STATIC_MECH']
    
    if len(no_temp_row) == 0 or len(no_static_row) == 0:
        print(f"{algo}: insufficient data for discovery comparison")
        continue
    
    no_temp = no_temp_row['degradation_pct'].values[0]
    no_temp_std = no_temp_row['degradation_std_pct'].values[0]
    no_static = no_static_row['degradation_pct'].values[0]
    no_static_std = no_static_row['degradation_std_pct'].values[0]
    
    print(f"{algo}:")
    print(f"   Removing temporal features: {no_temp:+6.2f}% ± {no_temp_std:5.2f}%")
    print(f"   Removing static features:   {no_static:+6.2f}% ± {no_static_std:5.2f}%")
    
    diff = abs(no_temp - no_static)
    if no_temp > no_static:
        print(f"   Algorithm relies MORE on temporal features (diff: {diff:.2f} pp)")
    elif no_static > no_temp:
        print(f"   Algorithm relies MORE on static features (diff: {diff:.2f} pp)")
    else:
        print(f"   Algorithm relies equally on both")
    print()

# Final summary table
print("=" * 70)
print("FINAL SUMMARY TABLE (suitable for paper)")
print("=" * 70)
print()
print(degradation_df.to_string(index=False))
print()
print("Files generated:")
print(f"   {SUMMARY_FILE}")
print(f"   {DEGRADATION_FILE}")
print()
print("Next step: run 'python make_figures.py' to generate visualization")
