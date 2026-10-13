#!/usr/bin/env python3
"""
03d_analyze_30seed_capacity_sweep.py
====================================

Summarises the seeded 30-initialization LSTM runs produced by
    02b_run_capacity_sweep.py --n-seeds 30
for Arm A (full cohort), Arm B (715 interior patients) and Arm B clean (the
670 Arm B patients with a fitted EC50).

The script first checks the run files: total row count, exactly 30 seeds
(42 to 71) in every planned cell, no duplicate (arm, configuration, units,
seed) keys, no unplanned cells, and the fixed seeding recorded on every row.
Nothing is summarised or tested if any of these fails.

For every arm x capacity x configuration cell it reports the mean MAE, the SD
across the run-level MAEs, and the mean R². Every other configuration is
compared with PURE_ML in the same arm and capacity: absolute and percent
difference, paired Wilcoxon signed-rank test by seed, and bootstrap 95%
intervals on both differences.

SIGN CONVENTION: difference = PURE_ML - configuration; percent difference =
(PURE_ML - configuration) / configuration x 100. POSITIVE means the
configuration beats PURE_ML. For FULL_HYBRID this is the "Pure-ML degradation"
of results/processed/capacity_summary.md, with the same sign.

HOLM FAMILIES: the FULL_HYBRID versus PURE_ML comparisons are corrected across
the capacities within each arm. The four Arm B clean 4-unit configurations
compared with PURE_ML form a separate family.

PAIRING: runs are paired by shared seed, shared shuffle order and partially
shared initial weights. The LSTM input kernel and the first Dense kernel
differ between configurations because the feature counts differ. The shared
part is checked here from init_shared_sha256 before any paired test is run.

USAGE:
    python src/03d_analyze_30seed_capacity_sweep.py

OUTPUT:
    results/processed/capacity_summary_30seed.csv       one row per cell
    results/processed/capacity_comparisons_30seed.csv   one row per comparison
    results/processed/capacity_summary_30seed.md

AUTHOR: Christopher Morris
"""

import os
import sys
import platform

import numpy as np
import pandas as pd
import scipy
from scipy.stats import wilcoxon

N_SEEDS = 30
BASE_SEED = 42
EXPECTED_SEEDING = 'keras.utils.set_random_seed'
BASELINE = 'PURE_ML'
PRIMARY = 'FULL_HYBRID'
N_BOOTSTRAP = 10000
BOOTSTRAP_SEED = 42

ARMS = {
    'full': ('A (full cohort)', 'capacity_sweep_arm_a_full_30seed_results.csv'),
    'interior': ('B (interior, 715)', 'capacity_sweep_arm_b_interior_30seed_results.csv'),
    'interior_clean': ('B clean (fitted EC50, 670)', 'capacity_sweep_arm_b_clean_30seed_results.csv'),
}
# The planned grid: (arm, units) -> configurations
PLANNED = {
    **{('full', u): ['FULL_HYBRID', 'PURE_ML'] for u in (4, 8, 16, 32)},
    **{('interior', u): ['FULL_HYBRID', 'PURE_ML'] for u in (4, 8, 16, 32)},
    ('interior_clean', 4): ['FULL_HYBRID', 'NO_TEMPORAL', 'NO_STATIC_MECH', 'PURE_ML', 'ONLY_MECH'],
}
N_PLANNED_RUNS = sum(len(configs) for configs in PLANNED.values()) * N_SEEDS
ABLATION_CELL = ('interior_clean', 4)
KEY_COLS = ['arm', 'configuration', 'lstm_units', 'seed']
OUT_CSV = os.path.join('results', 'processed', 'capacity_summary_30seed.csv')
OUT_COMPARISONS = os.path.join('results', 'processed', 'capacity_comparisons_30seed.csv')
OUT_MD = os.path.join('results', 'processed', 'capacity_summary_30seed.md')
EXPECTED_SEEDS = set(range(BASE_SEED, BASE_SEED + N_SEEDS))


def holm(pvalues):
    """Holm step-down adjusted p-values, in the input order."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


def bootstrap_cis(config_mae, baseline_mae, rng):
    """Percentile 95% intervals on the absolute and the percent difference,
    resampling seeds (matched pairs) with replacement."""
    idx = rng.integers(0, len(config_mae), size=(N_BOOTSTRAP, len(config_mae)))
    config_means = config_mae[idx].mean(axis=1)
    baseline_means = baseline_mae[idx].mean(axis=1)
    diff = baseline_means - config_means
    pct = diff / config_means * 100
    return (float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5)),
            float(np.percentile(pct, 2.5)), float(np.percentile(pct, 97.5)))


# ---------------------------------------------------------------------------
# Data checks
# ---------------------------------------------------------------------------
runs = []
problems = []
for arm, (label, path) in ARMS.items():
    if not os.path.exists(path):
        problems.append(f"{path} is missing.")
        continue
    arm_runs = pd.read_csv(path)
    if 'seeding' not in arm_runs.columns:
        problems.append(f"{path} has no seeding column; it predates the seeding fix.")
        continue
    n_bad_seeding = int((arm_runs['seeding'] != EXPECTED_SEEDING).sum())
    if n_bad_seeding:
        problems.append(f"{path} contains {n_bad_seeding} runs not seeded with "
                        f"{EXPECTED_SEEDING}.")
    if set(arm_runs['arm']) != {arm}:
        problems.append(f"{path} contains arms {sorted(set(arm_runs['arm']))}; expected {arm}.")
    runs.append(arm_runs)
if not runs:
    print("ERROR: no 30-seed result files found.")
    sys.exit(1)
runs = pd.concat(runs, ignore_index=True)

n_duplicates = int(runs.duplicated(KEY_COLS).sum())
if n_duplicates:
    problems.append(f"{n_duplicates} duplicate (arm, configuration, units, seed) keys.")
if len(runs) != N_PLANNED_RUNS:
    problems.append(f"{len(runs)} rows in total; the planned grid has {N_PLANNED_RUNS}.")

cell_rows = []
for (arm, units), configs in PLANNED.items():
    for config in configs:
        seeds = runs[(runs['arm'] == arm) & (runs['lstm_units'] == units)
                     & (runs['configuration'] == config)]['seed']
        ok = len(seeds) == N_SEEDS and set(seeds) == EXPECTED_SEEDS
        cell_rows.append({'arm': arm, 'configuration': config, 'lstm_units': units,
                          'n_rows': len(seeds), 'n_distinct_seeds': seeds.nunique(),
                          'seeds_42_to_71': ok})
        if not ok:
            problems.append(f"{ARMS[arm][0]}, {units} units, {config}: {len(seeds)} rows, "
                            f"missing seeds {sorted(EXPECTED_SEEDS - set(seeds))}, "
                            f"unexpected seeds {sorted(set(seeds) - EXPECTED_SEEDS)}.")
cell_table = pd.DataFrame(cell_rows)
planned_cells = {(arm, config, units) for (arm, units), configs in PLANNED.items()
                 for config in configs}
unplanned = set(zip(runs['arm'], runs['configuration'], runs['lstm_units'])) - planned_cells
if unplanned:
    problems.append(f"Rows in cells outside the planned grid: {sorted(unplanned)}.")

print("=" * 70)
print("DATA CHECKS")
print("=" * 70)
print(f"Rows: {len(runs)} (planned {N_PLANNED_RUNS}); duplicate keys: {n_duplicates}; "
      f"rows seeded with {EXPECTED_SEEDING}: "
      f"{int((runs['seeding'] == EXPECTED_SEEDING).sum())}")
print(cell_table.to_string(index=False))
if problems:
    print("\nERROR: the run files failed the data checks; nothing was summarised.")
    for p in problems:
        print(f"   {p}")
    sys.exit(1)
print(f"All {len(cell_table)} cells have exactly {N_SEEDS} runs, seeds "
      f"{BASE_SEED} to {BASE_SEED + N_SEEDS - 1}.\n")

# ---------------------------------------------------------------------------
# Cell summaries and comparisons with PURE_ML
# ---------------------------------------------------------------------------
rng = np.random.default_rng(BOOTSTRAP_SEED)
rows = []
comparisons = []
for (arm, units), configs in PLANNED.items():
    cell = runs[(runs['arm'] == arm) & (runs['lstm_units'] == units)]
    by_config = {c: cell[cell['configuration'] == c].set_index('seed').sort_index()
                 for c in configs}
    base = by_config[BASELINE]
    for config in configs:
        d = by_config[config]
        rows.append({
            'arm': arm,
            'arm_label': ARMS[arm][0],
            'lstm_units': units,
            'configuration': config,
            'n_runs': len(d),
            'n_patients': int(d['n_patients'].iloc[0]),
            'n_train_patients': int(d['n_train_patients'].iloc[0]),
            'n_test_patients': int(d['n_test_patients'].iloc[0]),
            'n_train_instances': int(d['n_train_instances'].iloc[0]),
            'n_test_instances': int(d['n_test_instances'].iloc[0]),
            'mae_mean': float(d['mae'].mean()),
            'mae_sd_across_runs': float(d['mae'].std(ddof=1)),
            'mae_min': float(d['mae'].min()),
            'mae_max': float(d['mae'].max()),
            'r2_mean': float(d['r2'].mean()),
        })
        if config == BASELINE:
            continue
        # Pairing check: same seed must give the same initial values for
        # every weight tensor whose shape is shared by the two configurations.
        shared_same = int((base['init_shared_sha256'] == d['init_shared_sha256']).sum())
        if shared_same != N_SEEDS:
            print(f"ERROR: {ARMS[arm][0]}, {units} units, {config} vs {BASELINE}: shared "
                  f"initial weights identical for only {shared_same} of {N_SEEDS} seeds.")
            sys.exit(1)
        diffs = (base['mae'] - d['mae']).to_numpy()
        abs_lo, abs_hi, pct_lo, pct_hi = bootstrap_cis(
            d['mae'].to_numpy(), base['mae'].to_numpy(), rng)
        comparison = {
            'arm': arm,
            'arm_label': ARMS[arm][0],
            'lstm_units': units,
            'configuration': config,
            'baseline': BASELINE,
            'n_pairs': N_SEEDS,
            'n_seeds_shared_init_identical': shared_same,
            'configuration_mae_mean': float(d['mae'].mean()),
            'pure_ml_mae_mean': float(base['mae'].mean()),
            'abs_diff_pure_minus_config': float(diffs.mean()),
            'abs_diff_boot_ci_low': abs_lo,
            'abs_diff_boot_ci_high': abs_hi,
            'pct_diff_pure_vs_config': float(diffs.mean() / d['mae'].mean() * 100),
            'pct_diff_boot_ci_low': pct_lo,
            'pct_diff_boot_ci_high': pct_hi,
            'paired_diff_sd': float(diffs.std(ddof=1)),
            'paired_dz': float(diffs.mean() / diffs.std(ddof=1)),
            'n_seeds_config_better': int((diffs > 0).sum()),
            'wilcoxon_p': float(wilcoxon(diffs, zero_method='wilcox',
                                         alternative='two-sided').pvalue),
        }
        if config == PRIMARY:
            comparisons.append({'family': f'capacity_{arm}', **comparison})
        if (arm, units) == ABLATION_CELL:
            comparisons.append({'family': f'configurations_{arm}_{units}u', **comparison})

summary = pd.DataFrame(rows)
comparisons = pd.DataFrame(comparisons)
comparisons['holm_p'] = comparisons.groupby('family')['wilcoxon_p'].transform(holm)
comparisons['holm_family_size'] = comparisons.groupby('family')['wilcoxon_p'].transform('size')

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
summary.to_csv(OUT_CSV, index=False)
comparisons.to_csv(OUT_COMPARISONS, index=False)


def fmt_p(value):
    return f"{value:.2e}" if value < 0.001 else f"{value:.3f}"


COMPARISON_HEADER = [
    '| configuration MAE | PURE_ML MAE | absolute difference (95% interval) | '
    'percent difference (95% interval) | dz | seeds where configuration is better | '
    'Wilcoxon p | Holm p |',
    '---:|---:|---|---|---:|---:|---:|---:|',
]


def comparison_cells(r):
    return (f"{r['configuration_mae_mean']:.4f} | {r['pure_ml_mae_mean']:.4f} | "
            f"{r['abs_diff_pure_minus_config']:+.4f} ({r['abs_diff_boot_ci_low']:+.4f} to "
            f"{r['abs_diff_boot_ci_high']:+.4f}) | "
            f"{r['pct_diff_pure_vs_config']:+.2f}% ({r['pct_diff_boot_ci_low']:+.2f}% to "
            f"{r['pct_diff_boot_ci_high']:+.2f}%) | {r['paired_dz']:+.2f} | "
            f"{int(r['n_seeds_config_better'])} of {N_SEEDS} | "
            f"{fmt_p(r['wilcoxon_p'])} | {fmt_p(r['holm_p'])} |")


capacity = comparisons[comparisons['family'].str.startswith('capacity_')]
ablation = comparisons[comparisons['family'].str.startswith('configurations_')]

md = ['# Capacity sweep summary, 30 seeded initializations', '']
md += [
    f"Generated by `src/03d_analyze_30seed_capacity_sweep.py` with Python "
    f"{platform.python_version()}, pandas {pd.__version__}, NumPy {np.__version__}, "
    f"SciPy {scipy.__version__}.",
    '',
    '## Data checks',
    '',
    f"- {len(runs)} runs in total, {n_duplicates} duplicate (arm, configuration, units, seed) keys.",
    f"- All {len(cell_table)} planned cells have exactly {N_SEEDS} runs, seeds {BASE_SEED} to "
    f"{BASE_SEED + N_SEEDS - 1}. No rows fall outside the planned grid.",
    f"- All {len(runs)} rows record `{EXPECTED_SEEDING}`, called before model construction. "
    "The earlier 10-initialization files were seeded with `np.random.seed` and "
    "`tf.random.set_seed` only and are not reproducible by seed; none of those runs is here.",
    '',
    '| arm | configuration | units | runs | distinct seeds |',
    '|---|---|---:|---:|---:|',
]
md += [f"| {ARMS[r['arm']][0]} | {r['configuration']} | {r['lstm_units']} | {r['n_rows']} | "
       f"{r['n_distinct_seeds']} |" for _, r in cell_table.iterrows()]
md += [
    '',
    '## Pairing basis',
    '',
    "Each comparison pairs the two configurations run with the same seed in the same arm "
    "and capacity. The pairs share the seed, the shuffle order of the training instances, "
    "and part of the initial weights: the LSTM recurrent kernel, the Dense(8) kernel, the "
    "output kernel and all biases. **The LSTM input kernel and the first Dense kernel "
    "differ**, because their shapes depend on the number of input features, so the two "
    "members of a pair do not start from an identical model.",
    '',
    f"The shared part was checked from the stored hashes: in all {len(comparisons)} "
    f"comparisons below, all {N_SEEDS} seeds give bit-identical initial values for every "
    "weight tensor whose shape is the same in both configurations.",
    '',
    '## Method',
    '',
    "- SD is the spread across the run-level MAEs (one MAE per initialization).",
    "- Difference = PURE_ML - configuration; percent difference = (PURE_ML - configuration) "
    "/ configuration x 100. **Positive means the configuration beats PURE_ML.** For "
    "FULL_HYBRID this has the same sign as the Pure-ML degradation in `capacity_summary.md`.",
    f"- Wilcoxon: two-sided signed-rank test on the {N_SEEDS} within-seed MAE differences.",
    "- Holm: the FULL_HYBRID versus PURE_ML comparisons are corrected across the capacities "
    "within each arm (four in Arm A, four in Arm B, one in Arm B clean, where the corrected "
    "value equals the raw one). The four Arm B clean 4-unit configurations compared with "
    "PURE_ML are a separate family of four.",
    f"- Bootstrap intervals: percentile 95%, {N_BOOTSTRAP} resamples of the {N_SEEDS} seeds "
    "(pairs kept together). They reflect variation across initializations on a fixed "
    "train/test split, not variation across patients.",
    "- dz is the mean within-seed difference divided by the SD of the within-seed differences.",
    '',
    '## 4-unit FULL_HYBRID versus PURE_ML',
    '',
    '| arm ' + COMPARISON_HEADER[0],
    '|---|' + COMPARISON_HEADER[1],
]
md += [f"| {r['arm_label']} | {comparison_cells(r)}"
       for _, r in capacity[capacity['lstm_units'] == 4].iterrows()]
md += ['', "Holm p here is corrected across the capacities of the same arm.", '']

for arm, (label, _) in ARMS.items():
    arm_rows = summary[summary['arm'] == arm]
    f0 = arm_rows.iloc[0]
    md += [f"## Arm {label}", '',
           f"Train {f0['n_train_patients']} patients ({f0['n_train_instances']} instances), "
           f"test {f0['n_test_patients']} patients ({f0['n_test_instances']} instances).", '',
           '| units | configuration | runs | MAE mean | MAE SD | MAE min | MAE max | mean R² |',
           '|---:|---|---:|---:|---:|---:|---:|---:|']
    md += [f"| {r['lstm_units']} | {r['configuration']} | {r['n_runs']} | {r['mae_mean']:.4f} | "
           f"{r['mae_sd_across_runs']:.4f} | {r['mae_min']:.4f} | {r['mae_max']:.4f} | "
           f"{r['r2_mean']:.4f} |" for _, r in arm_rows.iterrows()]
    arm_capacity = capacity[capacity['arm'] == arm]
    md += ['', f"### FULL_HYBRID versus PURE_ML by capacity, arm {label}", '',
           f"Holm family: {len(arm_capacity)} comparison(s).", '',
           '| units ' + COMPARISON_HEADER[0],
           '|---:|' + COMPARISON_HEADER[1]]
    md += [f"| {r['lstm_units']} | {comparison_cells(r)}" for _, r in arm_capacity.iterrows()]
    md.append('')

md += [f"## Arm {ARMS[ABLATION_CELL[0]][0]}, {ABLATION_CELL[1]} units: every configuration "
       "versus PURE_ML", '',
       f"Holm family: {len(ablation)} comparisons. PURE_ML itself is the baseline "
       f"(MAE {ablation['pure_ml_mae_mean'].iloc[0]:.4f}).", '',
       '| configuration ' + COMPARISON_HEADER[0],
       '|---|' + COMPARISON_HEADER[1]]
md += [f"| {r['configuration']} | {comparison_cells(r)}" for _, r in ablation.iterrows()]
md.append('')

with open(OUT_MD, 'w', encoding='utf-8') as f:
    f.write('\n'.join(md))

print('\n'.join(md))
print(f"Wrote {OUT_CSV}")
print(f"Wrote {OUT_COMPARISONS}")
print(f"Wrote {OUT_MD}")
