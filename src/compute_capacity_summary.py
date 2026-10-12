#!/usr/bin/env python3
"""Compute a readable capacity-sweep summary for the three study arms.

This script reads the three arm-level sweep result files and writes:
- results/processed/capacity_summary_table.csv
- results/processed/capacity_summary.md

It reports the full MAE summary for each arm × size × configuration, flags cells
with fewer than 10 runs, and computes matched FULL_HYBRID vs PURE_ML comparisons
using paired Wilcoxon tests and Holm correction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
ARM_FILES = {
    'full': ROOT / 'capacity_sweep_arm_a_full_results.csv',
    'interior': ROOT / 'capacity_sweep_arm_b_interior_results.csv',
    'matched': ROOT / 'capacity_sweep_arm_c_matched_results.csv',
}
CONFIG_ORDER = ['FULL_HYBRID', 'NO_TEMPORAL', 'NO_STATIC_MECH', 'PURE_ML', 'ONLY_MECH']
ARM_LABELS = {
    'full': 'A (full cohort)',
    'interior': 'B (interior cohort)',
    'matched': 'C (matched cohort)',
}


def holm_adjust(pvals: list[float] | np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    if p.size == 0:
        return np.array([], dtype=float)
    order = np.argsort(p)
    sorted_p = p[order]
    adjusted = np.empty_like(sorted_p)
    running = 0.0
    m = len(sorted_p)
    for idx, val in enumerate(sorted_p, start=1):
        running = max(running, (m - idx + 1) * val)
        adjusted[idx - 1] = min(1.0, running)
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result


def format_number(value: float | int | str | None, digits: int = 4):
    if value is None or pd.isna(value):
        return 'LOW_N'
    return f'{float(value):.{digits}f}'


def compute_summary_rows() -> pd.DataFrame:
    summary_rows = []
    for arm_name, path in ARM_FILES.items():
        df = pd.read_csv(path)
        df = df.drop_duplicates(
            subset=['arm', 'algorithm', 'configuration', 'seed', 'lstm_units'],
            keep='last',
        )
        for units in sorted(df['lstm_units'].unique()):
            size_df = df[df['lstm_units'] == units].copy()
            for config in CONFIG_ORDER:
                cell = size_df[size_df['configuration'] == config].copy()
                n_runs = len(cell)
                if n_runs < 10:
                    summary_rows.append({
                        'arm': arm_name,
                        'lstm_units': int(units),
                        'configuration': config,
                        'n_runs': int(n_runs),
                        'mae_mean': np.nan,
                        'mae_sd': np.nan,
                        'mae_min': np.nan,
                        'mae_max': np.nan,
                        'r2_mean': np.nan,
                        'low_n_flag': 'LOW_N',
                    })
                    continue

                mae = cell['mae'].astype(float)
                r2 = cell['r2'].astype(float)
                summary_rows.append({
                    'arm': arm_name,
                    'lstm_units': int(units),
                    'configuration': config,
                    'n_runs': int(n_runs),
                    'mae_mean': float(mae.mean()),
                    'mae_sd': float(mae.std(ddof=1)),
                    'mae_min': float(mae.min()),
                    'mae_max': float(mae.max()),
                    'r2_mean': float(r2.mean()),
                    'low_n_flag': '',
                })
    return pd.DataFrame(summary_rows).sort_values(['arm', 'lstm_units', 'configuration']).reset_index(drop=True)


def compute_comparison_rows() -> pd.DataFrame:
    comparison_rows = []
    pvals = []
    pair_metadata = []
    for arm_name, path in ARM_FILES.items():
        df = pd.read_csv(path)
        df = df.drop_duplicates(
            subset=['arm', 'algorithm', 'configuration', 'seed', 'lstm_units'],
            keep='last',
        )
        for units in sorted(df['lstm_units'].unique()):
            size_df = df[df['lstm_units'] == units].copy()
            full = size_df[size_df['configuration'] == 'FULL_HYBRID'].copy()
            pure = size_df[size_df['configuration'] == 'PURE_ML'].copy()
            if full.empty or pure.empty:
                continue
            paired = full.merge(pure, on='seed', how='inner', suffixes=('_full', '_pure'))
            n_pair = len(paired)

            if n_pair < 10:
                comparison_rows.append({
                    'arm': arm_name,
                    'lstm_units': int(units),
                    'n_paired_runs': int(n_pair),
                    'full_hybrid_mae_mean': float(full['mae'].mean()) if len(full) >= 10 else np.nan,
                    'pure_ml_mae_mean': float(pure['mae'].mean()) if len(pure) >= 10 else np.nan,
                    'mae_abs_diff': np.nan,
                    'pct_improvement': np.nan,
                    'wilcoxon_p': np.nan,
                    'holm_p': np.nan,
                    'low_n_flag': 'LOW_N',
                })
                continue

            full_mean = float(paired['mae_full'].mean())
            pure_mean = float(paired['mae_pure'].mean())
            abs_diff = abs(full_mean - pure_mean)
            pct_improvement = 100.0 * (full_mean - pure_mean) / full_mean if full_mean != 0 else np.nan
            diffs = paired['mae_full'] - paired['mae_pure']
            if np.allclose(diffs.to_numpy(), 0.0):
                pval = 1.0
            else:
                try:
                    pval = float(wilcoxon(diffs.to_numpy(), zero_method='wilcox', alternative='two-sided').pvalue)
                except ValueError:
                    pval = 1.0
            comparison_rows.append({
                'arm': arm_name,
                'lstm_units': int(units),
                'n_paired_runs': int(n_pair),
                'full_hybrid_mae_mean': full_mean,
                'pure_ml_mae_mean': pure_mean,
                'mae_abs_diff': abs_diff,
                'pct_improvement': pct_improvement,
                'wilcoxon_p': pval,
                'holm_p': np.nan,
                'low_n_flag': '',
            })
            pvals.append(pval)
            pair_metadata.append((arm_name, int(units)))

    comparison_df = pd.DataFrame(comparison_rows).sort_values(['arm', 'lstm_units']).reset_index(drop=True)
    if pvals:
        adjusted = holm_adjust(pvals)
        for (arm_name, units), adj in zip(pair_metadata, adjusted):
            mask = (comparison_df['arm'] == arm_name) & (comparison_df['lstm_units'] == units)
            comparison_df.loc[mask, 'holm_p'] = float(adj)
    return comparison_df


def write_markdown(summary_df: pd.DataFrame, comparison_df: pd.DataFrame) -> None:
    md_lines = [
        '# Capacity sweep summary',
        '',
        '> Cells marked as LOW_N have fewer than 10 runs and are excluded from mean/SD reporting; no value is fabricated or interpolated.',
        '',
    ]

    for arm_name in ['full', 'interior', 'matched']:
        arm_df = summary_df[summary_df['arm'] == arm_name].copy().sort_values(['lstm_units', 'configuration'])
        md_lines.append(f'## Arm {ARM_LABELS[arm_name]}')
        md_lines.append('')
        md_lines.append('| hidden_units | configuration | n_runs | MAE mean | MAE SD | MAE min | MAE max | mean R² | flag |')
        md_lines.append('|---:|---|---:|---:|---:|---:|---:|---:|---|')
        for _, row in arm_df.iterrows():
            md_lines.append(
                f"| {int(row['lstm_units'])} | {row['configuration']} | {int(row['n_runs'])} | "
                f"{format_number(row['mae_mean'])} | {format_number(row['mae_sd'])} | {format_number(row['mae_min'])} | {format_number(row['mae_max'])} | "
                f"{format_number(row['r2_mean'])} | {row['low_n_flag']} |"
            )
        md_lines.append('')

        comp_df = comparison_df[comparison_df['arm'] == arm_name].copy().sort_values('lstm_units')
        md_lines.append(f'### FULL_HYBRID vs PURE_ML comparison for arm {ARM_LABELS[arm_name]}')
        md_lines.append('')
        md_lines.append('| hidden_units | FULL_HYBRID MAE mean | PURE_ML MAE mean | absolute diff | percent improvement | paired Wilcoxon p | Holm p | note |')
        md_lines.append('|---:|---:|---:|---:|---:|---:|---:|---|')
        if comp_df.empty:
            md_lines.append('| — | — | — | — | — | — | — | no matched comparison |')
        else:
            for _, row in comp_df.iterrows():
                md_lines.append(
                    f"| {int(row['lstm_units'])} | {format_number(row['full_hybrid_mae_mean'])} | {format_number(row['pure_ml_mae_mean'])} | "
                    f"{format_number(row['mae_abs_diff'])} | {format_number(row['pct_improvement'], digits=2)}% | "
                    f"{format_number(row['wilcoxon_p'], digits=3)} | {format_number(row['holm_p'], digits=3)} | {row['low_n_flag'] if row['low_n_flag'] else 'matched runs'} |"
                )
        md_lines.append('')

    output_path = ROOT / 'results' / 'processed' / 'capacity_summary.md'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(md_lines) + '\n', encoding='utf-8')


def main() -> None:
    summary_df = compute_summary_rows()
    comparison_df = compute_comparison_rows()
    comparison_columns = [
        'arm', 'lstm_units', 'n_paired_runs', 'full_hybrid_mae_mean',
        'pure_ml_mae_mean', 'mae_abs_diff', 'pct_improvement',
        'wilcoxon_p', 'holm_p', 'low_n_flag',
    ]
    summary_df = summary_df.merge(
        comparison_df[comparison_columns],
        on=['arm', 'lstm_units'],
        how='left',
        suffixes=('', '_comparison'),
    )

    processed_dir = ROOT / 'results' / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)

    summary_table_path = processed_dir / 'capacity_summary_table.csv'
    summary_df.to_csv(summary_table_path, index=False)

    write_markdown(summary_df, comparison_df)

    print(f'Wrote {summary_table_path}')
    print(f'Wrote {processed_dir / "capacity_summary.md"}')
    print('\nSummary table preview:')
    print(summary_df.to_string(index=False))
    print('\nComparison table preview:')
    print(comparison_df.to_string(index=False))


if __name__ == '__main__':
    main()
