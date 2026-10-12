#!/usr/bin/env python3
"""Generate the corrected figure set from the post-fix processed results."""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / 'src'
FIG_DIR = ROOT / 'figures'
PROCESSED_DIR = ROOT / 'results' / 'processed'
CACHE_FILE = ROOT / 'cached_features.pkl'

sys.path.insert(0, str(SRC_DIR))
from compute_population_baseline import (
    BASE_SEED,
    compute_full_cohort_split,
    fit_scale_anchors,
    predictions_with_anchors,
)
from lib.publication_plots import setup_publication_style


def save_figure(fig, stem: str):
    for ext in ('png', 'pdf'):
        path = FIG_DIR / f'{stem}.{ext}'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f'  wrote {path.relative_to(ROOT)}')
    plt.close(fig)


def ensure_inputs(*paths: Path):
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f'Missing required source file: {path}')


def plot_capacity_sweep_by_arm():
    summary = pd.read_csv(PROCESSED_DIR / 'capacity_summary_table.csv')
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    arm_labels = {
        'full': 'A (full cohort)',
        'interior': 'B (interior cohort)',
        'matched': 'C (matched cohort)',
    }
    for ax, arm_name in zip(axes, ['full', 'interior', 'matched']):
        arm_df = summary[summary['arm'] == arm_name].copy()
        arm_df = arm_df.sort_values(['lstm_units', 'configuration'])
        for config in ['FULL_HYBRID', 'PURE_ML']:
            sub = arm_df[arm_df['configuration'] == config].sort_values('lstm_units')
            if sub.empty:
                continue
            x = sub['lstm_units'].astype(int).to_numpy()
            y = sub['mae_mean'].astype(float).to_numpy()
            yerr = sub['mae_sd'].astype(float).to_numpy()
            color = '#0072B2' if config == 'FULL_HYBRID' else '#D55E00'
            label = 'Full Hybrid' if config == 'FULL_HYBRID' else 'Pure ML'
            ax.errorbar(x, y, yerr=yerr, fmt='-o', color=color, linewidth=2, label=label, capsize=4)
        ax.set_title(f'Arm {arm_labels[arm_name].split()[0]}')
        ax.set_xlabel('LSTM hidden units')
        ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.3)
        ax.set_axisbelow(True)
        ax.set_xticks(sorted(arm_df['lstm_units'].dropna().unique().tolist()))
    axes[0].set_ylabel('MAE')
    axes[0].legend(loc='best')
    fig.suptitle('Capacity sweep by arm: corrected post-fix MAE vs hidden units')
    fig.tight_layout()
    save_figure(fig, 'capacity_sweep_by_arm')


def plot_arm_b_mechanistic():
    summary = pd.read_csv(PROCESSED_DIR / 'capacity_summary_table.csv')
    arm_b = summary[(summary['arm'] == 'interior') & (summary['lstm_units'] == 4)].copy()
    arm_b = arm_b.sort_values('configuration')
    mech = pd.read_csv(PROCESSED_DIR / 'arm_b_mechanistic_comparison.csv')
    mech_rows = mech[mech['cohort'].str.contains('Arm B 4-unit capacity sweep', na=False)].copy()
    mech_rows['configuration'] = mech_rows['cohort'].str.replace('Arm B 4-unit capacity sweep: ', '', regex=False)
    mech_rows['mae'] = mech_rows['mae'].astype(float)
    mech_rows['sd_run_mae_means'] = mech_rows['sd_run_mae_means'].astype(float)

    ordered = ['FULL_HYBRID', 'NO_STATIC_MECH', 'NO_TEMPORAL', 'PURE_ML', 'ONLY_MECH']
    config_rows = []
    for config in ordered:
        if config in {'FULL_HYBRID', 'PURE_ML'}:
            row = arm_b[arm_b['configuration'] == config].iloc[0]
            config_rows.append({
                'configuration': config,
                'mae_mean': float(row['mae_mean']),
                'mae_sd': float(row['mae_sd']),
            })
        else:
            item = mech_rows[mech_rows['configuration'] == config]
            if item.empty:
                continue
            config_rows.append({
                'configuration': config,
                'mae_mean': float(item['mae'].iat[0]),
                'mae_sd': float(item['sd_run_mae_means'].iat[0]) if pd.notna(item['sd_run_mae_means'].iat[0]) else 0.0,
            })

    df = pd.DataFrame(config_rows).sort_values('mae_mean').reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(df))
    colors = ['#000000', '#E69F00', '#0072B2', '#D55E00', '#009E73']
    y = df['mae_mean'].astype(float).to_numpy()
    yerr = df['mae_sd'].fillna(0.0).astype(float).to_numpy()
    ax.bar(x, y, yerr=yerr, capsize=4, color=colors[:len(df)], edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', ' ') for c in df['configuration']], rotation=20, ha='right')
    ax.set_ylabel('MAE')
    ax.set_title('Arm B: 4-unit mechanistic ablation (interior cohort)')
    ax.text(
        0.5, 0.98,
        'Y-axis truncated at 0.30 to emphasize the spread among the better-performing variants.',
        transform=ax.transAxes,
        ha='center', va='top', fontsize=9,
    )
    ax.set_ylim(0.30, max(y) * 1.15)
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_figure(fig, 'arm_b_mechanistic_4u')


def plot_arm_a_pre_vs_post():
    legacy = pd.read_csv(ROOT / 'capacity_sweep_results.csv')
    current = pd.read_csv(ROOT / 'capacity_sweep_arm_a_full_results.csv')
    legacy = legacy[legacy['algorithm'] == 'LSTM'].copy()
    current = current[current['algorithm'] == 'LSTM'].copy()
    for frame in (legacy, current):
        frame['lstm_units'] = frame['lstm_units'].astype(int)
        frame['configuration'] = frame['configuration'].astype(str)

    legacy_summary = legacy[legacy['configuration'].isin(['FULL_HYBRID', 'PURE_ML'])].groupby(['lstm_units', 'configuration'])['mae'].mean().reset_index()
    current_summary = current[current['configuration'].isin(['FULL_HYBRID', 'PURE_ML'])].groupby(['lstm_units', 'configuration'])['mae'].mean().reset_index()

    legacy_std = legacy[legacy['configuration'].isin(['FULL_HYBRID', 'PURE_ML'])].groupby(['lstm_units', 'configuration'])['mae'].std(ddof=1).reset_index().rename(columns={'mae': 'mae_sd'})
    current_std = current[current['configuration'].isin(['FULL_HYBRID', 'PURE_ML'])].groupby(['lstm_units', 'configuration'])['mae'].std(ddof=1).reset_index().rename(columns={'mae': 'mae_sd'})
    legacy_summary = legacy_summary.merge(legacy_std, on=['lstm_units', 'configuration'], how='left')
    current_summary = current_summary.merge(current_std, on=['lstm_units', 'configuration'], how='left')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    series_colors = {'FULL_HYBRID': '#0072B2', 'PURE_ML': '#D55E00'}
    panel_specs = [
        ('Pre-fix Arm A: commit 2fa4c4c', legacy_summary, '#0072B2'),
        ('Post-fix Arm A: commit a60be12', current_summary, '#0072B2'),
    ]
    for ax, (title, data, _) in zip(axes, panel_specs):
        for config in ['FULL_HYBRID', 'PURE_ML']:
            sub = data[data['configuration'] == config].sort_values('lstm_units')
            ax.errorbar(
                sub['lstm_units'],
                sub['mae'],
                yerr=sub['mae_sd'].fillna(0.0),
                fmt='-o',
                label=config.replace('_', ' '),
                color=series_colors[config],
                capsize=4,
                linewidth=2,
            )
        ax.set_xlabel('LSTM hidden units')
        ax.set_ylabel('MAE')
        ax.set_title(title)
        ax.tick_params(axis='both', which='both', labelleft=True, labelbottom=True)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)
        ax.legend(loc='best')
        ax.text(0.5, 1.02, title.split('commit ')[-1], transform=ax.transAxes, ha='center', va='bottom', fontsize=9)
    fig.suptitle('Arm A pre-fix vs post-fix artifact check: full-cohort runs only; pre-fix commit 2fa4c4c vs corrected run a60be12')
    fig.tight_layout()
    save_figure(fig, 'arm_a_pre_post_fix')


def plot_scale_recalibration():
    ensure_inputs(CACHE_FILE, PROCESSED_DIR / 'baseline_scale_calibration.csv')
    with open(CACHE_FILE, 'rb') as fh:
        df = pickle.load(fh)
    train_df, test_df = compute_full_cohort_split(df)
    anchors = fit_scale_anchors(train_df)
    ce = test_df['Ce'].to_numpy(dtype=float)
    true_y = test_df['actual_sas'].to_numpy(dtype=float)
    published_pred = predictions_with_anchors(ce, 3.08, e0=7.0, emax=6.0)
    calibrated_pred = predictions_with_anchors(ce, test_df['EC50_calibrated'].to_numpy(dtype=float), anchors['e0'], anchors['emax'])

    observed_bins = np.linspace(true_y.min(), true_y.max(), 11)
    observed_groups = pd.cut(true_y, bins=observed_bins, include_lowest=True)

    published_by_group = []
    calibrated_by_group = []
    labels = []
    for group in observed_groups.unique():
        mask = observed_groups == group
        if mask.sum() < 3:
            continue
        published_by_group.append(published_pred[mask])
        calibrated_by_group.append(calibrated_pred[mask])
        labels.append(f'{group.left:.1f}-{group.right:.1f}')

    fig, ax = plt.subplots(figsize=(9, 6))
    positions = np.arange(1, len(labels) + 1)
    published_bp = ax.boxplot(published_by_group, positions=positions - 0.15, widths=0.25, patch_artist=True, manage_ticks=False)
    calibrated_bp = ax.boxplot(calibrated_by_group, positions=positions + 0.15, widths=0.25, patch_artist=True, manage_ticks=False)
    for box in published_bp['boxes']:
        box.set(facecolor='#999999', alpha=0.6, edgecolor='black')
    for box in calibrated_bp['boxes']:
        box.set(facecolor='#0072B2', alpha=0.7, edgecolor='black')
    lims = [min(true_y.min(), published_pred.min(), calibrated_pred.min()), max(true_y.max(), published_pred.max(), calibrated_pred.max())]
    ax.plot(lims, lims, '--', color='black', linewidth=1, label='Identity line')
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_xlabel('Observed SAS bin')
    ax.set_ylabel('Predicted SAS')
    ax.set_title('Scale recalibration: published anchors vs cohort-calibrated anchors')
    ax.legend(['Published anchors', 'Calibrated anchors', 'Identity line'], loc='best', frameon=False)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, 'scale_recalibration')


def plot_baseline_comparison():
    baseline = pd.read_csv(PROCESSED_DIR / 'baseline_comparison.csv')
    keep = [
        'Persistence (SAS stays at sas_last)',
        'XGBoost',
        'LSTM PURE_ML, 4 units',
        'LSTM FULL_HYBRID, 4 units',
        'Eleveld, population EC50 (training-calibrated E0/Emax)',
        'Eleveld, personalized EC50 (same training-calibrated E0/Emax)',
    ]
    plot_df = baseline[baseline['method'].isin(keep)].copy()
    plot_df['method'] = plot_df['method'].str.replace('LSTM ', '', regex=False)
    plot_df = plot_df.sort_values('mae', ascending=True).reset_index(drop=True)
    persistence_value = float(plot_df.loc[plot_df['method'].str.contains('Persistence', case=False), 'mae'].iloc[0])
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#6b6b6b', '#E69F00', '#D55E00', '#0072B2', '#009E73', '#CC79A7']
    ax.bar(np.arange(len(plot_df)), plot_df['mae'].astype(float), color=colors[:len(plot_df)])
    ax.axhline(persistence_value, color='black', linestyle='--', linewidth=1.2, label='Persistence reference')
    ax.set_xticks(np.arange(len(plot_df)))
    ax.set_xticklabels(plot_df['method'], rotation=20, ha='right')
    ax.set_ylabel('MAE')
    ax.set_title('Baseline comparison: training-calibrated Eleveld vs clinical and ML baselines')
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', frameon=False)
    fig.tight_layout()
    save_figure(fig, 'baseline_comparison')


def generate_cohort_figures():
    script = ROOT / 'src' / '05_cohort_figures.py'
    ensure_inputs(script, CACHE_FILE, ROOT / 'data' / 'kidney_subgroups_patients.csv')
    subprocess.run([sys.executable, str(script)], cwd=str(ROOT), check=True)


def write_manifest():
    figures = [
        ('capacity_sweep_by_arm.png', 'src/04b_plot_capacity_sweep.py', ['results/processed/capacity_summary_table.csv']),
        ('capacity_sweep_by_arm.pdf', 'src/04b_plot_capacity_sweep.py', ['results/processed/capacity_summary_table.csv']),
        ('arm_b_mechanistic_4u.png', 'src/06_generate_corrected_figures.py', ['results/processed/capacity_summary_table.csv', 'results/processed/arm_b_mechanistic_comparison.csv']),
        ('arm_b_mechanistic_4u.pdf', 'src/06_generate_corrected_figures.py', ['results/processed/capacity_summary_table.csv', 'results/processed/arm_b_mechanistic_comparison.csv']),
        ('arm_a_pre_post_fix.png', 'src/06_generate_corrected_figures.py', ['capacity_sweep_results.csv', 'capacity_sweep_arm_a_full_results.csv']),
        ('arm_a_pre_post_fix.pdf', 'src/06_generate_corrected_figures.py', ['capacity_sweep_results.csv', 'capacity_sweep_arm_a_full_results.csv']),
        ('scale_recalibration.png', 'src/06_generate_corrected_figures.py', ['cached_features.pkl', 'results/processed/baseline_scale_calibration.csv']),
        ('scale_recalibration.pdf', 'src/06_generate_corrected_figures.py', ['cached_features.pkl', 'results/processed/baseline_scale_calibration.csv']),
        ('baseline_comparison.png', 'src/06_generate_corrected_figures.py', ['results/processed/baseline_comparison.csv']),
        ('baseline_comparison.pdf', 'src/06_generate_corrected_figures.py', ['results/processed/baseline_comparison.csv']),
    ]

    cohort_files = sorted((FIG_DIR / 'cohort').glob('*'))
    for path in cohort_files:
        if path.suffix.lower() in {'.png', '.pdf'}:
            figures.append((path.name, 'src/05_cohort_figures.py', ['cached_features.pkl', 'data/kidney_subgroups_patients.csv', 'cohort_per_patient_mae.csv']))

    lines = [
        '# Figure manifest',
        '',
        '| figure | generating script | sources | date generated |',
        '| --- | --- | --- | --- |',
    ]
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    for file_name, script_name, sources in figures:
        source_text = ', '.join(sources)
        lines.append(f'| {file_name} | {script_name} | {source_text} | {now} |')
    manifest_path = FIG_DIR / 'FIGURE_MANIFEST.md'
    manifest_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Wrote {manifest_path.relative_to(ROOT)}')


def main():
    setup_publication_style()
    FIG_DIR.mkdir(exist_ok=True)
    print(f'Started corrected-figure regeneration at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    plot_capacity_sweep_by_arm()
    plot_arm_b_mechanistic()
    plot_arm_a_pre_vs_post()
    plot_scale_recalibration()
    plot_baseline_comparison()
    generate_cohort_figures()
    write_manifest()
    print('Completed corrected figure regeneration.')


if __name__ == '__main__':
    main()
