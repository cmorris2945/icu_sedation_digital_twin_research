#!/usr/bin/env python3
"""Generate the corrected capacity-sweep figure set from the processed summary."""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from lib.publication_plots import setup_publication_style

SUMMARY_FILE = os.path.join('results', 'processed', 'capacity_summary_table.csv')
FIGURES_DIR = 'figures'

setup_publication_style()

if not os.path.exists(SUMMARY_FILE):
    raise FileNotFoundError(f'Missing post-fix source file: {SUMMARY_FILE}')

os.makedirs(FIGURES_DIR, exist_ok=True)
summary_df = pd.read_csv(SUMMARY_FILE)

# Keep the name used by the existing workflow while switching the data source to the corrected processed table.
arm_order = ['full', 'interior', 'matched']
arm_labels = {'full': 'A (full cohort)', 'interior': 'B (interior cohort)', 'matched': 'C (matched cohort)'}

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
for ax, arm_name in zip(axes, arm_order):
    arm_df = summary_df[summary_df['arm'] == arm_name].copy()
    arm_df = arm_df.sort_values(['lstm_units', 'configuration'])
    units = sorted(arm_df['lstm_units'].dropna().unique().tolist())
    for config in ['FULL_HYBRID', 'PURE_ML']:
        sub = arm_df[arm_df['configuration'] == config].sort_values('lstm_units')
        if sub.empty:
            continue
        x = sub['lstm_units'].astype(int).to_numpy()
        y = sub['mae_mean'].astype(float).to_numpy()
        yerr = sub['mae_sd'].astype(float).to_numpy()
        label = 'Full Hybrid' if config == 'FULL_HYBRID' else 'Pure ML'
        color = '#0072B2' if config == 'FULL_HYBRID' else '#D55E00'
        ax.errorbar(x, y, yerr=yerr, fmt='-o', color=color, linewidth=2, label=label, capsize=4)
    ax.set_title(f'Arm {arm_labels[arm_name].split()[0]}')
    ax.set_xlabel('LSTM hidden units')
    ax.set_xticks(units)
    ax.set_xticklabels([str(u) for u in units])
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.3)
    ax.set_axisbelow(True)

axes[0].set_ylabel('MAE')
axes[0].legend(loc='best')
fig.suptitle('Capacity sweep by arm: corrected post-fix MAE against hidden units')
fig.tight_layout()
for fmt in ('png', 'pdf'):
    path = os.path.join(FIGURES_DIR, f'capacity_sweep_by_arm.{fmt}')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f'Wrote {path}')
plt.close(fig)

print('All corrected capacity-sweep figures written to figures/.')
