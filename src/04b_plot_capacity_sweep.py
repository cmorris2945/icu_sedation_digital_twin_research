#!/usr/bin/env python3
"""
04b_plot_capacity_sweep.py
==========================

This script generates publication-quality figures from the capacity-sweep
analysis produced by 03b_analyze_capacity_sweep.py. Two figures are produced:

1. capacity_sweep_main.png/pdf - line plot with LSTM size on the x-axis and
   degradation from FULL_HYBRID baseline on the y-axis, one line per feature
   configuration. Shaded bands represent ± std across seeds. The shape of the
   lines tells the visual story of whether feature contribution depends on
   model capacity.

2. capacity_heatmap.png/pdf - heatmap of mean degradation values across the
   size x configuration grid. Diverging color scheme centered on zero so
   positive and negative degradations are visually distinct.

The same publication style settings as the main ablation figures are used
for visual consistency in the paper.

USAGE:
    python 04b_plot_capacity_sweep.py

REQUIREMENTS:
    capacity_sweep_summary.csv in the current directory

OUTPUT:
    figures/capacity_sweep_main.{png,pdf}
    figures/capacity_heatmap.{png,pdf}

AUTHOR: Christopher Morris
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import sys

SUMMARY_FILE = 'capacity_sweep_summary.csv'
FIGURES_DIR = 'figures'

# Same publication style as the main ablation figures
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'figure.facecolor': 'white',
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'savefig.bbox': 'tight',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# Colorblind-accessible palette extended for five lines
CONFIG_COLORS = {
    'FULL_HYBRID':    '#000000',  # Black, baseline reference
    'NO_TEMPORAL':    '#0072B2',  # Blue
    'NO_STATIC_MECH': '#E69F00',  # Orange
    'PURE_ML':        '#D55E00',  # Vermillion
    'ONLY_MECH':      '#009E73',  # Green
}

# Configurations in display order
CONFIG_ORDER = ['FULL_HYBRID', 'NO_TEMPORAL', 'NO_STATIC_MECH', 'PURE_ML', 'ONLY_MECH']

print("=" * 70)
print("CAPACITY SWEEP FIGURES")
print("=" * 70)

if not os.path.exists(SUMMARY_FILE):
    print(f"ERROR: Summary file not found: {SUMMARY_FILE}")
    print("Please run 'python 03b_analyze_capacity_sweep.py' first.")
    sys.exit(1)

os.makedirs(FIGURES_DIR, exist_ok=True)

print(f"\nLoading summary from {SUMMARY_FILE}...")
summary_df = pd.read_csv(SUMMARY_FILE)
n_seeds_lstm = int(summary_df['n_runs'].max())
sizes = sorted(summary_df['lstm_units'].unique().tolist())
print(f"   {len(sizes)} sizes: {sizes}")
print(f"   {summary_df['configuration'].nunique()} configurations")
print(f"   max seeds per cell: {n_seeds_lstm}")

# ===================================================================
# Figure 1: capacity_sweep_main, line plot per configuration
# ===================================================================
print("\nGenerating main capacity sweep figure...")

fig, ax = plt.subplots(figsize=(10, 6))

for config in CONFIG_ORDER:
    sub = (summary_df[summary_df['configuration'] == config]
           .sort_values('lstm_units'))
    if len(sub) == 0:
        continue
    x = sub['lstm_units'].values
    y = sub['degradation_pct'].values
    yerr = sub['degradation_std_pct'].values

    color = CONFIG_COLORS[config]
    label = config.replace('_', ' ')

    if config == 'FULL_HYBRID':
        # Baseline is at 0 by definition. Draw as a faint dashed reference line
        # rather than a noisy line that just sits at y=0.
        ax.plot(x, y, color=color, linestyle='--', linewidth=1.2,
                marker='o', markersize=5, label=f'{label} (baseline)',
                alpha=0.6, zorder=1)
    else:
        ax.plot(x, y, color=color, linewidth=2, marker='o', markersize=6,
                label=label, zorder=3)
        ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.15,
                        zorder=2)

ax.axhline(y=0, color='black', linestyle='-', linewidth=0.7, zorder=0)
ax.set_xscale('log', base=2)
ax.set_xticks(sizes)
ax.set_xticklabels([str(s) for s in sizes])
ax.set_xlabel('LSTM hidden units', fontweight='bold')
ax.set_ylabel('MAE degradation from Full Hybrid (%)', fontweight='bold')
ax.set_title(f'LSTM capacity sweep: feature contribution vs model size '
             f'(n={n_seeds_lstm} seeds per cell)', fontweight='bold', pad=10)
ax.yaxis.grid(True, alpha=0.3, linestyle='--')
ax.set_axisbelow(True)
ax.legend(loc='best', frameon=False, ncol=2)

plt.tight_layout()

png_path = os.path.join(FIGURES_DIR, 'capacity_sweep_main.png')
pdf_path = os.path.join(FIGURES_DIR, 'capacity_sweep_main.pdf')
plt.savefig(png_path, dpi=300, bbox_inches='tight')
plt.savefig(pdf_path, bbox_inches='tight')
plt.close()

print(f"   Saved {png_path}")
print(f"   Saved {pdf_path}")

# ===================================================================
# Figure 2: capacity_heatmap, sizes x configurations heatmap of degradation
# ===================================================================
print("\nGenerating capacity heatmap figure...")

deg_pivot = (summary_df.pivot(index='lstm_units', columns='configuration',
                               values='degradation_pct')
             .reindex(columns=CONFIG_ORDER)
             .sort_index())

# Use a diverging colormap centered at zero so positive and negative
# degradations are visually distinct.
vmin = float(min(deg_pivot.min().min(), -1))
vmax = float(max(deg_pivot.max().max(), 1))
abs_max = max(abs(vmin), abs(vmax))

fig, ax = plt.subplots(figsize=(9, 4.5))
im = ax.imshow(deg_pivot.values, aspect='auto', cmap='RdBu_r',
               vmin=-abs_max, vmax=abs_max)

# Annotate cells with values
for i in range(deg_pivot.shape[0]):
    for j in range(deg_pivot.shape[1]):
        val = deg_pivot.values[i, j]
        text_color = 'white' if abs(val) > abs_max * 0.55 else 'black'
        ax.text(j, i, f'{val:+.1f}%', ha='center', va='center',
                color=text_color, fontsize=10, fontweight='bold')

ax.set_xticks(range(len(CONFIG_ORDER)))
ax.set_xticklabels([c.replace('_', ' ') for c in CONFIG_ORDER],
                   rotation=15, ha='right')
ax.set_yticks(range(len(deg_pivot.index)))
ax.set_yticklabels([str(int(s)) for s in deg_pivot.index])
ax.set_xlabel('Feature configuration', fontweight='bold')
ax.set_ylabel('LSTM hidden units', fontweight='bold')
ax.set_title(f'LSTM capacity heatmap: mean MAE degradation (%) from Full Hybrid '
             f'(n={n_seeds_lstm} seeds per cell)', fontweight='bold', pad=10)

cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cbar.set_label('Degradation (%)', fontweight='bold')

plt.tight_layout()

png_path = os.path.join(FIGURES_DIR, 'capacity_heatmap.png')
pdf_path = os.path.join(FIGURES_DIR, 'capacity_heatmap.pdf')
plt.savefig(png_path, dpi=300, bbox_inches='tight')
plt.savefig(pdf_path, bbox_inches='tight')
plt.close()

print(f"   Saved {png_path}")
print(f"   Saved {pdf_path}")

print("\n" + "=" * 70)
print("All capacity sweep figures generated successfully")
print("=" * 70)
