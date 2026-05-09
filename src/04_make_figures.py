#!/usr/bin/env python3
"""
make_figures.py
===============

This script generates publication-quality figures from the ablation results.
The figures use a colorblind-accessible palette and are saved in both PNG
format for embedding in manuscripts and PDF format for final publication.

The main figure is a grouped bar chart showing the degradation in mean
absolute error for each feature configuration, with separate bars for
XGBoost and LSTM. The visual comparison between the two algorithms is
what makes our computational discovery legible to readers.

USAGE:
    python make_figures.py

REQUIREMENTS:
    degradation_analysis.csv in the current directory

OUTPUT:
    figures/ablation_main.png and figures/ablation_main.pdf

AUTHOR: Christopher Morris
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import sys

DEGRADATION_FILE = 'degradation_analysis.csv'
FIGURES_DIR = 'figures'

# Configure matplotlib for publication quality. These settings are based on
# typical journal requirements with colorblind-accessible colors.
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

# Colorblind-accessible palette based on Wong 2011 Nature Methods recommendations
COLOR_XGB = '#E69F00'   # Orange for XGBoost
COLOR_LSTM = '#0072B2'  # Blue for LSTM

print("=" * 70)
print("FIGURE GENERATION")
print("=" * 70)

# Verify input file exists
if not os.path.exists(DEGRADATION_FILE):
    print(f"ERROR: Degradation analysis file not found: {DEGRADATION_FILE}")
    print("Please run 'python analyze_results.py' first.")
    sys.exit(1)

# Create figures directory
os.makedirs(FIGURES_DIR, exist_ok=True)

# Load the degradation data
print(f"\nLoading degradation analysis from {DEGRADATION_FILE}...")
degradation_df = pd.read_csv(DEGRADATION_FILE)
print(f"   Loaded {len(degradation_df)} rows")

lstm_rows = degradation_df.loc[degradation_df['algorithm'] == 'LSTM', 'n_runs']
lstm_n_seeds = int(lstm_rows.max()) if len(lstm_rows) else 0

# Pivot the data so configurations are rows and algorithms are columns
plot_data = degradation_df.pivot(
    index='configuration', 
    columns='algorithm', 
    values='degradation_pct'
)

plot_std = degradation_df.pivot(
    index='configuration',
    columns='algorithm',
    values='degradation_std_pct'
)

# Order configurations in a meaningful way for the figure
config_order = ['FULL_HYBRID', 'NO_TEMPORAL', 'NO_STATIC_MECH', 'PURE_ML', 'ONLY_MECH']
plot_data = plot_data.reindex(config_order)
plot_std = plot_std.reindex(config_order)

# Create the main figure
print("\nGenerating main ablation figure...")

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(plot_data.index))
width = 0.35

# XGBoost bars (deterministic, no error bars needed)
xgb_values = plot_data['XGBoost'].values if 'XGBoost' in plot_data.columns else np.zeros(len(x))
bars1 = ax.bar(x - width/2, xgb_values, width, 
               label='XGBoost (deterministic)', 
               color=COLOR_XGB, edgecolor='black', linewidth=0.5)

# LSTM bars with error bars from multi-seed standard deviation
lstm_values = plot_data['LSTM'].values if 'LSTM' in plot_data.columns else np.zeros(len(x))
lstm_errors = plot_std['LSTM'].values if 'LSTM' in plot_std.columns else np.zeros(len(x))
bars2 = ax.bar(x + width/2, lstm_values, width,
               yerr=lstm_errors, capsize=4,
               label=f'LSTM (mean ± std, n={lstm_n_seeds} seeds)',
               color=COLOR_LSTM, edgecolor='black', linewidth=0.5,
               error_kw={'linewidth': 1, 'ecolor': 'black'})

# Annotate each bar with its value
for bars, values in [(bars1, xgb_values), (bars2, lstm_values)]:
    for bar, value in zip(bars, values):
        if abs(value) > 0.5:  # Only label bars with meaningful values
            height = bar.get_height()
            ax.annotate(
                f'{value:+.1f}%',
                xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 4 if height >= 0 else -14),
                textcoords='offset points',
                ha='center', 
                va='bottom' if height >= 0 else 'top',
                fontsize=9, 
                fontweight='bold'
            )

# Reference line at zero
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.7)

# Labels and styling
ax.set_xlabel('Feature Configuration', fontweight='bold')
ax.set_ylabel('MAE Degradation from Full Hybrid (%)', fontweight='bold')
ax.set_title('Ablation Study: Algorithm-Dependent Mechanistic Feature Contributions',
             fontweight='bold', pad=10)

ax.set_xticks(x)
labels = plot_data.index.tolist()
# Replace underscores with spaces for cleaner labels
clean_labels = [l.replace('_', ' ') for l in labels]
ax.set_xticklabels(clean_labels, rotation=15, ha='right')

ax.legend(loc='upper left', frameon=False)
ax.yaxis.grid(True, alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

plt.tight_layout()

# Save in both formats
png_path = os.path.join(FIGURES_DIR, 'ablation_main.png')
pdf_path = os.path.join(FIGURES_DIR, 'ablation_main.pdf')
plt.savefig(png_path, dpi=300, bbox_inches='tight')
plt.savefig(pdf_path, bbox_inches='tight')
plt.close()

print(f"   Saved {png_path}")
print(f"   Saved {pdf_path}")

# Create a second figure showing absolute MAE values for context
print("\nGenerating absolute MAE figure...")

# Pivot for absolute values
mae_data = degradation_df.pivot(
    index='configuration',
    columns='algorithm',
    values='mae_mean'
).reindex(config_order)

mae_std = degradation_df.pivot(
    index='configuration',
    columns='algorithm',
    values='mae_std'
).reindex(config_order)

fig, ax = plt.subplots(figsize=(10, 6))

xgb_vals = mae_data['XGBoost'].values if 'XGBoost' in mae_data.columns else np.zeros(len(x))
lstm_vals = mae_data['LSTM'].values if 'LSTM' in mae_data.columns else np.zeros(len(x))
lstm_errs = mae_std['LSTM'].values if 'LSTM' in mae_std.columns else np.zeros(len(x))

ax.bar(x - width/2, xgb_vals, width,
       label='XGBoost (deterministic)',
       color=COLOR_XGB, edgecolor='black', linewidth=0.5)

ax.bar(x + width/2, lstm_vals, width,
       yerr=lstm_errs, capsize=4,
       label=f'LSTM (mean ± std, n={lstm_n_seeds} seeds)',
       color=COLOR_LSTM, edgecolor='black', linewidth=0.5,
       error_kw={'linewidth': 1, 'ecolor': 'black'})

ax.set_xlabel('Feature Configuration', fontweight='bold')
ax.set_ylabel('Mean Absolute Error', fontweight='bold')
ax.set_title('Absolute Performance Across Feature Configurations',
             fontweight='bold', pad=10)

ax.set_xticks(x)
ax.set_xticklabels(clean_labels, rotation=15, ha='right')
ax.legend(loc='upper left', frameon=False)
ax.yaxis.grid(True, alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

plt.tight_layout()

png_path2 = os.path.join(FIGURES_DIR, 'absolute_mae.png')
pdf_path2 = os.path.join(FIGURES_DIR, 'absolute_mae.pdf')
plt.savefig(png_path2, dpi=300, bbox_inches='tight')
plt.savefig(pdf_path2, bbox_inches='tight')
plt.close()

print(f"   Saved {png_path2}")
print(f"   Saved {pdf_path2}")

print("\n" + "=" * 70)
print("All figures generated successfully")
print("=" * 70)
print()
print("Figures are saved in the 'figures' subfolder.")
print("PNG files are for embedding in manuscripts.")
print("PDF files are vector format for final publication.")
