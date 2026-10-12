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

CURRENT_SOURCE = os.path.join('results', 'processed', 'capacity_summary_table.csv')
FIGURES_DIR = 'figures'

print("=" * 70)
print("LEGACY ABLATION FIGURE GUARD")
print("=" * 70)
print('The legacy degradation_analysis.csv is a pre-fix artifact and is intentionally not used.')
print(f'Current post-fix source: {CURRENT_SOURCE}')
print('This repository now uses the processed capacity summary and baseline outputs instead.')
print('No stale result file was read.')
print('')

# Keep the script runnable without introducing stale-data reads.
if not os.path.exists(CURRENT_SOURCE):
    print(f'ERROR: required post-fix source is missing: {CURRENT_SOURCE}')
    sys.exit(1)

# Create figures directory for the current output set if needed.
os.makedirs(FIGURES_DIR, exist_ok=True)
print('Skipped legacy ablation generation; the current corrected pipeline is the capacity sweep and baseline figures.')
print('This script intentionally avoids the stale degradation_analysis.csv artifact.')
print('Done.')
