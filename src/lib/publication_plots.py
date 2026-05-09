"""
publication_plots.py

This module provides standardized plotting utilities that produce
publication-quality figures suitable for medical and scientific journals.
Centralizing these settings in one module ensures that all figures
across the project have consistent visual style, which makes the
research look more professional and helps reviewers focus on the
content rather than visual inconsistencies.

The styling choices follow conventions that work well for medical
journals. We use a colorblind-accessible palette based on the Wong
2011 Nature Methods recommendations, which means people with the
most common forms of color blindness can still distinguish all the
colors in our figures. We use sans-serif fonts at sizes that remain
readable when figures are reduced for print. We remove the top and
right axis spines for a cleaner look that is now standard in most
journal publications. We use 300 DPI for raster outputs which is the
minimum acceptable resolution for most journals.

The module exposes simple high-level functions that handle the
common figure types in our research, plus utility functions for
setting up the matplotlib environment and saving figures in multiple
formats.

AUTHOR: Christopher Morris
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import os


# Colorblind-accessible color palette based on Wong 2011 Nature Methods.
# These specific hex codes have been tested to be distinguishable by
# people with most common forms of color blindness while still looking
# good in print.
COLORS = {
    'blue': '#0072B2',
    'orange': '#E69F00',
    'green': '#009E73',
    'yellow': '#F0E442',
    'sky_blue': '#56B4E9',
    'vermillion': '#D55E00',
    'purple': '#CC79A7',
    'black': '#000000',
}

# Standard color assignments for our specific algorithms. Using the
# same colors consistently across all figures helps readers track
# which algorithm is which without having to constantly reference
# the legend.
ALGORITHM_COLORS = {
    'XGBoost': COLORS['orange'],
    'Random Forest': COLORS['green'],
    'SVM': COLORS['purple'],
    'MLP': COLORS['vermillion'],
    'LSTM': COLORS['blue'],
    'GRU': COLORS['sky_blue'],
    'Pure ML': COLORS['orange'],
    'Hybrid': COLORS['blue'],
}


def setup_publication_style():
    """
    Configure matplotlib for publication-quality output. Call this once
    at the start of any plotting script. The settings affect all
    subsequently created figures until matplotlib is restarted.
    
    The settings target medical journal requirements specifically.
    Font sizes are calibrated for readability when figures are reduced
    to single-column width in a journal. The DPI is set to 300 which
    is the minimum for print publication. Spines are removed for a
    modern look. Fonts are set to embed properly in PDFs which is
    required by some journals.
    """
    plt.rcParams.update({
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'figure.facecolor': 'white',
        'figure.figsize': (6.5, 4.5),
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
        'savefig.pad_inches': 0.05,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })


def save_figure(fig, name, output_dir='figures', formats=None):
    """
    Save a matplotlib figure in multiple formats with consistent settings.
    
    By default we save in both PNG format which is convenient for
    embedding in manuscripts during writing and review, and PDF format
    which is the vector format most journals require for final
    publication. The PNG and PDF versions are saved in subfolders so
    they stay organized as the project grows.
    
    Args:
        fig: matplotlib Figure object to save
        name: filename without extension
        output_dir: directory to save in, will be created if needed
        formats: list of formats to save, defaults to png and pdf
    """
    if formats is None:
        formats = ['png', 'pdf']
    
    saved_paths = []
    for fmt in formats:
        subdir = os.path.join(output_dir, fmt)
        os.makedirs(subdir, exist_ok=True)
        path = os.path.join(subdir, f'{name}.{fmt}')
        fig.savefig(path, dpi=300, bbox_inches='tight')
        saved_paths.append(path)
    
    return saved_paths
