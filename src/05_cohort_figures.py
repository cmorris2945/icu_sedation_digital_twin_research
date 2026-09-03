#!/usr/bin/env python3
"""
05_cohort_figures.py
====================

Cohort description figures and per-patient error analysis, requested by
Dr. Kahveci: what does the cohort actually look like, and does the model's
error depend on any patient characteristic?

The script has two parts.

PART 1 builds cohort distributions. cached_features.pkl is instance-level
(30,931 rows), so it is collapsed to one row per patient (1,489 rows) before
anything is plotted. Patient-level demographics are taken from the raw
extraction CSV rather than from the pkl, because 01_build_features.py has
already substituted defaults for missing weight and height by the time the
pkl is written, and those defaults must not be mistaken for measurements.

PART 2 computes per-patient mean absolute error. 02_run_ablation.py and
02b_run_capacity_sweep.py only ever save aggregate MAE, so per-patient error
does not exist anywhere on disk and has to be regenerated. This script
retrains the best configuration from the capacity sweep (FULL_HYBRID at
lstm_units=4) with the identical patient-level split and the identical
architecture and training settings, across the same ten seeds, and keeps the
per-instance absolute errors on the test set.

Because retraining is the slow part, per-patient errors are cached to
cohort_per_patient_mae.csv. Re-running the script reuses that cache and
regenerates only the figures. Pass --force to retrain from scratch.

A NOTE ON IMPUTED VALUES
    398 of 1,490 patients (26.7%) have no recorded height and 18 (1.2%) have
    no recorded weight. 01_build_features.py assigns 170 cm and 75 kg to
    these. Any BMI computed from a defaulted height is fictitious, so every
    height- and BMI-derived statistic in this script uses MEASURED values
    only, and the excluded counts are reported on the figures and in the
    output tables. One panel deliberately shows the as-modelled height
    distribution, spike and all, because that spike is what the models
    actually saw and it belongs in the paper's limitations.

USAGE:
    python 05_cohort_figures.py [--force]

REQUIREMENTS:
    cached_features.pkl
    data/kidney_subgroups_patients.csv

OUTPUT:
    cohort_table1.csv                 descriptive statistics, one row per variable
    cohort_correlations.csv           Pearson and Spearman with p-values
    cohort_per_patient_mae.csv        per-patient test errors (cache)
    figures/cohort/*.png, *.pdf       four figures

EXPECTED RUNTIME:
    About 10 minutes on CPU for the ten retraining runs, then seconds for the
    figures. Near-instant on a re-run, because the errors are cached.

AUTHOR: Christopher Morris
"""

import os
import sys
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Make src/lib importable regardless of where the script is invoked from
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.publication_plots import setup_publication_style, COLORS

# ---------------------------------------------------------------------------
# Configuration. The model settings below are COPIED from
# 02b_run_capacity_sweep.py and must stay identical to it, or the per-patient
# errors will not correspond to the published 4-unit FULL_HYBRID result.
# ---------------------------------------------------------------------------
CACHE_FILE = 'cached_features.pkl'
PATIENTS_FILE = os.path.join('data', 'kidney_subgroups_patients.csv')
FIG_DIR = os.path.join('figures', 'cohort')
TABLE1_FILE = 'cohort_table1.csv'
CORR_FILE = 'cohort_correlations.csv'
MAE_CACHE_FILE = 'cohort_per_patient_mae.csv'

BASE_SEED = 42
SEQUENCE_LENGTH = 10
LSTM_UNITS = 4          # best configuration in the capacity sweep
FIXED_EPOCHS = 50
N_SEEDS = 10
SWEEP_REFERENCE_MAE = 0.3514    # capacity_sweep_summary.csv, units=4 FULL_HYBRID
SWEEP_REFERENCE_SD = 0.0175

# Single hue for magnitude, second hue only for the imputed-value overlay.
# Both drawn from the Wong palette in lib/publication_plots.py and checked
# for colourblind separation before use.
C_MAIN = COLORS['blue']
C_ALT = COLORS['orange']
C_MUTED = '#6b6b6b'

FORCE = '--force' in sys.argv

print("=" * 70)
print("COHORT FIGURES AND PER-PATIENT ERROR ANALYSIS")
print("=" * 70)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

for f in (CACHE_FILE, PATIENTS_FILE):
    if not os.path.exists(f):
        print(f"ERROR: required file not found: {f}")
        sys.exit(1)

os.makedirs(FIG_DIR, exist_ok=True)
setup_publication_style()


def save_figure(fig, name):
    """Write a figure to figures/cohort/ as both PNG and PDF, matching the
    flat layout the capacity sweep figures already use."""
    paths = []
    for fmt in ('png', 'pdf'):
        path = os.path.join(FIG_DIR, f'{name}.{fmt}')
        fig.savefig(path, dpi=300, bbox_inches='tight')
        paths.append(path)
    plt.close(fig)
    for p in paths:
        print(f"   Saved {p}")


def style_axis(ax):
    """Recessive grid and axes so the data carries the figure."""
    ax.grid(axis='y', color='#e6e6e6', linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


# ===========================================================================
# LOAD AND COLLAPSE TO PATIENT LEVEL
# ===========================================================================
print("Loading data...")
with open(CACHE_FILE, 'rb') as fh:
    inst = pickle.load(fh)
patients_raw = pd.read_csv(PATIENTS_FILE)

print(f"   instance level : {len(inst):,} rows, {inst['stay_id'].nunique():,} patients")
print(f"   extraction CSV : {len(patients_raw):,} patients")

# Static per-patient columns take the first value; they are constant within a
# patient by construction. propofol_rate varies per instance, so it is averaged.
pat = (inst.groupby('stay_id')
            .agg(age=('age', 'first'),
                 ec50=('EC50_calibrated', 'first'),
                 mean_propofol_rate=('propofol_rate', 'mean'),
                 n_instances=('actual_sas', 'size'))
            .reset_index())

# Weight and height come from the RAW extraction, not the pkl, so that the
# defaults 01_build_features.py substitutes are identifiable as missing.
pat = pat.merge(
    patients_raw[['stay_id', 'gender', 'weight', 'height']],
    on='stay_id', how='left'
)
pat['weight_measured'] = pat['weight'].notna()
pat['height_measured'] = pat['height'].notna()

# BMI only where BOTH inputs are real measurements.
both = pat['weight_measured'] & pat['height_measured']
pat['bmi'] = np.nan
pat.loc[both, 'bmi'] = pat.loc[both, 'weight'] / (pat.loc[both, 'height'] / 100.0) ** 2

# The as-modelled height, defaults included, purely so the artefact can be shown.
pat['height_as_modelled'] = pat['height'].fillna(170.0)

n_pat = len(pat)
n_h_missing = int((~pat['height_measured']).sum())
n_w_missing = int((~pat['weight_measured']).sum())
n_bmi = int(pat['bmi'].notna().sum())
print(f"   analysis cohort: {n_pat:,} patients")
print(f"   height missing : {n_h_missing} ({100*n_h_missing/n_pat:.1f}%) -> excluded from height/BMI")
print(f"   weight missing : {n_w_missing} ({100*n_w_missing/n_pat:.1f}%) -> excluded from weight/BMI")
print(f"   BMI computable : {n_bmi:,} patients ({100*n_bmi/n_pat:.1f}%)")

# Data-quality check. MIMIC charted weights and heights include recording
# errors, and these values feed the PK simulation through the mcg/kg/min ->
# mg/min conversion, so implausible extremes are worth surfacing rather than
# silently plotting.
IMPLAUSIBLE = [
    ('weight', 30, 250, 'kg'),
    ('height', 130, 210, 'cm'),
    ('bmi', 12, 70, 'kg/m2'),
]
flagged_any = False
for col, lo, hi, unit in IMPLAUSIBLE:
    s = pat[col].dropna()
    bad = s[(s < lo) | (s > hi)]
    if len(bad):
        flagged_any = True
        print(f"   DATA QUALITY: {len(bad)} patients have {col} outside "
              f"{lo}-{hi} {unit} (min {s.min():.1f}, max {s.max():.1f}). "
              f"Retained as recorded, not winsorised.")
if not flagged_any:
    print("   DATA QUALITY: no implausible weight/height/BMI values found.")

# Comorbidity flags: use only what the CSV actually contains.
FLAG_COLS = [c for c in patients_raw.columns if c.startswith('has_')]
pat = pat.merge(patients_raw[['stay_id'] + FLAG_COLS], on='stay_id', how='left')
print(f"   comorbidity flags found: {len(FLAG_COLS)}")
print()


# ===========================================================================
# PART 1a - TABLE 1
# ===========================================================================
print("Building cohort_table1.csv...")

CONTINUOUS = [
    ('age', 'Age (years)', 'all patients'),
    ('weight', 'Weight (kg)', 'measured only'),
    ('height', 'Height (cm)', 'measured only'),
    ('bmi', 'BMI (kg/m2)', 'measured weight and height only'),
    ('ec50', 'Calibrated EC50 (ug/mL)', 'all patients'),
    ('mean_propofol_rate', 'Mean propofol rate (mcg/kg/min)', 'mean over instances, zeros included'),
    ('n_instances', 'Prediction instances per patient', 'all patients'),
]

rows = []
for col, label, note in CONTINUOUS:
    s = pat[col].dropna()
    rows.append({
        'variable': label, 'type': 'continuous',
        'n': len(s), 'n_missing': int(pat[col].isna().sum()),
        'mean': round(float(s.mean()), 3), 'sd': round(float(s.std()), 3),
        'median': round(float(s.median()), 3),
        'q1': round(float(s.quantile(0.25)), 3),
        'q3': round(float(s.quantile(0.75)), 3),
        'min': round(float(s.min()), 3), 'max': round(float(s.max()), 3),
        'n_positive': '', 'pct_positive': '', 'note': note,
    })

for val, label in (('M', 'Sex: male'), ('F', 'Sex: female')):
    k = int((pat['gender'] == val).sum())
    rows.append({
        'variable': label, 'type': 'binary', 'n': n_pat, 'n_missing': 0,
        'mean': '', 'sd': '', 'median': '', 'q1': '', 'q3': '', 'min': '', 'max': '',
        'n_positive': k, 'pct_positive': round(100 * k / n_pat, 2), 'note': '',
    })

for col in FLAG_COLS:
    k = int(pat[col].sum())
    rows.append({
        'variable': col.replace('has_', 'Comorbidity: '), 'type': 'binary',
        'n': n_pat, 'n_missing': 0,
        'mean': '', 'sd': '', 'median': '', 'q1': '', 'q3': '', 'min': '', 'max': '',
        'n_positive': k, 'pct_positive': round(100 * k / n_pat, 2),
        'note': 'legacy Phase 1 flag; see instructions.txt Section 9 for known defects',
    })

# Imputation is part of describing the cohort honestly, so it is a table row.
for label, k in (('Height imputed to 170 cm', n_h_missing),
                 ('Weight imputed to 75 kg', n_w_missing)):
    rows.append({
        'variable': label, 'type': 'binary', 'n': n_pat, 'n_missing': 0,
        'mean': '', 'sd': '', 'median': '', 'q1': '', 'q3': '', 'min': '', 'max': '',
        'n_positive': k, 'pct_positive': round(100 * k / n_pat, 2),
        'note': 'substituted by 01_build_features.py; excluded from the stats above',
    })

table1 = pd.DataFrame(rows)
table1.to_csv(TABLE1_FILE, index=False)
print(f"   wrote {TABLE1_FILE} ({len(table1)} rows)")
print()


# ===========================================================================
# PART 1b - DISTRIBUTION FIGURES
# ===========================================================================
print("Generating distribution figures...")

HISTS = [
    ('age', 'Age (years)', 'all patients', None),
    ('weight', 'Weight (kg)', f'measured only, {n_w_missing} excluded', None),
    ('height', 'Height (cm)', f'measured only, {n_h_missing} excluded', None),
    ('bmi', 'BMI (kg/m$^2$)', f'measured only, n={n_bmi}', None),
    ('ec50', 'Calibrated EC50 ($\\mu$g/mL)', 'all patients', 'ec50'),
    ('mean_propofol_rate', 'Mean propofol rate (mcg/kg/min)', 'zeros included', None),
    ('n_instances', 'Prediction instances per patient', 'all patients', None),
]

fig, axes = plt.subplots(2, 4, figsize=(15, 7))
axes = axes.ravel()

for ax, (col, label, sub, special) in zip(axes, HISTS):
    s = pat[col].dropna()
    ax.hist(s, bins=30, color=C_MAIN, edgecolor='white', linewidth=0.4, zorder=2)
    ax.set_xlabel(label)
    ax.set_ylabel('Patients')
    ax.set_title(sub, fontsize=9, color=C_MUTED, loc='left')
    style_axis(ax)
    if special == 'ec50':
        # More than half the cohort is pinned at the optimiser's lower bound.
        # That is the single most important thing about this variable.
        n_lo = int((s <= 0.5001).sum())
        n_hi = int((s >= 9.999).sum())
        ax.axvline(0.5, color=C_ALT, linewidth=1.5, linestyle='--', zorder=3)
        ax.annotate(f'{100*n_lo/len(s):.0f}% pinned\nat lower bound',
                    xy=(0.5, ax.get_ylim()[1] * 0.72),
                    xytext=(2.0, ax.get_ylim()[1] * 0.72),
                    color=C_ALT, fontsize=8,
                    arrowprops=dict(arrowstyle='->', color=C_ALT, lw=1.0))
        print(f"   EC50 at lower bound: {n_lo} ({100*n_lo/len(s):.1f}%), "
              f"upper bound: {n_hi} ({100*n_hi/len(s):.1f}%)")

# Eighth panel: the height distribution the models actually consumed, so the
# imputation spike is visible rather than buried in a caption.
# The imputed values are all exactly 170.0, so histogramming them produces a
# degenerate zero-width bin that renders as an invisible hairline. Draw them
# as a single bar one bin wide instead, which is what the spike actually is.
ax = axes[7]
meas_h = pat.loc[pat['height_measured'], 'height']
h_bins = np.histogram_bin_edges(meas_h, bins=30)
ax.hist(meas_h, bins=h_bins,
        color=C_MAIN, edgecolor='white', linewidth=0.4, zorder=2,
        label=f'Measured (n={n_pat - n_h_missing})')
ax.bar(170.0, n_h_missing, width=(h_bins[1] - h_bins[0]),
       color=C_ALT, edgecolor='white', linewidth=0.4, zorder=3,
       label=f'Imputed 170 cm (n={n_h_missing})')
ax.set_xlabel('Height (cm), as seen by the models')
ax.set_ylabel('Patients')
ax.set_title('imputation artefact', fontsize=9, color=C_MUTED, loc='left')
ax.legend(frameon=False, fontsize=8)
style_axis(ax)

fig.suptitle(f'Cohort distributions (n = {n_pat:,} patients)', fontsize=13, x=0.02, ha='left')
fig.tight_layout(rect=(0, 0, 1, 0.97))
save_figure(fig, 'cohort_distributions')

# --- sex and comorbidity ---------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5),
                               gridspec_kw={'width_ratios': [1, 2.1]})

sex_counts = pat['gender'].value_counts().reindex(['M', 'F'])
bars = ax1.bar(['Male', 'Female'], sex_counts.values, color=C_MAIN,
               width=0.6, zorder=2)
for b, v in zip(bars, sex_counts.values):
    ax1.text(b.get_x() + b.get_width() / 2, v + n_pat * 0.012,
             f'{v:,}\n({100*v/n_pat:.1f}%)', ha='center', va='bottom', fontsize=9)
ax1.set_ylabel('Patients')
ax1.set_ylim(0, sex_counts.max() * 1.22)
ax1.set_title('Sex', fontsize=11, loc='left')
style_axis(ax1)

prev = (pat[FLAG_COLS].sum() / n_pat * 100).sort_values()
labels = [c.replace('has_', '').replace('_', ' ') for c in prev.index]
ax2.barh(labels, prev.values, color=C_MAIN, height=0.68, zorder=2)
for y, v in enumerate(prev.values):
    ax2.text(v + prev.max() * 0.015, y, f'{v:.1f}%', va='center', fontsize=8.5)
ax2.set_xlabel('Prevalence (% of cohort)')
ax2.set_xlim(0, prev.max() * 1.18)
ax2.set_title('Kidney comorbidity flags (legacy Phase 1; not used by the models)',
              fontsize=10, loc='left')
ax2.grid(axis='x', color='#e6e6e6', linewidth=0.6, zorder=0)
ax2.set_axisbelow(True)

fig.tight_layout()
save_figure(fig, 'cohort_sex_comorbidity')
print()


# ===========================================================================
# PART 2a - PER-PATIENT MAE (requires retraining)
# ===========================================================================

def compute_per_patient_mae():
    """Retrain 4-unit FULL_HYBRID across the sweep's ten seeds and return
    per-patient test errors plus the per-seed overall MAE.

    Everything here mirrors 02b_run_capacity_sweep.py exactly: the same
    derived Ce features, the same patient-level split with random_state=42,
    the same architecture, the same fixed 50 epochs and 0.15 validation
    split. Only the retention of per-instance predictions is new.
    """
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, concatenate
    from tensorflow.keras.optimizers import Adam
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error

    df = inst.copy()
    ce_seqs = np.array([np.array(s) for s in df['ce_sequence']])
    df['ce_seq_mean'] = ce_seqs.mean(axis=1)
    df['ce_seq_std'] = ce_seqs.std(axis=1)
    df['ce_seq_trend'] = ce_seqs[:, -1] - ce_seqs[:, 0]
    df['ce_seq_max'] = ce_seqs.max(axis=1)
    df['ce_seq_last'] = ce_seqs[:, -1]

    unique_patients = df['stay_id'].unique()
    train_patients, test_patients = train_test_split(
        unique_patients, test_size=0.2, random_state=BASE_SEED
    )
    train_df = df[df['stay_id'].isin(train_patients)].reset_index(drop=True)
    test_df = df[df['stay_id'].isin(test_patients)].reset_index(drop=True)
    print(f"   split: {len(train_patients)} train / {len(test_patients)} test patients")
    print(f"          {len(train_df):,} train / {len(test_df):,} test instances")

    CLINICAL = ['age', 'gender', 'weight', 'height', 'sas_last', 'sas_mean',
                'sas_std', 'sas_trend', 'propofol_rate', 'time_hours']
    STATIC_MECH = ['Ce', 'Cp', 'EC50_calibrated', 'mech_pred_pop', 'mech_pred_calib']
    static_feats = CLINICAL + STATIC_MECH      # FULL_HYBRID

    sas_train = np.array([np.array(s) for s in train_df['sas_sequence']])
    sas_test = np.array([np.array(s) for s in test_df['sas_sequence']])
    ce_train = np.array([np.array(s) for s in train_df['ce_sequence']])
    ce_test = np.array([np.array(s) for s in test_df['ce_sequence']])
    seq_train = np.stack([sas_train, ce_train], axis=-1)     # use_ce_seq = True
    seq_test = np.stack([sas_test, ce_test], axis=-1)

    static_train = train_df[static_feats].values
    static_test = test_df[static_feats].values
    y_train = train_df['actual_sas'].values
    y_test = test_df['actual_sas'].values

    ss_static = StandardScaler()
    static_train_s = ss_static.fit_transform(static_train)
    static_test_s = ss_static.transform(static_test)

    ss_seq = StandardScaler()
    seq_train_s = ss_seq.fit_transform(
        seq_train.reshape(-1, seq_train.shape[-1])).reshape(seq_train.shape)
    seq_test_s = ss_seq.transform(
        seq_test.reshape(-1, seq_test.shape[-1])).reshape(seq_test.shape)

    abs_err = np.zeros((N_SEEDS, len(test_df)))
    seed_maes = []

    for i in range(N_SEEDS):
        seed = BASE_SEED + i
        np.random.seed(seed)
        tf.random.set_seed(seed)

        seq_input = Input(shape=(SEQUENCE_LENGTH, seq_train.shape[-1]), name='seq')
        lstm_out = LSTM(LSTM_UNITS)(seq_input)
        lstm_out = Dropout(0.2)(lstm_out)
        static_input = Input(shape=(static_train.shape[1],), name='static')
        combined = concatenate([lstm_out, static_input])
        x = Dense(16, activation='relu')(combined)
        x = Dropout(0.2)(x)
        x = Dense(8, activation='relu')(x)
        output = Dense(1)(x)

        model = Model(inputs=[seq_input, static_input], outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        model.fit([seq_train_s, static_train_s], y_train,
                  epochs=FIXED_EPOCHS, batch_size=64,
                  validation_split=0.15, verbose=0)

        y_pred = model.predict([seq_test_s, static_test_s], verbose=0).flatten()
        abs_err[i] = np.abs(y_pred - y_test)
        mae = float(mean_absolute_error(y_test, y_pred))
        seed_maes.append(mae)
        print(f"   [{datetime.now().strftime('%H:%M:%S')}] seed {seed}: "
              f"test MAE = {mae:.4f}")

    # Per instance, average the absolute error over seeds; then average within
    # patient. Averaging over seeds first is what makes the per-patient number
    # stable enough to correlate against.
    test_out = pd.DataFrame({
        'stay_id': test_df['stay_id'].values,
        'abs_err': abs_err.mean(axis=0),
    })
    per_patient = (test_out.groupby('stay_id')['abs_err']
                   .agg(patient_mae='mean', n_test_instances='size')
                   .reset_index())
    return per_patient, seed_maes


if os.path.exists(MAE_CACHE_FILE) and not FORCE:
    print(f"Reusing cached per-patient errors from {MAE_CACHE_FILE} (--force to retrain).")
    per_patient = pd.read_csv(MAE_CACHE_FILE)
    seed_maes = None
else:
    print(f"Retraining FULL_HYBRID at lstm_units={LSTM_UNITS}, {N_SEEDS} seeds. "
          f"This is the slow part.")
    per_patient, seed_maes = compute_per_patient_mae()
    per_patient.to_csv(MAE_CACHE_FILE, index=False)
    print(f"   wrote {MAE_CACHE_FILE}")

print()
print("Sanity check against the published capacity sweep:")
if seed_maes is not None:
    m, sd = float(np.mean(seed_maes)), float(np.std(seed_maes, ddof=1))
    print(f"   this run   : instance-weighted test MAE = {m:.4f} +/- {sd:.4f} "
          f"over {N_SEEDS} seeds")
    print(f"   sweep      : {SWEEP_REFERENCE_MAE:.4f} +/- {SWEEP_REFERENCE_SD:.4f} "
          f"(capacity_sweep_summary.csv, units=4 FULL_HYBRID)")
    delta = m - SWEEP_REFERENCE_MAE
    print(f"   difference : {delta:+.4f} "
          f"({abs(delta)/SWEEP_REFERENCE_SD:.2f} sweep SDs)")
    if abs(delta) <= 2 * SWEEP_REFERENCE_SD:
        print("   VERDICT: consistent with the published result.")
    else:
        print("   VERDICT: NOT consistent. Investigate before using these errors.")
else:
    print("   skipped, per-patient errors came from cache.")

pw = float(per_patient['patient_mae'].mean())
print(f"   patient-weighted mean of per-patient MAE = {pw:.4f}")
print("   (differs from the instance-weighted figure because patients "
      "contribute unequal instance counts; both are correct, they answer "
      "different questions)")
print()


# ===========================================================================
# PART 2a-bis - NAIVE BASELINES
# ===========================================================================
# Not in the original request, but the per-patient errors made it unavoidable:
# a large share of test instances carry the same SAS as the previous
# observation, so "predict the last observed SAS" is a strong baseline and has
# to be reported. It needs no training and is fully deterministic.
print("Naive baselines on the identical test split...")

from sklearn.model_selection import train_test_split as _tts
from sklearn.metrics import mean_absolute_error as _mae

_up = inst['stay_id'].unique()
_tr, _te = _tts(_up, test_size=0.2, random_state=BASE_SEED)
_train = inst[inst['stay_id'].isin(_tr)]
_test = inst[inst['stay_id'].isin(_te)].copy()
_y = _test['actual_sas'].values

_unchanged = float((_test['actual_sas'] == _test['sas_last']).mean())
baselines = {
    'persistence (predict sas_last)': _mae(_y, _test['sas_last'].values),
    'window mean (predict sas_mean)': _mae(_y, _test['sas_mean'].values),
    'global train mean': _mae(_y, np.full(len(_y), _train['actual_sas'].mean())),
}

print(f"   test set: {len(_test):,} instances, {_test['stay_id'].nunique()} patients")
print(f"   SAS unchanged from previous observation in "
      f"{100*_unchanged:.1f}% of test instances")
for name, val in baselines.items():
    print(f"   {name:<32s} MAE = {val:.4f}")
print(f"   {'4-unit FULL_HYBRID (published)':<32s} MAE = {SWEEP_REFERENCE_MAE:.4f}")

_test['persistence_err'] = (_test['actual_sas'] - _test['sas_last']).abs()
_pers_pat = _test.groupby('stay_id')['persistence_err'].mean().rename('persistence_mae')
_cmp = per_patient.set_index('stay_id')[['patient_mae']].join(_pers_pat, how='inner').dropna()
_wstat, _wp = stats.wilcoxon(_cmp['patient_mae'], _cmp['persistence_mae'])
_model_wins = int((_cmp['patient_mae'] < _cmp['persistence_mae']).sum())

print(f"   paired over {len(_cmp)} test patients: model {_cmp['patient_mae'].mean():.4f} "
      f"vs persistence {_cmp['persistence_mae'].mean():.4f}, "
      f"Wilcoxon p = {_wp:.3g}")
print(f"   the model beats persistence for {_model_wins}/{len(_cmp)} patients "
      f"({100*_model_wins/len(_cmp):.1f}%)")
if baselines['persistence (predict sas_last)'] < SWEEP_REFERENCE_MAE:
    print("   *** PERSISTENCE OUTPERFORMS THE BEST MODEL IN THE CAPACITY SWEEP. ***")
    print("   *** This baseline must appear in the manuscript. See              ***")
    print("   *** instructions.txt Section 9.                                   ***")

# Instance-weighted and patient-weighted MAE are different quantities, so the
# table reports both and never compares across the two columns.
_rows = []
for name, val in baselines.items():
    col = {'persistence (predict sas_last)': 'persistence_err'}.get(name)
    _rows.append({
        'method': name, 'requires_training': False,
        'test_mae_instance_weighted': round(float(val), 4),
        'test_mae_patient_weighted': (
            round(float(_pers_pat.mean()), 4) if col else ''),
    })
_rows.append({
    'method': f'LSTM FULL_HYBRID {LSTM_UNITS} units (this run, {N_SEEDS}-seed mean)',
    'requires_training': True,
    'test_mae_instance_weighted': (round(float(np.mean(seed_maes)), 4)
                                   if seed_maes is not None else ''),
    'test_mae_patient_weighted': round(float(pw), 4),
})
_rows.append({
    'method': 'LSTM FULL_HYBRID 4 units (published capacity sweep)',
    'requires_training': True,
    'test_mae_instance_weighted': SWEEP_REFERENCE_MAE,
    'test_mae_patient_weighted': '',
})
pd.DataFrame(_rows).to_csv('cohort_baseline_comparison.csv', index=False)
print("   wrote cohort_baseline_comparison.csv")
print()


# ===========================================================================
# PART 2b - CORRELATIONS AND SCATTER PLOTS
# ===========================================================================
print("Computing correlations...")

pat_mae = pat.merge(per_patient, on='stay_id', how='inner')
print(f"   per-patient MAE available for {len(pat_mae)} test patients")

# EC50 is pinned at an optimiser bound for a large share of the cohort, which
# makes any correlation on the full cohort hard to interpret. Report the
# interior subset alongside it rather than instead of it.
interior = pat[(pat['ec50'] > 0.5001) & (pat['ec50'] < 9.999)]
print(f"   EC50 strictly interior for {len(interior)} of {n_pat} patients "
      f"({100*len(interior)/n_pat:.1f}%)")


def correlate(frame, xcol, ycol, xlabel, ylabel, subset):
    d = frame[[xcol, ycol]].dropna()
    if len(d) < 3:
        return None
    pr, pp = stats.pearsonr(d[xcol], d[ycol])
    sr, sp = stats.spearmanr(d[xcol], d[ycol])
    return {
        'y': ylabel, 'x': xlabel, 'subset': subset, 'n': len(d),
        'pearson_r': round(float(pr), 4), 'pearson_p': round(float(pp), 5),
        'spearman_rho': round(float(sr), 4), 'spearman_p': round(float(sp), 5),
    }


MAE_PAIRS = [('age', 'Age (years)'), ('weight', 'Weight (kg)'),
             ('bmi', 'BMI (kg/m$^2$)'), ('n_instances', 'Instances per patient')]
EC50_PAIRS = [('age', 'Age (years)'), ('weight', 'Weight (kg)'),
              ('bmi', 'BMI (kg/m$^2$)')]

corr_rows = []
for col, label in MAE_PAIRS:
    r = correlate(pat_mae, col, 'patient_mae', label.replace('$^2$', '2'),
                  'Per-patient MAE', 'test patients')
    if r:
        corr_rows.append(r)
for col, label in EC50_PAIRS:
    r = correlate(pat, col, 'ec50', label.replace('$^2$', '2'),
                  'Calibrated EC50', 'all patients')
    if r:
        corr_rows.append(r)
    r = correlate(interior, col, 'ec50', label.replace('$^2$', '2'),
                  'Calibrated EC50', 'EC50 interior only')
    if r:
        corr_rows.append(r)

corr = pd.DataFrame(corr_rows)

# Holm correction over the primary family only: the seven pre-specified tests,
# excluding the interior-only sensitivity rows.
primary = corr['subset'] != 'EC50 interior only'
for method in ('pearson', 'spearman'):
    p = corr.loc[primary, f'{method}_p'].values
    order = np.argsort(p)
    adj = np.empty(len(p))
    running = 0.0
    for rank, idx in enumerate(order):
        val = (len(p) - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    corr.loc[primary, f'{method}_p_holm'] = np.round(adj, 5)
corr['holm_family'] = np.where(primary, f'primary (n={int(primary.sum())} tests)',
                               'sensitivity, not corrected')
corr.to_csv(CORR_FILE, index=False)
print(f"   wrote {CORR_FILE} ({len(corr)} rows)")
print()
print(corr[['y', 'x', 'subset', 'n', 'pearson_r', 'pearson_p',
            'spearman_rho', 'spearman_p']].to_string(index=False))
print()


def scatter_panel(ax, frame, xcol, ycol, xlabel, ylabel):
    d = frame[[xcol, ycol]].dropna()
    ax.scatter(d[xcol], d[ycol], s=22, color=C_MAIN, alpha=0.45,
               edgecolor='white', linewidth=0.3, zorder=2)
    if len(d) >= 3:
        b, a = np.polyfit(d[xcol], d[ycol], 1)
        xs = np.linspace(d[xcol].min(), d[xcol].max(), 100)
        ax.plot(xs, a + b * xs, color=C_ALT, linewidth=2, zorder=3)
        pr, pp = stats.pearsonr(d[xcol], d[ycol])
        sr, sp = stats.spearmanr(d[xcol], d[ycol])
        ax.text(0.03, 0.97,
                f'Pearson r = {pr:.3f} (p = {pp:.3f})\n'
                f'Spearman $\\rho$ = {sr:.3f} (p = {sp:.3f})\n'
                f'n = {len(d)}',
                transform=ax.transAxes, va='top', ha='left', fontsize=8,
                color=C_MUTED,
                bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                          edgecolor='#e0e0e0', linewidth=0.6))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(color='#ededed', linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


print("Generating scatter figures...")

fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
for ax, (col, label) in zip(axes.ravel(), MAE_PAIRS):
    scatter_panel(ax, pat_mae, col, 'patient_mae', label, 'Per-patient MAE (SAS units)')
fig.suptitle(f'Per-patient error vs patient characteristics '
             f'(FULL_HYBRID, {LSTM_UNITS} units, {N_SEEDS} seeds, '
             f'n = {len(pat_mae)} test patients)',
             fontsize=12, x=0.02, ha='left')
fig.tight_layout(rect=(0, 0, 1, 0.96))
save_figure(fig, 'cohort_mae_scatter')

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
for ax, (col, label) in zip(axes, EC50_PAIRS):
    scatter_panel(ax, pat, col, 'ec50', label, 'Calibrated EC50 ($\\mu$g/mL)')
    ax.axhline(0.5, color=C_MUTED, linewidth=1, linestyle='--', zorder=1)
fig.suptitle(f'Calibrated EC50 vs patient characteristics (n = {n_pat:,} patients; '
             f'the dashed line is the optimiser lower bound, where '
             f'{100*(pat["ec50"] <= 0.5001).mean():.0f}% of patients sit)',
             fontsize=11, x=0.02, ha='left')
fig.tight_layout(rect=(0, 0, 1, 0.94))
save_figure(fig, 'cohort_ec50_scatter')

print()
print("=" * 70)
print("Files generated:")
for f in (TABLE1_FILE, CORR_FILE, MAE_CACHE_FILE):
    print(f"   {f}")
print(f"   {FIG_DIR}/cohort_distributions.png/.pdf")
print(f"   {FIG_DIR}/cohort_sex_comorbidity.png/.pdf")
print(f"   {FIG_DIR}/cohort_mae_scatter.png/.pdf")
print(f"   {FIG_DIR}/cohort_ec50_scatter.png/.pdf")
print()
print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
