#!/usr/bin/env python3
"""
02c_run_arm_b_clean_4unit.py
============================

Reruns the Arm B 4-unit FULL_HYBRID versus PURE_ML comparison on the 670
Arm B patients whose EC50 was actually fitted, excluding the 45 patients
assigned the population default 3.08 because their calibration window had no
usable observations.

The train/test assignment is the existing Arm B assignment. The 715-patient
Arm B split from 02b_run_capacity_sweep.py is reproduced exactly and the
default-assigned patients are then dropped from each side; nothing is re-split.

Two cohorts are trained in this script:
    clean   the 670 fitted patients (the requested experiment)
    rerun   all 715 Arm B patients again, in this environment (control)

The control exists because the stored Arm B runs were produced in a different
Python environment and per-seed results are not bitwise reproducible. Without
it, a difference between the stored 715-patient result and the new 670-patient
result could not be attributed to the cohort. Pass --skip-control to omit it.

--repeat-control LABEL retrains the 715-patient cohort again in a fresh
process and stores it under LABEL, so that two runs of the same seed in the
same environment can be compared. --config and --seeds restrict what is run.

The model, hyperparameters, seeds and run order are those of
02b_run_capacity_sweep.py. The existing Arm B results file is read, never
written.

The runs made on 2026-10-12 and stored in arm_b_clean_4unit_runs.csv used
np.random.seed + tf.random.set_seed only and are not reproducible by seed.
Those runs, their comparison table and their log are kept in
results/superseded/pre_seeding_fix/, not in the output paths below.
Seeding was changed to keras.utils.set_random_seed afterwards; the seeded
30-initialization runs are produced by 02b_run_capacity_sweep.py --n-seeds 30
(--arm interior_clean for the 670-patient cohort).

USAGE:
    python src/02c_run_arm_b_clean_4unit.py

REQUIREMENTS:
    cached_features.pkl and capacity_sweep_arm_b_interior_results.csv in the
    current directory

OUTPUT:
    results/processed/arm_b_clean_4unit_runs.csv        one row per training run
    results/processed/arm_b_clean_4unit_comparison.csv  side-by-side summary
    arm_b_clean_4unit.log

AUTHOR: Christopher Morris
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import pickle
import os
import sys
import argparse
import platform
warnings.filterwarnings('ignore')

import scipy
import sklearn
from scipy.stats import wilcoxon
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Suppress TensorFlow's verbose logging output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, concatenate
from tensorflow.keras.optimizers import Adam

# Configuration, identical to 02b_run_capacity_sweep.py
CACHE_FILE = 'cached_features.pkl'
EXISTING_ARM_B_FILE = 'capacity_sweep_arm_b_interior_results.csv'
RUNS_FILE = os.path.join('results', 'processed', 'arm_b_clean_4unit_runs.csv')
COMPARISON_FILE = os.path.join('results', 'processed', 'arm_b_clean_4unit_comparison.csv')
LOG_FILE = 'arm_b_clean_4unit.log'

BASE_SEED = 42
SEQUENCE_LENGTH = 10
LSTM_UNITS = 4
FIXED_EPOCHS = 50
N_SEEDS = 10
EC50_BOUND = 0.5
EC50_TOL = 1e-3
EC50_POP = 3.08

parser = argparse.ArgumentParser(description='Rerun Arm B at 4 units without the default-assigned patients.')
parser.add_argument('--skip-control', action='store_true',
                    help='Do not retrain the 715-patient Arm B in this environment.')
parser.add_argument('--repeat-control', metavar='LABEL', default=None,
                    help='Retrain only the 715-patient Arm B and record it under LABEL, '
                         'leaving earlier runs in place. Used to test run-to-run determinism.')
parser.add_argument('--config', nargs='+', default=None,
                    help='Configuration(s) to run; defaults to both.')
parser.add_argument('--seeds', nargs='+', type=int, default=None,
                    help='Seed(s) to run; defaults to 42-51.')
ARGS = parser.parse_args()


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


log_stream = open(LOG_FILE, 'a', encoding='utf-8', buffering=1)
sys.stdout = Tee(sys.stdout, log_stream)
sys.stderr = Tee(sys.stderr, log_stream)

print("=" * 70)
print("ARM B 4-UNIT RERUN WITHOUT DEFAULT-ASSIGNED PATIENTS")
print("=" * 70)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Log file: {LOG_FILE}")
print(f"Python {platform.python_version()}, pandas {pd.__version__}, "
      f"NumPy {np.__version__}, SciPy {scipy.__version__}, "
      f"scikit-learn {sklearn.__version__}, TensorFlow {tf.__version__}")

for required in (CACHE_FILE, EXISTING_ARM_B_FILE):
    if not os.path.exists(required):
        print(f"ERROR: required file not found: {required}")
        sys.exit(1)

print(f"\nLoading features from {CACHE_FILE}...")
with open(CACHE_FILE, 'rb') as f:
    df = pickle.load(f)
print(f"   {len(df)} prediction instances from {df['stay_id'].nunique()} patients")

# Arm B classification, identical to 02b_run_capacity_sweep.py
ec50_by_patient = df.groupby('stay_id')['EC50_calibrated'].first()
pinned_low_ids = set(ec50_by_patient.index[
    (ec50_by_patient - EC50_BOUND).abs() <= EC50_TOL
])
pinned_high_ids = set(ec50_by_patient.index[
    (ec50_by_patient - 10.0).abs() <= EC50_TOL
])
interior_ids = set(ec50_by_patient.index) - pinned_low_ids - pinned_high_ids
default_ids = set(ec50_by_patient.index[ec50_by_patient == EC50_POP])

# Existing Arm B split, reproduced exactly; not re-split below
arm_df = df[df['stay_id'].isin(interior_ids)].reset_index(drop=True)
unique_patients = arm_df['stay_id'].unique()
train_patients, test_patients = train_test_split(
    unique_patients, test_size=0.2, random_state=BASE_SEED
)

assert len(pinned_high_ids) == 0, "pinned-high values present; cache is not the post-fix cache"
assert len(interior_ids) == 715, f"expected 715 Arm B patients, found {len(interior_ids)}"
assert (len(train_patients), len(test_patients)) == (572, 143), "Arm B split does not match 572/143"
assert default_ids <= interior_ids, "a default-assigned patient is outside Arm B"

COHORTS = {}
for cohort, drop_ids in (('clean', default_ids), ('rerun', set())):
    keep_train = [p for p in train_patients if p not in drop_ids]
    keep_test = [p for p in test_patients if p not in drop_ids]
    COHORTS[cohort] = {
        'train_df': arm_df[arm_df['stay_id'].isin(keep_train)].reset_index(drop=True),
        'test_df': arm_df[arm_df['stay_id'].isin(keep_test)].reset_index(drop=True),
        'n_train_patients': len(keep_train),
        'n_test_patients': len(keep_test),
    }
COHORT_LABELS = {
    'clean': 'Arm B clean: 670 fitted patients, defaults excluded',
    'rerun': 'Arm B control: all 715 patients retrained in this environment',
}

print("\nArm B composition:")
print(f"   Arm B patients: {len(interior_ids)} (train {len(train_patients)}, test {len(test_patients)})")
print(f"   assigned the population default {EC50_POP}: {len(default_ids)} "
      f"(train {len(set(train_patients) & default_ids)}, test {len(set(test_patients) & default_ids)})")
for cohort, c in COHORTS.items():
    print(f"   {cohort}: train {c['n_train_patients']} patients ({len(c['train_df'])} instances), "
          f"test {c['n_test_patients']} patients ({len(c['test_df'])} instances)")

# Feature groups and configurations, identical to 02b_run_capacity_sweep.py
CLINICAL = ['age', 'gender', 'weight', 'height', 'sas_last', 'sas_mean',
            'sas_std', 'sas_trend', 'propofol_rate', 'time_hours']
STATIC_MECH = ['Ce', 'Cp', 'EC50_calibrated', 'mech_pred_pop', 'mech_pred_calib']

CONFIGS = {
    'FULL_HYBRID': {
        'lstm_static': CLINICAL + STATIC_MECH,
        'use_ce_seq': True,
    },
    'PURE_ML': {
        'lstm_static': CLINICAL,
        'use_ce_seq': False,
    },
}


def save_result(result):
    """Append a single result to the runs CSV while preventing duplicate rows
    from repeated resume attempts or interrupted reruns."""
    key_cols = ['cohort', 'algorithm', 'configuration', 'seed', 'lstm_units']
    df_row = pd.DataFrame([result])
    if os.path.exists(RUNS_FILE):
        existing = pd.read_csv(RUNS_FILE)
        combined = pd.concat([existing, df_row], ignore_index=True)
        deduped = combined.drop_duplicates(subset=key_cols, keep='last')
        deduped.to_csv(RUNS_FILE, index=False)
    else:
        df_row.to_csv(RUNS_FILE, index=False)


def run_lstm(config, seed, lstm_units, train_df, test_df):
    """Train and evaluate an LSTM. Body identical to run_lstm in
    02b_run_capacity_sweep.py except that the frames are passed in."""
    # Seeds Python's random, NumPy and TensorFlow. Keras 3 draws its initializer
    # and dropout seeds from Python's random, so seeding NumPy and TensorFlow
    # alone does not fix the initial weights.
    keras.utils.set_random_seed(seed)

    static_feats = config['lstm_static']
    use_ce_seq = config['use_ce_seq']

    sas_train = np.array([np.array(s) for s in train_df['sas_sequence']])
    sas_test = np.array([np.array(s) for s in test_df['sas_sequence']])

    if use_ce_seq:
        ce_train = np.array([np.array(s) for s in train_df['ce_sequence']])
        ce_test = np.array([np.array(s) for s in test_df['ce_sequence']])
        seq_train = np.stack([sas_train, ce_train], axis=-1)
        seq_test = np.stack([sas_test, ce_test], axis=-1)
    else:
        seq_train = sas_train.reshape(-1, SEQUENCE_LENGTH, 1)
        seq_test = sas_test.reshape(-1, SEQUENCE_LENGTH, 1)

    static_train = train_df[static_feats].values
    static_test = test_df[static_feats].values
    y_train = train_df['actual_sas'].values
    y_test = test_df['actual_sas'].values

    ss_static = StandardScaler()
    static_train_s = ss_static.fit_transform(static_train)
    static_test_s = ss_static.transform(static_test)

    ss_seq = StandardScaler()
    seq_train_flat = seq_train.reshape(-1, seq_train.shape[-1])
    seq_test_flat = seq_test.reshape(-1, seq_test.shape[-1])
    seq_train_s = ss_seq.fit_transform(seq_train_flat).reshape(seq_train.shape)
    seq_test_s = ss_seq.transform(seq_test_flat).reshape(seq_test.shape)

    n_static = static_train.shape[1]
    n_seq = seq_train.shape[-1]

    seq_input = Input(shape=(SEQUENCE_LENGTH, n_seq), name='seq')
    lstm_out = LSTM(lstm_units)(seq_input)
    lstm_out = Dropout(0.2)(lstm_out)

    static_input = Input(shape=(n_static,), name='static')

    combined = concatenate([lstm_out, static_input])
    x = Dense(16, activation='relu')(combined)
    x = Dropout(0.2)(x)
    x = Dense(8, activation='relu')(x)
    output = Dense(1)(x)

    model = Model(inputs=[seq_input, static_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

    history = model.fit(
        [seq_train_s, static_train_s], y_train,
        epochs=FIXED_EPOCHS, batch_size=64,
        validation_split=0.15,
        verbose=0
    )

    y_pred = model.predict([seq_test_s, static_test_s], verbose=0).flatten()

    return {
        'mae': float(mean_absolute_error(y_test, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
        'r2': float(r2_score(y_test, y_pred)),
        'epochs': FIXED_EPOCHS,
        'final_train_loss': float(history.history['loss'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1]),
    }


if ARGS.repeat_control:
    assert ARGS.repeat_control not in COHORTS, "label is reserved"
    COHORTS[ARGS.repeat_control] = COHORTS['rerun']
    COHORT_LABELS[ARGS.repeat_control] = (
        f"Arm B control repeat '{ARGS.repeat_control}': all 715 patients")
    cohorts_to_run = [ARGS.repeat_control]
else:
    cohorts_to_run = ['clean'] if ARGS.skip_control else ['clean', 'rerun']
configs_to_run = [(name, config) for name, config in CONFIGS.items()
                  if ARGS.config is None or name in ARGS.config]
seeds_to_run = ARGS.seeds if ARGS.seeds is not None else [
    BASE_SEED + offset for offset in range(N_SEEDS)]
os.makedirs(os.path.dirname(RUNS_FILE), exist_ok=True)

completed_runs = set()
if os.path.exists(RUNS_FILE):
    existing = pd.read_csv(RUNS_FILE)
    completed_runs = {
        (row['cohort'], row['configuration'], int(row['seed']))
        for _, row in existing.iterrows()
    }
    print(f"\nFound {len(completed_runs)} completed runs in {RUNS_FILE}; resuming")

n_done_now = 0
n_skipped = 0

for cohort in cohorts_to_run:
    c = COHORTS[cohort]
    print(f"{'=' * 70}\n{COHORT_LABELS[cohort]}\n{'=' * 70}")
    print(f"LSTM_UNITS = {LSTM_UNITS}")
    for config_name, config in configs_to_run:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] units={LSTM_UNITS} "
              f"config={config_name}")
        for seed in seeds_to_run:
            if (cohort, config_name, seed) not in completed_runs:
                result = run_lstm(config, seed, LSTM_UNITS, c['train_df'], c['test_df'])
                save_result({
                    'cohort': cohort,
                    'algorithm': 'LSTM',
                    'configuration': config_name,
                    'seed': seed,
                    'lstm_units': LSTM_UNITS,
                    'n_patients': c['n_train_patients'] + c['n_test_patients'],
                    'n_train_patients': c['n_train_patients'],
                    'n_test_patients': c['n_test_patients'],
                    'n_train_instances': len(c['train_df']),
                    'n_test_instances': len(c['test_df']),
                    **result,
                })
                print(f"   seed={seed}: MAE={result['mae']:.4f}, "
                      f"R²={result['r2']:.4f}")
                n_done_now += 1
            else:
                n_skipped += 1
                print(f"   seed={seed}: already done, skipping")

print(f"\nCompleted {n_done_now} new runs, skipped {n_skipped} previously completed")


def summarise(runs, label, source, counts):
    """One comparison row from the ten FULL_HYBRID and ten PURE_ML runs."""
    full = runs[runs['configuration'] == 'FULL_HYBRID']
    pure = runs[runs['configuration'] == 'PURE_ML']
    paired = full.merge(pure, on='seed', how='inner', suffixes=('_full', '_pure'))
    assert len(paired) == N_SEEDS and len(full) == N_SEEDS and len(pure) == N_SEEDS, \
        f"{label}: expected {N_SEEDS} matched runs, found {len(paired)}"
    full_mean = float(paired['mae_full'].mean())
    pure_mean = float(paired['mae_pure'].mean())
    diffs = paired['mae_full'] - paired['mae_pure']
    return {
        'cohort': label,
        'source': source,
        **counts,
        'lstm_units': LSTM_UNITS,
        'n_paired_runs': len(paired),
        'full_hybrid_mae_mean': full_mean,
        'full_hybrid_mae_sd_across_runs': float(paired['mae_full'].std(ddof=1)),
        'full_hybrid_r2_mean': float(paired['r2_full'].mean()),
        'pure_ml_mae_mean': pure_mean,
        'pure_ml_mae_sd_across_runs': float(paired['mae_pure'].std(ddof=1)),
        'pure_ml_r2_mean': float(paired['r2_pure'].mean()),
        'absolute_diff_pure_minus_full': pure_mean - full_mean,
        # Positive means removing the mechanistic features made MAE worse
        'pct_diff_pure_vs_full': (pure_mean - full_mean) / full_mean * 100,
        'n_runs_full_hybrid_better': int((diffs < 0).sum()),
        'wilcoxon_p': float(wilcoxon(diffs.to_numpy(), zero_method='wilcox',
                                     alternative='two-sided').pvalue),
    }


def cohort_counts(c):
    return {
        'n_patients': c['n_train_patients'] + c['n_test_patients'],
        'n_train_patients': c['n_train_patients'],
        'n_test_patients': c['n_test_patients'],
        'n_train_instances': len(c['train_df']),
        'n_test_instances': len(c['test_df']),
    }


stored = pd.read_csv(EXISTING_ARM_B_FILE)
stored = stored[(stored['lstm_units'] == LSTM_UNITS)
                & stored['configuration'].isin(['FULL_HYBRID', 'PURE_ML'])]
new_runs = pd.read_csv(RUNS_FILE)

rows = [summarise(stored, 'arm_b_715_stored', EXISTING_ARM_B_FILE,
                  cohort_counts(COHORTS['rerun']))]
# Every complete retraining of the 715-patient cohort; partial ones are skipped
for cohort in new_runs['cohort'].unique():
    if cohort == 'clean' or (new_runs['cohort'] == cohort).sum() != 2 * N_SEEDS:
        continue
    label = ('arm_b_715_rerun_this_environment' if cohort == 'rerun'
             else f'arm_b_715_{cohort}_this_environment')
    rows.append(summarise(new_runs[new_runs['cohort'] == cohort], label, RUNS_FILE,
                          cohort_counts(COHORTS['rerun'])))
rows.append(summarise(new_runs[new_runs['cohort'] == 'clean'],
                      'arm_b_670_clean', RUNS_FILE,
                      cohort_counts(COHORTS['clean'])))

comparison = pd.DataFrame(rows)
comparison.to_csv(COMPARISON_FILE, index=False)

print(f"\nWrote {COMPARISON_FILE}")
with pd.option_context('display.width', 200, 'display.max_columns', None,
                       'display.float_format', '{:.4f}'.format):
    print(comparison.drop(columns=['source']).T)

print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
