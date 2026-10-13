#!/usr/bin/env python3
"""
02b_run_capacity_sweep.py
=========================

This script runs the LSTM capacity sweep experiment using the cached features
built by 01_build_features.py. It tests four LSTM hidden-unit sizes (4, 8, 16,
32) across all five feature configurations from the original ablation, with
ten random seeds per combination, to evaluate whether smaller LSTMs benefit
more from mechanistic features than larger LSTMs overfit on this dataset size.

The motivation comes from the 10-seed ablation run at LSTM_UNITS=16, where
LSTM showed no consistent benefit from any feature group. That result is
consistent with overfitting on ~25k training instances, and the question this
sweep is designed to answer is whether the relationship between LSTM capacity
and feature-augmentation benefit is monotonic. If smaller LSTMs show positive
degradation when mechanistic features are removed while larger ones do not,
that is a real finding about model capacity matching dataset size and gives
the paper a defensible computational discovery.

XGBoost is not rerun here because it does not depend on LSTM capacity, and
its numbers are already on file from the original ablation.

The script saves results incrementally and is resumable. Resume keys are
the tuple (configuration, seed, lstm_units), so a partial run can pick up
from where it left off without redoing completed work.

USAGE:
    python 02b_run_capacity_sweep.py

REQUIREMENTS:
    cached_features.pkl in the current directory (created by 01_build_features.py)

OUTPUT:
    capacity_sweep_results.csv - one row per LSTM training run, with an
    additional lstm_units column relative to ablation_results.csv

EXPECTED RUNTIME:
    Approximately 4-6 hours on a typical laptop CPU. Total runs: 4 sizes x
    5 configurations x 10 seeds = 200 LSTM trainings. The 32-unit runs are
    the slowest, the 4-unit runs are the fastest.

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
import hashlib
import time
warnings.filterwarnings('ignore')

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

# Configuration
CACHE_FILE = 'cached_features.pkl'
ARM_FILE_STEMS = {
    'full': 'arm_a_full',
    'interior': 'arm_b_interior',
    'matched': 'arm_c_matched',
    'interior_clean': 'arm_b_clean',
}
# '--arm all' is the original three-arm design; interior_clean runs only when named
ALL_ARMS = ['full', 'interior', 'matched']

BASE_SEED = 42
SEQUENCE_LENGTH = 10
LSTM_UNITS_LIST = [4, 8, 16, 32]    # Capacity sweep over LSTM hidden-unit sizes
FIXED_EPOCHS = 50                   # Same fixed-epoch setup as the main ablation
DEFAULT_N_SEEDS = 10                # Same seed count as the main ablation
EC50_BOUND = 0.5
EC50_TOL = 1e-3
EC50_POP = 3.08                     # Assigned when the calibration window is empty
# Recorded with every run. Rows without this column were produced before
# 2026-10-12 with np.random.seed + tf.random.set_seed only, which leaves the
# Keras 3 weight initializers unseeded; those runs are not reproducible by seed.
SEEDING = 'keras.utils.set_random_seed'

parser = argparse.ArgumentParser(description='Run the three-arm capacity sweep.')
parser.add_argument('--arm', choices=['all', *ARM_FILE_STEMS], default='all')
parser.add_argument('--config', nargs='+', default=None,
                    help='Configuration(s) to run; defaults to all.')
parser.add_argument('--units', nargs='+', type=int, choices=[4, 8, 16, 32],
                    default=None,
                    help='LSTM unit size(s) to run; defaults to all sizes.')
parser.add_argument('--n-seeds', type=int, default=DEFAULT_N_SEEDS,
                    help='Initializations per cell (seeds 42, 43, ...). Any value other '
                         'than 10 writes to capacity_sweep_*_<n>seed_results.csv.')
parser.add_argument('--log-file', default=None,
                    help='Log file to append to; defaults to the sweep log for --n-seeds.')
ARGS = parser.parse_args()

N_SEEDS = ARGS.n_seeds
RUN_TAG = '' if N_SEEDS == DEFAULT_N_SEEDS else f'_{N_SEEDS}seed'
RESULTS_FILES = {
    arm: f'capacity_sweep_{stem}{RUN_TAG}_results.csv'
    for arm, stem in ARM_FILE_STEMS.items()
}
LOG_FILE = ARGS.log_file or f'three_arm_capacity_sweep{RUN_TAG}.log'


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
print("LSTM CAPACITY SWEEP")
print("=" * 70)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Log file: {LOG_FILE}")
print()

# Verify cache file exists
if not os.path.exists(CACHE_FILE):
    print(f"ERROR: Feature cache file not found: {CACHE_FILE}")
    print("Please run 'python 01_build_features.py' first.")
    sys.exit(1)

# Load cached features
print(f"\nLoading features from {CACHE_FILE}...")
with open(CACHE_FILE, 'rb') as f:
    df = pickle.load(f)
print(f"   {len(df)} prediction instances from {df['stay_id'].nunique()} patients")

# Compute temporal statistics from Ce sequences. These are not used by the LSTM
# directly (LSTM takes the full sequence) but the loader matches the original
# ablation script for parity.
ce_seqs = np.array([np.array(s) for s in df['ce_sequence']])
df['ce_seq_mean'] = ce_seqs.mean(axis=1)
df['ce_seq_std'] = ce_seqs.std(axis=1)
df['ce_seq_trend'] = ce_seqs[:, -1] - ce_seqs[:, 0]
df['ce_seq_max'] = ce_seqs.max(axis=1)
df['ce_seq_last'] = ce_seqs[:, -1]

ec50_by_patient = df.groupby('stay_id')['EC50_calibrated'].first()
pinned_low_ids = set(ec50_by_patient.index[
    (ec50_by_patient - EC50_BOUND).abs() <= EC50_TOL
])
pinned_high_ids = set(ec50_by_patient.index[
    (ec50_by_patient - 10.0).abs() <= EC50_TOL
])
interior_ids = set(ec50_by_patient.index) - pinned_low_ids - pinned_high_ids
default_ids = set(ec50_by_patient.index[ec50_by_patient == EC50_POP])
interior_clean_ids = interior_ids - default_ids
full_ids = set(ec50_by_patient.index)
matched_ids = set(np.random.RandomState(BASE_SEED).choice(
    np.array(sorted(full_ids)), size=len(interior_ids), replace=False
))

print("\nPatient classification from regenerated cache:")
print(f"   pinned-low: {len(pinned_low_ids)}")
print(f"   pinned-high: {len(pinned_high_ids)}")
print(f"   interior: {len(interior_ids)}")
print(f"   interior, assigned the population default {EC50_POP}: "
      f"{len(interior_ids & default_ids)}; fitted: {len(interior_clean_ids)}")
print(f"   arm C size-matched sample: {len(matched_ids)} (sampling seed {BASE_SEED})")
if pinned_high_ids:
    print("WARNING: pinned-high values remain in the regenerated cache")
else:
    print("Confirmed: no pinned-high values remain")
if N_SEEDS == DEFAULT_N_SEEDS:
    pd.DataFrame([
        {'arm': 'A_full', 'n_patients': len(full_ids), 'sampling_seed': np.nan},
        {'arm': 'B_interior', 'n_patients': len(interior_ids), 'sampling_seed': np.nan},
        {'arm': 'C_matched', 'n_patients': len(matched_ids), 'sampling_seed': BASE_SEED},
    ]).to_csv('three_arm_cohort_counts.csv', index=False)

# Feature groups, identical to the main ablation
CLINICAL = ['age', 'gender', 'weight', 'height', 'sas_last', 'sas_mean',
            'sas_std', 'sas_trend', 'propofol_rate', 'time_hours']
STATIC_MECH = ['Ce', 'Cp', 'EC50_calibrated', 'mech_pred_pop', 'mech_pred_calib']
TEMP_DERIVED = ['ce_seq_mean', 'ce_seq_std', 'ce_seq_trend', 'ce_seq_max', 'ce_seq_last']

# Five ablation configurations, identical to the main ablation. The xgb_features
# fields are kept for reference and parity but unused here.
CONFIGS = {
    'FULL_HYBRID': {
        'description': 'All features (reference baseline)',
        'xgb_features': CLINICAL + STATIC_MECH + TEMP_DERIVED,
        'lstm_static': CLINICAL + STATIC_MECH,
        'use_ce_seq': True,
    },
    'NO_TEMPORAL': {
        'description': 'Remove temporal mechanistic features',
        'xgb_features': CLINICAL + STATIC_MECH,
        'lstm_static': CLINICAL + STATIC_MECH,
        'use_ce_seq': False,
    },
    'NO_STATIC_MECH': {
        'description': 'Remove static mechanistic features',
        'xgb_features': CLINICAL + TEMP_DERIVED,
        'lstm_static': CLINICAL,
        'use_ce_seq': True,
    },
    'PURE_ML': {
        'description': 'Remove all mechanistic features',
        'xgb_features': CLINICAL,
        'lstm_static': CLINICAL,
        'use_ce_seq': False,
    },
    'ONLY_MECH': {
        'description': 'Keep only mechanistic features',
        'xgb_features': STATIC_MECH + TEMP_DERIVED,
        'lstm_static': STATIC_MECH,
        'use_ce_seq': True,
    },
}

if ARGS.config:
    unknown_configs = sorted(set(ARGS.config) - set(CONFIGS))
    if unknown_configs:
        parser.error(f"unknown configuration(s): {', '.join(unknown_configs)}")
configurations_to_run = [
    (name, config) for name, config in CONFIGS.items()
    if ARGS.config is None or name in ARGS.config
]


def acquire_results_lock(results_file):
    """Block until this process holds the lock file that serializes access to
    a results CSV; returns the lock file path for os.remove."""
    lock_file = results_file + '.lock'
    while True:
        try:
            os.close(os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            return lock_file
        except FileExistsError:
            time.sleep(0.2)


def read_completed(results_file):
    """Resume keys (configuration, seed, lstm_units) currently in a results CSV."""
    if not os.path.exists(results_file):
        return set()
    lock_file = acquire_results_lock(results_file)
    try:
        existing = pd.read_csv(results_file)
    finally:
        os.remove(lock_file)
    return {
        (row['configuration'], int(row['seed']), int(row['lstm_units']))
        for _, row in existing.iterrows()
    }


# Byte offset of the OS-level lock inside a cell lock file, placed past the
# text so the file stays readable while it is held.
CELL_LOCK_OFFSET = 1024


def cell_lock_path(results_file, config_name, lstm_units):
    return f'{results_file[:-len(".csv")]}.{config_name}.{lstm_units}u.cell.lock'


def acquire_cell_lock(results_file, config_name, lstm_units):
    """Claim one (arm, configuration, units) cell for this process. Returns an
    open handle, or None when another live worker holds the cell.

    The lock file is written before training and removed after, and the claim
    itself is an OS-level lock on the open handle. The operating system drops
    that lock when the process dies, so a lock file left behind by a killed
    worker does not block a restart."""
    path = cell_lock_path(results_file, config_name, lstm_units)
    handle = open(path, 'a+', encoding='utf-8')
    try:
        os.lseek(handle.fileno(), CELL_LOCK_OFFSET, os.SEEK_SET)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.lockf(handle, fcntl.LOCK_EX | fcntl.LOCK_NB, 1, CELL_LOCK_OFFSET)
    except OSError:
        handle.close()
        return None
    handle.truncate(0)
    handle.write(f"pid={os.getpid()} "
                 f"started={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    handle.flush()
    return handle


def release_cell_lock(handle):
    path = handle.name
    handle.close()
    try:
        os.remove(path)
    except OSError:
        pass  # another worker claimed the cell between close and remove


def save_result(result, results_file):
    """Append a single result to the output CSV while preventing accidental
    duplicate rows from repeated resume attempts or interrupted reruns."""
    key_cols = ['arm', 'algorithm', 'configuration', 'seed', 'lstm_units']
    df_row = pd.DataFrame([result])
    # Several processes may write the same file (one per capacity), so the
    # read-append-write below is serialized with a lock file.
    lock_file = acquire_results_lock(results_file)
    try:
        if os.path.exists(results_file):
            existing = pd.read_csv(results_file)
            combined = pd.concat([existing, df_row], ignore_index=True)
            deduped = combined.drop_duplicates(subset=key_cols, keep='last')
            deduped.to_csv(results_file, index=False)
        else:
            df_row.to_csv(results_file, index=False)
    finally:
        os.remove(lock_file)


def weights_digest(weights):
    """SHA-256 over the raw bytes of a list of weight arrays."""
    digest = hashlib.sha256()
    for w in weights:
        digest.update(np.ascontiguousarray(w).tobytes())
    return digest.hexdigest()


# Positions in model.get_weights() whose shape does not depend on the feature
# configuration: LSTM recurrent kernel and bias, Dense(16) bias, Dense(8) kernel
# and bias, output kernel and bias. The LSTM input kernel (0) and the Dense(16)
# kernel (3) change shape with the number of sequence channels and static features.
CONFIG_INVARIANT_WEIGHTS = [1, 2, 4, 5, 6, 7, 8]


def run_lstm(config, seed, lstm_units):
    """Train and evaluate an LSTM with the given configuration, random seed,
    and hidden-unit count. Architecture is identical to 02_run_ablation.py
    except the LSTM hidden size is parameterized."""
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
    init_weights = model.get_weights()

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
        'seeding': SEEDING,
        'init_weights_sha256': weights_digest(init_weights),
        'init_shared_sha256': weights_digest(
            [init_weights[i] for i in CONFIG_INVARIANT_WEIGHTS]),
    }


ARM_PATIENTS = {
    'full': full_ids,
    'interior': interior_ids,
    'matched': matched_ids,
    'interior_clean': interior_clean_ids,
}
# interior_clean keeps the Arm B train/test assignment: the Arm B split is
# reproduced and the default-assigned patients are dropped from each side.
SPLIT_PARENT = {'interior_clean': 'interior'}
ARM_LABELS = {
    'full': 'A: full cohort',
    'interior': 'B: interior patients only',
    'matched': 'C: size-matched random full-cohort sample',
    'interior_clean': 'B clean: interior patients with a fitted EC50 (defaults excluded)',
}
arms_to_run = ALL_ARMS if ARGS.arm == 'all' else [ARGS.arm]
units_to_run = ARGS.units if ARGS.units is not None else LSTM_UNITS_LIST
total_runs = len(arms_to_run) * len(units_to_run) * len(configurations_to_run) * N_SEEDS
print(f"\nConfigurations: {', '.join(name for name, _ in configurations_to_run)}")
print(f"Running {len(arms_to_run)} arm(s), {total_runs} LSTM trainings total")
print()

n_done_now = 0
n_skipped = 0

for arm in arms_to_run:
    results_file = RESULTS_FILES[arm]
    completed_runs = set()
    if os.path.exists(results_file):
        existing = pd.read_csv(results_file)
        if 'seeding' not in existing.columns:
            print(f"ERROR: {results_file} was produced with the earlier, unseeded "
                  "initialization and cannot be resumed with the current seeding. "
                  "Use --n-seeds to write a new file, or move the old file aside.")
            sys.exit(1)
        completed_runs = {
            (row['configuration'], int(row['seed']), int(row['lstm_units']))
            for _, row in existing.iterrows()
        }
        print(f"Found {len(completed_runs)} completed runs in {results_file}; resuming")

    split_arm = SPLIT_PARENT.get(arm, arm)
    arm_df = df[df['stay_id'].isin(ARM_PATIENTS[split_arm])].reset_index(drop=True)
    unique_patients = arm_df['stay_id'].unique()
    train_patients, test_patients = train_test_split(
        unique_patients, test_size=0.2, random_state=BASE_SEED
    )
    if split_arm != arm:
        train_patients = [p for p in train_patients if p in ARM_PATIENTS[arm]]
        test_patients = [p for p in test_patients if p in ARM_PATIENTS[arm]]
        arm_df = arm_df[arm_df['stay_id'].isin(ARM_PATIENTS[arm])].reset_index(drop=True)
        unique_patients = arm_df['stay_id'].unique()
    train_df = arm_df[arm_df['stay_id'].isin(train_patients)].reset_index(drop=True)
    test_df = arm_df[arm_df['stay_id'].isin(test_patients)].reset_index(drop=True)
    print(f"{'=' * 70}\nARM {ARM_LABELS[arm]}\n{'=' * 70}")
    print(f"   Patients: {len(unique_patients)}; train {len(train_patients)} "
          f"({len(train_df)} instances), test {len(test_patients)} "
          f"({len(test_df)} instances)")
    print(f"   Results: {results_file}")

    for lstm_units in units_to_run:
        print(f"LSTM_UNITS = {lstm_units}")
        for config_name, config in configurations_to_run:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] units={lstm_units} "
                  f"config={config_name}")
            # One worker per cell: the completed set is only a snapshot, so two
            # workers on the same cell would both train it.
            cell_lock = acquire_cell_lock(results_file, config_name, lstm_units)
            if cell_lock is None:
                print(f"ERROR: cell arm={arm} config={config_name} "
                      f"units={lstm_units} is held by another worker "
                      f"({cell_lock_path(results_file, config_name, lstm_units)}); "
                      "exiting instead of racing it")
                sys.exit(1)
            try:
                # Re-read under the lock: another worker may have finished
                # this cell since the arm-level read above.
                completed_runs = read_completed(results_file)
                for seed_offset in range(N_SEEDS):
                    seed = BASE_SEED + seed_offset
                    if (config_name, seed, lstm_units) not in completed_runs:
                        result = run_lstm(config, seed, lstm_units)
                        save_result({
                            'arm': arm,
                            'algorithm': 'LSTM',
                            'configuration': config_name,
                            'seed': seed,
                            'lstm_units': lstm_units,
                            'n_patients': len(unique_patients),
                            'n_train_patients': len(train_patients),
                            'n_test_patients': len(test_patients),
                            'n_train_instances': len(train_df),
                            'n_test_instances': len(test_df),
                            **result,
                        }, results_file)
                        print(f"   seed={seed}: MAE={result['mae']:.4f}, "
                              f"R²={result['r2']:.4f}")
                        n_done_now += 1
                    else:
                        n_skipped += 1
                        print(f"   seed={seed}: already done, skipping")
            finally:
                release_cell_lock(cell_lock)

print(f"\nCompleted {n_done_now} new runs, skipped {n_skipped} previously completed")
print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
