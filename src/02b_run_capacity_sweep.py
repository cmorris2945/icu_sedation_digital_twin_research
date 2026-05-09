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
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Suppress TensorFlow's verbose logging output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, concatenate
from tensorflow.keras.optimizers import Adam

# Configuration
CACHE_FILE = 'cached_features.pkl'
RESULTS_FILE = 'capacity_sweep_results.csv'

BASE_SEED = 42
SEQUENCE_LENGTH = 10
LSTM_UNITS_LIST = [4, 8, 16, 32]    # Capacity sweep over LSTM hidden-unit sizes
FIXED_EPOCHS = 50                   # Same fixed-epoch setup as the main ablation
N_SEEDS = 10                        # Same seed count as the main ablation

print("=" * 70)
print("LSTM CAPACITY SWEEP")
print("=" * 70)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Verify cache file exists
if not os.path.exists(CACHE_FILE):
    print(f"ERROR: Feature cache file not found: {CACHE_FILE}")
    print("Please run 'python 01_build_features.py' first.")
    sys.exit(1)

# Resumability: track which (configuration, seed, lstm_units) tuples already exist
completed_runs = set()
if os.path.exists(RESULTS_FILE):
    existing = pd.read_csv(RESULTS_FILE)
    for _, row in existing.iterrows():
        completed_runs.add((row['configuration'], int(row['seed']), int(row['lstm_units'])))
    print(f"Found {len(completed_runs)} previously completed runs in {RESULTS_FILE}")
    print("Will skip these and only run remaining experiments")
else:
    print("No previous results found, starting from scratch")

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

# Same patient-level train/test split as the original ablation, using the same
# random_state, so capacity sweep results are directly comparable.
unique_patients = df['stay_id'].unique()
train_patients, test_patients = train_test_split(
    unique_patients, test_size=0.2, random_state=BASE_SEED
)
train_df = df[df['stay_id'].isin(train_patients)].reset_index(drop=True)
test_df = df[df['stay_id'].isin(test_patients)].reset_index(drop=True)
print(f"   Train: {len(train_patients)} patients, {len(train_df)} observations")
print(f"   Test:  {len(test_patients)} patients, {len(test_df)} observations")

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


def save_result(result):
    """Append a single result to the output CSV. Incremental write so partial
    progress is preserved if the script is interrupted."""
    df_row = pd.DataFrame([result])
    if os.path.exists(RESULTS_FILE):
        df_row.to_csv(RESULTS_FILE, mode='a', header=False, index=False)
    else:
        df_row.to_csv(RESULTS_FILE, index=False)


def run_lstm(config, seed, lstm_units):
    """Train and evaluate an LSTM with the given configuration, random seed,
    and hidden-unit count. Architecture is identical to 02_run_ablation.py
    except the LSTM hidden size is parameterized."""
    np.random.seed(seed)
    tf.random.set_seed(seed)

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


total_runs = len(LSTM_UNITS_LIST) * len(CONFIGS) * N_SEEDS
print(f"\nRunning experiments...")
print(f"Total expected: {len(LSTM_UNITS_LIST)} sizes x {len(CONFIGS)} configs x "
      f"{N_SEEDS} seeds = {total_runs} runs")
print()

n_done_now = 0
n_skipped = 0

for lstm_units in LSTM_UNITS_LIST:
    print(f"{'=' * 70}")
    print(f"LSTM_UNITS = {lstm_units}")
    print(f"{'=' * 70}")

    for config_name, config in CONFIGS.items():
        print(f"[{datetime.now().strftime('%H:%M:%S')}] units={lstm_units} "
              f"config={config_name}")
        print(f"   {config['description']}")

        for seed_offset in range(N_SEEDS):
            seed = BASE_SEED + seed_offset
            if (config_name, seed, lstm_units) not in completed_runs:
                result = run_lstm(config, seed, lstm_units)
                save_result({
                    'algorithm': 'LSTM',
                    'configuration': config_name,
                    'seed': seed,
                    'lstm_units': lstm_units,
                    **result,
                })
                print(f"   units={lstm_units} seed={seed}: "
                      f"MAE={result['mae']:.4f}, R²={result['r2']:.4f}, "
                      f"train_loss={result['final_train_loss']:.4f}, "
                      f"val_loss={result['final_val_loss']:.4f}")
                n_done_now += 1
            else:
                n_skipped += 1
                print(f"   units={lstm_units} seed={seed}: already done, skipping")

        print()

print(f"Completed {n_done_now} new runs, skipped {n_skipped} previously completed")
print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\nNext step: run 'python 03b_analyze_capacity_sweep.py' to summarize")
