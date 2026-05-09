#!/usr/bin/env python3
"""
run_ablation.py
===============

This script runs the full ablation study using the cached features built
by build_features.py. It tests five different feature configurations
against both XGBoost and LSTM, with multiple random seeds for the LSTM
to assess result stability.

The five configurations are designed to isolate the contribution of
different feature groups. The FULL_HYBRID configuration uses everything
and serves as our reference point. The NO_TEMPORAL configuration removes
the temporal mechanistic features to test how much the time-series of
drug concentration matters. The NO_STATIC_MECH configuration removes
the static mechanistic features like calibrated EC50 to test how much
patient-specific calibration matters. The PURE_ML configuration removes
all mechanistic features to establish a baseline. The ONLY_MECH
configuration removes clinical features to test whether mechanistic
features alone are sufficient.

The script saves results incrementally after every single training run,
so if anything goes wrong we still have whatever data was collected up
to that point. The script is also resumable. If you run it after a
partial completion, it will detect what has already been done and only
run the remaining experiments.

USAGE:
    python run_ablation.py

REQUIREMENTS:
    cached_features.pkl in the current directory (created by build_features.py)

OUTPUT:
    ablation_results.csv - one row per training run with full results

EXPECTED RUNTIME:
    Approximately 30-45 minutes on a typical laptop CPU.
    Faster if you have GPU support enabled.

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

import xgboost as xgb

# Configuration
CACHE_FILE = 'cached_features.pkl'
RESULTS_FILE = 'ablation_results.csv'

BASE_SEED = 42
SEQUENCE_LENGTH = 10
LSTM_UNITS = 16
FIXED_EPOCHS = 50           # No early stopping for stability
N_SEEDS = 10                # Multiple seeds for LSTM stability assessment

print("=" * 70)
print("ABLATION STUDY")
print("=" * 70)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Verify cache file exists
if not os.path.exists(CACHE_FILE):
    print(f"ERROR: Feature cache file not found: {CACHE_FILE}")
    print("Please run 'python build_features.py' first.")
    sys.exit(1)

# Check what has already been completed for resumability
completed_runs = set()
if os.path.exists(RESULTS_FILE):
    existing = pd.read_csv(RESULTS_FILE)
    for _, row in existing.iterrows():
        completed_runs.add((row['algorithm'], row['configuration'], row['seed']))
    print(f"Found {len(completed_runs)} previously completed runs in {RESULTS_FILE}")
    print("Will skip these and only run remaining experiments")
else:
    print("No previous results found, starting from scratch")

# Load cached features
print(f"\nLoading features from {CACHE_FILE}...")
with open(CACHE_FILE, 'rb') as f:
    df = pickle.load(f)
print(f"   {len(df)} prediction instances from {df['stay_id'].nunique()} patients")

# Compute temporal statistics from Ce sequences for use as XGBoost features.
# These let tree-based models access summarized temporal information without
# being able to consume the actual sequence input that LSTM uses.
ce_seqs = np.array([np.array(s) for s in df['ce_sequence']])
df['ce_seq_mean'] = ce_seqs.mean(axis=1)
df['ce_seq_std'] = ce_seqs.std(axis=1)
df['ce_seq_trend'] = ce_seqs[:, -1] - ce_seqs[:, 0]
df['ce_seq_max'] = ce_seqs.max(axis=1)
df['ce_seq_last'] = ce_seqs[:, -1]

# Patient-level train/test split. Same patient never appears in both sets,
# which prevents data leakage and gives a more honest estimate of how the
# model would generalize to new patients.
unique_patients = df['stay_id'].unique()
train_patients, test_patients = train_test_split(
    unique_patients, test_size=0.2, random_state=BASE_SEED
)
train_df = df[df['stay_id'].isin(train_patients)].reset_index(drop=True)
test_df = df[df['stay_id'].isin(test_patients)].reset_index(drop=True)
print(f"   Train: {len(train_patients)} patients, {len(train_df)} observations")
print(f"   Test:  {len(test_patients)} patients, {len(test_df)} observations")

# Define feature groups
CLINICAL = ['age', 'gender', 'weight', 'height', 'sas_last', 'sas_mean',
            'sas_std', 'sas_trend', 'propofol_rate', 'time_hours']
STATIC_MECH = ['Ce', 'Cp', 'EC50_calibrated', 'mech_pred_pop', 'mech_pred_calib']
TEMP_DERIVED = ['ce_seq_mean', 'ce_seq_std', 'ce_seq_trend', 'ce_seq_max', 'ce_seq_last']

# Define the five ablation configurations
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
    """
    Append a single result to the output CSV file. The incremental saving
    means that if the script is interrupted we still have whatever data was
    collected up to that point.
    """
    df_row = pd.DataFrame([result])
    if os.path.exists(RESULTS_FILE):
        df_row.to_csv(RESULTS_FILE, mode='a', header=False, index=False)
    else:
        df_row.to_csv(RESULTS_FILE, index=False)


def run_xgb(config):
    """
    Train and evaluate XGBoost with the given feature configuration. XGBoost
    is deterministic given a random seed, so a single run is sufficient.
    The model takes only static features as input.
    """
    feats = config['xgb_features']
    X_train = train_df[feats].values
    X_test = test_df[feats].values
    y_train = train_df['actual_sas'].values
    y_test = test_df['actual_sas'].values
    
    model = xgb.XGBRegressor(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        random_state=BASE_SEED, verbosity=0
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return {
        'mae': float(mean_absolute_error(y_test, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
        'r2': float(r2_score(y_test, y_pred)),
    }


def run_lstm(config, seed):
    """
    Train and evaluate LSTM with the given configuration and random seed.
    Different seeds will produce different results because of the random
    weight initialization. We run multiple seeds to assess this variance.
    
    The LSTM has two input branches that merge before the final dense
    layers. The sequence input takes SAS history and optionally Ce history.
    The static input takes whatever clinical or mechanistic features the
    configuration calls for.
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    static_feats = config['lstm_static']
    use_ce_seq = config['use_ce_seq']
    
    # Build sequence inputs. We always include SAS history. We include Ce
    # history only when the configuration calls for temporal features.
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
    
    # Static features and target values
    static_train = train_df[static_feats].values
    static_test = test_df[static_feats].values
    y_train = train_df['actual_sas'].values
    y_test = test_df['actual_sas'].values
    
    # Scale features. Static and sequence inputs are scaled separately because
    # they have different ranges and meanings.
    ss_static = StandardScaler()
    static_train_s = ss_static.fit_transform(static_train)
    static_test_s = ss_static.transform(static_test)
    
    ss_seq = StandardScaler()
    seq_train_flat = seq_train.reshape(-1, seq_train.shape[-1])
    seq_test_flat = seq_test.reshape(-1, seq_test.shape[-1])
    seq_train_s = ss_seq.fit_transform(seq_train_flat).reshape(seq_train.shape)
    seq_test_s = ss_seq.transform(seq_test_flat).reshape(seq_test.shape)
    
    # Build the dual-input LSTM model
    n_static = static_train.shape[1]
    n_seq = seq_train.shape[-1]
    
    seq_input = Input(shape=(SEQUENCE_LENGTH, n_seq), name='seq')
    lstm_out = LSTM(LSTM_UNITS)(seq_input)
    lstm_out = Dropout(0.2)(lstm_out)
    
    static_input = Input(shape=(n_static,), name='static')
    
    combined = concatenate([lstm_out, static_input])
    x = Dense(16, activation='relu')(combined)
    x = Dropout(0.2)(x)
    x = Dense(8, activation='relu')(x)
    output = Dense(1)(x)
    
    model = Model(inputs=[seq_input, static_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    
    # Train for fixed number of epochs without early stopping, which we
    # found gives more stable results across random seeds.
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


# Run all experiments, skipping any that have already been completed
print(f"\nRunning experiments...")
print(f"Total expected: {len(CONFIGS)} configs x ({1} XGBoost + {N_SEEDS} LSTM seeds) "
      f"= {len(CONFIGS) * (1 + N_SEEDS)} runs")
print()

n_done_now = 0
n_skipped = 0

for config_name, config in CONFIGS.items():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Configuration: {config_name}")
    print(f"   {config['description']}")
    
    # XGBoost run
    if ('XGBoost', config_name, BASE_SEED) not in completed_runs:
        result = run_xgb(config)
        save_result({
            'algorithm': 'XGBoost',
            'configuration': config_name,
            'seed': BASE_SEED,
            **result,
            'epochs': 0,
            'final_train_loss': 0,
            'final_val_loss': 0,
        })
        print(f"   XGBoost: MAE={result['mae']:.4f}, R²={result['r2']:.4f}")
        n_done_now += 1
    else:
        n_skipped += 1
        print(f"   XGBoost: already done, skipping")
    
    # LSTM runs with multiple seeds
    for seed_offset in range(N_SEEDS):
        seed = BASE_SEED + seed_offset
        if ('LSTM', config_name, seed) not in completed_runs:
            result = run_lstm(config, seed)
            save_result({
                'algorithm': 'LSTM',
                'configuration': config_name,
                'seed': seed,
                **result,
            })
            print(f"   LSTM seed {seed}: MAE={result['mae']:.4f}, "
                  f"R²={result['r2']:.4f}, "
                  f"train_loss={result['final_train_loss']:.4f}, "
                  f"val_loss={result['final_val_loss']:.4f}")
            n_done_now += 1
        else:
            n_skipped += 1
            print(f"   LSTM seed {seed}: already done, skipping")
    
    print()

print(f"Completed {n_done_now} new runs, skipped {n_skipped} previously completed")
print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\nNext step: run 'python analyze_results.py' to generate summary statistics")
