#!/usr/bin/env python3
"""
Compute the clinical-practice baseline comparison using the same test split and
identical prediction instances as the capacity sweep.

This script writes a single machine-readable CSV to
results/processed/baseline_comparison.csv with the rows requested in the
project brief:

- Persistence (SAS stays at sas_last)
- Eleveld, population EC50
- Eleveld, personalized EC50
- XGBoost
- LSTM PURE_ML, 4 units
- LSTM FULL_HYBRID, 4 units
- E0 re-anchored Eleveld sensitivity rows (kept separate from primary rows)
- Eleveld variants with E0/Emax calibrated on training patients only

All methods are evaluated on the exact same hold-out patient set as the
published full-cohort capacity sweep (20% of unique patients, random_state=42).
"""

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Suppress TensorFlow noise.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

tf.get_logger().setLevel('ERROR')
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

ROOT = Path(__file__).resolve().parent.parent
CACHE_FILE = ROOT / 'cached_features.pkl'
RESULTS_FILE = ROOT / 'results' / 'processed' / 'baseline_comparison.csv'
CALIBRATION_FILE = ROOT / 'results' / 'processed' / 'baseline_scale_calibration.csv'
CAPACITY_SWEEP_FILE = ROOT / 'capacity_sweep_arm_a_full_results.csv'

BASE_SEED = 42
SEQUENCE_LENGTH = 10
LSTM_UNITS = 4
FIXED_EPOCHS = 50

CLINICAL = ['age', 'gender', 'weight', 'height', 'sas_last', 'sas_mean',
            'sas_std', 'sas_trend', 'propofol_rate', 'time_hours']
STATIC_MECH = ['Ce', 'Cp', 'EC50_calibrated', 'mech_pred_pop', 'mech_pred_calib']


def sigmoid_emax(ce, ec50, gamma=1.47, e0=7.0, emax=6.0):
    ce = np.asarray(ce, dtype=float)
    return e0 - emax * (ce ** gamma) / (ec50 ** gamma + ce ** gamma)


def fit_scale_anchors(train_df):
    """Fit E0/Emax by least squares, with patient-cluster robust 95% CIs.

    The point estimates solve y = E0 - Emax * f(Ce; EC50=3.08). Confidence
    intervals use a CR1 cluster-robust covariance by stay_id to account for
    repeated observations within patients. These are cohort SAS-scale
    calibration parameters, not pharmacological parameter estimates.
    """
    ce = train_df['Ce'].to_numpy(dtype=float)
    y = train_df['actual_sas'].to_numpy(dtype=float)
    groups = train_df['stay_id'].to_numpy()
    fraction = ce ** 1.47 / (3.08 ** 1.47 + ce ** 1.47)
    design = np.column_stack([np.ones(len(train_df)), -fraction])
    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    residuals = y - design @ coefficients

    xtx_inv = np.linalg.inv(design.T @ design)
    meat = np.zeros((2, 2), dtype=float)
    unique_groups = pd.unique(groups)
    for group in unique_groups:
        mask = groups == group
        score = design[mask].T @ residuals[mask]
        meat += np.outer(score, score)

    n_obs, n_parameters = design.shape
    n_groups = len(unique_groups)
    if n_groups <= 1 or n_obs <= n_parameters:
        raise ValueError('Insufficient training patients/instances for scale-anchor confidence intervals.')
    correction = (n_groups / (n_groups - 1)) * ((n_obs - 1) / (n_obs - n_parameters))
    covariance = correction * xtx_inv @ meat @ xtx_inv
    standard_errors = np.sqrt(np.diag(covariance))
    critical_value = stats.t.ppf(0.975, df=n_groups - 1)
    lower = coefficients - critical_value * standard_errors
    upper = coefficients + critical_value * standard_errors

    return {
        'e0': float(coefficients[0]),
        'e0_ci_lower': float(lower[0]),
        'e0_ci_upper': float(upper[0]),
        'emax': float(coefficients[1]),
        'emax_ci_lower': float(lower[1]),
        'emax_ci_upper': float(upper[1]),
        'n_train_instances': int(n_obs),
        'n_train_patients': int(n_groups),
        'ci_method': '95% t intervals; CR1 cluster-robust covariance by stay_id',
        'interpretation': (
            'E0 and Emax are scale-calibration parameters fitted to this '
            'cohort’s SAS charting, not pharmacological estimates.'
        ),
    }


def predictions_with_anchors(ce, ec50, e0, emax):
    ce = np.asarray(ce, dtype=float)
    ec50 = np.asarray(ec50, dtype=float)
    return sigmoid_emax(ce, ec50, gamma=1.47, e0=e0, emax=emax)


def compute_full_cohort_split(df):
    unique_patients = df['stay_id'].unique()
    train_patients, test_patients = train_test_split(
        unique_patients,
        test_size=0.2,
        random_state=BASE_SEED,
    )
    train_df = df[df['stay_id'].isin(train_patients)].reset_index(drop=True)
    test_df = df[df['stay_id'].isin(test_patients)].reset_index(drop=True)
    return train_df, test_df


def patient_mean_mae(y_true, y_pred, stay_ids):
    df = pd.DataFrame({
        'stay_id': stay_ids,
        'abs_err': np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)),
    })
    return df.groupby('stay_id')['abs_err'].mean()


def paired_wilcoxon_p(reference_mae, comparison_mae):
    ref = pd.Series(reference_mae)
    cmp = pd.Series(comparison_mae)
    merged = pd.concat([ref.rename('reference'), cmp.rename('comparison')], axis=1).dropna()
    if merged.empty:
        return float('nan')
    diff = merged['comparison'] - merged['reference']
    if np.allclose(diff.to_numpy(), 0):
        return 1.0
    try:
        return float(stats.wilcoxon(diff.to_numpy(), zero_method='wilcox', alternative='two-sided').pvalue)
    except ValueError:
        return float('nan')


def xgb_predict(train_df, test_df, features):
    X_train = train_df[features].values
    X_test = test_df[features].values
    y_train = train_df['actual_sas'].values
    y_test = test_df['actual_sas'].values
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=BASE_SEED,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return y_test, preds, model


def run_lstm(train_df, test_df, static_feats, use_ce_seq=False, seed=BASE_SEED):
    np.random.seed(seed)
    tf.random.set_seed(seed)

    sas_train = np.array([np.asarray(s, dtype=float) for s in train_df['sas_sequence']])
    sas_test = np.array([np.asarray(s, dtype=float) for s in test_df['sas_sequence']])

    if use_ce_seq:
        ce_train = np.array([np.asarray(s, dtype=float) for s in train_df['ce_sequence']])
        ce_test = np.array([np.asarray(s, dtype=float) for s in test_df['ce_sequence']])
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
    history = model.fit(
        [seq_train_s, static_train_s],
        y_train,
        epochs=FIXED_EPOCHS,
        batch_size=64,
        validation_split=0.15,
        verbose=0,
    )
    preds = model.predict([seq_test_s, static_test_s], verbose=0).flatten()
    return y_test, preds, history


def make_record(method, mae, sd_test_instance_abs_errors, r2, p_vs_ours,
                sd_ten_run_mae_means='', e0_anchor='', e0_anchor_basis='',
                emax_anchor='', emax_ci_lower='', emax_ci_upper='',
                e0_ci_lower='', e0_ci_upper='', n_train_instances='',
                interpretation=''):
    return {
        'method': method,
        'mae': float(mae),
        'sd_test_instance_abs_errors': (
            float(sd_test_instance_abs_errors)
            if sd_test_instance_abs_errors != '' and pd.notna(sd_test_instance_abs_errors)
            else ''
        ),
        'sd_ten_run_mae_means': (
            float(sd_ten_run_mae_means)
            if sd_ten_run_mae_means != '' and pd.notna(sd_ten_run_mae_means)
            else ''
        ),
        'r2': float(r2),
        'p_vs_ours': '' if p_vs_ours is None else float(p_vs_ours),
        'e0_anchor': e0_anchor,
        'e0_anchor_basis': e0_anchor_basis,
        'e0_ci_lower': e0_ci_lower,
        'e0_ci_upper': e0_ci_upper,
        'emax_anchor': emax_anchor,
        'emax_ci_lower': emax_ci_lower,
        'emax_ci_upper': emax_ci_upper,
        'n_train_instances': n_train_instances,
        'interpretation': interpretation,
    }


def main():
    with open(CACHE_FILE, 'rb') as f:
        df = pickle.load(f)

    train_df, test_df = compute_full_cohort_split(df)
    y_true = test_df['actual_sas'].to_numpy()
    stay_ids = test_df['stay_id'].to_numpy()
    scale_fit = fit_scale_anchors(train_df)
    drug_free = test_df['Ce'].eq(0)
    if not drug_free.any():
        raise ValueError('No zero-Ce test instances are available for the E0 sensitivity anchor.')
    e0_empirical = float(test_df.loc[drug_free, 'actual_sas'].mean())

    # Deterministic baselines.
    persistence_pred = test_df['sas_last'].to_numpy()
    persistence_mae = mean_absolute_error(y_true, persistence_pred)
    persistence_r2 = r2_score(y_true, persistence_pred)
    persistence_sd = float(np.std(np.abs(y_true - persistence_pred), ddof=1)) if len(y_true) > 1 else 0.0

    population_pred = test_df['mech_pred_pop'].to_numpy()
    population_mae = mean_absolute_error(y_true, population_pred)
    population_r2 = r2_score(y_true, population_pred)
    population_sd = float(np.std(np.abs(y_true - population_pred), ddof=1)) if len(y_true) > 1 else 0.0

    personalized_pred = test_df['mech_pred_calib'].to_numpy()
    personalized_mae = mean_absolute_error(y_true, personalized_pred)
    personalized_r2 = r2_score(y_true, personalized_pred)
    personalized_sd = float(np.std(np.abs(y_true - personalized_pred), ddof=1)) if len(y_true) > 1 else 0.0

    # Sensitivity only: retain the primary E0=7 scores above, and shift the
    # intercept to the observed test-set SAS mean at exactly zero modeled Ce.
    population_reanchored = population_pred + (e0_empirical - 7.0)
    personalized_reanchored = personalized_pred + (e0_empirical - 7.0)
    population_reanchored_mae = mean_absolute_error(y_true, population_reanchored)
    population_reanchored_sd = float(np.std(np.abs(y_true - population_reanchored), ddof=1))
    population_reanchored_r2 = r2_score(y_true, population_reanchored)
    personalized_reanchored_mae = mean_absolute_error(y_true, personalized_reanchored)
    personalized_reanchored_sd = float(np.std(np.abs(y_true - personalized_reanchored), ddof=1))
    personalized_reanchored_r2 = r2_score(y_true, personalized_reanchored)

    population_calibrated = predictions_with_anchors(
        test_df['Ce'], 3.08, scale_fit['e0'], scale_fit['emax']
    )
    personalized_calibrated = predictions_with_anchors(
        test_df['Ce'], test_df['EC50_calibrated'],
        scale_fit['e0'], scale_fit['emax'],
    )
    population_calibrated_mae = mean_absolute_error(y_true, population_calibrated)
    population_calibrated_sd = float(np.std(np.abs(y_true - population_calibrated), ddof=1))
    population_calibrated_r2 = r2_score(y_true, population_calibrated)
    personalized_calibrated_mae = mean_absolute_error(y_true, personalized_calibrated)
    personalized_calibrated_sd = float(np.std(np.abs(y_true - personalized_calibrated), ddof=1))
    personalized_calibrated_r2 = r2_score(y_true, personalized_calibrated)

    # Train the two model baselines needed for paired p-values.
    xgb_y_test, xgb_pred, _ = xgb_predict(train_df, test_df, CLINICAL)
    xgb_mae = mean_absolute_error(xgb_y_test, xgb_pred)
    xgb_r2 = r2_score(xgb_y_test, xgb_pred)
    xgb_sd = float(np.std(np.abs(xgb_y_test - xgb_pred), ddof=1)) if len(xgb_y_test) > 1 else 0.0

    full_hybrid_y_test, full_hybrid_pred, _ = run_lstm(
        train_df,
        test_df,
        CLINICAL + STATIC_MECH,
        use_ce_seq=True,
        seed=BASE_SEED,
    )
    full_hybrid_mae = mean_absolute_error(full_hybrid_y_test, full_hybrid_pred)
    full_hybrid_r2 = r2_score(full_hybrid_y_test, full_hybrid_pred)
    full_hybrid_sd = float(np.std(np.abs(full_hybrid_y_test - full_hybrid_pred), ddof=1)) if len(full_hybrid_y_test) > 1 else 0.0

    # Bias-corrected p-values against the canonical seed-42 full-hybrid model.
    ours_patient_mae = patient_mean_mae(full_hybrid_y_test, full_hybrid_pred, stay_ids)

    sweep = pd.read_csv(CAPACITY_SWEEP_FILE)
    full_hybrid_10 = sweep[
        (sweep['algorithm'] == 'LSTM') &
        (sweep['configuration'] == 'FULL_HYBRID') &
        (sweep['lstm_units'] == 4)
    ]
    pure_ml_10 = sweep[
        (sweep['algorithm'] == 'LSTM') &
        (sweep['configuration'] == 'PURE_ML') &
        (sweep['lstm_units'] == 4)
    ]
    if len(full_hybrid_10) != 10 or len(pure_ml_10) != 10:
        raise ValueError('Expected 10 full-cohort capacity-sweep runs per 4-unit LSTM configuration.')

    baseline_records = [
        make_record(
            'Persistence (SAS stays at sas_last)',
            persistence_mae,
            persistence_sd,
            persistence_r2,
            paired_wilcoxon_p(
                patient_mean_mae(y_true, persistence_pred, stay_ids),
                ours_patient_mae,
            ),
        ),
        make_record(
            'Eleveld, population EC50',
            population_mae,
            population_sd,
            population_r2,
            paired_wilcoxon_p(
                patient_mean_mae(y_true, population_pred, stay_ids),
                ours_patient_mae,
            ),
            e0_anchor=7.0,
            e0_anchor_basis='Published fixed E0',
        ),
        make_record(
            'Eleveld, personalized EC50',
            personalized_mae,
            personalized_sd,
            personalized_r2,
            paired_wilcoxon_p(
                patient_mean_mae(y_true, personalized_pred, stay_ids),
                ours_patient_mae,
            ),
            e0_anchor=7.0,
            e0_anchor_basis='Published fixed E0',
        ),
        make_record(
            'Eleveld, population EC50 (UNUSABLE n=3 anchor diagnostic; not primary)',
            population_reanchored_mae,
            population_reanchored_sd,
            population_reanchored_r2,
            None,
            e0_anchor=e0_empirical,
            e0_anchor_basis=(
                f'UNUSABLE diagnostic: mean observed SAS where test Ce == 0 '
                f'(n={int(drug_free.sum())})'
            ),
        ),
        make_record(
            'Eleveld, personalized EC50 (UNUSABLE n=3 anchor diagnostic; not primary)',
            personalized_reanchored_mae,
            personalized_reanchored_sd,
            personalized_reanchored_r2,
            None,
            e0_anchor=e0_empirical,
            e0_anchor_basis=(
                f'UNUSABLE diagnostic: mean observed SAS where test Ce == 0 '
                f'(n={int(drug_free.sum())})'
            ),
        ),
        make_record(
            'Eleveld, population EC50 (training-calibrated E0/Emax)',
            population_calibrated_mae,
            population_calibrated_sd,
            population_calibrated_r2,
            None,
            e0_anchor=scale_fit['e0'],
            e0_anchor_basis='Training-patient least-squares fit; EC50 fixed at 3.08',
            emax_anchor=scale_fit['emax'],
            emax_ci_lower=scale_fit['emax_ci_lower'],
            emax_ci_upper=scale_fit['emax_ci_upper'],
            e0_ci_lower=scale_fit['e0_ci_lower'],
            e0_ci_upper=scale_fit['e0_ci_upper'],
            n_train_instances=scale_fit['n_train_instances'],
            interpretation=scale_fit['interpretation'],
        ),
        make_record(
            'Eleveld, personalized EC50 (same training-calibrated E0/Emax)',
            personalized_calibrated_mae,
            personalized_calibrated_sd,
            personalized_calibrated_r2,
            None,
            e0_anchor=scale_fit['e0'],
            e0_anchor_basis='Training-patient least-squares fit; EC50 fixed at 3.08 for fitting',
            emax_anchor=scale_fit['emax'],
            emax_ci_lower=scale_fit['emax_ci_lower'],
            emax_ci_upper=scale_fit['emax_ci_upper'],
            e0_ci_lower=scale_fit['e0_ci_lower'],
            e0_ci_upper=scale_fit['e0_ci_upper'],
            n_train_instances=scale_fit['n_train_instances'],
            interpretation=scale_fit['interpretation'],
        ),
        make_record(
            'XGBoost',
            xgb_mae,
            xgb_sd,
            xgb_r2,
            paired_wilcoxon_p(
                patient_mean_mae(xgb_y_test, xgb_pred, stay_ids),
                ours_patient_mae,
            ),
        ),
        make_record(
            'LSTM PURE_ML, 4 units',
            pure_ml_10['mae'].mean(),
            '',
            pure_ml_10['r2'].mean(),
            None,
            sd_ten_run_mae_means=pure_ml_10['mae'].std(ddof=1),
        ),
    ]

    our_summary = {
        'method': 'LSTM FULL_HYBRID, 4 units',
        'mae': float(full_hybrid_10['mae'].mean()),
        'sd_test_instance_abs_errors': '',
        'sd_ten_run_mae_means': float(full_hybrid_10['mae'].std(ddof=1)),
        'r2': float(full_hybrid_10['r2'].mean()),
        'p_vs_ours': '',
        'e0_anchor': '',
        'e0_anchor_basis': '',
        'e0_ci_lower': '',
        'e0_ci_upper': '',
        'emax_anchor': '',
        'emax_ci_lower': '',
        'emax_ci_upper': '',
        'n_train_instances': '',
        'interpretation': '',
    }

    records = baseline_records + [our_summary]
    output = pd.DataFrame(
        records,
        columns=[
            'method', 'mae', 'sd_test_instance_abs_errors',
            'sd_ten_run_mae_means', 'r2', 'p_vs_ours', 'e0_anchor',
            'e0_anchor_basis', 'e0_ci_lower', 'e0_ci_upper',
            'emax_anchor', 'emax_ci_lower', 'emax_ci_upper',
            'n_train_instances', 'interpretation',
        ],
    )
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(RESULTS_FILE, index=False)
    pd.DataFrame([scale_fit]).to_csv(CALIBRATION_FILE, index=False)
    print(f'Wrote {RESULTS_FILE}')
    print(f'Wrote {CALIBRATION_FILE}')
    print(scale_fit)
    print(output.to_string(index=False))


if __name__ == '__main__':
    main()
