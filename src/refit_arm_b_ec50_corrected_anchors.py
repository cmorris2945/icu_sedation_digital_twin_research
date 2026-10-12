#!/usr/bin/env python3
"""Refit patient-specific PK/PD parameters on Arm B with corrected anchors.

The existing cached EC50 values and cache file are read-only. This script
rebuilds Ce trajectories from the raw infusion records to fit each patient's
first-half SAS observations and scores only that patient's held-out half.
Variant B's shrinkage coefficient is selected using only Arm B training
patients, by their held-out-half pooled MAE.
"""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from scipy import stats
from scipy.integrate import odeint
from scipy.optimize import minimize_scalar
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data'
CACHE_FILE = ROOT / 'cached_features.pkl'
OUT_SUMMARY = ROOT / 'results' / 'processed' / 'arm_b_ec50_refit_comparison.csv'
OUT_PATIENTS = ROOT / 'results' / 'processed' / 'arm_b_ec50_refit_patient_level.csv'
OUT_SHRINKAGE = ROOT / 'results' / 'processed' / 'arm_b_ec50_shrinkage_selection.csv'

BASE_SEED = 42
SEQUENCE_LENGTH = 10
SIM_MINUTES = 72 * 60
EC50_POP = 3.08
EC50_BOUNDS = (0.5, 10.0)
E0_CAL = 3.9866
EMAX_CAL = 1.7315
E0_BOUNDS = (1.0, 7.0)  # SAS scale limits for variant C's fitted baseline.
GAMMA = 1.47

PK = {
    'V1': 6.28,
    'V2': 25.5,
    'V3': 273.0,
    'CL': 1.79,
    'Q2': 1.75,
    'Q3': 1.11,
    'ke0': 0.146,
}

# Training-only CV grid. Penalty term added to mean squared SAS error:
# lambda * (EC50 - 3.08)^2 / n_usable. Thus identical lambda shrinks a
# small-n patient's estimate more strongly than a large-n patient's estimate.
SHRINKAGE_GRID = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]


def pk_ode(y, t, rate_array):
    a1, a2, a3, ce = y
    c1 = a1 / PK['V1']
    idx = int(t)
    rate_in = rate_array[idx] if 0 <= idx < len(rate_array) else 0.0
    da1 = (
        rate_in
        - (PK['CL'] / PK['V1']) * a1
        - (PK['Q2'] / PK['V1']) * a1 + (PK['Q2'] / PK['V2']) * a2
        - (PK['Q3'] / PK['V1']) * a1 + (PK['Q3'] / PK['V3']) * a3
    )
    da2 = (PK['Q2'] / PK['V1']) * a1 - (PK['Q2'] / PK['V2']) * a2
    da3 = (PK['Q3'] / PK['V1']) * a1 - (PK['Q3'] / PK['V3']) * a3
    dce = PK['ke0'] * (c1 - ce)
    return [da1, da2, da3, dce]


def simulate_ce(propofol_events, weight_kg):
    """Reproduce the 72-hour PK trajectory calculation in 01_build_features."""
    t_grid = np.arange(SIM_MINUTES + 1)
    rate_array = np.zeros(len(t_grid), dtype=float)
    if len(propofol_events):
        first_time = propofol_events['starttime'].min()
        for event in propofol_events.itertuples(index=False):
            start_min = int((event.starttime - first_time).total_seconds() / 60)
            end_min = int((event.endtime - first_time).total_seconds() / 60)
            rate_mg_min = 0.0
            if pd.notna(event.rate) and event.rate > 0:
                # Source SQL expresses rates in mcg/kg/min, as in the builder.
                rate_mg_min = (event.rate * weight_kg) / 1000.0
            if 0 <= start_min < len(rate_array) and end_min < len(rate_array):
                rate_array[start_min:end_min + 1] = rate_mg_min
    solution = odeint(
        pk_ode, [0.0, 0.0, 0.0, 0.0], t_grid,
        args=(rate_array,), mxstep=5000,
    )
    return solution[:, 3], propofol_events['starttime'].min()


def sas_ce_arrays(patient_sas, patient_propofol, weight_kg):
    patient_sas = patient_sas.sort_values('charttime')
    ce_trajectory, infusion_start = simulate_ce(patient_propofol, weight_kg)
    times = (
        (patient_sas['charttime'] - infusion_start).dt.total_seconds().to_numpy() / 60
    ).astype(int)
    sas = patient_sas['sas_score'].to_numpy(dtype=float)
    return times, sas, ce_trajectory


def sigmoid(ce, ec50, e0, emax):
    ce = np.asarray(ce, dtype=float)
    ec50 = np.asarray(ec50, dtype=float)
    return e0 - emax * ce ** GAMMA / (ec50 ** GAMMA + ce ** GAMMA)


def usable_split(times, sas, ce_trajectory):
    n_obs = len(sas)
    n_calib = max(SEQUENCE_LENGTH, n_obs // 2)
    calib_indices = np.array([
        i for i in range(n_calib)
        if 0 <= times[i] < len(ce_trajectory)
    ], dtype=int)
    test_indices = np.array([
        i for i in range(n_calib, n_obs)
        if 0 <= times[i] < len(ce_trajectory)
    ], dtype=int)
    return n_calib, calib_indices, test_indices


def optimize_ec50(ce, sas, e0, emax, shrinkage=0.0):
    n_usable = len(sas)
    if not n_usable:
        return EC50_POP, False

    def objective(candidate):
        residual = sigmoid(ce, candidate, e0, emax) - sas
        mse = float(np.mean(residual ** 2))
        penalty = shrinkage * (candidate - EC50_POP) ** 2 / n_usable
        return mse + penalty

    result = minimize_scalar(
        objective, bounds=EC50_BOUNDS, method='bounded',
        options={'xatol': 1e-8},
    )
    estimate = float(result.x)
    at_boundary = (
        estimate <= EC50_BOUNDS[0] + 1e-3
        or estimate >= EC50_BOUNDS[1] - 1e-3
    )
    return estimate, at_boundary


def optimize_e0(ce, sas):
    if not len(sas):
        return E0_CAL, False
    effect = EMAX_CAL * ce ** GAMMA / (EC50_POP ** GAMMA + ce ** GAMMA)
    unbounded = float(np.mean(sas + effect))
    estimate = float(np.clip(unbounded, *E0_BOUNDS))
    at_boundary = (
        estimate <= E0_BOUNDS[0] + 1e-10
        or estimate >= E0_BOUNDS[1] - 1e-10
    )
    return estimate, at_boundary


def fit_one_patient(times, sas, ce_trajectory, shrinkage=0.0):
    n_calib, calib_indices, test_indices = usable_split(times, sas, ce_trajectory)
    ce_calib = ce_trajectory[times[calib_indices]]
    sas_calib = sas[calib_indices]
    ce_test = ce_trajectory[times[test_indices]]
    sas_test = sas[test_indices]

    ec50_a, boundary_a = optimize_ec50(ce_calib, sas_calib, E0_CAL, EMAX_CAL)
    ec50_b, boundary_b = optimize_ec50(
        ce_calib, sas_calib, E0_CAL, EMAX_CAL, shrinkage=shrinkage
    )
    e0_c, boundary_c = optimize_e0(ce_calib, sas_calib)

    predictions = {
        'population': sigmoid(ce_test, EC50_POP, E0_CAL, EMAX_CAL),
        'A_ec50_refit': sigmoid(ce_test, ec50_a, E0_CAL, EMAX_CAL),
        'B_ec50_shrinkage': sigmoid(ce_test, ec50_b, E0_CAL, EMAX_CAL),
        'C_e0_refit': sigmoid(ce_test, EC50_POP, e0_c, EMAX_CAL),
    }
    return {
        'n_sas_observations': len(sas),
        'n_calib_nominal': n_calib,
        'n_calib_usable': len(calib_indices),
        'n_test_usable': len(test_indices),
        'ec50_A': ec50_a,
        'ec50_B': ec50_b,
        'e0_C': e0_c,
        'boundary_A': boundary_a,
        'boundary_B': boundary_b,
        'boundary_C': boundary_c,
        'fallback_no_calib': len(calib_indices) == 0,
        'test_indices': test_indices,
        'test_sas': sas_test,
        'predictions': predictions,
    }


def choose_shrinkage(training_records, patient_inputs):
    """Select lambda using only Arm B training-patient held-out-half MAE."""
    rows = []
    for weight in SHRINKAGE_GRID:
        actual_all = []
        pred_all = []
        for stay_id in training_records:
            times, sas, ce = patient_inputs[stay_id]
            fit = fit_one_patient(times, sas, ce, shrinkage=weight)
            actual_all.extend(fit['test_sas'])
            pred_all.extend(fit['predictions']['B_ec50_shrinkage'])
        mae = float(mean_absolute_error(actual_all, pred_all))
        rows.append({'shrinkage_weight': weight, 'train_patient_cv_mae': mae,
                     'n_train_patients': len(training_records),
                     'selection_data': 'Arm B training patients; each patient first-half fit, second-half score'})
    scores = pd.DataFrame(rows)
    best = scores.loc[scores['train_patient_cv_mae'].idxmin()]
    return float(best['shrinkage_weight']), scores


def patient_error_series(frame, column):
    return frame.groupby('stay_id')[column].mean()


def paired_patient_wilcoxon(frame, variant_column):
    paired = frame.groupby('stay_id')[[variant_column, 'population_abs_error']].mean().dropna()
    difference = paired[variant_column] - paired['population_abs_error']
    if np.allclose(difference.to_numpy(), 0):
        return 1.0
    return float(stats.wilcoxon(difference.to_numpy(), alternative='two-sided').pvalue)


def score_summary(eval_frame, method, pred_col, boundary_count, boundary_total,
                  p_value=None, shrinkage_weight=np.nan):
    y = eval_frame['actual_sas'].to_numpy(dtype=float)
    pred = eval_frame[pred_col].to_numpy(dtype=float)
    absolute_errors = np.abs(y - pred)
    return {
        'variant': method,
        'n_test_instances': len(eval_frame),
        'n_test_patients': eval_frame['stay_id'].nunique(),
        'mae': float(mean_absolute_error(y, pred)),
        'sd_test_instance_abs_errors': float(np.std(absolute_errors, ddof=1)),
        'r2': float(r2_score(y, pred)),
        'boundary_patients': int(boundary_count),
        'patients_fitted': int(boundary_total),
        'boundary_definition': (
            'EC50 at/within 0.001 of [0.5,10]'
            if method.startswith(('A_', 'B_'))
            else 'E0 clipped to SAS scale bounds [1,7]'
            if method.startswith('C_') else 'not applicable'
        ),
        'paired_wilcoxon_vs_population_patient_mae': p_value,
        'shrinkage_weight': shrinkage_weight,
        'shrinking_penalty': 'lambda*(EC50-3.08)^2/n_usable' if method.startswith('B_') else '',
    }


def main():
    with open(CACHE_FILE, 'rb') as f:
        cached = pickle.load(f)
    ec50_stored = cached.groupby('stay_id')['EC50_calibrated'].first()
    pinned = set(ec50_stored.index[(ec50_stored - 0.5).abs() <= 1e-3]) | set(
        ec50_stored.index[(ec50_stored - 10.0).abs() <= 1e-3]
    )
    arm_b_ids = set(ec50_stored.index) - pinned
    if len(arm_b_ids) != 715:
        raise ValueError(f'Expected 715 Arm B patients from stored EC50 classification; got {len(arm_b_ids)}.')
    train_ids, test_ids = train_test_split(
        np.array([sid for sid in cached['stay_id'].unique() if sid in arm_b_ids]),
        test_size=0.2, random_state=BASE_SEED,
    )
    train_ids, test_ids = set(train_ids), set(test_ids)
    if len(train_ids) != 572 or len(test_ids) != 143:
        raise ValueError('Reconstructed Arm B train/test split does not match the sweep (572/143).')

    patients = pd.read_csv(DATA_DIR / 'kidney_subgroups_patients.csv').set_index('stay_id')
    infusions = pd.read_csv(DATA_DIR / 'kidney_subgroups_propofol.csv')
    observations = pd.read_csv(DATA_DIR / 'kidney_subgroups_sas.csv')
    infusions['starttime'] = pd.to_datetime(infusions['starttime'])
    infusions['endtime'] = pd.to_datetime(infusions['endtime'])
    observations['charttime'] = pd.to_datetime(observations['charttime'])
    infusion_groups = {sid: g.sort_values('starttime') for sid, g in infusions.groupby('stay_id')}
    sas_groups = {sid: g.sort_values('charttime') for sid, g in observations.groupby('stay_id')}

    patient_inputs = {}
    fit_records = []
    boundary = {}
    for index, stay_id in enumerate(arm_b_ids, 1):
        if stay_id not in infusion_groups or stay_id not in sas_groups:
            raise ValueError(f'Missing raw SAS or infusion records for Arm B patient {stay_id}.')
        weight = patients.loc[stay_id, 'weight']
        if pd.isna(weight) or weight <= 0:
            weight = 75.0
        times, sas, ce = sas_ce_arrays(
            sas_groups[stay_id], infusion_groups[stay_id], float(weight)
        )
        patient_inputs[stay_id] = (times, sas, ce)
        if index % 100 == 0:
            print(f'Rebuilt PK/SAS arrays for {index}/{len(arm_b_ids)} Arm B patients')

    selected_lambda, cv_scores = choose_shrinkage(train_ids, patient_inputs)
    OUT_SHRINKAGE.parent.mkdir(parents=True, exist_ok=True)
    cv_scores['selected'] = cv_scores['shrinkage_weight'].eq(selected_lambda)
    cv_scores.to_csv(OUT_SHRINKAGE, index=False)

    for stay_id in sorted(test_ids):
        times, sas, ce = patient_inputs[stay_id]
        fit = fit_one_patient(times, sas, ce, shrinkage=selected_lambda)
        for position, (idx, sas_value) in enumerate(zip(fit['test_indices'], fit['test_sas'])):
            row = {
                'stay_id': stay_id,
                'observation_index': int(idx),
                'actual_sas': float(sas_value),
                'ce': float(ce[times[idx]]),
                'population_prediction': float(fit['predictions']['population'][position]),
                'A_prediction': float(fit['predictions']['A_ec50_refit'][position]),
                'B_prediction': float(fit['predictions']['B_ec50_shrinkage'][position]),
                'C_prediction': float(fit['predictions']['C_e0_refit'][position]),
                'ec50_A': fit['ec50_A'],
                'ec50_B': fit['ec50_B'],
                'e0_C': fit['e0_C'],
                'n_calib_usable': fit['n_calib_usable'],
                'shrinkage_weight': selected_lambda,
            }
            row['population_abs_error'] = abs(row['actual_sas'] - row['population_prediction'])
            row['A_abs_error'] = abs(row['actual_sas'] - row['A_prediction'])
            row['B_abs_error'] = abs(row['actual_sas'] - row['B_prediction'])
            row['C_abs_error'] = abs(row['actual_sas'] - row['C_prediction'])
            fit_records.append(row)
        boundary[stay_id] = {
            'A': fit['boundary_A'], 'B': fit['boundary_B'], 'C': fit['boundary_C'],
            'fallback': fit['fallback_no_calib'],
        }

    eval_frame = pd.DataFrame(fit_records)
    eval_frame.to_csv(OUT_PATIENTS, index=False)
    fallback_count = sum(item['fallback'] for item in boundary.values())
    n_test_patients = len(test_ids)

    p_values = {
        'A_abs_error': paired_patient_wilcoxon(eval_frame, 'A_abs_error'),
        'B_abs_error': paired_patient_wilcoxon(eval_frame, 'B_abs_error'),
        'C_abs_error': paired_patient_wilcoxon(eval_frame, 'C_abs_error'),
    }
    count_a = sum(item['A'] for item in boundary.values())
    count_b = sum(item['B'] for item in boundary.values())
    count_c = sum(item['C'] for item in boundary.values())

    summary = [
        score_summary(eval_frame, 'Population baseline (corrected anchors; EC50=3.08)',
                      'population_prediction', 0, n_test_patients),
        score_summary(eval_frame, 'A_EC50 refit; corrected anchors',
                      'A_prediction', count_a, n_test_patients,
                      p_values['A_abs_error']),
        score_summary(eval_frame, 'B_EC50 refit with shrinkage',
                      'B_prediction', count_b, n_test_patients,
                      p_values['B_abs_error'], selected_lambda),
        score_summary(eval_frame, 'C_E0 refit; EC50=3.08',
                      'C_prediction', count_c, n_test_patients,
                      p_values['C_abs_error']),
    ]
    summary_df = pd.DataFrame(summary)
    summary_df['anchor_E0_fixed'] = E0_CAL
    summary_df['anchor_Emax_fixed'] = EMAX_CAL
    summary_df['gamma_fixed'] = GAMMA
    summary_df['train_patient_count'] = len(train_ids)
    summary_df['test_patient_count'] = len(test_ids)
    summary_df['n_test_split_patients'] = len(test_ids)
    summary_df['n_test_patients_scored'] = eval_frame['stay_id'].nunique()
    summary_df['no_usable_calibration_patients_test'] = fallback_count
    summary_df['patients_fitted'] = [
        0, n_test_patients - fallback_count,
        n_test_patients - fallback_count, n_test_patients - fallback_count,
    ]
    summary_df['shrinkage_selection'] = (
        'Minimum pooled held-out-half MAE over Arm B training patients; '
        'grid searched over ' + ','.join(str(v) for v in SHRINKAGE_GRID)
    )
    summary_df.to_csv(OUT_SUMMARY, index=False)

    print(f'Arm B: {len(train_ids)} train / {len(test_ids)} test patients')
    print(f'Variant B shrinkage lambda: {selected_lambda}; selection based only on '
          'Arm B training patients’ held-out-half pooled MAE')
    print(f'No-usable-calibration test patients (population fallback): {fallback_count}')
    print(summary_df.to_string(index=False))
    print(f'Wrote {OUT_SUMMARY}')
    print(f'Wrote {OUT_PATIENTS}')
    print(f'Wrote {OUT_SHRINKAGE}')


if __name__ == '__main__':
    main()
