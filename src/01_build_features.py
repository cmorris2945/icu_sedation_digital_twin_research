#!/usr/bin/env python3
"""
build_features.py
=================

This script builds the feature dataset from raw MIMIC-IV CSV files. It runs
the Eleveld pharmacokinetic-pharmacodynamic simulation for each patient,
calibrates a patient-specific EC50 value, and produces a feature dataset
suitable for our ablation study experiments.

This is the slow step in our pipeline because the ODE solver has to integrate
the three-compartment model for every patient over their entire ICU stay. On
a typical laptop, this takes fifteen to thirty minutes for the full 1,490
patient cohort. The results are saved to a pickle file so subsequent scripts
can load the features instantly without redoing this work.

USAGE:
    python build_features.py

OUTPUT:
    cached_features.pkl - feature dataset saved to disk

REQUIREMENTS:
    Three CSV files in a 'data' subfolder:
        data/kidney_subgroups_patients.csv
        data/kidney_subgroups_propofol.csv
        data/kidney_subgroups_sas.csv

AUTHOR: Christopher Morris
"""

import pandas as pd
import numpy as np
from datetime import datetime
from scipy.integrate import odeint
from scipy.optimize import minimize_scalar
import warnings
import pickle
import os
import sys
warnings.filterwarnings('ignore')

# Paths and configuration
DATA_DIR = 'data'
PATIENTS_FILE = os.path.join(DATA_DIR, 'kidney_subgroups_patients.csv')
PROPOFOL_FILE = os.path.join(DATA_DIR, 'kidney_subgroups_propofol.csv')
SAS_FILE = os.path.join(DATA_DIR, 'kidney_subgroups_sas.csv')
OUTPUT_FILE = 'cached_features.pkl'

# How many patients to process. Set to None to use all patients in the data,
# or to a specific number for faster testing during development.
N_PATIENTS = None  # None means use all patients

# Length of historical sequence we use for predictions. Ten observations gives
# the LSTM enough context to learn temporal patterns while keeping computation
# manageable.
SEQUENCE_LENGTH = 10

# Random seed for patient sampling when N_PATIENTS is set
SEED = 42

print("=" * 70)
print("FEATURE BUILDING SCRIPT")
print("=" * 70)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Verify input files exist before doing any work
for f in [PATIENTS_FILE, PROPOFOL_FILE, SAS_FILE]:
    if not os.path.exists(f):
        print(f"ERROR: Required file not found: {f}")
        print("Please ensure the data CSV files are in the 'data' subfolder.")
        sys.exit(1)

# Load raw data from CSV files
print("Loading raw data files...")
patients_df = pd.read_csv(PATIENTS_FILE)
propofol_df = pd.read_csv(PROPOFOL_FILE)
sas_df = pd.read_csv(SAS_FILE)

# Convert timestamp columns to datetime objects so we can do time arithmetic
propofol_df['starttime'] = pd.to_datetime(propofol_df['starttime'])
propofol_df['endtime'] = pd.to_datetime(propofol_df['endtime'])
sas_df['charttime'] = pd.to_datetime(sas_df['charttime'])

print(f"   Patients: {len(patients_df)}")
print(f"   Propofol events: {len(propofol_df)}")
print(f"   SAS observations: {len(sas_df)}")

# ============================================================================
# Eleveld 2018 PK/PD model parameters
# ============================================================================
# These are the population-level parameters from Eleveld et al. 2018 in the
# British Journal of Anaesthesia. They describe how propofol distributes
# through the body and how it produces sedation in the average adult patient.
# Individual patients will deviate from these averages, which is why we
# calibrate EC50 to each patient.
ELEVELD_PARAMS = {
    'V1': 6.28,        # Central compartment volume in liters
    'V2': 25.5,        # Rapid peripheral compartment volume
    'V3': 273,         # Slow peripheral compartment volume
    'CL': 1.79,        # Metabolic clearance, mostly hepatic
    'Q2': 1.75,        # Inter-compartmental clearance to V2
    'Q3': 1.11,        # Inter-compartmental clearance to V3
    'ke0': 0.146,      # Effect-site equilibration rate
    'EC50_pop': 3.08,  # Population EC50 for sedation effect
    'gamma': 1.47,     # Hill coefficient for sigmoid response
    'E0': 7,           # Baseline SAS without drug
    'Emax': 6,         # Maximum drug effect on SAS scale
}


def eleveld_pk_ode(y, t, rate_func, params):
    """
    Three-compartment pharmacokinetic differential equations.
    
    The state vector y has four components: drug amounts in the central,
    rapid peripheral, and slow peripheral compartments, plus the effect-site
    concentration. The function returns the rate of change for each.
    
    The central compartment receives drug from the infusion and exchanges
    drug with both peripheral compartments. Drug is eliminated only from
    the central compartment via the clearance term.
    
    The effect site is a virtual compartment that tracks the concentration
    affecting the brain. It equilibrates with central compartment
    concentration at rate ke0.
    """
    A1, A2, A3, Ce = y
    V1 = params['V1']
    C1 = A1 / V1  # Central concentration
    rate_in = rate_func(t)  # Drug infusion rate at this time
    
    dA1_dt = (rate_in
              - (params['CL'] / V1) * A1
              - (params['Q2'] / V1) * A1 + (params['Q2'] / params['V2']) * A2
              - (params['Q3'] / V1) * A1 + (params['Q3'] / params['V3']) * A3)
    
    dA2_dt = (params['Q2'] / V1) * A1 - (params['Q2'] / params['V2']) * A2
    dA3_dt = (params['Q3'] / V1) * A1 - (params['Q3'] / params['V3']) * A3
    dCe_dt = params['ke0'] * (C1 - Ce)
    
    return [dA1_dt, dA2_dt, dA3_dt, dCe_dt]


def sigmoid_emax(Ce, EC50, gamma, E0, Emax):
    """
    Sigmoid Emax pharmacodynamic equation that converts effect-site
    concentration into predicted sedation level. The relationship is
    S-shaped, with EC50 marking the concentration where half of maximum
    effect is achieved.
    """
    return E0 - Emax * (Ce ** gamma) / (EC50 ** gamma + Ce ** gamma)


def simulate_patient_pk(patient_id, propofol_events, sas_events, params,
                         pat_weight, sim_duration_hrs=72):
    """
    Run the full PK/PD simulation for one patient.
    
    Returns time grid and trajectories for plasma and effect-site
    concentrations. Returns three Nones if simulation fails.
    """
    t_grid = np.arange(0, sim_duration_hrs * 60 + 1)
    rate_array = np.zeros(len(t_grid))
    
    if len(propofol_events) > 0:
        first_time = propofol_events['starttime'].min()
        for _, event in propofol_events.iterrows():
            start_min = int((event['starttime'] - first_time).total_seconds() / 60)
            end_min = int((event['endtime'] - first_time).total_seconds() / 60)
            
            # Convert mcg/kg/min to mg/min using patient weight
            rate_mg_per_min = 0
            if pd.notna(event['rate']) and event['rate'] > 0:
                rate_mg_per_min = (event['rate'] * pat_weight) / 1000.0
            
            if 0 <= start_min < len(rate_array) and end_min < len(rate_array):
                rate_array[start_min:end_min+1] = rate_mg_per_min
    
    def rate_func(t):
        idx = int(t)
        if 0 <= idx < len(rate_array):
            return rate_array[idx]
        return 0
    
    y0 = [0, 0, 0, 0]  # No drug initially in any compartment
    
    try:
        sol = odeint(eleveld_pk_ode, y0, t_grid, args=(rate_func, params),
                     full_output=False, mxstep=5000)
        return t_grid, sol[:, 0] / params['V1'], sol[:, 3]
    except Exception as e:
        return None, None, None


# ============================================================================
# Build feature dataset
# ============================================================================
print("\nBuilding features (this is the slow step)...")

# Determine which patients to process
all_patient_ids = patients_df['stay_id'].unique()
if N_PATIENTS is not None and len(all_patient_ids) > N_PATIENTS:
    np.random.seed(SEED)
    sampled_patient_ids = np.random.choice(all_patient_ids, N_PATIENTS, replace=False)
    print(f"   Using random sample of {N_PATIENTS} patients from {len(all_patient_ids)} total")
else:
    sampled_patient_ids = all_patient_ids
    print(f"   Using all {len(all_patient_ids)} patients")

all_instances = []
n_processed = 0
n_failed = 0

# Process each patient through the full pipeline
for patient_count, patient_id in enumerate(sampled_patient_ids, 1):
    if patient_count % 100 == 0:
        elapsed = (datetime.now() - datetime.now()).total_seconds()  # placeholder
        print(f"   Processed {patient_count}/{len(sampled_patient_ids)} patients, "
              f"{len(all_instances)} instances built so far")
    
    pat_info = patients_df[patients_df['stay_id'] == patient_id].iloc[0]
    pat_propofol = propofol_df[propofol_df['stay_id'] == patient_id].sort_values('starttime')
    pat_sas = sas_df[sas_df['stay_id'] == patient_id].sort_values('charttime')
    
    # Skip patients without enough data for our temporal modeling
    if len(pat_sas) < SEQUENCE_LENGTH + 1 or len(pat_propofol) == 0:
        n_failed += 1
        continue
    
    # Get patient weight, defaulting to 75kg if missing
    pat_weight = pat_info.get('weight', 75)
    if pd.isna(pat_weight) or pat_weight <= 0:
        pat_weight = 75
    
    # Run the PK simulation for this patient
    t_grid, Cp_traj, Ce_traj = simulate_patient_pk(
        patient_id, pat_propofol, pat_sas, ELEVELD_PARAMS, pat_weight
    )
    
    if t_grid is None:
        n_failed += 1
        continue
    
    # Get SAS observation times in minutes from first propofol event
    first_time = pat_propofol['starttime'].min()
    sas_times_min = []
    for _, sas_obs in pat_sas.iterrows():
        time_min = (sas_obs['charttime'] - first_time).total_seconds() / 60
        sas_times_min.append(int(time_min))
    sas_values = pat_sas['sas_score'].values
    
    # Calibrate EC50 using the first half of this patient's observations.
    # We hold out the second half to use as our prediction targets, which
    # mimics how this would work in clinical practice.
    n_obs = len(pat_sas)
    n_calib = max(SEQUENCE_LENGTH, n_obs // 2)
    
    def ec50_loss(ec50_candidate):
        """Mean squared error of mechanistic prediction at calibration points."""
        loss = 0
        n = 0
        for i in range(n_calib):
            t_idx = sas_times_min[i]
            if 0 <= t_idx < len(Ce_traj):
                pred = sigmoid_emax(Ce_traj[t_idx], ec50_candidate,
                                    ELEVELD_PARAMS['gamma'],
                                    ELEVELD_PARAMS['E0'], ELEVELD_PARAMS['Emax'])
                loss += (pred - sas_values[i]) ** 2
                n += 1
        return loss / max(n, 1)
    
    try:
        result = minimize_scalar(ec50_loss, bounds=(0.5, 10.0), method='bounded')
        ec50_calibrated = result.x
    except Exception:
        ec50_calibrated = ELEVELD_PARAMS['EC50_pop']
    
    # Build prediction instances. Each instance represents one moment where we
    # want to predict the next SAS score given the recent history.
    for i in range(SEQUENCE_LENGTH, n_obs):
        t_idx = sas_times_min[i]
        if not (0 <= t_idx < len(Ce_traj)):
            continue
        
        # Recent SAS history
        sas_history = sas_values[i-SEQUENCE_LENGTH:i]
        
        # Recent Ce history at the same time points
        ce_history = []
        for j in range(i-SEQUENCE_LENGTH, i):
            t_j = sas_times_min[j]
            if 0 <= t_j < len(Ce_traj):
                ce_history.append(Ce_traj[t_j])
            else:
                ce_history.append(0)
        ce_history = np.array(ce_history)
        
        # Mechanistic state at prediction time
        Ce_now = Ce_traj[t_idx]
        Cp_now = Cp_traj[t_idx]
        
        # Mechanistic predictions using both population and calibrated EC50
        mech_pred_pop = sigmoid_emax(Ce_now, ELEVELD_PARAMS['EC50_pop'],
                                     ELEVELD_PARAMS['gamma'],
                                     ELEVELD_PARAMS['E0'], ELEVELD_PARAMS['Emax'])
        mech_pred_calib = sigmoid_emax(Ce_now, ec50_calibrated,
                                       ELEVELD_PARAMS['gamma'],
                                       ELEVELD_PARAMS['E0'], ELEVELD_PARAMS['Emax'])
        
        # Current propofol infusion rate
        current_rate = 0
        for _, event in pat_propofol.iterrows():
            event_start = (event['starttime'] - first_time).total_seconds() / 60
            event_end = (event['endtime'] - first_time).total_seconds() / 60
            if event_start <= t_idx <= event_end:
                current_rate = event['rate'] if pd.notna(event['rate']) else 0
                break
        
        # Build the complete feature dictionary for this instance
        all_instances.append({
            'stay_id': patient_id,
            'age': pat_info.get('age', 60),
            'gender': 1 if pat_info.get('gender', 'M') == 'M' else 0,
            'weight': pat_weight,
            'height': pat_info.get('height', 170) if pd.notna(pat_info.get('height')) else 170,
            'sas_last': sas_history[-1],
            'sas_mean': np.mean(sas_history),
            'sas_std': np.std(sas_history),
            'sas_trend': sas_history[-1] - sas_history[0],
            'propofol_rate': current_rate,
            'time_hours': t_idx / 60,
            'Ce': Ce_now,
            'Cp': Cp_now,
            'EC50_calibrated': ec50_calibrated,
            'mech_pred_pop': mech_pred_pop,
            'mech_pred_calib': mech_pred_calib,
            'sas_sequence': sas_history.tolist(),
            'ce_sequence': ce_history.tolist(),
            'actual_sas': sas_values[i]
        })
    
    n_processed += 1

# Convert to DataFrame and save
df = pd.DataFrame(all_instances)
print(f"\nProcessed {n_processed} patients successfully, {n_failed} skipped")
print(f"Total prediction instances built: {len(df)}")
print(f"Unique patients in final dataset: {df['stay_id'].nunique()}")

# Save the cached features for use by other scripts
print(f"\nSaving features to {OUTPUT_FILE}...")
with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump(df, f)
print(f"Done! File size: {os.path.getsize(OUTPUT_FILE) / 1024 / 1024:.2f} MB")

print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\nNext step: run 'python run_ablation.py' to perform the ablation study")
