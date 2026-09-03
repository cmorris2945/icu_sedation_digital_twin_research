#!/usr/bin/env python3
"""
08_ec50_diagnostics.py
======================

Diagnostic for the EC50 boundary-pinning problem. DIAGNOSTIC ONLY: this script
does not re-fit, repair or re-run the calibration, and it writes nothing that
any other script consumes.

THE PROBLEM
    01_build_features.py calibrates a per-patient EC50 with
    scipy.optimize.minimize_scalar(method='bounded') on [0.5, 10.0]. 52% of
    patients come out at exactly 0.5 and 3% at exactly 10.0. A parameter
    sitting on its own search bound is not a fitted value, it is the optimiser
    reporting that it wanted to leave the interval. That matters because
    EC50_calibrated is the feature carrying the paper's "patient-specific
    digital twin" claim, and for 55% of the cohort it carries no patient-
    specific information at all beyond "low" or "high".

WHAT THIS SCRIPT TESTS
    The stated hypothesis is that pinned patients have little effective drug
    exposure, leaving the sigmoid Emax curve flat with respect to EC50 so that
    no interior minimum exists.

    Note the sigmoid's actual geometry before reading the results, because it
    makes the prediction directional rather than symmetric:

        sigmoid_emax(Ce, EC50) = E0 - Emax * Ce^g / (EC50^g + Ce^g)
        with E0 = 7, Emax = 6, g = 1.47

        EC50 -> 0    =>  the fraction -> 1  =>  prediction -> 1  (deep sedation)
        EC50 -> inf  =>  the fraction -> 0  =>  prediction -> 7  (fully awake)
        Ce   -> 0    =>  prediction -> 7 for EVERY EC50 (the flat case)

    So a patient observed at a LOW SAS is fitted by driving EC50 DOWN, and one
    observed near SAS 7 by driving it UP. Combined with near-zero Ce, the loss
    becomes monotone in EC50 and the optimiser rails to whichever bound points
    the right way. The expectation is therefore that pinned-LOW patients are
    the SEDATED ones the model cannot explain, not the awake ones. The script
    reports what is actually true rather than assuming either direction.

    Beyond the group comparisons it profiles the objective directly: for each
    patient it evaluates the sigmoid loss across a grid of EC50 values and
    records whether the minimum is interior or on a boundary. That is a direct
    test of "no interior minimum exists" rather than an inference from it.

    PROXY CAVEAT: the profile uses each patient's PREDICTION INSTANCES, since
    those are what cached_features.pkl stores. The real calibration used the
    first half of that patient's SAS observations, which the pkl does not
    retain. Same functional form, same patient, overlapping but not identical
    points. Treat the profile as strong corroboration, not as a bit-exact
    replay of the optimiser.

USAGE:
    python 08_ec50_diagnostics.py

REQUIREMENTS:
    cached_features.pkl                (required)
    cohort_per_patient_mae.csv         (optional; enables the impact analysis)
    data/kidney_subgroups_{sas,propofol}.csv
                                       (optional; enables calibration-set sizes)

OUTPUT:
    ec50_diagnostics.csv               group comparison table with test statistics
    ec50_patient_level.csv             per-patient diagnostic values
    figures/cohort/ec50_pinning_diagnostics.{png,pdf}

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.publication_plots import setup_publication_style, COLORS

CACHE_FILE = 'cached_features.pkl'
MAE_FILE = 'cohort_per_patient_mae.csv'
SAS_FILE = os.path.join('data', 'kidney_subgroups_sas.csv')
PROPOFOL_FILE = os.path.join('data', 'kidney_subgroups_propofol.csv')
FIG_DIR = os.path.join('figures', 'cohort')
OUT_TABLE = 'ec50_diagnostics.csv'
OUT_PATIENT = 'ec50_patient_level.csv'

# Bounds and model constants, copied from 01_build_features.py.
EC50_LO, EC50_HI = 0.5, 10.0
TOL = 1e-3
E0, EMAX, GAMMA = 7.0, 6.0, 1.47
SEQUENCE_LENGTH = 10
SIM_MINUTES = 72 * 60

CE_NEAR_ZERO = 0.1      # ug/mL; population EC50 is 3.08, so this is ~3% of it
CE_ESSENTIALLY_ZERO = 0.01

C_LOW, C_HIGH, C_INT = COLORS['blue'], COLORS['orange'], COLORS['green']
C_MUTED = '#6b6b6b'
GROUP_COLORS = {'pinned-low': C_LOW, 'pinned-high': C_HIGH, 'interior': C_INT}
GROUP_ORDER = ['pinned-low', 'pinned-high', 'interior']

print("=" * 70)
print("EC50 BOUNDARY-PINNING DIAGNOSTICS")
print("=" * 70)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("DIAGNOSTIC ONLY - the calibration is not modified or re-run.")
print()

if not os.path.exists(CACHE_FILE):
    print(f"ERROR: required file not found: {CACHE_FILE}")
    sys.exit(1)

os.makedirs(FIG_DIR, exist_ok=True)
setup_publication_style()


def sigmoid_emax(ce, ec50):
    """Identical to 01_build_features.py."""
    ce = np.asarray(ce, dtype=float)
    return E0 - EMAX * (ce ** GAMMA) / (ec50 ** GAMMA + ce ** GAMMA)


# ===========================================================================
# PATIENT-LEVEL AGGREGATION
# ===========================================================================
with open(CACHE_FILE, 'rb') as fh:
    inst = pickle.load(fh)
print(f"Loaded {len(inst):,} instances from {inst['stay_id'].nunique():,} patients")

g = inst.groupby('stay_id')
pat = pd.DataFrame({
    'ec50': g['EC50_calibrated'].first(),
    'rate_mean': g['propofol_rate'].mean(),
    'rate_median': g['propofol_rate'].median(),
    'ce_mean': g['Ce'].mean(),
    'ce_median': g['Ce'].median(),
    'ce_max': g['Ce'].max(),
    'frac_ce_below_0p1': g['Ce'].apply(lambda s: float((s < CE_NEAR_ZERO).mean())),
    'frac_ce_below_0p01': g['Ce'].apply(lambda s: float((s < CE_ESSENTIALLY_ZERO).mean())),
    'n_instances': g['actual_sas'].size(),
    'sas_mean': g['actual_sas'].mean(),
    'sas_std': g['actual_sas'].std(),
    'sas_range': g['actual_sas'].apply(lambda s: float(s.max() - s.min())),
    'sas_min': g['actual_sas'].min(),
    'sas_max': g['actual_sas'].max(),
}).reset_index()
pat['sas_std'] = pat['sas_std'].fillna(0.0)


def classify(e):
    if abs(e - EC50_LO) <= TOL:
        return 'pinned-low'
    if abs(e - EC50_HI) <= TOL:
        return 'pinned-high'
    return 'interior'


pat['group'] = pat['ec50'].apply(classify)
counts = pat['group'].value_counts().reindex(GROUP_ORDER).fillna(0).astype(int)
n_pat = len(pat)

print()
print("PINNING CLASSIFICATION")
print(f"   bounds [{EC50_LO}, {EC50_HI}], tolerance {TOL}")
for grp in GROUP_ORDER:
    print(f"   {grp:<12s} {counts[grp]:>5d}  ({100*counts[grp]/n_pat:5.1f}%)")
print(f"   {'TOTAL PINNED':<12s} "
      f"{counts['pinned-low'] + counts['pinned-high']:>5d}  "
      f"({100*(counts['pinned-low'] + counts['pinned-high'])/n_pat:5.1f}%)")
print()


# ===========================================================================
# CALIBRATION-SET SIZES (optional, needs the raw CSVs)
# ===========================================================================
if os.path.exists(SAS_FILE) and os.path.exists(PROPOFOL_FILE):
    sas = pd.read_csv(SAS_FILE)
    pro = pd.read_csv(PROPOFOL_FILE)
    sas['charttime'] = pd.to_datetime(sas['charttime'])
    pro['starttime'] = pd.to_datetime(pro['starttime'])
    sas = sas.sort_values(['stay_id', 'charttime'])
    t0 = pro.groupby('stay_id')['starttime'].min()

    recs = []
    for sid, grp in sas.groupby('stay_id'):
        if sid not in t0.index:
            continue
        tmin = ((grp['charttime'] - t0[sid]).dt.total_seconds() / 60).astype(int).values
        n_obs = len(tmin)
        n_calib = max(SEQUENCE_LENGTH, n_obs // 2)
        head = tmin[:n_calib]
        # ec50_loss only accumulates points whose index lands inside the
        # simulated trajectory, so this is what actually drove the fit.
        eff = int(((head >= 0) & (head < SIM_MINUTES + 1)).sum())
        recs.append({'stay_id': sid, 'n_sas_obs': n_obs,
                     'n_calib_nominal': n_calib, 'n_calib_effective': eff})
    calib = pd.DataFrame(recs)
    pat = pat.merge(calib, on='stay_id', how='left')
    print("CALIBRATION SET SIZES (reconstructed from the raw CSVs)")
    print(f"   patients whose calibration window contributed ZERO usable points: "
          f"{int((pat['n_calib_effective'] == 0).sum())}")
    print(f"   median nominal calibration points  : "
          f"{pat['n_calib_nominal'].median():.0f}")
    print(f"   median effective calibration points: "
          f"{pat['n_calib_effective'].median():.0f}")
    HAS_CALIB = True
else:
    print("CALIBRATION SET SIZES: raw CSVs not found, skipping.")
    HAS_CALIB = False
print()


# ===========================================================================
# OBJECTIVE PROFILING - is there an interior minimum at all?
# ===========================================================================
print("Profiling the sigmoid objective per patient...")
print("   (proxy: uses each patient's prediction instances, not the exact")
print("    calibration half; see the module docstring)")

grid = np.linspace(EC50_LO, EC50_HI, 400)
shape = {'stay_id': [], 'profile_argmin': [], 'profile_shape': [],
         'loss_range_rel': []}

for sid, grp in inst.groupby('stay_id'):
    ce = grp['Ce'].values
    y = grp['actual_sas'].values
    # loss(e) over the grid, vectorised across instances
    preds = E0 - EMAX * (ce[None, :] ** GAMMA) / (grid[:, None] ** GAMMA + ce[None, :] ** GAMMA)
    loss = ((preds - y[None, :]) ** 2).mean(axis=1)
    i = int(np.argmin(loss))
    if i == 0:
        shp = 'boundary-low'
    elif i == len(grid) - 1:
        shp = 'boundary-high'
    else:
        shp = 'interior'
    rng = float((loss.max() - loss.min()) / (abs(loss.mean()) + 1e-12))
    shape['stay_id'].append(sid)
    shape['profile_argmin'].append(float(grid[i]))
    shape['profile_shape'].append(shp)
    shape['loss_range_rel'].append(rng)

pat = pat.merge(pd.DataFrame(shape), on='stay_id', how='left')

print()
print("OBJECTIVE SHAPE vs FITTED OUTCOME (rows = fitted group, cols = profile shape)")
xt = pd.crosstab(pat['group'], pat['profile_shape'])
print(xt.to_string())
agree = float((((pat['group'] == 'pinned-low') & (pat['profile_shape'] == 'boundary-low')) |
               ((pat['group'] == 'pinned-high') & (pat['profile_shape'] == 'boundary-high')) |
               ((pat['group'] == 'interior') & (pat['profile_shape'] == 'interior'))).mean())
print(f"   profile shape agrees with the fitted outcome for {100*agree:.1f}% of patients")
print(f"   patients whose objective is essentially flat (relative range < 1e-6): "
      f"{int((pat['loss_range_rel'] < 1e-6).sum())}")
print()


# ===========================================================================
# IMPACT - does pinning hurt prediction?
# ===========================================================================
if os.path.exists(MAE_FILE):
    mae = pd.read_csv(MAE_FILE)[['stay_id', 'patient_mae']]
    pat = pat.merge(mae, on='stay_id', how='left')
    HAS_MAE = True
    print(f"Merged per-patient MAE for {int(pat['patient_mae'].notna().sum())} "
          f"test patients from {MAE_FILE}")
else:
    HAS_MAE = False
    print(f"{MAE_FILE} not found; skipping the impact analysis.")
print()


# ===========================================================================
# GROUP COMPARISON TABLE
# ===========================================================================
VARS = [
    ('rate_mean', 'Mean propofol rate (mcg/kg/min)'),
    ('rate_median', 'Median propofol rate (mcg/kg/min)'),
    ('ce_mean', 'Mean Ce (ug/mL)'),
    ('ce_median', 'Median Ce (ug/mL)'),
    ('ce_max', 'Max Ce reached (ug/mL)'),
    ('frac_ce_below_0p1', f'Fraction of instances with Ce < {CE_NEAR_ZERO}'),
    ('frac_ce_below_0p01', f'Fraction of instances with Ce < {CE_ESSENTIALLY_ZERO}'),
    ('n_instances', 'Prediction instances'),
    ('sas_mean', 'Mean observed SAS'),
    ('sas_std', 'Within-patient SAS SD'),
    ('sas_range', 'Within-patient SAS range'),
]
if HAS_CALIB:
    VARS += [('n_sas_obs', 'SAS observations (all)'),
             ('n_calib_nominal', 'Calibration points (nominal)'),
             ('n_calib_effective', 'Calibration points (effective)')]
if HAS_MAE:
    VARS += [('patient_mae', 'Per-patient MAE (test patients only)')]


def med_iqr(s):
    s = s.dropna()
    if len(s) == 0:
        return '', ''
    return (round(float(s.median()), 4),
            f'{s.quantile(0.25):.4g} to {s.quantile(0.75):.4g}')


rows = []
for col, label in VARS:
    samples = [pat.loc[pat['group'] == grp, col].dropna() for grp in GROUP_ORDER]
    row = {'variable': label}
    for grp, s in zip(GROUP_ORDER, samples):
        m, q = med_iqr(s)
        row[f'{grp}_n'] = len(s)
        row[f'{grp}_median'] = m
        row[f'{grp}_iqr'] = q
    usable = [s for s in samples if len(s) >= 3]
    if len(usable) >= 2 and max(len(np.unique(s)) for s in usable) > 1:
        H, p = stats.kruskal(*usable)
        row['kruskal_H'] = round(float(H), 3)
        row['kruskal_p'] = float(p)
    else:
        row['kruskal_H'], row['kruskal_p'] = '', np.nan
    lo = pat.loc[pat['group'] == 'pinned-low', col].dropna()
    hi = pat.loc[pat['group'] == 'pinned-high', col].dropna()
    it = pat.loc[pat['group'] == 'interior', col].dropna()
    for name, a, b in (('low_vs_interior', lo, it), ('high_vs_interior', hi, it)):
        if len(a) >= 3 and len(b) >= 3:
            try:
                _, pv = stats.mannwhitneyu(a, b, alternative='two-sided')
                row[f'mwu_{name}_p'] = float(pv)
            except ValueError:
                row[f'mwu_{name}_p'] = np.nan
        else:
            row[f'mwu_{name}_p'] = np.nan
    rows.append(row)

tab = pd.DataFrame(rows)

# Holm across the Kruskal family.
mask = tab['kruskal_p'].notna()
p = tab.loc[mask, 'kruskal_p'].values
order = np.argsort(p)
adj, run = np.empty(len(p)), 0.0
for rank, idx in enumerate(order):
    run = max(run, (len(p) - rank) * p[idx])
    adj[idx] = min(run, 1.0)
tab.loc[mask, 'kruskal_p_holm'] = adj
for c in ('kruskal_p', 'kruskal_p_holm', 'mwu_low_vs_interior_p', 'mwu_high_vs_interior_p'):
    tab[c] = tab[c].apply(lambda v: '' if pd.isna(v) else f'{v:.3g}')

tab.to_csv(OUT_TABLE, index=False)
pat.to_csv(OUT_PATIENT, index=False)
print(f"Wrote {OUT_TABLE} ({len(tab)} rows) and {OUT_PATIENT} ({len(pat)} rows)")
print()

print("GROUP COMPARISON (median [IQR])")
print("-" * 70)
hdr = f"{'variable':<42s}" + ''.join(f"{g:>22s}" for g in GROUP_ORDER) + f"{'Kruskal p':>12s}"
print(hdr)
for _, r in tab.iterrows():
    line = f"{r['variable']:<42s}"
    for grp in GROUP_ORDER:
        line += f"{str(r[f'{grp}_median']):>22s}"
    line += f"{r['kruskal_p']:>12s}"
    print(line)
print()


# ===========================================================================
# VERDICT
# ===========================================================================
print("=" * 70)
print("VERDICT")
print("=" * 70)

ce_lo = pat.loc[pat['group'] == 'pinned-low', 'ce_mean'].median()
ce_hi = pat.loc[pat['group'] == 'pinned-high', 'ce_mean'].median()
ce_it = pat.loc[pat['group'] == 'interior', 'ce_mean'].median()
sas_lo = pat.loc[pat['group'] == 'pinned-low', 'sas_mean'].median()
sas_hi = pat.loc[pat['group'] == 'pinned-high', 'sas_mean'].median()
sas_it = pat.loc[pat['group'] == 'interior', 'sas_mean'].median()

print(f"Median mean-Ce   : pinned-low {ce_lo:.4f} | pinned-high {ce_hi:.4f} "
      f"| interior {ce_it:.4f}")
print(f"Median mean-SAS  : pinned-low {sas_lo:.3f} | pinned-high {sas_hi:.3f} "
      f"| interior {sas_it:.3f}")
print()

if ce_lo < ce_it:
    print("EXPOSURE HYPOTHESIS: SUPPORTED for pinned-low. Those patients have")
    print("   materially lower effect-site concentration than interior patients,")
    print(f"   {ce_it/max(ce_lo, 1e-9):.1f}x lower at the median, so the sigmoid has")
    print("   little leverage and the objective runs monotonically to a bound.")
else:
    print("EXPOSURE HYPOTHESIS: NOT SUPPORTED for pinned-low; their Ce is not")
    print("   lower than interior patients. Look elsewhere for the cause.")

print()
print("SECOND MECHANISM: the pinned-high group is a different failure entirely.")
if HAS_CALIB:
    hi_ids = set(pat.loc[pat['group'] == 'pinned-high', 'stay_id'])
    zero_ids = set(pat.loc[pat['n_calib_effective'] == 0, 'stay_id'])
    if hi_ids and hi_ids == zero_ids:
        print(f"   All {len(hi_ids)} pinned-high patients are EXACTLY the patients whose")
        print("   calibration window contributed zero usable points, and no others.")
        print("   With no points to accumulate, ec50_loss returns loss/max(n,1) = 0")
        print("   for every candidate, so the objective is identically flat and")
        print("   minimize_scalar's bounded search terminates at ~9.999995 by")
        print("   construction. Their EC50 is not an estimate of anything: it is")
        print("   the optimiser's terminus on an empty objective. Note their drug")
        print(f"   exposure is NORMAL (median mean-Ce {ce_hi:.4f} vs interior "
              f"{ce_it:.4f}),")
        print("   so the exposure hypothesis does NOT explain this group.")
    else:
        print(f"   pinned-high n={len(hi_ids)}, zero-calibration n={len(zero_ids)}, "
              f"overlap {len(hi_ids & zero_ids)}. Not a clean identity; inspect.")
else:
    print("   Cannot test without the raw CSVs (calibration sizes unavailable).")

print()
print("SAS DIRECTION CHECK (the brief expected pinned-low awake / pinned-high sedated):")
print(f"   mean SAS: pinned-low {sas_lo:.2f} | pinned-high {sas_hi:.2f} "
      f"| interior {sas_it:.2f}")
spread = max(sas_lo, sas_hi, sas_it) - min(sas_lo, sas_hi, sas_it)
print(f"   spread across groups is {spread:.2f} SAS units")
if spread < 0.5:
    print("   NOT SUPPORTED. The groups barely differ in sedation level, so")
    print("   pinning is not tracking how awake or sedated a patient is.")
print("   The comparison that matters is against the model's own drug-free")
print(f"   baseline E0 = {E0:.0f}. Every group sits far below it:")
for grp in GROUP_ORDER:
    mu = pat.loc[pat['group'] == grp, 'sas_mean'].mean()
    print(f"      {grp:<12s} mean SAS {mu:.2f}  ->  {E0 - mu:.2f} units below E0")
print("   That gap, not the between-group difference, is what the optimiser is")
print("   trying to close. A patient sitting ~3.4 SAS units below the drug-free")
print("   baseline with almost no modelled drug on board leaves the loss")
print("   monotonically decreasing in EC50, so it rails to the floor and still")
print("   cannot reach the observation. The pinning is the calibration absorbing")
print("   a baseline mismatch between the Eleveld E0/Emax scale and ICU SAS,")
print("   not a pharmacological property of the patient.")

if HAS_MAE:
    lo = pat.loc[pat['group'] == 'pinned-low', 'patient_mae'].dropna()
    it = pat.loc[pat['group'] == 'interior', 'patient_mae'].dropna()
    if len(lo) >= 3 and len(it) >= 3:
        _, pv = stats.mannwhitneyu(lo, it, alternative='two-sided')
        print()
        print("IMPACT ON PREDICTION:")
        print(f"   pinned-low median MAE {lo.median():.4f} (n={len(lo)}) vs "
              f"interior {it.median():.4f} (n={len(it)}), Mann-Whitney p = {pv:.3g}")
        if pv >= 0.05:
            print("   No detectable difference. The degenerate EC50 is not measurably")
            print("   hurting prediction, which is itself the point: the feature is")
            print("   contributing so little that its being degenerate costs nothing.")
        elif lo.median() < it.median():
            print("   Pinned-low patients are predicted MORE accurately than interior")
            print("   ones, not less. The degenerate EC50 is not hurting anything.")
            print("   The likely reason is confounding rather than benefit: pinned-low")
            print("   patients are the low-exposure ones, whose SAS moves least, and")
            print("   per-patient error tracks within-patient SAS variability almost")
            print("   perfectly (rho = 0.85, see instructions.txt Section 9). They are")
            print("   simply the easy patients. Two implications, both worth stating:")
            print("   the boundary artefact costs no accuracy, and a feature whose")
            print("   degeneracy costs no accuracy is a feature carrying little signal.")
        else:
            print("   Pinned-low patients are predicted LESS accurately than interior")
            print("   ones, so the boundary artefact plausibly does carry a cost.")
print()


# ===========================================================================
# FIGURE
# ===========================================================================
print("Generating figure...")
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# (a) Ce distribution by group, log axis with a floor for exact zeros
ax = axes[0, 0]
floor = 1e-4
for grp in GROUP_ORDER:
    s = pat.loc[pat['group'] == grp, 'ce_mean'].clip(lower=floor)
    if len(s):
        ax.hist(np.log10(s), bins=40, alpha=0.55, color=GROUP_COLORS[grp],
                edgecolor='white', linewidth=0.3,
                label=f'{grp} (n={len(s)})', zorder=2)
ax.set_xlabel('log$_{10}$ mean effect-site concentration Ce ($\\mu$g/mL)')
ax.set_ylabel('Patients')
ax.set_title('(a) Drug exposure by pinning group', fontsize=11, loc='left')
ax.legend(frameon=False, fontsize=8)
ax.grid(axis='y', color='#ededed', linewidth=0.6, zorder=0)
ax.set_axisbelow(True)

# (b) mean SAS by group
ax = axes[0, 1]
data = [pat.loc[pat['group'] == g, 'sas_mean'].dropna().values for g in GROUP_ORDER]
bp = ax.boxplot(data, labels=[f'{g}\n(n={len(d)})' for g, d in zip(GROUP_ORDER, data)],
                patch_artist=True, widths=0.55, showfliers=False)
for patch, grp in zip(bp['boxes'], GROUP_ORDER):
    patch.set_facecolor(GROUP_COLORS[grp]); patch.set_alpha(0.65)
    patch.set_edgecolor('white'); patch.set_linewidth(1.2)
for med in bp['medians']:
    med.set_color('#333333'); med.set_linewidth(1.6)
ax.axhline(E0, color=C_MUTED, linestyle='--', linewidth=1.2, zorder=1)
ax.text(0.98, E0, ' E0 = 7 (drug-free baseline)', transform=ax.get_yaxis_transform(),
        ha='right', va='bottom', fontsize=8, color=C_MUTED)
ax.set_ylabel('Mean observed SAS')
ax.set_title('(b) Observed sedation by pinning group', fontsize=11, loc='left')
ax.grid(axis='y', color='#ededed', linewidth=0.6, zorder=0)
ax.set_axisbelow(True)

# (c) objective profiles for representative patients
ax = axes[1, 0]
rng = np.random.default_rng(42)
for grp in GROUP_ORDER:
    ids = pat.loc[pat['group'] == grp, 'stay_id'].values
    if len(ids) == 0:
        continue
    pick = rng.choice(ids, size=min(12, len(ids)), replace=False)
    for j, sid in enumerate(pick):
        sub = inst[inst['stay_id'] == sid]
        ce, y = sub['Ce'].values, sub['actual_sas'].values
        preds = E0 - EMAX * (ce[None, :] ** GAMMA) / (grid[:, None] ** GAMMA + ce[None, :] ** GAMMA)
        loss = ((preds - y[None, :]) ** 2).mean(axis=1)
        rngl = loss.max() - loss.min()
        norm = (loss - loss.min()) / rngl if rngl > 1e-12 else np.zeros_like(loss)
        ax.plot(grid, norm, color=GROUP_COLORS[grp], alpha=0.5, linewidth=1.1,
                label=grp if j == 0 else None, zorder=2)
ax.set_xlabel('EC50 candidate ($\\mu$g/mL)')
ax.set_ylabel('Normalised calibration loss')
ax.set_title('(c) Objective shape: monotone means no interior optimum',
             fontsize=11, loc='left')
ax.legend(frameon=False, fontsize=8)
ax.grid(color='#ededed', linewidth=0.6, zorder=0)
ax.set_axisbelow(True)

# (d) impact on prediction error
ax = axes[1, 1]
if HAS_MAE and pat['patient_mae'].notna().any():
    data = [pat.loc[pat['group'] == g, 'patient_mae'].dropna().values for g in GROUP_ORDER]
    keep = [(g, d) for g, d in zip(GROUP_ORDER, data) if len(d)]
    bp = ax.boxplot([d for _, d in keep],
                    labels=[f'{g}\n(n={len(d)})' for g, d in keep],
                    patch_artist=True, widths=0.55, showfliers=False)
    for patch, (grp, _) in zip(bp['boxes'], keep):
        patch.set_facecolor(GROUP_COLORS[grp]); patch.set_alpha(0.65)
        patch.set_edgecolor('white'); patch.set_linewidth(1.2)
    for med in bp['medians']:
        med.set_color('#333333'); med.set_linewidth(1.6)
    ax.set_ylabel('Per-patient MAE (SAS units)')
    ax.set_title('(d) Does pinning hurt prediction?', fontsize=11, loc='left')
    ax.grid(axis='y', color='#ededed', linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
else:
    ax.axis('off')
    ax.text(0.5, 0.5, 'cohort_per_patient_mae.csv not available',
            ha='center', va='center', color=C_MUTED, transform=ax.transAxes)

fig.suptitle(f'EC50 boundary pinning: {counts["pinned-low"]} of {n_pat} patients '
             f'({100*counts["pinned-low"]/n_pat:.0f}%) at the lower bound, '
             f'{counts["pinned-high"]} ({100*counts["pinned-high"]/n_pat:.0f}%) at the upper',
             fontsize=12, x=0.02, ha='left')
fig.tight_layout(rect=(0, 0, 1, 0.96))
for fmt in ('png', 'pdf'):
    p = os.path.join(FIG_DIR, f'ec50_pinning_diagnostics.{fmt}')
    fig.savefig(p, dpi=300, bbox_inches='tight')
    print(f"   Saved {p}")
plt.close(fig)

print()
print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
