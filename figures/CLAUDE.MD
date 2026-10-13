# ICU Sedation Digital Twin — read this first

## Where you are
Live repo: `icu_sedation_digital_twin_research/` (nested).
The outer `icu-sedation-research/` folder is a STALE SEPTEMBER COPY. Never work in it.

## What the paper now claims
The manuscript was rewritten on 2026-10-11 and is titled
"Patient-Specific Pharmacodynamic Parameters Are Not Identifiable from
Routine ICU Sedation Charting."

The live manuscript is in **Overleaf**, not this repo. `paper/main.tex` at HEAD is
the superseded September draft — do not treat it as current.

**The capacity finding is retracted.** Any text claiming mechanistic benefit
decays monotonically with LSTM size (10.4% / 4.5% / 0.7% / 0.6%, 4-unit MAE
0.3514) is from the pre-fix run at commit `2fa4c4c` and is an artifact. Do not
report it as a result.

## Current findings
- Eleveld scale anchors do not transfer to ICU SAS charting. Recalibrated on
  training patients: E0 = 3.74 (95% CI 3.68–3.79), Emax = 1.47 (1.20–1.74) vs
  published 7 and 6. Propofol explains ~9% of SAS variance.
- Patient-specific EC50 is NOT recoverable. Personalized loses to population
  under every anchor choice and all three corrected refit schemes.
- Mechanistic features give a small MAE improvement over PURE_ML at 4 units.
  Source: the seeded 30-initialization sweep (seeds 42–71),
  `results/processed/capacity_summary_30seed.md`. FULL_HYBRID vs PURE_ML:
  - Arm B (715): 0.3544 vs 0.3603, +1.66% (95% interval +0.54% to +2.72%),
    Wilcoxon p = 0.002, Holm 0.005.
  - Arm B clean (670 fitted): 0.3495 vs 0.3634, +3.98% (+2.94% to +5.07%),
    p = 1.64e-07. The result holds, and is larger, without the 45 defaults.
  - Arm A (1,489): 0.3668 vs 0.3746, +2.15% (+0.26% to +4.13%), p = 0.019,
    Holm 0.074. Not significant after correction.
  - At 16 and 32 units no arm shows a difference after Holm correction.
  The earlier "6.5% (0.347 vs 0.370, Holm 0.023)" for Arm B came from 10
  unseeded runs and is superseded. Do not report it.
- Persistence (SAS unchanged) scores MAE 0.2866 and beats every model.

## The arms
- Arm A — full cohort, 1,489 patients
- Arm B — 715 patients classified "interior"; actually 670 genuinely fitted
  plus 45 assigned the population default 3.08. NOT a clean cohort.
- Arm B clean — the 670 fitted patients, same train/test assignment as Arm B
  with the 45 dropped (535 train, 135 test). 4 units only, all five
  configurations.
- Arm C — 715 random patients, size-matched control. Run at 10 unseeded
  initializations only; not part of the 30-initialization sweep.

## The EC50 defect
`ec50_loss` normalized by `max(n, 1)`, returning 0 for every candidate when a
patient had no usable calibration observations. Bounded minimization then
terminated at the upper bound 9.999995. Affected exactly the 45 patients with
empty calibration windows. Those 45 shared an out-of-range constant that the
model exploited as a patient identifier, which manufactured the capacity result.
Fixed at `a60be12`; corrected routine returns the population value.

Post-fix classification: 774 pinned-low, 0 pinned-high, 715 "interior".

## The Keras seeding defect
Until 2026-10-12 the LSTM scripts called `np.random.seed` and
`tf.random.set_seed` only. Keras 3 takes its initializer and dropout seeds from
Python's `random`, so a seed did not fix the initial weights and the same seed
trained twice gave different results. Fixed with
`keras.utils.set_random_seed(seed)`. Seeded rows carry a `seeding` column;
rows without it are pre-fix.

## Which result files to trust
There are two defects, so there are three classes of file.

**Current — after both fixes.** Use these:
- `capacity_sweep_arm_a_full_30seed_results.csv`
- `capacity_sweep_arm_b_interior_30seed_results.csv`
- `capacity_sweep_arm_b_clean_30seed_results.csv`
- `results/processed/capacity_summary_30seed.{md,csv}`,
  `capacity_comparisons_30seed.csv`

Provenance for every number in them is in `instructions.txt` Section 12.

**After the EC50 fix, before the seeding fix.** Built on the corrected cache,
but not reproducible by seed. Never pool with the 30-seed rows and never cite
as a result:
- `capacity_sweep_arm_a_full_results.csv`,
  `capacity_sweep_arm_b_interior_results.csv`,
  `capacity_sweep_arm_c_matched_results.csv` (10 initializations, incomplete)
- `results/processed/capacity_summary.md`, `capacity_summary_table.csv`
- everything under `results/superseded/pre_seeding_fix/`

`capacity_sweep_arm_a_full_results.csv` is NOT pre-EC50-fix: its first session
started on 2026-10-07 after the corrected cache was built, and every session
log prints "pinned-high: 0". An earlier version of this file listed it as
pre-fix; that was wrong.

**Before the EC50 fix.** Never use as a source of truth:
- `capacity_sweep_results.csv`, `capacity_sweep_summary.csv`
- `ablation_results.csv`, `summary_statistics.csv`, `degradation_analysis.csv`
- `ec50_diagnostics.csv` (Sep 3)

`04_make_figures.py` refuses to read `degradation_analysis.csv` by design.

## Hard rules
- **Never commit patient-level data.** No `stay_id`, `subject_id` or `hadm_id`
  columns, no per-observation values, no `.pkl`. MIMIC-IV data use agreement.
- Never fabricate, estimate or interpolate a number. Say what is missing.
- Never cite a pre-fix output as evidence about post-fix behavior.
- Report results whichever way they come out.
- Show diffs before committing.

## Conventions
- **The two summaries use opposite signs.** In `capacity_summary_30seed.md` a
  POSITIVE percent difference means the mechanistic configuration is BETTER
  than PURE_ML. In the older `capacity_summary.md` a NEGATIVE percent change
  means the mechanistic model is BETTER.
- Two kinds of SD are reported and they are not comparable: spread across
  test-instance absolute errors (deterministic methods) and spread across the
  run-level MAEs (LSTM; 30 runs in the current sweep, 10 in the older files).
  Keep them in separate columns.

## Open items
1. Regenerate `ec50_diagnostics` and the per-patient MAE table from post-fix
   data.
2. The 10-initialization three-arm design was narrowed from 600 to 366 runs
   and `03c` cannot run on it. Arm C has no seeded runs, so the pre-registered
   Arm B minus Arm C contrast has no seeded result.
3. No 30-initialization run has been retrained to test whether final MAE, and
   not only the initial weights, is reproducible by seed.