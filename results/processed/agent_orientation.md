# Agent orientation report

## Working directory and confirmation

Working directory:
C:\Users\chris' pc\icu-sedation-research\icu_sedation_digital_twin_research

Confirmed: src/02b_run_capacity_sweep.py exists in this directory: YES

This is the active nested repo. The outer folder one level up is a stale duplicate/workspace snapshot and is not the live project root.

---

## 1) Pipeline in order: what runs when, what each script reads, and what each writes

1. src/01_build_features.py
   - Reads: data/kidney_subgroups_patients.csv, data/kidney_subgroups_propofol.csv, data/kidney_subgroups_sas.csv
   - Writes: cached_features.pkl
   - Purpose: run the Eleveld PK/PD simulation, calibrate each patient's EC50, build prediction instances, and create the feature cache used by all downstream analysis.

2. src/02_run_ablation.py
   - Reads: cached_features.pkl
   - Writes: ablation_results.csv
   - Purpose: 5 feature configurations x (1 XGBoost + 10 LSTM seeds) at 16 LSTM units.

3. src/03_analyze_results.py
   - Reads: ablation_results.csv
   - Writes: summary_statistics.csv, degradation_analysis.csv
   - Purpose: compute per-configuration MAE summaries and degradation percentages relative to FULL_HYBRID.

4. src/04_make_figures.py
   - Reads: degradation_analysis.csv (legacy path)
   - Writes: figures/ablation_main.{png,pdf}, figures/absolute_mae.{png,pdf}
   - Purpose: produce the older 16-unit ablation figures. This is not the active capacity-sweep figure stream.

5. src/02b_run_capacity_sweep.py
   - Reads: cached_features.pkl
   - Writes: capacity_sweep_arm_a_full_results.csv, capacity_sweep_arm_b_interior_results.csv, capacity_sweep_arm_c_matched_results.csv, three_arm_cohort_counts.csv, three_arm_capacity_sweep.log
   - Purpose: 4 LSTM sizes (4, 8, 16, 32) x 5 configurations x 10 seeds across three study arms.

6. src/03b_analyze_capacity_sweep.py
   - Reads: capacity_sweep_results.csv (legacy single-file sweep output)
   - Writes: capacity_sweep_summary.csv
   - Purpose: aggregate the older single-file capacity sweep into a per-size summary. This is a legacy path, not the active three-arm analysis.

7. src/04b_plot_capacity_sweep.py
   - Reads: capacity_sweep_summary.csv
   - Writes: figures/capacity_sweep_main.{png,pdf}, figures/capacity_heatmap.{png,pdf}
   - Purpose: generate the older capacity-sweep figures from the single-file summary.

8. src/03c_analyze_three_arm_capacity_sweep.py
   - Reads: capacity_sweep_arm_a_full_results.csv, capacity_sweep_arm_b_interior_results.csv, capacity_sweep_arm_c_matched_results.csv
   - Writes: three_arm_capacity_summary.csv, three_arm_pure_ml_comparison.csv, three_arm_capacity_trend_stats.csv
   - Purpose: compare the three arms directly and compute Spearman and exponential-decay trend summaries.

9. src/compute_capacity_summary.py
   - Reads: capacity_sweep_arm_a_full_results.csv, capacity_sweep_arm_b_interior_results.csv, capacity_sweep_arm_c_matched_results.csv
   - Writes: results/processed/capacity_summary_table.csv, results/processed/capacity_summary.md
   - Purpose: produce the processed summary used by the current post-fix analysis.

10. src/05_cohort_figures.py
    - Reads: cached_features.pkl, data/kidney_subgroups_patients.csv
    - Writes: cohort_table1.csv, cohort_correlations.csv, cohort_per_patient_mae.csv, figures/cohort/*
    - Purpose: cohort distributions, Table 1, correlations, and per-patient MAE analysis.

11. src/08_ec50_diagnostics.py
    - Reads: cached_features.pkl
    - Writes: ec50_diagnostics.csv, ec50_patient_level.csv, figures/cohort/ec50_pinning_diagnostics.{png,pdf}
    - Purpose: diagnose EC50 boundary pinning and report whether there is a true interior optimum.

12. src/09_ec50_cache_classification.py
    - Reads: cached_features.pkl
    - Writes: console summary only
    - Purpose: summarize how many patients are pinned-low, pinned-high, or interior in the current cache.

---

## 2) Current result files vs superseded files

### Current/active post-fix analysis files

The active workflow is built around the corrected EC50 cache and the three-arm sweep outputs:

- capacity_sweep_arm_a_full_results.csv
- capacity_sweep_arm_b_interior_results.csv
- capacity_sweep_arm_c_matched_results.csv
- results/processed/capacity_summary_table.csv
- results/processed/capacity_summary.md
- results/processed/baseline_comparison.csv
- results/processed/baseline_scale_calibration.csv
- results/processed/arm_b_mechanistic_comparison.csv

These are the files connected to the current fixed workflow and are the ones the active support scripts read.

### Legacy or superseded files

These are historical or legacy artifacts and should not be treated as sources of truth for the fixed-cache analysis:

- capacity_sweep_summary.csv
- capacity_sweep_results.csv
- degradation_analysis.csv
- summary_statistics.csv
- ablation_results_3seeds.csv
- docs/paper_draft.tex
- docs/paper_skeleton.md
- docs/study_guide_v*.md

### Specific note on the three flagged files

The task calls out three specific files:

- capacity_sweep_summary.csv
- capacity_sweep_arm_a_full_results.csv
- degradation_analysis.csv

The first and third are clearly legacy pre-fix/stale artifacts. `capacity_sweep_summary.csv` is the legacy single-file sweep summary from the old 2fa4c4c path. `degradation_analysis.csv` is the legacy 16-unit ablation summary from the pre-fix analysis.

The arm-A file needs a more careful distinction: `capacity_sweep_arm_a_full_results.csv` is not the old single-file `capacity_sweep_results.csv` from the pre-fix branch; it is the Arm A file generated in the corrected three-arm workflow after the EC50 fix. It is therefore part of the current fixed analysis, not part of the stale old root-level sweep summary. That said, it is still not the historical provenance file to use when auditing the pre-fix paper results. It is a corrected-current artifact, not a source of truth for the earlier published result tables.

### Does anything still read the stale files?

Yes. The legacy script reads are still present:

- src/03_analyze_results.py reads `degradation_analysis.csv`
- src/04_make_figures.py reads `degradation_analysis.csv`
- src/03b_analyze_capacity_sweep.py reads `capacity_sweep_summary.csv`
- src/04b_plot_capacity_sweep.py reads `capacity_sweep_summary.csv`
- src/05_cohort_figures.py still references the old 4-unit FULL_HYBRID value from the legacy capacity-sweep summary (`capacity_sweep_summary.csv`)

So the repo is mixed: active analysis has moved to the corrected three-arm files and processed summaries, but stale legacy readers remain in older scripts and comments.

---

## 3) What the three arms are and why the third exists

The three-arm design is implemented in src/02b_run_capacity_sweep.py:

- Arm A: full cohort
  - Includes all patients in the feature cache.
  - This is the broadest reference cohort.

- Arm B: interior-only cohort
  - Includes only patients with numerically interior EC50 values, i.e., not pinned to the optimization bounds.
  - This isolates the patients with a meaningful, data-driven EC50 estimate.

- Arm C: matched cohort
  - Randomly sampled without replacement from the full cohort, with size matched to Arm B.
  - This exists to control for the fact that Arm B has fewer patients; otherwise, differences between A and B might be due simply to smaller sample size rather than the EC50 composition.

This is why the third arm exists: it separates “EC50-interior-only effect” from “smaller-cohort effect.”

---

## 4) The EC50 calibration defect, the fix, and why it matters

The defect was diagnosed and fixed in the repo history.

### What the defect was

The calibration loop in src/01_build_features.py used a bounded optimizer over EC50 values in the interval [0.5, 10.0]. If a patient had no usable calibration timestamps inside the simulated 72-hour window, the objective became flat or effectively zero across the candidate set. In that case, the optimizer could terminate at the search boundary instead of producing a meaningful fitted value.

The script then wrote those boundary-returned values into `EC50_calibrated`, even though they were not patient-specific estimates. This made the model look “personalized” when the value was actually a solver artifact.

### Which commit fixed it

The fix is in commit `a60be12`:

- Commit message: `Fix-empty-EC50-calibration-and-document-capacity-sweep`
- The relevant helper is src/lib/ec50_calibration.py, which returns the population EC50 (`EC50_pop`, 3.08) if there are no usable calibration points.
- The call site in src/01_build_features.py was updated to use that helper.

The repo also shows a follow-up diagnostic commit: `236d596` (`Diagnose EC50 boundary pinning: two separate non-pharmacological causes`), which explains that the issue was non-pharmacological and separated into two patterns: pinned-low due to low exposure and pinned-high due to an empty calibration objective.

### Why it matters

This matters because the paper’s personalization story is built on the EC50 feature. If the value is not a true patient-specific estimate, then downstream mechanistic features are invalid and any model comparison built on those values is not trustworthy.

The repo explicitly records that the old pre-fix cache and results cannot be pooled with the corrected post-fix cache. That is the root reason the older root-level files are not a source of truth.

---

## 5) Contradictions and stale documentation in the repo

The project is currently in a partially inconsistent state. The main contradictions are:

- The repo still contains both legacy ablation outputs and newer three-arm sweep outputs.
- The original `instructions.txt` notes script inventory discrepancies and states that some files are “not in the repo” even though newer scripts and results are now present.
- `paper/main.tex` is a generic scaffold and not the 2026-10-11 rewritten manuscript described in the task note.
- `src/05_cohort_figures.py` still references the legacy 4-unit FULL_HYBRID value from the old summary file.
- `src/03_analyze_results.py` and `src/04_make_figures.py` still use the legacy pre-fix files as their inputs.

In other words, the repo contains both a corrected current pipeline and a large set of legacy artifacts that have not all been retired.

---

## 6) Questions to settle before trusting an analysis here

Before running a final analysis or writing results from this repo, these questions should be answered explicitly:

1. Is the canonical manuscript target the old 16-unit ablation workflow or the corrected three-arm capacity-sweep workflow?
2. Should legacy root-level files like `capacity_sweep_summary.csv`, `capacity_sweep_results.csv`, and `degradation_analysis.csv` be retained only as historical artifacts, or should they be regenerated for side-by-side comparison?
3. Is `results/processed/capacity_summary_table.csv` the authoritative current summary, or is there a separate finalized manuscript table to use instead?
4. Should the team ignore the stale 2fa4c4c-era outputs entirely, or keep them explicitly marked as historical only?

---

## Bottom line

The repo is in a mixed state: the fixed EC50 calibration and the three-arm sweep are the active analysis path, but the older root-level ablation and capacity summary artifacts remain in the working tree and are still referenced by some scripts. The real fix for the empty EC50 objective was commit `a60be12`, and it matters because the personalization claim depends on EC50 values being real patient-level estimates rather than optimizer artifacts. The legacy single-file files like `capacity_sweep_summary.csv` and `degradation_analysis.csv` are not trustworthy for current use, and any analysis should be anchored to the corrected three-arm result set and the processed summaries under `results/processed/` unless a separate manuscript decision explicitly overrides that.
