# Project inventory and orientation (live project)

Working directory (verified):
`C:\Users\chris' pc\icu-sedation-research\icu_sedation_digital_twin_research`

Confirmed presence:
`src/02b_run_capacity_sweep.py` exists in this directory: YES

## Which tree is the live project?

The live project is the nested directory:
`C:\Users\chris' pc\icu-sedation-research\icu_sedation_digital_twin_research`

Evidence from git:
- `git -C "C:\Users\chris' pc\icu-sedation-research\icu_sedation_digital_twin_research" rev-parse --show-toplevel` resolves to that exact nested directory.
- `git status` in the nested directory shows modified tracked files and untracked files in that repo, including:
  - `src/02b_run_capacity_sweep.py`
  - `capacity_sweep_arm_a_full_results.csv`
  - `instructions.txt`
  - `results/`
  - `src/compute_arm_b_mechanistic.py`
  - `src/refit_arm_b_ec50_corrected_anchors.py`
- The nested repo is on `main` and its branch points at `05835ab` (`Three-arm sweep setup, EC50 n=0 fix, partial arm A results`).

By contrast, the outer folder:
`C:\Users\chris' pc\icu-sedation-research`

is not the active repo root for the live project. It is a separate workspace-level directory that contains:
- older root-level output files from the earlier workflow (`ablation_results.csv`, `capacity_sweep_results.csv`, `summary_statistics.csv`, etc.)
- a nested untracked copy of the project directory (`icu_sedation_digital_twin_research/`)
- several stale or earlier-generation logs (`step01.log`, `step02.log`, `step03.log`, `step04.log`)
- a `venv/` directory and a separate status that shows the nested project directory as untracked

This outer folder is best described as a duplicate/stale workspace snapshot rather than the canonical live project. It contains an older draft/state of the project and an untracked nested copy of the current repo, not the active git root itself.

## Repo structure under the live nested project

Top-level folder tree (selected, excluding `.git`, `venv`, and local private data directories):

```text
C:\Users\chris' pc\icu-sedation-research\icu_sedation_digital_twin_research
├── README.md
├── CHANGELOG.md
├── initialize_git.sh
├── instructions.txt
├── ablation_results.csv
├── ablation_results_3seeds.csv
├── cached_features.pkl
├── capacity_sweep_arm_a_full_results.csv
├── capacity_sweep_arm_b_interior_results.csv
├── capacity_sweep_arm_c_matched_results.csv
├── capacity_sweep_results.csv
├── capacity_sweep_summary.csv
├── cohort_baseline_comparison.csv
├── cohort_correlations.csv
├── cohort_per_patient_mae.csv
├── cohort_table1.csv
├── data/
├── degradation_analysis.csv
├── docs/
├── ec50_diagnostics.csv
├── ec50_patient_level.csv
├── environment/
├── figures/
├── initialize_git.sh
├── paper/
├── queries/
├── results/
├── src/
│   ├── 01_build_features.py
│   ├── 02_run_ablation.py
│   ├── 02b_run_capacity_sweep.py
│   ├── 03_analyze_results.py
│   ├── 03b_analyze_capacity_sweep.py
│   ├── 04_make_figures.py
│   ├── 04b_plot_capacity_sweep.py
│   ├── 05_cohort_figures.py
│   ├── 08_ec50_diagnostics.py
│   ├── append_arm_b_capacity_comparisons.py
│   ├── compute_arm_b_mechanistic.py
│   ├── compute_capacity_summary.py
│   ├── compute_population_baseline.py
│   ├── refit_arm_b_ec50_corrected_anchors.py
│   └── lib/
├── summary_statistics.csv
├── three_arm_capacity_sweep.log
├── three_arm_cohort_counts.csv
├── tests/
└── ...
```

## Exact-path confirmation in the live repo

Verified in the nested repo:
- `results/processed/capacity_summary_table.csv` -> FOUND
- `results/processed/capacity_summary.md` -> FOUND
- `results/processed/baseline_comparison.csv` -> FOUND
- `results/processed/baseline_scale_calibration.csv` -> FOUND
- `results/processed/arm_b_mechanistic_comparison.csv` -> FOUND
- `src/compute_capacity_summary.py` -> FOUND
- `src/compute_population_baseline.py` -> FOUND
- `src/compute_arm_b_mechanistic.py` -> FOUND
- `src/refit_arm_b_ec50_corrected_anchors.py` -> FOUND
- `results/processed/manuscript_number_audit.md` -> NOT FOUND at the exact bare path, but the report exists under the processed folder in the earlier outer repo; not present here as the requested bare filename.

## Summary

The nested folder is the actual live project and is the repo that contains the current three-arm sweep work and the exact files the investigation is targeting. The outer folder is a separate, older/stale workspace copy that contains an earlier snapshot and a nested duplicate of the project, not the active git working tree.
