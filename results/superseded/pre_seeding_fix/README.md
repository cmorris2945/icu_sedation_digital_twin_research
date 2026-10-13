# Superseded: runs made before the Keras seeding fix

Nothing in this folder is a current result. Do not cite these numbers.

## The defect these files predate

Until 2026-10-12 the LSTM scripts seeded each run with `np.random.seed(seed)`
and `tf.random.set_seed(seed)` only. Keras 3 draws its weight-initializer and
dropout seeds from Python's `random` module, which those two calls do not seed,
so the initial weights were different every time a "seed" was run. The runs
were not reproducible by seed, and runs paired by seed did not share an
initialization.

The fix is `keras.utils.set_random_seed(seed)` at the top of `run_lstm`, in
`src/02b_run_capacity_sweep.py` and `src/02c_run_arm_b_clean_4unit.py`.

## What is here

All three files were written by `src/02c_run_arm_b_clean_4unit.py` on
2026-10-12 between 08:30 and 09:55, before the fix (the script was last edited
at 10:10 the same day). They were originally written to
`results/processed/` and the repository root and were moved here on 2026-10-13.

| file | contents |
|---|---|
| `arm_b_clean_4unit_runs.csv` | 64 unseeded 4-unit LSTM runs: 20 `clean` (670 fitted patients), 20 `rerun` and 20 `rerun2` (all 715 Arm B patients), 2 `probe_a` and 2 `probe_b` |
| `arm_b_clean_4unit_comparison.csv` | FULL_HYBRID versus PURE_ML summaries built from those runs and from the 10-initialization Arm B file |
| `arm_b_clean_4unit.log` | console log of the four sessions |

The rows carry no `seeding` column. That is how the pre-fix runs are told
apart from the seeded ones.

The `probe_a` and `probe_b` rows are the evidence for the defect: the same
cohort, configuration and seed, trained twice in the same environment, gave
different results (seed 46, PURE_ML: MAE 0.3843 and 0.3708).

## What replaces them

The seeded 30-initialization runs of the same 670-patient cohort:

- `capacity_sweep_arm_b_clean_30seed_results.csv` (repository root), from
  `python src/02b_run_capacity_sweep.py --arm interior_clean --units 4 --n-seeds 30`
- summarised in `results/processed/capacity_summary_30seed.md`

See `instructions.txt` Section 12 for the full provenance.
