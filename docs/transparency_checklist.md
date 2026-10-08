# Research Transparency and Reproducibility Checklist
## ICU Sedation Digital Twin Project
**Last Updated:** October 7, 2026

This document tracks all methodological decisions for full transparency and reproducibility.

---

## 1. Data Source

| Item | Value | Notes |
|------|-------|-------|
| Database | MIMIC-IV v3.1 | PhysioNet credentialed access |
| Access Method | Google BigQuery | Project: physionet-data |
| IRB/DUA | [Your IRB number] | [Date approved] |
| Extraction Date | 2026-10-07 | Fresh BigQuery pulls using the committed repository queries; reproducible for credentialed MIMIC-IV users. |
| Patient Cohort Size | 1,490 patients (current) | May change with comorbidity focus |

### SQL Queries Used
All three runnable queries are saved in `queries/` in this repository:
`extract_propofol_patients.sql`, `extract_propofol_infusions.sql`, and
`extract_sas_observations.sql`. The operator reports that the three CSV extracts
were freshly pulled from BigQuery on 2026-10-07 using these committed queries.
Re-extraction is reproducible for credentialed MIMIC-IV v3.1 users. Job IDs for
the 2026-10-07 refresh were not recorded and remain pending.

---

## 2. Cohort Selection Criteria

**CORRECTED 2026-09-02** against the recovered BigQuery SQL, and re-extracted
2026-10-07 using those repository queries. The previous version of this table stated an age filter that was
never applied and a SAS threshold of ≥2 assessments, which matched neither the
extraction filter (≥15) nor the modeling requirement (≥11).

| Criterion | Value | Justification |
|-----------|-------|---------------|
| Age | No filter applied | MIMIC-IV contains adult patients only by construction. Cohort is de facto adult (observed range 18–93), but no `age >= 18` predicate exists in any query. Do not describe one. |
| Propofol administration | ≥5 infusion events, itemid 222168, rate > 0 | Extraction filter. Biases the cohort toward sustained infusions; state this as a selection effect. |
| SAS assessments | ≥15 per ICU stay | Extraction filter (itemid 223753), `valuenum` restricted to 1–7. |
| SAS assessments (modeling) | ≥11 per ICU stay | Separate downstream requirement in `01_build_features.py` (10 history steps + 1 prediction target). Never binds, since the extraction filter already demands 15. **Not** the cohort inclusion criterion. |
| Simulation window | First 72 h after first propofol event | A second, downstream observation filter, not an SQL inclusion rule: discards 62.9% of extracted SAS observations (55.0% after the window and 7.9% before the first infusion). |
| ICU stay duration | No explicit filter | |
| Exclusions | None explicit | No pregnancy or diagnosis-based exclusions were applied. |

Age is computed, not read raw, using MIMIC's anchor-year adjustment:
`DATETIME_DIFF(i.intime, DATETIME(p.anchor_year, 1, 1), YEAR) + p.anchor_age`

Weight and height are `LEFT JOIN`ed from `mimiciv_3_1_derived.first_day_weight`
and `first_day_height`, so both may be NULL (weight 1.2%, height 26.7% of the
extract). `01_build_features.py` substitutes 75 kg and 170 cm respectively.

Cohort sizes: 1,490 patients extracted → 1,489 in the analysis dataset →
30,931 prediction instances. One patient yields zero prediction instances
because only 4 of its 254 SAS observations fall inside the 72-hour window;
this patient is silently absent from the instance-level cache. Weight was
missing for 18/1,490 patients (1.2%) and defaulted to 75 kg; height was missing
for 398/1,490 (26.7%) and defaulted to 170 cm.

---

## 3. Feature Engineering

### Raw Clinical Features (10 features)
| Feature | Source | Computation | Missing Data Handling |
|---------|--------|-------------|----------------------|
| age | patients table | Direct | Required (no missing) |
| gender | patients table | Binary encode (M=1, F=0) | Required |
| weight | [table] | [method] | [method] |
| height | [table] | [method] | [method] |
| sas_last | sedation scores | Most recent SAS before prediction time | Required |
| sas_mean | sedation scores | Mean of all prior SAS | Forward fill if <2 observations |
| sas_std | sedation scores | Std dev of all prior SAS | Set to 0 if <2 observations |
| sas_trend | sedation scores | Linear slope of last 3 SAS | Set to 0 if <3 observations |
| propofol_rate | medication table | Current infusion rate (mg/kg/hr) | [method] |
| time_hours | timestamps | Hours since ICU admission | Computed |

### Mechanistic Features (5 features)
| Feature | Source | Computation | Reference |
|---------|--------|-------------|-----------|
| Ce | Eleveld model | Effect-site concentration via 3-compartment + ke0 | Eleveld 2018, BJA |
| Cp | Eleveld model | Plasma concentration (central compartment) | Eleveld 2018, BJA |
| EC50_calibrated | Patient-specific fit | Fitted to patient's prior SAS-propofol pairs | See Section 4 |
| mech_pred_pop | Eleveld model | Predicted SAS from population EC50 | Sigmoid Emax |
| mech_pred_calib | Eleveld model | Predicted SAS from calibrated EC50 | Sigmoid Emax |

---

## 4. Eleveld PK/PD Model Implementation

### Model Reference
Eleveld DJ, Colin P, Absalom AR, Struys MMRF. Pharmacokinetic–pharmacodynamic model for propofol for broad application in anaesthesia and sedation. British Journal of Anaesthesia. 2018;120(5):942-959. doi:10.1016/j.bja.2018.01.018

### Population Parameters Used
| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| V1 | 6.28 | L | Central compartment volume |
| V2 | 25.5 | L | Rapid peripheral volume |
| V3 | 273 | L | Slow peripheral volume |
| CL | 1.79 | L/min | Metabolic clearance |
| Q2 | 1.75 | L/min | Inter-compartmental clearance (rapid) |
| Q3 | 1.11 | L/min | Inter-compartmental clearance (slow) |
| ke0 | 0.146 | min⁻¹ | Effect-site equilibration rate |
| EC50_pop | 3.08 | μg/mL | Population effect-site concentration for 50% effect |
| gamma | 1.47 | - | Hill coefficient (sigmoidicity) |

### Covariate Adjustments
[Document any weight/age/sex adjustments applied]

### ODE Solver
| Setting | Value |
|---------|-------|
| Method | scipy.integrate.odeint |
| Time step | 1 minute |
| Initial conditions | All compartments = 0 |

### EC50 Calibration Method
| Setting | Value |
|---------|-------|
| Optimization | scipy.optimize.minimize_scalar |
| Method | Brent's method (bounded) |
| Bounds | [0.5, 10.0] μg/mL |
| Objective | Mean Squared Error between predicted and observed SAS |
| Calibration observations | First `max(10, floor(n_SAS/2))` ordered SAS observations; only timestamps inside the 72-hour simulated trajectory contribute |
| Empty effective calibration window | Return population EC50 (3.08 μg/mL) without optimization |

**Zero-observation bug correction (2026-10-07).** Previously, the objective
divided accumulated squared error by `max(n, 1)`. If no calibration timestamp
fell within the simulated trajectory, the objective was identically zero and
bounded `minimize_scalar` returned approximately 9.999995 for 45 patients.
The fit encoded no patient information. The corrected implementation explicitly
returns population EC50 for zero usable points. This changes the feature matrix;
results from the old and corrected caches are distinct analyses and must not be
combined. See `instructions.txt` Section 11.6 for filenames and run provenance.

---

## 5. Model Configurations

### 5.1 XGBoost
| Hyperparameter | Value | Justification |
|----------------|-------|---------------|
| n_estimators | 100 | Standard default, sufficient for convergence |
| max_depth | 6 | Prevents overfitting while allowing complex interactions |
| learning_rate | 0.1 | Standard default |
| subsample | 1.0 | Default (no row subsampling) |
| colsample_bytree | 1.0 | Default (no column subsampling) |
| random_state | 42 | Reproducibility |
| objective | reg:squarederror | Regression task |

**Citation:** Chen T, Guestrin C. XGBoost: A Scalable Tree Boosting System. KDD 2016.

### 5.2 Random Forest
| Hyperparameter | Value | Justification |
|----------------|-------|---------------|
| n_estimators | 100 | Standard default |
| max_depth | 12 | Allows deeper trees than XGBoost (no boosting regularization) |
| min_samples_split | 2 | Default |
| min_samples_leaf | 1 | Default |
| random_state | 42 | Reproducibility |

**Citation:** Breiman L. Random Forests. Machine Learning. 2001;45(1):5-32.

### 5.3 Support Vector Machine (SVM)
| Hyperparameter | Value | Justification |
|----------------|-------|---------------|
| kernel | rbf | Radial basis function, good for nonlinear relationships |
| C | 1.0 | Default regularization (tested 0.1, 1.0, 10.0) |
| epsilon | 0.1 | Default for SVR |
| gamma | scale | 1/(n_features * X.var()) - adaptive |
| Sample size | 5000 | Computational constraint for kernel methods |

**Citation:** Smola AJ, Schölkopf B. A tutorial on support vector regression. Statistics and Computing. 2004;14(3):199-222.

### 5.4 Multi-Layer Perceptron (MLP)
| Hyperparameter | Value | Justification |
|----------------|-------|---------------|
| hidden_layer_sizes | (64, 32) | Two layers, decreasing width (standard architecture) |
| activation | relu | Standard for hidden layers (Nair & Hinton, 2010) |
| solver | adam | Adaptive learning rates (Kingma & Ba, 2015) |
| learning_rate_init | 0.001 | Adam default |
| alpha | 0.001 | L2 regularization |
| max_iter | 500 | With early stopping |
| early_stopping | True | Prevents overfitting |
| validation_fraction | 0.1 | 10% held out for early stopping |
| random_state | 42 | Reproducibility |

**Citations:**
- Nair V, Hinton GE. Rectified linear units improve restricted Boltzmann machines. ICML 2010.
- Kingma DP, Ba J. Adam: A method for stochastic optimization. ICLR 2015.

### 5.5 LSTM
| Hyperparameter | Value | Justification |
|----------------|-------|---------------|
| sequence_length | 10 | ~2-4 hours of SAS observations |
| lstm_units | 32 | Balance between capacity and overfitting |
| dense_layers | (32, 16) | Post-LSTM processing |
| dropout | 0.2 | Regularization |
| optimizer | Adam | Standard |
| learning_rate | 0.001 | Adam default |
| epochs | 100 | With early stopping |
| batch_size | 64 | Standard |
| early_stopping_patience | 10 | Epochs without improvement |
| validation_split | 0.1 | 10% held out |

**Sequence features:**
- Pure: 10-step SAS history
- Hybrid: 10-step SAS history + 10-step Ce history

**Static features:**
- Pure: age, gender, weight, height, propofol_rate, time_hours (6 features)
- Hybrid: Pure + Ce, Cp, EC50_calibrated, mech_pred_pop, mech_pred_calib (11 features)

**Citation:** Hochreiter S, Schmidhuber J. Long short-term memory. Neural Computation. 1997;9(8):1735-1780.

### 5.6 GRU
[Same as LSTM but with GRU cells]
**Note:** GRU showed training instability in initial experiments (MAE 0.893 vs 0.5-0.6 for others). Requires further tuning.

**Citation:** Cho K, et al. Learning phrase representations using RNN encoder-decoder for statistical machine translation. EMNLP 2014.

---

## 6. Evaluation Methodology

### Train/Test Split
| Setting | Value |
|---------|-------|
| Method | Patient-level split (no patient in both train and test) |
| Train fraction | 80% |
| Test fraction | 20% |
| Random seed | 42 |

### Metrics
| Metric | Formula | Interpretation |
|--------|---------|----------------|
| MAE | Σ|y - ŷ| / n | Average absolute error in SAS units |
| RMSE | √(Σ(y - ŷ)² / n) | Penalizes large errors more |
| R² | 1 - SS_res/SS_tot | Variance explained |

### Statistical Significance
[To be added: bootstrap confidence intervals, paired t-tests, etc.]

---

## 7. Explainability Methods

### SHAP (SHapley Additive exPlanations)
| Setting | Value |
|---------|-------|
| Method (tree models) | TreeExplainer |
| Method (neural networks) | DeepExplainer or GradientExplainer |
| Background samples | 100 (random from training set) |
| Reference | Lundberg SM, Lee SI. A unified approach to interpreting model predictions. NeurIPS 2017. |

### LIME (Local Interpretable Model-agnostic Explanations)
| Setting | Value |
|---------|-------|
| Kernel width | Default (0.75 * sqrt(n_features)) |
| Number of samples | 5000 |
| Reference | Ribeiro MT, Singh S, Guestrin C. "Why should I trust you?": Explaining the predictions of any classifier. KDD 2016. |

---

## 8. Software Environment

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.11.9 | Fresh Windows clone, 2026-10-07 |
| numpy | 2.4.6 | Numerical computing; newer than original 1.26.4 |
| pandas | 3.0.6 | Data manipulation; newer than original 2.x |
| scikit-learn | Installed; exact version pending | ML algorithms |
| xgboost | Installed; exact version pending | Gradient boosting |
| tensorflow | 2.21.0 | Neural networks |
| scipy | 1.17 | Optimization, ODEs |
| shap | Version not recorded for this run | Explainability |
| lime | Version not recorded for this run | Explainability |
| matplotlib | Version not recorded for this run | Visualization |

**Reproducibility check, 2026-10-07: PASSED.** On a fresh Windows clone,
`01_build_features.py` loaded 1,490 patients, 31,235 propofol events, and
113,536 SAS observations, then reproduced 30,931 instances from 1,489 patients
and a 9.54 MB cache. The post-fix build completed successfully with 1,490
processed and zero skipped. These cohort totals matched the original feature
build despite newer pandas and NumPy versions. Read-back classification of the
corrected cache found 774 pinned-low, zero pinned-high, 715 numerically
interior (670 fitted interior plus 45 population EC50 fallbacks at 3.08).
Previously published model results came from the pre-fix pipeline; the original
pre-fix pickle was overwritten before it could be preserved in this session.
A post-fix snapshot is retained locally only. Neither cache is to be committed.

---

## 9. Reproducibility Checklist

- [ ] All random seeds documented and set
- [ ] Exact data extraction queries saved
- [ ] All preprocessing steps documented
- [ ] All hyperparameters documented with justification
- [ ] Code available in repository
- [ ] Results include confidence intervals
- [ ] Negative results reported honestly
- [ ] Limitations acknowledged

### Three-Arm Capacity Sweep (post-fix cache)

The rerun tests 4, 8, 16, and 32 LSTM units, all five existing feature
configurations, and ten seeds (42–51) per cell. Architecture and training
settings match `src/02b_run_capacity_sweep.py`: 50 epochs, batch size 64,
validation split 0.15, and an 80/20 patient-level train/test split with
`random_state=42`.

| Arm | Cohort | Purpose |
|-----|--------|---------|
| A | Full corrected cohort | Reference population |
| B | Interior EC50 patients only | Tests feature benefit after removing boundary-pinned calibration values |
| C | Seed-42 random sample from the full cohort, matched to B's patient count | Controls for reduced sample size, isolating EC50 composition from dataset-size effects |

Arm C is essential: comparing B only with A would confound EC50-based patient
selection with the reduced number of training patients. The primary contrast
is Pure ML versus Full Hybrid degradation at each capacity in B compared with
the same contrast in C. Results, Spearman monotonicity, and exponential decay
fit are pending until all 600 runs complete. Do not report estimates before
then.

---

## 10. Version History

| Date | Change | Author |
|------|--------|--------|
| 2026-04-19 | Initial creation | Chris Morris |
| 2026-10-07 | Fresh BigQuery extracts, corrected cohort/provenance details, EC50 empty-window fallback, environment and three-arm capacity design | Chris Morris |
