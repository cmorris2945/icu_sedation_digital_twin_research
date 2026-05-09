# Progress Report: ICU Sedation Digital Twin Project
**Christopher Morris**  
**CIS 6905 Independent Study**  
**Advisor: Dr. Tamer Kahveci**  
**April 12, 2026**

---

## Summary

This report covers two main analyses from our last meeting: (1) the comprehensive kidney subgroup analysis you requested, and (2) testing the hybrid approach (Option D) to see if combining PK/PD with ML actually improves the method.

The short version: the kidney comorbidity angle didn't pan out the way I hoped, but the hybrid approach worked and I think that's the more interesting finding.

---

## 1. Comprehensive Kidney Subgroup Analysis

You asked me to run the EC50 calibration analysis across all kidney comorbidity subgroups, not just the binary "kidney failure yes/no" I had before.

### What I Did

I pulled data from MIMIC-IV for patients with various kidney conditions: AKI, CKD stages 1-5, diabetic nephropathy, hypertensive kidney disease, and dialysis patients. Then I ran the Eleveld PK/PD model with EC50 calibration on each subgroup and compared improvement rates.

### Results

| Subgroup | N Analyzed | Population MAE | Calibrated MAE | Improvement |
|----------|------------|----------------|----------------|-------------|
| No Kidney Disease | 78 | 2.91 ± 0.55 | 2.35 ± 0.74 | 19.6% |
| ANY Kidney Disease | 40 | 3.09 ± 0.60 | 2.50 ± 0.73 | 19.6% |
| AKI | 44 | 3.19 ± 0.64 | 2.75 ± 0.79 | 14.5% |
| CKD Stage 5 / ESRD | 43 | 3.08 ± 0.73 | 2.58 ± 0.85 | 17.3% |
| Hypertensive Kidney | 50 | 2.92 ± 0.65 | 2.33 ± 0.75 | 20.4% |
| Diabetic Nephropathy | 27 | 2.89 ± 0.52 | 2.44 ± 0.69 | 16.0% |
| On Dialysis | 10 | 3.16 ± 0.97 | 2.76 ± 0.95 | 13.1% |

### The Problem

I couldn't replicate the earlier finding that kidney failure patients benefit more from calibration. The t-test comparing no kidney disease vs. tight kidney failure (CKD5/ESRD/dialysis) gave p=0.45 — nowhere near significant.

The earlier analysis with 50+50 hand-selected patients showed kidney patients improving 20% vs 10% for non-kidney (p=0.003). But with the larger random sample, both groups improve about the same (~17-20%). I would think the earlier result was probably a false positive from small sample size.

### What This Means

The good news is that EC50 calibration helps everyone — roughly 15-20% improvement across all patient groups. That's consistent and robust.

The bad news is we can't claim that comorbidities (at least kidney disease) make patients benefit more from personalization. The "comorbidity-aware digital twin" angle seems weaker than I initially thought.

---

## 2. Hybrid Approach (Option D)

This is where things got more interesting. You mentioned that Option D was the most relevant direction — can we actually improve the method by combining PK/PD with ML?

### The Setup

I tested three models:

1. **Pure Mechanistic PK/PD** — Eleveld equations with calibrated EC50
2. **Pure XGBoost** — ML using only raw clinical features (demographics, propofol rates, recent SAS history)
3. **Hybrid XGBoost** — ML using raw clinical features PLUS mechanistic features (effect-site concentration Ce, plasma concentration Cp, calibrated EC50, and the PK/PD model's SAS predictions)

The question: does adding the mechanistic outputs to XGBoost improve predictions beyond what pure XGBoost can achieve?

### Results

| Model | MAE (SAS levels) | RMSE |
|-------|------------------|------|
| Mechanistic PK/PD | 2.51 | 2.82 |
| Pure XGBoost | 0.55 | 0.89 |
| **Hybrid XGBoost** | **0.46** | **0.73** |

The hybrid beat pure XGBoost by about 15%. It seems like the pharmacological information from the PK/PD model is adding something that ML alone can't extract from raw features.

### Feature Importance

Looking at what the hybrid model actually used:

| Feature | Importance | Type |
|---------|------------|------|
| sas_last | 0.703 | Raw clinical |
| sas_mean_6h | 0.048 | Raw clinical |
| sas_mean_4h | 0.035 | Raw clinical |
| mechanistic_pred_pop | 0.020 | Mechanistic |
| current_rate | 0.019 | Raw clinical |
| Ce (effect-site conc) | 0.015 | Mechanistic |

The most recent SAS dominates (which makes sense — sedation levels don't jump around randomly). But the mechanistic prediction shows up in the top features, meaning the PK/PD model is contributing useful signal.

### Why I Think This Matters

This answers the question you raised about XGBoost beating the mechanistic model 2:1. Yes, pure ML is more accurate than pure physiology — but physiology still adds value. The hybrid outperforms both.

I believe this supports the "digital twin" framing because it shows the mechanistic model isn't redundant. It captures something about the drug-effect relationship that improves predictions even when ML has access to all the raw data.

---

## 3. Robustness Test (Sample Size Reduction)

You mentioned testing robustness by reducing sample size 10%, 20%, etc. to see which model degrades faster. I ran this test.

### Results

| Training Data | Pure XGBoost MAE | Hybrid MAE | Hybrid Advantage |
|---------------|------------------|------------|------------------|
| 100% | 0.526 | 0.454 | +13.8% |
| 80% | 0.511 | 0.483 | +5.6% |
| 60% | 0.455 | 0.435 | +4.2% |
| 40% | 0.477 | 0.448 | +6.1% |
| 20% | 0.484 | 0.460 | +5.1% |
| 10% | 0.462 | 0.441 | +4.6% |

### What This Shows

The hybrid outperforms pure XGBoost at ALL training levels — even with only 10% of the training data. The mechanistic features provide a stable, consistent benefit regardless of sample size.

This suggests the PK/PD model contributes real physiological structure, not just additional noise that helps when you have lots of data. The hybrid should generalize better to new patient populations.

---

## 4. What I'm Still Thinking About

A few things I'm uncertain about:

**On the kidney results:** It's possible that kidney disease really doesn't affect propofol pharmacodynamics much. Propofol is primarily metabolized by the liver, not the kidneys. So from a mechanistic standpoint, maybe I should have expected this null result. Liver disease might be a better target if we want to pursue the comorbidity angle.

**On the hybrid:** The 15% improvement is solid, but I wonder if there's more to squeeze out. The mechanistic features had relatively low importance compared to the recent SAS scores. Maybe the hybrid is mostly just learning to trust the recent SAS and using the mechanistic stuff as a minor correction. I could try Option 2 (residual learning — train ML to predict the error of the mechanistic model) to see if framing it differently helps.

---

## 5. Proposed Next Steps

Based on where we are, I see a few options:

**Option A: Write it up now.** We have a positive hybrid result and robustness evidence. The story would be: "Pure ML beats pure physiology, but physiology+ML beats both — and this advantage holds across all training sizes." The kidney subgroup stuff becomes a secondary finding showing calibration helps universally.

**Option B: Try Option 2 (residual learning).** Instead of adding mechanistic features to XGBoost, train ML to predict the residual error of the PK/PD model. This keeps the mechanistic model "in charge" and uses ML as a correction layer. Might produce cleaner interpretability.

**Option C: Pivot to liver disease.** If we still want the comorbidity angle, liver disease makes more mechanistic sense for propofol (hepatic metabolism). Would need new BigQuery queries.

My instinct is that we have enough for a paper with the hybrid result + robustness test, but I wanted to check with you before deciding whether to do more experiments or start writing.

---

## Attached Files

1. `kidney_subgroup_analysis.png` — visualization of calibration benefit across kidney subgroups
2. `kidney_subgroup_summary.csv` — detailed results table
3. `hybrid_experiment_results.png` — comparison of three models
4. `hybrid_experiment_results.csv` — accuracy metrics
5. `hybrid_feature_importance.csv` — what the hybrid model learned to use
6. `robustness_test_results.png` — hybrid vs pure XGBoost across training sizes
7. `robustness_test_results.csv` — degradation analysis data

Let me know what you think and how you'd like to proceed.

— Chris
