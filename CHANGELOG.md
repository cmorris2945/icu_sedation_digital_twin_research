# Project Changelog

This document tracks significant developments in the research project over time. Each entry describes what changed, when it changed, and why the change was made. This kind of historical record serves several purposes for an academic research project. It helps the researcher remember what was tried and why decisions were made, which becomes invaluable when writing the methods section of a paper months or years after the experiments were performed. It demonstrates to advisors and committee members that the research has progressed through systematic investigation rather than random tinkering. And it provides reviewers and other researchers with context about how the final methodology emerged from the iterative scientific process.

Entries are listed in reverse chronological order with the most recent changes appearing first.

## May 2026: Repository Restructuring and Local Pipeline Development

The complete project was reorganized into a properly structured research repository following conventions used by medical informatics journals. The new structure separates source code from results, figures, queries, documentation, and environment specifications. This reorganization prepares the project for eventual publication and ensures that all artifacts needed for reproducibility are properly preserved.

A complete local execution pipeline was developed to allow running the full ablation study on a researcher's own computer rather than depending on remote compute resources. The pipeline consists of four scripts that should be run in sequence. The build features script handles the slow pharmacokinetic simulation step. The run ablation script performs all the model training. The analyze results script computes summary statistics. The make figures script generates publication quality visualizations.

All SQL queries used to extract data from MIMIC-IV through BigQuery were documented with template files in the queries directory. The actual query text needs to be filled in from the BigQuery query history before the project can be fully reproduced from this repository, but the structure and documentation is now in place.

## April 2026: LSTM Stability Investigation

The initial multi seed runs of the LSTM ablation revealed substantial instability across random initializations, with the same configuration producing mean absolute error values ranging from 0.35 to 0.45 depending on the random seed used. A focused investigation tested three potential causes including inconsistent early stopping triggering, model size mismatch with available training data, and learning rate optimization issues.

The investigation determined that early stopping was triggering at very different points across random seeds, with some runs stopping after only 18 epochs while others trained for the full 50 epochs. Switching to fixed epoch training without early stopping reduced the coefficient of variation across seeds from approximately 10 percent to approximately 4 percent. This made the LSTM results stable enough to draw scientific conclusions, though some intrinsic variance remains due to limited training data.

## April 2026: Multi-Algorithm Ablation Study

The first version of the ablation study was implemented and run on a sample of 400 patients. The study tested five different feature configurations against both XGBoost and LSTM. The XGBoost portion produced clean interpretable results showing that XGBoost relies somewhat more on static mechanistic features than on temporal mechanistic derived features, with a modest differential of about 1 percentage point.

The initial LSTM results appeared to support a much larger differential favoring static mechanistic features, but subsequent multi seed analysis revealed these initial numbers were artifacts of a single fortunate random initialization rather than genuine effects. The discovery hypothesis remains under investigation pending more stable LSTM results.

## April 2026: Progress Report and Hypothesis Formalization

A formal progress report was prepared documenting three research hypotheses. The primary hypothesis is that mechanistic feature augmentation improves machine learning prediction across algorithm families. The secondary hypothesis is that the improvement is consistent across multiple algorithm families demonstrating generalizability. The tertiary hypothesis is that the benefit is most pronounced in patients with hepatic impairment, though this hypothesis was later set aside in favor of a more general computational discovery focus.

The report identified three weaknesses in the initial work that needed to be addressed. The single algorithm focus on XGBoost was insufficient as a computer science contribution. The 15 percent improvement could be explained by random variation. The analysis scope across multiple kidney comorbidity subgroups was too unfocused to support a clear narrative.

## April 2026: Multi-Algorithm Comparison

The multi algorithm comparison was implemented testing six algorithm families on the kidney subgroups cohort. Initial results showed LSTM with 6.1 percent improvement, XGBoost with 3.0 percent improvement, Random Forest with 1.1 percent improvement, MLP with 0.3 percent improvement, SVM with no improvement, and GRU showing training instability that prevented meaningful comparison.

The 4 of 5 valid algorithms showing improvement provided preliminary support for the generalizability hypothesis, though the magnitude of benefit varied substantially across algorithms.

## March 2026: Kidney Comorbidity Null Result

A comprehensive analysis was conducted to test whether mechanistic feature augmentation provides differential benefits for patients with kidney disease. The analysis examined acute kidney injury, chronic kidney disease at all stages, dialysis patients, diabetic nephropathy, hypertensive kidney disease, and a combined kidney comorbidity group.

The results showed that EC50 calibration improved prediction by approximately 15 to 20 percent equally across all kidney subgroups, with no statistically significant differential benefit for any specific kidney comorbidity. This null result is mechanistically consistent with propofol pharmacology because propofol is hepatically metabolized through glucuronidation rather than renally cleared. Kidney disease does not substantially alter propofol pharmacokinetics.

The null result was reported honestly to the advisor and the project pivoted away from kidney specific comorbidities toward more general questions about hybrid mechanistic machine learning approaches.

## February 2026: Initial Hybrid Model Experiment

The first hybrid mechanistic machine learning model was implemented combining XGBoost with mechanistic features derived from the Eleveld 2018 propofol pharmacokinetic pharmacodynamic model. The experiment showed approximately 14.9 percent improvement in mean absolute error compared to pure XGBoost using only clinical features.

Robustness testing across different training set sizes from 10 percent to 100 percent of available data showed that the hybrid model outperformed pure machine learning at all training levels, providing initial evidence that the improvement was not an artifact of specific data subsets.

## January 2026: Project Initialization

The research project was initialized with the goal of investigating whether digital twin technology could improve ICU sedation management. The MIMIC-IV version 3.1 database was chosen as the primary data source because of its public availability to credentialed researchers and its substantial cohort of ICU patients with detailed medication and assessment records.

The Eleveld 2018 pharmacokinetic pharmacodynamic model was selected as the mechanistic foundation because it represents the current state of the art for propofol modeling and has been validated in multiple clinical settings. The model provides population level parameters that can be calibrated to individual patients, which is the foundation for our patient specific digital twin approach.

The Sedation Agitation Scale was chosen as the prediction target because it is widely used in ICU practice and has consistent records in the MIMIC-IV database. Other potential targets like the Bispectral Index were considered but rejected because BIS data is not consistently available in the ICU setting.

## How To Update This Document

When making significant changes to the research, add a new entry at the top of this document under a heading that includes the year and month plus a brief description of the change. Each entry should describe what changed, why it changed, and any implications for the research direction or methodology. Keep entries focused on substantive changes rather than minor edits or corrections. The goal is to provide a high level history that someone can read in a few minutes to understand how the project arrived at its current state.
