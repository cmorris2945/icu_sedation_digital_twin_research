# ICU Sedation Digital Twin Research

This repository contains the complete code, analyses, and documentation for research investigating whether mechanistic pharmacokinetic-pharmacodynamic modeling can be combined with machine learning to improve ICU sedation prediction. The work is part of a PhD research program at the University of Florida, Department of Computer and Information Science and Engineering.

## What This Research Is About

The fundamental question we are addressing is whether augmenting machine learning models with features derived from mechanistic propofol pharmacology produces better predictions of ICU sedation levels than using machine learning alone. We use the Eleveld 2018 pharmacokinetic-pharmacodynamic model as our mechanistic foundation, calibrate the EC50 parameter to individual patients to create patient-specific digital twins, and test the hybrid approach across multiple machine learning algorithm families.

A secondary question we investigate is whether different machine learning algorithm families benefit from mechanistic features in fundamentally different ways. Specifically, we hypothesize that recurrent neural networks like LSTM benefit more from temporal mechanistic features such as the time series of effect site drug concentration, while tree based models like XGBoost benefit more from static mechanistic features such as the patient calibrated EC50 value. If this hypothesis holds, it would represent a methodological contribution to the field of physiologically grounded machine learning by suggesting that mechanistic feature type should be matched to algorithm architecture.

The clinical motivation for this research is that ICU sedation management currently relies on population averaged dosing protocols that do not account for individual patient pharmacokinetic differences. Better prediction models could enable more personalized sedation management, potentially reducing complications associated with both under sedation and over sedation while shortening ICU stays.

## Repository Structure

The repository is organized into directories that separate different types of content. The src directory contains the Python source code that performs all the analyses, with a numbered prefix on each script indicating the order in which scripts should be executed for full reproduction. The src/lib subdirectory contains reusable utility modules that the analysis scripts import. The queries directory contains the SQL queries used to extract our patient cohort from the MIMIC-IV database via Google BigQuery. The results directory contains the outputs from completed analysis runs, organized into raw experimental outputs and processed summary statistics. The figures directory contains generated visualizations in both PNG format suitable for embedding in manuscripts and PDF format suitable for final publication. The docs directory contains all written documentation including methodology descriptions, study guides, and progress reports. The data directory holds local data files, though raw MIMIC-IV patient data is not included due to data use agreement restrictions. The environment directory contains Python environment specifications that ensure exact reproducibility of the software environment used to produce the results. The notebooks directory holds any exploratory Jupyter notebooks created during the research, separate from the production analysis scripts. The tests directory holds any unit tests verifying that key functions work correctly.

## Reproducing the Analysis

To reproduce our analysis from scratch, you need three things. First, a Python 3.10 or newer environment with the dependencies listed in environment/requirements.txt. Second, credentialed access to MIMIC-IV through PhysioNet, which requires completing ethics training and signing a data use agreement. Third, the ability to run Google BigQuery queries against the physionet-data project.

The reproduction workflow proceeds in five phases. The first phase is data extraction, where you run the SQL queries in the queries directory to pull the patient cohort and clinical data from MIMIC-IV into local CSV files. The second phase is feature building, where you run src/01_build_features.py to process the raw data through our Eleveld pharmacokinetic simulation and produce the feature dataset. This is the slow step in the pipeline and may take fifteen to thirty minutes for the full cohort. The third phase is the ablation study, where you run src/02_run_ablation.py to train all model configurations and produce the raw experimental results. This typically takes thirty to forty five minutes. The fourth phase is analysis, where you run src/03_analyze_results.py to compute summary statistics from the raw results. This runs in seconds. The fifth phase is figure generation, where you run src/04_make_figures.py to produce publication quality visualizations.

The scripts are designed so that each phase produces output files that the next phase consumes. This means you can rerun any phase independently if you want to modify something, without redoing the work of previous phases. The feature building phase in particular caches its output to a pickle file, which means subsequent runs of the analysis scripts load the features instantly rather than redoing the slow simulation.

## Data Availability

The raw MIMIC-IV patient data underlying this research is publicly available to credentialed researchers through PhysioNet at https://physionet.org/content/mimiciv/3.1/. Access requires completing the credentialing process which includes ethics training. The specific patient cohort used in our analysis can be reproduced by running the SQL queries in this repository against MIMIC-IV version 3.1.

We cannot redistribute the actual patient data due to the terms of the MIMIC-IV data use agreement. However, this does not impede reproducibility because any researcher with their own MIMIC-IV access can run our queries and obtain the same dataset. The combination of preserved queries plus preserved analysis code provides complete reproducibility within the bounds of legal data sharing constraints.

## Code Availability

All analysis code is available in this repository under an open license to be specified before publication. The code is organized to be both readable as documentation of methodology and executable as a reproducible analysis pipeline. We have made specific choices in the code organization to support these dual purposes, including extensive inline comments explaining methodological decisions, separation of utility code from analysis scripts, and incremental result saving so that long running analyses can be interrupted and resumed without losing progress.

The dependency on specific package versions is captured in environment/requirements.txt to enable exact reproduction of the software environment we used. We have tested the analyses with the package versions specified there. Newer versions may also work but have not been verified.

## Citation

When this research is published, the citation information will be added to this README. Until then, please contact the corresponding author for citation guidance if you reference this work.

The mechanistic pharmacokinetic model we build on should be cited as follows. Eleveld DJ, Colin P, Absalom AR, Struys MMRF. Pharmacokinetic pharmacodynamic model for propofol for broad application in anaesthesia and sedation. British Journal of Anaesthesia. 2018;120(5):942-959. doi:10.1016/j.bja.2018.01.018.

The MIMIC-IV database we use should be cited as follows. Johnson AEW, Bulgarelli L, Shen L, et al. MIMIC-IV, a freely accessible electronic health record dataset. Scientific Data. 2023;10:1. doi:10.1038/s41597-022-01899-x.

## Project History

This research has gone through several phases of investigation. An initial phase examined whether kidney comorbidities produce differential effects in propofol pharmacokinetic personalization. This investigation produced a null result because propofol is hepatically metabolized rather than renally cleared, so kidney disease does not substantially affect its pharmacokinetics. A second phase examined whether multiple machine learning algorithms benefit from hybrid mechanistic feature augmentation, with preliminary results showing modest but consistent improvements across XGBoost, Random Forest, MLP, and LSTM. The current phase focuses on understanding why algorithms differ in their response to mechanistic features through ablation studies, with the goal of producing a methodological contribution about hybrid mechanistic machine learning approaches.

The research is conducted under the supervision of the principal investigator and follows standard scientific practices for medical informatics research including patient level data splitting to prevent leakage, multiple random seeds for stochastic algorithms, statistical significance testing for primary findings, and full transparency about negative or null results.

## Contact and Collaboration

For questions about the methodology, requests for additional information, or potential collaboration, please reach out through appropriate academic channels. This research is part of a doctoral program and may be of interest to other researchers working at the intersection of pharmacology, machine learning, and critical care medicine.
