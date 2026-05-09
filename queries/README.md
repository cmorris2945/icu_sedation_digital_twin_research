# SQL Queries for MIMIC-IV Data Extraction

This directory holds the SQL queries that were used to extract our patient cohort and clinical data from the MIMIC-IV database via Google BigQuery. These queries are essential for reproducibility because they define exactly which patients are in our analysis and exactly what data we use for each one. Without preserving these queries, no researcher could reproduce our cohort, which means they could not reproduce any of our results.

The MIMIC-IV database itself cannot be redistributed because of the data use agreement that researchers must sign to obtain access. However, the queries that select from the database can be freely shared, and that is what enables reproducibility despite the access restrictions. Any researcher who obtains their own credentialed access through PhysioNet can run these exact queries and obtain the same dataset we used.

## Required Database Access

To run these queries, you need credentialed access to MIMIC-IV through PhysioNet. The credentialing process requires completing an ethics training course and signing a data use agreement. Detailed instructions are available at the PhysioNet website. Once credentialed, you can access MIMIC-IV either through downloading the CSV files directly or through Google Cloud BigQuery, which is what we use because it allows querying without downloading the full database.

The specific BigQuery project is physionet-data, and the relevant datasets within that project are mimiciv_3_1_hosp for hospital admission data, mimiciv_3_1_icu for ICU-specific data, and mimiciv_3_1_derived for tables derived from the raw data by the MIMIC-IV team.

## Query Organization

The queries are organized by what they extract. Each query is in its own SQL file with a header comment explaining what the query does, why it was written, what the inputs and outputs are, and any important assumptions or limitations. The query files should be considered part of the methods section of our paper and will be cited in the paper's code and data availability statement.

## Query Files

The file extract_propofol_patients.sql identifies the patient cohort by selecting adult ICU patients who received propofol infusions during their stay. This is the foundation query that all others build on, because it defines which patients are included in our analysis.

The file extract_propofol_infusions.sql pulls the detailed records of every propofol infusion event for the patients in our cohort. This includes start times, end times, infusion rates, and units. The infusion records drive the pharmacokinetic simulation in our analysis pipeline.

The file extract_sas_observations.sql pulls the Sedation-Agitation Scale assessments that clinicians recorded during each patient's ICU stay. These are the targets we are trying to predict, so the quality and completeness of these records directly affects what we can learn.

The file extract_demographics.sql pulls patient characteristics like age, gender, weight, and height. These are clinical features in our analysis that may correlate with sedation responses.

The file extract_kidney_comorbidities.sql identifies patients with various forms of kidney disease, which we used in our initial analysis that produced a null result. We preserve this query even though that subgroup analysis is not the focus of the final paper, because it documents our complete experimental history.

## How To Use These Queries

The queries are written for the BigQuery dialect of SQL. They should be runnable in the BigQuery web console at console.cloud.google.com/bigquery once you have authenticated with credentialed access to the physionet-data project.

To reproduce our analysis dataset, run the queries in this order. First run extract_propofol_patients.sql to establish the cohort. Save the results as kidney_subgroups_patients.csv since that is the filename our analysis scripts expect. Second run extract_demographics.sql for the same cohort. Third run extract_propofol_infusions.sql to get the infusion records. Save as kidney_subgroups_propofol.csv. Fourth run extract_sas_observations.sql to get the sedation assessments. Save as kidney_subgroups_sas.csv. Place all three CSV files in the data folder of the analysis project, and the analysis scripts will find them automatically.

## Important Notes About the Cohort

Our cohort name kidney_subgroups was chosen during an earlier phase of the project when we were investigating kidney disease as a potential focus area. We later determined through analysis that kidney comorbidity does not produce differential effects in propofol pharmacokinetics, since propofol is hepatically metabolized rather than renally cleared. However, the cohort itself is still appropriate for our current analysis because it consists of adult ICU patients who received propofol and had sedation assessments. The kidney subgroup labels in the data are simply ignored by our final analysis.

This naming history is preserved in this documentation because changing it now would break compatibility with our existing analysis scripts and require us to retest everything. In the final paper we will describe the cohort by its inclusion criteria rather than its historical name to avoid confusing readers.

## Pending Items

I still need to copy the actual SQL query files into this directory. They currently exist only in my BigQuery query history. The next time I work on this project I will paste each query into its corresponding SQL file in this directory so they are properly preserved as text files that can be version controlled and shared.

The queries should also include comments inline explaining what each clause does, since reviewers and other researchers may not be familiar with the specific structure of the MIMIC-IV tables. Adding these explanatory comments will be part of the documentation work I do before the paper submission.
