# Data Directory

This directory is where local data files belong when running the analysis pipeline. The contents of this directory are not committed to version control because the patient data we work with is governed by the MIMIC-IV data use agreement which prohibits redistribution. The directory itself is preserved in version control through this README file, but any actual data files placed here will be ignored by Git.

## What Should Go In This Directory

When you set up this project on a new machine to reproduce our analysis, you should place three CSV files in this directory. The file kidney_subgroups_patients.csv contains patient demographic information and cohort identifiers. The file kidney_subgroups_propofol.csv contains the detailed records of every propofol infusion event. The file kidney_subgroups_sas.csv contains the Sedation-Agitation Scale assessments that we use as our prediction targets. These three files together form the complete input dataset for our analysis pipeline.

The CSV files are produced by running the SQL queries in the queries directory against the MIMIC-IV database via Google BigQuery. Once you have credentialed access to MIMIC-IV through PhysioNet, you can run these queries and save the results as CSV files in this directory. The naming convention for the files is preserved from earlier phases of the project when we were investigating kidney comorbidities, but the data extraction logic is applicable to our current analysis as documented in the queries directory.

The processed_features subdirectory is where the analysis pipeline saves intermediate feature datasets after running the slow pharmacokinetic simulation step. These files are also excluded from version control because they can be regenerated from the raw data and the analysis code. Saving them locally makes subsequent analysis runs faster but does not require them to be shared.

## What Should Not Go In This Directory

Raw MIMIC-IV data files in their original format from PhysioNet should not be stored in this directory. The data use agreement requires that protected health data be kept in secure storage with appropriate access controls. If you are working with raw MIMIC-IV files, keep them in a separate location and use this directory only for derived files that you have created through your own queries.

Any files containing patient identifiers, original timestamps that could enable re-identification, free text clinical notes, or other particularly sensitive information should never be placed here regardless of whether they would be technically allowed by the data use agreement. As a general principle, this directory should contain only the minimum data necessary to run the analysis, and that data should be at the level of aggregation specified in our extraction queries.

## How To Verify Your Data Files Are Correct

After placing the CSV files in this directory, you can verify they have the expected structure by checking the column names. The patients file should have columns including stay_id, gender, age, weight, height, and various comorbidity flags. The propofol infusions file should have columns for stay_id, starttime, endtime, rate, and rateuom. The SAS observations file should have columns for stay_id, charttime, and sas_score.

If your files have different column names or structures than these, you may need to either modify the SQL queries to produce the expected format, or modify the analysis scripts to read whatever format your data has. The first option is generally preferable because it keeps the analysis scripts consistent with what we have been running.

## Data Security Considerations

Even though these CSV files are derived from MIMIC-IV and contain only the information we need for analysis, they still contain patient health data that requires appropriate handling. You should ensure that the disk where these files live is encrypted, that access to your computer requires a password, and that you do not back these files up to cloud services that have not been approved for protected health information storage. When you are done with the analysis, you should plan to securely delete the files rather than just moving them to trash.

If you are working in a shared environment such as a university workstation, take particular care that other users cannot access these files. Storing them in your private user directory rather than a shared location is essential. If you need to share results from the analysis with collaborators, share only the aggregated findings and never the underlying patient records.
