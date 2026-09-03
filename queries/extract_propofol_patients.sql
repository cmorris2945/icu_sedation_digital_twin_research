-- =====================================================================
-- extract_propofol_patients.sql         (BigQuery history: "QUERY 2")
-- =====================================================================
--
-- Purpose: Identify the patient cohort for our ICU sedation digital twin
-- analysis and attach demographics plus the legacy kidney comorbidity flags.
-- This is the foundation query: it produces kidney_subgroups_patients.csv,
-- which the other two queries' cohort CTEs reproduce independently.
--
-- Output columns (as they appear in kidney_subgroups_patients.csv):
--   stay_id, gender, age, weight, height,
--   has_AKI, has_CKD_Stage_1_2, has_CKD_Stage_3, has_CKD_Stage_4,
--   has_CKD_Stage_5_ESRD, has_CKD_Unspecified, has_Diabetic_Nephropathy,
--   has_Dialysis, has_Hypertensive_Kidney, has_Other_Kidney, has_ANY_Kidney
--
-- Inclusion criteria (CORRECTED 2026-09-02 against the recovered query):
--   - At least 15 SAS observations during the ICU stay
--   - At least 5 propofol infusion events during the ICU stay
--   - Propofol rate > 0
--   - SAS valuenum between 1 and 7
--
--   NO age filter is applied. MIMIC-IV contains adult patients only by
--   construction, so the cohort is adult (verified: min 18, max 93), but
--   there is no age >= 18 predicate in this query. Earlier versions of this
--   header claimed one, and claimed ">= 11 SAS observations". Both were
--   wrong. The 11 is the MODELING requirement enforced downstream in
--   01_build_features.py (10 history steps + 1 prediction target); it is not
--   the extraction filter and it never binds, since 15 is stricter.
--
-- Age is COMPUTED using MIMIC's anchor-year adjustment, not read raw.
--   Confirmed by the recovered text below:
--     DATETIME_DIFF(i.intime, DATETIME(p.anchor_year, 1, 1, 0, 0, 0), YEAR)
--       + p.anchor_age
--   (the YEAR unit and the "+ anchor_age" term fall just past the point
--   where the recovered text is cut off; the pattern is unambiguous, but
--   see the TRUNCATION notice below before treating it as verbatim)
--
-- Weight and height are LEFT JOINed from
--   mimiciv_3_1_derived.first_day_weight and first_day_height,
--   so both may be NULL (in the extract: weight 1.2%, height 26.7%).
--   The joins themselves fall past the truncation point and are NOT in the
--   recovered text.
--
-- Cohort definition (identical CTE block in QUERY 2, 3 and 4):
--   All three queries rebuild the same "rich_cohort" CTE rather than joining
--   against a saved patient table. The thresholds are HARD-CODED LITERALS:
--       HAVING COUNT(*) >= 15   (SAS observations per stay)
--       HAVING COUNT(*) >= 5    (propofol infusion events per stay)
--   Any change to a threshold must be made in all three files or the three
--   CSVs will describe different cohorts. QUERY 2 differs from 3 and 4 only
--   in that its CTEs also project sas_count and propofol_count.
--
-- Provenance:
--   Extraction date: 2026-04-12
--   BigQuery job ID: bquxjob_6570a768_19d84255827
--   itemids: propofol 222168 (icu.inputevents), SAS 223753 (icu.chartevents)
--   Kidney flags derived from hosp.diagnoses_icd, joined on hadm_id.
--
-- AUTHOR: Christopher Morris
-- =====================================================================
--
-- #####################################################################
-- ## PARTIAL RECOVERY - THIS QUERY IS INCOMPLETE AND WILL NOT RUN.   ##
-- ##                                                                 ##
-- ## Everything below is VERBATIM as supplied on 2026-09-02, but the ##
-- ## supplied text was cut off mid-expression inside the             ##
-- ## `demographics` CTE, at the DATETIME_DIFF call. Nothing has been ##
-- ## invented to complete it.                                        ##
-- ##                                                                 ##
-- ## STILL MISSING, after the truncation point:                      ##
-- ##   - the remainder of the age expression                        ##
-- ##     (YEAR unit argument and "+ p.anchor_age AS age")            ##
-- ##   - gender/weight/height projection details                     ##
-- ##   - the LEFT JOINs to mimiciv_3_1_derived.first_day_weight and  ##
-- ##     first_day_height                                            ##
-- ##   - the FROM/JOIN clauses of the demographics CTE               ##
-- ##   - the final SELECT that assembles demographics with           ##
-- ##     patient_kidney_summary into the output columns              ##
-- ##                                                                 ##
-- ## This is strictly better than the guessed template it replaced   ##
-- ## (which wrongly contained an anchor_age >= 18 filter and a >= 11 ##
-- ## SAS threshold), because every line present here is real. But it ##
-- ## is NOT yet a reproducible query. Retrieve the complete text     ##
-- ## with job ID bquxjob_6570a768_19d84255827 and replace this file. ##
-- #####################################################################

-- QUERY 2: Extract full patient data with kidney subgroup flags
-- Run this in BigQuery and download as CSV

WITH propofol_patients AS (
    SELECT DISTINCT stay_id
    FROM `physionet-data.mimiciv_3_1_icu.inputevents`
    WHERE itemid = 222168
    AND rate > 0
),

sas_patients AS (
    SELECT
        stay_id,
        COUNT(*) AS sas_count
    FROM `physionet-data.mimiciv_3_1_icu.chartevents`
    WHERE itemid = 223753
    AND valuenum BETWEEN 1 AND 7
    GROUP BY stay_id
    HAVING COUNT(*) >= 15  -- At least 15 SAS observations
),

propofol_counts AS (
    SELECT
        stay_id,
        COUNT(*) AS propofol_count
    FROM `physionet-data.mimiciv_3_1_icu.inputevents`
    WHERE itemid = 222168
    AND rate > 0
    GROUP BY stay_id
    HAVING COUNT(*) >= 5  -- At least 5 propofol events
),

-- Rich data cohort
rich_cohort AS (
    SELECT p.stay_id
    FROM propofol_patients p
    INNER JOIN sas_patients s ON p.stay_id = s.stay_id
    INNER JOIN propofol_counts pc ON p.stay_id = pc.stay_id
),

stay_info AS (
    SELECT i.stay_id, i.hadm_id, i.subject_id
    FROM `physionet-data.mimiciv_3_1_icu.icustays` i
    INNER JOIN rich_cohort r ON i.stay_id = r.stay_id
),

-- Categorize kidney diagnoses
kidney_diagnoses AS (
    SELECT
        s.stay_id,
        d.icd_code,
        CASE
            WHEN d.icd_code IN ('N170', 'N171', 'N172', 'N178', 'N179') THEN 'AKI'
            WHEN d.icd_code LIKE '584%' THEN 'AKI'
            WHEN d.icd_code IN ('N181', 'N182', '5851', '5852') THEN 'CKD_Stage_1_2'
            WHEN d.icd_code IN ('N183', 'N1830', 'N1831', 'N1832', '5853') THEN 'CKD_Stage_3'
            WHEN d.icd_code IN ('N184', '5854') THEN 'CKD_Stage_4'
            WHEN d.icd_code IN ('N185', 'N186', '5855', '5856') THEN 'CKD_Stage_5_ESRD'
            WHEN d.icd_code IN ('N189', 'N18', 'N188', '5859') THEN 'CKD_Unspecified'
            WHEN d.icd_code LIKE 'E102%' OR d.icd_code LIKE 'E112%' OR d.icd_code LIKE '2504%' THEN 'Diabetic_Nephropathy'
            WHEN d.icd_code = 'Z992' OR d.icd_code LIKE 'V451%' THEN 'Dialysis'
            WHEN d.icd_code LIKE 'I12%' OR d.icd_code LIKE 'I13%' OR d.icd_code LIKE '403%' OR d.icd_code LIKE '404%' THEN 'Hypertensive_Kidney'
            WHEN d.icd_code LIKE 'N0%' OR d.icd_code LIKE '58%' THEN 'Other_Kidney'
            ELSE NULL
        END AS kidney_category
    FROM stay_info s
    INNER JOIN `physionet-data.mimiciv_3_1_hosp.diagnoses_icd` d ON s.hadm_id = d.hadm_id
    WHERE
        d.icd_code LIKE 'N17%' OR d.icd_code LIKE 'N18%' OR d.icd_code LIKE 'N19%'
        OR d.icd_code LIKE 'E102%' OR d.icd_code LIKE 'E112%'
        OR d.icd_code = 'Z992' OR d.icd_code LIKE 'I12%' OR d.icd_code LIKE 'I13%'
        OR d.icd_code LIKE 'N0%'
        OR d.icd_code LIKE '584%' OR d.icd_code LIKE '585%' OR d.icd_code LIKE '586%'
        OR d.icd_code LIKE '2504%' OR d.icd_code LIKE 'V451%'
        OR d.icd_code LIKE '403%' OR d.icd_code LIKE '404%'
        OR d.icd_code LIKE '580%' OR d.icd_code LIKE '581%' OR d.icd_code LIKE '582%' OR d.icd_code LIKE '583%'
),

patient_kidney_summary AS (
    SELECT
        stay_id,
        MAX(CASE WHEN kidney_category = 'AKI' THEN 1 ELSE 0 END) AS has_AKI,
        MAX(CASE WHEN kidney_category = 'CKD_Stage_1_2' THEN 1 ELSE 0 END) AS has_CKD_Stage_1_2,
        MAX(CASE WHEN kidney_category = 'CKD_Stage_3' THEN 1 ELSE 0 END) AS has_CKD_Stage_3,
        MAX(CASE WHEN kidney_category = 'CKD_Stage_4' THEN 1 ELSE 0 END) AS has_CKD_Stage_4,
        MAX(CASE WHEN kidney_category = 'CKD_Stage_5_ESRD' THEN 1 ELSE 0 END) AS has_CKD_Stage_5_ESRD,
        MAX(CASE WHEN kidney_category = 'CKD_Unspecified' THEN 1 ELSE 0 END) AS has_CKD_Unspecified,
        MAX(CASE WHEN kidney_category = 'Diabetic_Nephropathy' THEN 1 ELSE 0 END) AS has_Diabetic_Nephropathy,
        MAX(CASE WHEN kidney_category = 'Dialysis' THEN 1 ELSE 0 END) AS has_Dialysis,
        MAX(CASE WHEN kidney_category = 'Hypertensive_Kidney' THEN 1 ELSE 0 END) AS has_Hypertensive_Kidney,
        MAX(CASE WHEN kidney_category = 'Other_Kidney' THEN 1 ELSE 0 END) AS has_Other_Kidney,
        MAX(CASE WHEN kidney_category IS NOT NULL THEN 1 ELSE 0 END) AS has_ANY_Kidney
    FROM kidney_diagnoses
    GROUP BY stay_id
),

-- Demographics
demographics AS (
    SELECT
        i.stay_id,
        p.gender,
        DATETIME_DIFF(i.intime, DATETIME(p.anchor_year, 1, 1, 0, 0, 0),

-- <<<<<<<<<<<<<<<<<<<< RECOVERED TEXT ENDS HERE >>>>>>>>>>>>>>>>>>>>
-- The supplied text stops mid-call on the line above. See the PARTIAL
-- RECOVERY banner at the top of this file for exactly what is missing.
-- Do not attempt to run this file. Do not cite it as the method until
-- the remainder is recovered.
