-- =====================================================================
-- extract_sas_observations.sql          (BigQuery history: "QUERY 4")
-- =====================================================================
--
-- Purpose: Extract all Sedation-Agitation Scale assessments for patients
-- in our cohort. The SAS is a clinical scale used in ICUs to quantify a
-- patient's level of sedation or agitation, and it is what we are trying
-- to predict with our digital twin model.
--
-- The Sedation-Agitation Scale uses a 7-point scale where lower numbers
-- mean more sedated and higher numbers mean more agitated. Specifically,
-- 1 means unarousable, 2 means very sedated, 3 means sedated, 4 means
-- calm and cooperative, 5 means agitated, 6 means very agitated, and 7
-- means dangerous agitation. In ICU practice, clinicians typically aim
-- for SAS scores of 3 or 4, which represent appropriate sedation levels
-- for mechanically ventilated patients.
--
-- Output columns:
--   stay_id: ICU stay identifier (joins to patient cohort)
--   charttime: timestamp when the SAS assessment was recorded
--   sas_score: the SAS value as a number from 1 to 7
--
-- How SAS values are actually read (CORRECTED 2026-09-02):
--   This query reads c.valuenum directly. It does NOT parse text values.
--   Earlier versions of this header described a COALESCE fallback that
--   parsed strings like "3-Sedated" via SUBSTR. That logic was never in
--   the query that produced our data, and describing it here was wrong.
--   Confirmed against the extract: all 113,536 SAS values are numeric,
--   all lie within 1-7, and none are null. Do not reintroduce the text
--   parsing description, and do not cite it in the manuscript.
--
-- Cohort definition (identical CTE block in QUERY 2, 3 and 4):
--   The three extraction queries each rebuild the same "rich_cohort" CTE
--   rather than joining against a saved patient table. The thresholds are
--   HARD-CODED LITERALS, not derived or parameterised:
--       HAVING COUNT(*) >= 15   (SAS observations per stay)
--       HAVING COUNT(*) >= 5    (propofol infusion events per stay)
--   Any change to a threshold must be made in all three files or the
--   three CSVs will describe different cohorts.
--
--   Note that propofol_patients (DISTINCT stay_id with rate > 0) is
--   logically redundant with propofol_counts (>= 5 such events): any stay
--   satisfying the latter satisfies the former. The join is harmless and
--   is preserved here exactly as run.
--
-- IMPORTANT - what this query does NOT restrict:
--   There is no time restriction anywhere in this query. The >= 15 count
--   and the final SELECT both span the patient's ENTIRE ICU stay. The
--   72-hour analysis window is applied downstream in
--   01_build_features.py, not here. Consequently ">= 15 SAS observations"
--   does NOT mean 15 usable observations: 62.9% of the rows this query
--   returns fall outside the analysis window and are discarded later.
--   See instructions.txt Section 11.1.
--
-- Provenance:
--   Extraction date: 2026-04-12
--   BigQuery job ID: bquxjob_796d82a8_19d84277e85
--   itemid: 223753 (SAS) in physionet-data.mimiciv_3_1_icu.chartevents
--   Verified against the extract: minimum 15 observations per patient
--   (hard floor), all scores within 1-7, no nulls.
--
--   NOTE: the ">= 15 observations" figure is the EXTRACTION filter. The
--   ">= 11" figure that appears elsewhere in this project is the downstream
--   MODELING requirement in 01_build_features.py (10 history + 1 target).
--   They are different things; do not conflate them in the manuscript.
--
-- AUTHOR: Christopher Morris
-- =====================================================================

-- QUERY 4: Get SAS data for all patients with rich data
-- Save as: kidney_subgroups_sas.csv

WITH propofol_patients AS (
    SELECT DISTINCT stay_id
    FROM `physionet-data.mimiciv_3_1_icu.inputevents`
    WHERE itemid = 222168
    AND rate > 0
),

sas_patients AS (
    SELECT stay_id
    FROM `physionet-data.mimiciv_3_1_icu.chartevents`
    WHERE itemid = 223753
    AND valuenum BETWEEN 1 AND 7
    GROUP BY stay_id
    HAVING COUNT(*) >= 15
),

propofol_counts AS (
    SELECT stay_id
    FROM `physionet-data.mimiciv_3_1_icu.inputevents`
    WHERE itemid = 222168
    AND rate > 0
    GROUP BY stay_id
    HAVING COUNT(*) >= 5
),

rich_cohort AS (
    SELECT p.stay_id
    FROM propofol_patients p
    INNER JOIN sas_patients s ON p.stay_id = s.stay_id
    INNER JOIN propofol_counts pc ON p.stay_id = pc.stay_id
)

SELECT
    c.stay_id,
    c.charttime,
    c.valuenum AS sas_score
FROM `physionet-data.mimiciv_3_1_icu.chartevents` c
INNER JOIN rich_cohort r ON c.stay_id = r.stay_id
WHERE c.itemid = 223753
AND c.valuenum BETWEEN 1 AND 7
ORDER BY c.stay_id, c.charttime;
