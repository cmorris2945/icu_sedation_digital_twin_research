-- =====================================================================
-- extract_propofol_infusions.sql        (BigQuery history: "QUERY 3")
-- =====================================================================
--
-- Purpose: Extract all propofol infusion events for patients in our cohort.
-- These records drive the pharmacokinetic simulation in our analysis pipeline,
-- since the simulation needs to know when each patient received drug, at what
-- rate, and for how long.
--
-- The output is one row per infusion event, where an event represents a
-- continuous period during which a specific propofol rate was active. When
-- the rate changes, that creates a new event record. When the infusion is
-- paused or stopped, that ends the current event.
--
-- Output columns:
--   stay_id: ICU stay identifier (joins to patient cohort)
--   starttime: when this infusion event began
--   endtime: when this infusion event ended
--   rate: infusion rate as a numeric value
--   rateuom: unit of measurement for the rate (typically mcg/kg/min)
--
-- Important notes about units:
--   The rate is stored in MIMIC-IV in the units that were used clinically,
--   which for propofol is almost always mcg/kg/min. Our analysis pipeline
--   converts this to mg/min using the patient's weight, since the Eleveld
--   PK model expects absolute rates rather than weight-normalized rates.
--   We preserve the original units in this output so the conversion is
--   clearly visible in the analysis code rather than hidden in the query.
--
--   CAVEAT (noted 2026-09-02): this query does NOT filter on rateuom, and
--   01_build_features.py converts every row as though it were mcg/kg/min
--   without checking. In our extract that is safe, because rateuom is
--   uniformly 'mcg/kg/min' across all 31,235 rows. It is not guaranteed by
--   the query. Anyone re-running this against a different MIMIC version
--   should re-check the distinct rateuom values before trusting the PK
--   simulation, since a mg/kg/hr row would be silently mis-scaled.
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
--   There is no time restriction. Infusion events are returned for the
--   patient's entire ICU stay. The 72-hour analysis window is applied
--   downstream in 01_build_features.py, which also drops any event whose
--   endtime falls past the window rather than clipping it.
--   See instructions.txt Section 11.1.
--
-- Provenance:
--   Extraction date: 2026-04-12
--   BigQuery job ID: bquxjob_10b927f2_19d84266954
--   itemid: 222168 (propofol) in physionet-data.mimiciv_3_1_icu.inputevents
--   Verified against the extract: minimum 5 events per patient (hard floor),
--   no non-positive or null rates, rateuom uniformly 'mcg/kg/min'.
--
-- AUTHOR: Christopher Morris
-- =====================================================================

-- QUERY 3: Get propofol data for all patients with rich data
-- Save as: kidney_subgroups_propofol.csv

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
    i.stay_id,
    i.starttime,
    i.endtime,
    i.rate,
    i.rateuom
FROM `physionet-data.mimiciv_3_1_icu.inputevents` i
INNER JOIN rich_cohort r ON i.stay_id = r.stay_id
WHERE i.itemid = 222168
AND i.rate > 0
ORDER BY i.stay_id, i.starttime;
