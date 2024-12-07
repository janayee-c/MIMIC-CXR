select
  ed_pneumonia.subject_id,
	ed_pneumonia.hadm_id,
	ed_pneumonia.ed_stay_id,
 FROM
(SELECT
  ed.subject_id,
  ed.hadm_id,
  ed.stay_id as ed_stay_id
FROM
  `physionet-data.mimic_ed.edstays` ed
INNER JOIN
  `physionet-data.mimic_ed.diagnosis` dg
ON
  ed.stay_id = dg.stay_id
WHERE
  LOWER(dg.icd_title) LIKE '%pneumonia%'
  AND LOWER(dg.icd_title) NOT LIKE '%without pneumonia%'
  AND LOWER(dg.icd_title) NOT LIKE '%vacc for strep pneumonia%'
  AND LOWER(dg.icd_title) NOT LIKE 'Klebsiella pneumoniae as the cause of diseases'
) as ed_pneumonia
INNER JOIN `physionet-data.mimic_core.admissions` ad ON ed_pneumonia.hadm_id = ad.hadm_id
left JOIN `physionet-data.mimic_hosp.diagnoses_icd` dg on ed_pneumonia.hadm_id = dg.hadm_id
INNER JOIN `physionet-data.mimic_hosp.d_icd_diagnoses` ddg ON dg.icd_code = ddg.icd_code
WHERE
--  lower(ddg.long_title) LIKE '%pneumonia%'
 dg.icd_code in ("486","J189","4829","J159","J188","48242","4821","J13","J15212","J181","481","J151","48241","J15211","J156","48284","48283","J150","J154","4820","J851","J158","48282","4822","J14","J180","48239","J155","48230","485","J1529","48281","J1520","48249","J168","J153","48289","48240")
 group by
  ed_pneumonia.subject_id,
	ed_pneumonia.hadm_id,
	ed_pneumonia.ed_stay_id
