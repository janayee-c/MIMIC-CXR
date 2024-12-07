SELECT
  icustays.subject_id,
  icustays.hadm_id,
  icustays.stay_id,
  CASE WHEN admissions.hospital_expire_flag = 1 THEN TRUE ELSE FALSE END AS hospital_expire_flag,
  CASE WHEN ventilation.ventilation_status = 'InvasiveVent' THEN TRUE ELSE FALSE END AS ventilation_status,
  COALESCE(sepsis3.sepsis3, FALSE) AS sepsis3,
  CASE WHEN admissions.hospital_expire_flag = 1 OR ventilation.ventilation_status = 'InvasiveVent' OR COALESCE(sepsis3.sepsis3, FALSE) THEN TRUE ELSE FALSE END AS severe
FROM
  `physionet-data.mimic_icu.icustays` icustays
LEFT JOIN `physionet-data.mimic_core.admissions` admissions ON icustays.hadm_id = admissions.hadm_id
LEFT JOIN `physionet-data.mimic_derived.ventilation` ventilation ON icustays.stay_id = ventilation.stay_id
LEFT JOIN `physionet-data.mimic_derived.sepsis3` sepsis3 ON icustays.stay_id = sepsis3.stay_id;
