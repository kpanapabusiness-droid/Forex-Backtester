# Arc 9 Step 4 - Extractability investigation + artefact production

Verdict: **FAIL**

Pipeline E gate: RF AUC >= 0.65. Pipeline D1 gate: RF AUC >= 0.6 AND exclusion <= 0.3.
Threshold sweep: [0.4, 0.5, 0.6, 0.7]. Selection: max precision with recall >= 0.6. v2.2 §3: no max-F1 fallback.

## Per-archetype results

| label | mode | n_total | n_pos | E RF AUC | E LR AUC | E status | E thr | E prec | E rec | D1 t | D1 AUC | D1 excl | D1 status | D1 thr | D1 prec | D1 rec | pipeline | pre_t SL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cluster_0_individual | individual | 2153 | 365 | 0.511 | 0.527 | fail | - | - | - | 1 | 0.626 | 0.013 | fail_threshold_sweep | - | - | - | **dies** | 2.0 |

## Outputs

- entry_features.csv
- predictability_angle_E.csv
- predictability_angle_D1.csv
- extractability_pass_list.csv
- threshold_sweep_E_<label>.csv (per surviving Pipeline E)
- threshold_sweep_D1_<label>.csv (per surviving Pipeline D1)
- archetype_<label>_E_classifier.joblib + _E_filter.yaml (per surviving E)
- archetype_<label>_D1_classifier.joblib + _D1_policy.yaml (per surviving D1)
