# Arc 11 v3.0 — Step 4 Extraction Summary

Per L_PROTOCOL §2 Step 4.

Candidate clusters: [0]
Feature columns: 32

## Per-cluster × classifier OOS AUC

| cluster | classifier | mean OOS AUC | fold AUCs | AUC-best thr | precision | recall | n_admitted |
|---:|---|---:|---|---:|---:|---:|---:|
| 0 | random_forest | 0.6873 | 0.6765;0.6528;0.7142;0.6747;0.7182 | 0.267 | 0.428 | 0.859 | 3871 |
| 0 | logistic | 0.6861 | 0.6890;0.6472;0.7039;0.6987;0.6918 | 0.248 | 0.438 | 0.880 | 3871 |
| 0 | lgbm | 0.6751 | 0.6428;0.6598;0.6971;0.6619;0.7139 | 0.131 | 0.413 | 0.893 | 4168 |

## Per-cluster best classifier + top-10 features
### Cluster 0
- Best classifier: **random_forest** (mean OOS AUC 0.6873)
- AUC-best threshold: 0.267
- Top-10 features (permutation importance):
  - w1_close_slope_sign
  - kijun_26_distance
  - day_of_week
  - break_magnitude_atr ← swing-derived
  - eur_strength_index
  - swing_low_distance_14 ← swing-derived
  - atr_vs_trailing_100
  - atr_percentile_100
  - distance_to_round_number
  - usd_strength_index
