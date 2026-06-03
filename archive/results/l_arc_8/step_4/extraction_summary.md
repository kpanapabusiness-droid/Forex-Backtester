# Arc 8 — Step 4 Extraction Summary

_Generated: 2026-05-22T10:49:45.960856+00:00Z_

- Candidate clusters processed: 1
- Classifiers tested: RF, LGBM, Logistic (Appendix A defaults)
- CV: 5-fold TimeSeriesSplit
- Feature catalogue: 23 (v3.0 Step 1)

## Per-cluster, per-model OOS AUC + threshold metrics

| Cluster | Model | Mean AUC | Std AUC | Fold AUCs | AUC-best thr | Precision | Recall | Trades | F1-best thr | F1 | F1 Trades |
|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 2 | rf | 0.5300 | 0.0204 | 0.4902;0.5401;0.5426;0.5454;0.5318 | 0.220 | 0.250 | 0.521 | 2,697 | 0.050 | 0.374 | 5,630 |
| 2 | logreg | 0.5227 | 0.0128 | 0.5199;0.5119;0.5210;0.5473;0.5133 | 0.240 | 0.260 | 0.338 | 1,684 | 0.050 | 0.374 | 5,623 |
| 2 | lgbm | 0.5212 | 0.0273 | 0.5142;0.5162;0.5531;0.4764;0.5463 | 0.160 | 0.244 | 0.567 | 3,007 | 0.060 | 0.372 | 5,160 |

## Best classifier per cluster (drives Step 5 A2/A6)

| Cluster | Best model | Mean AUC | AUC-best threshold | Pickle |
|---:|---|---:|---:|---|
| 2 | rf | 0.5300 | 0.2200 | `results\l_arc_8\step_4\classifiers\cluster_2_rf.pkl` |

## Top-10 permutation-importance features (per cluster)

### Cluster 2

| Feature | Perm. importance (mean) | Std |
|---|---:|---:|
| swing_low_distance_14 | +0.02307 | 0.00308 |
| prior_session_low_distance | +0.01934 | 0.00108 |
| atr_vs_trailing_100 | +0.01850 | 0.00428 |
| kijun_26_distance | +0.01814 | 0.00316 |
| dollar_bloc_state | +0.01785 | 0.00068 |
| atr_percentile_100 | +0.01602 | 0.00344 |
| spread_vs_trailing_100 | +0.01521 | 0.00132 |
| spread_percentile_100 | +0.01416 | 0.00087 |
| usd_strength_index | +0.01391 | 0.00099 |
| distance_to_round_number | +0.01363 | 0.00035 |


## Methodology notes

- Target = binary cluster membership (cluster_primary == cid).
- TimeSeriesSplit preserves chronological order: each fold's train is strictly before its test.
- Permutation importance computed on a model refit on full data (scoring=ROC-AUC, n_repeats=5).
- AUC-best threshold derived from Youden's J = TPR - FPR on the concatenated OOS proba stream.
- F1-best threshold derived from F1 sweep over [0.05, 0.95] in 0.01 steps.
- Logistic regression features standardised via StandardScaler per fold; RF and LGBM use raw values.
- Suspect-lineage features (cross_pair class) kept in training matrix per L_PROTOCOL Step 4 mechanic 7 — Step 6 will downgrade/kill candidates whose load-bearing features fail the producer audit. Lineage report at `step_1/feature_lineage.csv`.
