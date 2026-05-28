# Arc 10 v3.0 — Step 4 Extraction Summary

- Candidate clusters: [1]
- Features used: 37 (default 27 + arc-specific extras)
- Classifiers: ['lgbm', 'logistic', 'rf'] at Appendix A defaults

## Per-cluster AUC + threshold sweep

| Cluster | Classifier | n+ | mean_AUC | OOS_AUC | AUC-best thr | prec | recall | n_admit | F1-best thr | F1 | F1 prec | F1 rec |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| c1 | rf | 1528 | 0.5141 | 0.4949 | 0.50 | 0.465 | 0.215 | 596 | 0.05 | 0.469 | 0.469 | 1.000 |
| c1 | logistic | 1528 | 0.4976 | 0.4956 | 0.50 | 0.472 | 0.458 | 1250 | 0.11 | 0.469 | 0.469 | 0.998 |
| c1 | lgbm | 1528 | 0.5180 | 0.5059 | 0.50 | 0.470 | 0.497 | 1363 | 0.07 | 0.470 | 0.470 | 0.998 |

## Top-10 features per cluster (permutation importance on best classifier)


### c1 (best clf: lgbm, mean AUC: 0.5180)

| Rank | Feature | imp_mean | imp_std |
|---:|---|---:|---:|
| 1 | atr_14 | 0.02286 | 0.01099 |
| 2 | upper_fraction | 0.01154 | 0.00424 |
| 3 | prior_session_low_distance | 0.00664 | 0.00506 |
| 4 | prior_session_high_distance | 0.00507 | 0.00464 |
| 5 | distance_to_round_number | 0.00378 | 0.00780 |
| 6 | atr14_at_signal | 0.00367 | 0.00488 |
| 7 | kijun_26_distance | 0.00313 | 0.00434 |
| 8 | atr_vs_trailing_100 | 0.00188 | 0.00725 |
| 9 | w1_close_slope_sign | 0.00171 | 0.00411 |
| 10 | swing_low_distance_14 | 0.00112 | 0.01480 |

**Swing-feature audit (per dispatch §Step 4):**
- Swing-derived features in top 10: ['prior_session_low_distance', 'prior_session_high_distance', 'kijun_26_distance', 'swing_low_distance_14']
- swing_high_distance_14 / swing_low_distance_14 use rolling(.max/.min, window=14).shift(1) on mid-OHLC — one-sided lookback, no future bars. Causal. (Trailing N-bar swing, not centred ±N.)
- kijun_26_distance = kijun(high, low, 26).shift(1) — strictly prior bars. Causal.

## Gate readings (informational; arc continues regardless)

- L_PROTOCOL v3.0 §3 has no Step-4 AUC gate. Step 5 (WFO) is the only deployment gate.
- Cross-arc historical context: prior V-shape near-misses (Arc 7, Arc 10 v2.3) hovered Pipeline E ≈ 0.48-0.63 — feature-set bound.
