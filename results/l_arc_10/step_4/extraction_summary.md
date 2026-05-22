# Arc 10 v3.0 — Step 4 Extraction Summary

- Candidate clusters: [1]
- Features used: 37 (default 27 + arc-specific extras)
- Classifiers: ['lgbm', 'logistic', 'rf'] at Appendix A defaults

## Per-cluster AUC + threshold sweep

| Cluster | Classifier | n+ | mean_AUC | OOS_AUC | AUC-best thr | prec | recall | n_admit | F1-best thr | F1 | F1 prec | F1 rec |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| c1 | rf | 1528 | 0.5142 | 0.4941 | 0.50 | 0.456 | 0.213 | 601 | 0.05 | 0.469 | 0.469 | 1.000 |
| c1 | logistic | 1528 | 0.5012 | 0.5020 | 0.50 | 0.479 | 0.463 | 1247 | 0.13 | 0.469 | 0.469 | 0.998 |
| c1 | lgbm | 1528 | 0.5199 | 0.5095 | 0.50 | 0.479 | 0.484 | 1304 | 0.06 | 0.470 | 0.470 | 0.998 |

## Top-10 features per cluster (permutation importance on best classifier)


### c1 (best clf: lgbm, mean AUC: 0.5199)

| Rank | Feature | imp_mean | imp_std |
|---:|---|---:|---:|
| 1 | atr_14 | 0.02682 | 0.01002 |
| 2 | prior_session_low_distance | 0.01159 | 0.00643 |
| 3 | prior_session_high_distance | 0.00956 | 0.00454 |
| 4 | upper_fraction | 0.00701 | 0.00522 |
| 5 | distance_to_round_number | 0.00689 | 0.00497 |
| 6 | atr14_at_signal | 0.00674 | 0.00295 |
| 7 | eur_strength_index | 0.00397 | 0.00474 |
| 8 | atr_percentile_100 | 0.00397 | 0.00400 |
| 9 | swing_low_distance_14 | 0.00217 | 0.01045 |
| 10 | atr_vs_trailing_100 | 0.00164 | 0.00485 |

**Swing-feature audit (per dispatch §Step 4):**
- Swing-derived features in top 10: ['prior_session_low_distance', 'prior_session_high_distance', 'swing_low_distance_14']
- swing_high_distance_14 / swing_low_distance_14 use rolling(.max/.min, window=14).shift(1) on mid-OHLC — one-sided lookback, no future bars. Causal. (Trailing N-bar swing, not centred ±N.)

## Gate readings (informational; arc continues regardless)

- L_PROTOCOL v3.0 §3 has no Step-4 AUC gate. Step 5 (WFO) is the only deployment gate.
- Cross-arc historical context: prior V-shape near-misses (Arc 7, Arc 10 v2.3) hovered Pipeline E ≈ 0.48-0.63 — feature-set bound.
