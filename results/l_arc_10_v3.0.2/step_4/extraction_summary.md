# Arc 10 v3.0 — Step 4 Extraction Summary

- Candidate clusters: [0]
- Features used: 37 (default 27 + arc-specific extras)
- Classifiers: ['lgbm', 'logistic', 'rf'] at Appendix A defaults

## Per-cluster AUC + threshold sweep

| Cluster | Classifier | n+ | mean_AUC | OOS_AUC | AUC-best thr | prec | recall | n_admit | F1-best thr | F1 | F1 prec | F1 rec |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| c0 | rf | 1493 | 0.5098 | 0.4935 | 0.50 | 0.459 | 0.285 | 778 | 0.05 | 0.477 | 0.477 | 1.000 |
| c0 | logistic | 1493 | 0.5131 | 0.5075 | 0.50 | 0.484 | 0.391 | 1010 | 0.20 | 0.477 | 0.477 | 0.999 |
| c0 | lgbm | 1493 | 0.5103 | 0.5003 | 0.50 | 0.474 | 0.475 | 1252 | 0.09 | 0.478 | 0.478 | 0.996 |

## Top-10 features per cluster (permutation importance on best classifier)


### c0 (best clf: logistic, mean AUC: 0.5131)

| Rank | Feature | imp_mean | imp_std |
|---:|---|---:|---:|
| 1 | L0_value | 0.05614 | 0.01215 |
| 2 | L1_value | 0.05236 | 0.01154 |
| 3 | atr14_at_signal | 0.04840 | 0.01113 |
| 4 | kijun_26_distance | 0.04337 | 0.01204 |
| 5 | L1_to_atr_proximity | 0.02105 | 0.01173 |
| 6 | d1_close_slope_magnitude | 0.01277 | 0.00416 |
| 7 | session_dead | 0.01226 | 0.00698 |
| 8 | session_tokyo | 0.00841 | 0.00944 |
| 9 | prior_session_low_distance | 0.00734 | 0.00456 |
| 10 | spread_percentile_100 | 0.00285 | 0.00162 |

**Swing-feature audit (per dispatch §Step 4):**
- Swing-derived features in top 10: ['L0_value', 'L1_value', 'kijun_26_distance', 'L1_to_atr_proximity', 'prior_session_low_distance']
- L1/L0 features carried verbatim from the DLR producer (signals/lchar_dlr_long.py). Producer uses ±3-bar swing-low detector with right-edge offset 4 (confirmation-lag, NOT Arc 9's centred-at-signal failure mode). Producer-level trace passed at intent stage.
- kijun_26_distance = kijun(high, low, 26).shift(1) — strictly prior bars. Causal.

## Gate readings (informational; arc continues regardless)

- L_PROTOCOL v3.0 §3 has no Step-4 AUC gate. Step 5 (WFO) is the only deployment gate.
- Cross-arc historical context: prior V-shape near-misses (Arc 7, Arc 10 v2.3) hovered Pipeline E ≈ 0.48-0.63 — feature-set bound.
