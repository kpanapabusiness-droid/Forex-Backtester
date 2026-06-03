# Arc 8 — Step 2 Cluster Summary

_Generated: 2026-05-22T07:12:46.839401+00:00Z_

- **Trades clustered:** 6,757
- **K sweep:** [2, 3, 4, 5, 6]
- **Primary K (max silhouette):** 3
- **Primary silhouette:** 0.3544

## Silhouette by K

| K | Silhouette | Inertia |
|---:|---:|---:|
| 2 | 0.3171 | 21572.64 |
| 3 | 0.3544 | 15275.65 |
| 4 | 0.3473 | 12500.05 |
| 5 | 0.3132 | 10620.05 |
| 6 | 0.3072 | 9335.54 |

## Per-cluster characterisation (primary K)

| Cluster | Size | Share | Modal tag (share) | Mono | Peaks | TTP_rel | DD_depth | Recovery | Mean R | p50 R | MFE p50 | MAE p50 | Bars held |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1,657 | 24.52% | Monotonic down (88.35%) | 0.138 | 0.77 | 0.085 | -1.068 | -5.238 | -1.000 | -1.000 | 0.138 | -1.171 | 5.4 |
| 1 | 3,560 | 52.69% | Mixed (80.79%) | 0.457 | 7.21 | 0.429 | -2.174 | -1.347 | -0.984 | -1.000 | 0.935 | -1.164 | 29.2 |
| 2 | 1,540 | 22.79% | V-shape recovery (43.12%) | 0.513 | 51.54 | 0.633 | -5.036 | 0.275 | +2.831 | +1.958 | 5.645 | -0.820 | 201.7 |

## Shape-tag value counts (across full pool)

| Tag | Count | Share |
|---|---:|---:|
| Mixed | 3,129 | 46.31% |
| Monotonic down | 1,930 | 28.56% |
| Bimodal | 837 | 12.39% |
| V-shape recovery | 664 | 9.83% |
| Stepwise climber | 197 | 2.92% |

## Quartile cache (drives shape-tag rules)

```
{
  "mono_q25": 0.3333333333333333,
  "mono_q75": 0.5072463768115942,
  "peaks_q25": 1.0,
  "peaks_q50": 5.0,
  "peaks_q75": 21.0,
  "dd_q25": -3.323956557714056,
  "rec_q75": -0.19655317091721475,
  "mfe_q50": 0.9161350132342304
}
```

Notes:
- Clustering features (5): mono, n_local_peaks, ttp_rel, dd_depth, recovery_score. Standardised via StandardScaler before KMeans.
- All K values reported above; primary K used for Step 3 capturability sweep.
- If max silhouette below 0.30, see L_PROTOCOL §2 Step 2 failure-diagnostics path (continue with single-cluster assignment, flag in this report).
