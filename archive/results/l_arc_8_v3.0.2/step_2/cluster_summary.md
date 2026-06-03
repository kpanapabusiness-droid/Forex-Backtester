# l_arc_8_v3.0.2 — Step 2 Cluster Summary

_Generated: 2026-05-25T09:13:22.516060+00:00Z_

- **Trades clustered:** 6,612
- **K sweep:** [2, 3, 4, 5, 6]
- **Primary K (max silhouette):** 3
- **Primary silhouette:** 0.3624

## Arc 8 v3.0 reference (UTC + multi_tf all-NaN) — informational

Arc 8 v3.0 surfaced 3 clusters: c0 Monotonic_down (~25%), c1 Choppy (~52%), c2 V-shape (~23%). Under 5ers_eet + multi_tf restored + W1 canonical, the K-sweep and cluster boundaries are re-determined from scratch on path-shape features (which are R-unit, signal-pool-derived — independent of bar-storage convention). Material archetype shifts are flagged below.

## Silhouette by K

| K | Silhouette | Inertia |
|---:|---:|---:|
| 2 | 0.3269 | 20688.82 |
| 3 | 0.3624 | 14625.70 |
| 4 | 0.3611 | 11832.55 |
| 5 | 0.3161 | 10048.40 |
| 6 | 0.3161 | 8721.01 |

## Per-cluster characterisation (primary K)

| Cluster | Size | Share | Modal tag (share) | Mono | Peaks | TTP_rel | DD_depth | Recovery | Mean R | p50 R | MFE p50 | MAE p50 | Bars held |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3,418 | 51.69% | Mixed (81.30%) | 0.454 | 7.04 | 0.424 | -2.187 | -1.377 | -0.991 | -1.000 | 0.968 | -1.171 | 28.7 |
| 1 | 1,628 | 24.62% | Monotonic down (89.13%) | 0.135 | 0.73 | 0.082 | -1.055 | -5.175 | -0.999 | -1.000 | 0.136 | -1.176 | 5.2 |
| 2 | 1,566 | 23.68% | Bimodal (41.76%) | 0.514 | 50.05 | 0.639 | -5.002 | 0.260 | +2.689 | +1.649 | 5.663 | -0.864 | 198.9 |

## Shape-tag value counts (across full pool)

| Tag | Count | Share |
|---|---:|---:|
| Mixed | 3,036 | 45.92% |
| Monotonic down | 1,913 | 28.93% |
| Bimodal | 826 | 12.49% |
| V-shape recovery | 627 | 9.48% |
| Stepwise climber | 210 | 3.18% |

## Quartile cache (drives shape-tag rules)

```
{
  "mono_q25": 0.3333333333333333,
  "mono_q75": 0.5098039215686274,
  "peaks_q25": 1.0,
  "peaks_q50": 5.0,
  "peaks_q75": 21.0,
  "dd_q25": -3.374276286047506,
  "rec_q75": -0.1953318421125163,
  "mfe_q50": 0.9610315483327853
}
```

Notes:
- Clustering features (5): mono, n_local_peaks, ttp_rel, dd_depth, recovery_score. Standardised via StandardScaler before KMeans.
- All K values reported above; primary K used for Step 3 capturability sweep.
- If max silhouette below 0.30, see L_PROTOCOL §2 Step 2 failure-diagnostics path.
