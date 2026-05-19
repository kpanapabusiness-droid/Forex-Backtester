# Arc 10 — Step 2 path-shape clustering summary

Protocol: `L_ARC_PROTOCOL.md` v2.1.2 §§6, 11, 17

## Verdict
**PASS** — K=3, silhouette 0.4525; §6 gate clean; 0 assigned / 2 tentative / 0 boundary / 1 unassigned.

## K selection
- K chosen: **3**
- Silhouette at chosen K: 0.4525
- Tie tolerance applied (within ±0.01 absolute): yes
  - k_best by raw silhouette: K=5 (0.4622); tied set: [3, 4, 5, 7]; smaller K preferred (parsimony, v2.1 Open-12 closure).

## Silhouette sweep

| K | silhouette | min cluster n | max cluster % | gate pass |
|---:|---:|---:|---:|:---:|
| 3 | 0.4525 | 228 | 38.40% | PASS |
| 4 | 0.4585 | 114 | 32.79% | PASS |
| 5 | 0.4622 | 45 | 32.54% | PASS |
| 6 | 0.4508 | 34 | 32.54% | PASS |
| 7 | 0.4567 | 30 | 32.29% | PASS |

## Degenerate features

| Feature | modal-bin mass | degenerate (>80%) |
|---|---:|:---:|
| monotonicity_ratio_in_profit | 31.80% | no |
| local_peaks_count | 33.54% | no |
| pullback_magnitude_median | 45.89% | no |
| time_to_peak_mfe_relative | 25.19% | no |

Degenerate count: **0 / 4** — none.

## Archetype assignments (K=3)

| cluster | n | size_frac | centroid (mono / peaks / pullback / ttp_rel) | archetype | status | notes |
|---:|---:|---:|---|---|:---:|---|
| 0 | 266 | 0.3317 | 0.0160 / 0.53 / 0.0068 / 0.0656 | tentative_Early-peak hold OR Peak-and-collapse | tentative | Step 3 pct_peak_and_collapse: <0.30 → Early-peak hold; >=0.50 → Peak-and-collapse |
| 1 | 228 | 0.2843 | 0.5211 / 21.92 / 0.7338 / 0.6867 | tentative_V-shape recovery | tentative | Step 3 MAE-before-peak >= 5 bars confirmation |
| 2 | 308 | 0.3840 | 0.5470 / 5.56 / 0.2643 / 0.3621 | unassigned | unassigned |  |

**Same-archetype clusters:** none.

**Boundary clusters:** none.

**Unassigned clusters:**
- cluster 2: centroid (mono / peaks / pullback / ttp_rel) = 0.5470 / 5.56 / 0.2643 / 0.3621

## Path-shape feature distributions (full pool)

| feature | p5 | p25 | p50 | p75 | p95 |
|---|---:|---:|---:|---:|---:|
| monotonicity_ratio_in_profit | 0 | 0 | 0.4851 | 0.5396 | 0.6667 |
| local_peaks_count | 0 | 1 | 4 | 11.75 | 34 |
| pullback_magnitude_median | 0 | 0 | 0.2107 | 0.5388 | 0.9751 |
| time_to_peak_mfe_relative | 0 | 0 | 0.3333 | 0.5905 | 0.8961 |

## Determinism

**Gate: PASS**

| File | run 1 sha256 | run 2 sha256 | match |
|---|---|---|:---:|
| `archetype_assignments.csv` | `e99b377e79f74fbc…` | `e99b377e79f74fbc…` | YES |
| `centroids_K3.csv` | `8bd36d2a4a329079…` | `8bd36d2a4a329079…` | YES |
| `centroids_K4.csv` | `00f2d21e5dece3da…` | `00f2d21e5dece3da…` | YES |
| `centroids_K5.csv` | `a88d55d0c82f5219…` | `a88d55d0c82f5219…` | YES |
| `centroids_K6.csv` | `78731baef8215311…` | `78731baef8215311…` | YES |
| `centroids_K7.csv` | `eb17f21d2465b282…` | `eb17f21d2465b282…` | YES |
| `clusters_K3.csv` | `93199b1c90e0723f…` | `93199b1c90e0723f…` | YES |
| `clusters_K4.csv` | `4fa69c14d0bf1611…` | `4fa69c14d0bf1611…` | YES |
| `clusters_K5.csv` | `b51c1980b3461e2f…` | `b51c1980b3461e2f…` | YES |
| `clusters_K6.csv` | `240998336f68f918…` | `240998336f68f918…` | YES |
| `clusters_K7.csv` | `be995a1ebdebc41a…` | `be995a1ebdebc41a…` | YES |
| `path_features.csv` | `eeceb49abaa3af4d…` | `eeceb49abaa3af4d…` | YES |
| `silhouette_K3.txt` | `ade39da96257203d…` | `ade39da96257203d…` | YES |
| `silhouette_K4.txt` | `5aef39cffca15cba…` | `5aef39cffca15cba…` | YES |
| `silhouette_K5.txt` | `b2a8612442a4aec7…` | `b2a8612442a4aec7…` | YES |
| `silhouette_K6.txt` | `6f55056a740b51bd…` | `6f55056a740b51bd…` | YES |
| `silhouette_K7.txt` | `2b25247ac31d17b4…` | `2b25247ac31d17b4…` | YES |
| `silhouette_sweep.csv` | `9211a9a95023e355…` | `9211a9a95023e355…` | YES |

## Files

- `results\l_arc_10\step2/archetype_assignments.csv`
- `results\l_arc_10\step2/centroids_K3.csv`
- `results\l_arc_10\step2/centroids_K4.csv`
- `results\l_arc_10\step2/centroids_K5.csv`
- `results\l_arc_10\step2/centroids_K6.csv`
- `results\l_arc_10\step2/centroids_K7.csv`
- `results\l_arc_10\step2/clusters_K3.csv`
- `results\l_arc_10\step2/clusters_K4.csv`
- `results\l_arc_10\step2/clusters_K5.csv`
- `results\l_arc_10\step2/clusters_K6.csv`
- `results\l_arc_10\step2/clusters_K7.csv`
- `results\l_arc_10\step2/path_features.csv`
- `results\l_arc_10\step2/silhouette_K3.txt`
- `results\l_arc_10\step2/silhouette_K4.txt`
- `results\l_arc_10\step2/silhouette_K5.txt`
- `results\l_arc_10\step2/silhouette_K6.txt`
- `results\l_arc_10\step2/silhouette_K7.txt`
- `results\l_arc_10\step2/silhouette_sweep.csv`
- `results\l_arc_10\step2/feature_histograms.png`
- `results\l_arc_10\step2/STEP2_SUMMARY.md`
- `configs/l_arc_10/step2.yaml`
- `scripts/l_arc_10/step2_cluster.py`

## Step 1 commit
hash: `362a085b984f8402254acb154593a629b20a997e`

## Step 2 commit
hash: _pending_

