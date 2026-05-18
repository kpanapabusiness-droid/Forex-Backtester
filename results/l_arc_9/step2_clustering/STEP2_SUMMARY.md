# Arc 9 Step 2 - Clustering

Verdict: **PASS** (chosen K = 3)

## K-sweep (silhouette + gate)

| K | silhouette | max_cluster_frac | min_cluster_n | gate | fail_reason |
|---|---|---|---|---|---|
| 3 | 0.4247 | 0.4677 | 365 | PASS | - |
| 4 | 0.4193 | 0.3093 | 332 | PASS | - |
| 5 | 0.4032 | 0.2968 | 246 | PASS | - |
| 6 | 0.4168 | 0.2601 | 120 | PASS | - |
| 7 | 0.4055 | 0.2601 | 120 | PASS | - |

K selection rule: highest silhouette satisfying gate, smaller K wins within 0.01 absolute. **Selected K = 3**.

## Archetype assignment at K=3

| cluster_id | n | frac | mono | local_peaks | pullback | ttp_rel | archetype |
|---|---|---|---|---|---|---|---|
| 0 | 365 | 0.170 | 0.534 | 31.44 | 0.580 | 0.771 | **Unclassified** |
| 1 | 781 | 0.363 | 0.074 | 0.76 | 0.028 | 0.129 | **Early-peak hold** |
| 2 | 1007 | 0.468 | 0.528 | 7.28 | 0.510 | 0.477 | **Unclassified** |

## Path-feature degeneracy

| feature | max_modal_frac | degenerate (>0.80) |
|---|---|---|
| monotonicity_ratio_in_profit | 0.312 | no |
| local_peaks_count | 0.191 | no |
| pullback_magnitude_median | 0.358 | no |
| time_to_peak_mfe_relative | 0.191 | no |

Degenerate count: 0 (halt at >= 2)

## Outputs

- path_features.csv (2153 rows)
- silhouette_summary.csv
- silhouette_K<k>.txt (K=3..7)
- clusters_K<k>.csv (K=3..7)
- centroids_K<k>.csv (K=3..7)
- archetype_assignments.csv (at K=3)
