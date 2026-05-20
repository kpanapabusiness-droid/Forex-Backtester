# Arc 2 redo2 — Step 2 path-shape clustering summary

Protocol: `L_ARC_PROTOCOL.md` v2.1.1 §§1, 6, 11

**Step 2 disposition: PASS**

Gate (§6): silhouette >= 0.3, max cluster fraction <= 0.9, all clusters >= 30 trades. 5 of 5 sweep K values pass.
Chosen K: **4** (silhouette 0.4834, max_frac 0.3482, min_count 1706). Selection rule (v2.1.1 §6): k_best=4 at silhouette 0.4834; tied set (within ±0.01 absolute): [4]; selected smallest K within tolerance.

## Path-shape feature distributions

| Feature | p1 | p5 | p25 | p50 | p75 | p95 | p99 | %=0 | %=1 | modal-bin mass | degenerate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| monotonicity_ratio_in_profit | 0 | 0 | 0 | 0.4865 | 0.5385 | 0.6667 | 1 | 31.16% | 2.92% | 31.16% | no |
| local_peaks_count | 0 | 0 | 1 | 3 | 10 | 24 | 31 | 22.17% | 12.59% | 22.17% | no |
| pullback_magnitude_median | 0 | 0 | 0 | 0.1199 | 0.3414 | 0.7244 | 1.17 | 43.85% | 0.00% | 46.27% | no |
| time_to_peak_mfe_relative | 0 | 0 | 0 | 0.3223 | 0.6169 | 0.9421 | 0.9835 | 28.27% | 0.00% | 28.33% | no |

Degenerate features (modal bin > 80%): 0 of 4 — none

## K-sweep results

| K | silhouette | max cluster fraction | min cluster count | gate |
|---:|---:|---:|---:|:---|
| 3 | 0.4615 | 0.3671 | 3727 | PASS |
| 4 | 0.4834 | 0.3482 | 1706 | PASS |
| 5 | 0.4657 | 0.3154 | 871 | PASS |
| 6 | 0.4689 | 0.3139 | 536 | PASS |
| 7 | 0.4621 | 0.3129 | 373 | PASS |

## Chosen K=4 — cluster centroids and assignments

| cluster_id | monotonicity | local_peaks | pullback | time_to_peak_rel | size | fraction | archetype | matching rule | boundary |
|---:|---:|---:|---:|---:|---:|---:|---|---|:---:|
| 0 | 0.5482 | 4.73 | 0.1300 | 0.3408 | 4270 | 0.3482 | unclassified | no §11 pattern matched | no |
| 1 | 0.5382 | 20.07 | 0.3398 | 0.7744 | 2285 | 0.1863 | Stepwise climber | monotonicity>=0.50 AND local_peaks in [5,30] AND pullback<=0.5 AND time_to_peak_rel>=0.50 | no |
| 2 | 0.0147 | 0.53 | 0.0112 | 0.0432 | 4001 | 0.3263 | Early-peak hold OR Peak-and-collapse | time_to_peak_rel<=0.30 (disambiguation needs Step-3 pct_peak_and_collapse) | YES |
| 3 | 0.5121 | 7.46 | 0.7002 | 0.5905 | 1706 | 0.1391 | unclassified | no §11 pattern matched | no |

Boundary-flagged clusters: 1 (2)

## Cluster interpretation notes

- Cluster 0 (size 0.3482): unclassified — moderate climber, ttp in [0.30, 0.50] gap between Early-peak hold and Monotone ascent §11 patterns. Centroid mono=0.5482, peaks=4.73, pull=0.1300, ttp=0.3408.
- Cluster 2 (size 0.3263): centroid mono=0.0147, pull=0.0112 are near zero → trades that mostly never went in profit (degenerate path-shape). Step-3 fwd_mfe_p50 check will likely separate these from genuine Early-peak hold trades.
- Cluster 3 (size 0.1391): unclassified — looks like Stepwise climber with deeper pullbacks (pullback 0.70 > 0.5 threshold). Centroid mono=0.5121, peaks=7.46, pull=0.7002, ttp=0.5905.

## PR #129 concordance (K=5)

- Status: **MATCH**
- Detail: closest cluster_id=3: mono=0.5415 (target 0.54), local_peaks=22.02 (target 26.5), size_frac=0.1441 (target 0.155); checks: mono<=0.10 True, peaks<=10 True, size<=0.05 True
- Reference: PR #129 reported K=5 archetype 1 with monotonicity ~0.54, local_peaks ~26.5, size_fraction ~0.155.

## Jaccard overlap vs prior arc_2_redo K=4

| fork \ prior | 0 | 1 | 2 | 3 | best |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.0002 | 0.9813 | 0.0000 | 0.0025 | prior 1 @ 0.9813 |
| 1 | 0.0005 | 0.0029 | 0.9848 | 0.0000 | prior 2 @ 0.9848 |
| 2 | 0.0005 | 0.0004 | 0.0000 | 0.9933 | prior 3 @ 0.9933 |
| 3 | 0.9667 | 0.0062 | 0.0035 | 0.0000 | prior 0 @ 0.9667 |

Average best-match Jaccard: **0.9815**. Interpretation: near-1.0 average → fork clustering decomposes the pool identically to the prior redo (minor reassignments at cluster boundaries are expected due to the 1-bar definition change in is_held vs still_open).

## Determinism (CSV byte-identical)

**Gate: PASS**

Run-1 CSV hashes:
- `archetype_assignments.csv`: `f385f610384351ddf079b6be652fd02fdc1ddcfb29fa8cc1ee6b12f001c9738a`
- `centroids_K3.csv`: `c8b3f3c754cfac4cd9ca217fdfbe946f4608da595445adec6a57b977af95c966`
- `centroids_K4.csv`: `a06393be96a59d0381a5a7a70d5c1ed8520bf36a6565604c4e5718128fa785b2`
- `centroids_K5.csv`: `c351e3015b7333ae9494f9e435723fdde18fe0ce1a7a405cdfbd1723686a960c`
- `centroids_K6.csv`: `1df07ecc77216c2b9e3ed4f9359d761e7150234b637d71d0ee48addf5e71015e`
- `centroids_K7.csv`: `3151deafe2b47a91d5e9302fea56c8b060bbd5e4fd3e9fe46ec27b1911658bc0`
- `clusters_K3.csv`: `e5ad098f29491226f5ce4328b43f4bfcde3dde64ae8cfdd165a0ed83c6251fa0`
- `clusters_K4.csv`: `07c6b506e6111a5c97d5d80011c78b37e6f668d650f2b7903ade6f469e94c471`
- `clusters_K5.csv`: `0accf498aba3bc9cc281f521b9b057bfd95b0ac12497664d08b36168d276632f`
- `clusters_K6.csv`: `0b7ade7eec8ce7f0786e9105e4c10d291f3e495c5c00620b28b68c7377cfe7fa`
- `clusters_K7.csv`: `e124da9f353f6598bb0ffa54b83220d9e25587d70238811cfeead6b1dbb67e53`
- `feature_diagnostics.csv`: `6d7d450954d992f5ca891cc52c6c3ef4138d345a9b6377e3095179f5645ef103`
- `jaccard_vs_prior_K4.csv`: `38af054cc32b159aa99f59b985080e2042a56a9b970c3986a98e1655d411bdd5`
- `path_features.csv`: `95636c84de1d63bcdf57866574b14bbcfcb2ee5c04fa32081083efa927a5a618`
- `silhouette_sweep.csv`: `82c18d817ecb052897b8a743214fc6cfb557d69307b5ae44f8a9091d43688002`

Run-2 CSV hashes:
- `archetype_assignments.csv`: `f385f610384351ddf079b6be652fd02fdc1ddcfb29fa8cc1ee6b12f001c9738a` (MATCH)
- `centroids_K3.csv`: `c8b3f3c754cfac4cd9ca217fdfbe946f4608da595445adec6a57b977af95c966` (MATCH)
- `centroids_K4.csv`: `a06393be96a59d0381a5a7a70d5c1ed8520bf36a6565604c4e5718128fa785b2` (MATCH)
- `centroids_K5.csv`: `c351e3015b7333ae9494f9e435723fdde18fe0ce1a7a405cdfbd1723686a960c` (MATCH)
- `centroids_K6.csv`: `1df07ecc77216c2b9e3ed4f9359d761e7150234b637d71d0ee48addf5e71015e` (MATCH)
- `centroids_K7.csv`: `3151deafe2b47a91d5e9302fea56c8b060bbd5e4fd3e9fe46ec27b1911658bc0` (MATCH)
- `clusters_K3.csv`: `e5ad098f29491226f5ce4328b43f4bfcde3dde64ae8cfdd165a0ed83c6251fa0` (MATCH)
- `clusters_K4.csv`: `07c6b506e6111a5c97d5d80011c78b37e6f668d650f2b7903ade6f469e94c471` (MATCH)
- `clusters_K5.csv`: `0accf498aba3bc9cc281f521b9b057bfd95b0ac12497664d08b36168d276632f` (MATCH)
- `clusters_K6.csv`: `0b7ade7eec8ce7f0786e9105e4c10d291f3e495c5c00620b28b68c7377cfe7fa` (MATCH)
- `clusters_K7.csv`: `e124da9f353f6598bb0ffa54b83220d9e25587d70238811cfeead6b1dbb67e53` (MATCH)
- `feature_diagnostics.csv`: `6d7d450954d992f5ca891cc52c6c3ef4138d345a9b6377e3095179f5645ef103` (MATCH)
- `jaccard_vs_prior_K4.csv`: `38af054cc32b159aa99f59b985080e2042a56a9b970c3986a98e1655d411bdd5` (MATCH)
- `path_features.csv`: `95636c84de1d63bcdf57866574b14bbcfcb2ee5c04fa32081083efa927a5a618` (MATCH)
- `silhouette_sweep.csv`: `82c18d817ecb052897b8a743214fc6cfb557d69307b5ae44f8a9091d43688002` (MATCH)

