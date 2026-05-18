# Arc 11 — Step 2 path-shape clustering summary

Protocol: `L_ARC_PROTOCOL.md` v2.1.2 §§6, 11, 17

## Verdict
**PASS** — K=4, silhouette 0.4692; §6 gate clean; 0 assigned / 3 tentative / 0 boundary / 1 unassigned.

## K selection
- K chosen: **4**
- Silhouette at chosen K: 0.4692
- Tie tolerance applied (within ±0.01 absolute): no
  - k_best by raw silhouette: K=4; tied set: [4]; no parsimony divergence.

## Silhouette sweep

| K | silhouette | min cluster n | max cluster % | gate pass |
|---:|---:|---:|---:|:---:|
| 3 | 0.4506 | 687 | 36.93% | PASS |
| 4 | 0.4692 | 324 | 32.36% | PASS |
| 5 | 0.4588 | 204 | 32.23% | PASS |
| 6 | 0.4461 | 160 | 32.19% | PASS |
| 7 | 0.4559 | 93 | 31.45% | PASS |

## Degenerate features

| Feature | modal-bin mass | degenerate (>80%) |
|---|---:|:---:|
| monotonicity_ratio_in_profit | 31.19% | no |
| local_peaks_count | 34.32% | no |
| pullback_magnitude_median | 46.28% | no |
| time_to_peak_mfe_relative | 26.79% | no |

Degenerate count: **0 / 4** — none.

## Archetype assignments (K=4)

| cluster | n | size_frac | centroid (mono / peaks / pullback / ttp_rel) | archetype | status | notes |
|---:|---:|---:|---|---|:---:|---|
| 0 | 744 | 0.3236 | 0.0106 / 0.45 / 0.0079 / 0.0616 | tentative_Early-peak hold OR Peak-and-collapse | tentative | Step 3 pct_peak_and_collapse: <0.30 → Early-peak hold; >=0.50 → Peak-and-collapse |
| 1 | 324 | 0.1409 | 0.5362 / 32.48 / 0.5429 / 0.7682 | tentative_V-shape recovery | tentative | Step 3 MAE-before-peak >= 5 bars confirmation |
| 2 | 666 | 0.2897 | 0.5611 / 4.67 / 0.1495 / 0.3070 | unassigned (near: Early-peak hold OR Peak-and-collapse) | unassigned | near: Early-peak hold OR Peak-and-collapse |
| 3 | 565 | 0.2458 | 0.5092 / 9.50 / 0.7701 / 0.5637 | tentative_V-shape recovery | tentative | Step 3 MAE-before-peak >= 5 bars confirmation |

**Same-archetype clusters (downstream Step 3 evaluates per-cluster AND per-aggregate):**
- `tentative_V-shape recovery` → clusters [1, 3]

**Boundary clusters:** none.

**Unassigned clusters:**
- cluster 2: centroid (mono / peaks / pullback / ttp_rel) = 0.5611 / 4.67 / 0.1495 / 0.3070
  notes: near: Early-peak hold OR Peak-and-collapse

## Path-shape feature distributions (full pool)

| feature | p5 | p25 | p50 | p75 | p95 |
|---|---:|---:|---:|---:|---:|
| monotonicity_ratio_in_profit | 0 | 0 | 0.4889 | 0.5356 | 0.6858 |
| local_peaks_count | 0 | 1 | 3 | 11.5 | 36 |
| pullback_magnitude_median | 0 | 0 | 0.2061 | 0.5413 | 0.9973 |
| time_to_peak_mfe_relative | 0 | 0 | 0.3333 | 0.5872 | 0.9184 |

## Determinism

**Gate: PASS**

| File | run 1 sha256 | run 2 sha256 | match |
|---|---|---|:---:|
| `archetype_assignments.csv` | `43e522eca40a2ca4…` | `43e522eca40a2ca4…` | YES |
| `centroids_K3.csv` | `98e13a56d551f65e…` | `98e13a56d551f65e…` | YES |
| `centroids_K4.csv` | `2ba49fcb43bebe6d…` | `2ba49fcb43bebe6d…` | YES |
| `centroids_K5.csv` | `9bfd1ce23fe928e7…` | `9bfd1ce23fe928e7…` | YES |
| `centroids_K6.csv` | `54a6a43e2549e371…` | `54a6a43e2549e371…` | YES |
| `centroids_K7.csv` | `8f510594b3ab4d98…` | `8f510594b3ab4d98…` | YES |
| `clusters_K3.csv` | `63c5c82c92c08b83…` | `63c5c82c92c08b83…` | YES |
| `clusters_K4.csv` | `a380b57badf84dd7…` | `a380b57badf84dd7…` | YES |
| `clusters_K5.csv` | `c8f953b10c896c89…` | `c8f953b10c896c89…` | YES |
| `clusters_K6.csv` | `ec692dd566e19ed8…` | `ec692dd566e19ed8…` | YES |
| `clusters_K7.csv` | `719247fe745c541f…` | `719247fe745c541f…` | YES |
| `path_features.csv` | `6b334a01d04832f3…` | `6b334a01d04832f3…` | YES |
| `silhouette_K3.txt` | `fd23f29ba2c16acd…` | `fd23f29ba2c16acd…` | YES |
| `silhouette_K4.txt` | `cfa61795117a1d3c…` | `cfa61795117a1d3c…` | YES |
| `silhouette_K5.txt` | `ef236476896c55fd…` | `ef236476896c55fd…` | YES |
| `silhouette_K6.txt` | `003a1996575ab6bb…` | `003a1996575ab6bb…` | YES |
| `silhouette_K7.txt` | `4d50d417afe82904…` | `4d50d417afe82904…` | YES |
| `silhouette_sweep.csv` | `37f734ec37fdf675…` | `37f734ec37fdf675…` | YES |

## Files

- `results\l_arc_11\step2/archetype_assignments.csv`
- `results\l_arc_11\step2/centroids_K3.csv`
- `results\l_arc_11\step2/centroids_K4.csv`
- `results\l_arc_11\step2/centroids_K5.csv`
- `results\l_arc_11\step2/centroids_K6.csv`
- `results\l_arc_11\step2/centroids_K7.csv`
- `results\l_arc_11\step2/clusters_K3.csv`
- `results\l_arc_11\step2/clusters_K4.csv`
- `results\l_arc_11\step2/clusters_K5.csv`
- `results\l_arc_11\step2/clusters_K6.csv`
- `results\l_arc_11\step2/clusters_K7.csv`
- `results\l_arc_11\step2/path_features.csv`
- `results\l_arc_11\step2/silhouette_K3.txt`
- `results\l_arc_11\step2/silhouette_K4.txt`
- `results\l_arc_11\step2/silhouette_K5.txt`
- `results\l_arc_11\step2/silhouette_K6.txt`
- `results\l_arc_11\step2/silhouette_K7.txt`
- `results\l_arc_11\step2/silhouette_sweep.csv`
- `results\l_arc_11\step2/feature_histograms.png`
- `results\l_arc_11\step2/STEP2_SUMMARY.md`
- `configs/l_arc_11/step2.yaml`
- `scripts/l_arc_11/step2_cluster.py`

## Step 1 commit
hash: `766b452125c7c04ccfb172c57eb0a4480765d5b7`

## Step 2 commit
hash: _pending_

