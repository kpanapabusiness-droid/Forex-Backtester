# Arc 8 — Step 2 path-shape clustering summary

Protocol: `L_ARC_PROTOCOL.md` v2.3 stack (base v2.1.2 + v2.2 + v2.3 amendments); §§6, 11, 17 unchanged

## Verdict
**PASS** — K=4, silhouette 0.4762; §6 gate clean; 0 assigned / 3 tentative / 0 boundary / 1 unassigned.

## K selection
- K chosen: **4**
- Silhouette at chosen K: 0.4762
- Tie tolerance applied (within ±0.01 absolute): no
  - k_best by raw silhouette: K=4; tied set: [4]; no parsimony divergence.

## Silhouette sweep

| K | silhouette | min cluster n | max cluster % | gate pass |
|---:|---:|---:|---:|:---:|
| 3 | 0.4622 | 221 | 46.27% | PASS |
| 4 | 0.4762 | 177 | 32.33% | PASS |
| 5 | 0.4578 | 149 | 32.03% | PASS |
| 6 | 0.4630 | 54 | 31.35% | PASS |
| 7 | 0.4570 | 50 | 31.27% | PASS |

## Degenerate features

| Feature | modal-bin mass | degenerate (>80%) |
|---|---:|:---:|
| monotonicity_ratio_in_profit | 31.42% | no |
| local_peaks_count | 34.36% | no |
| pullback_magnitude_median | 44.76% | no |
| time_to_peak_mfe_relative | 26.30% | no |

Degenerate count: **0 / 4** — none.

## Archetype assignments (K=4)

| cluster | n | size_frac | centroid (mono / peaks / pullback / ttp_rel) | archetype | status | notes |
|---:|---:|---:|---|---|:---:|---|
| 0 | 316 | 0.2381 | 0.5652 / 4.26 / 0.1237 / 0.3138 | unassigned (near: Early-peak hold OR Peak-and-collapse) | unassigned | near: Early-peak hold OR Peak-and-collapse |
| 1 | 177 | 0.1334 | 0.5397 / 33.82 / 0.5484 / 0.7783 | tentative_V-shape recovery | tentative | Step 3 MAE-before-peak >= 5 bars confirmation |
| 2 | 429 | 0.3233 | 0.0099 / 0.48 / 0.0078 / 0.0592 | tentative_Early-peak hold OR Peak-and-collapse | tentative | Step 3 pct_peak_and_collapse: <0.30 → Early-peak hold; >=0.50 → Peak-and-collapse |
| 3 | 405 | 0.3052 | 0.5070 / 10.42 / 0.7141 / 0.5450 | tentative_V-shape recovery | tentative | Step 3 MAE-before-peak >= 5 bars confirmation |

**Same-archetype clusters (downstream Step 3 evaluates per-cluster AND per-aggregate):**
- `tentative_V-shape recovery` → clusters [1, 3]

**Boundary clusters:** none.

**Unassigned clusters:**
- cluster 0: centroid (mono / peaks / pullback / ttp_rel) = 0.5652 / 4.26 / 0.1237 / 0.3138
  notes: near: Early-peak hold OR Peak-and-collapse

## Path-shape feature distributions (full pool)

| feature | p5 | p25 | p50 | p75 | p95 |
|---|---:|---:|---:|---:|---:|
| monotonicity_ratio_in_profit | 0 | 0 | 0.491 | 0.5375 | 0.6667 |
| local_peaks_count | 0 | 1 | 4 | 12 | 35 |
| pullback_magnitude_median | 0 | 0 | 0.2572 | 0.5697 | 0.9823 |
| time_to_peak_mfe_relative | 0 | 0 | 0.3415 | 0.6057 | 0.9212 |

## Determinism

**Gate: PASS**

| File | run 1 sha256 | run 2 sha256 | match |
|---|---|---|:---:|
| `archetype_assignments.csv` | `a950647bb4a01c9b…` | `a950647bb4a01c9b…` | YES |
| `centroids_K3.csv` | `1cfd875e279b4329…` | `1cfd875e279b4329…` | YES |
| `centroids_K4.csv` | `d2a4ba2037be0865…` | `d2a4ba2037be0865…` | YES |
| `centroids_K5.csv` | `2c1545f2309c22ef…` | `2c1545f2309c22ef…` | YES |
| `centroids_K6.csv` | `09df4b6a22e80819…` | `09df4b6a22e80819…` | YES |
| `centroids_K7.csv` | `ff7c46bc942f5124…` | `ff7c46bc942f5124…` | YES |
| `clusters_K3.csv` | `fd7e6bdc1a981036…` | `fd7e6bdc1a981036…` | YES |
| `clusters_K4.csv` | `b81c3c268291d453…` | `b81c3c268291d453…` | YES |
| `clusters_K5.csv` | `cd95594eac4fc778…` | `cd95594eac4fc778…` | YES |
| `clusters_K6.csv` | `c83430b1fdc23985…` | `c83430b1fdc23985…` | YES |
| `clusters_K7.csv` | `46ff10b909488523…` | `46ff10b909488523…` | YES |
| `path_features.csv` | `735d0a45e597ac27…` | `735d0a45e597ac27…` | YES |
| `silhouette_K3.txt` | `f3e2cc8381f8111c…` | `f3e2cc8381f8111c…` | YES |
| `silhouette_K4.txt` | `7f5118ef5f641343…` | `7f5118ef5f641343…` | YES |
| `silhouette_K5.txt` | `b87e0479604b49ff…` | `b87e0479604b49ff…` | YES |
| `silhouette_K6.txt` | `0df6c7babbc0c37a…` | `0df6c7babbc0c37a…` | YES |
| `silhouette_K7.txt` | `73640130ab624ece…` | `73640130ab624ece…` | YES |
| `silhouette_sweep.csv` | `6e0eba2aedb23b88…` | `6e0eba2aedb23b88…` | YES |

## Files

- `results\l_arc_8\step2/archetype_assignments.csv`
- `results\l_arc_8\step2/centroids_K3.csv`
- `results\l_arc_8\step2/centroids_K4.csv`
- `results\l_arc_8\step2/centroids_K5.csv`
- `results\l_arc_8\step2/centroids_K6.csv`
- `results\l_arc_8\step2/centroids_K7.csv`
- `results\l_arc_8\step2/clusters_K3.csv`
- `results\l_arc_8\step2/clusters_K4.csv`
- `results\l_arc_8\step2/clusters_K5.csv`
- `results\l_arc_8\step2/clusters_K6.csv`
- `results\l_arc_8\step2/clusters_K7.csv`
- `results\l_arc_8\step2/path_features.csv`
- `results\l_arc_8\step2/silhouette_K3.txt`
- `results\l_arc_8\step2/silhouette_K4.txt`
- `results\l_arc_8\step2/silhouette_K5.txt`
- `results\l_arc_8\step2/silhouette_K6.txt`
- `results\l_arc_8\step2/silhouette_K7.txt`
- `results\l_arc_8\step2/silhouette_sweep.csv`
- `results\l_arc_8\step2/feature_histograms.png`
- `results\l_arc_8\step2/STEP2_SUMMARY.md`
- `configs/l_arc_8/step2.yaml`
- `scripts/l_arc_8/step2_cluster.py`

## Step 1 commit
hash: `3c5f943c08bd61b143d665a8a9b5d05f5ee5ae75`

## Step 2 commit
hash: _pending_

