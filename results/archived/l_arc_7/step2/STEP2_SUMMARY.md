# Arc 7 — Step 2 path-shape clustering summary

Protocol: `L_ARC_PROTOCOL.md` v2.1.2 §§6, 11, 17

## Verdict
**PASS** — K=4, silhouette 0.4263; §6 gate clean; 0 assigned / 4 tentative / 0 boundary / 0 unassigned.

## K selection
- K chosen: **4**
- Silhouette at chosen K: 0.4263
- Tie tolerance applied (within ±0.01 absolute): no
  - k_best by raw silhouette: K=4; tied set: [4, 5, 6]; no parsimony divergence.

## Silhouette sweep

| K | silhouette | min cluster n | max cluster % | gate pass |
|---:|---:|---:|---:|:---:|
| 3 | 0.4091 | 375 | 39.98% | PASS |
| 4 | 0.4263 | 185 | 29.89% | PASS |
| 5 | 0.4167 | 120 | 27.17% | PASS |
| 6 | 0.4242 | 67 | 26.40% | PASS |
| 7 | 0.4123 | 67 | 26.32% | PASS |

## Degenerate features

| Feature | modal-bin mass | degenerate (>80%) |
|---|---:|:---:|
| monotonicity_ratio_in_profit | 26.32% | no |
| local_peaks_count | 28.80% | no |
| pullback_magnitude_median | 41.77% | no |
| time_to_peak_mfe_relative | 20.89% | no |

Degenerate count: **0 / 4** — none.

## Archetype assignments (K=4)

| cluster | n | size_frac | centroid (mono / peaks / pullback / ttp_rel) | archetype | status | notes |
|---:|---:|---:|---|---|:---:|---|
| 0 | 385 | 0.2989 | 0.5547 / 4.44 / 0.1578 / 0.2977 | tentative_Early-peak hold OR Peak-and-collapse | tentative | Step 3 pct_peak_and_collapse: <0.30 → Early-peak hold; >=0.50 → Peak-and-collapse |
| 1 | 185 | 0.1436 | 0.5363 / 33.51 / 0.5672 / 0.7302 | tentative_V-shape recovery | tentative | Step 3 MAE-before-peak >= 5 bars confirmation |
| 2 | 353 | 0.2741 | 0.0136 / 0.59 / 0.0089 / 0.0817 | tentative_Early-peak hold OR Peak-and-collapse | tentative | Step 3 pct_peak_and_collapse: <0.30 → Early-peak hold; >=0.50 → Peak-and-collapse |
| 3 | 365 | 0.2834 | 0.4975 / 9.84 / 0.7622 / 0.5673 | tentative_V-shape recovery | tentative | Step 3 MAE-before-peak >= 5 bars confirmation |

**Same-archetype clusters (downstream Step 3 evaluates per-cluster AND per-aggregate):**
- `tentative_Early-peak hold OR Peak-and-collapse` → clusters [0, 2]
- `tentative_V-shape recovery` → clusters [1, 3]

**Boundary clusters:** none.

**Unassigned clusters:** none.

## Path-shape feature distributions (full pool)

| feature | p5 | p25 | p50 | p75 | p95 |
|---|---:|---:|---:|---:|---:|
| monotonicity_ratio_in_profit | 0 | 0 | 0.496 | 0.5333 | 0.6883 |
| local_peaks_count | 0 | 1 | 4 | 12 | 36 |
| pullback_magnitude_median | 0 | 0 | 0.2993 | 0.5835 | 1.031 |
| time_to_peak_mfe_relative | 0 | 0.125 | 0.3636 | 0.6 | 0.8741 |

## Determinism

**Gate: PASS**

| File | run 1 sha256 | run 2 sha256 | match |
|---|---|---|:---:|
| `archetype_assignments.csv` | `65720a9e8ea9323c…` | `65720a9e8ea9323c…` | YES |
| `centroids_K3.csv` | `6ae3fadcd308c37b…` | `6ae3fadcd308c37b…` | YES |
| `centroids_K4.csv` | `596bed0e6af29910…` | `596bed0e6af29910…` | YES |
| `centroids_K5.csv` | `e6425e024187abeb…` | `e6425e024187abeb…` | YES |
| `centroids_K6.csv` | `ec2c4663a64c9bd6…` | `ec2c4663a64c9bd6…` | YES |
| `centroids_K7.csv` | `2e93d8a39358b2ae…` | `2e93d8a39358b2ae…` | YES |
| `clusters_K3.csv` | `57757f4e9169b6db…` | `57757f4e9169b6db…` | YES |
| `clusters_K4.csv` | `a931e7f373a19d2b…` | `a931e7f373a19d2b…` | YES |
| `clusters_K5.csv` | `5fd8be3b38ae5aa0…` | `5fd8be3b38ae5aa0…` | YES |
| `clusters_K6.csv` | `03b20fe6d62430cb…` | `03b20fe6d62430cb…` | YES |
| `clusters_K7.csv` | `fc144b813a224005…` | `fc144b813a224005…` | YES |
| `path_features.csv` | `d85f5700d715071e…` | `d85f5700d715071e…` | YES |
| `silhouette_K3.txt` | `6fdb7b62a998caba…` | `6fdb7b62a998caba…` | YES |
| `silhouette_K4.txt` | `91cc886bdb81923a…` | `91cc886bdb81923a…` | YES |
| `silhouette_K5.txt` | `ca01469f80c4b76f…` | `ca01469f80c4b76f…` | YES |
| `silhouette_K6.txt` | `8c1293c686457411…` | `8c1293c686457411…` | YES |
| `silhouette_K7.txt` | `56d9808dca35338a…` | `56d9808dca35338a…` | YES |
| `silhouette_sweep.csv` | `48b7032fa7ed2318…` | `48b7032fa7ed2318…` | YES |

## Files

- `results\l_arc_7\step2/archetype_assignments.csv`
- `results\l_arc_7\step2/centroids_K3.csv`
- `results\l_arc_7\step2/centroids_K4.csv`
- `results\l_arc_7\step2/centroids_K5.csv`
- `results\l_arc_7\step2/centroids_K6.csv`
- `results\l_arc_7\step2/centroids_K7.csv`
- `results\l_arc_7\step2/clusters_K3.csv`
- `results\l_arc_7\step2/clusters_K4.csv`
- `results\l_arc_7\step2/clusters_K5.csv`
- `results\l_arc_7\step2/clusters_K6.csv`
- `results\l_arc_7\step2/clusters_K7.csv`
- `results\l_arc_7\step2/path_features.csv`
- `results\l_arc_7\step2/silhouette_K3.txt`
- `results\l_arc_7\step2/silhouette_K4.txt`
- `results\l_arc_7\step2/silhouette_K5.txt`
- `results\l_arc_7\step2/silhouette_K6.txt`
- `results\l_arc_7\step2/silhouette_K7.txt`
- `results\l_arc_7\step2/silhouette_sweep.csv`
- `results\l_arc_7\step2/feature_histograms.png`
- `results\l_arc_7\step2/STEP2_SUMMARY.md`
- `configs/arc_7/step2.yaml`
- `scripts/arc_7/step2_cluster.py`

## Step 1 commit
hash: `59de33a3d6c9459ad688d523c8e391892504b72a`

## Step 2 commit
hash: _pending_

