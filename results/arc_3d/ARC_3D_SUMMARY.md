# Arc 3D — diagnostic tail: SL × aggregation sweep

> **Status**: diagnostic-only, generated 2026-05-16. Does **not** modify the Arc 3
> CLEAN-NULL verdict, the L_ARC_PROTOCOL v2.0 §2 floor numbers, or the §11
> archetype patterns. Produces evidence for v2.1 calibration of
> Open-13, Open-14, Open-15. Authorised by `docs/arc_results/ARC_3_RESULT.md`
> "Diagnostic tail (Arc 3D) — recommended" section.

> **Scope**: TRIAL__volatility_regime__d1_atr_top_decile__any__h_120 (Arc 3
> signal, registry Entry 3). 3 SL settings × 2 aggregation modes = 6 cells.
> Same signal, pair set (28 FX), data window (2020-10-01 → 2026-01-31),
> horizon (120 bars), exposure cap, risk, and spread treatment as Arc 3.

> **Spread treatment**: per-bar historical from MT5 `spread` column in
> `data/1hr/*.csv`, floor-applied via `configs/spread_floors_5ers.yaml`
> (locked, sha256 a613b4ce64...). Identical to Arc 3 baseline.

---

## Cell index

| Cell | SL | Mode | Pool size | Chosen K | Silhouette | trades_all sha256 |
|---|---|---|---|---|---|---|
| A1 | 2.0 × ATR | aggregated | 2568 | 7 | 0.4177 | `47c5290d70954cc04674574fb3567889bb5f610daba8c9ece80939cd1b63ba88` |
| A2 | 2.0 × ATR | un_aggregated | 2568 | 7 | 0.4177 | (same as A1) |
| B1 | 3.0 × ATR | aggregated | 1919 | 4 | 0.4135 | `2209e1848d577e0485ee24dd5c4bca9c4b898a84a48bd955e9de7c08d5779785` |
| B2 | 3.0 × ATR | un_aggregated | 1919 | 4 | 0.4135 | (same as B1) |
| C1 | 4.0 × ATR | aggregated | 1643 | 4 | 0.4082 | `59aad7e2e046d67054b596cb22ed577ab3716683701113ce995ba6e48a1ad280` |
| C2 | 4.0 × ATR | un_aggregated | 1643 | 4 | 0.4082 | (same as C1) |

All six step1 runs deterministic (byte-identical two-run). All step2 runs
deterministic. step3_3d is pure data-processing (no RNG).

---

## §1 — Reproduction check (Cell A1 vs Arc 3 baseline)

**Cell A1 reproduces Arc 3 byte-for-byte.** This is the BLOCKER gate: if A1
diverges from `results/l_arc_3/`, the entire diagnostic is unauthorised. It
does not diverge.

### Step 1 byte-identical hashes

| File | Arc 3 baseline | A1 reproduction | Match |
|---|---|---|---|
| trades_all.csv | `47c5290d70954cc04674574fb3567889bb5f610daba8c9ece80939cd1b63ba88` | `47c5290d70954cc04674574fb3567889bb5f610daba8c9ece80939cd1b63ba88` | ✅ |
| trades_paths.csv | `c32d598fa84b2e730353600813fa0aef4abc44026f88416432922f092bf14624` | `c32d598fa84b2e730353600813fa0aef4abc44026f88416432922f092bf14624` | ✅ |

### Step 2 byte-identical hashes

| File | Arc 3 baseline | A1 reproduction | Match |
|---|---|---|---|
| clusters_K7.csv | `9326f6de6a333b45…` | `9326f6de6a333b45…` | ✅ |
| centroids_K7.csv | `27235581a1f5e754…` | `27235581a1f5e754…` | ✅ |
| archetype_assignments.csv | `68333dedec44d2bc…` | `68333dedec44d2bc…` | ✅ |
| path_features.csv | `ced70d4a430dbc45…` | `ced70d4a430dbc45…` | ✅ |

### Step 3 per-criterion outcomes — A1 vs Arc 3 baseline

| Archetype | mono | peaks | mfe_p50 | reach_1R | wrong_way | shape_tag | size | A1 conjunctive | Arc 3 baseline |
|---|---|---|---|---|---|---|---|---|---|
| Stepwise climber (2+4) | 0.5594 PASS | 16.73 PASS | 3.34R PASS | 0.836 PASS | 0.383 FAIL | bimodal FAIL | 0.275 PASS | FAIL (4/6) | FAIL (4/6) |
| Early-peak hold (0+3) | 0.2508 FAIL | 1.22 PASS | 0.35R FAIL | 0.120 FAIL | 0.988 FAIL | no_magnitude FAIL | 0.400 PASS | FAIL (2/6) | FAIL (2/6) |
| Cluster 1 (unassigned) | 0.4989 FAIL | 8.70 PASS | 1.95R PASS | 0.809 PASS | 0.732 FAIL | no_magnitude FAIL | 0.192 PASS | FAIL (3/6) | FAIL (3/6) |
| Cluster 6 (unassigned) | 0.5037 FAIL | 8.37 PASS | 2.18R PASS | 0.835 PASS | 0.617 FAIL | heavy_right_tail PASS | 0.090 FAIL | FAIL (3/6) | FAIL (3/6) |
| Cluster 5 (unassigned) | 0.0181 FAIL | 1.55 PASS | 0.27R FAIL | 0.018 FAIL | 1.000 FAIL | no_magnitude FAIL | 0.043 FAIL | FAIL (1/6) | FAIL (1/6) |

Note: the "(N/6)" count uses the 6-criterion conjunctive form where direction
(reach_1R AND wrong_way) is one combined criterion, matching Arc 3 step3's
accounting. The Arc 3D archetype_summaries.csv breaks reach_1R and wrong_way
into two columns for per-criterion exposure as required by the prompt.

**Verdict: A1 reproduces Arc 3 exactly. Diagnostic authorised to proceed.**

---

## §2 — Per-cell capturability table

Six rows. For each cell × archetype/cluster: size, centroid_mono,
centroid_peaks, mfe_p50, reach_1R, wrong_way, shape_tag, and the 6-criterion
conjunctive verdict (with "(N/6)" count). Highlights mark cells where the
group passes 5 of 6 (anything but shape_tag and the conjunctive verdict).

### Cell A1 — SL=2.0, aggregated  (K=7, 5 groups: 2 §11 archetypes + 3 unassigned)

| Group | size_frac | mono | peaks | mfe_p50 | reach_1R | wrong_way | shape_tag | gates | verdict |
|---|---|---|---|---|---|---|---|---|---|
| Stepwise climber (2+4) | 0.275 | 0.5594 | 16.73 | 3.34R | 0.836 | 0.383 | bimodal | 4/6 | FAIL |
| Early-peak hold (0+3) | 0.400 | 0.2508 | 1.22 | 0.35R | 0.120 | 0.988 | no_magnitude | 2/6 | FAIL |
| unassigned (cluster 1) | 0.192 | 0.4989 | 8.70 | 1.95R | 0.809 | 0.732 | no_magnitude | 3/6 | FAIL |
| unassigned (cluster 5) | 0.043 | 0.0181 | 1.55 | 0.27R | 0.018 | 1.000 | no_magnitude | 1/6 | FAIL |
| unassigned (cluster 6) | 0.090 | 0.5037 | 8.37 | 2.18R | 0.835 | 0.617 | heavy_right_tail | 3/6 | FAIL |

### Cell A2 — SL=2.0, un-aggregated  (7 clusters)

| Group | size_frac | mono | peaks | mfe_p50 | reach_1R | wrong_way | shape_tag | gates | verdict |
|---|---|---|---|---|---|---|---|---|---|
| cluster_0 (Early-peak hold) | 0.230 | 0.0084 | 0.25 | 0.16R | 0.007 | 1.000 | no_magnitude | 2/6 | FAIL |
| cluster_1 (unassigned) | 0.192 | 0.4989 | 8.70 | 1.95R | 0.809 | 0.732 | no_magnitude | 3/6 | FAIL |
| cluster_2 (Stepwise climber) | 0.148 | **0.5489** | 24.42 | **4.82R** | **1.000** | **0.047** | unclassified | **4/6** | FAIL |
| cluster_3 (Early-peak hold) | 0.170 | 0.5785 | 2.53 | 0.68R | 0.272 | 0.973 | no_magnitude | 3/6 | FAIL |
| cluster_4 (Stepwise climber) | 0.127 | 0.5716 | 7.79 | 1.33R | 0.645 | 0.774 | no_magnitude | 3/6 | FAIL |
| cluster_5 (unassigned) | 0.043 | 0.0181 | 1.55 | 0.27R | 0.018 | 1.000 | no_magnitude | 1/6 | FAIL |
| cluster_6 (unassigned) | 0.090 | 0.5037 | 8.37 | 2.18R | 0.835 | 0.617 | heavy_right_tail | 3/6 | FAIL |

**Cluster 2 alone** (Open-14 evidence): wrong_way collapses from 0.383
(aggregated 2+4) to 0.047 (cluster 2 alone). Aggregation was masking a
structurally cleaner sub-archetype. cluster 2 fails monotonicity by **0.001**
(0.5489 vs 0.55 floor) — a hair, and Arc 3D's tightest near-miss outside C1.

### Cell B1 — SL=3.0, aggregated  (K=4, 4 groups: 2 archetypes + 2 unassigned)

| Group | size_frac | mono | peaks | mfe_p50 | reach_1R | wrong_way | shape_tag | gates | verdict |
|---|---|---|---|---|---|---|---|---|---|
| Stepwise climber (cluster 0) | 0.254 | 0.5429 | 21.52 | **1.97R** | **0.979** | **0.084** | bimodal | 4/6 | FAIL |
| Early-peak hold (cluster 1) | 0.188 | 0.0064 | 0.48 | 0.13R | 0.000 | 1.000 | no_magnitude | 2/6 | FAIL |
| unassigned (cluster 2) | 0.214 | 0.5027 | 8.11 | 1.32R | 0.663 | 0.571 | heavy_right_tail | 3/6 | FAIL |
| unassigned (cluster 3) | 0.345 | 0.5348 | 5.78 | 0.73R | 0.361 | 0.814 | no_magnitude | 2/6 | FAIL |

### Cell B2 — SL=3.0, un-aggregated  (4 clusters)

Identical to B1 in this cell (K=4 → no §6 aggregation possible since Stepwise
climber is a single cluster; Early-peak hold same). Cluster-level table
matches B1 row-for-row.

### Cell C1 — SL=4.0, aggregated  (K=4, 4 groups: 2 archetypes + 2 unassigned)

| Group | size_frac | mono | peaks | mfe_p50 | reach_1R | wrong_way | shape_tag | gates | verdict |
|---|---|---|---|---|---|---|---|---|---|
| **Stepwise climber (cluster 0)** | **0.287** | **0.5424** ❌ | **21.04** ✅ | **2.10R** ✅ | **0.936** ✅ | **0.051** ✅ | **tight_unimodal** ✅ | **5/6** | **FAIL (mono only)** |
| Early-peak hold (cluster 1) | 0.145 | 0.0135 | 0.54 | 0.10R | 0.000 | 0.996 | no_magnitude | 2/6 | FAIL |
| unassigned (cluster 2) | 0.376 | 0.5325 | 6.56 | 0.70R | 0.311 | 0.718 | no_magnitude | 2/6 | FAIL |
| unassigned (cluster 3) | 0.192 | 0.5031 | 8.19 | 1.05R | 0.524 | 0.492 | heavy_right_tail | 3/6 | FAIL |

### Cell C2 — SL=4.0, un-aggregated  (4 clusters)

Identical to C1 (Stepwise climber is one cluster; aggregation no-op).

---

## §3 — SL response curves

### Stepwise climber (aggregated)

Tracking the same-archetype cohort across A1 → B1 → C1. At SL=2.0 the cohort
is clusters 2+4 (aggregated); at SL=3.0 and SL=4.0 it's a single cluster
(cluster 0) because the wider-SL clustering collapses Arc 3's split into one
band. Identity is matched by centroid signature (mono ≈ 0.54, peaks ≈ 17–21,
ttp_rel ≥ 0.80) — clearly the same archetype.

| Criterion | A1 (SL=2.0) | B1 (SL=3.0) | C1 (SL=4.0) | Direction | Floor | A1 / B1 / C1 verdict |
|---|---|---|---|---|---|---|
| size_fraction | 0.275 | 0.254 | 0.287 | ↔ | ≥ 0.10 | PASS / PASS / PASS |
| centroid_monotonicity | 0.559 | 0.543 | 0.542 | ↓ | ≥ 0.55 | **PASS** / **FAIL** / **FAIL** |
| centroid_local_peaks | 16.73 | 21.52 | 21.04 | ↑ | in [5, 30] | PASS / PASS / PASS |
| mfe_h240_p50 | 3.34R | 1.97R | 2.10R | ↓ (R-rescale) | ≥ 1.5R | PASS / PASS / PASS |
| frac_reach_1R | 0.836 | 0.979 | 0.936 | ↑ | ≥ 0.70 | PASS / PASS / PASS |
| **frac_wrong_way** | **0.383** | **0.084** | **0.051** | **↓↓** | ≤ 0.30 | **FAIL** / **PASS** / **PASS** |
| **shape_tag** | **bimodal** | **bimodal** | **tight_unimodal** | flips at SL=4 | ∈ {tight_unimodal, heavy_right_tail} | **FAIL** / **FAIL** / **PASS** |
| **gates_passed** | **4/6** | **4/6** | **5/6** | | | |

Two structural facts surface:

1. **wrong_way drops 7.5× across the SL sweep** (0.383 → 0.051). Open-15
   confirmed at large magnitude. The 2.0 × ATR_1H SL on a 120-bar horizon
   was structurally too tight; wider SLs cleanly satisfy the floor.

2. **shape_tag flips bimodal → tight_unimodal at SL=4.0**. The bimodality
   at SL=2.0 was an artifact of the SL setting, not an intrinsic property
   of the archetype. With wider SL the right-mode mass dominates the
   distribution and the IQR/skew settle into the tight_unimodal classifier.

3. **monotonicity drops slightly with wider SL (0.559 → 0.542)** and crosses
   the floor at SL=3.0. This is a feature-mechanical effect: wider 1R means
   each bar's close_r is a smaller relative move, so the monotonicity ratio
   captures less "in-profit ratchets". The 0.55 floor was calibrated under
   SL=2.0 anchor measurements; calibration may need adjustment when SL
   changes.

### Un-aggregated cluster 2 (Arc 3 baseline) vs un-aggregated cluster 0 (B2, C2)

Tracking the cleanest single Stepwise sub-cluster — Arc 3's cluster 2 (the
"24 small steps" sub-archetype) — against its narrowest centroid analogue at
wider SLs:

| Property | A2 cluster_2 (SL=2.0) | B2 cluster_0 (SL=3.0) | C2 cluster_0 (SL=4.0) |
|---|---|---|---|
| size_fraction | 0.148 | 0.254 | 0.287 |
| mono centroid | 0.5489 | 0.5429 | 0.5424 |
| peaks centroid | 24.42 | 21.52 | 21.04 |
| pullback centroid | 0.498 | 0.330 | 0.262 |
| ttp_rel centroid | 0.816 | 0.807 | 0.826 |
| mfe_p50 | 4.82R | 1.97R | 2.10R |
| reach_1R | 1.000 | 0.979 | 0.936 |
| wrong_way | 0.047 | 0.084 | 0.051 |
| shape_tag | unclassified | bimodal | tight_unimodal |
| gates_passed | 4/6 | 4/6 | 5/6 |

The wider-SL clusters absorb cluster 4's mass into cluster 0 (the wider SL
no longer separates "many small steps" from "few bigger moves" cleanly). So
the SL=4.0 Stepwise cohort is a strict superset of Arc 3's cluster 2 — its
wrong_way 5.1% reflects both cluster 2 (which was 4.7% alone at SL=2.0) and
the surviving cluster-4-like mass at the wider SL.

### Early-peak hold (cluster 1 at SL=3,4 vs aggregate 0+3 at SL=2.0)

| Property | A1 (0+3) | B1 (1) | C1 (1) |
|---|---|---|---|
| size_fraction | 0.400 | 0.188 | 0.145 |
| mono | 0.251 | 0.006 | 0.013 |
| mfe_p50 | 0.35R | 0.13R | 0.10R |
| reach_1R | 0.120 | 0.000 | 0.000 |
| wrong_way | 0.988 | 1.000 | 0.996 |
| shape_tag | no_magnitude | no_magnitude | no_magnitude |

Early-peak hold gets worse at wider SLs (mfe_p50 in R units halves and halves
again as 1R widens). It was already a clear FAIL at SL=2.0; wider SL doesn't
rescue it. Open-15's wider-SL benefit is archetype-specific — it cures the
high-magnitude Stepwise cohort's wrong_way without touching the low-magnitude
Early-peak cohort's structural failure.

---

## §4 — Cluster-identity drift

Wider SL produces fundamentally different clusterings of the same signal:

| SL | Chosen K | Silhouette | Range across K∈{3..7} | Stepwise cluster shape |
|---|---|---|---|---|
| 2.0 | 7 | 0.4177 | 0.4027–0.4177 (0.015) | split into two: cluster 2 ("many small steps", peaks 24) and cluster 4 ("few bigger moves", peaks 8) |
| 3.0 | 4 | 0.4135 | — | merged into one: cluster 0 (peaks 21.5) |
| 4.0 | 4 | 0.4082 | — | merged into one: cluster 0 (peaks 21.0) |

The collapse from K=7 → K=4 is the most important drift fact:

- At SL=2.0 the silhouette is roughly flat across K=3–7 (range 0.015 —
  effectively noise per Open-12). The K=7 choice was a 0.0021-margin
  tiebreaker.
- At SL=3.0 and SL=4.0 the silhouette has a clearer K=4 maximum. The wider
  R-scaling tightens the within-cluster variance for the larger cohorts and
  no longer rewards K=7's finer splits.
- Cluster 2 vs cluster 4 at SL=2.0 — the §6 same-archetype aggregation that
  Arc 3 applied — collapses cleanly into a single Stepwise cluster at wider
  SL. **This is direct evidence that the §6 aggregation rule was, in this
  case, doing the right thing in spirit** (one underlying archetype, two
  noisy sub-clusters at K=7). But it also confirms Open-14's concern: at
  K=7 the two sub-clusters had **structurally different wrong_way**
  (cluster 2 alone: 4.7%; cluster 4 alone: 77.4%) — masked by the aggregate.

The Early-peak hold cohort also collapses (cluster 0 + cluster 3 at SL=2.0
merge into cluster 1 at SL=3.0/4.0) but doesn't change verdict — it's a
structural FAIL on magnitude at every SL.

### Centroid signature tracking — Stepwise climber across SLs

| SL | Cluster ID | mono | peaks | pullback | ttp_rel | Matches Arc 3 Stepwise pattern? |
|---|---|---|---|---|---|---|
| 2.0 | cluster 2 | 0.549 | 24.42 | 0.498 | 0.816 | ✅ |
| 2.0 | cluster 4 | 0.572 | 7.79 | 0.426 | 0.787 | ✅ |
| 3.0 | cluster 0 | 0.543 | 21.52 | 0.330 | 0.807 | ✅ |
| 4.0 | cluster 0 | 0.542 | 21.04 | 0.262 | 0.826 | ✅ |

Centroid signature is stable enough to track across SLs by pattern matching
(all four pass `mono ≥ 0.50`, `peaks ∈ [5, 30]`, `ttp_rel ≥ 0.50`, the §11
Stepwise pattern). Cluster ID renumbering between SL=2.0 and SL=3.0/4.0 is
just because K changed — no archetype was lost.

---

## §5 — Open-13 / Open-14 / Open-15 evidence verdicts

### Open-13 — §2 shape_tag floor / §11 row-7 bimodal incompatibility

**Verdict: weaker support than the Arc 3 result doc anticipated.**

The Open-13 amendment was proposed on the premise that the Stepwise
climber's bimodal shape_tag is an intrinsic property of the archetype that
§2's shape_tag floor incorrectly excludes. Arc 3D shows:

- **At SL=4.0 the Stepwise climber's shape_tag flips to tight_unimodal**
  (cell C1/C2). The bimodality at SL=2.0 was an SL-induced artifact, not an
  intrinsic property. The right-mode of the SL=2.0 bimodal distribution
  becomes the centre of mass when 1R is wider; the left mode (the stop-out
  cluster at exactly -1R) shrinks proportionally because fewer trades hit
  SL.
- The Open-13 amendment ("admit bimodal when accompanied by §11 row-7
  routing AND modes meet ≥ 1R separation") would have admitted the SL=2.0
  cohort. But fixing the SL also fixes the bimodality without needing the
  amendment.

**Implication for v2.1:** Open-13's bimodal admission rule is a backstop
that solves a problem the SL choice itself created. The cleaner fix is the
Open-15 SL-horizon rescaling. If Open-15 lands, Open-13 may not be
load-bearing.

That said, Open-13 remains a worthwhile safety net — bimodality may arise
in other arcs from genuinely two-mode dynamics, not just SL artifacts.
Documenting the SL-sensitivity in v2.1 is more important than admitting
bimodal in §2.

### Open-14 — Same-archetype aggregation destroys capturable sub-clusters

**Verdict: confirmed with caveats.**

Direct evidence from Cell A2 cluster-level breakdown:

| Cluster | local_peaks | pct_peak_and_collapse | wrong_way | Conclusion |
|---|---|---|---|---|
| 2 (alone) | 24.42 | 0.126 | **0.047** | structurally clean |
| 4 (alone) | 7.79 | 0.474 | **0.774** | structurally damaged |
| 2+4 (aggregated) | 16.73 | 0.287 | 0.383 | dominated by cluster 4's mass |

Cluster 4's `pct_peak_and_collapse` is **3.7×** cluster 2's. Despite both
sharing the §11 Stepwise pattern, they are forward-geometrically distinct.
The §6 same-archetype rule masked cluster 2's clean wrong_way (4.7%) by
averaging it with cluster 4's high wrong_way (77.4%) → aggregate 38.3%
fails the floor; cluster 2 alone passes by margin.

**Caveat: at wider SL the two sub-clusters merge into one cluster.** B1/C1's
single Stepwise cluster has wrong_way 8.4%/5.1% (not the 4.7% of cluster 2
alone) — because the wider-SL pool no longer separates the structurally
clean sub-mass from the post-collapse trades. So the v2.1 amendment for
Open-14 (conditional aggregation on cluster-pair disparity) is most
load-bearing **at narrow SL where K is high**. If Open-15 fixes SL, K
collapses and Open-14's mechanism becomes less prominent.

**Implication for v2.1:** Open-14 is real but interacts strongly with Open-15.
The cleanest path is to fix Open-15 first; Open-14 then becomes a smaller
calibration concern.

### Open-15 — SL-distance / hold-horizon asymmetry inflates frac_wrong_way

**Verdict: confirmed at the largest magnitude of the three Open-13/14/15
items, with high confidence.**

Stepwise climber frac_wrong_way across SLs:

| SL | wrong_way (aggregated/single Stepwise cluster) | Margin to floor (0.30) |
|---|---|---|
| 2.0 | 0.383 | −0.083 (FAIL) |
| 3.0 | 0.084 | +0.216 (PASS) |
| 4.0 | 0.051 | +0.249 (PASS) |

The reduction is monotonic and large. The expected halving from a 2x SL
widening is conservative — the actual reduction from SL=2.0 → SL=4.0 is
7.5x.

Cluster 1 (unassigned, Peak-and-collapse near-miss in Arc 3) shows similar
behaviour at SL=2.0 → 3.0 → 4.0 if we track by centroid signature, but at
wider SLs the cluster identity drifts and the directly-comparable cohort is
the SL=3.0 cluster 2 (heavy_right_tail) and SL=4.0 cluster 3 (heavy_right_tail).
Both have wrong_way ≤ 0.72 at SL=3.0 → 0.49 at SL=4.0, consistent with
Open-15's mechanism, but on a population that doesn't pass anyway because of
other floor failures.

**Implication for v2.1:** Open-15 SL-horizon rescaling (proposed default
`SL = 2.0 × ATR × √(h/24)`, which is ~4.5×ATR_1H for h=120) is well supported.
At SL=4.0 the Stepwise cohort fails only on monotonicity (0.5424 vs 0.55).
A combined Open-15 fix + monotonicity-floor recalibration (small downward
adjustment of the floor at wider SLs) would push this cohort into pass
territory.

The Brownian-consistent reference at √(h/24) × 2.0 = ~4.5 × ATR_1H for h=120
sits between this study's SL=4.0 and SL=5.0; SL=4.0 is the conservative
near-reference value and already produces a near-pass.

---

## §6 — Any cell where all 5 non-shape_tag floors pass

**Answer: No cell where all 5 non-shape_tag floors pass.**

The closest case is **C1/C2 Stepwise climber at SL=4.0**:

| Criterion | Value | Floor | Status |
|---|---|---|---|
| centroid_monotonicity | 0.5424 | ≥ 0.55 | **FAIL (margin −0.0076)** |
| centroid_local_peaks | 21.04 | ∈ [5, 30] | PASS |
| mfe_h240_p50 | 2.10R | ≥ 1.5R | PASS |
| frac_reach_1R | 0.936 | ≥ 0.70 | PASS |
| frac_wrong_way | 0.051 | ≤ 0.30 | PASS |
| shape_tag | tight_unimodal | ∈ allowed set | PASS |
| size_fraction | 0.287 | ≥ 0.10 | PASS |

5 of 6 conjunctive criteria pass (or 6 of 7 if reach_1R/wrong_way separated).
The single failure is **monotonicity**, not shape_tag. This is structurally
different from the Open-13 hypothesis that the only floor in the way of
Stepwise's pass was shape_tag.

The runner-up is **A2 cluster_2 (Stepwise climber alone) at SL=2.0**:

| Criterion | Value | Floor | Status |
|---|---|---|---|
| centroid_monotonicity | 0.5489 | ≥ 0.55 | **FAIL (margin −0.0011)** |
| centroid_local_peaks | 24.42 | ∈ [5, 30] | PASS |
| mfe_h240_p50 | 4.82R | ≥ 1.5R | PASS |
| frac_reach_1R | 1.000 | ≥ 0.70 | PASS |
| frac_wrong_way | 0.047 | ≤ 0.30 | PASS |
| shape_tag | unclassified | ∈ allowed set | **FAIL** |
| size_fraction | 0.148 | ≥ 0.10 | PASS |

4 of 6 conjunctive pass. Two failures: monotonicity (0.001 miss) and
shape_tag (unclassified — distribution is high-variance, doesn't fit the
classifier rules). Note shape_tag is **unclassified** not **bimodal** — the
cluster 2 alone distribution is not bimodal; the aggregate (2+4) is.

**The headline evidence Arc 3D produces is:**

> The Stepwise climber archetype passes 5 of 6 §2 floors under SL=4.0,
> failing only on monotonicity by 0.0076. Both Open-13 (bimodal shape_tag)
> and Open-15 (wrong_way) failures dissolve under wider SL. A new failure
> mode — monotonicity sensitivity to SL choice — surfaces in their place.

---

## §7 — What the data does NOT show (anti-evidence)

Motivated-reasoning risk is real here; the diagnostic was conceived with
hypotheses about what wider SL would do. Several things did not pan out
the way the Arc 3 result doc's "Stop-loss distance / horizon asymmetry"
section suggested.

### (a) "Doubling SL halves wrong_way mass"

**Stronger than predicted.** The Arc 3 result doc estimated that doubling SL
would halve wrong_way (to ~20%). Actual: 0.383 (SL=2.0) → 0.051 (SL=4.0) is
**7.5x** reduction, not 2x. The structural mechanism is strong.

But also: "MFE p50 in R units halves to ~1.7R" was predicted; actual is
3.34R (SL=2.0) → 2.10R (SL=4.0), a 1.6x reduction — close to predicted
but not exactly half. **Reach_1R held up better than predicted**: 0.836
(SL=2.0) → 0.936 (SL=4.0) **increased**, contradicting the Arc 3 result
doc's expectation that reach_1R would drop to ~60%. The mechanism is:
wider SL means fewer trades stopped out, so more trades have time to
reach 1R MFE in absolute terms before time-exit.

### (b) "Reach_1R drops to ~60%"

**Not observed.** Reach_1R is ~94% at SL=4.0 (cell C1/C2), not 60%.
Prediction was wrong on direction. Reason: the Stepwise climber's MFE is
driven by trade duration, not by being close to SL — its trades run long
and reach distant peaks. Wider SL preserves duration (fewer SL exits) →
preserves MFE reach.

### (c) Open-13 standalone fix would have helped

**Partially undermined.** Open-13 was proposed to address the shape_tag
floor blocking Stepwise climber. At SL=4.0 the shape_tag flips to
tight_unimodal (admissible) without any §2 change. Fixing Open-15 (SL
choice) dissolves Open-13's specific problem case. Open-13 remains
worthwhile as a safety net but the Stepwise-climber-as-Open-13-evidence
framing weakens.

### (d) Wider SL is monotonically better

**Not true.** At SL=4.0 the Stepwise cohort fails monotonicity (0.5424 vs
0.55 floor) — a fresh failure mode appearing at the wider SL. The
monotonicity_ratio_in_profit feature is SL-sensitive in a way the protocol
hasn't accounted for. **Wider SL trades one failure (wrong_way) for another
(monotonicity).** This is a real anti-finding — the simplest "just go to
SL=4.0" recommendation would not pass §2 either.

### (e) Un-aggregation unlocks a sub-cluster cleanly

**Partially true, but narrow.** Cluster 2 alone at SL=2.0 (Cell A2) passes
mfe_p50, reach_1R, wrong_way, size cleanly — it's the cleanest cohort in the
entire 6-cell sweep on those four criteria. But it fails monotonicity by
0.001 and its shape_tag is unclassified (not in admissible set). So even the
most-favourable un-aggregation under Open-14 still doesn't pass §2. Open-14
gets one floor right (wrong_way) but doesn't surface a 5-of-non-shape_tag
case at SL=2.0.

### (f) Combined Open-13 + Open-14 + Open-15 would unlock the Stepwise cohort

**Partially correct, but more complicated than expected.** A combined fix:

- Open-15 (SL=4.0 default for h=120) → wrong_way passes, shape_tag passes,
  but **monotonicity now fails** by 0.0076.
- Open-14 (un-aggregation) → no-op at SL=4.0 since Stepwise is a single cluster.
- Open-13 (bimodal admission) → no-op at SL=4.0 since shape_tag is already
  tight_unimodal.

The combined fix needs **a fourth change** to land the Stepwise cohort: a
monotonicity floor adjustment, or a monotonicity-feature definition that's
SL-invariant. Neither is currently in the v2.1 backlog.

### (g) The Arc 3 verdict could be reversed

**No.** The whole diagnostic operates under fixed Arc 3 verdict (CLEAN-NULL,
locked). Nothing in this study moves the §2 floors or §11 patterns. The
near-passes documented here (mono 0.5424 vs 0.55 at SL=4.0; mono 0.5489 vs
0.55 at A2 cluster 2) are not within-arc passes — they're cross-arc
calibration evidence. Per project methodology, the discipline of accepting
CLEAN-NULL is what makes the evidence trustworthy.

---

## Cross-arc implications

**v2.1 calibration priorities, in order:**

1. **Open-15 (SL-horizon rescaling)** — highest confidence, largest magnitude
   effect. Adopting SL = `2.0 × ATR × √(h/24)` as the arc-level default
   resolves wrong_way (Open-15's stated target) and shape_tag bimodality
   (Open-13's stated target) simultaneously. Calibrate against KH-24 anchor
   (h=320 effectively, given KH-24 trades run longer) — under v2.1's
   rescaling rule, KH-24's SL would be slightly wider than the current 2x;
   need to verify the K=4 archetype 3 D1 pass is preserved.

2. **Monotonicity feature SL-invariance** — new item, surfaced by Arc 3D.
   The monotonicity_ratio_in_profit feature decreases slightly as SL widens
   (mechanical effect; not a property of the underlying signal). Either:
   (a) redefine the feature to be SL-invariant (e.g., normalise close_r
   thresholds to a fraction of mfe_so_far_r rather than absolute R), or
   (b) recalibrate the §2 monotonicity floor in conjunction with Open-15.
   Without this, Open-15's adoption may shift the §2 failure mode from
   wrong_way to monotonicity rather than removing it.

3. **Open-14 (conditional aggregation)** — confirmed but second-order.
   Most load-bearing at narrow SL. After Open-15 lands, the K=high regime
   is less common (B1/C1 chose K=4). Lower priority but still real.

4. **Open-13 (shape_tag bimodal admission)** — backstop value only.
   Stepwise climber's bimodality dissolves under Open-15. Other arcs may
   surface bimodal cohorts that aren't SL artifacts; Open-13 remains a
   reasonable safety net but is no longer the urgent item Arc 3's result
   doc suggested.

**KH-24 anchor cross-check (Open-15 calibration):**

Required before any Open-15 commit: under SL = 2.0 × ATR × √(h_kh24/24)
with h_kh24 = the effective KH-24 holding period (typically ~30 bars on
4H = ~5 days), what is the SL multiplier and does KH-24's K=4 archetype 3
still pass v2.0 extractability (D1 at t=3)? If a wider arc-level SL breaks
KH-24's anchor pass, Open-15 needs the archetype-specific override path
(Open-15 option (a), Step-4 SL adjustments) rather than the arc-level
default (option (b)).

This calibration is not in scope for Arc 3D — it's the next step before
v2.1 commit.

---

## File index

```
results/arc_3d/
├── sl_2.0/
│   ├── step1/         trades_all.csv, trades_paths.csv, manifest.json
│   ├── step2/         clusters_K{3..7}.csv, centroids_K{3..7}.csv,
│   │                  silhouette_K{3..7}.txt, archetype_assignments.csv,
│   │                  path_features.csv, step2_diagnostics.json
│   └── step3/
│       ├── aggregated/     archetype_summaries.csv, step3_3d_diagnostics.json
│       └── un_aggregated/  archetype_summaries.csv, step3_3d_diagnostics.json
├── sl_3.0/  (same structure; K=4 chosen)
├── sl_4.0/  (same structure; K=4 chosen)
└── ARC_3D_SUMMARY.md   (this document)
```

Scripts:
- `scripts/arc_3/step1_backtest.py` — engine-fix: now reads
  `cfg["exit"]["hard_stop"]["multiplier"]` instead of hardcoded 2.0.
  Value-identical to Arc 3 baseline when multiplier=2.0 (proven by
  Cell A1 byte-identical hashes).
- `scripts/arc_3/step2_clustering.py` — unmodified.
- `scripts/arc_3d/step3_3d.py` — new. Dynamic archetype grouping from
  step2 archetype_assignments.csv; supports `--mode aggregated` and
  `--mode un_aggregated`; per-criterion PASS/FAIL columns.

Configs:
- `configs/wfo_l_arc_3d_sl2.yaml` — SL=2.0, output sl_2.0/step1
- `configs/wfo_l_arc_3d_sl3.yaml` — SL=3.0, output sl_3.0/step1
- `configs/wfo_l_arc_3d_sl4.yaml` — SL=4.0, output sl_4.0/step1

---

## Validation checklist

- [x] All 6 cells produced
- [x] step1 byte-identical determinism per SL (verified via --verify-determinism)
- [x] A1 reproduces Arc 3 baseline byte-for-byte (step1 + step2 sha256s match;
      step3 per-cluster centroid + per-criterion §2 outcomes match)
- [x] All 7 required sections present in this summary
- [x] Per-cluster §2 results expose individual criterion pass/fail (separate
      reach_1R / wrong_way columns plus the 6-criterion conjunctive count)
- [x] Cluster-identity tracking by centroid signature, not cluster ID
- [x] No §2 floor numbers changed anywhere
- [x] No §11 archetype patterns changed
- [x] Spread treatment unchanged (per-bar from data/1hr/, floored via
      configs/spread_floors_5ers.yaml)
