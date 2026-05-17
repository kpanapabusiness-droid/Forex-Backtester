# Open-18 Replay #2 — KH-24 v2.0 c4 under v2.1.1 — RESULT

## Status

Opened 2026-05-17 under L_ARC_PROTOCOL v2.1.1 §16 Open-18.
Closed 2026-05-17. Final disposition: **PASS_v2_1_1** — primary cohort c4 rescued at SL=1.0×ATR (composite 0.304). Secondary surprise: c1 (43% of pool, near-stepwise) also passes individually at SL=2.0×ATR with composite 0.387.

Anchor refresh status: **DEFER** — chat-side synthesis after all three Open-18 replays land.

## Cohort under test

| Field | Value |
|---|---|
| Source arc | KH-24 v2.0 self-test (`arc_kh24_v2`) |
| Pool | 842 trades, K=5 clusters |
| Primary cohort of interest | c4 (trend-rider, n=122, 14.5% pool) |
| v2.0 verdict on c4 | FAIL §2 — mono 0.530 (full-window, miss 0.020), shape_tag=scattered (87.7% cap-binders censoring) |
| v2.1.1 hypothesis | c4 rescued via pre-peak + MFE-based shape_tag |
| Anchor role | If rescued, c4 becomes candidate for §14 anchor refresh — final decision chat-side |

## Verdict by cluster

| row_type | id | archetype_label | n | size_frac | selected_SL | composite | mono_pre_peak | shape_tag | verdict | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| cluster | c0 | peak_and_collapse | 193 | 0.229 | none | N/A | 0.114 | heavy_right_tail | FAIL_NO_PASSING_SL | mono and reach_1R fail at every SL; v2.0 peak-and-collapse, no rescue |
| cluster | c1 | unresolved_f22d041e | 365 | 0.434 | 2.0× | 0.387 | 0.609 | heavy_right_tail | **PASS** | unexpected survivor; near-stepwise centroid; all six SLs pass shape; 2.0× wins composite |
| cluster | c2 | peak_and_collapse | 112 | 0.133 | none | N/A | 0.694 | heavy_right_tail | FAIL_NO_PASSING_SL | mono pre-peak clean but reach_1R and fwd_mfe_p50 fail; 4/6 SLs hit shape_tag dead zone |
| cluster | c3 | unresolved_dd77bf0f | 50 | 0.059 | none | N/A | 0.348 | heavy_right_tail | FAIL_NO_PASSING_SL | reach_1R fails at every SL (best 0.48 at 0.5×); v2.0 near-monotone |
| cluster | c4 | unresolved_11fee4a0 | 122 | 0.145 | 1.0× | 0.304 | 0.566 | heavy_right_tail | **PASS** | primary cohort; rescued; selection blocked at wider SLs by shape_tag dead zone (Finding 2) |
| aggregate | agg_peak_and_collapse | peak_and_collapse | 305 | 0.362 | none | N/A | 0.327 | heavy_right_tail | FAIL_NO_PASSING_SL | c0+c2 aggregate fails magnitude (best fwd_mfe_p50 0.69R) and reach_1R (best 0.36) at every SL |

Source: `archetype_summaries.csv`.

## c4 under v2.1.1 (primary cohort)

SL sweep on c4 (n=122, 14.5% of pool):

| SL ×ATR | mono pre-peak | local_peaks | fwd_mfe_p50 R_cand | fwd_mfe_p95 R_cand | reach_1R | wrong_way pre-peak | cap_bind | shape_tag (p95/p50) | local_peaks | composite |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 0.605 | 8.83 | 1.86 | 40.79 | 0.664 | 0.377 | 0.189 | heavy_right_tail (21.9) | PASS | N/A (reach_1R + wrong_way) |
| **1.0** | **0.566** | **18.68** | **7.51** | **24.11** | **0.770** | **0.082** | **0.443** | **heavy_right_tail (3.21)** | **PASS** | **0.304 ← selected** |
| 1.5 | 0.564 | 25.47 | 8.07 | 18.69 | 0.869 | 0.008 | 0.664 | unclassified (2.32) | PASS | N/A (shape — dead zone) |
| 2.0 | 0.557 | 30.94 | 6.65 | 14.59 | 1.000 | 0.000 | 0.877 | unclassified (2.19) | DEFERRED | N/A (shape — dead zone) |
| 3.0 | 0.557 | 30.94 | 4.43 | 9.73 | 0.992 | 0.000 | 0.984 | unclassified (2.19) | DEFERRED | N/A (shape — dead zone) |
| 4.0 | 0.557 | 30.94 | 3.33 | 7.29 | 0.984 | 0.000 | 1.000 | unclassified (2.19) | DEFERRED | N/A (shape — dead zone) |

Composite tiebreaker not invoked (1.0× is the only SL with all numerical floors passing). Source: `c4_sl_sweep.csv`, `c4_distribution.csv`.

### v2.0 vs v2.1.1 (c4 at selected SL)

| Metric | v2.0 (full-window, 2.0×ATR) | v2.1.1 (selected SL = 1.0×ATR) |
|---|---|---|
| selected SL | 2.0×ATR (single) | 1.0×ATR |
| monotonicity | 0.530 (full-window) | 0.566 (pre-peak) |
| shape_tag | scattered (final_r) | heavy_right_tail (MFE-based) |
| frac_wrong_way | 0.000 (Def B full-window) | 0.082 (Def C pre-peak) |
| fwd_mfe_h240_p50 | 6.65R (R_sim) | 7.51R (R_candidate) |
| cap_binding_rate | 0.877 | 0.443 |
| §2 verdict | FAIL | PASS |
| composite | N/A | 0.304 |

The v2.0 → v2.1.1 deltas: mono +0.036 (just crosses 0.55 floor); shape_tag flips from censoring-artifact `scattered` to `heavy_right_tail`; cap-binding halved (0.877 → 0.443); fwd_mfe_p50 actually rises (6.65 → 7.51) because R-denomination shrinks at the tighter SL. wrong_way Def C ticks up (0.000 → 0.082) because Def C measures pre-peak MAE breaches against the tighter candidate threshold, not the wider one. Net: §2 floors all pass conjunctively, primary hypothesis (§16 Open-18) confirmed for c4.

## Other clusters

**c1 (unresolved_f22d041e, near stepwise, n=365 — 43.4% of pool — UNEXPECTED PASS):** Five of six SLs cleared shape_tag (heavy_right_tail), composite peaked at 2.0×ATR with 0.387. Pre-peak mono 0.609, fwd_mfe_p50 1.70R, reach_1R 0.742, wrong_way 0.014, local_peaks centroid 8.40 (cleanly inside Stepwise 5–30 range). cap_binding only 0.090 — the cohort is mostly SL-resolved at 2.0×, not censored. v2.0 had labelled c1 unresolved with full-window mono 0.501 (just under 0.55); pre-peak measurement lifted it +0.108 to 0.609. This is the cleanest surviving cohort by every measure and was not on the §16 Open-18 hypothesis list — see Finding 6 below.

**c0 (peak_and_collapse, n=193):** Dies at every SL on mono (centroid 0.11–0.13, far under 0.55) and on reach_1R (best 0.24 at 0.5×). frac_wrong_way_pre_peak ranges 0.76 → 0.005 across SLs, but the cohort's structural shape (early peak, immediate collapse) is not rescuable. v2.0 verdict reproduced.

**c2 (peak_and_collapse, n=112):** mono pre-peak clean (0.69–0.89) but cohort lacks magnitude — fwd_mfe_p50 1.16R at the tightest SL and falls below 1R at every wider candidate. reach_1R fails too. 4/6 SLs hit the shape_tag dead zone (Finding 1), but this is moot since numerical floors block first.

**c3 (unresolved_dd77bf0f, near monotone, n=50):** reach_1R fails at every SL (best 0.48 at SL=0.5×). mono pre-peak 0.35–0.58 across SLs. shape_tag heavy_right_tail everywhere. Smallest cluster in the pool (5.9%); marginally above size floor (0.10) at SL=0.5× (0.059 — actually below 0.10 floor), so even if numerical floors had passed, the size floor would block.

## Per-cluster vs per-aggregate routing

Per §7 routing: the v2.0 K=5 archetype labels (from `ARC_KH24_V2_RESULT.md`) place c0 and c2 in `peak_and_collapse`; c1, c3, c4 each carry unique `unresolved_*` provisional labels. One aggregate fires:

- **agg_peak_and_collapse** (c0 + c2, n=305, 36.2% of pool): FAIL_NO_PASSING_SL. At every SL the aggregate cohort fails magnitude (fwd_mfe_p50 best 0.69R at 0.5×, well under 1.5R floor) and reach_1R (best 0.36). Neither constituent rescues the other.

c1, c3, c4 produce no aggregates (singleton labels). Disposition per `cluster_routing.csv`:

| cluster | individual_passes | aggregate_passes | disposition |
|---|---|---|---|
| c0 | False | False | dies |
| c1 | **True** | n/a (singleton) | individual_passes_only |
| c2 | False | False | dies |
| c3 | False | n/a (singleton) | dies |
| c4 | **True** | n/a (singleton) | individual_passes_only |

Surviving routings to Step 4+ (under chat-side decisions on anchor refresh and protocol calibration): c1 and c4 as individual cohorts.

## Cross-replay calibration flags

### Finding 1 — shape_tag classifier dead zone (2.0 < p95/p50 ≤ 3.0)

The MFE-based shape_tag classifier (v2.1.1) admits only the boundary categories:
- p95/p50 ≤ 2.0 → `tight_unimodal` (admitted at §2)
- p95/p50 > 3.0 → `heavy_right_tail` (admitted at §2)
- (2.0, 3.0] → `unclassified` (rejected at §2)

Moderate-tail unimodal distributions land in this gap. In this replay 8 sweep cells out of 36 (5 individual clusters × 6 SLs + 1 aggregate × 6 SLs) hit the dead zone:

- **c2:** 4 cells (SL=1.5, 2.0, 3.0, 4.0; p95/p50 = 2.73, 2.69, 2.69, 2.69) — moot for c2 since numerical floors block independently
- **c4:** 4 cells (SL=1.5, 2.0, 3.0, 4.0; p95/p50 = 2.32, 2.19, 2.19, 2.19) — **load-bearing** because every cell except 1.0× is otherwise structurally PASS

c4's wider-SL configurations have numerical floors strictly cleaner than the selected 1.0× (see Finding 2), but shape_tag blocks them all. The selected 1.0× sits in `heavy_right_tail` only because the distribution still has enough right-tail extension at the tighter R-denomination (p95/p50 = 3.21).

These cutoffs are not in protocol §7; they originated at task-prompt design. Cross-arc calibration question: should the classifier admit moderate-tail unimodal distributions? Defer until Replay #1 (Arc 3 Stepwise) and Replay #3 (Arc 2 c2) land.

### Finding 2 — selection driven by classifier mechanics, not structural truth (c4)

At SL=2.0×ATR (the v2.0 reference frame), c4's numerical floors are strictly cleaner than at SL=1.0×: mono pre-peak 0.557 (vs 0.566 at 1.0× — within rounding noise), fwd_mfe_p50 6.65R (vs 7.51R, but this is R-denomination flattering 1.0× — in absolute ATR units 2.0× peak is bigger), reach_1R 1.000 (vs 0.770), wrong_way 0.000 (vs 0.082). Only shape_tag (`unclassified` due to dead zone) and local_peaks (centroid 30.94 — see Finding 4) block 2.0×. Composite landed at 1.0× because 1.0× is the *only* SL that clears shape_tag, not because 1.0× is the structurally optimal SL for c4.

This pattern does not repeat in c1: c1's selected SL=2.0× passes shape_tag (heavy_right_tail across all six SLs) — c1 is composite-selected on structural grounds. c0, c2, c3 fail numerical floors before shape_tag is consequential.

### Finding 3 — mono pre-peak margin thin (c4)

c4 crosses the 0.55 mono floor with +0.016 margin at the selected SL (0.566 vs 0.55). Hypothesis directionally confirmed — pre-peak measurement does rescue c4 — but the post-peak choppiness effect is smaller than the v2.0 closure narrative implied. The full-window → pre-peak delta is +0.036 (0.530 → 0.566), of which only +0.016 lands above the floor.

For c1 the delta is much larger (full-window 0.501 → pre-peak 0.609 at selected SL, +0.108) and the margin is +0.059 — a more robust pass.

### Finding 4 — local_peaks ceiling deferrals

Only c4 produces DEFERRED status on the local_peaks floor (centroid 30.94 at SL=2.0×, 3.0×, 4.0× — outside the §11 Stepwise 5–30 range but within the relaxed ≤35 ceiling encoded in the YAML for counterfactual reporting). At c4's selected SL=1.0×, the pre-peak local_peaks centroid drops to 18.68 — cleanly inside the Stepwise range. So the ceiling-extension question (LIVE doc Issue #2) is moot for c4's actual selection.

If the classifier dead zone were resolved and c4 selected at 2.0× instead, the local_peaks deferral would become live. None of the other surviving or aggregate cohorts touch this floor.

### Finding 5 — cap-binding partial reduction

cap_binding_rate at each surviving cohort's selected SL:

| cohort | selected SL | cap_binding_rate | v2.0 cap_bind (full-window 2.0×) |
|---|---|---|---|
| c1 | 2.0×ATR | 0.090 | n/a (v2.0 didn't report per-cluster) |
| c4 | 1.0×ATR | 0.443 | 0.877 |

c4's cap-binding halved at the tighter selected SL but is still substantial (44.3% of trades hit the 240-bar window cap without SL touch). §5 pool-level auto-extend trigger (>20%) is not reached at the pool level (c4 alone is 14.5% × 0.443 ≈ 6.4% of pool), but the cohort is still partly censored. c1 by contrast is mostly SL-resolved at its selected SL (9.0% cap-bind) — a cleaner cohort.

### Finding 6 — c1 rescued (not in §16 hypothesis)

c1 (43% of pool, the largest cluster) was not on the §16 Open-18 hypothesis list — that list named c4 (this replay), Arc 3 Stepwise climber, and Arc 2 redo c2. v2.0 logged c1 as `unresolved_f22d041e`, failing only clean_shape (full-window mono 0.501, miss 0.049) with otherwise clean §2 metrics (heavy_right_tail, fwd_mfe_p50 1.70R, reach_1R 0.742, wrong_way 0.060). Under v2.1.1's pre-peak measurement, c1's mono lifts to 0.609 at the composite-selected SL=2.0×ATR — a +0.108 delta, far larger than c4's +0.036.

That this surfaces in the *bare-signal* 842-trade population means a second structurally clean cohort exists at the entry stage of KH-24 beyond the post-filter c4 of the §14 anchor. Implications for cross-arc anchor refresh and KH-24 v2.0 disposition are chat-side concerns.

## Anchor refresh implications

c4 at SL=1.0×ATR is anchor-eligible by population characteristics (passes §2 under v2.1.1 conjunctively). c1 at SL=2.0×ATR is also anchor-eligible by the same test, with a larger pool and cleaner cap-binding profile. **Recommendation: DEFER anchor refresh.** Reasons:

- c4's selection is driven by the shape_tag dead zone (Finding 1), not structural truth (Finding 2)
- c4's selected SL 1.0×ATR is half the original §14 reference (2.0×ATR) — different population characterization, even though §14 wording says "re-run KH-24 v2.0 self-test"
- c4's cap-binding still substantial (44.3% at selected SL)
- c1 wasn't on the §16 hypothesis list but passes more robustly than c4 — anchor refresh now would have to choose between c4 (intended cohort) and c1 (cleaner cohort), with no protocol guidance
- §14 anchor was historically measured on the *filtered deployed* population (214 trades, post-1H CIR + currency cap); both c1 and c4 are from the *bare signal* (842 trades, no filters) — different denominator and selection criteria

Chat-side synthesis after Replays #1 and #3 land should:
- Weigh c4 and c1 as candidates from this replay alongside any Arc 3 Stepwise / Arc 2 c2 survivors
- Decide whether to refresh §14 with bare-signal c4, bare-signal c1, or hold the deployed-population reference
- Address the shape_tag dead zone calibration before any anchor commitment

If c4 had failed v2.1.1: anchor refresh blocked; §14 wording stays "refresh pending." It did not fail, so the decision becomes live but is deferred.

## Interpretation

v2.1.1's two main fixes — pre-peak monotonicity and MFE-based shape_tag — both did real work on c4. Pre-peak mono lifted the cohort over the 0.55 floor (+0.036 delta from full-window, +0.016 over the floor). MFE-based shape_tag flipped the censoring-artifact `scattered` label to `heavy_right_tail` at the selected SL. The §16 Open-18 hypothesis (c4 rescued via post-peak fix and final_r censoring fix) is directionally correct.

But the rescue is partial in character. c4's selected SL is half the original v2.0 reference, driven by a classifier dead zone, with cap-binding still 44.3%. The wide-SL configurations where c4's trend-rider structure is most apparent are blocked by mechanics (dead zone), not numerics.

The surprise finding is c1 — the largest cohort in the pool, near-stepwise by centroid, passes more robustly than c4. v2.0 missed it on the same mono-floor mechanism that caught c4, and v2.1.1's pre-peak fix lifts it by +0.108 (vs c4's +0.036). c1 is the cohort the protocol's mechanics most clearly rescue.

Across the full 5-cluster replay: 2/5 clusters (c1, c4) survived §2 under v2.1.1 individually; the c0+c2 peak_and_collapse aggregate fails. The two survivors are both near-stepwise by centroid and both heavy_right_tail by MFE shape_tag — i.e. the rescued archetype is *trend-following with intra-trade peaks*, which is structurally consistent with KH-24's deployed system. v2.1.1 changes the v2.0 outcome from "0 of 5 clusters survive" to "2 of 5 clusters survive plus 1 aggregate evaluated and fails" — a non-trivial expansion of the protocol's discriminative behaviour against the same signal.

## Files

All under `results/replays_v2_1_1/kh24_v2_c4/`:

- `archetype_summaries.csv` — per cluster + aggregate row, selected SL, composite, verdict
- `c{0,1,2,3,4}_sl_sweep.csv` — full per-SL sweep per cluster (6 rows each)
- `c{0,1,2,3,4}_selected_sl.csv` — single-row selection per cluster
- `c{0,1,2,3,4}_distribution.csv` — distribution detail at selected SL (or best-attempt SL for non-survivors)
- `agg_peak_and_collapse_sl_sweep.csv` — aggregate sweep for c0+c2
- `agg_peak_and_collapse_selected_sl.csv` — aggregate selection (none — fails)
- `agg_peak_and_collapse_distribution.csv` — aggregate distribution at best-attempt SL
- `capturability_pass_list.csv` — surviving archetypes (c1, c4)
- `cluster_routing.csv` — per-cluster individual/aggregate disposition
- `REPLAY_RESULT.md` (this file)

Step 1 commits: b552fd1, c148dd8 (scaffold + ruff cleanup)
Step 2 commit: [filled in post-commit by chat]

### Note on Step 2 generation

The Step 1 script (`scripts/replays_v2_1_1/kh24_v2_c4/step3.py`) implements per-cluster SL sweep and selection but did not implement the aggregate-by-archetype-label evaluation or the archetype/pass-list/routing/distribution output writers that Step 2 requires. Per the Step 2 dispatch, the script is locked unless a real bug surfaces. Rather than modify the script, this Step 2 used a one-off helper (not committed) that imports the locked primitives (`evaluate_sl_for_cluster`, `select_sl`, `classify_shape_tag`, `load_inputs`, `write_cluster_outputs`) and computes the missing pieces. All per-cluster sweep outputs are byte-identical to running the script directly; aggregate and summary outputs are produced by the helper. Determinism verified by re-running the helper to a second directory and diffing all output files (identical).
