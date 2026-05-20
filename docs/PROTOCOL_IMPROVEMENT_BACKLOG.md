# Protocol Improvement Backlog — v2.0 → v2.1

> Source: review of `ARC_KH24_V2_RESULT.md`, `ARC_2_REDO_RESULT.md`, `ARC_3_RESULT.md` plus chat-raised issues 2026-05-16.
> Status: record-only. Per §1.8 / §12, no protocol edits applied within-arc. Items here feed the post-Arc-5 cross-arc calibration review (or earlier if the §2 pattern continues into Arc 4).
> Calibration anchor preservation rule (§14): every proposed change must be checked against "does KH-24 K=4 archetype 3 still clear v2.0 extractability (E or D1) under the new rule?" Any fix that breaks the anchor is rejected by construction.

---

## Summary status

| Resolution | Count | Items |
|---|---|---|
| Resolved in v2.1 / v2.1.1 | 11 | P0.1 (v2.1 + v2.1.1 composite refinement), P0.2, P0.3, P0.4, P0.5, P1.6, P1.9, P1.11 (via PR #131), P2.13, P2.15, P3.16 |
| Partial in v2.1 | 1 | P1.8 |
| Closed in v2.2 | 4 | §8 max-F1 fallback (v2.2 §3); mid-arc analyst sign-off carve-outs spanning §9/§12/§16a (v2.2 §1/§2/§5/§6); FIFO arc selection state file (v2.2 §4 new §15b); live-execution equivalence (v2.2 §7 new §1a, asserted not closed). Note: v2.2 §1 sign-flip mechanisation obsoleted by v2.3 §9 removal — the gate it mechanised no longer exists. |
| Closed in v2.3 | 3 | Open-22 (full-pool gate at §9, structural removal in v2.3 §1); Open-23 (Pipeline D1 cost-language, documentation correction in v2.3 §4); Open-24 (Pipeline D1 pre-t SL per archetype, protocol spec in v2.3 §5 — engine PR pending) |
| Still open | 5 | P1.7 (refresh execution — pending KH-24 v2.0 re-run only under v2.1.1), P1.10, P1.12, P2.14, **Open-25 (v2.4 §1.5 entry-separability gate — Arc 8 closure 2026-05-18; evidence Arcs 4 RERUN / 5 / 8; HIGH PRIORITY)** |
| Partial in v2.2 | 1 | Open-21 (Step 4 deployability gate) — proposal (a) strict-mode max-F1 fallback closed by v2.2 §3; alternates (b) recall floor 0.30 + (c) AUC floor 0.70 remain on calibration backlog |

Last updated: 2026-05-19 alongside Arc 8 + Arc 9 + Arc 10 + Arc 11 closure housekeeping. **Arc 8** PR-HHHL long CLOSED 2026-05-18 HALT_DEPLOYMENT; v2.4 §1.5 entry-separability gate proposed as Open-25 (HIGH PRIORITY, evidence Arcs 4 RERUN / 5 / 8). **Arc 9** closed STEP_4_KILL_REAFFIRMED 2026-05-19 after producer-leak patch; v2.x Amendment 1 added (producer-level causal audit dimension, highest priority); v2.x §8 D1 feature-budget expansion WITHDRAWN; v2.x §3 threshold-grid replacement WEAKENED. **Arc 10** DLR closed STEP_4_HALT 2026-05-18 per §16a Path A (first arc end-to-end under v2.3); post-closure EXP-01–06 + WFO pair (base / oracle c1) complete; Open-06 WEAKENED, Open-04 DEFERRED; five new items + two promoted candidates (`L1_minus_L0_atr`, V-shape cross-arc abstraction) documented in "Arc 10 cross-arc items" section. **Arc 11** SHB long 4H closed-HALT 2026-05-18; Pipeline DE + `min_observation_bars` filed as v2.4 candidates. Prior: 2026-05-18 L_ARC_PROTOCOL v2.3 amendment landed (Step 5 cross-fold stability removed; Step 6 WFO renumbered as Step 5; Open-22/23/24 closed in protocol; engine PR `feat/open-24-pre-t-sl-per-archetype` merged 2026-05-19 as PR #146); v2.2 amendment landed earlier same day.

---

## v2.x Amendment Proposals (revised 2026-05-19 after Arc 9 producer-leak incident)

Full text in `L_ARC_PROTOCOL_v2_x_AMENDMENT_PROPOSAL.md` (revised). Pre-PR requirements at the proposal doc's Migration table.

### v2.x Amendment 1 — Producer-level causal audit dimension (NEW, highest priority)

**Status:** Proposed (drafted 2026-05-19)
**Priority:** Highest — failure mode demonstrated, costs were severe, fix is mechanical and cheap
**Evidence base:** Arc 9 producer-leak incident (single arc, direct empirical)
**Source:** `L_ARC_PROTOCOL_v2_x_AMENDMENT_PROPOSAL.md` Amendment 1

#### Problem

The existing audit framework checks join-level causality and end-to-end probability reproduction. It does not check whether values within joined rows are causally constructed. Arc 9 incident: ±10-bar centred swing detector at D1 frame level produced features whose values at each row depended on up to 10 future bars relative to the row's date. `merge_asof` join was correct; values inside the joined rows were not. Original audit returned 8/8 GREEN. Classifier AUC inflated by +0.23 on the leaked features. Deployment chain initiated on fake economics. External audit detected the miss.

#### Proposal

For every feature in a classifier's feature matrix, the value at any given row must depend only on data with timestamp ≤ that row's nominal date (after any specified lag). Standard mathematical definitions that use centred or bilateral windows are non-causal by default and must be replaced with one-sided or confirmation-lag variants.

Audit requirement: every classifier audit must include per-feature producer-level causal verification, distinct from join-level causality and end-to-end probability reproduction.

#### Pre-PR requirements

- KH-24 producer-level audit (anchor preservation verification)
- Causal-only feature library implementation
- Updated CC lookahead audit dispatch template (dimension 9 = producer-level causal verification)

#### Related items

- `results/l_arc_9/INCIDENT_2026_05_19_ARC_9_PRODUCER_LEAK.md`
- `results/l_arc_9/ARC_9_CLOSURE.md`
- `L_ARC_PROTOCOL_v2_x_AMENDMENT_PROPOSAL.md`

---

### v2.x §8 D1 feature-budget expansion — WITHDRAWN 2026-05-19

Status changed from "proposed" to "withdrawn." Arc 9 evidence (originally cited as direct empirical support) was on a classifier with leaked features. With causally-clean features, the AUC lift from D1 + session feature expansion collapses to ≈ 0. Proposal requires different empirical support before re-listing.

---

### v2.x §3 threshold-grid replacement — WEAKENED 2026-05-19

Arc 9's empirical contribution (Candidate B at threshold 0.05 producing +11pp ROI over Candidate A) is invalidated by leaked classifier. Arc 7 calibration recovery experiment becomes the load-bearing evidence. Arc 7 calibration recovery test recommended as separate dispatch.

---

### v2.x Step 5 fold-1 warmup convention — LIVE

Survives independent of the Arc 9 leak (data-window mismatch with KH-24 anchor; applies to any future arc whose data window starts at F1 OOS_start).

---

### v2.x Worst-day DD as standard Step 5 output — LIVE

Methodology engineering-level, signal-agnostic. Arc 9 numerical results invalidated; methodology itself survives and applies to every future Step 5 evaluation. KH-24 worst-day DD characterisation recommended pre-PR (independent of v2.x landing).

---

## P0 — Protocol currently not producing systems

Three arcs have closed at Step 3 §2 floors with archetypes carrying strong forward magnitude (KH-24 v2.0 c4 fwd_mfe_p50 6.65R / reach_1R 1.000; Arc 2 c2 t-stat +52.17 / mfe_p50 5.83R; Arc 3 Stepwise mfe_p50 3.34R / reach_1R 83.6%). The pattern is now systemic, not coincidental. These four items address it.

### P0.1 — Path-quality metrics measure full held-window, not path-to-peak

**Status:** RESOLVED in v2.1 protocol amendment 2026-05-17 (§2/§7 pre-peak metrics); refined in v2.1.1 (§7 capturability composite for SL selection among passing SLs). v1.3 forward-window extension already provides the SL-free continuation required — no engine PR needed (corrected in v2.1.1). Closed-arc re-runs under v2.1.1 (Open-18) runnable on existing `trades_paths.csv`.

**Original status (2026-05-16):** new, user-raised 2026-05-16. Subsumes part of Open-01 (which currently scopes only path-shape clustering features, not §2 forward-geometry).

**Problem statement.** §2 capturability uses metrics that average or aggregate over the full held window: `frac_wrong_way`, `monotonicity_ratio_in_profit`, `local_peaks_count`, `pullback_magnitude_median`. A trade that ascends cleanly to +5R MFE, then crashes back to −1R MAE on the way to the time exit, gets penalised on every one of these metrics — even though the *path to peak* (the portion we actually need to be clean to extract R via a trailing stop or MFE-lock) was textbook clean.

**Why it's an issue.** Capturability asks: "if we knew this trade belonged to this archetype, could we extract R from it?" The answer depends almost entirely on path quality *up to peak MFE*. Post-peak behaviour matters for exit policy design (peak-and-collapse archetype, V-shape archetype) and for archetype identification, but it should not contribute to whether the archetype clears the capturability gate. The current metrics conflate these two concerns.

This is the cleanest single explanation for why three different signals, with three different mechanisms, all surface high-magnitude archetypes that fail §2. The cohorts have real edge to peak MFE; the metrics measure the noise that arrives after the peak and reject the cohorts on that basis.

**Predicted root cause.** The protocol inherited "monotonicity over the whole in-profit window" and "frac_wrong_way over the whole held window" from the v1.0 era where the clustering basis was forward-geometry magnitude — there, "did the trade survive cleanly to its time exit" was the right question because the exit policy was implicit / fixed. v2.0 separates capturability from extractability and introduces archetype-specific exit policies (§11), which means the metric definitions should have been bisected at the same time but weren't. Open-01 ackowledges this for clustering features; the §2 forward-geometry floors slipped through.

**Resolution.** Split each affected §2 metric into pre-peak and post-peak variants. Pre-peak variants gate capturability; post-peak variants inform archetype identification and exit-policy choice.

| Current metric | Pre-peak variant | Post-peak variant |
|---|---|---|
| frac_wrong_way | MAE ≤ −1R reached before peak MFE bar | MAE drawdown from peak MFE ≥ X R |
| monotonicity_ratio_in_profit | among bars 0..peak_mfe_bar with close_r > 0 | (not needed at §2 — exit policy concern) |
| local_peaks_count | bars 0..peak_mfe_bar | bars peak_mfe_bar..exit |
| pullback_magnitude_median | peak pairs in 0..peak_mfe_bar | post-peak retracement magnitude |
| pct_peak_and_collapse | (this is already post-peak by definition — keep) | — |

§2 capturability floors then apply to pre-peak variants only. Post-peak variants flow into §11 routing (peak-and-collapse vs V-shape vs sustained vs split-exit).

Operationalisation: peak MFE bar is observable in `trades_paths.csv` post-hoc. No new data collection needed. Implementation is a Step 1 → Step 3 derived-metrics change, not a backtester change.

**Impact on KH-24 anchor.** KH-24 K=4 archetype 3 has frac_wrong_way 0.04 and monotonicity 0.576 under current full-window definitions. Pre-peak variants will be at least as favourable (the trade either had MAE before peak or not — pre-peak frac_wrong_way ≤ current frac_wrong_way for any trade). Anchor preservation holds.

**Impact on closed arcs.** Re-evaluation under pre-peak variants is mechanical from existing trades_paths data. Will quantify how much of the §2 failure pattern is genuine path-quality failure vs post-peak contamination of the metrics. If pre-peak metrics show e.g. Arc 3 Stepwise wrong_way ~20% (vs 38% full-window), that's strong evidence the rest of the §2 issues collapse to this one.

**Priority.** P0 highest. Almost certainly the largest single lever on the §2-failure pattern. Cheaper than every other P0 fix (no SL changes, no aggregation logic, no new clustering — just recomputing existing metrics on a clipped window).

**Dependencies.** Independent. Output feeds calibration of P0.4 (the SL/horizon fix becomes less urgent if pre-peak metrics already pass) and P0.5 (the monotonicity floor calibration becomes a different question once measured pre-peak only).

---

### P0.2 — §2 shape_tag floor excludes bimodal; §11 row 7 routes bimodal

**Status:** RESOLVED in v2.1 (§7 bimodal_separated test, §2 admit, §11 row 7 routing).

**Original status (2026-05-16):** Open-13 in protocol §16. Highest-priority cross-arc item in current backlog.

**Problem statement.** §2 requires `shape_tag ∈ {tight_unimodal, heavy_right_tail}`. §11 row 7 defines bimodal fwd_mfe distribution (two modes ≥ 1R apart) as a valid archetype with its own exit policy: "Half-off at TP1 (lower mode), trail remainder." The shape that §11 has an explicit policy for cannot reach §11 because §2 kills it.

**Evidence.** Arc 3 Stepwise climber (clusters 2+4, 27.5% pool, n=707): passes monotonicity 0.559, mfe_p50 3.34R, reach_1R 83.6%, size cleanly; fails only on shape_tag=bimodal and frac_wrong_way 0.383. Final R distribution textbook split: p25 −1.00R, p50 +1.85R, p75 +3.80R — exactly the distribution §11 row 7 was written for.

**Why it's an issue.** Internal protocol contradiction. The capturability framing implies "single mode + tail" is the only clean structure, but a bimodal distribution with separated modes is exactly the case where a split exit (TP1 at lower mode, trail at upper) harvests measurable R from both sub-populations. Bimodal-with-separation is a structured outcome, not noise.

**Resolution.** §2 admits `bimodal` when:
- modes meet ≥ 1R separation criterion (Hartigan dip test for bimodality + explicit mode-distance check)
- AND the archetype is routed to §11 row 7

Operationally: introduce a new shape_tag value `bimodal_separated`. §2 allows `{tight_unimodal, heavy_right_tail, bimodal_separated}`. Generic `bimodal` (modes too close, or one mode << other) stays excluded — that's a noisy distribution, not a structured one. §6/§7 shape_tag taxonomy splits `bimodal` accordingly.

**Impact on KH-24 anchor.** KH-24 K=4 archetype 3 shape_tag is currently `bimodal (right-mode dominates)` per §14. Under the new taxonomy this would route to `bimodal_unstructured` (one mode dominates, modes not ≥ 1R apart in equal mass) — still excluded — OR `bimodal_separated` depending on operationalisation. Need to verify before commit. If KH-24 anchor archetype lands in `bimodal_unstructured`, anchor preservation works (anchor passes via D1 t=3 already, independent of shape_tag).

**Priority.** P0. Concrete signal-rescue lever; one of the closures (Arc 3) directly attributable.

**Dependencies.** Defines operationalisation for `bimodal_separated`. Use Hartigan dip statistic at p < 0.05 + min-mode-mass ≥ 0.20 + mode separation ≥ 1R as a starting spec.

---

### P0.3 — Same-archetype aggregation hides capturable sub-clusters

**Status:** RESOLVED in v2.1 (§7 per-cluster + per-aggregate evaluation).

**Original status (2026-05-16):** Open-14 in protocol §16.

**Problem statement.** §6 aggregates same-archetype clusters before §2 evaluation. §2 mixes path-shape criteria (mono, local_peaks) with forward-geometry criteria (frac_wrong_way, frac_reach_1R). Clusters sharing an archetype label by centroid pattern can differ substantially on forward geometry — aggregation evaluates a hybrid that does not exist as a real trade population.

**Evidence.**
- Arc 3 Early-peak hold = clusters 0+3. Cluster 0 mono 0.008. Cluster 3 mono 0.579 (passes §2 floor alone). Aggregated mono 0.251. Aggregation killed cluster 3's individual capturability.
- Arc 3 Stepwise climber = clusters 2+4. Cluster 2 local_peaks 24.42 / pc 0.126. Cluster 4 local_peaks 7.79 / pc 0.474. Aggregated wrong_way 38.3% probably hides cluster 2 ≈ 25–30% (passes) and cluster 4 ≈ 50%+ (fails).

**Why it's an issue.** The "share an exit policy" rationale only holds if both clusters get the same downstream treatment. If cluster 2 passes §2 and cluster 4 fails, the shared-policy logic collapses — cluster 4 dies regardless of cluster 2's outcome. Aggregating first then evaluating uses the failing cluster's path-shape to filter out the passing cluster — exactly backwards.

**Resolution.** §2 evaluation per-cluster. §6 aggregation rule reframes as "candidate aggregation": clusters that individually clear §2 AND share centroid pattern get aggregated for Step 4+ exit-policy design. Clusters that individually fail §2 die regardless of same-label siblings.

Side benefit: this disambiguates the §11 boundary-cluster rule. Currently "boundary clusters: assign by empirical test on per-fold internal validation." Under per-cluster §2 eval, boundary cases that fail §2 just die — no validation runaround needed.

**Operational definition for "share an exit policy":** clusters share the same §11 row AND their per-§2-floor disparity ≤ X% (e.g., 10%) on each criterion. Otherwise treat as separate sub-archetypes carrying the same §11 exit policy label but evaluated and deployed independently.

**Impact on KH-24 anchor.** KH-24 K=4 archetype 3 is a single cluster, not an aggregate. Rule change does not touch it. Anchor preservation holds.

**Priority.** P0. Direct evidence from Arc 3. Quantified evidence for Arc 3D (cluster 2 vs cluster 4 separately) would close this completely.

**Dependencies.** Independent. Output may interact with P0.1 (if pre-peak metrics narrow the disparity between clusters that share an archetype, aggregation becomes more defensible — measure before deciding).

---

### P0.4 — SL distance / horizon asymmetry structurally inflates frac_wrong_way

**Status:** RESOLVED in v2.1 (§7 SL sweep, §11 SL column demoted, §17 R-unit definition updated).

**Original status (2026-05-16):** Open-15 in protocol §16.

**Problem statement.** When SL distance (in volatility units) is small relative to expected horizon price movement, false stop-outs are statistically guaranteed regardless of signal quality. Arc 3 used 2.0 × ATR_1H on 120-bar horizon — total expected movement ~√120 × per-bar-σ, so 2 ATR ≈ 0.18σ of horizon. Even a directionally neutral random walk should breach this distance frequently.

**Evidence.** Arc 3 — all three full-evaluation archetypes failed on frac_wrong_way:
- Early-peak hold 98.8% (aggregation artefact)
- Stepwise climber 38.3%
- Cluster 1 73.2%

**Why it's an issue.** §2's `frac_wrong_way ≤ 0.30` floor was calibrated with KH-24 in mind, where SL = 2 × ATR_4H on a 4H signal with ~40-day forward window. The SL-to-horizon ratio in KH-24's frame is different by an order of magnitude from Arc 3's 2×ATR_1H on 120 1H bars. A fixed wrong_way ceiling applied to populations with widely varying SL/horizon ratios filters by ratio, not by signal quality.

**Resolution.** Three options, increasing structural scope:

1. **Archetype-specific initial SLs (set at Step 4).** §11 already nominally provides per-archetype SLs (Monotone 1R, Stepwise 1.3R, Early-peak 0.8R, etc.). These are R-multipliers on top of an arc-level base. The arc-level base is what's miscalibrated; §11 multipliers don't compensate for arc-level SL/horizon asymmetry. Requires Step 4 to actually run, which Arc 3 didn't reach.

2. **Arc-level SL scaled to horizon at arc open.** `SL = 2.0 × ATR × √(h/24)`. For h=120 this is ~4.5× ATR_1H. Brownian-motion-consistent default, applies cleanly to any horizon, becomes v2.1 protocol default.

3. **Horizon-aware frac_wrong_way floor.** Keep SL fixed, scale the §2 ceiling by horizon/SL ratio. Embeds the asymmetry in the gate rather than fixing at source — less clean.

Path 2 is the cleanest fix. Path 1 still needed at Step 4 regardless.

**Interaction with P0.1.** If pre-peak frac_wrong_way is much lower than full-window (which is the prediction), Path 2 becomes less critical. Run P0.1 first, see what frac_wrong_way values land on, then decide whether P0.4 is still needed or to what extent. Plausible outcome: pre-peak frac_wrong_way at current SL is comfortably under 0.30 for capturable archetypes, and P0.4 becomes a refinement rather than a blocking fix.

**Impact on KH-24 anchor.** Path 2 would change KH-24's SL from 2× ATR_4H to 2× ATR × √(40-day-h/4-h) ≈ 7.7× ATR_4H — a major change to live signal. **Path 2 cannot be retrofit to KH-24 without rebuilding the entire system.** Anchor preservation requires: Path 2 applies forward to new arcs only; KH-24 keeps current SL by exception, anchored on its already-passing forward geometry under v1.0 era SL. Document this exception explicitly if Path 2 is adopted.

**Priority.** P0 in current framing but possibly P1 after P0.1 lands. Measurement (Arc 3D) tells us which.

**Dependencies.** Measure pre-peak frac_wrong_way (P0.1) first. Calibrate P0.4 on result.

---

### P0.5 — §2 monotonicity floor 0.55 may be too high

**Status:** RESOLVED in v2.1 — monotonicity now computed pre-peak per §7. Floor stays 0.55. Closed-arc re-run will confirm whether near-miss pattern resolves.

**Original status (2026-05-16):** Cross-arc calibration backlog, HIGH priority.

**Problem statement.** Three arcs near-miss the 0.55 floor with otherwise strong cohorts:
- KH-24 v2.0 c4: missed by 0.020, fwd_mfe_p50 6.65R, frac_reach_1R 1.000, frac_wrong_way 0.000
- Arc 2 redo c2: missed by 0.009, fwd_mfe_p50 5.83R, t-stat +52.17 (n=2,278)
- Arc 3 Stepwise: passes by 0.009 (mono 0.559), but fails on other §2 criteria

**Why it's an issue.** Margin misses are small (≤ 0.02). The cohorts on the other side of the miss have textbook-edge forward geometry. The floor was calibrated against KH-24's filtered deployed population (mono 0.576), where v1.0's 1H CIR + currency cap filters had pre-selected for cleaner-shaped paths. Applied to bare or differently-filtered signals, the floor is structurally hostile.

**Predicted root cause.** Almost certainly an artefact of P0.1. Monotonicity over the full in-profit window includes post-peak choppiness that has nothing to do with whether the trade can be captured. Recomputed pre-peak only, the same cohorts will likely show monotonicity well above 0.55.

**Resolution.** Two parts.

1. Apply P0.1 first. Recompute monotonicity on bars 0..peak_mfe_bar. Almost certainly the near-misses become comfortable passes.

2. If pre-peak monotonicity still surfaces marginal cohorts: consider whether the conjunctive AND should soften to "k of 6 §2 criteria" (Open-09), or whether monotonicity centroid should be replaced with monotonicity-median-across-cluster-trades (more robust to outliers).

**Impact on KH-24 anchor.** Anchor at mono 0.576 — passes 0.55 by 0.026. Floor reduction (if pursued) is anchor-safe. P0.1-driven pre-peak monotonicity for anchor will be ≥ 0.576 (clean ascent before peak).

**Priority.** P0 in current framing; almost certainly subsumed by P0.1.

**Dependencies.** P0.1.

---

## P1 — Methodology / documentation gaps

### P1.6 — `frac_wrong_way` definition missing from §17 glossary

**Status:** RESOLVED in v2.1 (§17 Def C).

**Original status (2026-05-16):** Cross-arc calibration backlog, MEDIUM priority.

**Problem statement.** The protocol uses `frac_wrong_way` throughout §2 but the §17 glossary does not define it. Three plausible definitions exist:

- Def A: `final_r ≤ −0.5R`. Gives nonsense on hard-SL designs (trades stopped out at −1R are always counted; trades that recover from MAE to slight profit are always missed).
- Def B (KH-24 v2.0 self-test): MAE ≤ −1R reached before MFE > 0.5R reached, OR MFE > 0.5R never reached. The "wrong from outset" interpretation.
- Def C (P0.1, proposed): MAE ≤ −1R reached before peak MFE bar.

Arc 3 used... unclear; closure doc doesn't specify. Arc 2 redo doesn't specify. KH-24 v2.0 explicitly ratifies Def B.

**Why it's an issue.** Three arcs may have used three different definitions. Cross-arc comparison is invalid if the definition isn't held constant. The §14 anchor numbers are measured under whatever definition was in use at the time and may not match the protocol-as-written.

**Resolution.** Two-step.

1. Ratify Def C (P0.1) as the protocol definition. Add to §17:
   > **frac_wrong_way:** Fraction of trades where MAE ≤ −1R is reached on or before the peak MFE bar. Trades that breach −1R only *after* peak MFE are post-peak collapse and are counted by `pct_peak_and_collapse`, not `frac_wrong_way`.

2. Re-evaluate KH-24 anchor under Def C. The anchor's current 0.04 wrong_way is under Def B; Def C value will be ≤ Def B value, so anchor still passes the 0.30 floor.

**Priority.** P1. Fix as part of P0.1 commit (same edit lands).

**Dependencies.** P0.1.

---

### P1.7 — §14 anchor population vs §15 pool floor structural mismatch

**Status:** OPEN — v2.1.1 §14 defines refresh path; refresh execution pending KH-24 v2.0 re-run only. No engine PR needed — v1.3 forward extension predates the requirement and already provides the SL-free observation (engine-reality correction in v2.1.1).

**Original status (2026-05-16):** Cross-arc calibration backlog, HIGH priority. Surfaced by KH-24 v2.0 self-test.

**Problem statement.** §14's anchor numbers (mono 0.576, fwd_mfe_p50 5.40R, etc.) are measured on KH-24's filtered 214-trade deployed population. §15's pool floor of ≥ 500 trades structurally excludes that population. The protocol's anchor describes one population; the protocol's mechanics describe a different one.

**Evidence.** KH-24 v2.0 self-test (bare signal, no filters) produced 842 trades. Its closest archetype-equivalent (c4) is 122 trades — close to deployed (214) but on a different filter regime. Re-deriving the anchor on the unfiltered v2.0-eligible pool produces fundamentally different numbers.

**Why it's an issue.** Two problems.

1. Anchor preservation rule (§14) is conceptually fragile. "Don't break the anchor" is the rule, but the anchor isn't reproducible under v2.0 mechanics — what does "break" mean for a population the protocol can't construct?

2. Cross-arc calibration decisions might be made on a comparison ("does this change preserve the anchor?") that's effectively comparing v2.0-eligible cohorts to v1.0-filtered ones. The comparison is structurally invalid.

**Resolution.** Two-step.

1. Re-derive §14 on a v2.0-compatible KH-24 population. Run KH-24 v2.0 self-test through Steps 1-4 and use the resulting "best capturable + extractable archetype" as the new anchor. (Currently the self-test died at Step 3, so this requires fixing P0.1 / P0.4 first.)

2. If no v2.0-compatible KH-24 anchor exists after P0 fixes, document the anchor mismatch explicitly: "anchor is a v1.0 reference, v2.0 cannot reproduce it; calibration changes are evaluated on forward arcs not against the historical anchor."

**Priority.** P1. Blocked by P0.1, P0.4.

**Dependencies.** P0.1, P0.4.

---

### P1.8 — shape_tag rules don't compensate for forward-window censoring

**Status:** PARTIAL in v2.1 — bimodal_separated admission covers part of shape_tag tightness. Censoring-vs-shape_tag tension remains pending re-run evidence.

**Original status (2026-05-16):** Cross-arc calibration backlog, HIGH priority. Surfaced by KH-24 v2.0 self-test.

**Problem statement.** When N% of a cohort hits the forward-window cap, their final_r is censored (recorded at whatever point on a still-running trajectory the cap fell). Different trades got clipped at different points, producing wide spread in final_r even when underlying MFE distribution is clean.

**Evidence.** KH-24 v2.0 c4: 87.7% of trades hit the 240-bar cap. shape_tag classified as "scattered" from censored final_r distribution. Underlying fwd_mfe distribution was heavy_right_tail with p50 6.65R.

**Why it's an issue.** shape_tag is supposed to characterise the structural shape of cohort outcomes. When measured on censored final_r, it characterises the censoring pattern instead. False rejection of capturable cohorts.

**Resolution.** Three candidates, increasing scope:

1. **Censor-aware shape_tag.** Compute shape_tag only on uncensored trades (those that exited before the cap) if uncensored sub-population is ≥ 50.

2. **Derive shape_tag from MFE distribution, not final_r.** MFE is not censored by the time-exit cap — the trade's peak MFE within the window is observable regardless of whether the trade is still running. This is the principled fix.

3. **Skip shape_tag entirely for cohorts with > 50% cap-binders.** Coarsest fix; loses information.

Option 2 cleanest. Implementation: redefine shape_tag bins on `fwd_mfe_h240` percentiles (`p95/p50` for heavy_right_tail, dip test for bimodal_separated, etc.), not on `final_r`.

**Impact on KH-24 anchor.** Anchor currently passes shape_tag (via "tight_unimodal" or "heavy_right_tail" depending on operationalisation). Recomputing on MFE distribution: anchor's fwd_mfe_p50 5.40R, fwd_mfe_p95 likely ~10R+ — heavy_right_tail safe. Anchor preservation holds.

**Priority.** P1. Interacts with P0.2 (bimodal admission) — both touch shape_tag taxonomy.

**Dependencies.** P0.2 (joint redesign of shape_tag taxonomy).

---

### P1.9 — 240-bar forward window too tight for slow trend-following signals

**Status:** RESOLVED in v2.1 (§5 forward window auto-extend at >20% cap-bind, 2× extension).

**Original status (2026-05-16):** Cross-arc calibration backlog, HIGH priority. Surfaced by KH-24 v2.0 self-test.

**Problem statement.** 240 bars = 40 days for 4H. Trend-following signals that run for weeks hit the cap before maturing. Cap-binding distorts both forward-geometry measurement (P1.8) and exit-policy design.

**Evidence.** KH-24 v2.0: 16.7% pool-level cap-binding, 87.7% on c4 (the trend-rider cluster).

**Why it's an issue.** 240 is a one-size-fits-all cap. For mean-reverting or short-horizon signals it's plenty; for trend-followers on 4H it's tight. The cap directly inflates cap-binder fraction, which feeds P1.8.

**Resolution.** Per-arc-configurable forward window in arc config YAML. Default 240 bars retained; arcs can extend to e.g. 480 (80 days for 4H) when signal class warrants. Rule: if Step 1 reports > 25% cap-binding, flag for window extension at arc open of next phase.

**Impact on KH-24 anchor.** Anchor was measured at 240-bar window. Re-running at e.g. 480 bars produces different numbers. Anchor preservation requires either: (a) re-derive anchor at the new window for affected arcs, or (b) document anchor as window-specific and not portable across window settings.

**Priority.** P1. Interacts with P1.8.

**Dependencies.** P1.8.

---

### P1.10 — §11 archetype centroid patterns are first-pass priors

**Status:** OPEN — §11 empirical refresh deferred until Arc 4 + Arc 5 data.

**Original status (2026-05-16):** Open-07 in protocol §16. Cross-arc evidence accumulating.

**Problem statement.** §11 patterns are first-pass priors. Across three arcs most clusters land `unresolved_*` provisional labels. The Random walk and Peak-and-collapse rows have overly tight ceilings (P2.14 below); Stepwise climber and Monotone ascent rows have not been stress-tested against the cluster centroid distributions actually observed.

**Evidence.**
- Arc 3 K=7 → 7 clusters, only 4 mapped to §11 archetypes (Early-peak hold × 2, Stepwise climber × 2), 3 unassigned.
- Arc 2 redo K=4 → 4 clusters, only 2 mapped (Stepwise, Early-peak hold), 2 unassigned.
- KH-24 v2.0 K=5 → 5 clusters, mostly unresolved.

**Why it's an issue.** §11 is supposed to be the exit-policy library. If 50%+ of observed clusters are unresolved, the library is incomplete. Unresolved clusters can't be evaluated past Step 3.

**Resolution.** Empirical §11 refresh after Arc 5. Method:
1. Collect centroid distributions across all closed arcs (KH-24, Arc 2 redo, Arc 3, Arc 4, Arc 5).
2. Cluster the centroids themselves to identify natural archetype regions.
3. Match each region to an exit-policy intuition; commit as new §11 rows.

**Priority.** P1. Wait for Arc 4 + Arc 5 to produce the data needed for refresh.

**Dependencies.** Arc 4, Arc 5 completion.

---

### P1.11 — §3 Pipeline D1 "break-even close" wording

**Status:** RESOLVED in PR #131 (§3 wording fix; "close at market on bar N+1 open").

**Original status (2026-05-16):** chat-raised 2026-05-16.

**Problem statement.** §3 reads: "Trades classifier deems untradeable at bar N: close at break-even or small loss." Wording is loose. At bar N the trade has moved — it could be in profit, at entry, or underwater. "Break-even" literally means closing at entry price, which isn't always available (price has moved) and isn't always what you'd want.

**Resolution.** Revise §3 step 4 to:
> "Trades classifier deems untradeable at bar N: exit at bar N+1 open at market. Realised R is whatever the trade has accrued by then."

Removes the implicit assumption about realised outcome. Same applies to §11 row 6 (Random walk Pipeline D1): "Close at break-even at bar N" → "Exit at bar N+1 open at market."

**Impact on KH-24 anchor.** Anchor is Pipeline D1 t=3 per §14. Wording change is editorial — no impact on backtest mechanics. Anchor preservation holds.

**Priority.** P1. Editorial fix; can land any time.

**Dependencies.** None.

---

### P1.12 — Workflow / commits confused when running arcs in parallel

**Status:** OPEN — main-only-except-engine convention applies in practice but not yet explicit in §13. Editorial fix can land any time.

**Original status (2026-05-16):** chat-raised 2026-05-16.

**Problem statement.** §13 routes arc analysis/results direct-to-main; engine/signal/locked-config/CI/protocol changes need PRs. When two arcs run in parallel chats, governance files (CLAUDE.md, STATUS.md, SESSION_ZERO.md, candidate registry) get touched by both and create merge confusion. CC's session-start branch is unreliable (KH-24 v2.0 Step 1 landed on `arc/l_arc_3_step1`, Step 3 on `feat/d1-pipeline` — both cherry-picked back to main).

**Resolution.** Two options.

1. **Strict main-only workflow.** All arc work goes direct to main. Engine changes still PR (the safety case for engine PRs is independent of arc parallelism). Governance file edits coordinate via chat handover.

2. **Lockfile convention.** Per-arc lockfile in repo (`.arc_<N>.lock`) — when an arc is active, governance file edits require chat-level coordination. Mechanical but explicit.

Option 1 is what's effectively in §13 already for arc analysis. The confusion is about engine-touching changes (KH-24 v2.0 Step 1's analysis script imported `core.spread_floor` as library-use which technically falls outside §13's PR-required list). Tighten the rule: any change touching `core/`, `scripts/phase_kgl_v2_4h_wfo.py`, or `signals/` requires PR — including library imports if the import path crosses module boundaries.

**Priority.** P1. Operational discipline fix.

**Dependencies.** None.

---

## P2 — Calibration refinements

### P2.13 — Open-12 silhouette tie tolerance

**Status:** RESOLVED in v2.1 (§6 tolerance 0.01 absolute).

**Problem statement.** §6 K selection rule "ties: smaller K preferred" lacks tolerance definition. Arc 3 chose K=7 on a 0.0021 margin over K=4. Across K ∈ {3..7} the range was 0.0165 — effectively noise.

**Resolution.** Add tolerance: smaller K preferred when silhouette gap < max(0.01 absolute, 5% relative). Spec into §6.

**Impact on KH-24 anchor.** KH-24 K=4 selection holds (K=4 was the highest-silhouette selection at the time). No impact.

**Priority.** P2.

---

### P2.14 — §11 Random walk / Peak-and-collapse rows over-specified

**Status:** OPEN — folds into §11 empirical refresh (P1.10) after Arcs 4-5.

**Problem statement.**
- Arc 3 cluster 6 matched Random walk's positive criteria (local_peaks 8.37, pullback 1.083R) but failed monotonicity ≤ 0.30 ceiling at 0.504.
- Arc 3 cluster 1 matched Peak-and-collapse signature strongly (pc=0.663) but with peak timing 0.385 vs §11's ≤ 0.30 ceiling.

**Resolution.** Two options:
1. Widen Random walk monotonicity ceiling to ≤ 0.55 OR split into "strict random walk" (mono ≤ 0.30) and "loose random walk" (0.30 < mono ≤ 0.55) variants.
2. Widen Peak-and-collapse time_to_peak ceiling to ≤ 0.40 OR split similarly.

Fold into P1.10 §11 empirical refresh.

**Priority.** P2.

---

### P2.15 — Per-pair n distribution stability concern

**Status:** RESOLVED in v2.1 (§9 per-pair stability reporting at Step 5).

**Problem statement.** KH-24 v2.0 self-test had 15/28 pairs flagged < 30 trades in pool. §5 keeps them in the pool but structural concern remains: if a downstream archetype concentrates in low-n pairs, cross-pair stability is suspect.

**Resolution.** Two candidates:
1. Per-archetype per-pair stability check at Step 5 — for each surviving archetype, report contribution per pair, flag if > 50% from < 5 pairs.
2. Pool-level rule that low-n pairs are excluded from clusters they don't reach a minimum count in.

Option 1 lighter-touch (reporting, not exclusion). Fold into Step 5 spec.

**Priority.** P2.

---

## P3 — Housekeeping

### P3.16 — Open-08 (`pullback_magnitude_median` degeneracy) empirically resolved

**Status:** RESOLVED in v2.1 (§16 Open-08 closed as resolved).

**Original status (2026-05-16):** Open-08 in protocol §16.

**Resolution.** KH-24 v2.0 self-test empirically refutes degeneracy concern (mode fraction 0.31, well under 80% threshold). Close Open-08 as resolved. Editorial fix.

**Priority.** P3.

---

## Arc 4 cross-arc items (2026-05-17)

Items raised by Arc 4 closure (`bar_range_top_decile__neg__h_001`; CLEAN-NULL on transaction-cost truth). Spread floor file replacement is the highest-priority blocker for all future arc work.

### HIGHEST priority — added 2026-05-17 from Arc 4 closure

- **Spread floor file replacement** — `configs/spread_floors_5ers.yaml`'s uniform 0.1 pip floor under-models real spreads by 3-48x per pair. Per-pair empirical floors from HistData audit (or MT5 broker snapshot) required. Locked-file change. Blocks all future arc work.
  Source: Arc 4 closure 2026-05-17.

### HIGH priority — added 2026-05-17

- **Phase Zero spread validation** — Add spread floor validation step before Step 1 plumbing for all future L arcs. Refresh tick-based audit every 6 months. Affects `L_ARC_PROTOCOL.md` §5 and `WORKFLOW.md`.
  Source: Arc 4 closure 2026-05-17.

- **F1 structural leakage** — L arc pool starts 2020-10-01 = F1 OOS start. No honest WFO training data exists for F1. Affects every L arc retroactively (magnitude not direction). Options: pool back-extend to pre-2020, drop F1 from L arc evaluation, or alternate fold structure for L arcs.
  Source: Arc 4 Step 5C 2026-05-17.

### MEDIUM priority — added 2026-05-17

- **Session-aware spread modeling** — Per-pair × per-session floors may be required for accurate cost modeling on signals that fire outside London/NY overlap. Defer until per-pair floor in place and next arc's behaviour observed.
  Source: Arc 4 closure 2026-05-17.

- **Convention (b) MTM DD as §10 default** — 5ers measures account equity in real-time; convention (a) closed-trade ordering understates DD by 14-63%. Convention (b) should become §10's default gate metric. Affects protocol §9 and §10 wording.
  Source: Arc 4 Step 5B-refit 2026-05-17.

- **Step 5 simulator default — apply exit spread** — Post-hoc simulator templates should enforce S/2 exit spread by default per SPREAD_SEMANTICS_LOCK.md. Arc 4's omission was prompt-author error; the simulator template should make it impossible to skip.
  Source: Arc 4 Step 5B-spread 2026-05-17.

- **D1 threshold grid specification** — §3 locks Pipeline E grid {0.40, 0.50, 0.60, 0.70} but never explicitly locks D1's grid. Both Arc 4 and Arc 5 hit this. Lock D1 grid as {base_rate, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50} in protocol amendment.
  Source: Arc 4 Step 5A + Arc 5 Step 4b 2026-05-17.

### LOW priority — added 2026-05-17

- **§11 row 2 deep-pullback tolerance** — Arc 4 cluster 1 carried pullback 0.676R against row 2's ≤0.5R rule. Closed arc, so doesn't matter for Arc 4, but centroid pattern boundaries in §11 deserve empirical refinement once more arcs land.
  Source: Arc 4 Step 4→5 boundary decision 2026-05-17.

---

## Arc 6 cross-arc items (2026-05-17)

Three items raised by Arc 6 closure (failed-breakout reversal long, out-of-registry; DIES at Step 4 deployability). Queued for the cross-arc calibration session before the next out-of-registry arc opens. Numbering follows protocol §16 Open-NN convention (Open-20 reserved/rejected for the realised-R-under-fixed-SL framing per v2.1.2 CHANGELOG; next available is Open-21).

### Open-21 (new): Step 4 deployability gate

- **Source:** Arc 6 closure 2026-05-17, Step 4 disposition.
- **Problem.** §8 arc-level gate at D1 RF AUC ≥ 0.60 passes mechanically even when the threshold sweep falls back to max-F1 at sub-1% recall. Resulting admission rate is non-deployable. Arc 6 demonstration: both Step 3 survivors cleared D1 AUC ≥ 0.60 (c0 0.602 at t=4; c2 0.630 at t=1, growing to 0.711 at t=10) but neither cluster admitted recall ≥ 0.60 at any threshold ∈ {0.40, 0.50, 0.60, 0.70}; both fell back to max-F1 — c0 precision 0.333 recall 0.009 (~3 trades), c2 precision 0.250 recall 0.004 (~1 trade) across the 5-year, 1,564-trade pool. Arithmetic: AUC 0.60 at 15–21% positive class admits a max precision at recall 0.60 of ~0.20–0.25 — barely better than base rate. The §8 design point (recall ≥ 0.60 AND meaningful precision) effectively requires AUC ≈ 0.75+ at this class balance.
- **Proposal (recommended): (a) Strict mode.** Threshold sweep must select on max-precision subject to recall ≥ 0.60. Max-F1 fallback triggers cluster-dies, not graceful pass. The §8 rule's stated intent already encodes recall ≥ 0.60 as the deployability line; make it the gate. Minimal protocol change.
- **Alt proposals.**
  - (b) **Recall floor at admission threshold ≥ 0.30** (or 0.40) regardless of selection rule. Below that, cluster dies.
  - (c) **Higher AUC floor.** Raise §8 D1 AUC threshold 0.60 → 0.70. At 0.70, recall-0.60 thresholds become achievable for class balances in the 15–25% range.
- **Anchor preservation.** KH-24 K=4 archetype 3 passes Pipeline D1 at t=3 with RF AUC 0.638 and exclusion 15.4%. Under (a) the anchor's recall ≥ 0.60 sweep needs verification — anchor numbers were measured under v2.0 mechanics that did not separate the sweep selection from the AUC pass. Verification required before commit.
- **Status.** Queued for cross-arc calibration session before next out-of-registry arc opens. Likely v2.1.3 amendment.

### Open-17 expansion: Tiebreak 1 noise floor

- **Source:** Arc 6 Step 3 c2 SL selection 2026-05-17.
- **Problem.** §7 Tiebreak 1 (larger peak_mfe in ATR units) fires on sub-noise margins. Arc 6 c2 selection: peak_mfe_atr 13.40 (X=3.0) vs 13.38 (X=2.0) — 0.02 ATR absolute, 0.15% relative — flipped SL from X=2.0 to X=3.0 at identical composite (0.6162). Economic consequence: ~50% capital efficiency loss at identical path quality (same dollar risk per trade, ~1.5× dollar MFE under X=2.0). Tiebreak 1's stated purpose ("reward larger physical capture") is not satisfied at noise-level differences.
- **Proposal.** Require `peak_mfe_atr_margin ≥ 0.10 ATR` OR `≥ 1% relative` before Tiebreak 1 applies; otherwise fall through to Tiebreak 2 (parsimony / smaller SL). Preserves the rule's intent while eliminating the noise-driven flip.
- **Anchor preservation.** Anchor (KH-24 K=4 archetype 3) is a single SL selection upstream of Tiebreak 1; rule change does not touch it.
- **Status.** Queued for cross-arc calibration session. Pairs naturally with Open-21 review.

### (unnumbered note) reach_1R floor noise tolerance

- **Source:** Arc 6 Step 3 c3 disposition 2026-05-17.
- **Problem.** c3 dies at `reach_1R = 0.697` vs 0.70 floor — a 0.003 absolute margin, within sampling noise for n=511 (binomial se at p=0.70, n=511 → 1.96·se ≈ 0.040). Within-arc thresholds don't move per §1.8.
- **Question.** Does the floor need a binomial-noise tolerance — e.g., `reach_1R ≥ 0.70 − 1.96 × se`?
- **Trade-off.** Tolerance restores marginal clusters (would meaningfully widen the gate at typical n) but weakens the gate's discriminative power. Same tension exists in principle for every §2 hard floor.
- **Status.** Cross-arc note — both-sides argument required, not a clear-cut calibration. Not blocking next arc.

---

## Arc 4 rerun + Arc 5 closure items (2026-05-18)

Three items raised by the Arc 4 re-run closure (FAIL Step 6 under p50 floors) plus the Arc 5 closure (SHELVED Step 6 FAIL). Both arcs PASSED §9 admit-only stability and FAILED §10 full-pool deployment. Cross-arc structural finding: Pipeline D1 carries mandatory reject-pool + early-exit-pool cost that §9's admit-only framing cannot see. Numbering continues the Open-NN sequence (Open-21 was the last assigned).

### Open-22 — Full-pool gate at §9 or earlier (HIGH) — CLOSED in v2.3

- **Description.** §9 currently evaluates admit-only stability. Arc 4 and Arc 5 both PASS §9 and FAIL §10 full-pool deployment. The protocol burns Steps 1-5 of compute and analyst time on arcs whose architectural failure mode is invisible until Step 6.

    Candidate amendments (historical, pre-v2.3):
    - Add a full-pool variant to §9 (require both admit-only AND full-pool sign-consistency to advance)
    - Add a Step 4 full-pool preview (admit/reject/early-exit decomposition with expectancy estimate at the classifier-locking gate)
    - Restructure §9 + §10 sequencing so the deployment-blocking metric is the primary stability gate
    - Pre-Step 6 admit/reject/early-exit expectancy summary as standardised Step 4 output

- **Surface arc.** Arc 4 rerun (2026-05-18) + Arc 5 closure (recent).
- **Resolution (v2.3, 2026-05-18).** Closed by structural removal of §9 in v2.3 §1. Step 5 (WFO, was Step 6) measures full-pool by construction. No replacement gate needed. See "Resolved in v2.3 amendment" section below.
- **Status.** CLOSED in v2.3.

### Open-23 — §8 Pipeline D1 cost-language correction (MEDIUM) — CLOSED in v2.3

- **Description.** §8 describes Pipeline D1 reject pool as "near-break-even small loss after spread, given the short hold and pre-t SL." Empirical evidence from Arc 4 + Arc 5:

    | Arc | Reject mean R | Reject % of signals | Early-exit mean R | Early-exit % |
    |---|---:|---:|---:|---:|
    | Arc 4 cluster 1 | −0.232 | 32.2% | −0.685 | 10.6% |
    | Arc 5 (per closure) | ~−0.46 | ~78% (Arc 5 specific) | — | — |

    §8 wording understated reject-pool cost. Empirically the reject pool costs ~−0.15 to −0.46R per trade depending on classifier discrimination strength, and the early-exit pool (pre-t SL hits before t=1) is a separate architectural cost at ~−0.45 to −0.69R on ~10-15% of signals.

- **Surface arc.** Arc 4 rerun + Arc 5 closure.
- **Resolution (v2.3, 2026-05-18).** Closed by documentation correction in v2.3 §4. §3 and §8 Pipeline D1 wording updated with empirical cost ranges; full-pool R = admit + reject + pre-t-loss contributions; evaluated at Step 5 WFO. See "Resolved in v2.3 amendment" section below.
- **Status.** CLOSED in v2.3.

### Open-24 — Pipeline D1 pre-t SL per archetype (MEDIUM) — CLOSED in protocol in v2.3; engine PR pending

- **Description.** Pre-t SL of 2×ATR (§8 Pipeline D1 default) fires on 10-15% of signals before the classifier evaluates at bar t. This is pure architectural cost — not classifier-induced, not exit-policy-induced. The classifier cannot filter these trades and the bail-out doesn't apply. Arc 4 cluster 1 saw 1,005 / 9,474 signals (10.6%) hit pre-t SL at −0.685R mean.

    Candidate structural responses (historical, pre-v2.3):
    - Widen uniform pre-t SL (reduces early-exit rate; increases per-event loss)
    - Shorten t (less time for SL to fire; smaller classifier feature set)
    - Hybrid: per-archetype pre-t SL calibrated to early-exit rate observed at Step 1

- **Surface arc.** Arc 4 rerun.
- **Resolution (v2.3, 2026-05-18).** Closed in protocol via v2.3 §5: pre-t SL = cluster's Step 3 selected SL multiplier (was uniform 2.0×ATR). Engine PR pending to expose per-archetype `pre_t_sl_atr_multiplier` field (default 2.0 for backward compatibility / anchor preservation). Per-archetype YAML schema extension; Step 3 archetype YAML emission writes `pre_t_sl_atr_multiplier = selected_sl_multiplier`. See "Resolved in v2.3 amendment" section below.
- **Status.** CLOSED in protocol; engine PR pending.

---

## Cross-cutting observations

### Two arcs (KH-24 v2.0, Arc 2 redo, Arc 3) all close FAIL at Step 3 §2 floors

Pattern is now systemic. Three arcs, three signals, three mechanisms — same gate failure. The §2-cluster of P0 items (P0.1, P0.2, P0.3, P0.4, P0.5) is collectively the dominant signal from v2.0's first three closed arcs.

If Arc 4 makes four-of-four, this graduates from "post-Arc-5 review" to "blocking the protocol from producing systems" and the calibration review should be expedited rather than waiting for Arc 5.

### Ordering of fixes

Within P0:
1. P0.1 (pre-peak metrics) — independent, cheapest, almost certainly the largest lever.
2. P0.2 (shape_tag bimodal) — independent of P0.1.
3. P0.3 (per-cluster §2 eval) — independent of P0.1, P0.2; reduces dependence on aggregation logic correctness.
4. P0.4 (SL/horizon) — measure pre-peak frac_wrong_way (P0.1) first; calibrate need on result.
5. P0.5 (mono floor) — almost certainly subsumed by P0.1.

Recommend executing P0.1 first (mechanical, cheap, almost certainly resolves much of the pattern), then re-measuring closed arcs to scope the rest.

### Anchor preservation rule survives all P0 fixes

P0.1, P0.2, P0.3, P0.5 are anchor-safe by construction (anchor values are at least as favourable under new metric definitions, anchor already passes the looser shape_tag taxonomy via D1, anchor is a single cluster not aggregated, anchor monotonicity comfortably clears any plausible loosened floor).

P0.4 is the exception: Path 2 (SL scaled to horizon) cannot retrofit to KH-24 without rebuilding the live system. Anchor preservation requires Path 2 to apply forward only, with KH-24 documented as a per-system exception.

---

## Resolved in v2.3 amendment (2026-05-18)

L_ARC_PROTOCOL v2.3 amendment landed 2026-05-18 (`L_ARC_PROTOCOL_v2_3_AMENDMENT.md`). Seven §0 changes; three items from this backlog close:

### CLOSED in v2.3 §1 — Open-22 full-pool gate at §9 (structural removal)

- **Surface:** Arc 4 RERUN (2026-05-18) + Arc 5 closure. Both arcs PASSED §9 admit-only stability and FAILED §10 full-pool deployment. §9 evaluated admit-set fold metrics; deployed-system economics for Pipeline D1 depend on full-pool (admit + reject + early-exit). The gate measured the wrong population.
- **Resolution.** v2.3 §1 removes §9 entirely. Pipeline is now 1-2-3-4-5 with Step 5 = WFO (renumbered from Step 6). Step 5 WFO runs the full backtester across all 7 OOS folds with real execution — every signal in the population goes through the engine, including reject-set early closures with their actual costs. Step 5 inherently measures full-pool. No separate Step 5 question that Step 6 doesn't answer more accurately.
- **Compute trade-off acknowledged.** Step 5 was cheap (uses already-trained classifier + admit set); Step 5 WFO is expensive (full WFO). "Filter bad archetypes before spending Step 6 compute" rationale doesn't justify the misleading-data problem documented in Arc 4 RERUN's deployment-fatal failure mode.
- **Anchor preservation.** KH-24 K=4 archetype 3 passes Step 5 WFO by deployment (worst-fold ROI +1.92%, worst-fold DD 6.37%, all 7 OOS folds positive). v2.2 Step 5 cross-fold stability (now removed) had been satisfied trivially by the same fold data. No interaction.

### CLOSED in v2.3 §4 — Open-23 Pipeline D1 cost-language correction (documentation)

- **Surface:** Arc 5 closure surfaced the gap; Arc 4 RERUN reinforced.
- **Resolution.** v2.3 §4 updates §3 and §8 Pipeline D1 wording. §3 Pipeline D1 description now reads: "Predict at bar t which trades to close vs continue. Rejected trades close at bar t with cost ~−0.15 to −0.46R; this is empirical, not a parameter." §8 Pipeline D1 row now references full-pool R = admit-weighted + reject-weighted + pre-t-loss contributions, evaluated at Step 5 WFO. Three-population structure documented (admit / rejected / pre-t losers) with empirical cost bounds from closed-arc evidence.
- **No threshold change.** Documentation correction only. Arc 5's KILL disposition was correct under v2.2; the gap was that an analyst reading §3 / §8 in v2.2 might infer rejected trades were cost-free.
- **Anchor preservation.** Anchor uses Pipeline D1 at t=3. Documentation correction does not change anchor evaluation.

### CLOSED in v2.3 §5 — Open-24 Pipeline D1 pre-t SL per archetype (protocol; engine PR pending)

- **Surface:** Arc 4 RERUN — uniform 2.0×ATR pre-t SL fired on 10.6% of signals at −0.685R mean, structurally separate cost from classifier-induced reject pool.
- **Resolution.** v2.3 §5 specifies pre-t SL = cluster's Step 3 selected SL multiplier (was uniform 2.0×ATR for all D1 archetypes). The SL distance selected at §7 SL sweep (per cluster, by capturability composite) is the SL used pre-t for that cluster's Pipeline D1 deployment.
- **Engine impact.** Pipeline D1 engine (post-PR2) supports per-archetype config for §11 exit policies. v2.3 spec requires the per-archetype D1 config to additionally express `pre_t_sl_atr_multiplier` (default 2.0 for backward compatibility with v2.2 behaviour and anchor preservation). Engine PR scope: add field to per-archetype D1 YAML schema; default 2.0; read at trade entry; apply to pre-t SL distance calculation; Step 3 archetype YAML emission writes `pre_t_sl_atr_multiplier = selected_sl_multiplier`. Can land independently of v2.3 protocol amendment doc.
- **Anchor preservation.** KH-24 K=4 archetype 3 Step 3 selected SL = 2.0×ATR (matches v2.0 anchor metrics fwd_mfe_p50 measured at 2.0×ATR frame). Pre-t SL under v2.3 = 2.0×ATR (cluster's Step 3 selected) = identical to v2.2's uniform 2.0×ATR. Open-24 spec change is a no-op for the anchor.

### Cross-cutting note on v2.2 §1 obsoletion

v2.2 §1 (sign-flip mechanisation) is OBSOLETED by v2.3 §1 — it mechanised the Step 5 gate 1 override; the gate no longer exists. v2.2 §2/§3/§4/§7 stand unchanged; v2.2 §5 (§16a) and §6 (halt point) updated for renumbering. Recorded for change-tracking, not a calibration loss.

---

## Resolved in v2.2 amendment (2026-05-18)

L_ARC_PROTOCOL v2.2 amendment landed 2026-05-18 (`L_ARC_PROTOCOL_v2_2_AMENDMENT.md`). Seven §0 changes; four items from this backlog (and cross-arc surface) close or partially close:

### CLOSED in v2.2 §3 — Step 4 max-F1 fallback

- **Surface:** Arc 6 closure (Open-21 original surface), Arc 7 closure (case study).
- **Resolution.** §8 threshold sweep must satisfy recall ≥ 0.60. No max-F1 fallback. If no threshold in {0.40, 0.50, 0.60, 0.70} satisfies recall ≥ 0.60, archetype dies at Step 4. Applies to Pipeline E, Pipeline D1, and Tier 2 lift candidate threshold sweeps.
- **Open-21 partial closure.** Proposal (a) "strict mode" is now mechanical. Alternates (b) recall floor 0.30 and (c) AUC floor 0.70 remain on the calibration backlog as potential further tightening; v2.2 §3 implements the minimum-mechanical fix.
- **Anchor preservation.** KH-24 K=4 archetype 3, Pipeline D1 at t=3, RF AUC 0.638 — cohort large enough that 60% admit is mechanical. No interaction.

### CLOSED in v2.2 §1 / §2 / §5 / §6 — Mid-arc analyst sign-off carve-outs

- **Surface:** every closed arc (KH-24 v2.0, Arc 2 redo, Arc 3, Arc 4, Arc 5, Arc 6, Arc 7) — chat-judgement carve-outs at §9 single-fold flip, §12 stack-freely, and ambiguous KILL/HALT disposition rules. Under serial execution these were cheap; under parallel CC execution they became serialisation bottlenecks.
- **Resolution.** v2.2 §1 mechanises §9 single-fold sign-flip (no chat override; mandatory diagnostic logging). v2.2 §2 caps §12 Tier 2 lift at ≤ 5 candidates intersection-only. v2.2 §5 adds new §16a KILL vs HALT mechanical disposition rule (single-criterion + cohort viability + near-miss/strong-magnitude). v2.2 §6 explicitly removes mid-arc analyst sign-off between arc-open and end of step 5 (§13 update).
- **Empirical basis.** Historical pattern: every carve-out was resolved the way the new rule resolves it. No archetype has tested more than 3 lift candidates historically (cap at 5 is bounded permission without practical constraint). Every archetype that died on §9 sign consistency also failed another gate.
- **Anchor preservation.** KH-24 K=4 archetype 3 — all 7 folds positive under v2.0 evaluation (§9 untriggered); no Tier 2 lift evaluated (§12 untriggered); passes all gates (§16a untriggered). No interaction.

### CLOSED in v2.2 §4 — Arc selection FIFO via state file

- **Surface:** Arc 6 (out-of-registry insertion via `discovery/lomega_regime_conditional`) demonstrated that implicit arc selection ("analyst picks") doesn't scale.
- **Resolution.** New §15b. CC consults `results/ARC_QUEUE.md` at arc start; picks topmost Unrun entry; transitions to Active with timestamp + branch name. Supports registry entries (`LCHAR_TOPN_REGISTRY.md` Entry K) AND standalone signal specs (`signal_spec_<name>_v<version>.md`). Git-level concurrency. Analyst override via direct edit.
- **Companion file landed.** `results/ARC_QUEUE.md` initialised 2026-05-18; empty Active and Unrun at landing; Closed section populated by housekeeping pass.

### ASSERTED in v2.2 §7 — Live-execution equivalence

- **Surface:** Arc 4 spread fix landing (2026-05-17) + PR2 landing (2026-05-17 PR #135). The engine now complies post-PR2 + spread fix; v2.2 makes the contract explicit.
- **Resolution.** New §1a. Steps 1 and 6 must execute under SPREAD_SEMANTICS_LOCK-equivalent semantics: entry timing (signal at bar t close → entry at bar t+1 open), spread costs (real per-bar MT5 bid/ask from execution bar; floor file is fallback only), intrabar SL/TS (triggers on mid, fills on bid/ask), D1 features (one-day lag), volume veto (no entry / no trade row / no spread). Step 6 additionally must apply §11 archetype-specific exit policy.
- **No behaviour change.** The assertion prevents silent re-introduction of pre-PR2 / pre-spread-fix divergence in future arc scripts.

---

## Arc 5 cross-arc items (2026-05-17, from `arc-5-closure` branch)

Eight items raised by Arc 5 closure (`mtf_alignment.2_down_mixed.kijun` h=120; SHELVED at Step 6 FAIL). Pipeline D1 + new spreads + full-pool WFO accounting surfaced rejected-pool adverse selection as the dominant cost — the protocol §9 admit-only framing missed the failure Step 6's full-pool reckoning caught. Numbering uses the `P-<id>` convention from the closure doc (distinct from Open-NN sequence).

### P0 — Protocol must surface full-pool economics before Step 6

- **P-§9-FRAMING (P0).** §9 sign-consistency and DD-ratio must be measured on full-pool strategy R (admits with their pipeline outcomes + rejects with their pipeline outcomes), not admit-only R. Protocol wording is currently ambiguous and reads as admit-only in practice. Under corrected framing, Arc 5 would have failed §9 at fold 4 (admit mean barely positive, full-pool mean negative). Step 6 would have been redundant.
- **P-D1-VIABILITY (P0).** Pipeline D1 viability check: signals with > X% bar-0/1 SL-hit rate at deployed baseline SL should be flagged for D1 unsuitability at Step 4. Arc 5 had 7.9%. Suggested threshold: 5%.
- **P-D1-REJECT-BIAS (P0).** Document rejected-pool selection bias: classifier rejection at bar t is itself a prediction signal correlated with continued adverse drift. Mean R of rejected pool ≠ unconditional bar-t baseline. Add to §3 Pipeline D1 description with calibration data from Arc 5 (−0.46R rejected vs +0.025R unconditional).

### P1 — Threshold selection

- **P-F9-RESELECT (P1).** F9 threshold selection should use the metric that gates ship decision (worst-fold compounded ROI subject to DD ceiling), not an intermediate measurement (admit-set precision/recall). Currently Step 4b selects on admit-only proxies that don't capture full-strategy cost.

### P2 — Housekeeping

- **P-CLUSTERING-LEAKAGE (P2).** Open-10 leakage status: c3 confirmed clean (Arc 5 Step 5.5 audit); c1 unresolved (audit invalid due to match formula saturation). Per-fold clustering should become default rather than full-pool, with full-pool retained as comparison only.
- **P-SPREAD-FLOOR-DOC (P2).** Spread floor file docstring drift — claims "applies only when raw spread is zero" but new p50 calibration applies to 58.6% of execution-bar entries. Update docstring + governance.
- **P-§11-MATCH-FORMULA (P2).** §11 row 2 pattern matching formula `min(1, (30-peaks)/25)` saturates at peaks ≤ 5, conflating Monotone-ascent (row 1, peaks ≤ 4) and Stepwise-climber (row 2, peaks 5-30) regions. Add a peaks lower-bound or a "preferred peaks range" specification for cleaner archetype matching.
- **P-OPEN-18-RECONCILE (P2).** STATUS.md / Open-18 priority queue had multiple inaccuracies discovered during Arc 5: `l_arc_4` Step 4/5 scaffolding existed undocumented; Arc 2 redo2 (v2.1.1 schema fork) existed undocumented; Open-18 KH-24 anchor replay scaffolding existed undocumented. Reconcile STATUS with actual repo state.

### Status

P0 items combine with Arc 4 RERUN's Open-22/23/24 (below) to form the cross-arc Pipeline D1 full-pool gating question. Owner: cross-arc calibration session. Queued for next protocol amendment cycle (likely v2.3 or v2.1.3 — pre-v2.2 numbering rules).

---

## Arc 7 cross-arc items (2026-05-17, from `phase/l_arc_7` branch)

Seven items raised by Arc 7 closure (liquidity sweep + reclaim long; CLEAN-NULL at Step 4). First capturable-not-extractable closure of record. Cross-arc items are documented in the closure doc and forwarded here for batch review.

### NEW

1. **Capturable-extractable gap as recognised closure category.** Arc 7 is the case study: PASS §7 with 3 V-shape units passing §2 conjunctively at composite > 0.37; FAIL §8 with 0/6 unit × pipeline AUCs clearing gate. This is not the same failure mode as a §2-fail at Step 3. v2.2 should consider an explicit closure pathway and commentary for arcs that PASS §7 but FAIL §8.

2. **SL-selection vs class-imbalance tension.** Composite-maximising SLs at §7 can compress the success distribution past extractability viability. Arc 7 c1: SL=4×ATR maximised §7 composite (0.617) but drove base success to 0.778, leaving only 41 negatives in n=185 for the classifier to learn from. Candidate v2.2 amendments: (a) §8 re-sweeps SLs and reports AUC × class-balance jointly; (b) §7 composite includes a class-balance regulariser; (c) leave §7 alone but flag the tension in §17.

3. **Open-04 external features escalation.** Arc 7 supplies concrete evidence that in-protocol features (Pipeline E + D1) can be insufficient even for capturable cohorts. Macro / session / cross-asset feature pipelines are now backed by empirical case for v2.2 commission.

### VALIDATED

4. **v2.1.2 `≠ scattered` floor.** First arc-of-record where the relaxed floor was load-bearing. All three Step 3 survivors carried `shape_tag = unclassified`. Under the prior floor (`∈ {tight_unimodal, heavy_right_tail, bimodal_separated}`), Arc 7 would have died at Step 3 with the wrong diagnosis. Closure proves the relaxation admits cohorts that subsequently get killed by other gates for the right reasons. Floor stays.

### UNRESOLVED

5. **§11 Stepwise pullback ≤ 0.5R ceiling.** Arc 7 c1 was the test case (mono 0.536, peaks 33.5, ttp_rel 0.73, pullback 0.567). Did not deploy. We don't know whether the §11 pullback ≤ 0.5R ceiling is over-strict in practice because no Arc 7 unit cleared §8. Question persists for future arcs.

### CLEANUP

6. **§15a text vs `_flatten_bar_path_for_trade` impl gap.** On `mfe_so_far_r` semantics: text says `close_r` running max; impl uses `high_r` intrabar. Arc 7 followed impl. Reconcile protocol text at next calibration review.

7. **Dispatch halt-criterion phrasing.** "Cross-arc bar-overlap" is a finding, not a halt. Future dispatches use overlap-vs-KH-24 as the live-system check; cross-arc overlap is a portfolio-composition note (Open-05).

### Status

Items 1-3 (NEW) are candidates for the v2.3 / v2.1.3 calibration cycle. Items 5 (UNRESOLVED) and 6, 7 (CLEANUP) carry to backlog. Item 4 (VALIDATED) is informational.

---

## Open-25 — v2.4 §1.5 entry-separability gate (proposed; Arc 8 closure)

**Status:** Proposed
**Priority:** High — third consecutive arc failure of the same structural pattern
**First proposed:** 2026-05-18 (Arc 8 closure)
**Evidence base:** Arcs 4 RERUN, 5, 8 — all PASS admit-only, FAIL full-pool ship gates
**Closure docs:** `docs/archive/arc_results/ARC_4_RERUN_RESULT.md`, `docs/archive/arc_results/ARC_5_RESULT.md`, `results/l_arc_8/ARC_8_CLOSURE.md`

### Problem

The v2.3 protocol stack optimises admit-only economics through Step 4 (extractability) and only evaluates full-pool deployment ROI at Step 5 (WFO). When path-shape clustering identifies a winning archetype but that archetype is not predictable from entry-time observables, the Step 4 classifier admits a large fraction of non-winning trades because they share entry-bar geometry with the winners. Full-pool deployment economics then dilute the archetype's edge to negative aggregate ROI.

Three consecutive arcs have failed this way:

| Arc | Admit-only Sharpe | Full-pool worst DD | Failure mechanism |
|---|---:|---:|---|
| 4 RERUN | (PASS) | (FAIL) | Reject pool 32% × −0.232R + early-exit pool 11% × −0.685R |
| 5 | (PASS) | (FAIL) | Rejected pool 78% × −0.46R |
| 8 | 1.14–1.44 | 15.6%–19.0% | Admit rate 70–89% × non-c1 mean ≈ −0.3R |

This is structural, not arc-specific. Per-arc patches (better classifiers, different SLs, alternative exit policies) cannot fix it — the discriminating information genuinely isn't available at entry. Confirmed empirically by Arc 8's post-Step-5 entry-feature-overlap diagnostic (commit `7d9109e`): all 18 entry features have mean overlap coefficient > 0.78 between c1 and non-c1; multiclass RF on cluster IDs at entry achieves c1 1-vs-rest AUC 0.547 (≈ coin-flip) and precision@recall=0.60 = 0.149 (zero lift over base rate 0.133).

### Proposal

Add a §1.5 gate to the protocol, executed **before** Step 1 simulation compute is committed.

**Gate procedure:**
1. Run a Step-1-spec smoke pool (n ≥ 200 trades; subset of pairs OK)
2. Apply Step 2 clustering (same K-selection logic as full Step 2)
3. Train a multiclass RandomForestClassifier on entry-time features (Step 4 Angle E feature set) with target = cluster ID
4. 5-fold TimeSeriesSplit; report one-vs-rest AUC and precision@recall=0.60 per cluster
5. Identify the winning-archetype candidate cluster(s) by mean final_r within cluster

**Gate threshold:**
- Winning-cluster one-vs-rest precision@recall=0.60 ≥ 0.30 (entry features only) → PASS, proceed to full Step 1
- < 0.30 → HALT, arc closed pre-Step-1 with structural-separability finding logged

**Rationale for 0.30 threshold:**
- Arc 8 c1 entry-time precision@recall=0.60 = 0.149 → would have caught Arc 8 at §1.5 (saved Steps 1–5 + diagnostics compute)
- 0.30 is below the §10 deployment-viable threshold of 0.40 — preserves arcs that are marginal at entry but might be rescuable by post-entry confirmation (rare path, but kept open)
- Threshold should be validated against Arcs 4 and 5 historical data as a regression test before v2.4 release

### Cost-benefit

**Compute savings per blocked arc:** ~10–30 hours of CC time across Step 1 sim + clustering + capturability + extractability + WFO + diagnostics. Three arcs over the last cycle would have been blocked here.

**Implementation cost:** ~1 dispatch's worth of CC time for the §1.5 script + ~30 min smoke-pool compute per arc.

**Risk of over-blocking:** Low. The 0.30 threshold is below deployment-viable, so arcs that could rescue with post-entry confirmation aren't auto-killed. Need regression test against Arcs 4 / 5 to verify they would have failed §1.5 (likely yes based on their failure patterns).

### Open design questions

1. **Smoke-pool sample size and pair selection** — does 200 trades from 5 pairs reliably predict full-pool separability? Probably, but worth a quick sensitivity analysis.
2. **Cluster K selection at §1.5** — full Step 2 uses silhouette-driven K selection. §1.5 needs to mirror this; risk of K=2 vs K=4 changing the gate's verdict.
3. **What about post-entry-viable archetypes** that fail §1.5 by entry-feature stats but pass at t=5–10? Current proposal would block these. Could add a second-pass at t=5 multiclass — but doubles compute and we don't yet know if any signal in the framework benefits.
4. **Regression test corpus** — Arcs 4 RERUN, 5, 8 should fail §1.5. What about Arc 2 (passed full pipeline)? It should pass §1.5 — verify before v2.4 release.

### Acceptance criteria for v2.4 release

- §1.5 script implemented and tested deterministically
- Regression test against Arcs 4 RERUN, 5, 8 confirms all three fail the gate
- Regression test against Arc 2 (or other historically successful arc) confirms it passes the gate
- Protocol doc updated with §1.5 spec, threshold, halt-on-fail logic
- Existing arc workflow updated to call §1.5 before Step 1

### Related items

- See `results/l_arc_8/ARC_8_CLOSURE.md` for full Arc 8 context
- See `results/l_arc_8/diagnostics/entry_feature_overlap/DIAGNOSTIC_SUMMARY.md` for the multiclass-at-entry analysis that informs the threshold choice
- See `results/l_arc_8/diagnostics/COMBINED_DIAGNOSTIC_SUMMARY.md` for the Path 1 / Path 2 follow-up diagnostics confirming the failure is structural (Path 1 marginal at t=12 with AUC 0.755 but precision@recall=0.60 = 0.274; Path 2 dead — no mechanical filter satisfies c1_ret ≥ 0.80 ∧ c2_ret ≤ 0.30 ∧ pool ≥ 500)
- Builds on Open-22 / Open-23 / Open-24 (closed in v2.3 protocol-level; Open-25 is the cross-arc systemic-pattern response that those individual closures did not address)

---

## Arc 10 cross-arc items (2026-05-18, from `claude/charming-mcnulty-8160e0`)

Arc 10 (DLR — D1 swing-low rejection long, out-of-registry `signal_spec_d1_swing_low_rejection_long_v0.1.md`) closed `STEP_4_HALT` per §16a Path A. First arc to run end-to-end under v2.3. Steps 1-3 PASS clean; Step 4 disjunctive §8 fails near-miss (c1 V-shape recovery E AUC 0.6296 margin −0.0204; D1 AUC 0.5897 margin −0.0103). Post-closure experimentation (EXP-01–06) + WFO pair conducted over §16a at chat-side direction. Items below derive from the closure + experimentation + WFO. Closure doc: `docs/archive/arc_results/ARC_10_RESULT.md`.

### Status changes to existing items

#### Open-06 — AUC threshold recalibration — WEAKENED

Originally argued for relaxation based on three V-shape near-miss data points (Arc 6, Arc 7, Arc 10). Per EXP-05, **Arc 6 reclassified as Stepwise climber, not V-shape** (its own closure doc was Stepwise; the V-shape mislabel propagated through prior cross-arc synthesis). Two clean V-shape data points remain on record (Arc 7, Arc 10). Threshold relaxation that admits the Arc 6 + 10 pair mixes archetypes; a clean V-shape pair (Arc 7 + Arc 10) requires deeper relaxation (binding at E ≥ 0.536 via Arc 7 agg_c1_c3, not E ≥ 0.600 via Arc 6 + Arc 10).

**Status:** WEAKENED. Reassess in v2.4. Cross-arc V-shape pooling (EXP-05 pool AUC 0.6348 at gap −0.015) may make threshold relaxation unnecessary if a pooled-cohort classifier clears 0.65 at scale; defer Open-06 decision until cross-arc clusterifier build (next-dispatch rank #2) lands.

#### Open-04 — external-feature commission — DEFERRED

EXP-06 on Arc 10 probed two candidate features (D1 Kijun distance, session dummies) — both produced negative AUC delta on Arc 10 alone (small-n saturation: 25 baseline features + n=228 → adding features adds noise faster than signal). Single-arc evidence inconclusive.

**Status:** DEFERRED. Needs multi-arc reproduction on Arc 8 / 9 / 11 (when their step1+step4 outputs land) before any commission proposal. Re-run EXP-06 as a multi-arc probe in the v2.4 cycle.

### New backlog items

#### §16a Path A — disjunctive-gate ambiguity (MEDIUM)

"Single criterion fail with margin < 0.03 → HALT" wording is unclear when a disjunctive Step 4 gate (E OR D1) fails on both criteria simultaneously. Two readings:
- **Strict:** 2 individual AUC thresholds fail → 2 criteria → KILL.
- **Compound:** the §8 disjunctive extractability gate fails as a single gate → HALT.

Arc 10 invoked the compound reading because (a) AUC is explicitly named in §16a Path A numeric-criterion list, (b) both margins < 0.03 absolute, (c) all earlier-step gates pass cleanly. Compound-vs-strict ambiguity flagged in Arc 10 closure with note that chat may overrule to KILL at v2.4 cycle.

**Action for v2.4:** formalise reading in §16a. Audit Arc 6 / Arc 7 closures for consistency (both also failed disjunctive Step 4; check what reading they invoked).

#### Reverse FE methodology framework (LOW)

Activity class currently defined only in Arc 10 closure (§"Why we can't filter to c1") and post-closure dispatch (`ARC_10_REVERSE_FE_DISPATCH.md`). Lift to framework methodology doc if one exists; otherwise create one. Includes:
- Anti-snooping protocol: pre-registered hypothesis catalog, hashed before validation
- Hold-out conventions at small N (Arc 10 c1 n=228 is the working reference)
- Routing rules between classifier path (extend feature envelope, re-test through Step 4) vs filter path (deterministic conditions, validate on post-filter trade-set P&L)

**Action for v2.4:** decide on framework-level vs ad-hoc treatment. If framework-level, draft methodology doc + integrate with L_ARC_PROTOCOL §§7-8 wording.

#### Within-cluster heterogeneity probing (LOW)

Step 3 capturability composite of 0.4934 on Arc 10 c1 (V-shape recovery, n=228) hints at heterogeneity. Deep-V vs shallow-V sub-clustering may produce a cleaner-signature sub-type. Methodology candidate:
- Step 2 extension: optional second-pass K-means within capturable cluster
- Reverse FE Stage 1: qualitative characterisation surfaces sub-types before classifier work

**Action:** explore in next reverse-FE dispatch on Arc 10 c1 (rank #1).

#### Q2-2022 misattribution audit (LOW)

Arc 10's fold-2 was originally narrated as "Q2 2022 USD strength regime" in the original closure. EXP-04 confirmed the actual fold-2 window is **2023-07-13 → 2024-06-05**. Possible same misattribution elsewhere — closure docs across all arcs should be reviewed for "Q2 2022" references and dates verified against actual fold definitions.

**Action:** repo-wide audit task. Cheap; combine with next housekeeping pass.

#### Filter-path validation regime (MEDIUM)

No precedent in current L_ARC_PROTOCOL for validating a deterministic filter against post-filter trade-set P&L rather than classifier AUC. Filter path is the rank #3 dispatch from Arc 10 closure recommendations and needs an acceptance regime defined before it dispatches downstream.

Candidates for acceptance criteria (filter-path dispatch should pre-register):
- Worst-fold Sharpe ≥ X (calibrate vs Arc 10 oracle Sharpe 4.61 — realistic filter likely much lower)
- Worst-fold expectancy ≥ X R/trade
- Worst-fold max DD ≤ X%
- Trade count per fold ≥ X (sanity floor; below it Sharpe is noise)

**Action for v2.4:** define acceptance criteria before any filter-path probe dispatches. Cross-reference with §10 pass-deployable / pass-viable thresholds — filter path may need its own tier.

### Promoted candidates from Arc 10 research

#### `L1_minus_L0_atr` cross-arc feature promotion (LOW)

EXP-02 single-feature LOO drop is +0.028 on Arc 10 c1 — 116% of the total HTF feature-class lift (+0.024 over generic baseline). Removing this one feature alone drops AUC by more than removing all HTF features combined (because the remaining HTF features partially compensate for each other; `L1_minus_L0_atr` is structurally unique). This is the D1 HL slope magnitude in ATR units — the structural condition that distinguishes "shallow uptrend HL pullback" from "steep uptrend HL pullback" at the swing-low rejection moment.

**Action:** promote to standard cross-arc feature catalog (analogue to KH-24's `d1_close_in_range`) pending reproduction on Arc 8 (PR-HHHL — has HL structure), Arc 9 (IB-trend — different structural anchor), Arc 11 (SHB — swing-high anchor; mirror form may apply). If 2+ arcs surface analogous structural-magnitude features, formalise feature class.

#### V-shape archetype as cross-arc abstraction (MEDIUM)

EXP-05 cross-arc V-shape pool (Arc 7 c3 + Arc 10 c1) at common SL=3.0×ATR with generic 17-feature subset produces pooled AUC 0.6348 (gap −0.015 to gate). +0.029 over Arc 10 alone; +0.140 over Arc 7 c3 alone at re-imposed SL. WFO oracle on Arc 10 c1 confirms real OOS edge if cluster ID known at entry (Sharpe annualised 4.61). Combined evidence: V-shape recovery is a cross-arc deployable abstraction, but realisable classifier ceiling is at-the-gate, not above (EXP-01 P(realisable AUC ≥ 0.65) = 12.5%).

**Action:** build deployable abstraction pending reverse-FE-fed classifier (rank #2 next dispatch from Arc 10 closure). Arc 6 is BLOCKED for V-shape pool (Stepwise, not V-shape, per EXP-05 reclass).

### Status

Items above accumulate as inputs for the v2.4 cross-arc calibration cycle. Open-06 weakened, Open-04 deferred; five new items, two promoted candidates. Anti-pattern observation: dispatches that propagated the "three V-shape cohorts" framing did so based on stale Arc 6 archetype labelling — same-day audit before cross-arc synthesis would have caught this. Add to v2.4 cycle hygiene checklist.

---

## Document control

| Field | Value |
|---|---|
| Created | 2026-05-16 |
| Source | review of three closed arc results + chat-raised issues |
| Status | record-only; no protocol edits applied |
| Review trigger | post-Arc-5 cross-arc calibration review (or earlier if Arc 4 closes on §2 floors) |
| Anchor rule | §14 KH-24 K=4 archetype 3 preservation checked per item |
| v2.1 amendment date | 2026-05-17 — summary status block above tracks per-item resolution |
| v2.1.1 amendment date | 2026-05-17 — combined refinements + engine-reality corrections; P0.1 refined to composite selection, P1.7 unblocked from engine PR (re-run only) |
| Arc 6 cross-arc items added | 2026-05-17 — Open-21 (Step 4 deployability gate, new), Open-17 expansion (Tiebreak 1 noise floor), unnumbered reach_1R noise tolerance note |
| Arc 4 RERUN + Arc 5 items added | 2026-05-18 — Open-22/23/24 (Pipeline D1 full-pool gating); Arc 5 P-series (P0: §9-FRAMING, D1-VIABILITY, D1-REJECT-BIAS; P1: F9-RESELECT; P2: CLUSTERING-LEAKAGE, SPREAD-FLOOR-DOC, §11-MATCH-FORMULA, OPEN-18-RECONCILE) |
| Arc 7 cross-arc items added | 2026-05-18 (housekeeping pass) — 7 items (3 NEW, 1 VALIDATED, 1 UNRESOLVED, 2 CLEANUP) from `phase/l_arc_7` closure doc |
| v2.2 amendment date | 2026-05-18 — `L_ARC_PROTOCOL_v2_2_AMENDMENT.md`. Closed in v2.2: §8 max-F1 fallback (v2.2 §3, closing Arc 6/7 case), mid-arc analyst sign-off carve-outs (v2.2 §1/§2/§5/§6), FIFO arc selection (v2.2 §4 new §15b), live-execution equivalence asserted (v2.2 §7 new §1a). Open-21 partial: proposal (a) strict-mode closed; (b)/(c) on backlog. Open-22/23/24 (Pipeline D1 full-pool gating) NOT closed by v2.2 — addressed in v2.3 (row below). |
| v2.3 amendment date | 2026-05-18 — `L_ARC_PROTOCOL_v2_3_AMENDMENT.md`. Closed in v2.3: Open-22 (v2.3 §1 structural removal of §9); Open-23 (v2.3 §4 §3/§8 cost-language correction); Open-24 (v2.3 §5 per-archetype pre-t SL spec; engine PR pending for `pre_t_sl_atr_multiplier`). Step 5 cross-fold stability removed; Step 6 WFO renumbered as Step 5; orchestrator halt point shifted end of Step 5 → end of Step 4; v2.2 §1 sign-flip mechanisation OBSOLETED (gate no longer exists); §16a position-5 semantic shifted to WFO; §1a Step 1 + Step 5 (was Step 6). New informal register at `SHELVED_ARCS.md`. Anchor preservation verified (KH-24 K=4 archetype 3 passes Step 5 WFO by deployment; Step 3 selected SL = 2.0×ATR matches v2.2 uniform pre-t SL — Open-24 no-op for anchor). Companion file: `prompts/cc_arc_orchestrator_template.md` updated to v1.1. |
| Arc 8 cross-arc items added | 2026-05-19 (housekeeping pass) — Open-25 (v2.4 §1.5 entry-separability gate, HIGH PRIORITY) from Arc 8 closure 2026-05-18. Arc 8 (PR-HHHL long) HALT_DEPLOYMENT — Steps 1-4 PASS, Step 5 WFO FAIL §10 ship gates; 3rd consecutive Open-22/23/24 admit-only-vs-deployment failure (Arcs 4 RERUN, 5, 8). Open-25 is the cross-arc systemic-pattern response that Open-22/23/24's individual v2.3 closures did not address. Closure doc: `results/l_arc_8/ARC_8_CLOSURE.md`. |
| Engine PR #146 (Open-24) | 2026-05-19 — `feat/open-24-pre-t-sl-per-archetype` merged into main. Implements per-archetype `pre_t_sl_atr_multiplier` consumption. Closes Open-24 engine-side. Consumed by Arc 8 c1 D1 policy YAML (pre_t_sl_atr_multiplier=4.0) before Arc 8 closure. |
| Arc 10 cross-arc items added | 2026-05-19 (meta-doc alignment pass) — Arc 10 closure (STEP_4_HALT) + post-closure experimentation (EXP-01–06) + WFO pair (base + oracle c1). Status changes: Open-06 WEAKENED (Arc 6 reclassified Stepwise per EXP-05; clean V-shape pair Arc 7+10 only); Open-04 DEFERRED (EXP-06 single-arc probe inconclusive; needs Arc 8/9/11). New items: §16a Path A disjunctive-gate ambiguity (MEDIUM); reverse FE methodology framework (LOW); within-cluster heterogeneity probing (LOW); Q2-2022 misattribution audit (LOW); filter-path validation regime (MEDIUM). Promoted candidates: `L1_minus_L0_atr` cross-arc feature (LOW); V-shape archetype cross-arc abstraction (MEDIUM). |

---

## Arc 11 cross-arc items (2026-05-18; housekeeping landed 2026-05-19)

Source: `results/l_arc_11/ARC_11_CLOSURE.md`. Arc 11 closed CLOSED-HALT at Step 4 (§16a Path A near-miss, best AUC 0.5728, margin 0.027). 4 post-closure experimental sessions surfaced amendment candidates, pattern documentation, and strike list below. All entries doc-only — no protocol mutation by Arc 11 housekeeping.

### Amendment candidates (priority: live for v2.4)

**[Arc 11] Pipeline DE (Deferred-Entry)** — NEW amendment proposal

- Architecture: enter at bar `t` or not at all, no in-trade classifier
- Gate: `AUC ≥ 0.60` + sign-consistency + pre-t SL filter rate `< 40%`
- Default `t`: per-archetype, sweep `[1, 16]`
- Pairs with: `min_observation_bars` registry parameter (below)
- Distinct from Pipeline D1 (post-entry in-trade decision) and Pipeline E (entry-bar decision)
- Source: Arc 11 filter_diag (Exp 3) Regime B + sig_improve (Exp 4) Stage 2. Only direction across four feature regimes that moves Pipeline E AUC above the §8 0.60 line on this signal class.

**[Arc 11] `min_observation_bars` archetype-registry parameter** — NEW amendment proposal

- Required per-archetype parameter, gates §16a HALT decision
- Protocol: test Pipeline DE variants at `min_observation_bars` before declaring near-miss HALT on Step 4 disjunctive AUC gate
- Implication: cleaner failure attribution for capturable-not-extractable cohorts (DD/trade-count grounds, not AUC margin grounds)
- Source: Arc 11 closure synthesis. Implication for Arc 11 specifically: would have continued to DE evaluation, produced the same +3.11%/+17.23%/18% DD result, then HALT'd on DD/trade-count grounds instead of AUC-margin near-miss

### Pattern documentation (priority: cross-arc synthesis)

**[Arc 11] Capturable-not-extractable pattern (Arc 6 + Arc 11 confirmation)** — UPGRADE from speculative to confirmed

- Two clean instances of same shape: cohort carries real structural edge (Arc 11 c1: `fwd_mfe_p50 = 4.48R`, `reach_1R = 100%`; Arc 6 c2: `mfe_p50 = 4.47R`, `ww_pp = 0.000`); entry-time features cannot resolve which trades realise it
- Calibration: Pipeline E AUC ceiling ~0.55 on 4H entry-bar features across three independent feature regimes (baseline, multi-TF including D1+1H, reframed target reach_1R / mfe≥2R). Likely structural to the feature/timeframe combination, not the classifier choice.
- Discriminating information lives in post-signal price action (timing > features as extractability lever) — multi-TF and feature redesign do not help
- Cross-reference target: any future arc with `mfe_p50 ≥ 3R` + `reach_1R ≥ 80%` + Pipeline E AUC < 0.55 should auto-flag this pattern and route to Pipeline DE evaluation before §16a HALT consideration

### Strike list (priority: prevent re-proposal)

**[Arc 11] Empirically retired directions for SHB long 4H**

The following directions have been tested and disconfirmed for this signal/timeframe combination. Do not re-propose without new evidence:

1. Pipeline E on entry-bar features (AUC ceiling 0.52)
2. Pipeline D post-entry on c1 cohort (AUC 0.40–0.45 — worse than random)
3. Multi-TF feature extension D1+1H (+0.024 AUC, dead)
4. Reframed supervision target reach_1R or mfe ≥ 2R (worse than cluster ID)
5. Trigger-bar mechanical filters (0/27 single rules, 0/3 pairs pass c1_ret ≥ 0.80 AND c2_ret ≤ 0.30)
6. Sizing without filtering (full pool negative EV at every tier: 0.25% / 0.50% / 1.00% → worst-fold −17% / −32% / −56%)
7. "Relax AUC gate when mfe_p50 ≥ 3R" (disconfirmed by no-oracle test; even with c1 mfe_p50 4.48R the no-oracle live system fails on every gate)
8. DD-relaxation amendment for capturable-not-extractable cohorts (best combo DD/ROI ratio 1.04 too poor to justify; risk-scaling does not rescue)

Note: directions 1, 3, 4 may still apply to other signals/timeframes; this strike list is signal-specific to SHB long 4H.

### Files

- Closure doc: `results/l_arc_11/ARC_11_CLOSURE.md`
- Live doc: `results/l_arc_11/ARC_11_LIVE.md`
- Experimental outputs: `results/l_arc_11/experimental_s5/`, `results/l_arc_11/filter_diag/`, `results/l_arc_11/sig_improve/`
