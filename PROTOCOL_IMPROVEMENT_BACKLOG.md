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
| Still open | 4 | P1.7 (refresh execution — pending KH-24 v2.0 re-run only under v2.1.1), P1.10, P1.12, P2.14 |
| Partial in v2.2 | 1 | Open-21 (Step 4 deployability gate) — proposal (a) strict-mode max-F1 fallback closed by v2.2 §3; alternates (b) recall floor 0.30 + (c) AUC floor 0.70 remain on calibration backlog |

Last updated: 2026-05-18 alongside L_ARC_PROTOCOL v2.3 amendment (Step 5 cross-fold stability removed; Step 6 WFO renumbered as Step 5; Open-22/23/24 closed in protocol with engine PR pending for Open-24). v2.2 amendment landed earlier same day.

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
