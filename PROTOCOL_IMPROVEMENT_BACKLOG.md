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
| Still open | 4 | P1.7 (refresh execution — pending KH-24 v2.0 re-run only under v2.1.1), P1.10, P1.12, P2.14 |

Last updated: 2026-05-17 alongside L_ARC_PROTOCOL v2.1.1 amendment (combined refinements + engine-reality corrections).

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

## Arc 5 cross-arc items (2026-05-17)

Eight items raised by Arc 5 closure (`mtf_alignment.2_down_mixed.kijun` at h=120, registry Entry 5; SHELVED at Step 6 FAIL under PR 2 + new spreads + full-pool WFO accounting). The three P0 items below take priority over further arc dispatches — they're protocol-level corrections, not arc-specific.

### P-§9-FRAMING — P0

- **Source:** Arc 5 Step 5 vs Step 6 reconciliation 2026-05-17.
- **Problem.** §9 sign-consistency and DD-ratio gates must be measured on full-pool strategy R (admits with their pipeline outcomes + rejects with their pipeline outcomes), not admit-only R. Protocol §9 wording is currently ambiguous and reads as admit-only in practice. Under admit-only Arc 5 passed §9 cleanly (7/7 positive admit mean R, both clusters). Under full-pool framing Arc 5 would have failed at fold 4 (admit mean barely positive, full-pool mean negative). Same bug applies to every Pipeline D1 arc with high rejection rate and adverse-selected rejection (i.e., every competent classifier — by construction, competent rejection correlates with bad outcomes).
- **Empirical evidence (Arc 5).** Cluster 1 admit-set §9 PASS (sign 7/7, DD ratio 1.17 under PR 2). Full-pool §9 evaluated retroactively would have flagged fold 4 (mean R −0.094, ROI −34.49% compounded). Step 6 caught the negative expectancy that admit-only §9 missed.
- **Action.** Explicit protocol clarification in §9 — sign-consistency, size variance (already full-pool), and DD-ratio all measured on the full-pool equity curve including rejected-trade pipeline outcomes. Update Step 5 result-doc template to require full-pool metrics. Re-evaluate any in-flight or recently-closed D1 arcs under corrected framing.
- **Anchor preservation.** KH-24 K=4 archetype 3 passes Pipeline D1 at t=3 with exclusion 15.4% (vs Arc 5's ~80% rejection rate). Lower rejection rate means smaller rejected-pool contribution to full-pool R; anchor preservation depends on actual KH-24 rejected-pool mean R, not currently measured. Verification required before commit.
- **Status.** Highest-priority cross-arc item. Likely v2.1.3 amendment.

### P-D1-VIABILITY — P0

- **Source:** Arc 5 Step 6 economic decomposition 2026-05-17.
- **Problem.** Pipeline D1 enters every signal and pays the SL cost on any trade that hits SL before bar t=1 (where the classifier evaluates). Signals with high bar-0/1 SL-hit rate at the deployed baseline SL accumulate unavoidable per-trade losses that the classifier cannot mitigate. Arc 5 had 7.9% bar-0/1 hits (976/12,348), some from spread-blowout fills on news days (Jan 2015 SNB unpeg drove individual losses to −6.87R from 440-pip spreads). Mean early-exit R ≈ −1.1 in 2×ATR R-frame.
- **Proposal.** Add a Step 4 gate that measures bar-0/1 SL-hit rate on the OOS pool under the candidate's deployed-baseline SL. If > 5%, flag D1 as unsuitable (cluster routes to E only, or arc dies if E also fails). Threshold (5%) is a starting calibration point informed by Arc 5; cross-arc calibration may refine.
- **Anchor preservation.** KH-24 K=4 archetype 3 bar-0/1 SL-hit rate not currently measured; needs measurement before commit. Verification required.
- **Status.** P0. Pairs with P-§9-FRAMING (both are D1-framework corrections). Likely v2.1.3 amendment.

### P-D1-REJECT-BIAS — P0

- **Source:** Arc 5 Step 6 tier decomposition 2026-05-17.
- **Problem.** Pipeline D1 description (§3) does not document that classifier rejection at bar t is itself a prediction signal correlated with continued adverse drift. A competent classifier rejects when path-so-far at t=1 looks adverse (negative MAE, low MFE, monotone-down close_r); that adversity correlates strongly with bar-2 drift. Rejected-pool close-at-market cost is therefore materially worse than unconditional bar-2 R baseline.
- **Empirical calibration (Arc 5).** Rejected pool (~78% of trades): mean R **−0.46** (denominated in c3 R-frame). Unconditional bar-2 R baseline (all trades, conditional only on bars_held ≥ 3): mean R **+0.025**. Order-of-magnitude difference. Tier C in Arc 5 ensemble (both classifiers reject): mean R −0.46 (1,789 trades). Selection bias is structural, not an Arc 5 idiosyncrasy.
- **Action.** Add explicit framing in §3 Pipeline D1 description: "Classifier rejection at bar t is itself a prediction signal; rejected pool's pipeline-outcome R is expected to be materially negative under a competent classifier. Strategy ROI accounting must include rejected-pool pipeline outcomes." Include Arc 5 calibration data (−0.46 vs +0.025) as reference.
- **Anchor preservation.** Doc-only change. Anchor unaffected.
- **Status.** P0. Pairs with P-§9-FRAMING. Likely v2.1.3 amendment.

### P-F9-RESELECT — P1

- **Source:** Arc 5 Step 4b threshold selection 2026-05-17.
- **Problem.** F9 threshold selection at Step 4b currently uses admit-set precision/recall proxies (max-precision subject to recall ≥ 0.60). These don't capture full-strategy cost (rejected-pool drag + early-exit losses). A threshold that maximises admit precision may not minimise the full-strategy negative R from rejection drag. Under-spec'd in §8 for D1 specifically (E grid is explicit; D1 grid was extended in Arc 5 within-arc as F9 decision).
- **Proposal.** Revise §8 threshold-selection rule to operate on full-strategy WFO R (worst-fold compounded ROI subject to DD ceiling), not admit-only precision/recall. Requires Step 4b to know about Step 6 mechanics — explicit dependency to document.
- **Anchor preservation.** KH-24's D1 threshold at t=3 was selected under v2.0 mechanics; verification under new rule required before commit.
- **Status.** P1. Lower priority than the three P0s because it's a downstream refinement once full-pool framing lands.

### P-CLUSTERING-LEAKAGE — P1

- **Source:** Arc 5 Step 5.5 audit 2026-05-17, plus Open-10 carryover.
- **Problem.** Open-10 leakage status: Arc 5 Step 5.5 confirmed c3 leak-free (Jaccard 0.93 — full-pool clustering identity reproduces under per-fold clustering); c1 audit invalid due to §11 row 2 match formula saturation (see P-§11-MATCH-FORMULA below). Per-fold clustering should become default rather than full-pool, with full-pool retained as comparison only.
- **Proposal.** Revise §6 Step 2 spec to make per-fold clustering canonical for WFO-bound arcs. Full-pool clustering retained as a parallel diagnostic. Step 5.5 audit framework (Jaccard vs full-pool) becomes a standard Step 2 deliverable.
- **Anchor preservation.** KH-24 K=4 anchor clustering under per-fold methodology requires re-measurement. Verification required.
- **Status.** P1. Closes Open-10 with empirical methodology rather than just resolving a single instance.

### P-§11-MATCH-FORMULA — P2

- **Source:** Arc 5 Step 5.5 c1 audit invalidation 2026-05-17.
- **Problem.** §11 row 2 (Stepwise climber) pattern matching formula `min(1, (30 − peaks) / 25)` saturates at peaks ≤ 5, returning the same match score (1.0) for any centroid with peaks 0-5. This conflates row 1 (Monotone-ascent, peaks ≤ 4) and row 2 (Stepwise-climber, peaks 5-30) match regions. Arc 5 Step 5.5 audit-c1 was systematically misrouted to baseline c0 (peaks 4.73, near-Monotone-ascent) instead of baseline c1 (peaks 20.07, true Stepwise) specifically because of this saturation. Jaccard 0.0 across all folds — different cluster, not different version of same cluster.
- **Proposal.** Add a peaks lower-bound to row 2's match criteria (e.g., peaks ≥ 5 required, not just peaks ≤ 30/50), or specify a preferred-peaks-range that informs match-score weighting. Same review may apply to other §11 rows with one-sided ceiling/floor.
- **Anchor preservation.** KH-24 K=4 archetype 3 has peaks 14.19 — well within preferred Stepwise range — anchor preservation holds under any reasonable peaks lower-bound.
- **Status.** P2. Cleanup item; doesn't change protocol behaviour for typical centroids, only resolves the saturation edge case.

### P-SPREAD-FLOOR-DOC — P2

- **Source:** Arc 5 Phase 1 (Step 1 re-run under new spreads) 2026-05-17.
- **Problem.** `configs/spread_floors_5ers.yaml` docstring claims "applies only when raw spread is zero" — accurate for the old uniform 0.1-pip floor (mechanically equivalent to raw native > 0). False for the new per-pair active-session p50 calibration: the loader (`core/spread_floor.py`, `_spread_pips_at_bar`) applies `max(raw_pips, floor)` — so under the new floors, much larger fractions of execution bars are floored upward. Arc 5 measurement: 58.6% of matched-trade entry-bar spreads strictly raised by the new floor.
- **Proposal.** Revise spread file docstring to reflect post-calibration semantics: "Loader applies max(raw_pips, floor). Under per-pair active-session p50 calibration, floor activates on a material fraction of execution bars (~50-60% in Arc 5 measurement)."
- **Anchor preservation.** Doc-only change. KH-24 doesn't load this file (raw MT5 per-bar spread used directly).
- **Status.** P2. Cleanup item; doesn't change behaviour, only documentation accuracy.

### P-OPEN-18-RECONCILE — P2

- **Source:** Arc 5 multi-step provenance discovery 2026-05-17.
- **Problem.** STATUS.md / Open-18 priority queue had multiple inaccuracies discovered during Arc 5 work: `l_arc_4` Step 4/5/6 scaffolding existed undocumented; Arc 2 redo2 (v2.1.1 schema fork with `is_held` column + real-market forward R-fields) existed undocumented; Open-18 KH-24 anchor replay scaffolding existed undocumented. Arc 5 had to rediscover these via repo archaeology.
- **Proposal.** Walk through repo and reconcile STATUS.md (Recent Closures, Active Research, Open Items) with actual state. Add a STATUS section for "in-flight scaffolding" that captures work-in-progress trees not yet promoted to a named arc.
- **Anchor preservation.** Doc-only.
- **Status.** P2. Process-improvement item; reduces friction for future arcs.

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
| Arc 5 cross-arc items added | 2026-05-17 — 8 items: P-§9-FRAMING (P0), P-D1-VIABILITY (P0), P-D1-REJECT-BIAS (P0), P-F9-RESELECT (P1), P-CLUSTERING-LEAKAGE (P1), P-§11-MATCH-FORMULA (P2), P-SPREAD-FLOOR-DOC (P2), P-OPEN-18-RECONCILE (P2). Three P0s take priority over further arc dispatches — Pipeline D1 framework requires correction before next D1-based arc runs. |
