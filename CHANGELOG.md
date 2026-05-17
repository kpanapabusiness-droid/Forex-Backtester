# Changelog

## ENGINE GENERALISATION | 2026-05-17 | PR #138 MERGED
Engine becomes config-driven for signal, timeframe, time-exit, and spread floor. KH-24 remains byte-identical (sha256 verified pre/post). Engine is now ready to run signals beyond KH-24 across any timeframe with configurable max-life caps and spread-floor application.
- New: `core/signal_adapter.py` — SignalAdapter Protocol, import_class helper, validate_aux_declaration
- New: `signals/kb_exhaustion_bar_adapter.py` — wraps KH-24 signal logic; deferred import avoids module-load cycle
- Patched: `scripts/phase_kgl_v2_4h_wfo.py` — adapter wiring, signal_tf + data config block, aux loading gated by adapter declaration, time_exit_bars cascade tail, _apply_spread_floor at four entry sites, sys.modules alias preventing __main__-vs-package double-load byte-identity break
- Patched: `configs/wfo_kh24.yaml` — adds signal_adapter / signal_adapter_kwargs / signal_tf / data block with KH-24 defaults
- Tests: 22 new tests in tests/test_engine_generalisation.py; full suite 1075 passed, 62 skipped
- KH-24 byte-identical: sha256 08118567a6ef…58e80ab0 verified pre- and post-PR
- Recovery: parallel CC-session worktree collision mid-build lost two adapter files; recovered via isolated git worktree pattern (now standard going forward)

## PIPELINE D1 PR 2 — STEPWISE CLIMBER POLICY | 2026-05-17 | PR #135 MERGED
§11 row 2 (Stepwise climber) exit policy + per-fold classifier dispatch land. ApplyPolicy decision pathway from PR #131 made concrete for the first §11 archetype. Other §11 rows deferred until an arc surfaces a Step-4 consumer.
- New: `core/exit_policies.py` — StepwiseClimberPolicy with MFE-lock at 1R + 0.75R trail-from-high; SL at lock-fire = max(BE_floor, trail_stop)
- Extended: `core/d1_pipeline.py` — D1Hook loads one joblib per fold per archetype; evaluate() takes fold_id; per-fold classifier dispatch with fail-loud on missing fold path
- Patched: `scripts/phase_kgl_v2_4h_wfo.py` — per-bar policy update_per_bar between bar_path append and SL check; fold_id threaded from WFO loop; audit fields (mfe_lock_fired_bar, trail_active_from_bar, classifier_fold_id) persisted to trades_all.csv
- Tests: 72 tests (20 exit-policy + 52 D1-pipeline), all passing
- Trail-suppression resolved against Step 5 simulator (scripts/arc_4/step5_stability.py:269-318): lock-fire bar arms AND uses trail in same-bar SL check
- KH-24 unaffected; D1_HOOK = None for all KH-24 configs
- Deferred: §11 rows 1, 3, 4, 5, 6, 7 exit policies — built per concrete consumer when an arc surfaces a Step-4 archetype needing them

## ARC 6 CLOSED | 2026-05-17 | DIES AT STEP 4 DEPLOYABILITY
Arc 6 (failed-breakout reversal long, out-of-registry insertion on `discovery/lomega_regime_conditional`) opened and closed same day. Steps 1–4 ran consecutively; all four mechanical PASSes.
- Step 1 PASS: pool 1,564 trades over 2020-10-01 → 2026-01-31, determinism `e57528...`, KH-24 co-fire 0.0000 by structural exclusion (bullish reclaim vs bearish exhaustion), cap-binding 17.65%
- Step 2 PASS: K=4 chosen, silhouette 0.4795; 4 clusters labelled (1 Stepwise, 1 early_peak, 2 unassigned — 54% pool)
- Step 3 PASS 2/4: c2 (Stepwise climber, n=242) SL=3.0×ATR composite 0.616; c0 (Stepwise-boundary, n=334) SL=2.0×ATR composite 0.384. c1 dies on early_peak structural mfe_p50 ceiling; c3 dies on reach_1R 0.697 vs 0.70 (0.003 absolute margin, within sampling noise)
- Step 4 PASS mechanically / FAIL substantively: Pipeline E fails both clusters (best AUC 0.600 c0 / 0.590 c2 vs 0.65 floor); Pipeline D1 clears AUC ≥ 0.60 (c0 t=4 RF AUC 0.602; c2 t=1 RF AUC 0.630) but threshold sweep collapses to max-F1 fallback at sub-1% recall — c0 precision 0.333 recall 0.009 (~3 admitted), c2 precision 0.250 recall 0.004 (~1 admitted) across the 5-year pool
- Steps 5 and 6 NOT EXECUTED: ≤ 1 trade/fold makes Step 5 stability noise-dominated and Step 6 WFO uninformative; arc dies at Step 4 deployability
- c2 D1 AUC monotone in t (0.630 t=1 → 0.711 t=10) — path-so-far information has real discriminative power; smallest-t rule + threshold-sweep design point combined to bury the deployable hook
- Calibration items raised: **Open-21 (new)** — §8 Step 4 deployability gate (strict-mode threshold sweep recommended; max-F1 fallback = cluster-dies, not graceful pass); **Open-17 expansion** — Tiebreak 1 noise floor (Arc 6 c2 selection flipped X=2.0→3.0 on 0.02 ATR / 0.15% relative at identical composite 0.6162); plus unnumbered cross-arc note on reach_1R floor noise tolerance
- Spec v0.2 erratum: literal v0.1 `swing_low_N = min(low[t-N..t-1])` is mathematically unsatisfiable; corrected to `min(low[t-N-M..t-M-1])` (verified Arc 6 Step 1 no-lookahead spot check 5/5). New file: `docs/signal_spec_failed_breakout_long_v0.2.md`
- Signal NOT permanently eliminated — path quality clean (c2 mfe_p50 4.47R, ww_pp 0.000); may return under richer feature regime, multi-TF entry context, or ensemble approach. Logged under CLAUDE.md "Not eliminated, but flagged"
- Commits: `b178ea4` (Step 1), `fd660ab` (Step 2), `f56b945` (Step 3), `8648c1b` (Step 4) on `discovery/lomega_regime_conditional`
- Closure: `docs/arc_results/ARC_6_RESULT.md`
- Arc 7 unaffected and continues on `phase/l_arc_7`
- Live system KH-24 unaffected and unchanged

## L_ARC_PROTOCOL v2.1.2 AMENDMENT | 2026-05-17 | doc-only

Open-18 cross-replay synthesis amendment. §2 internal-consistency floor relaxed from categorical admit list to single exclusion (`≠ scattered`); §11 Stepwise climber local_peaks ceiling extended 5-30 → 5-50; §15a arc Step 1 schema requirement added (Open-19 closure mechanism). Open-15 and Open-18 closed as empirically validated.

### Substantive changes

- §2 floor: `shape_tag ∈ {tight_unimodal, heavy_right_tail, bimodal_separated}` → `shape_tag ≠ scattered`. Mechanically: detection of "no structure" (scattered) replaces taxonomic admission. Cohorts with detectable structure but no clean label (unclassified, moderate-tail unimodal, multi-modal without clean separation) now pass §2's internal-consistency floor and route to §11 by centroid pattern.
- §11 Stepwise climber row: local_peaks 5-30 → 5-50. Empirical centroids from KH-24 c4 (30.94), Arc 3 c2 (24.42), Arc 2 redo2 cid 1 (20.07) cluster near or above the previous ceiling with same exit-policy profile.
- §7 fallback text rewritten to reference relaxed §2 floor.
- §15a added (arc Step 1 schema requirement): real-market R-fields out to entry+240, `is_held` column distinguishing held vs forward-observation bars. Reference implementation: `_flatten_bar_path_for_trade`. Engine refactor to signal-agnostic utility deferred.
- §17 glossary: `scattered` and `unclassified` operational definitions added.
- §16 Open-15 status augmented with cross-replay empirical confirmation.
- §16 Open-18 status → CLOSED (synthesis complete, three replays passed).
- §16 Open-19 added + closed (via §15a).

### Open items resolved

- Open-15: CLOSED — empirically confirmed across three replays
- Open-18: CLOSED — synthesis complete (Arc 3 Stepwise, KH-24 v2.0 c4, Arc 2 redo2 cid 1 all PASS)
- Open-19: CLOSED — §15a schema requirement landed

### Items deliberately NOT added

- Open-20 (raised in Arc 2 redo2 closure re: cohorts with negative final_r under fixed-SL re-imposition): not added to protocol. Reframed as Step 4+ measurement question — cohorts with high pct_peak_and_collapse should have realised R measured under the trailing-stop exit policy from §11, not under fixed-SL re-imposition at Step 3. The trailing stop *is* the realised-R protection; §2 is correctly a capturability gate, not a realised-R gate.

### Anchor preservation

KH-24 K=4 archetype 3 centroid (mono 0.576, local_peaks 14.19, pullback 0.020, ttp_rel 0.847) routes to Stepwise climber under both old [5,30] and new [5,50] local_peaks ranges. bimodal shape_tag passes under both old admit list and new `≠ scattered` floor. No routing change. §14 anchor explicitly not refreshed (user decision — deployed-pop reference holds).

### Files

- L_ARC_PROTOCOL.md → v2.1.2
- STATUS.md updated
- SESSION_ZERO.md version bump

## OPEN-18 CROSS-REPLAY SYNTHESIS | 2026-05-17 | empirical validation

Three parallel Open-18 replays completed under v2.1.1 — all PASS. Synthesis closed v2.1.2 amendment same day.

### Replay results

**Replay #1 — Arc 3 Stepwise climber (PASS):**
- c2 individual: PASS at SL=1.5×ATR, composite 0.473, shape_tag `bimodal_separated` (Hartigan dip + min-mode-mass + ≥1R separation all fired)
- Aggregate (c2+c4): PASS at SL=2.0×ATR, composite 0.467
- c4 individual: FAIL on magnitude/reach (small-magnitude archetype distinct from c2)
- Pre-peak Def C: 38.3% → 1.7% wrong_way on aggregate (~20× reduction, much larger than analyst predicted)
- Forks fired: C (genuine cluster-level bimodal_separated, not aggregation artifact) + F (44-58% of trades peak after Arc 3's 120-bar time exit; rescue is partly horizon-extension)
- Results: `results/replays_v2_1_1/arc_3_stepwise/REPLAY_RESULT.md`

**Replay #2 — KH-24 v2.0 c4 (PASS):**
- c1 (unresolved, near-stepwise): PASS at SL=2.0×ATR, composite 0.387
- c4 (unresolved, near-stepwise): PASS at SL=1.0×ATR, composite 0.304
- c0, c2, c3, c0+c2 aggregate: FAIL (peak-and-collapse cohorts, no magnitude edge)
- Findings: shape_tag dead-zone correlates with cap-binding (informs §5 + Amendment A architecture); pre-peak mono lift varies (c1 +0.108, c4 +0.036)
- Results: `results/replays_v2_1_1/kh24_v2_c4/REPLAY_RESULT.md`

**Replay #3 — Arc 2 redo2 cid 1 (PASS, Step 4+ deferred):**
- cid 1 (Stepwise climber): PASS at SL=3.0×ATR, composite 0.593, n=2285 (18.6% of pool), final_r_mean +3.148 (t=35.06)
- cid 3 (unclassified): clears §2 path-shape floors but negative final_r_mean under SL re-imposition; shelved as evidence for Open-20 (not added to protocol — see v2.1.2 entry above)
- Pre-peak Def C: 34.0% → 0.0% wrong_way on cid 1 (total rescue)
- bimodal_separated REJECTED on mode separation (0.48R < 1.0R required)
- New ops surfaced: Open-19 (Step 1 schema portability) closed in v2.1.2 §15a
- Results: `results/l_arc_2_redo2/step3/STEP3_SUMMARY.md` (full Arc 2 redo2 artefacts under `results/l_arc_2_redo2/`)

### Synthesis verdict

Pre-peak Def C is the dominant rescue mechanism — three independent cohorts, three signal classes, three near-total wrong_way reductions. The Open-15/Open-18 hypothesis is the load-bearing v2.1.1 mechanism.

bimodal_separated (Open-13) validated narrowly — one positive (Arc 3 c2). Useful admit for a specific archetype, not workhorse infrastructure.

§2 categorical floor + §11 Stepwise ceiling were the architectural gaps surfaced by the replays; v2.1.2 amendment addresses them.

### Closes

- Open-15: validated (cross-replay empirical confirmation)
- Open-18: complete (3/3 replays passed, synthesis written into v2.1.2)

## L_ARC_PROTOCOL v2.1.1 AMENDMENT | 2026-05-17 | doc-only

Combined patch: v2.1 refinements + v2.1 engine-reality corrections.

### v2.1.1 refinements (substantive)

- §7 SL selection rule: smallest-SL-passes → **capturability composite maximiser** among SLs passing all §2 floors. Composite = (mono_pre_peak − 0.55) + (frac_reach_1R − 0.70) + (0.30 − frac_wrong_way_pre_peak). Tiebreaker 1: physical-MFE in ATR units. Tiebreaker 2: smaller SL. Closes selection bias toward marginal-pass tight SLs.
- §5 forward window auto-extend: Step 2 clustering **re-runs on extended-window paths**. K may differ; extended-window K is the documented value. Previously unspecified.
- §11 row 7 routing precedence: when cluster matches base archetype AND passes bimodal_separated test, **both exit policies advance** through Step 4-6; Step 6 ships higher worst-fold ROI. Previously ambiguous.
- §10 multi-cluster ship decision: **one cluster per arc** — highest worst-fold ROI subject to DD ≤ 8%. Other survivors archived as portfolio candidates (Open-05). Previously unspecified.

### v2.1 engine-reality corrections (wording)

v2.1's §1 item 25, §5 "SL-free path recording" subsection, §7 SL sweep mechanics, §14 anchor refresh list, §16 Open-15, §17 SL-free path glossary all referenced an "engine PR pending" for SL-free path recording. CC's halt while attempting that PR established that the engine already has a v1.3 forward-window extension (`scripts/phase_kgl_v2_4h_wfo.py::_flatten_bar_path_for_trade`, lines 104-213) that emits forward observation bars (`is_held=0`) regardless of exit reason. No new column or schema change is needed — the existing forward bars provide what §7 SL sweep requires.

v2.1.1 corrects the wording across all eight locations:
- §0 v2.1 changes table: "Path recording" row removed (v1.3 extension predates v2.1, not a v2.0→v2.1 engine change)
- §1 item 25: rewritten to confirm SL sweep operates on existing `is_held=0` forward bars
- §5: "SL-free path recording" subsection replaced with clarification citing v1.3 extension
- §7: SL sweep mechanics step 2 references actual schema (`is_held=1` + `is_held=0`)
- §14: anchor refresh dependency #1 (engine PR) removed; only KH-24 v2.0 re-run blocking
- §16 Open-15: status rewritten to cite v1.3 extension as existing mechanism
- §16 Open-18 (new): empirical replays runnable immediately on existing trades_paths.csv
- §17: SL-free path glossary entry rewritten without `post_sl_hypothetical` reference

### Open items added

- Open-17: capturability composite weighting calibration (v2.2 item, weights currently equal as first-pass prior)
- Open-18: empirical replay of three v2.0-killed cohorts (Arc 2 redo c2, KH-24 v2.0 c4, Arc 3 Stepwise climber) under v2.1.1 rules to validate the pre-peak + composite + bimodal_separated mechanisms

### Branch / PR impact

- `feat/sl-free-path-recording` branch superseded and deleted (no PR opened)
- Anchor refresh blocked only on KH-24 v2.0 re-run analysis (no engine wait)
- Open-18 replays runnable immediately on existing `trades_paths.csv` from closed arcs

### Anchor preservation

All v2.1.1 changes either strictly more permissive than v2.1 (composite selects among passing SLs; row 7 adds candidate exit policy; window auto-extend doesn't fire for anchor population which is below 20% trigger) or constrain only ship-decision logic (post-§2). No §2 floor numbers change. KH-24 K=4 archetype 3 cannot fail under v2.1.1 if it passed under v2.1.

### Files
- L_ARC_PROTOCOL.md → v2.1.1
- STATUS.md updated
- SESSION_ZERO.md version bump
- PROTOCOL_IMPROVEMENT_BACKLOG.md P0.1 and P1.7 status text corrected

## L_ARC_PROTOCOL v2.1 AMENDMENT | 2026-05-17 | doc-only
Protocol amendment closing Open-08, Open-12, Open-13, Open-14, Open-15; partial close on Open-01 (forward-geometry only). Doc-only — engine work (SL-free path recording) deferred to separate engine PR; closed-arc re-runs under v2.1 deferred to post-engine-PR task.
Resolutions:
- Open-08 (`pullback_magnitude_median` degeneracy): empirically refuted by KH-24 v2.0 self-test (modal mass 0.31); closed as resolved
- Open-12 (silhouette tie tolerance): defined as 0.01 absolute in §6
- Open-13 (§2 / §11 row 7 bimodal incompatibility): §2 admits `bimodal_separated` shape_tag under Hartigan dip + mass + separation test in §7; routes to §11 row 7
- Open-14 (same-archetype aggregation kills capturable sub-clusters): §7 evaluates per-cluster AND per-aggregate; cluster proceeds if either passes
- Open-15 (SL/horizon asymmetry inflates frac_wrong_way): addressed via SL-free measurement (engine PR pending) + per-archetype SL sweep at Step 3
Substantive changes:
- §2 forward-geometry metrics split pre-peak (capturability gate) vs post-peak (exit-policy info)
- §7 SL sweep replaces single-SL evaluation; smallest-SL-that-passes selected per archetype
- §11 SL column demoted to prior; actual SL = Step 3 sweep result
- §6 K-selection tie tolerance 0.01 absolute
- §5 forward window auto-extend at >20% pool-level cap-binding
- §17 frac_wrong_way Def C ratified (pre-peak)
- §16 Open-16 added: 1R consistency across arcs (accepted as v2.1 design choice)
Calibration anchor refresh pending engine PR + KH-24 v2.0 re-run.
Files:
- L_ARC_PROTOCOL.md → v2.1
- STATUS.md updated for v2.1 active protocol
- SESSION_ZERO.md version reference bump
- PROTOCOL_IMPROVEMENT_BACKLOG.md marked resolved items

## PIPELINE D1 BACKTESTER EXTENSION | 2026-05-17 | PR #131 MERGED
Pipeline D1 plumbing landed per L_ARC_PROTOCOL v2.0 §3. Conditional exits keyed on mid-trade classifier output at bar N now supported. No D1 archetypes exist yet — extension is unexercised pending an arc producing a Step 4 candidate.
- New: core/features_path_so_far.py — path-so-far feature builder, single source of truth for offline training + runtime engine
- New: core/d1_pipeline.py — D1Hook class, Close / ApplyPolicy / Hold decision dataclasses, YAML config loader
- Patched: scripts/phase_kgl_v2_4h_wfo.py — hook fires at line ~1352 (pre-Priority-1 SL check, post-bar_path append); entry features stored at trade open; D1_HOOK global with YAML loader in main(); byte-identical when D1_HOOK is None
- Patched: L_ARC_PROTOCOL.md §3 — Pipeline D1 wording fixed: pre-t SL = 2.0 × ATR(14), "close at N+1 open" replaces "break-even close" framing; PR 1 / PR 2 scope split documented
- Tests: 41 new tests in tests/test_d1_pipeline.py, all passing; CI green (push + pull_request)
- Scope: PR 1 of 2. PR 2 (per-archetype §11 exit policies — MFE-lock, custom trail distances, TP1 half-off + trail, archetype-specific SL on classifier accept) deferred until an arc surfaces archetypes that need them. Trigger: Arc 4 or Arc 5 (or post-3D Stepwise climber if Open-13 amendment lands) produces an archetype reaching Step 4 with a §11 row the current engine cannot express.
- Live system KH-24 unaffected; D1_HOOK = None for all current configs

## KH-24 V2.0 SELF-TEST ARC CLOSED | 2026-05-16 | HALT AT STEP 3
Protocol self-test arc on the bare `kb_exhaustion_bar` signal opened and closed same day under `L_ARC_PROTOCOL.md` v2.0.
- Signal: bare KH-24 (C1-C6, C8, C9; C7 disabled); long-only, 4H, 28 FX, 1R = 2.0 × ATR(14), 240-bar forward window
- Step 1 plumbing PASS: pool 842 trades, deterministic, no lookahead, spread per `SPREAD_SEMANTICS_LOCK.md`
- Step 2 clustering PASS: K=5 chosen (silhouette 0.4327)
- Step 3 capturability FAIL: 0/5 clusters passed §2 conjunctively → arc dies per §7 (STEP3_FAIL_NO_CAPTURABLE_ARCHETYPE)
- Best contender c4 (trend-rider, n=122, 14.5% of pool): fwd_mfe_p50 6.65R, frac_reach_1R 1.000, frac_wrong_way 0.000 — missed monotonicity floor by 0.020 (mono=0.530 vs ≥0.55) AND shape_tag=scattered due to 87.7% forward-window cap-binding (censored final_r distribution)
- §14 calibration anchor non-reproducible on bare signal — anchor was measured on KH-24's filtered deployed 214-trade population, not bare 842-trade signal; structural mismatch with §15 pool floor
- Open-08 closed as resolved: `pullback_magnitude_median` operational definition empirically non-degenerate (mode fraction 0.31 on KH-24 paths, well under 0.80)
- Pattern flag: two-of-two arcs closed at Step 3 §2 floors on 2026-05-16 (KH-24 v2.0 + Arc 2 redo) — different signals, same gate failure mode (monotonicity / shape_tag)
- 8 cross-arc calibration candidates added to post-Arc-5 backlog: §2 monotonicity floor (0.55), shape_tag vs censoring, 240-bar forward window for 4H signals, §14-anchor / §15-pool mismatch, §17 `frac_wrong_way` disambiguation (Def B ratified), §11 archetype priors empirical refinement, per-pair n distribution stability, §16 Open-08 closure
- Pipeline D1 backtester extension work continues in separate chat — independent of this arc's closure
- KH-24 v1.0 deployment unaffected and unchanged
- Outputs: `results/arc_kh24_v2/ARC_KH24_V2_RESULT.md` + step1/, step2/, step3/ subfolders
- 77/77 CI tests passing under `tests/arc_kh24_v2/`

## ARC 3 CLOSED | 2026-05-16 | CLEAN-NULL at Step 3 (with reviewer flags)
Arc 3 (`TRIAL__volatility_regime__d1_atr_top_decile__any__h_120`) closes CLEAN-NULL with three reviewer-flagged opportunities for v2.1 calibration.
- Step 1 PASS: 2568 trades over 2020-10-01 → 2026-01-31, determinism byte-identical
- Step 2 PASS: K=7 chosen (silhouette 0.4177); 7 clusters → 2 named archetypes + 3 unassigned
- Step 3 FAIL: zero archetypes pass §2 conjunctive floors
  - Stepwise climber (27.5%, n=707): fails 2/6 — passes mono 0.559 / mfe_p50 3.34R / reach_1R 83.6% / size; killed by shape_tag=bimodal and wrong_way 38.3%. Median final_r +1.85R. Textbook §11 row-7 split-exit. Highest-priority v2.1 evidence.
  - Early-peak hold (40.0%): fails 5/6; aggregation of cluster 0 with cluster 3 destroyed clean sub-cluster (Open-14 evidence)
  - Cluster 1 (19.2%): fails 3/6; peak-and-collapse signature drives wrong_way
- Three reviewer flags in closure doc:
  - Stepwise climber opportunity → Open-13 (§2/§11 row-7 bimodal incompatibility, HIGH)
  - Aggregation hiding real archetypes → Open-14
  - SL/horizon asymmetry inflating wrong_way → Open-15
- Diagnostic tail (Arc 3D) recommended — 2D sweep (3 SL distances × 2 aggregation modes), ~30-60 min compute, generates direct evidence for Open-13/14/15. Reviewer decision pending.
- Result doc: `docs/arc_results/ARC_3_RESULT.md`
- Arc data: `results/l_arc_3/`
- Live system KH-24 unaffected and unchanged

## ARC 2 REDO CLOSED | 2026-05-16 | KILL AT STEP 3
Arc 2 redo opened and closed same day under L_ARC_PROTOCOL v2.0.
- Signal: TRIAL__mtf_alignment__2_down_mixed__kijun__h_120 (LCHAR_TOPN_REGISTRY.md Entry 2)
- Step 1 plumbing PASS: pool 12,262 trades, 28 pairs, deterministic, no lookahead
- Step 2 clustering PASS: K=4 chosen via silhouette 0.4778
- Step 3 capturability FAIL: 0/4 archetypes passed §2 hard floors; arc dies per §7
- Headline: cluster 2 (Stepwise climber, 18.6%) had fwd_mfe_p50 5.83R, frac_reach_1R 99.65%, final_r_mean +3.18R, t-stat +52.17 — failed on monotonicity 0.5414 (vs ≥0.55), frac_wrong_way 0.3051 (vs ≤0.30), shape_tag unclassified
- Cross-arc carryover: Open-09 evidence (hard floors false-killing high-magnitude cohorts), shape_tag definition pressure for high-body distributions
- Methodology cross-validation: v1.x verbatim WFO and v2.0 path-shape both FAIL on this signal — strong evidence signal is intractable not artefact
- Outputs: `results/l_arc_2_redo/ARC_2_REDO_RESULT.md` + step1/, step2/, step3/ subfolders

## L_ARC_PROTOCOL v2.0 LOCKED | 2026-05-16 | STRUCTURAL REWRITE
`L_ARC_PROTOCOL.md` v2.0 locked. Supersedes v1.0 + amendments (v1.1, v1.2, v1.3) for Arcs 3+. v1.x archived under `archive/`.
- Clustering basis shifts from forward-geometry magnitude to outcome-blind path-shape (monotonicity, local_peaks, pullback, time_to_peak_rel)
- Two-pipeline extractability: Pipeline E (entry-filter, RF AUC ≥ 0.65) and Pipeline D1 (deferred identification at bar N, RF AUC ≥ 0.60, ≤ 30% trades-exited-before-t)
- Two gates sequenced: capturability (§2 floors) → extractability (§2 disjunctive)
- Calibration anchor: KH-24 K=4 archetype 3 (passes via Pipeline D1 at t=3, RF AUC 0.638, exclusion 15.4%)
- Documentation: one live arc doc per arc (`ARC_<N>_LIVE.md`), finalised at end as `ARC_<N>_RESULT.md` — replaces per-step phase-doc workflow
- Workflow: direct-to-main for analysis scripts/results/calibration; PR required for backtester core, signal definitions, locked configs, CI, protocol doc
- Filter stacking: two-tier rule — Tier 1 (single classifier / feature subset / stack of 2) to clear gate, Tier 2 lift candidates after
- Exit-family map (§11): centroid pattern → archetype label → exit policy, per pipeline
- Evidence base: PR #129 (archetype diagnostic), PR #130 (predictability investigation)
- v1.0 + v1.1/v1.2 amendments + ops spec v1.0 moved to `archive/`
- Governance docs (CLAUDE.md, STATUS.md, SESSION_ZERO.md, README.md, WORKFLOW.md, NEW_CHAT_HANDOVER.md) updated to reference v2.0 as active protocol
- WORKFLOW.md / NEW_CHAT_HANDOVER.md flagged as v1.x-shaped — candidates for rewrite when Arc 3 opens
- Pipeline D1 backtester extension is the next planned engine PR (conditional exits keyed on mid-trade classifier output at bar N)

## L_ARC_PROTOCOL v1.0 LOCKED | 2026-05-13 | METHODOLOGY REDESIGN | SUPERSEDED 2026-05-16
`archive/L_ARC_PROTOCOL_v1_0.md` (v1.0) and `archive/L_ARC_OPERATIONAL_SPEC_v1_0.md` (v1.0) locked. Six-step extractability protocol replaces L6.0 verbatim-as-gate framing.
- Six steps: verbatim run (plumbing), descriptive trade-path analysis, extractability verdict, filter/exit derivation, re-characterisation, joint WFO
- Dual-tier WFO disposition: PASS-DEPLOYABLE (worst-fold annualised ROI >5%, mean >8%, DD <8%) / PASS-VIABLE (worst-fold ROI >0%, DD <8%) / clean-null
- Dual-gate step 3 verdict: AUC AND forward-geometry effect size AND cluster size ≥15% AND fold stability
- Component-ranked filter candidate scoring (no false-precision composite); held-out fold check on exits (derive folds 1–5, validate folds 6–7)
- BH haircut as reporting tiers (1/2/3), not gates; Tier 3 candidates require mechanical AND evidence-based justification
- Operational-cost haircut applied at PASS-DEPLOYABLE evaluation; 5ers schedule version recorded in CSV header
- Annualisation uses calendar-day formula; folds < 90 OOS days excluded from worst-fold annualisation
- Arc 1 redo doubles as protocol calibration check (`concurrent_signals_within_3h` must surface as ≥ Tier 2 predictor)
- Supersedes: L6.0 §9 (no filter rescue, verbatim-as-gate), §14 disposition rules; L6.0 §14.3 feature schema and §4/§5 WFO/pair structure carry forward
- Arc 1 and Arc 2 reopened for redo under new protocol; PHASE_L6_ARC1_OPEN, _P2_OPEN, _ARC2_OPEN marked SUPERSEDED
- WORKFLOW.md v2: phase docs co-located in `results/<arc>/` permanently (replaces `docs/` convention)
- CLAUDE.md, README.md, STATUS.md, SESSION_ZERO.md, NEW_CHAT_HANDOVER.md updated to reference new protocol

## L-ARC OPENED | 2026-05-09 | PLANNED
Active research direction shifts to the L characterization arc.
- Bottom-up exploratory data analysis approach
- Four-layer atlas: univariate, multi-timeframe, cross-pair, conditional structure
- Output is descriptive (statistics with CIs), not predictive
- Top-N candidates proceed to signal-testing via pre-registered ranking rule locked in L0
- Independent of KH-24; evaluates timeframe, direction, signal class, and pair-set fresh
- L arc proper begins after L0 methodology lock is drafted and signed off in next chat
- Source of truth: `L_ARC_PLAN.md` (SUPERSEDED on 2026-05-13 — atlas built, registry produced, signal-testing now under `L_ARC_PROTOCOL.md` v1.0)

## KH ARC CLOSED | 2026-05-09 | STRUCTURAL CEILING
Combined verdict from three diagnostic phases: KH-24 worst-fold ROI of +1.92% is the structural ceiling for that signal.
- Path A items A1, A5 closed by evidence; A2, A3, A4, A6, A7 deferred
- Path B1 closed by KI arc (1H port failed: t=0.095)
- Path B2 superseded by L arc; B3, B4 deferred or gated
- KH-24 remains live, locked, unchanged on VPS
- KH research roadmap updated to reflect closure
- Project research direction pivots to L characterization arc

## KH-29 | 2026-05-09 | EXIT-SIDE DIAGNOSTIC | AMBIGUOUS
Tested whether fold 7's MFE shrinkage was an exit-logic defect (kijun or trail) or a trend-extension defect.
- Reference cohort: folds 1+2+3 winners (n=53). Test cohort: fold 7 winners (n=14).
- Fold 7 median MFE_R 1.695 vs reference 1.961 (gap 0.27R; threshold for clean indictment 0.40R)
- Fold 7 median realized_R 0.672 vs reference 0.903 (deficit 0.23R; threshold 0.50R)
- Kijun_d1 exit rate: fold 7 35.7% vs reference 18.9% (+16.8pp; threshold 15pp)
- Fold 7 kijun-exit cohort capture (0.536) HIGHER than fold 7 trail-exit capture (0.454)
- All three verdict gates fail by margin. Per locked AMBIGUOUS rule, recommendation is pivot to B2/B3 without further KH-arc work.
- Outputs: `results/kh29/PHASE_KH29_RESULT.md`, `per_trade_excursions.csv`, `cohort_stats.csv`

## KH-28 | 2026-05-09 | REGIME DIAGNOSTIC | STRUCTURAL
Tested whether fold 7 weakness is addressable via signal-time regime selection.
- Candidate variables: R1 cross_pair_atr_ratio, R2 cross_pair_trend_strength, R3 cross_pair_dispersion, R4 pair_atr_ratio (control)
- Group A: fold 7 losers (n=13). Group B: non-fold-7 winners (n=93).
- No variable passed both p<0.05 AND protective direction
- R2 closest miss: right direction, p=0.077, n=13 (underpowered)
- R1 caveat: JPY-pair dominated by literal spec interpretation; equal-weight per-pair-normalized rebuild not done
- Trade-level diagnostic: fold 7 win rate matches good folds; deficit is in winning R magnitude (+0.83R vs +1.61R fold 1)
- Verdict STRUCTURAL on entry side. Triggered KH-29 to test exit side.
- Outputs: `results/kh28/PHASE_KH28_RESULT.md`, `regime_variables.csv`, `discrimination_results.csv`

## KH-27 | 2026-05-09 | RE-ENTRY PRE-FLIGHT | KILL
Tested whether extending the exposure cap to re-entries would have prevented fold 7 KH-25 losses.
- Pre-flight only — no full WFO. Pre-commit gate locked before running.
- Found re-entries fire AFTER original has exited via kh14_bar6 (10/10 sampled). Original never in open-positions set at re-entry time.
- 18 OOS re-entries; 0 blocked by extended cap; 0 of 3 fold 7 losses blocked
- Fold 7 losses are months apart (2025-04-30, 2025-10-01, 2025-12-26) with zero overlapping exposure
- "Correlated re-entry losses" framing in original KH-25 phase doc not supported by data — erratum added
- Verdict KILL by locked gate (0 of 3 fold 7 losses blocked). KH-27 full WFO not run.
- Outputs: `results/kh27_preflight/PHASE_KH27_PREFLIGHT.md`, `reentry_cap_analysis.csv`

## KH-24 | 2026-04-20 | GATE PASS ✓
First gate-passing result in the KH arc.
Change: h1_last_bar_close_in_range threshold 0.624 → 0.28
Base: KH-22 (exposure cap=2 + h1 filter T=0.624)
Result: worst-fold ROI +1.92% (F7), worst-fold DD 6.37% (F1)
All 7 OOS folds positive. 214 trades (328 baseline, −35%).
System lock: `KH24_SYSTEM_LOCK.md`
This is the new baseline. Do not modify wfo_kh24.yaml.

## [2.0.0] - 2025-09-30
- Stable baseline v2.0 release (config-driven, invariant-checked)
- Golden Standard WL/S classification finalized
- Walk-Forward v1 + Monte Carlo (baseline)
- Spread affects PnL only (invariant)
- Audit immutability (entry TP/SL fields)

## v1.9.9-hardstop
- New Golden Standard: Hard-Stop Realism
  - Intrabar touch of TP/BE/TS exits immediately.
  - Trailing Stop activates & updates on closes only (monotone), exits intrabar when touched.
  - Pre-TP1 C1 reversal scratch ≈ 0 PnL (tolerance allows spread).
- Invariants preserved: audit fields immutable; spreads change PnL only, not trade counts.
