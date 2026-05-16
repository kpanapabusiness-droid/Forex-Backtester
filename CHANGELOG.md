# Changelog

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
