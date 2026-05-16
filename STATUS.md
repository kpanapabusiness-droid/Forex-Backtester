# STATUS

> Tight current-state snapshot. For full context, read `SESSION_ZERO.md` first.
> For methodology, read `L_ARC_PROTOCOL.md` (v2.0, self-contained).
> Last updated: 2026-05-16 — three arcs CLOSED at Step 3 §2 floors (Arc 2 redo KILL, KH-24 v2.0 self-test HALT, Arc 3 CLEAN-NULL); Arc 4 next.

---

## Active protocol

- Active protocol: L_ARC_PROTOCOL v2.0
- Calibration anchor: KH-24 K=4 archetype 3 (passes via Pipeline D1 at t=3)
- Next engine PR: Pipeline D1 backtester extension (conditional exits at bar N)

---

## Current Phase

**Arc 3 closed CLEAN-NULL at Step 3 under `L_ARC_PROTOCOL.md` v2.0.** Zero archetypes pass §2 capturability floors as drawn. Stepwise climber (27.5% pool, n=707) is the closest call — passes mono / mfe_p50 3.34R / reach_1R 83.6% / size cleanly but killed by §2 shape_tag floor excluding bimodal, despite §11 row 7 defining bimodal as a valid archetype.

Five cross-arc items logged for v2.1: Open-12 (silhouette tie tolerance), Open-13 (§2/§11 row-7 incompatibility — highest priority), Open-14 (same-archetype aggregation rule), Open-15 (SL/horizon asymmetry), plus Open-07 evidence.

**Diagnostic tail (Arc 3D) recommended in closure doc** — reviewer decision pending whether to run before Arc 4 opens.

**Next:** Arc 4 (registry Entry 4: `TRIAL__univariate_extreme__bar_range_top_decile__neg__h_001`).

Order under v2.0: Arc 4 → Arc 5.

---

## Live System (Out of Scope for L Arc)

| Item | Value |
| --- | --- |
| System | KH-24 (kb_exhaustion_bar + cap=2 + 1H CIR T=0.28) |
| Direction | Long only |
| Timeframe | 4H primary, D1 regime filter (lag-1 convention) |
| Pairs | 28 FX |
| Risk | 1% of reset floor balance |
| Gate | PASS — all 7 OOS folds positive |
| Worst-fold ROI | +1.92% (Fold 7) |
| Worst-fold DD | 6.37% (Fold 1) |
| Total OOS trades | 214 across Oct 2020 – Jan 2026 |
| Status | Live on Contabo VPS / 5ers |
| Spec | `docs/KH24_SYSTEM_LOCK.md` |
| Config | `configs/wfo_kh24.yaml` (locked, do not modify) |
| EA | `KH24_EA.mq5` v2.01 |

KH-24 is locked and unchanged. No modifications without an explicit modification phase.

---

## Active Research

L arc signal testing under `L_ARC_PROTOCOL.md` v2.0 (Arcs 3+). v1.x archive at `archive/` for historical Arc 1, Arc 2 reference.

| Item | Detail |
| --- | --- |
| Protocol | `L_ARC_PROTOCOL.md` v2.0 (self-contained; v1.x ops spec archived) |
| Signals | Top-N from `docs/LCHAR_TOPN_REGISTRY.md` — 5 arcs scheduled |
| Approach | Six-step pipeline: plumbing → path-shape clustering → capturability → extractability (E or D1) → cross-fold stability → WFO |
| Current arc | Arc 4 — opens after Arc 3 closure (CLEAN-NULL) and any authorised Arc 3D diagnostic |
| Calibration anchor | KH-24 K=4 archetype 3 (passes via Pipeline D1 at t=3) |
| Risk per trade | 0.5% × reset floor balance |
| Pair set | All 28 FX, same as KH-24 |
| WFO | 7 anchored expanding folds, OOS Oct 2020 – Jan 2026 |
| Folder layout | `results/l_arc_N/` co-located per WORKFLOW.md v2 |

---

## Protocol Tier Thresholds (Step 6 Disposition)

**PASS-DEPLOYABLE:**
- Worst-fold annualised ROI > 5% (net of spread, net of operational-cost haircut)
- Worst-fold max DD < 8%
- Mean fold annualised ROI > 8%
- Trades per fold ≥ scaled floor (15 at OOS ≥ 180 days; scaled down to min 5 below)

**PASS-VIABLE:**
- Worst-fold ROI > 0% (any positive, gross of haircut)
- Worst-fold max DD < 8%
- Trades per fold ≥ scaled floor

**CLEAN-NULL:** does not meet PASS-VIABLE thresholds; or any worst-fold DD ≥ 8%.

Annualisation: `fold_raw_ROI × (365 / fold_OOS_days)`. Folds < 90 OOS days excluded from worst-fold annualisation calculation.

---

## Recent Closures

| Phase | Verdict | Finding |
| --- | --- | --- |
| Arc 3 (l_arc_3) | CLEAN-NULL at Step 3 (2026-05-16) | TRIAL__volatility_regime__d1_atr_top_decile__any__h_120 — zero archetypes pass §2 as drawn; Stepwise climber profile shows real edge (mfe_p50 3.34R, reach_1R 83.6%, median final_r +1.85R) but killed by §2/§11-row-7 bimodal incompatibility; three reviewer flags + five cross-arc items |
| KH-24 v2.0 self-test (arc_kh24_v2) | HALT at Step 3 (2026-05-16) | Bare `kb_exhaustion_bar` under v2.0 — protocol self-test. 0/5 clusters cleared §2 conjunctively → arc dies per §7. Best contender c4 (trend-rider, n=122, fwd_mfe_p50 6.65R, frac_reach_1R 1.000, frac_wrong_way 0.000) missed monotonicity floor by 0.020 AND shape_tag=scattered from 87.7% forward-window cap-binding. §14 anchor non-reproducible on bare signal (anchor measured on filtered deployed population). 8 cross-arc items added. Open-08 closed as resolved. |
| Arc 2 redo | KILL at Step 3 (2026-05-16) | All 4 archetypes failed §2 capturability under hard floors; cluster 2 (Stepwise climber) carried strong magnitude but unextractable paths. Cross-arc carryover for v2.x calibration: Open-09 evidence, shape_tag definition pressure. |
| L_ARC_PROTOCOL v2.0 | LOCKED 2026-05-16 | Path-shape clustering + two-pipeline E/D1 extractability; KH-24 K=4 archetype 3 = calibration anchor; v1.x archived for Arcs 1, 2 historical reference |
| v2.0 predictability investigation | DELIVERED (PR #130) | Evidence base for v2.0 extractability gate |
| v2.0 archetype diagnostic | DELIVERED (PR #129) | Evidence base for path-shape clustering on KH-24 + Arc 1 + Arc 2 |
| L_ARC_PROTOCOL design | LOCKED v1.0 (2026-05-13, superseded by v2.0) | Six-step extractability protocol replaces L6.0 verbatim-as-gate framing |
| L_ARC_OPERATIONAL_SPEC design | LOCKED v1.0 | Deliverables, angle catalogues, scoring tables, effect size definitions |
| L6_0_METHODOLOGY_LOCK | SUPERSEDED | §9, §14 disposition rules superseded; feature schema (§14.3) and pair-set/WFO structure (§5, §4) carry forward |
| PHASE_L6_ARC1_OPEN, _P2_OPEN, _ARC2_OPEN | SUPERSEDED | Replaced by Arc 1/2 redo arc-open docs under new protocol |
| WORKFLOW.md | UPDATED v2 | Folder convention: phase docs co-located in `results/<arc>/`, permanent |

---

## Open Items

| Item | Priority | Notes |
| --- | --- | --- |
| Arc 3D diagnostic tail decision | HIGH | Reviewer to decide on running 2D SL × aggregation sweep before Arc 4 |
| Arc 4 opens under v2.0 | HIGH | Step 1 plumbing on registry Entry 4 (1-bar horizon, univariate-extreme `neg`) |
| Pipeline D1 backtester extension | HIGH | Conditional exits at bar N; next engine PR per v2.0 §13. Work continues in separate chat. |
| v2.1 calibration tracking | MEDIUM | Cross-arc items accumulating from Arcs 2 redo, KH-24 v2.0, Arc 3 — see Cross-arc calibration backlog below |

No outstanding bugs or issues against KH-24. No pending fixes against the backtester.

---

## Watch Items

| Item | Status | Notes |
| --- | --- | --- |
| §2 calibration pattern | 2/3 arcs hit it | 2026-05-16 closures: Arc 2 redo and KH-24 v2.0 self-test both failed §2 on monotonicity / shape_tag with strong-edge cohorts (KH-24 c4: fwd_mfe_p50 6.65R, reach_1R 1.000, missed mono by 0.020; Arc 2 redo c2: t-stat +52.17, missed mono by 0.009). Arc 3 closed CLEAN-NULL on related §2 floors. If Arc 4 closes the same way, §2 calibration moves from watch item to blocking the protocol from producing systems. Post-Arc-5 calibration review now has concrete inputs. |

---

## Cross-Arc Calibration Backlog (post-Arc-5 review)

Items accumulating from arc closures under v2.0. Per §1.8 within-arc thresholds do not move; per §12 cross-arc calibration is governed and requires a calibration document + chat-level approval. The list below is record-only — no protocol or threshold edits yet.

| Item | Source arc(s) | Priority | Notes |
| --- | --- | --- | --- |
| §2 monotonicity floor (0.55) calibration | KH-24 v2.0, Arc 2 redo | HIGH | KH-24 v2.0 c4 missed by 0.020 with structurally perfect profile; c1 by 0.049 with heavy_right_tail. Arc 2 redo c2 by 0.009 with t-stat +52.17 (n=2,278). Calibration review should consider whether 0.55 is too high, whether conjunctive AND should soften to "k of 6," or whether monotonicity should be median-across-cluster-trades rather than centroid. |
| shape_tag definitions vs forward-window censoring | KH-24 v2.0 | HIGH | When N% of a cohort hits the window cap, their final_r is censored. KH-24 v2.0 c4: 87.7% cap-binders → shape_tag=scattered despite underlying MFE distribution being clean and right-tailed. Candidates: censor-aware shape_tag, shape_tag derived from MFE not final_r, or skip shape_tag for cohorts with > 50% cap-binders. |
| 240-bar forward window for 4H signals | KH-24 v2.0 | HIGH | 16.7% pool-level cap-binding on bare KH-24; 87.7% on c4. Too tight for slow trend-following signals. Candidates: per-arc-configurable window, window scaled to expected hold time, or window extension when cap-binding exceeds a threshold. |
| §14 anchor population vs §15 pool floor mismatch | KH-24 v2.0 | HIGH | §14 numbers measured on KH-24's filtered 214-trade deployed population; §15 pool floor (≥ 500) excludes that population. Anchor and protocol describe non-overlapping populations. Calibration review should either re-derive §14 on a v2.0-compatible population or document the mismatch explicitly. |
| §17 `frac_wrong_way` definition | KH-24 v2.0 | MEDIUM | Ratify Def B (wrong-from-outset: MAE ≤ -1R before MFE > 0.5R, or MFE > 0.5R never reached) as protocol's intent. Add explicit definition to §17 glossary. Def A (final_r ≤ -0.5) gives nonsense on hard-SL designs. |
| §16 Open-08 closure | KH-24 v2.0 | LOW | `pullback_magnitude_median` operational definition (close_r dip between peaks) is empirically non-degenerate on bare KH-24 (mode fraction 0.31). Open-08 can be closed as resolved. |
| §11 archetype priors empirical refinement (Open-07) | KH-24 v2.0, Arc 2 redo, Arc 3 | MEDIUM | None of §11 patterns matched cleanly across the three arcs. Most clusters labelled `unresolved_*`. Cross-arc evidence accumulating that §11 centroid ranges are first-pass priors needing empirical refinement. |
| Per-pair n distribution stability concern | KH-24 v2.0 | LOW | 15/28 pairs flagged < 30 trades in bare KH-24 pool. §5 keeps them; structural concern remains if downstream archetypes concentrate in low-n pairs. Candidates: per-archetype per-pair stability check at Step 5, or pool-level rule excluding low-n pairs from clusters they don't reach a minimum count in. |
| Open-12 silhouette tie tolerance | Arc 3 | MEDIUM | (from Arc 3 closure doc) |
| Open-13 §2/§11 row-7 bimodal incompatibility | Arc 3 | HIGH | (from Arc 3 closure doc; highest priority) |
| Open-14 same-archetype aggregation rule | Arc 3 | MEDIUM | (from Arc 3 closure doc) |
| Open-15 SL/horizon asymmetry inflating wrong_way | Arc 3 | MEDIUM | (from Arc 3 closure doc) |

---

## Permanently Eliminated (post-KH closure + post-L6.0)

- KH-25 re-entry exposure cap hypothesis (KH-27 KILL)
- KH-24 fold 7 entry-side regime selection (KH-28 STRUCTURAL)
- KH-24 fold 7 exit-side indictment (KH-29 AMBIGUOUS)
- KI arc: 1H timeframe port of KH-24 signal (mean R 0.004, t=0.095)
- L6.0 verbatim-as-gate framing as arc disposition rule (replaced by extractability protocol)

Earlier eliminations (KGL_V2 era, JL forward bias, NNFX, exit indicator sweeps, etc.) recorded in `CLAUDE.md`, `docs/KH_Research_Roadmap.md`, and `project_brief.md`.

Note: Arc 2 signal (mtf_alignment.2_down_mixed.kijun, h=120) shelved 2026-05-16 as "real edge, not extractable under v2.0 as drawn" — see `results/l_arc_2_redo/ARC_2_REDO_RESULT.md`. Not permanently eliminated; reopenable contingent on v2.x calibration amendment.

---

## Results Locations

| Result | Location |
| --- | --- |
| KH-24 live system | `results/kh24/` |
| KH-27 pre-flight | `results/kh27_preflight/` |
| KH-28 regime analysis | `results/kh28/` |
| KH-29 excursion analysis | `results/kh29/` |
| L characterisation atlas | `results/lchar/` |
| L arc signal testing (current) | `results/l_arc_N/` (folder convention inherited from v1.x ops spec §2) |
| Arc 2 redo (closed KILL) | `results/l_arc_2_redo/` |
| KH-24 v2.0 self-test (closed HALT) | `results/arc_kh24_v2/` + `results/arc_kh24_v2/ARC_KH24_V2_RESULT.md` |
| Arc 3 (closed CLEAN-NULL) | `results/l_arc_3/` + `docs/arc_results/ARC_3_RESULT.md` |

---

## See Also

- `SESSION_ZERO.md` — full current state primer
- `L_ARC_PROTOCOL.md` — active research methodology (read first)
- `archive/L_ARC_OPERATIONAL_SPEC_v1_0.md` — v1.x per-step deliverables (historical; v2.0 protocol is self-contained)
- `docs/LCHAR_TOPN_REGISTRY.md` — the 5 candidate signals being tested
- `WORKFLOW.md` v2 — phase management and folder convention
- `docs/KH_Research_Roadmap.md` — closed and deferred KH-arc items
- `CHANGELOG.md` — full phase history
- `project_brief.md` — long-form project history and locked decisions

---

*This file is intentionally tight. For substance, read SESSION_ZERO.md and the protocol docs.*
