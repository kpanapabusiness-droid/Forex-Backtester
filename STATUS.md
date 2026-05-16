# STATUS

> Tight current-state snapshot. For full context, read `SESSION_ZERO.md` first.
> For methodology, read `L_ARC_PROTOCOL.md` (v2.0, self-contained).
> Last updated: 2026-05-16 — Arc 3 CLOSED (CLEAN-NULL at Step 3); Arc 2 redo also CLOSED (KILL at Step 3) earlier same day.

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
| Pipeline D1 backtester extension | HIGH | Conditional exits at bar N; next engine PR per v2.0 §13 |
| v2.1 calibration tracking | MEDIUM | Five cross-arc items from Arc 3 (Open-12 through Open-15 plus Open-07 evidence) |

No outstanding bugs or issues against KH-24. No pending fixes against the backtester.

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
