# STATUS

> Tight current-state snapshot. For full context, read `SESSION_ZERO.md` first.
> For methodology, read `L_ARC_PROTOCOL.md` (v2.1.1, self-contained).
> Last updated: 2026-05-17 — L_ARC_PROTOCOL v2.1.1 amendment landed (v2.1 refinements + engine-reality corrections). Engine PR for SL-free path recording superseded — v1.3 forward extension already provides the required SL-free observation. Open-18 empirical replays of v2.0-killed cohorts runnable now.

---

## Active protocol

- Active protocol: L_ARC_PROTOCOL v2.1.1 (amendment landed 2026-05-17)
- Calibration anchor: KH-24 K=4 archetype 3 — v2.0 values; refresh under v2.1.1 pending KH-24 v2.0 re-run only (no engine work needed; v1.3 forward extension already provides SL-free observation)
- Next engine PR: none currently planned. PR #131 (D1 plumbing) is merged; D1 PR 2 (per-archetype §11 exit policies) deferred until an arc surfaces a Step-4 consumer
- Next chat task: Open-18 empirical replays — Arc 3 Stepwise climber under v2.1.1 first (highest priority, designed-for cohort), then KH-24 v2.0 c4 (anchor refresh), then Arc 2 redo c2 (cross-arc validation)

---

## Current Phase

**L_ARC_PROTOCOL v2.1.1 amendment landed 2026-05-17.** Combined patch with two parts:

**v2.1.1 substantive refinements:**
- §7 SL selection: capturability composite maximiser (replaces smallest-SL-passes)
- §5 forward window auto-extend: Step 2 re-runs on extended-window paths
- §11 row 7 bimodal_separated: routing precedence resolved — both base archetype and split-exit policies advance through Step 4-6, Step 6 ships best
- §10 multi-cluster ship decision: one cluster per arc, highest worst-fold ROI subject to DD ≤ 8%

**v2.1 engine-reality corrections:**
- v2.1 introduced language about an "engine PR pending" for SL-free path recording
- CC's halt while attempting that PR established that the engine already has a v1.3 forward-window extension (`_flatten_bar_path_for_trade`) that emits forward bars (`is_held=0`) regardless of exit reason
- v2.1.1 corrects the wording across §1, §5, §7, §14, §16, §17 — no engine change implied or needed
- `feat/sl-free-path-recording` branch superseded and deleted

**Next: Open-18 empirical replays.** Three v2.0-killed cohorts (Arc 2 redo c2, KH-24 v2.0 c4, Arc 3 Stepwise climber) become testable under v2.1.1 rules immediately — no Step 1 re-run, no engine work, just re-running Step 3 analysis on existing `trades_paths.csv`. Recommended order: Arc 3 Stepwise climber → KH-24 v2.0 c4 (refreshes §14 anchor) → Arc 2 redo c2.

**Arc 4** opens after Open-18 replays validate v2.1.1 mechanics.

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
| L_ARC_PROTOCOL v2.1.1 amendment | LANDED (2026-05-17) | Combined patch: v2.1.1 refinements (§7 capturability composite, §5 re-cluster on extend, §11 row 7 routing precedence, §10 ship rule) + v2.1 engine-reality corrections (§1, §5, §7, §14, §16, §17 wording fixes — v1.3 forward extension already provides SL-free observation; engine PR was unneeded). `feat/sl-free-path-recording` branch superseded. Open-18 replays of v2.0-killed cohorts now runnable. |
| L_ARC_PROTOCOL v2.1 amendment | LANDED (2026-05-17) | Doc-only protocol amendment closing Open-08/12/13/14/15 (partial Open-01). Pre-peak §2 metrics, SL sweep at Step 3, per-cluster + per-aggregate evaluation, bimodal_separated admission, SL column demotion in §11, K-tie tolerance, forward window auto-extend. Engine PR + closed-arc re-runs follow. |
| Pipeline D1 backtester extension | DELIVERED (PR #131, 2026-05-17) | D1 plumbing + close-at-market policy; 41 tests; byte-identical when D1_HOOK=None. PR 2 (per-archetype §11 exit policies) deferred — awaiting a Step-4 archetype consumer. |
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
| Open-18 empirical replays | HIGH | Run v2.1.1 Step 3 analysis on existing `trades_paths.csv` for: Arc 3 Stepwise climber (first), KH-24 v2.0 c4 (anchor refresh), Arc 2 redo c2 (cross-arc). No engine work — v1.3 forward bars (`is_held=0`) already in schema. |
| §14 anchor refresh | HIGH | Pick refreshed anchor from KH-24 v2.0 c4 replay result. Only blocker is the replay above. |
| Arc 3D diagnostic tail | SUPERSEDED | v2.1 resolved Open-13/14/15; Open-18 replays serve the same purpose. Decision moot. |
| Arc 4 opens under v2.1.1 | HIGH | Opens after Open-18 replays validate v2.1.1 mechanics + anchor refresh lands. Registry Entry 4 (1-bar horizon, univariate-extreme `neg`). |
| Pipeline D1 PR 2 (per-archetype §11 exit policies) | DEFERRED | Awaiting an arc to surface an archetype reaching Step 4 with a §11 row the current engine cannot express. No build until concrete consumer surfaces. |

No outstanding bugs or issues against KH-24. No pending fixes against the backtester.

---

## Watch Items

| Item | Status | Notes |
| --- | --- | --- |
| §2 calibration pattern | 2/3 arcs hit it | 2026-05-16 closures: Arc 2 redo and KH-24 v2.0 self-test both failed §2 on monotonicity / shape_tag with strong-edge cohorts (KH-24 c4: fwd_mfe_p50 6.65R, reach_1R 1.000, missed mono by 0.020; Arc 2 redo c2: t-stat +52.17, missed mono by 0.009). Arc 3 closed CLEAN-NULL on related §2 floors. If Arc 4 closes the same way, §2 calibration moves from watch item to blocking the protocol from producing systems. Post-Arc-5 calibration review now has concrete inputs. |

---

## Cross-Arc Calibration Backlog (post-Arc-5 review)

Items accumulating from arc closures under v2.0. Per §1.8 within-arc thresholds do not move; per §12 cross-arc calibration is governed and requires a calibration document + chat-level approval. The 2026-05-17 v2.1 amendment resolved or partially resolved most of the items below; items still requiring evidence / execution are kept under "Active backlog".

### Resolved in v2.1 (2026-05-17 amendment)

| Item | Source arc(s) | Resolution |
| --- | --- | --- |
| §17 `frac_wrong_way` definition | KH-24 v2.0 | CLOSED in v2.1 — §17 Def C ratified (MAE ≤ −1R on or before peak_mfe_bar) |
| §16 Open-08 closure | KH-24 v2.0 | CLOSED 2026-05-17 — `pullback_magnitude_median` empirically non-degenerate; §16 marked resolved |
| Open-12 silhouette tie tolerance | Arc 3 | CLOSED in v2.1 — §6 tolerance 0.01 absolute silhouette |
| Open-13 §2/§11 row-7 bimodal incompatibility | Arc 3 | CLOSED in v2.1 — `bimodal_separated` admitted at §2 under Hartigan dip + mass + separation test; routes to §11 row 7 |
| Open-14 same-archetype aggregation rule | Arc 3 | CLOSED in v2.1 — §7 evaluates per-cluster AND per-aggregate; cluster proceeds if either passes |
| Open-15 SL/horizon asymmetry inflating wrong_way | Arc 3 | CLOSED in v2.1 / v2.1.1 — addressed via SL-free measurement (uses v1.3 forward-window extension already in engine; no PR needed) + per-archetype SL sweep at §7 (capturability composite selection under v2.1.1) |
| 240-bar forward window for 4H signals | KH-24 v2.0 | CLOSED in v2.1 — §5 forward window auto-extend at >20% pool-level cap-binding (2× extension default) |
| Per-pair n distribution stability concern | KH-24 v2.0 | CLOSED in v2.1 — §9 per-pair stability reporting added at Step 5 (informational; flags > 50% concentration in < 5 pairs) |

### Active backlog (still open or partial)

| Item | Source arc(s) | Priority | Notes |
| --- | --- | --- | --- |
| §2 monotonicity floor (0.55) calibration | KH-24 v2.0, Arc 2 redo | ADDRESSED in v2.1.1 | Pre-peak measurement subsumes the full-window biasing; Open-18 replays validate empirically. Floor stays 0.55 pending replay outcomes — re-open if replays surface residual near-misses. |
| shape_tag definitions vs forward-window censoring | KH-24 v2.0 | HIGH | v2.1 `bimodal_separated` admission covers part of the tightness. Censoring-vs-shape_tag tension (e.g. KH-24 v2.0 c4 87.7% cap-binders → shape_tag=scattered) remains until re-run evidence. Forward window auto-extend reduces but doesn't eliminate. |
| §14 anchor population vs §15 pool floor mismatch | KH-24 v2.0 | HIGH | v2.1.1 §14 documents refresh path. Execution pending KH-24 v2.0 re-run only (no engine work needed — v1.3 forward extension already provides SL-free observation; Open-18 replay #2 produces the refreshed anchor). |
| §11 archetype priors empirical refinement (Open-07) | KH-24 v2.0, Arc 2 redo, Arc 3 | MEDIUM | v2.1 demotes §11 SL column to prior (actual SL = Step 3 sweep). Centroid-pattern refresh remains open; deferred until Arc 4 + Arc 5 produce additional evidence. |

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
