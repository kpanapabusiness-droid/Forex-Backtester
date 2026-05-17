# STATUS

> Tight current-state snapshot. For full context, read `SESSION_ZERO.md` first.
> For methodology, read `L_ARC_PROTOCOL.md` (v2.1.2, self-contained).
> Last updated: 2026-05-17 — Arc 4 closed CLEAN-NULL on transaction-cost truth; `configs/spread_floors_5ers.yaml` flagged for per-pair empirical replacement (HistData audit shows uniform 0.1 pip floor under-models real spreads 3-48x per pair). KH-24 live deployment unchanged; WFO claim downgraded to pass-viable retroactively. Backtester extension closed (PR #131 + #135 + #138 merged). Arc 6 closed DIES at Step 4 deployability.

---

## Active protocol

- Active protocol: L_ARC_PROTOCOL v2.1.2 (amendment landed 2026-05-17)
- Calibration anchor: KH-24 K=4 archetype 3 — v2.0 values; deployed-pop reference held (no refresh from Open-18 replays per user decision; v2.1.2 anchor preservation verified — centroid still routes to Stepwise under 5-50 local_peaks range)
- Next engine PR: none currently planned. Backtester extension complete: PR #131 (D1 plumbing), PR #135 (D1 PR 2 Stepwise climber policy + per-fold classifiers), PR #138 (engine generalisation: signal adapter + TF pluggability + time-exit + spread floor). §11 rows 1, 3, 4, 5, 6, 7 exit policies deferred until an arc surfaces a Step-4 consumer needing them.
- **BLOCKER:** `configs/spread_floors_5ers.yaml` flagged for per-pair empirical floor replacement before Arc 5+ proceeds. See `docs/SPREAD_FLOOR_AUDIT_FINDING.md`.
- Next chat task: spread floor file replacement under SPREAD_SEMANTICS_LOCK governance; then Open-18 replays; then Arc 5. Cross-arc calibration session for Open-21 (Step 4 deployability gate) + Open-17 expansion (Tiebreak 1 noise floor) queued.

---

## Current Phase

**L_ARC_PROTOCOL v2.1.2 amendment landed 2026-05-17.** Synthesis of Open-18's three parallel replays:

- **Replay #1 Arc 3 Stepwise (PASS):** c2 individual fires bimodal_separated genuinely (composite 0.473 at SL=1.5×ATR). Aggregate PASS at SL=2.0, composite 0.467. Pre-peak Def C: 38.3% → 1.7% wrong_way.
- **Replay #2 KH-24 v2.0 c4 (PASS):** c1 (n=365, SL=2.0, composite 0.387) and c4 (n=122, SL=1.0, composite 0.304) both rescue. Surfaced shape_tag dead-zone correlation with cap-binding.
- **Replay #3 Arc 2 redo2 cid 1 (PASS):** SL=3.0×ATR, composite 0.593, n=2285 (18.6% of pool). Pre-peak Def C: 34% → 0% wrong_way.

**Synthesis verdict:** Pre-peak Def C is the dominant rescue mechanism across all three. bimodal_separated validated narrowly (one positive: Arc 3 c2). §2 categorical floor was mixing capability gating with classification gating; v2.1.2 relaxes it. §11 Stepwise local_peaks ceiling was a first-pass prior; empirical evidence justifies 5-50.

**v2.1.2 substantive changes:**
- §2 internal-consistency floor: `∈ {tight_unimodal, heavy_right_tail, bimodal_separated}` → `≠ scattered`
- §11 Stepwise local_peaks ceiling: 5-30 → 5-50
- §15a added: arc Step 1 schema requirement (Open-19 closure mechanism)
- Open-15, Open-18 closed (validated); Open-19 closed
- Open-20 deliberately NOT added: reframed as Step 4 measurement question (high-pct_peak_and_collapse cohorts measured under trailing-stop exit, not fixed-SL re-imposition; the trailing stop *is* the realised-R protection)

**Anchor preservation:** KH-24 K=4 archetype 3 centroid (mono 0.576, local_peaks 14.19, pullback 0.020, ttp_rel 0.847) routes to Stepwise climber under both old [5,30] and new [5,50] ranges; bimodal shape_tag passes `≠ scattered`. No routing change.

**Arc 4 closed CLEAN-NULL 2026-05-17.** Signal had real edge through Step 5 under modeled spreads; real-spread reconciliation (HistData audit) showed cost model under-modeled spreads by 3-48x per pair, killing F6 worst-fold sign consistency on Arc 4 cluster 1. Spread floor replacement is now the priority blocker for Arc 5+ work. Open-18 replays also blocked. KH-24 live deployment unchanged; WFO claim downgraded to pass-viable retroactively. See `docs/arc_results/ARC_4_RESULT.md` and `docs/SPREAD_FLOOR_AUDIT_FINDING.md`.

---

## Arc 6 closure (2026-05-17)

Arc 6 (`signal_spec_failed_breakout_long_v0.2`, out-of-registry insertion on `discovery/lomega_regime_conditional`) opened and closed same day. Steps 1–4 ran consecutively; all four mechanical PASSes.

**Disposition:** **DIES at Step 4 — deployability-level failure on Pipeline D1 recall collapse.**

Pipeline E fails both Step 3 survivors (c0 best AUC 0.600, c2 best AUC 0.590; 0.65 floor). Pipeline D1 mechanically clears AUC ≥ 0.60 (c0 0.602 at t=4; c2 0.630 at t=1), but neither cluster achieves recall ≥ 0.60 at any threshold, so the §8 threshold sweep falls back to max-F1. Resulting admission: ~3 trades (c0) / ~1 trade (c2) across the 5-year, 28-pair, 1,564-trade pool. WFO at Step 6 would see ≤ 1 trade/fold; Steps 5–6 not executed.

c2's D1 AUC growth with t (0.630 at t=1 → 0.711 at t=10) is the strongest single signal-quality finding — path-so-far information has real discriminative power, but the smallest-t rule + threshold-sweep design point combined to bury the deployable hook.

**Calibration items queued (cross-arc session before next out-of-registry arc):**
- **Open-21 (new) — Step 4 deployability gate:** §8 currently passes mechanically when D1 AUC ≥ 0.60 even when the threshold sweep falls back to max-F1 at sub-1% recall. Recommended fix (a): strict-mode threshold sweep — max-F1 fallback triggers cluster-dies, not graceful pass. Alts: (b) recall floor at admit ≥ 0.30; (c) raise D1 AUC floor 0.60 → 0.70.
- **Open-17 expansion — Tiebreak 1 noise floor:** §7 Tiebreak 1 fired at 0.02 ATR / 0.15% relative margin in Arc 6 c2 SL selection (X=3.0 over X=2.0 at identical composite 0.6162). Proposal: require `peak_mfe_atr_margin ≥ 0.10 ATR OR ≥ 1% relative` before Tiebreak 1 applies.
- **(unnumbered note) reach_1R floor noise tolerance:** c3 died at 0.697 vs 0.70 floor (0.003 margin within sampling noise at n=511). Cross-arc question whether floor needs a binomial-noise tolerance; both-sides argument required, not blocking next arc.

Arc 6 also produced a spec v0.2 erratum (`docs/signal_spec_failed_breakout_long_v0.2.md`): the literal v0.1 `swing_low_N = min(low[t-N..t-1])` is mathematically unsatisfiable; corrected to `min(low[t-N-M..t-M-1])`.

Full closure: `docs/arc_results/ARC_6_RESULT.md`. Arc 6 signal is NOT permanently eliminated — path quality is clean at v2.1.2 §2 (c2 Stepwise, mfe_p50 4.47R, ww_pp 0.000) and may return under a richer feature regime, multi-TF entry context, or ensemble approach.

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
| WFO claim retrospective | Pass-viable on real-spread reconciliation (not pass-deployable as originally claimed); live unchanged |

KH-24 is locked and unchanged. No modifications without an explicit modification phase.

---

## Active Research

L arc signal testing under `L_ARC_PROTOCOL.md` v2.1.2 (Arcs 4+). v2.0 governed Arc 3 (closed); v1.x archive at `archive/` for historical Arc 1, Arc 2 reference.

| Item | Detail |
| --- | --- |
| Protocol | `L_ARC_PROTOCOL.md` v2.1.2 (self-contained; v1.x ops spec archived) |
| Signals | Top-N from `docs/LCHAR_TOPN_REGISTRY.md` — 5 arcs scheduled |
| Approach | Six-step pipeline: plumbing → path-shape clustering → capturability → extractability (E or D1) → cross-fold stability → WFO |
| Current arc | Arc 4 — active, LCHAR Entry 4 (`TRIAL__univariate_extreme__bar_range_top_decile__neg__h_001`) |
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
| Arc 4 (l_arc_4) | CLEAN-NULL at Step 5 retroactive (2026-05-17) | bar_range_top_decile__neg__h_001 — first L arc to reach Step 5 PASS; cluster 1 D1 AUC 0.667 pass-deployable at 0.20% risk under modeled spreads; killed by HistData spread audit showing floor file under-models real spreads 3-48x per pair; F6 ann ROI flips from +10.08% to ~−5.6% under real spreads; §9.A retroactive fail; 8 cross-arc calibration items |
| Engine generalisation | DELIVERED (PR #138, 2026-05-17) | Engine config-driven for signal, timeframe, time-exit, spread floor. KH-24 byte-identical (sha256 verified). Arc 4 engine prerequisites complete. |
| Pipeline D1 PR 2 — Stepwise climber policy | DELIVERED (PR #135, 2026-05-17) | §11 row 2 exit policy + per-fold classifier dispatch land. ApplyPolicy made concrete. Trail-suppression resolved against Step 5 simulator. Other §11 rows deferred. |
| Arc 6 (out-of-registry; failed-breakout reversal long) | DIES at Step 4 deployability (2026-05-17) | Steps 1–4 all mechanical PASS. Pipeline E best AUC 0.600 / 0.590 vs 0.65 floor; Pipeline D1 mechanically clears AUC ≥ 0.60 but threshold sweep collapses to max-F1 at sub-1% recall (~3-4 admitted trades / 5-year pool). Two calibration items raised: Open-21 (Step 4 deployability gate, new) + Open-17 expansion (Tiebreak 1 noise floor); plus unnumbered cross-arc note on reach_1R floor noise tolerance. Spec v0.2 erratum: `swing_low_N = min(low[t-N-M..t-M-1])`. See `docs/arc_results/ARC_6_RESULT.md`. |
| L_ARC_PROTOCOL v2.1.2 amendment | LANDED (2026-05-17) | §2 categorical shape_tag floor relaxed to `≠ scattered`; §11 Stepwise local_peaks ceiling 5-30 → 5-50; §15a arc Step 1 schema requirement (Open-19 closure). Open-15, Open-18, Open-19 closed. Open-20 reframed as Step 4 measurement (not a §2 gate). Anchor preservation verified. |
| Open-18 cross-replay synthesis | COMPLETE (2026-05-17) | 3/3 replays passed: Arc 3 Stepwise (c2 + aggregate), KH-24 v2.0 c4 (c1 + c4), Arc 2 redo2 cid 1. Pre-peak Def C validated as dominant rescue mechanism (38% → 0-2% wrong_way across cohorts). bimodal_separated validated narrowly. See `results/replays_v2_1_1/` for full evidence. |
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
| `configs/spread_floors_5ers.yaml` replacement | HIGHEST | Per-pair empirical floors from HistData audit at p10. Locked-file change. SPREAD_SEMANTICS_LOCK governance required. Blocks all future arcs. |
| Phase Zero spread validation | HIGH | Add to L arc workflow before Step 1 plumbing; refresh tick audit every 6 months. |
| MT5 vs HistData spread cross-check | MEDIUM | Dump 5 days of 5ers MT5 spread snapshots from Contabo VPS, validate audit assumptions. |
| §11 row policies for rows 1, 3, 4, 5, 6, 7 | DEFERRED | Row 2 (Stepwise climber) shipped in PR #135. Other rows added when an arc surfaces a Step-4 archetype consumer needing them. No build until concrete consumer surfaces. |

No outstanding bugs or issues against KH-24. No pending fixes against the backtester. §14 anchor refresh explicitly held (deployed-pop reference, no change).

---

## Watch Items

| Item | Status | Notes |
| --- | --- | --- |
| §2 calibration pattern | ADDRESSED in v2.1.2 | The §2 monotonicity / shape_tag wall seen at 2026-05-16 closures (KH-24 c4, Arc 2 redo c2, Arc 3 Stepwise) was empirically resolved by Open-18 replays under v2.1.1 (pre-peak Def C + capturability composite) and architecturally addressed by v2.1.2 (`≠ scattered` floor + Stepwise ceiling extension). 3/3 arcs hit §2-related issues now; Arc 4 escalates to a different failure mode (cost-model invalidity surfaces real edge as undeployable). Spread floor finding is bigger than §2 calibration — affects every prior closure's magnitude (not direction). |

---

## Cross-Arc Calibration Backlog (post-Arc-5 review)

Items accumulating from arc closures under v2.0. Per §1.8 within-arc thresholds do not move; per §12 cross-arc calibration is governed and requires a calibration document + chat-level approval. The 2026-05-17 v2.1 amendment resolved or partially resolved most of the items below; items still requiring evidence / execution are kept under "Active backlog".

### Resolved in v2.1 / v2.1.1 / v2.1.2 amendments

| Item | Source arc(s) | Resolution |
| --- | --- | --- |
| §17 `frac_wrong_way` definition | KH-24 v2.0 | CLOSED in v2.1 — §17 Def C ratified (MAE ≤ −1R on or before peak_mfe_bar) |
| §16 Open-08 closure | KH-24 v2.0 | CLOSED 2026-05-17 — `pullback_magnitude_median` empirically non-degenerate; §16 marked resolved |
| Open-12 silhouette tie tolerance | Arc 3 | CLOSED in v2.1 — §6 tolerance 0.01 absolute silhouette |
| Open-13 §2/§11 row-7 bimodal incompatibility | Arc 3 | CLOSED in v2.1 — `bimodal_separated` admitted at §2 under Hartigan dip + mass + separation test; routes to §11 row 7 |
| Open-14 same-archetype aggregation rule | Arc 3 | CLOSED in v2.1 — §7 evaluates per-cluster AND per-aggregate; cluster proceeds if either passes |
| Open-15 SL/horizon asymmetry inflating wrong_way | Arc 3 | CLOSED in v2.1 / v2.1.1 — addressed via SL-free measurement (uses v1.3 forward-window extension already in engine; no PR needed) + per-archetype SL sweep at §7 (capturability composite selection under v2.1.1); empirically confirmed across three Open-18 replays in v2.1.2 |
| 240-bar forward window for 4H signals | KH-24 v2.0 | CLOSED in v2.1 — §5 forward window auto-extend at >20% pool-level cap-binding (2× extension default) |
| Per-pair n distribution stability concern | KH-24 v2.0 | CLOSED in v2.1 — §9 per-pair stability reporting added at Step 5 (informational; flags > 50% concentration in < 5 pairs) |
| §2 monotonicity floor (0.55) calibration | KH-24 v2.0, Arc 2 redo | RESOLVED in v2.1.2 — pre-peak measurement subsumes the full-window bias; three Open-18 replays confirm empirically. Floor unchanged; near-miss failure mode no longer reproduces. |
| shape_tag definitions vs forward-window censoring | KH-24 v2.0 | PARTIALLY RESOLVED in v2.1.2 — §2 categorical floor relaxed to `≠ scattered`; cap-binding correlation diagnostic carried forward as Watch / Active backlog item below. |
| §14 anchor population vs §15 pool floor mismatch | KH-24 v2.0 | HELD — user decision: deployed-pop reference unchanged. KH-24 v2.0 c4 replay produced c1/c4 candidates but no refresh applied. v2.1.2 anchor preservation verified under the existing v2.0 reference. |

### Active backlog (still open or partial)

| Item | Source arc(s) | Priority | Notes |
| --- | --- | --- | --- |
| §11 archetype priors empirical refinement (Open-07) | KH-24 v2.0, Arc 2 redo, Arc 3 | MEDIUM | v2.1 demoted §11 SL column to prior; v2.1.2 extends Stepwise local_peaks ceiling 5-30 → 5-50 based on Open-18 empirical centroids. Other §11 rows still first-pass priors; centroid-pattern refresh remains open. Deferred until Arc 4 + Arc 5 produce additional evidence. |
| Cap-binding / shape_tag dead-zone diagnostic | KH-24 v2.0 c4 replay | LOW | Replay #2 surfaced correlation between p95/p50 dead-zone (2.0, 3.0] and forward-window cap-binding. v2.1.2's `≠ scattered` floor avoids over-rejecting on this; §5 auto-extend addresses upstream. Track whether the correlation resurfaces under Arc 4+. |
| Real-spread floor file replacement | Arc 4 | HIGHEST | Per-pair empirical floors from HistData audit (or MT5 broker snapshot) required to replace uniform 0.1 pip floor that under-models real spreads 3-48x. Locked-file change under SPREAD_SEMANTICS_LOCK.md governance. Blocks all future arc work. |
| Spread audit as Phase Zero | Arc 4 | HIGH | All future L arcs must validate spread floor file against current broker quote before Step 1 plumbing runs; refresh tick audit every 6 months. Affects WORKFLOW.md v2 and `L_ARC_PROTOCOL.md` §5. |
| Session-aware spread modeling | Arc 4 | MEDIUM | Per-pair × per-session floors may be required for accurate cost modeling on cross-pair signals. Defer until per-pair floor in place and next arc's behaviour observed. |
| Step 5 simulator default — apply exit spread | Arc 4 | MEDIUM | Post-hoc simulator templates should enforce S/2 exit spread by default per SPREAD_SEMANTICS_LOCK.md. Arc 4's omission was prompt-author error; the simulator template should make it impossible to skip. |
| Convention (b) MTM DD as §10 default | Arc 4 | MEDIUM | 5ers measures account equity in real-time; convention (a) closed-trade ordering understates DD by 14-63%. Convention (b) should become §10's default gate metric. Affects protocol §9 and §10 wording. |
| F1 structural leakage | Arc 4 | HIGH | L arc pool starts 2020-10-01 = F1 OOS start. No honest WFO training data exists for F1. Affects every L arc retroactively (magnitude not direction). Options: pool back-extend to pre-2020, drop F1 from L arc evaluation, or alternate fold structure for L arcs. |
| D1 threshold grid specification | Arc 4 | MEDIUM | §3 locks Pipeline E grid {0.40, 0.50, 0.60, 0.70} but never explicitly locks D1's grid. Both Arc 4 and Arc 5 hit this. Lock D1 grid as {base_rate, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50} in protocol amendment. |

---

## Permanently Eliminated (post-KH closure + post-L6.0)

- KH-25 re-entry exposure cap hypothesis (KH-27 KILL)
- KH-24 fold 7 entry-side regime selection (KH-28 STRUCTURAL)
- KH-24 fold 7 exit-side indictment (KH-29 AMBIGUOUS)
- KI arc: 1H timeframe port of KH-24 signal (mean R 0.004, t=0.095)
- L6.0 verbatim-as-gate framing as arc disposition rule (replaced by extractability protocol)

Earlier eliminations (KGL_V2 era, JL forward bias, NNFX, exit indicator sweeps, etc.) recorded in `CLAUDE.md`, `docs/KH_Research_Roadmap.md`, and `project_brief.md`.

Note: Arc 2 signal (mtf_alignment.2_down_mixed.kijun, h=120) shelved 2026-05-16 as "real edge, not extractable under v2.0 as drawn" — see `results/l_arc_2_redo/ARC_2_REDO_RESULT.md`. Not permanently eliminated; reopenable contingent on v2.x calibration amendment.

Note: `bar_range_top_decile__neg__h_001` (Arc 4 signal) is SHELVED pending spread floor calibration. Not permanently eliminated — re-evaluatable under corrected spread file. See `docs/arc_results/ARC_4_RESULT.md`.

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
| Arc 4 (closed CLEAN-NULL on transaction-cost truth) | `results/l_arc_4/` + `docs/arc_results/ARC_4_RESULT.md` + `docs/SPREAD_FLOOR_AUDIT_FINDING.md` |
| Arc 6 (closed DIES at Step 4) | `results/arc_6/` (on `discovery/lomega_regime_conditional`) + `docs/arc_results/ARC_6_RESULT.md` |

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
