# SESSION ZERO — Forex Ignition Rebuild
> 5-minute primer. Read this first, then read `L_ARC_PROTOCOL.md` (v2.1.1, self-contained) for the active research methodology.
> Last updated: 2026-05-17 — L_ARC_PROTOCOL v2.1.1 amendment landed (v2.1 refinements + engine-reality corrections). PR #131 D1 backtester plumbing merged. Previously planned engine PR for SL-free path recording superseded — v1.3 forward extension already provides the required SL-free observation. Next: Open-18 empirical replays of v2.0-killed cohorts (runnable on existing `trades_paths.csv`), then Arc 4 under v2.1.1.

---

## Pointers

- Active protocol: `L_ARC_PROTOCOL.md` (v2.1.1)
- v1.x archive: `archive/`

---

## Current State

**ACTIVE WORK: L ARC SIGNAL TESTING UNDER L_ARC_PROTOCOL v2.1.1 — v2.1.1 AMENDMENT LANDED 2026-05-17, OPEN-18 REPLAYS NEXT, THEN ARC 4**
**LIVE SYSTEM: KH-24 (locked, unchanged, running on VPS)**

Three arcs closed on 2026-05-16, all at Step 3 §2 floors: Arc 2 redo (KILL), KH-24 v2.0 self-test (HALT), Arc 3 (CLEAN-NULL). KH-24 v2.0 self-test was a protocol self-test on the bare KH-24 signal — its closure surfaces the same §2 monotonicity / shape_tag failure mode seen on Arc 2 redo on a different signal with a different mechanism (forward-window censoring of the trend-rider cohort). The §14 calibration anchor was measured on KH-24's filtered deployed population (not bare signal), so v2.0 cannot self-validate on its own anchor as drawn. Cross-arc calibration backlog grew by 8 items, with §2 monotonicity floor and shape_tag-vs-censoring as the top priorities. Open-08 is closed as resolved (pullback_magnitude_median operational definition empirically non-degenerate on KH-24 paths). Arc 3 closure doc flags three reviewer items — Stepwise climber opportunity (passes 4/6 §2 floors cleanly), aggregation rule destroying capturable sub-clusters, SL/horizon asymmetry inflating wrong_way. Five cross-arc items logged for v2.1 from Arc 3. Diagnostic tail (Arc 3D) recommended before Arc 4 — reviewer decision pending. Results: `docs/arc_results/ARC_3_RESULT.md`, `results/arc_kh24_v2/ARC_KH24_V2_RESULT.md`, `results/l_arc_2_redo/ARC_2_REDO_RESULT.md`.

Pipeline D1 backtester extension is in progress in a separate chat — value independent of these closures, carries forward to all future D1-routed archetypes.

Pattern flag: two-of-three closures hit the same §2 monotonicity / shape_tag wall with structurally credible cohorts (KH-24 v2.0 c4: fwd_mfe_p50 6.65R, frac_reach_1R 1.000, frac_wrong_way 0.000, missed mono by 0.020; Arc 2 redo c2: t-stat +52.17, missed mono by 0.009). If Arc 4 closes the same way, §2 calibration moves from "watch item" to blocking the protocol — post-Arc-5 calibration review has concrete inputs.

Next: Arc 4 — registry Entry 4 (1-bar horizon, univariate-extreme family, structurally different from Arc 3).

As of 2026-05-17, `L_ARC_PROTOCOL.md` v2.1.1 is the active protocol. v2.1.1 combines v2.1's refinements (substantive: §7 capturability composite SL selection, §5 Step 2 re-clustering on window-extend, §11 row 7 routing precedence, §10 multi-cluster ship decision) with engine-reality corrections that supersede v2.1's references to a pending SL-free path-recording engine PR. CC's halt while attempting that PR established that `scripts/phase_kgl_v2_4h_wfo.py::_flatten_bar_path_for_trade` (v1.3) already emits `is_held=0` forward observation bars from entry to entry+240 regardless of exit reason — providing the SL-free continuation §7 SL sweep needs. v2.1.1 corrects §1/§5/§7/§14/§16/§17 wording accordingly; no engine change implied or needed. The `feat/sl-free-path-recording` branch is superseded and deleted. Open-18 empirical replays (Arc 3 Stepwise climber first, then KH-24 v2.0 c4 for anchor refresh, then Arc 2 redo c2) are runnable now on existing `trades_paths.csv`. KH-24 K=4 archetype 3 remains the calibration anchor with refresh pending re-run only. Arcs 1 and 2 ran under v1.x and are historical; v2.0 governed Arc 3; v2.1.1 governs Arc 4 onward.

### Live system

| Item | Value |
| --- | --- |
| System | KH-24 (baseline signal + exposure cap=2 + 1H CIR T=0.28) |
| Spec | `docs/KH24_SYSTEM_LOCK.md` |
| Config | `configs/wfo_kh24.yaml` (locked, do not modify) |
| Results | `results/kh24/` |
| EA | `KH24_EA.mq5` v2.01 deployed on Contabo VPS |
| Broker | 5ers |
| WFO gate | PASS — worst-fold ROI +1.92% (F7), worst-fold DD 6.37% (F1), all 7 OOS folds positive |
| OOS trades | 214 across Oct 2020 – Jan 2026 |

KH-24 is locked and out of scope for L arc work. No modifications without an explicit modification phase.

### Active research direction

**L arc signal testing, Arc 4 next under v2.1.1.** Source of truth: `L_ARC_PROTOCOL.md` v2.1.1 — self-contained for methodology, deliverables, gates, and workflow. v1.x ops spec is archived for historical reference.

The v2.1.1 L arc tests each registry signal through a six-step pipeline:
1. Plumbing (deterministic full-pool generation; pool ≥ 500)
2. Path-shape clustering (outcome-blind features: monotonicity, local_peaks, pullback, time_to_peak_rel)
3. Capturability characterisation (per §2 hard floors: clean shape + meaningful magnitude)
4. Extractability + artefact production (Pipeline E entry-filter AUC ≥ 0.65 OR Pipeline D1 deferred-policy AUC ≥ 0.60)
5. Cross-fold stability (sign consistency, size variance, DD ceiling)
6. WFO truth + pass-deployable / pass-viable gate

Arcs 1 and 2 are historical (ran under v1.x). v2.0 governed Arc 3; v2.1.1 governs Arc 4 onward. Calibration anchor: KH-24 K=4 archetype 3, which passes v2.0 extractability via Pipeline D1 at t=3 (RF AUC 0.638, exclusion 15.4%). Anchor refresh under v2.1.1 metrics pending KH-24 v2.0 self-test re-run only (no engine work needed — v1.3 forward extension already provides SL-free observation).

### Tool assignments (unchanged)

- This chat: strategy, research interpretation, decisions, step 3 verdicts, step 4 candidate selection
- Claude Code: multi-file features, atlas computation scripts, WFO runs, cluster fits, predictor scans
- Cursor: single-file patches, YAML edits, doc updates
- GPT-4 and Aider: permanently excluded

### Non-negotiable rules (unchanged)

- `build_ex_ante_bounded_population` — mandatory
- Signal bar N close → entry bar N+1 open — absolute
- D1 one-day lag — never revert
- No lookahead anywhere
- WFO worst-fold is the only judge at step 6
- One change per phase, pre-committed gate, accept the result
- Per-bar MT5 spread data — never hardcoded defaults
- Risk = 0.5% of reset floor balance (L6 convention)
- Clean labels = evaluation tool only, never used for population selection
- KH-24 configs (`wfo_kh24.yaml`, `wfo_baseline_clean.yaml`): never modify
- Every phase gets a result document regardless of pass or fail
- Full distributions reported for every metric (no medians-only)
- All result docs go in `results/<arc_name>/` per `WORKFLOW.md` v2 folder convention

---

## What the System Is

KH-24 is a long-only, trend-pullback system on 28 FX pairs, 4H timeframe. It identifies bearish exhaustion bars within D1 bullish trends and enters long on the following bar. The edge comes from trailing stop exits on trades that develop into sustained trends.

Full specification: `docs/KH24_SYSTEM_LOCK.md` — read this before writing any MQL5 code or modifying the live system.

The L arc is methodologically distinct from KH-24 development. It does not assume continuation of KH-24's signal class, timeframe, direction, or pair-set decisions. Each L arc tests a registry signal on its own merits.

---

## How We Got Here (Short Version)

**KGL_V2 era (Sep 2025 – Apr 2026).** First WFO-passing system on FTMO data. Switched to 5ers broker. C7 volume gate failed on 5ers tick volume. C7 removed, system re-validated.

**KH arc (Apr 2026).** Sequential refinement. KH-22 added exposure cap=2. KH-24 added 1H CIR T=0.28. KH-24 was the first configuration to pass the WFO gate across all 7 OOS folds. Locked and deployed live on 2026-04-20.

**Post-lock diagnostic arc (May 2026).** Three diagnostic phases (KH-27, KH-28, KH-29) established that KH-24's +1.92% worst-fold ROI is the structural ceiling for that signal. KH arc closed; pivot to L arc.

**L characterization arc (May 2026).** Bottom-up exploratory atlas across four layers (univariate, multi-timeframe, cross-pair, conditional). Output: `docs/LCHAR_TOPN_REGISTRY.md` listing top-N candidate signals ranked by deflated Sharpe ratio.

**L6 signal-testing arc (May 2026).** Initial L6.0 methodology lock used verbatim-as-gate framing. Arcs 1 and 2 ran and FAILED verbatim WFO (DD 39%–91%) — expected outcome of the framing, but the disposition rule closed those arcs prematurely. Arc 1 P2 (with concurrent_signals_within_3h filter, CH-001) passed under that framing; the work proved real edge existed but the protocol needed redesign.

**L arc protocol redesign (2026-05-13).** `L_ARC_PROTOCOL.md` v1.0 locked. Replaces verbatim-as-gate with six-step extractability protocol. Includes PASS-DEPLOYABLE vs PASS-VIABLE tiers, effect-size + AUC dual-gate verdict, post-arc routing, Arc 1 redo as calibration check. `archive/L_ARC_OPERATIONAL_SPEC_v1_0.md` specifies deliverables, angle catalogues, scoring tables. (v1.x superseded by v2.0 on 2026-05-16; both archived.)

---

## Phase History

*Note: when applying SESSION_ZERO updates, preserve any pre-existing Phase History entries below this line. New entries are appended at the top. The full list is what remains in the file.*

### 2026-05-16 — KH-24 v2.0 self-test arc closed (HALT at Step 3)

`results/arc_kh24_v2/ARC_KH24_V2_RESULT.md` written. Protocol self-test on the bare `kb_exhaustion_bar` signal (C1-C6, C8, C9; long-only; 4H; 1R = 2 × ATR(14); 240-bar forward window) — opened and closed same day. Step 1 PASS (pool 842, deterministic, no lookahead). Step 2 PASS (K=5, silhouette 0.4327, five clusters). Step 3 FAIL — zero of 5 clusters cleared §2 conjunctively → arc halts per §7 (STEP3_FAIL_NO_CAPTURABLE_ARCHETYPE). Best contender c4 (trend-rider cohort, n=122, 14.5% of pool): fwd_mfe_p50 6.65R, frac_reach_1R 1.000, frac_wrong_way 0.000 — missed monotonicity floor by 0.020 (mono=0.530 vs 0.55) and shape_tag was `scattered` because 87.7% of c4 trades hit the 240-bar forward-window cap, censoring final_r distribution. Two technical failures on a structurally-edge-carrying cohort. The §14 calibration anchor (mono 0.576) was measured on KH-24's deployed 214-trade population — already filtered through 1H CIR + currency cap — so v2.0 cannot mechanically rediscover, on bare signal, what v1.0's hand-crafted filters extracted. KH-24 v1.0 deployment unaffected. 8 cross-arc calibration items logged for post-Arc-5 review (top: §2 monotonicity floor, shape_tag censoring, 240-bar window, §14-vs-§15 population mismatch, §17 frac_wrong_way disambiguation). Open-08 closed as resolved (pullback_magnitude_median operational definition empirically non-degenerate). Pipeline D1 backtester extension work in separate chat continues. 77/77 CI tests passing.

### 2026-05-16 — Arc 3 closed CLEAN-NULL at Step 3 (with reviewer flags)

`TRIAL__volatility_regime__d1_atr_top_decile__any__h_120` taken through v2.0 Steps 1-3. Step 1 PASS (2568 trades). Step 2 PASS (K=7). Step 3 FAIL — zero archetypes pass §2. Stepwise climber (27.5%, n=707) is the closest call: passes mono 0.559, mfe_p50 3.34R (2.2× floor), reach_1R 83.6%, size; killed only by shape_tag=bimodal (excluded by §2 despite §11 row 7 defining the exit policy) and wrong_way 38.3%. Median final_r +1.85R. Three reviewer flags in closure doc. Five v2.1 cross-arc items: Open-12, Open-13 (highest priority), Open-14, Open-15, Open-07 evidence. Arc 3D diagnostic tail recommended. Result: `docs/arc_results/ARC_3_RESULT.md`.

### 2026-05-16 — Arc 2 redo CLOSED: KILL at Step 3

`results/l_arc_2_redo/ARC_2_REDO_RESULT.md` written. Arc 2 redo opened and closed same day. Step 1 (plumbing, pool 12,262) and Step 2 (K=4 clustering, silhouette 0.4778) passed clean. Step 3 capturability failed for all 4 archetypes — cluster 2 (Stepwise climber) carried overwhelming forward magnitude (fwd_mfe_p50 5.83R, t-stat +52.17, n=2,278) but missed monotonicity floor by 0.0086 and frac_wrong_way ceiling by 0.0051, plus failed shape_tag. v2.0 protocol working as designed — strong magnitude on oscillatory paths is exactly what §2 floors are designed to filter. Cross-arc carryover: Open-09 evidence, shape_tag definition pressure for high-magnitude cohorts. Arc 3 unaffected and continues. Methodology cross-validation: v1.x verbatim WFO and v2.0 path-shape gate both close FAIL on this signal via different mechanisms — strong evidence the signal is intractable, not a methodology artefact.

### 2026-05-16 — L_ARC_PROTOCOL v2.0 locked; Arc 3 opens under v2.0

`L_ARC_PROTOCOL.md` v2.0 locked. Replaces v1.0 + amendments as active protocol for Arcs 3+. Structural rewrite: clustering basis shifts from forward-geometry magnitude to outcome-blind path-shape; single entry-filter pipeline replaced by two-pipeline structure (E entry-filter + D1 deferred-identification); two-gate sequencing (capturability §2 → extractability §2); calibration anchor KH-24 K=4 archetype 3 (passes via Pipeline D1 at t=3); documentation shifts from per-step result docs to one live arc doc per protocol §13; workflow shifts to direct-to-main for analysis with PRs for engine changes per §13. v1.0 + amendments archived at `archive/L_ARC_PROTOCOL_v1_0.md` etc. for Arcs 1, 2 historical reference. Evidence base: PR #129 (archetype diagnostic), PR #130 (predictability investigation). Governance docs (CLAUDE.md, STATUS.md, SESSION_ZERO.md) updated.

### 2026-05-13 — L_ARC_PROTOCOL v1.0 locked; Arc 1 redo opens

`archive/L_ARC_PROTOCOL_v1_0.md` (v1.0) and `archive/L_ARC_OPERATIONAL_SPEC_v1_0.md` (v1.0) locked. Supersedes L6.0 §9 (no filter rescue, verbatim-as-gate) and §14 disposition rules; L6.0 feature schema (§14.3) and pair-set / WFO structure carry forward. Arc 1 and Arc 2 reopened for redo under the new protocol. Arc 1 redo doubles as the protocol calibration check — `concurrent_signals_within_3h` must surface as ≥ Tier 2 predictor in step 3, or the protocol is miscalibrated. WORKFLOW.md v2: phase docs now co-located in `results/<arc>/` permanently. CLAUDE.md, README.md, STATUS.md updated to point at new protocol. (v1.x in turn superseded by v2.0 on 2026-05-16; both archived.)

### 2026-05-09 — L characterization arc closed; L6 signal-testing opened (then redesigned)

L arc atlas complete. `docs/LCHAR_TOPN_REGISTRY.md` produced top-N candidate signals ranked by DSR. L6.0 methodology lock written. Arcs 1 and 2 run under L6.0; both FAILed verbatim WFO. Arc 1 P2 (CH-001 concurrent_signals filter) passed under L6.0 framing — confirms real edge. Subsequent re-planning concluded the verbatim-as-gate framing was structurally wrong; protocol redesigned, see 2026-05-13 entry.

### 2026-05-09 — KH arc closed

Three diagnostic phases (KH-27 KILL, KH-28 STRUCTURAL, KH-29 AMBIGUOUS) closed the question of whether KH-24's fold 7 weakness is addressable. Combined verdict: structural ceiling. KH-24 stays live unchanged.

*[Earlier phase history entries from prior SESSION_ZERO versions remain below — preserve verbatim.]*
