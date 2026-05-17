# SESSION ZERO — Forex Ignition Rebuild
> 5-minute primer. Read this first, then read `L_ARC_PROTOCOL.md` (v2.1.2, self-contained) for the active research methodology.
> Last updated: 2026-05-17 — Arc 6 (failed-breakout reversal long, out-of-registry) closed DIES at Step 4 deployability. Open-21 (new) + Open-17 expansion queued for cross-arc calibration before next out-of-registry arc. Arc 7 active.

---

## Pointers

- Active protocol: `L_ARC_PROTOCOL.md` (v2.1.2)
- v1.x archive: `archive/`

---

## Current State

**ACTIVE WORK: L ARC SIGNAL TESTING UNDER L_ARC_PROTOCOL v2.1.2 — ARC 6 (OUT-OF-REGISTRY) CLOSED 2026-05-17 DIES AT STEP 4 DEPLOYABILITY; ARC 7 ACTIVE**
**LIVE SYSTEM: KH-24 (locked, unchanged, running on VPS)**

Three arcs closed on 2026-05-16 (Arc 2 redo KILL, KH-24 v2.0 self-test HALT, Arc 3 CLEAN-NULL) all hit the same §2 monotonicity / shape_tag wall with structurally credible cohorts. v2.1 (2026-05-17) introduced pre-peak measurement + SL sweep + capturability composite + bimodal_separated admit. v2.1.1 (2026-05-17) refined SL selection to capturability composite maximiser.

Open-18 cross-replay synthesis closed 2026-05-17. Three parallel replays under v2.1.1 all PASS:
- Replay #1 Arc 3 Stepwise: c2 individual + aggregate PASS; pre-peak Def C 38.3% → 1.7% wrong_way
- Replay #2 KH-24 v2.0 c4: c1 + c4 both PASS as unresolved near-stepwise
- Replay #3 Arc 2 redo2 cid 1: PASS at SL=3.0, composite 0.593; pre-peak Def C 34% → 0%

Synthesis verdict: pre-peak Def C is the dominant rescue mechanism across all three. bimodal_separated validated narrowly (one positive: Arc 3 c2). v2.1.2 amendment (landed 2026-05-17) addresses the §2 categorical floor + §11 Stepwise ceiling gaps the replays surfaced:
- §2 internal-consistency floor: `∈ {tight_unimodal, heavy_right_tail, bimodal_separated}` → `≠ scattered`
- §11 Stepwise local_peaks ceiling: 5-30 → 5-50
- §15a added: arc Step 1 schema requirement (Open-19 closure)
- Open-15, Open-18, Open-19 closed

§14 anchor not refreshed (user decision — deployed-pop reference holds; v2.1.2 anchor preservation verified). Open-20 (raised in Arc 2 redo2 closure) not added to protocol — reframed as Step 4+ measurement question (high-pct_peak_and_collapse cohorts measured under trailing-stop exit, not fixed-SL re-imposition).

Arc 4 active under v2.1.2 on LCHAR Entry 4 (`TRIAL__univariate_extreme__bar_range_top_decile__neg__h_001`) — 1-bar horizon, univariate-extreme family, structurally different from Arc 3.

**Arc 6 (out-of-registry; failed-breakout reversal long) closed 2026-05-17 DIES at Step 4 deployability.** Steps 1–4 ran consecutively on `discovery/lomega_regime_conditional`; all four mechanical PASSes. Pipeline E failed both Step 3 survivors (best AUC 0.600 / 0.590 vs 0.65 floor). Pipeline D1 mechanically cleared AUC ≥ 0.60 (c0 0.602 at t=4; c2 0.630 at t=1) but neither cluster achieves recall ≥ 0.60 at any threshold, so §8 threshold sweep collapses to max-F1 — ~3 admitted trades (c0) / ~1 trade (c2) across the 5-year, 1,564-trade pool. Steps 5–6 not executed. Two calibration items raised: Open-21 (Step 4 deployability gate, new) + Open-17 expansion (Tiebreak 1 noise floor); plus an unnumbered cross-arc note on reach_1R floor noise tolerance. Spec v0.2 erratum (`docs/signal_spec_failed_breakout_long_v0.2.md`): `swing_low_N = min(low[t-N-M..t-M-1])`. Signal NOT permanently eliminated — path quality clean, may return under richer feature regime / multi-TF / ensemble. Full closure: `docs/arc_results/ARC_6_RESULT.md`.

**Arc 7 active** on `phase/l_arc_7`. Step 1 PASS (1288 trades), Step 2 PASS (K=4, sil=0.4263), Step 3 PASS (3 units survive §2: c1, c3, agg_c1_c3 all V-shape recovery family).

Results: `results/replays_v2_1_1/arc_3_stepwise/REPLAY_RESULT.md`, `results/replays_v2_1_1/kh24_v2_c4/REPLAY_RESULT.md`, `results/l_arc_2_redo2/`, `docs/arc_results/ARC_3_RESULT.md`, `results/arc_kh24_v2/ARC_KH24_V2_RESULT.md`, `results/l_arc_2_redo/ARC_2_REDO_RESULT.md`.

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

**L arc signal testing, Arc 4 active under v2.1.2.** Source of truth: `L_ARC_PROTOCOL.md` v2.1.2 — self-contained for methodology, deliverables, gates, and workflow. v1.x ops spec is archived for historical reference.

The v2.1.2 L arc tests each registry signal through a six-step pipeline:
1. Plumbing (deterministic full-pool generation; pool ≥ 500)
2. Path-shape clustering (outcome-blind features: monotonicity, local_peaks, pullback, time_to_peak_rel)
3. Capturability characterisation (per §2 hard floors: clean shape + meaningful magnitude)
4. Extractability + artefact production (Pipeline E entry-filter AUC ≥ 0.65 OR Pipeline D1 deferred-policy AUC ≥ 0.60)
5. Cross-fold stability (sign consistency, size variance, DD ceiling)
6. WFO truth + pass-deployable / pass-viable gate

Arcs 1 and 2 are historical (ran under v1.x). v2.0 governed Arc 3; v2.1.2 governs Arc 4 onward. Calibration anchor: KH-24 K=4 archetype 3, which passes v2.0 extractability via Pipeline D1 at t=3 (RF AUC 0.638, exclusion 15.4%). Anchor not refreshed from Open-18 replays per user decision — deployed-pop reference holds. v2.1.2 anchor preservation verified (centroid still routes to Stepwise under 5-50 local_peaks range; bimodal shape_tag passes `≠ scattered`).

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

### 2026-05-17 — Arc 6 (out-of-registry; failed-breakout reversal long) closed DIES at Step 4 deployability

`docs/arc_results/ARC_6_RESULT.md` written. Arc 6 dispatched out-of-registry on `discovery/lomega_regime_conditional`; opened and closed same day. Signal under test: failed-breakout reversal at swing low (long-only, 4H, 28 FX pairs). Steps 1–4 ran consecutively; all four mechanical PASSes. Step 1 PASS (pool 1,564; det e57528...; KH-24 co-fire 0.0000 by structural exclusion; cap-bind 17.65%; spec erratum locked — `swing_low_N = min(low[t-N-M..t-M-1])`). Step 2 PASS (K=4, silhouette 0.4795; 4 clusters labelled across early_peak / Stepwise / two unassigned). Step 3 PASS 2/4 (c2 Stepwise SL=3.0×ATR composite 0.616; c0 Stepwise-boundary SL=2.0×ATR composite 0.384). Step 4 PASS mechanically / FAIL substantively: Pipeline E fails both clusters (best AUC 0.600 c0, 0.590 c2; floor 0.65); Pipeline D1 clears AUC ≥ 0.60 floor (c0 t=4 RF AUC 0.602; c2 t=1 RF AUC 0.630) but neither achieves recall ≥ 0.60 at any threshold, so §8 sweep falls back to max-F1 — c0 precision 0.333 recall 0.009 (~3 trades), c2 precision 0.250 recall 0.004 (~1 trade). WFO would see ≤ 1 trade/fold; Steps 5–6 not executed. c2 D1 AUC grows monotonically with t (0.630 t=1 → 0.711 t=10) — path-so-far information has real discriminative power, the strongest single signal-quality finding in Arc 6. Calibration items queued: **Open-21 (new)** — §8 Step 4 deployability gate: strict-mode threshold sweep recommended (max-F1 fallback = cluster-dies, not graceful pass); **Open-17 expansion** — Tiebreak 1 noise floor (Arc 6 c2 selection flipped X=2.0→3.0 on a 0.02 ATR / 0.15% relative margin at identical composite 0.6162); plus unnumbered cross-arc note on reach_1R floor noise tolerance (c3 died 0.697 vs 0.70 by 0.003, within sampling noise at n=511). Signal NOT permanently eliminated — path quality clean (c2 mfe_p50 4.47R, ww_pp 0.000); may return under richer feature regime, multi-TF entry context, or ensemble approach. Commits: b178ea4 (Step 1), fd660ab (Step 2), f56b945 (Step 3), 8648c1b (Step 4) on `discovery/lomega_regime_conditional`. Arc 7 unaffected and continues on `phase/l_arc_7`.

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
