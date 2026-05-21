# CLAUDE.md — Forex Ignition Rebuild
> Last updated: 2026-05-19 | Phase: Arc 10 closed STEP_4_HALT on `claude/charming-mcnulty-8160e0`; post-closure research (experimentation + WFO pair) complete and committed. Arcs 8/9/11 run in parallel CC sessions (queue ownership delegated to those sessions). KH-24 live deployment unchanged.
> First file any AI assistant reads. Reflects where the project ACTUALLY is.

---

## Active protocol

L_ARC_PROTOCOL v2.0 (`L_ARC_PROTOCOL.md`) governs Arcs 3+. Path-shape clustering + two-pipeline (E entry-filter / D1 deferred-identification) extractability gate. Calibration anchor: KH-24 K=4 archetype 3 (passes via Pipeline D1 at t=3). Next engine PR: Pipeline D1 backtester extension (conditional exits at bar N).

Historical: Arcs 1, 2 ran under v1.0 protocol (`archive/L_ARC_PROTOCOL_v1_0.md` + v1.1/v1.2 amendments).

---

## Read These First, In Order

1. **`L_ARC_PROTOCOL.md`** — methodology of record for all L arc signal-testing work. Locked v2.0. Self-contained: deliverables, gates, exit-family map, and workflow all live in the v2.0 doc.
2. **`SESSION_ZERO.md`** — 5-minute primer on current state.
3. **`STATUS.md`** — tight current-state snapshot.

For v1.x historical reference (Arcs 1, 2): `archive/L_ARC_PROTOCOL_v1_0.md`, `archive/L_ARC_OPERATIONAL_SPEC_v1_0.md`, plus v1.1/v1.2 amendments in the same folder.

Then, depending on scope:
- Touching the live system → `docs/KH24_SYSTEM_LOCK.md`
- Running an L arc step → the arc's `PHASE_L_ARC_N_OPEN.md` under `results/l_arc_N/`
- Phase close → `WORKFLOW.md` v2

---

## CRITICAL: Current State

**Live system KH-24 is locked, passing, deployed.** Out of scope for L arc work; do not modify without an explicit modification phase.

**Active research: L arc signal testing under `L_ARC_PROTOCOL.md` v2.1.2 base + v2.2 + v2.3 amendments.** Arc 10 (DLR — D1 swing-low rejection long) closed `STEP_4_HALT` 2026-05-18 per §16a Path A. First arc to run end-to-end under v2.3 (5-step pipeline, halt at end of Step 4). Step 4 near-miss: c1 V-shape recovery E AUC 0.6296 (margin −0.0204), D1 AUC 0.5897 (margin −0.0103); disjunctive §8 fails on both. Post-closure experimentation pass (EXP-01–06) and WFO pair (base + oracle c1) conducted as research probes over §16a at chat-side direction. Oracle WFO Sharpe 4.61 vs base −1.29 (gap +5.90) — synthesis recommends BUILD clusterifier; explicit DO NOT DEPLOY. EXP-01 bootstrap shows realisable classifier P(AUC ≥ 0.65) = 12.5%. Closure: `docs/archive/arc_results/ARC_10_RESULT.md`. Two material corrections to prior framing: Arc 6 reclassified Stepwise (not V-shape, per EXP-05); fold-2 date 2023-07 → 2024-06 (not Q2 2022, per EXP-04). Open-06 (AUC threshold) weakened; Open-04 (external features) deferred pending Arc 8/9/11. Arc 4 RE-RUN closed FAIL Step 6 2026-05-18 — Pipeline D1 reject + early-exit pool drag swamps admit-pool edge (Open-22/23/24 spawned, closed in v2.3 with engine PR pending for Open-24). Arc 3 closed CLEAN-NULL at Step 3 (2026-05-16). Arcs 5/6/7 closed under v2.1.2 (see STATUS.md "Recent Closures"). Arcs 1, 2 historical (ran under v1.x). KH-24 live deployment unaffected.

---

## What the Project Is

A long-only 4H trend-pullback system (KH-24) is in production. Parallel research arcs (the L arc series) are testing the top-N signals from the L characterization atlas (`docs/LCHAR_TOPN_REGISTRY.md`) through a six-step pipeline that ends with a WFO gate. The goal of L arc work is one or more PASS-DEPLOYABLE survivor systems that complement or supersede KH-24.

The signal under test in any given L arc is from the registry. From Arc 3 onward the test follows `L_ARC_PROTOCOL.md` v2.0 exactly — v2.0 is self-contained for deliverables, gates, and workflow.

---

## Locked Philosophy

| Rule | Detail |
|------|--------|
| Structure-first | Signal is price structure, not indicator |
| WFO worst-fold is the only judge at step 6 | Average fold, best fold — irrelevant |
| Dual-tier disposition at step 6 | PASS-DEPLOYABLE / PASS-VIABLE / clean-null; DD < 8% applies to both PASS tiers |
| Ex-ante population always | `build_ex_ante_bounded_population` — no exceptions |
| No lookahead / no repainting | Hard invariant; lookahead-invariant tests required at every step |
| Config-driven (YAML only) | No hardcoded parameters |
| Volume = veto only | Never generates trades |
| Clean labels = evaluation only | Never in population selection |
| Full distributions, never medians-only | No metric summarised as a single number |
| Effect size before significance | AUC + forward-geometry effect size both required in step 3 |
| Within-arc thresholds do not move | Calibration adjustments are cross-arc only (v2.0 §12) |

---

## KH-24 System Parameters (Locked, Out of Scope for L Arc)

```
Signal:     kb_exhaustion_bar (c1–c6, c8, c9)
            c7 DISABLED — volume gate removed
Direction:  Long only
Timeframe:  4H with D1 regime filter (one-day lag)
Pairs:      28 FX currency pairs
Broker:     5ers
Data:       data/4hr/, data/daily/, data/1hr/
Entry:      Bar N+1 open after signal on bar N close
Stop:       Entry price - 2.0 × ATR(14) [entry price anchor]
Trail:      Activates at close ≥ entry + 2.0 ATR (close-based)
            1.5 ATR behind highest close, bar-close updates only
Exits:      trailing_stop | kijun_d1 | stoploss
Risk:       1.0% of current reset floor balance (KH-24 era; L arc uses 0.5%)
Filters:    exposure cap=2; 1H CIR T=0.28
Spread:     Per-bar MT5 data — never hardcoded
D1 align:   One-day lag — each 4H bar sees prior calendar day's D1 close
```

KH-24 WFO gate: PASS. Worst-fold ROI +1.92% (F7); worst-fold DD 6.37% (F1); 214 trades across Oct 2020–Jan 2026; all 7 folds positive. Live on Contabo VPS / 5ers.

---

## L Arc Configuration (Per L_ARC_PROTOCOL v2.0)

```
Signal:        Per docs/LCHAR_TOPN_REGISTRY.md entry (currently 5 signals, arcs 1–5)
Direction:     Long only (all registry signals)
Timeframe:     1H primary (registry-defined)
Pairs:         28 FX, same set as KH-24
Entry:         Bar N+1 open
SL:            2.0 × ATR(14)_1H from entry price
Time exit:     Bar N+1+h open (h from registry entry)
Spread:        configs/spread_floors_5ers.yaml (locked, sha256 in arc-open doc)
Exposure cap:  Max 1 open position per pair (no currency cap, no concurrent-trade cap)
Risk:          0.5% of reset floor balance (L6 convention)
WFO:           7 anchored expanding folds, OOS Oct 2020 – Jan 2026
```

Cost accounting: spread is the only per-trade cost. Commission/swap/slippage applied as aggregate haircut at PASS-DEPLOYABLE evaluation only (per operational spec §7.4).

---

## Engine Status

- Python backtester: D1 lookahead fix applied (one-day lag). Source of truth.
- EA v2.01: KH-24 deployed on VPS. No L arc EAs exist yet — porting opens only when a PASS-DEPLOYABLE survivor exists.
- Determinism: byte-identical outputs on re-run required. CI-enforced.

Key scripts:
- `scripts/phase_kgl_v2_4h_wfo.py` — WFO runner
- `scripts/lchar/run_layer4.py` — canonical L registry signal source
- `configs/wfo_kh24.yaml` — locked KH-24 config (do not modify)
- `configs/spread_floors_5ers.yaml` — locked spread floor (do not modify)

---

## What Has Been Permanently Eliminated

- `jd_rf_evt_02_bounded_operational` — forward bias, fabricated results
- Clean labels in population selection
- Forward-conditioned dataset construction
- Indicator-driven C1 sweeps as primary research
- Exit indicator sweeps (57 tested, zero passed)
- Full NNFX stack as strategy
- GPT-4 / Aider for any implementation task
- Short signals (negative lift confirmed Phase KC; short-mirror exploratory run also failed)
- `signal_flip` exit (cuts winners); `kijun_4h` exit (fires on normal pullbacks)
- D1b slope filter (net negative across folds)
- Choppiness gate (redistributes damage)
- TP1 half-off structure (inferior to no-TP1)
- Currency exposure cap (KH era; superseded by exposure cap=2 in KH-22)
- `agree_count` gate (too rare)
- FOMC proximity filter (p=0.889)
- C7 volume gate on 5ers data (broker-specific, no lift validated)
- Range/ATR ceiling at 1.25× (fold 7 negative)
- 2% risk on 5ers data (daily cap breach, reset sim breach)
- Same-day D1 alignment (lookahead; permanently replaced by one-day lag)
- KH-25 re-entry exposure cap (KH-27 KILL — re-entries fire post-original-exit)
- 1H timeframe port of KH-24 (KI arc: mean R 0.004, t=0.095)
- L6.0 verbatim-as-gate framing (replaced by `L_ARC_PROTOCOL.md` v1.0; v1.x in turn superseded by v2.0 for Arcs 3+)
- Arc 2 signal (mtf_alignment.2_down_mixed.kijun, h=120) — SHELVED 2026-05-16, not permanently eliminated; cluster 2 has strong magnitude (fwd_mfe_p50 5.83R, t-stat +52) on unextractable paths, reopenable via v2.x calibration; see `results/l_arc_2_redo/ARC_2_REDO_RESULT.md`
- Arc 4 signal (`bar_range_top_decile__neg__h_001`, 1H) — SHELVED 2026-05-17 then re-evaluated under corrected p50 spread floors 2026-05-18. Re-run verdict: FAIL Step 6 under §10 full-pool deployment reckoning. Admit-pool edge intact (+0.125R per trade); structural failure on reject-pool (−0.232R × 32%) + early-exit-pool (−0.685R × 11%) drag. Same Pipeline D1 architectural failure as Arc 5. Signal not permanently eliminated — Pipeline E feasibility could be revisited as long-shot, but Step 4 E AUC 0.55 (gate 0.65) is the blocker. See `docs/archive/arc_results/ARC_4_RERUN_RESULT.md`.

## Cross-arc lessons

- **Pipeline D1 admit-only economics ≠ deployment economics (Arc 4 + Arc 5).** Two arcs in a row PASSED §9 admit-only stability and FAILED §10 full-pool deployment. Pipeline D1 carries mandatory cost on the reject pool (~−0.2 to −0.5R per rejected trade, classifier-discrimination-dependent) and on the early-exit pool (~−0.5 to −0.7R on 10-15% of signal flow, pre-t SL hits before classifier evaluates). Any Pipeline D1 candidate's deployment viability is `(admit_rate × admit_mean) vs (reject_rate × |reject_mean|) + (early_exit_rate × |early_exit_mean|)` — both Arc 4 and Arc 5 had costs ~2× the edge. Full-pool reporting mandatory at Step 4+. See `docs/archive/arc_results/ARC_4_RERUN_RESULT.md` and `PROTOCOL_IMPROVEMENT_BACKLOG.md` (Open-22/23/24).

- **Spread-floor changes are not population-invariant under exposure caps.** Changing the spread floor file shifts entry/exit fill prices, which shifts when stops fire, which shifts when the `max_concurrent_per_pair` cap releases, which shifts admission for subsequent signals. Trade pool can drift ±1-2% from a pure cost-model change. Path features (mid-based) remain spread-independent; PnL and exposure-derived metrics do not. Future arcs swapping spread files should expect Step 1 pool drift and propagate through Step 2 cluster sizes + Step 4 per-fold classifier retraining.

- **V-shape recovery: capturable, near-miss extractable, cross-arc deployable (Arc 7 + Arc 10).** Two V-shape near-misses on record (Arc 7 c1/c3/agg, Arc 10 c1) — capturable at Step 3 but missing Step 4 disjunctive E/D1 AUC gate. Cross-arc pool (EXP-05) closes the gap: pooled AUC 0.6348 with generic 17-feature subset vs 0.6057 (Arc 10 alone) / 0.4954 (Arc 7 c3 alone at common SL). WFO oracle on Arc 10 c1 confirms real OOS edge if cluster ID known at entry (Sharpe 4.61, expectancy 1.55R/trade). Single load-bearing feature: `L1_minus_L0_atr` (D1 HL slope magnitude) carries 116% of HTF LOO drop on Arc 10. Reading: V-shape archetype is a cross-arc deployable abstraction once a classifier with extended feature envelope is built; Arc 10 alone is in the noise zone (EXP-01 P(AUC ≥ 0.65) under bootstrap = 12.5%). Cross-arc clusterifier build is the leading v2.4 candidate. See `docs/archive/arc_results/ARC_10_RESULT.md` and `results/l_arc_10/experiments/ARC_10_EXPERIMENT_SYNTHESIS.md`.

### Not eliminated, but flagged

- Failed-breakout reversal long (Arc 6, 2026-05-17): path quality clean at v2.1.2 Step 3 (c2 **Stepwise climber**, mfe_p50=4.47R, ww_pp=0.000) but entry-time predictability below 0.65 deployability bar (best Pipeline E AUC 0.600 / 0.590); D1 admission collapses on threshold sweep. Not permanently eliminated; may return under richer feature regime, multi-TF entry context, or ensemble approach. See `docs/archive/arc_results/ARC_6_RESULT.md`. *(Note: Arc 6 was previously narrated as V-shape in some Arc 10 docs; per EXP-05 it is Stepwise.)*
- D1 swing-low rejection long (Arc 10 DLR, 2026-05-18): path quality clean at v2.3 Step 3 (c1 V-shape recovery, composite 0.4934 at SL=3.0×ATR, fwd_mfe_p50 3.08R, wrong_way_pp 0.000) but entry-time predictability near-miss on disjunctive §8 — c1 E AUC 0.6296 (margin −0.0204), D1 AUC 0.5897 (margin −0.0103). Both Path A near-miss < 0.03. Post-closure WFO oracle Sharpe 4.61 vs base −1.29 (gap +5.90) confirms structural OOS edge if cluster ID known at entry; realisable classifier ceiling pending feature-envelope expansion (per closure §"Why we can't filter to c1"). Not permanently eliminated; cross-arc clusterifier build with Arc 7 c3 is the leading v2.4 candidate (EXP-05 pool AUC 0.6348). See `docs/archive/arc_results/ARC_10_RESULT.md`.

---

## Vocabulary (post-Arc-10)

Terms used across post-Arc-10 dispatches and downstream docs. Defined here; referenced elsewhere — do not redefine.

- **Reverse FE.** Envelope-expansion activity that runs outside the gated pipeline. Method: qualitative characterisation of the target cluster's entries → encoded hypothesis catalog (pre-registered, hashed before validation) → cheap separation tests → routing to classifier or filter path. Distinct from Step 4, which evaluates a fixed envelope and does not iterate on failure. Anti-snooping protocol mandatory.
- **Classifier path.** Extend the entry-time feature envelope via new feature families (multi-TF trend alignment, pre-entry pattern context, volatility-regime descriptors, within-cluster sub-clustering), re-test through Step 4 on the extended envelope. Validation: AUC against existing gate (E ≥ 0.65 / D1 ≥ 0.60).
- **Filter path.** Hand-engineer deterministic entry-time conditions (e.g. `D1 slope > X AND compression ratio < Y AND realised-vol-percentile > Z`) that select cluster-like setups by construction. Validate on post-filter trade-set P&L (Sharpe, expectancy, max DD), not classifier AUC. Sidesteps the AUC gate entirely. More robust at small N; interpretable. Acceptance criteria pending (backlog item).
- **§16a Path A — disjunctive-gate ambiguity.** "Single criterion fail with margin < 0.03 → HALT" is ambiguous when a disjunctive Step 4 gate (E OR D1) fails on both criteria. Default reading: compound (Step 4 as one §8 gate → HALT). Strict reading (two numeric criteria → KILL) is available and should be documented in the closure when invoked. Arc 10 invoked compound. v2.4 cycle is the natural place to formalise.

## Conventions

- **Queue ownership.** `results/ARC_QUEUE.md` is owned by exactly one CC session at any time. Sessions that are not the queue owner must not modify it. Parallel sessions on the same project must coordinate via the queue owner. Arc 10 ran on `claude/charming-mcnulty-8160e0` while Arcs 8/9/11 ran in a parallel session that held queue ownership.
- **Determinism baseline.** `random_state=42`, `n_jobs=1`, `lineterminator="\n"` throughout for any work that must be byte-identical-reproducible. Audited via two-run sha256 comparison.
- **Closure docs land at `docs/archive/arc_results/ARC_<N>_RESULT.md`** per L_ARC_PROTOCOL §13. LIVE docs are retired at closure. (Arc 7 onward.)

## Activity catalog (downstream-route names)

- **Pipeline E / Pipeline D1** — classifier-gated extractability per protocol §3 / §8.
- **Reverse FE diagnostic** — envelope expansion outside the gated pipeline (pre-registered).
- **Cross-arc clusterifier build** — multi-arc pool + extended feature catalog (e.g. V-shape pool with `L1_minus_L0_atr` mandatory).
- **Filter-path probe** — deterministic conditions, P&L validation, post-filter trade-set metrics.
- **v2.X calibration packet** — cross-arc cycle bundling closures + experimentation + WFO evidence for protocol amendments.

---

## Tool Assignments

| Task | Tool |
|------|------|
| Strategy, research planning, decisions, verdicts | This chat (Opus 4.7) |
| Multi-file features, atlas computation, WFO runs, cluster fits, predictor scans | Claude Code (Opus 4.7) |
| Single-file patches, YAML edits, doc updates | Cursor (Sonnet 4.6) |
| MT5 / MQL5 implementation | Claude Code |

GPT-4 and Aider are permanently excluded from all implementation work.

---

## Folder Convention (v2, Locked 2026-05-13)

All result documents are co-located with their artefacts under the arc folder.
- L arc work: `results/l_arc_N/<step_subfolder>/...` (folder convention inherited from v1.x ops spec §2; v2.0 keeps the same layout).
- Future non-L-arc work: same pattern under `results/<arc_name>/`.
- `docs/` retained for non-arc-specific system specs (e.g. `docs/KH24_SYSTEM_LOCK.md`).

See `WORKFLOW.md` v2 for details.

---

## Risk Parameters (Prop Firm Constraints)

- Prop firm: 5ers
- Account constraints: max DD 10%, daily DD 5% — breach closes account permanently
- Per-trade risk: KH-24 uses 1%; L arc uses 0.5%
- Step 6 gate: DD < 8% applies to both PASS-DEPLOYABLE and PASS-VIABLE tiers; 8% is safety margin against the 10% prop limit

---

## Methodology in One Line

WFO worst-fold at dual-tier disposition is the only judge of success. Pre-committed gates, accepted results, every phase a documented finding regardless of pass or fail. The protocol is the barrel; chat creativity is the aim within the barrel.
