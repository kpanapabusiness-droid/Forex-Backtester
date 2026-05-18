# CLAUDE.md — Forex Ignition Rebuild
> Last updated: 2026-05-17 | Phase: Arc 4 CLOSED (CLEAN-NULL on transaction-cost truth); spread floor file replaced 2026-05-17 with per-pair p50 values, arc work unblocked
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

**Active research: L arc signal testing under `L_ARC_PROTOCOL.md` v2.1.2.** Arc 4 closed CLEAN-NULL on transaction-cost truth 2026-05-17 — signal reached pass-deployable through Step 5 under modeled spreads; HistData spread audit revealed `configs/spread_floors_5ers.yaml` under-models real spreads by 3-48x per pair, killing F6 worst-fold sign consistency under real-spread reconciliation. **Spread floor file replaced 2026-05-17 with per-pair p50 values from HistData 2024-2025 audit. `configs/spread_floors_5ers.yaml` is now calibration-curated (no longer generated; source script `scripts/lchar/compute_spread_floors.py` retired). Arc work unblocked.** See `docs/calibration_decisions/SPREAD_FLOOR_CALIBRATION_DECISION_2026-05-17.md` for calibration rationale, `docs/SPREAD_FLOOR_AUDIT_FINDING.md` (RESOLVED section) for original finding, and `docs/arc_results/ARC_4_RESULT.md` for the arc context that surfaced the issue. Arc 3 closed CLEAN-NULL at Step 3 (2026-05-16). Arcs 1 and 2 are historical (ran under v1.x).

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
- Arc 4 signal (`bar_range_top_decile__neg__h_001`, 1H) — SHELVED 2026-05-17, not permanently eliminated; real edge through Step 5, killed by transaction-cost truth under HistData spread audit (cluster 1 D1 AUC 0.667, pass-deployable at 0.20% risk under modeled spreads; F6 ann ROI flips +10.08% → ~−5.6% under real-spread reconciliation). Re-evaluatable under corrected spread file. See `docs/arc_results/ARC_4_RESULT.md`.

### Not eliminated, but flagged

- Failed-breakout reversal long (Arc 6, 2026-05-17): path quality clean at v2.1.2 Step 3 (c2 Stepwise, mfe_p50=4.47R, ww_pp=0.000) but entry-time predictability below 0.65 deployability bar (best Pipeline E AUC 0.600 / 0.590); D1 admission collapses on threshold sweep. Not permanently eliminated; may return under richer feature regime, multi-TF entry context, or ensemble approach. See `docs/arc_results/ARC_6_RESULT.md`.

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
