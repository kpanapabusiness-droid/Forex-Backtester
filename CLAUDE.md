# CLAUDE.md — Forex Ignition Rebuild
> Last updated: 2026-05-13 | Phase: L arc signal testing, Arc 1 redo opens under L_ARC_PROTOCOL v1.0
> First file any AI assistant reads. Reflects where the project ACTUALLY is.

---

## Read These First, In Order

1. **`L_ARC_PROTOCOL.md`** — methodology of record for all L arc signal-testing work. Locked v1.0.
2. **`L_ARC_OPERATIONAL_SPEC.md`** — deliverables, folder layout, angle catalogues, scoring tables. Locked v1.0.
3. **`SESSION_ZERO.md`** — 5-minute primer on current state.
4. **`STATUS.md`** — tight current-state snapshot.

Then, depending on scope:
- Touching the live system → `docs/KH24_SYSTEM_LOCK.md`
- Running an L arc step → the arc's `PHASE_L_ARC_N_OPEN.md` under `results/l_arc_N/`
- Phase close → `WORKFLOW.md` v2

---

## CRITICAL: Current State

**Live system KH-24 is locked, passing, deployed.** Out of scope for L arc work; do not modify without an explicit modification phase.

**Active research: L arc signal testing under `L_ARC_PROTOCOL.md` v1.0.** Arc 1 redo is the current/next phase. Arc 1 redo doubles as the protocol calibration check (`concurrent_signals_within_3h` must surface as ≥ Tier 2 predictor in step 3, or halt and investigate).

---

## What the Project Is

A long-only 4H trend-pullback system (KH-24) is in production. Parallel research arcs (the L arc series) are testing the top-N signals from the L characterization atlas (`docs/LCHAR_TOPN_REGISTRY.md`) through a six-step pipeline that ends with a WFO gate. The goal of L arc work is one or more PASS-DEPLOYABLE survivor systems that complement or supersede KH-24.

The signal under test in any given L arc is from the registry. The test follows `L_ARC_PROTOCOL.md` v1.0 exactly. Per-arc deliverables are specified in `L_ARC_OPERATIONAL_SPEC.md` v1.0.

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
| Within-arc thresholds do not move | Calibration adjustments are cross-arc only, via v1.1 patch |

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

## L Arc Configuration (Per L_ARC_PROTOCOL v1.0)

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
- L6.0 verbatim-as-gate framing (replaced by `L_ARC_PROTOCOL.md` v1.0)

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
- L arc work: `results/l_arc_N/<step_subfolder>/...` per `L_ARC_OPERATIONAL_SPEC.md` §2.
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
