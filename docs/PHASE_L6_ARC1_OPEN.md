> # ⚠️ SUPERSEDED 2026-05-13
>
> This arc-open doc was written under L6.0 verbatim-as-gate framing, now SUPERSEDED by `L_ARC_PROTOCOL.md` v1.0.
>
> **Arc 1 is being redone** under the new protocol. The new arc-open doc will live at `results/l_arc_1/PHASE_L_ARC_1_OPEN.md` per `L_ARC_OPERATIONAL_SPEC.md` §2 folder structure.
>
> **What's retained from this original arc:** the step 1 trade-set (signal computation is unchanged; re-validation per `L_ARC_PROTOCOL.md` §5 confirms bit-identical match against canonical source).
>
> **What's discarded:** the verbatim WFO disposition (FAIL was an artefact of the framing, not a verdict on the signal's extractability).
>
> Arc 1 redo doubles as the protocol calibration check. See `L_ARC_PROTOCOL.md` §15.
>
> ---

# PHASE_L6_ARC1_OPEN — Arc 1 Opening Phase Document

**Arc:** L6+ Arc 1 (verbatim WFO of L registry rank 1)
**Status:** LOCKED (pending sign-off)
**Locked date:** 2026-05-10
**Predecessor docs:** `docs/L6_0_METHODOLOGY_LOCK.md`, `LCHAR_TOPN_REGISTRY.md` Entry 1, `configs/spread_floors_5ers.yaml` (sha256 `a613b4ce641c8d5218490531770a4924204029dedaa80fb24111beb61bd15547`)
**Modification rule:** No modifications to this document after sign-off without an explicit re-planning phase. Any modification resets the WFO; results from a pre-modification run are not pooled with results from a post-modification run.

---

## 1. Purpose

Lock the verbatim tradable conversion of L registry rank 1 (`TRIAL__univariate_extreme__abs_return_top_decile__neg__h_001`) into a runnable WFO specification, ready for Claude Code execution.

The signal definition itself is locked at L4 / the registry. This document locks only the conversion: entry rule, exit rule, SL spec, position sizing, fold layout, the YAML config path, and the result-document path. No filters, no trail, no structural exits — verbatim per L6.0 §9.

---

## 2. Signal Definition (verbatim from `LCHAR_TOPN_REGISTRY.md` Entry 1)

### 2.1 L4 condition

A signal fires at bar N close, on signal_TF = 1H, when **both** sub-conditions hold:

1. **Univariate extreme:** `abs(log_return[N])` strictly greater than the top-decile threshold of the trailing 100 1H bars (per pair × signal_TF). The trailing window is bars `N-100` through `N-1` inclusive; bar N is excluded from the threshold computation. Threshold is the 90th percentile of `abs(log_return)` over those 100 bars, computed at bar N close. No lookahead.
2. **Direction sub-spec `_neg`:** `close[N] < open[N]` (signal bar is a down bar).

`log_return[N] = ln(close[N] / close[N-1])`.

### 2.2 Direction of trade (per L6.0 §6)

**LONG.** Pooled conditional mean return for this trial is +0.0306 ATR-normalised; sign-rule yields long. The `_neg` sub-spec describes the signal bar (a large down bar), not the trade.

### 2.3 Horizon

`h = 1 bar` (1H). The L4 trial measured next-1-bar return. The verbatim time exit is therefore at bar N+2 open (1 bar after entry; see §3.3).

---

## 3. Tradable Conversion

### 3.1 Entry trigger

Signal fires at bar N close as defined in §2.1. Entry is queued for the next bar's open.

### 3.2 Entry execution

**Bar N+1 open.** Never same-bar. Per KH-24 convention.

If the pair already has an open position at bar N+1 open, the signal is dropped (§3.7). If the spread floor cannot be applied (file load failure), the WFO halts hard — never silently entered without floor.

### 3.3 Time exit

**Bar N+2 open.** This implements the L4 horizon h=1: the trade is open for exactly one 1H bar between bar N+1 open and bar N+2 open.

Time exit price is bar N+2 open. Spread is applied per `resolve_spread_pips` with floor (§3.6).

### 3.4 Stop loss

`sl_price = entry_price - 2.0 × ATR(14)` at signal_TF = 1H.

- ATR is Wilder-smoothed ATR(14) computed on 1H bars, evaluated at bar N close (no lookahead).
- SL anchored to entry price, not to signal bar low (KH-24 convention).
- SL fires intrabar between bar N+1 open and bar N+2 open: if low of bar N+1 ≤ sl_price, the trade exits at sl_price. Otherwise, time exit at §3.3.
- SL has priority over time exit when both would trigger on the same bar.

Note: at 1H × 2.0 ATR over a 1-bar hold, SL is structurally far away; most exits will be time exits. This is mechanically correct per L4 measurement and is not a defect to engineer around in phase 1.

### 3.5 No other exits

Per L6.0 §9 verbatim phase 1:

- No trailing stop
- No structural exits (no kijun cross, no signal flip, no MTF override)
- No filters (no h1_close_in_range, no MTF confirmation, no D1 trend filter)
- No exposure cap (no currency cap, no max-concurrent-positions cap)

Phase 1 is the verbatim test of the condition itself, not of a layered system.

### 3.6 Spread

Per L6.0 §7. `spread_floor.enabled: true`, source `configs/spread_floors_5ers.yaml`, expected_body_sha256 `a613b4ce641c8d5218490531770a4924204029dedaa80fb24111beb61bd15547`. Applied at entry and time exit and SL fill via `resolve_spread_pips`.

### 3.7 Concurrent positions per pair

**Maximum one open position per pair at any time.** If a new signal fires on a pair that already has an open position, the new signal is dropped (logged, counted, but not entered). At h=1 this is mostly moot (prior trade closes 1 bar after entry); included for safety.

### 3.8 Daily DD guard

**None.** Daily DD is logged per fold but does not halt within-fold execution. Per L6.0 §8 ("operational only — logged per fold, not gated"). Live-deployment 5ers daily-DD enforcement is a downstream concern, not a phase-1 backtest concern.

---

## 4. Position Sizing

Per L6.0 §8 / KH-24 convention:

```
risk_per_trade = 0.01 × reset_floor_balance
sl_distance_price = 2.0 × ATR(14)_at_signal_TF_1H_at_bar_N
position_size_units = risk_per_trade / (sl_distance_price × pip_value_per_unit)
```

- `reset_floor_balance` recomputed at start of each calendar month (5ers convention).
- `pip_value_per_unit` per pair convention; backtester handles JPY-quoted vs non-JPY internally.
- Position size is rounded to the nearest 0.01 lot (or backtester's existing convention; do not introduce new rounding).

---

## 5. Pair Set

All 28 pairs per L6.0 §5 / KH-24:

```
AUD_CAD AUD_CHF AUD_JPY AUD_NZD AUD_USD CAD_CHF CAD_JPY CHF_JPY
EUR_AUD EUR_CAD EUR_CHF EUR_GBP EUR_JPY EUR_NZD EUR_USD GBP_AUD
GBP_CAD GBP_CHF GBP_JPY GBP_NZD GBP_USD NZD_CAD NZD_CHF NZD_JPY
NZD_USD USD_CAD USD_CHF USD_JPY
```

---

## 6. WFO Structure

Per L6.0 §4. Inherited from KH-24 §3 verbatim except for the trades-per-fold floor.

| Property | Value |
|----------|-------|
| Method | Anchored expanding window |
| Folds | 7 |
| IS period | 4 years per fold start |
| OOS period | 9 months per fold |
| OOS coverage | 2020-10-01 → 2026-01-01 |
| Signal computation | Always at signal_TF = 1H, per-pair |
| ATR computation | Wilder ATR(14) at 1H |

### 6.1 Fold OOS windows

| Fold | OOS Start | OOS End |
|------|-----------|---------|
| 1 | 2020-10-01 | 2021-07-01 |
| 2 | 2021-07-01 | 2022-04-01 |
| 3 | 2022-04-01 | 2023-01-01 |
| 4 | 2023-01-01 | 2023-10-01 |
| 5 | 2023-10-01 | 2024-07-01 |
| 6 | 2024-07-01 | 2025-04-01 |
| 7 | 2025-04-01 | 2026-01-01 |

### 6.2 IS period for signal computation

The trailing-100-bar abs_return decile threshold (§2.1.1) and the ATR(14) (§3.4) are **rolling** per-bar computations. They use only the trailing 100 / 14 bars at any given bar N. This is naturally walk-forward by construction; no IS/OOS partitioning of these computations is needed beyond the standard rolling lookback.

There are no fitted parameters in the signal. The "IS period" in WFO terminology is therefore a no-op for signal definition — the signal computation on OOS bars uses only the data trailing those bars. The IS period exists in the fold layout for consistency with KH-24's structure and for any future phase that adds fitted parameters.

### 6.3 Data identity with L arc (mandatory invariant)

The WFO operates on the **exact same data files** the L arc characterized. Specifically:

- **Pair set:** identical (28 pairs per §5, matches L0 §2 / L registry pair-set)
- **Timeframe:** identical (1H, matches registry Entry 1 `signal_tf = 1H`)
- **Data window:** Arc 1 OOS coverage 2020-10-01 → 2026-01-01 (right-exclusive on data end 2025-12-31) matches the L arc window 2020-10-01 → 2025-12-31 verbatim
- **Source files:** identical — `data/1hr/<pair>.csv` per `DATA_AUDIT_REPORT.md` discovery, no extension, no substitution
- **Signal computation:** rolling 100-bar abs(log_return) 90th-percentile threshold + `close[N] < open[N]` filter, identical to the L4 trial's `univariate_extreme.abs_return_top_decile.neg.h_001` measurement bar-for-bar

The only difference between what L4 measured and what Arc 1 tests is what happens **after** the signal fires:

- L4: measured the conditional next-1-bar log return distribution at the signal bar (no execution, no costs)
- Arc 1: enters at bar N+1 open, exits at bar N+2 open (or SL hit), pays per-bar MT5 spread with the §7 floor, sized at 1% of reset floor, walked across 7 anchored expanding folds

If the L4 condition has tradable edge after realistic execution costs, Arc 1's gate will reflect it. If it doesn't, Arc 1 nulls cleanly. The WFO is the realistic-execution counterpart of the L4 descriptive measurement — same condition, same data, same bar-for-bar identity, with execution layered on top.

This invariant is mandatory. If the WFO's signal-firing bar set diverges from what the L4 trial would identify on the same data, that's a bug, not a deviation. The result doc must include a sanity check confirming the signal-firing bar count on the OOS window matches the L4-trial-equivalent bar count to within rounding (any divergence > 0 fires investigation).

---

## 7. Gate

Per L6.0 §4. Verbatim:

A WFO PASSES the gate if and only if all three conditions hold across all 7 folds:

1. Worst-fold ROI > 0%
2. Worst-fold max DD < 8%
3. Trades per fold ≥ 15

If any fold fails any of the three, the gate FAILS and Arc 1 is closed as a verbatim null per L6.0 §9 (no filter rescue).

---

## 8. YAML Config

The arc is config-driven via `configs/wfo_l6_arc1.yaml`. The config is created by Claude Code per the spec below; this document defines the locked content.

### 8.1 Required config block (illustrative — Claude Code uses backtester schema)

```yaml
phase: l6_arc1_verbatim
description: |
  L6+ Arc 1 — verbatim WFO of L registry rank 1
  TRIAL__univariate_extreme__abs_return_top_decile__neg__h_001
  Per docs/PHASE_L6_ARC1_OPEN.md and docs/L6_0_METHODOLOGY_LOCK.md

signal:
  type: l4_univariate_extreme
  signal_tf: 1H
  base: abs_log_return
  lookback_bars: 100
  threshold_quantile: 0.90
  threshold_comparison: strict_greater
  direction_filter: neg          # signal bar close < open
  trade_direction: long          # per L6.0 §6 sign rule

entry:
  bar_offset: 1                  # bar N+1 open
  execution: open

exit:
  hard_stop:
    type: atr_distance
    atr_period: 14
    atr_smoothing: wilder
    atr_tf: 1H
    multiplier: 2.0
    anchor: entry_price
  time_exit:
    bars_after_entry: 1          # exit at bar N+2 open
  trail: null
  structural_exits: []

filters: []                      # phase 1 verbatim — no filters

exposure:
  max_concurrent_per_pair: 1
  currency_cap: null             # no cap at phase 1
  daily_dd_guard: false

risk:
  basis: reset_floor_balance
  pct_per_trade: 0.01

spread_floor:
  enabled: true
  source: configs/spread_floors_5ers.yaml
  expected_body_sha256: a613b4ce641c8d5218490531770a4924204029dedaa80fb24111beb61bd15547

pairs:                           # all 28, per L6.0 §5
  - AUD_CAD
  - AUD_CHF
  # ... (full list per §5)

walk_forward:
  method: anchored_expanding
  n_folds: 7
  is_period_years: 4
  oos_period_months: 9
  oos_start: 2020-10-01
  oos_end: 2026-01-01

output:
  results_dir: results/l6/arc1/
  trades_csv: trades_all.csv
  fold_results_csv: wfo_fold_results.csv
  summary_txt: wfo_summary.txt
```

The YAML must validate against `BACKTESTER_SCHEMA.json` (post-spread-floor schema, sha256 `5096e8ae457f8f74841e85023e635cfc31f3c6d6ad87ee93418c7c88d80acac5`).

### 8.2 Config schema gap

If `signal.type: l4_univariate_extreme` is not yet supported by the existing backtester schema, the Arc 1 execution prompt must implement it as a one-change addition before running the WFO. The signal computation is mechanical (rolling 100-bar abs(log_return) decile + sign-of-bar filter) and must produce zero lookahead by construction. This is an expected schema extension, not a deviation; document it in the WFO result doc.

---

## 9. Outputs

Claude Code produces, in `results/l6/arc1/`:

| File | Description |
|------|-------------|
| `trades_all.csv` | Every taken trade across 7 OOS folds with entry/exit timestamps, entry/exit prices, SL, R, MAE/MFE, fold ID, pair |
| `wfo_fold_results.csv` | Per-fold metrics: trades, ROI, max DD, win%, mean R, gate disposition |
| `wfo_summary.txt` | Gate summary: per-fold table, worst-fold ROI, worst-fold DD, trades-per-fold floor check, overall PASS/FAIL |
| `signals_log.csv` | Every fired signal (taken or dropped) with reason for drop if applicable — for sanity checking signal flow |
| `PHASE_L6_ARC1_RESULT.md` | Result document (mandatory regardless of gate outcome) |

---

## 10. Sanity Checks (mandatory in result document)

The result doc must include a sanity-check section reporting at minimum:

| Check | Pass criterion |
|-------|----------------|
| Spread floor hash | Matches locked sha256, logged at WFO start |
| Spread floor application count | > 0, pct_floored within audit range 4–24% per pair |
| SL distance | Exactly 2.0 × ATR(14)_1H from entry on all taken trades |
| Direction assertion | All taken trades long; SL < entry on every trade |
| Lookahead | Zero — threshold and ATR computed using only bars ≤ N at signal time |
| Same-bar entries | Zero — every entry at bar N+1 open, never bar N |
| Concurrent-per-pair | Zero violations of §3.7 |
| Trade count per fold | ≥ 15 (or gate fails on condition 3) |
| L4-bar-identity | Signal-firing bar count on OOS window matches the L4-trial-equivalent count bar-for-bar (any divergence > 0 triggers investigation per §6.3) |

A failed sanity check halts the result doc and triggers re-run. Sanity checks are not optional, regardless of gate disposition.

---

## 11. Pre-Committed Expectations (for honesty)

Recording what the planner expects before seeing the result, so the gate evaluation is honest:

- **Trade frequency:** Top decile × 28 pairs × ~9-month OOS fold ≈ a few hundred to ~1k signals per fold. After the `_neg` sub-spec (~50% of signals) and the per-pair concurrent cap, expect 100–500 taken trades per fold. Likely well above the 15-trade floor.
- **Per-trade economics:** L4 pooled mean +0.0306 ATR-normalised over 1 bar. Typical 1H ATR ≈ 0.2–0.5 of typical spread. Net edge per trade after spread floor is plausibly small to break-even; this is the hardest test.
- **Fold variance:** 7 folds covering 2020–2026 span COVID, 2022 USD strength, 2024 carry-trade unwinds, 2025 reflation. Worst-fold DD on 100–500 trades per fold without exposure cap could plausibly exceed 8%, especially in correlated-pair-cluster fold periods. **Fold 1 (Oct 2020 – Jul 2021) and Fold 6 (Jul 2024 – Apr 2025) are the highest-prior-risk folds for DD failure.** No exposure cap means correlated EUR/CHF/JPY pair clustering can stack losses on a single regime move.
- **Most likely null mode:** worst-fold ROI ≤ 0% on 1–2 folds, gate fails on condition 1.
- **Most likely pass mode:** all folds positive but small (worst-fold ROI < 1%), gate barely passes; subsequent phases add filters / exposure caps to harden.

These are not predictions; they are anchors against which the actual result will be read. If the result diverges sharply from any of them, that's worth investigating before drawing conclusions.

---

## 12. What Subsequent Phases (if Arc 1 verbatim passes) Would Test

Listed for context, not in scope for this doc:

- Phase 2: add KH-24-style currency exposure cap (max 2 per currency)
- Phase 3: add a 1H or 4H confirmation filter analogous to KH-24's `h1_close_in_range`
- Phase 4: add an ATR trail (analogous to KH-24's 1.5× trail)
- Phase 5: pair-set restriction by per-fold IS-only per-pair Sharpe

Each is one change per phase per L6.0 §9. None are pre-committed; planner decides post-Arc-1-verbatim.

---

## 13. Out of Scope (consolidated)

- Modifications to L registry, spread floor file, or any locked artefact
- Short-side testing (trade direction is long per §2.2)
- Alternative horizons (h=24, h=120 are separate registry entries → separate arcs)
- Alternative measurement bases (`bar_range_top_decile_neg` is rank 4 → Arc 4)
- KH-24 modifications
- Live deployment of any kind

---

## 14. Tool Usage for Arc 1 Execution

- **This chat (planning):** completed once L6.0 and this Arc 1 opening doc are signed off
- **A separate Arc 1 execution chat (Opus 4.7):** issues the Claude Code prompt that runs the WFO and produces the result doc
- **Claude Code (Opus 4.7, xhigh effort):** all backtester work, WFO execution, signal-type schema extension if needed, signal flow logging, result doc generation
- **Cursor:** YAML config commit, post-result-doc patches if needed
- **GPT-4 and Aider:** permanently excluded

---

## 15. Sign-Off

Once this document is committed, Arc 1's verbatim spec is locked. The Arc 1 execution chat issues the WFO prompt against the spec verbatim — no methodology drift between this doc and the run.

**Signed off by:** _______________
**Date:** _______________

---

*End of Arc 1 Opening Phase Document. Companion document at `results/l6/arc1/PHASE_L6_ARC1_RESULT.md` produced post-WFO regardless of gate disposition.*
