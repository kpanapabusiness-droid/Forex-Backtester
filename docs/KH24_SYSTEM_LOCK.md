# KH24 System Lock
> As of 2026-04-20. This is day 0 for all future research.
> Replaces KGL_V2_FINAL_SYSTEM.md as the living spec.

---

## Section 1 — Status

**KH-24 is the validated base system as of 2026-04-20.**

Gate: **PASS**
Worst-fold ROI: **+1.92%** (fold 7)
Worst-fold DD: **6.37%** (fold 1)
All 7 WFO folds positive.
First gate pass in the KH arc.

This document is the new day 0. All future phases add to or subtract from
this system. Do not modify this document — create a new phase document for
each change.

---

## Section 2 — System Components

The system has three components, each solving a different problem.

---

### Component 1 — Baseline Signal (unchanged from KGL V2 Final)

All conditions evaluated at 4H bar N close. Entry at bar N+1 open.

**Entry conditions:**

| Condition | Rule |
|-----------|------|
| C1 | `close < open` |
| C2 | `abs(close - open) / ATR(14) >= 0.5` |
| C3 | `(close - low) / (high - low) <= 0.24` |
| C4 | `close > 4H Kijun(26)` |
| C5 | `close <= 4H Kijun(26) + 1.0 × ATR(14)` |
| C6 | `(close - close[N-10]) / ATR(14) <= -0.5` |
| C7 | **DISABLED** — volume gate, permanently removed |
| C8 | `prev D1 close > prev D1 Kijun(26)` [lag-1] |
| C9 | `prev D1 close <= prev D1 Kijun(26) + 1.0 × prev D1 ATR(14)` |

**Entry:** Bar N+1 open (never same-bar)

**Stop loss:** `entry_price - 2.0 × ATR(14)` — anchored to entry price, not signal bar low

**Trail:**
- Activates when bar close `>= entry_price + 2.0 × ATR` (close-based, never intrabar)
- Level: `max_close_since_activation - 1.5 × ATR`
- Ratchets up on close only — fills next bar open when triggered intrabar

**Exits (priority order):**
1. Hard stop loss — intrabar touch of `entry_price - 2.0 × ATR`
2. Trailing stop — intrabar touch of trail level once active
3. kijun_d1 — prev D1 close < prev D1 Kijun(26), 1-bar confirmation

**Permanently disabled exits:** signal_flip, kijun_4h

**Direction:** Long only. Short permanently closed.

**Risk:** 1% of current reset floor balance per trade.

**Timeframe:** 4H primary, D1 regime filter (D1 values use lag-1 convention — never forward-fill)

**ATR:** Wilder smoothed ATR(14) for both 4H and D1

**Kijun-sen:** `(highest high + lowest low) / 2` over period 26 — not a moving average

**Spread:** Per-bar MT5 spread applied on entry only — never hardcoded

**Pairs (28):**

| | | | |
|-|-|-|-|
| AUD_CAD | AUD_CHF | AUD_JPY | AUD_NZD |
| AUD_USD | CAD_CHF | CAD_JPY | CHF_JPY |
| EUR_AUD | EUR_CAD | EUR_CHF | EUR_GBP |
| EUR_JPY | EUR_NZD | EUR_USD | GBP_AUD |
| GBP_CAD | GBP_CHF | GBP_JPY | GBP_NZD |
| GBP_USD | NZD_CAD | NZD_CHF | NZD_JPY |
| NZD_USD | USD_CAD | USD_CHF | USD_JPY |

---

### Component 2 — Currency Exposure Cap = 2

**Problem solved:** Fold 1 DD (9.01% → 6.37%)

**Rule:** Maximum 2 concurrent open trades sharing any single currency.
Evaluated at entry time using only currently open trades — no lookahead.

**Example:** If EURUSD and EURGBP are open, any new EUR pair signal is blocked
until one of those trades closes.

**YAML param:** `currency_exposure_cap: 2`

Already implemented in the backtester.

---

### Component 3 — 1H Close-in-Range Filter at T = 0.28

**Problem solved:** Fold 6/7 ROI (negative → +3.24% / +1.92%)

**Rule:** At signal bar N close, compute for the last fully-closed 1H bar:

```
h1_last_bar_close_in_range = (h1_close - h1_low) / (h1_high - h1_low)
```

- If `> 0.28`: skip signal entirely
- If `<= 0.28`: proceed to entry
- If `h1_high == h1_low`: treat as 0.5 (neutral — allow entry)

**1H bar selection:** Last completed 1H bar at or before bar N close time.
The 1H bar is fully closed before the 4H signal fires — no lookahead.

**YAML param:** `kh24_h1_range_threshold: 0.28`

**Interpretation:** The filter blocks signals where the 1H bar closed in the
upper 72% of its range, indicating 1H momentum is still bullish. When the
4H exhaustion bar fires too late into a 1H bounce, the entry quality
degrades. Winners occur when the 1H is still bearish (closed in the bottom
28% of its range), confirming the 4H pullback is genuine and not already
exhausting itself at the 1H level.

---

## Section 3 — WFO Results

**Method:** Anchored expanding window, 7 folds
**IS period:** 4 years per fold start
**OOS period:** 9 months per fold
**OOS coverage:** Oct 2020 – Jan 2026
**Gate:** worst-fold ROI > 0% AND worst-fold DD < 8%

### Fold Table

| Fold | OOS Start | OOS End | Trades | ROI | DD | Win% | Mean R | Gate |
|------|-----------|---------|--------|-----|----|------|--------|------|
| 1 | 2020-10-01 | 2021-07-01 | 41 | +13.35% | 6.37% | 43.9% | +0.341 | PASS |
| 2 | 2021-07-01 | 2022-04-01 | 36 | +9.63%  | 4.45% | 58.3% | +0.278 | PASS |
| 3 | 2022-04-01 | 2023-01-01 | 25 | +11.90% | 4.43% | 56.0% | +0.479 | PASS |
| 4 | 2023-01-01 | 2023-10-01 | 32 | +3.32%  | 3.80% | 46.9% | +0.118 | PASS |
| 5 | 2023-10-01 | 2024-07-01 | 23 | +6.23%  | 3.09% | 52.2% | +0.283 | PASS |
| 6 | 2024-07-01 | 2025-04-01 | 30 | +3.24%  | 5.03% | 43.3% | +0.140 | PASS |
| 7 | 2025-04-01 | 2026-01-01 | 27 | +1.92%  | 4.06% | 51.9% | +0.082 | PASS |

**Total OOS trades:** 214
**Worst-fold ROI:** +1.92% (F7)
**Worst-fold DD:** 6.37% (F1)
**Gate: PASS — all 7 folds positive**

### Signal Flow (OOS)

| Stage | Count |
|-------|-------|
| Baseline OOS signals (no h1 filter, no cap) | 328 |
| Filtered by h1 filter (h1_last_bar_close_in_range > 0.28) | 107 |
| Blocked by exposure cap | 7 |
| **Trades taken** | **214** |

35% reduction vs baseline. The h1 filter removes 33% of candidates; the cap
blocks 7 further.

### Sanity Checks

| Check | Result |
|-------|--------|
| h1_last_bar_close_in_range: min=0.000, max=0.280, mean=0.123 | **PASS** |
| SL distance: exactly 2.0× ATR on all original trades | **PASS** |
| D1 lag: all D1 values use lag-1 convention | **PASS** |

No taken trade violates the h1 threshold. All stop distances are exactly 2.0×
ATR from entry. D1 values are never forward-filled.

---

## Section 4 — What Is Permanently Locked

These decisions are final. Do not revisit without exceptional evidence.

| Rule | Reason |
|------|--------|
| Signal direction: long only | Short lift confirmed negative (Phase KC) |
| Entry bar: N+1 open always | No same-bar entries ever |
| D1 lag-1 convention | Forward-fill introduces lookahead |
| SL: 2.0× ATR from entry price | Not signal bar low |
| Risk: 1% of reset floor balance | 2% breaches 5ers daily cap |
| kijun_d1: 1-bar confirmation | 2-bar (KH-8) reduces trade quality |
| kijun_4h exit: disabled | Fires on normal pullbacks, cuts winners |
| signal_flip exit: disabled | Cuts winners, net negative |
| C7 volume gate: disabled | 5ers data incompatibility, no lift validated |
| Aider: excluded | Permanently, all implementation work |
| GPT-4: excluded | Permanently, all implementation work |

---

## Section 5 — What Is Open for Research

**Add one change at a time on top of KH-24. Gate unchanged: worst-fold ROI > 0%, worst-fold DD < 8%.**

| Phase | Change | Source |
|-------|--------|--------|
| KH-25 | Re-entry logic (KH-16 style) on top of KH-24 | KH-16 passed on its own |
| KH-26 | Broader 1H timeframe signal confirmation research | 1H filter showed promise |
| Future | Re-entry threshold sweep | Only if KH-25 passes |

**Do not test:**

- Anything that changes the baseline signal conditions (C1–C6, C8, C9)
- Anything that removes the exposure cap or 1H filter
- Short signals
- Exit indicator sweeps (57 tested, zero passed)
- Full NNFX stack

---

## Section 6 — Files and Configs

### Locked Configs (never modify)

| File | Description |
|------|-------------|
| `configs/wfo_baseline_clean.yaml` | Original baseline — no cap, no h1 filter |
| `configs/wfo_kh24.yaml` | KH-24 system lock — do not modify |

### Key Results

| File | Description |
|------|-------------|
| `results/kh24/trades_all.csv` | 214 OOS trades |
| `results/kh24/wfo_fold_results_4h.csv` | Fold-level metrics |
| `results/kh24/wfo_summary_4h.txt` | Gate summary |
| `results/kh24/PHASE_KH24_RESULT.md` | Phase result record |

### Backtester Entry Point

```
python scripts/phase_kgl_v2_4h_wfo.py -c configs/wfo_kh24.yaml
```
