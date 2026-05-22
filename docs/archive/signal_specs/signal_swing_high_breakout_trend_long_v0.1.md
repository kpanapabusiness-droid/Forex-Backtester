# Signal Spec — Swing-High Breakout in Trend, Long (`signal_swing_high_breakout_trend_long_v0.1`)

> **Reconstructed 2026-05-22** from the producer-module docstring at
> [signals/lchar_swing_high_breakout_trend.py:1-37](../../../signals/lchar_swing_high_breakout_trend.py)
> and the locked-parameter assertions in
> [attic/scripts/l_arc_11/step1_backtest.py:519-526](../../../attic/scripts/l_arc_11/step1_backtest.py).
> The producer module is the source of truth — this file is a faithful
> transcription, not a re-design. Any divergence between this file and
> the producer module is a bug in this file.

---

## Identity

- **Family:** `trend_continuation_structural_breakout`
- **Base:** `swing_high_breakout`
- **Trial ID:** `ARC_11__swing_high_breakout_trend__long__4h__h_240`
- **Direction:** long
- **Signal TF:** 4H (locked)
- **Producer module:** `signals.lchar_swing_high_breakout_trend` — function `compute_signal(df_4h, signal_col="signal") -> df`

---

## Locked parameters

| Symbol | Value | Code constant |
|---|---:|---|
| Swing definition window (each side) | 3 | `SWING_K` |
| Trend-filter lookback (swing-low search) | 30 | `TREND_FILTER_LOOKBACK` |
| H_ref lookback (swing-high search) | 20 | `H_REF_LOOKBACK` |
| Right-edge offset (causal envelope) | 4 | `RIGHT_EDGE_OFFSET` |
| ATR period | 14 | `ATR_PERIOD` |
| ATR smoothing | Wilder (causal recursive) | `_wilder_atr` |
| Break buffer (× ATR) | 0.10 | `BREAK_BUFFER_ATR` |
| Close-position threshold | 0.5 (close in upper half) | `CLOSE_UPPER_HALF_MIN` |
| Refractory bars between signals | 20 | `REFRACTORY_BARS` |

---

## Swing definitions (3-bar local extreme — confirmation-lag causal)

For any bar position `k` on the 4H series:

- **swing-high at `k`** iff `high[k] > max(high[k-3..k-1])` AND `high[k] > max(high[k+1..k+3])`
- **swing-low at `k`**  iff `low[k]  < min(low[k-3..k-1])`  AND `low[k]  < min(low[k+1..k+3])`

The detection function uses both sides of bar `k`, so on its own it is non-causal. The **causal envelope** (below) makes the *consumption* of these swings causal at trigger time.

### Causal envelope (Step 1 right-edge audit reference)

At signal-bar `t`, only swings with `k ≤ t - 4` (`RIGHT_EDGE_OFFSET=4`) are identifiable / consumable. Implication: the latest swing position consumed has right-side detection bars at positions `(t-4)+1..(t-4)+3 = t-3..t-1`, i.e. all data closed strictly before bar `t` open (and trivially before the bar `t+1` entry).

This is the "confirmation-lag variant" of swing detection — distinct from non-causal centred smoothers, ±N-bar centred pivots, ZigZag, and bilateral change-point detectors, which the Arc 9 lesson (per `CLAUDE.md`) and L_PROTOCOL §1 forbid.

**Right-edge audit (mandatory at Step 1):** for every emitted trade, the bar-offset between H_ref and the signal bar must be ≥ 4; trades with offset < 4 must be 0.

---

## Trend filter (structural, no MA — Arc 9 convention)

1. Identify swing-lows in the window `[t - 30, t - 4]` (i.e. with `k ≤ t-4`, `RIGHT_EDGE_OFFSET=4`).
2. Require ≥ 1 such swing-low exists. If none, no signal.
3. Require `close[t-1] > min(swing-low values in window)`.

Note that condition 3 uses `close[t-1]` (the bar prior to the signal bar), not `close[t]`. This is preserved from the v2-era producer.

---

## Reference swing-high (`H_ref`)

1. Identify swing-highs in the window `[t - 20, t - 4]` (with `k ≤ t-4`).
2. `H_ref = high[k*]` where `k* = max{k : swing-high at k, k in window}` (most-recent identifiable swing-high).
3. Require `H_ref` exists (else no signal).

`H_ref` is the level the breakout fires against.

---

## Break trigger at bar `t` (long signal)

All three required:

- **3a. Decisive break with buffer:** `close[t] > H_ref + 0.10 × ATR(14)[t]`
- **3b. Bullish close:** `close[t] > open[t]`
- **3c. Close in upper half:** `(close[t] - low[t]) / (high[t] - low[t]) ≥ 0.5`

ATR(14) at bar `t` is Wilder-smoothed using TR values for bars `0..t` only — causal recursive form `atr[i] = (atr[i-1] · (period-1) + tr[i]) / period`, seeded as `atr[period-1] = mean(tr[0..period-1])`.

---

## Spacing & entry

- **4a. Refractory:** `t - last_signal_t ≥ 20` (`REFRACTORY_BARS=20`). The "last signal" tracker resets per pair.
- **4b. Entry:** bar `t+1` open. Long fill at `open_ask[t+1]` (per `core.sim.fill` long-entry semantics in v3.0).

---

## Surfaced columns (at signal-bar close)

The producer also emits the following columns alongside the boolean `signal` for downstream analysis (Step 4 audit, Step 2/3 path features, diagnostic surfaces):

| Column | Type | Definition |
|---|---|---|
| `signal` | bool | All gating conditions met (full signal fire). |
| `prefilter_pass` | bool | Trend-filter PASS + H_ref exists + bullish close + decisive break (all conditions except close-upper-half + refractory). Diagnostic only. |
| `h_ref` | float | `high[k*]` of the most-recent identifiable swing-high. |
| `h_ref_bar_offset` | float | `t - k*` (bars between H_ref bar and signal bar; ≥ 4 by construction). |
| `break_magnitude_atr` | float | `(close[t] - h_ref) / ATR(14)[t]`. NaN if no H_ref. |
| `close_position` | float | `(close[t] - low[t]) / (high[t] - low[t])`. NaN if `high[t] == low[t]`. |
| `trend_filter_swing_low` | float | `min(low[k] for k in trend-filter window)`. NaN if no swing-low in window. |
| `atr14` | float | Causal Wilder ATR(14) at bar `t`. |

These five (h_ref, h_ref_bar_offset, break_magnitude_atr, close_position, trend_filter_swing_low) constitute the per-arc signal-specific features added to the L_PROTOCOL §2 Step 1 default 27-feature catalog.

---

## Permanently locked decisions (do not vary in Arc 11)

- 3-bar swing definition (not 2, not 5). Single-side detection (not centred).
- 4H signal TF only (no MTF in this signal).
- Long-only (no short mirror).
- `RIGHT_EDGE_OFFSET=4` (constraint that makes swing consumption causal).
- 0.10 × ATR break buffer, close-upper-half ≥ 0.5, 20-bar refractory.
- Time horizon for the natural exit: 240 4H bars (= 40 calendar days) — locked at Step 1 sim; SL multiplier is swept at Step 3.

---

## Out of scope for this spec

- SL sweep (Step 3): `SL ∈ {1.5, 2.0, 2.5, 3.0, 3.5, 4.0} × ATR(14)`.
- Exit policy variants (Step 5): per L_PROTOCOL Appendix B, archetype-matched.
- Exposure cap (Step 5): `max_concurrent_per_currency ∈ {2, unlimited}`.
- Architecture search (Step 5 ML mechanics).
- Feature engineering for classifier (handled by `core.features.pipeline`, not this signal spec).

---

## Relationship to v2.x context

Arc 11 ran under L_ARC_PROTOCOL v2.3 in a prior epoch (closed `STEP_4_HALT` per the v2-era `docs/STATUS.md` line 5 — second capturable-not-extractable closure of record under v2.3 §16a Path A). The signal *definition* in this spec is identical between v2 and v3 — only the engine evaluating it changed. Per dispatch §1 paragraph 2: "Prior v2.x findings for this signal exist in `ARC_HISTORY.md`; treat them as historical record, NOT as expectations or targets. v3.0 is a different engine ... Cluster archetypes, surviving cohorts, and verdicts must come from the v3 evidence alone."

---

## End

Spec locked at v0.1. Changes require a new spec version (v0.2, ...) and a new arc.
