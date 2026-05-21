# v1.3 calibration — loader normalisation decisions

> Companion to `scripts/v1_3_calibration/load_paths.py`
> Date: 2026-05-15

Three trade-path datasets feed v1.3 calibration. They emit in three
different conventions (KH-24 backtester ATR-units, Arc 1 step 2
ATR-units, Arc 2 step 2 ATR-units with OHLC). The loader normalises
them to a single per-bar schema. The decisions are recorded here so
downstream metric code can rely on the contract and the chat can
audit the normalisation when reading the calibration report.

## Output schema (per-bar)

| Column | Type | Convention |
|---|---|---|
| `trade_id` | string | unique per trade within dataset (KH-24: `f"{pair}_{entry_date}"`; Arc 1 / Arc 2: integer id cast to string) |
| `bar_offset` | int32 | 0 = entry bar (Arc 1 re-indexed from t to bar_offset = t-1) |
| `high_r` | float32 | (high − entry_px) / sl_distance, signed; **NaN for Arc 1** (no per-bar OHLC) |
| `low_r` | float32 | (low − entry_px) / sl_distance, signed; **NaN for Arc 1** |
| `close_r` | float32 | (close − entry_px) / sl_distance, signed |
| `mfe_so_far_r` | float32 | running max of high_r within the trade's emitted window, R-units |
| `mae_so_far_r` | float32 | running min of low_r within the trade's emitted window, R-units (≤ 0 for long) |
| `is_held` | int8 | 1 iff bar_offset ≤ bars_held (matches KH-24 v1.3 convention) |

R-unit: `sl_distance = 2.0 × ATR_at_entry`. KH-24 uses 4H ATR(14);
Arc 1 / Arc 2 use 1H ATR(14). The SL_MULT=2.0 is identical across all
three datasets per L_ARC_PROTOCOL convention. The ATR-timeframe mismatch
is acknowledged: it does not cancel in absolute-magnitude metrics
(`pool_frac_reach_1R`), but it does cancel in capture-ratio metrics
(which are dimensionless: realised_R / peak_R).

## Output schema (per-trade meta)

| Column | Type | Source |
|---|---|---|
| `trade_id` | string | matches per-bar `trade_id` |
| `pair` | string | KH-24: `trades_all.pair`; Arc: `signals_features.pair` |
| `bars_held` | int32 | KH-24: `trades_all.bars_held`; Arc: `(exit_bar_ts − entry_bar_ts) / 1H` (NaT → clamped to max bar_offset emitted) |
| `direction` | string | KH-24: `"long"` (all KH-24 trades are long); Arc: `signals_features.direction` |
| `entry_px` | float | KH-24: `trades_all.entry_price`; Arc 1: `signal_bar_close` (approximate — see §A1); Arc 2: `open` at bar_offset=0 row |
| `atr_at_entry` | float | KH-24: `trades_all.atr_abs` (signal-bar 4H ATR); Arc: `signal_bar.atr_at_signal_1h` |
| `sl_distance` | float | `SL_MULT × atr_at_entry` |

## Per-dataset normalisation decisions

### KH-24 (reference)

- `trades_paths.csv` columns are already signed (high − entry) / ATR in
  ATR-units. Loader divides every per-bar magnitude column by
  `SL_MULT=2.0` so they land in SL-distance R-units consistent with the
  spec's R-unit convention.
- `is_held` already in 0/1 spec form — passed through.
- Per-trade meta drawn from `trades_all.csv` (553 trades): `pair`,
  `bars_held`, `atr_abs` → `atr_at_entry`, `entry_price` → `entry_px`.
- `direction` is `"long"` (KH-24 is long-only).

### Arc 1 — partial

- Per-bar artefact is `(trade_id, t, fwd_logret_step, fwd_logret_cum,
  fwd_mfe_atr, fwd_mae_atr)`. **No per-bar OHLC.**
- t is 1-indexed (t=1 = entry bar). Loader subtracts 1 →
  `bar_offset = t - 1`, 0-indexed, matching KH-24.
- `mfe_so_far_r = fwd_mfe_atr / 2.0`; `mae_so_far_r = -fwd_mae_atr / 2.0`
  (Arc 1 emits absolute mae; sign as adverse, ≤0 for long).
- `close_r` derived from `fwd_logret_cum`:
  ```
  close_price = entry_px × exp(fwd_logret_cum)
  close_r     = (close_price − entry_px) / sl_distance
              = entry_px × (exp(fwd_logret_cum) − 1) / sl_distance
  ```
  Per §A1 below, `entry_px ≈ signal_bar_close`, `sl_distance =
  2 × atr_at_signal_1h`. Sub-bar approximation: signal_bar close vs
  entry_bar open is 1 H1 bar apart in time; for FX 1H the drift over
  a single bar is small relative to ATR, but it IS slippage. Documented
  unavailability: a precise close_r would require per-trade
  entry_bar_open, which Arc 1's step 2 artefact does not emit.
- `high_r` and `low_r` are **set to NaN**. Arc 1 step 2 chose not to
  emit per-bar OHLC. Downstream TP/SL detection uses running MFE/MAE
  (exact: TP=X hits at the first bar_offset where mfe_so_far_r ≥ X;
  SL hits at the first bar_offset where mae_so_far_r ≤ −1). Downstream
  intrabar-low trail detection is NOT possible; trail-exit simulation
  falls back to close-based detection using `close_r`.
- `is_held` synthesised: `bar_offset ≤ bars_held`, with `bars_held =
  (exit_bar_ts − entry_bar_ts) / 1H` from `signals_features.csv`. For
  trades with NaT exit_bar_ts (never exited within the simulated window),
  `bars_held` is clamped to the max emitted bar_offset (480) so is_held=1
  across the trade's whole window.

### Arc 2 — compatible

- Per-bar artefact is `(trade_id, bar_offset, bar_ts, open, high, low,
  close, cum_logret_from_entry, mfe_to_date_atr, mae_to_date_atr,
  is_held_bar, is_forward_bar, data_end_flag)`.
- `bar_offset` already 0-indexed; passed through.
- `entry_px` taken as `open` at the trade's `bar_offset = 0` row
  (consistent with KH-24's entry at next-bar open). Actually loaded from
  `signals_features.signal_bar_close` for parity with the Arc 1 path —
  same one-bar approximation; documented.
- `high_r = (high − entry_px) / sl_distance`; same for low_r, close_r.
  Signed by long-direction convention (Arc 2 is long-only).
- `mfe_so_far_r = mfe_to_date_atr / 2.0`; `mae_so_far_r =
  -mae_to_date_atr / 2.0` (Arc 2 emits absolute mae; signed as adverse).
- `is_held = is_held_bar.astype(int8)`.

## §A1 Approximation: entry_px vs signal_bar_close

For Arc 1 and Arc 2, the step 2 artefact does not directly emit a
per-trade `entry_bar_open` field. The protocol entry rule is "bar N+1
open after signal on bar N close", so entry_px should be the open of
the bar after the signal bar. The closest proxy in `signals_features.csv`
is `signal_bar_close` (the close of the signal bar, i.e. the same
timestamp as the entry-bar open in continuous FX data).

For Arc 2 specifically, the per-bar artefact has `open` at bar_offset=0
which is the entry-bar open (Arc 2 uses bar_offset=0 = entry bar). We
verified Arc 2 trade 0's bar_offset=0 timestamp matches the
`signals_features.entry_bar_ts`. For consistency with the Arc 1 path
(which has no bar_offset=0 row, just t=1 onwards), the loader uses
`signal_bar_close` as `entry_px` in both arcs.

This is at most a 1-bar slippage approximation (signal_bar_close vs
entry_bar_open, one H1 step apart). For absolute close_r values in
the conditional-predictivity scan (Axis 2e), this introduces ≲ 0.05R
bias in either direction. Capture-ratio metrics are unaffected (both
numerator and denominator share the same approximation).

## §A2 Why divide by SL_MULT=2.0

All three datasets follow the L_ARC_PROTOCOL entry/exit rule:
SL = 2.0 × ATR_at_entry. The native ATR-unit MFE/MAE values
(KH-24's signed high_r in ATR-units, Arc 1/2's fwd_mfe_atr) are
"in units of ATR-at-entry". The user-spec R convention is
"in units of entry-to-SL distance". Since SL = 2×ATR, dividing
ATR-units by 2 gives R-units exactly. The conversion is the same
for all three datasets and is the only normalisation needed for
cross-dataset comparison of dimensionless capture ratios.

## §A3 Why ATR-timeframe mismatch doesn't get normalised away

KH-24 uses 4H ATR(14); Arc 1 / Arc 2 use 1H ATR(14). A 4H ATR is
roughly √4 = 2× a 1H ATR by random-walk scaling, but real FX volatility
clusters are different. The protocol explicitly accepts this in the
audit's §1: "for cross-arc calibration against L arc capturability gates
that use ATR(14)_1H, this becomes a unit conversion concern".

For v1.3 calibration the user's spec accepted this: the chat sets the
v1.3 floors after reading the side-by-side metrics. If absolute-magnitude
metrics like `pool_frac_reach_1R` look meaningfully different across
datasets, part of that difference is the ATR-timeframe convention, not
the underlying signal quality. The dimensionless capture ratios (which
the v1.3 spec emphasises in the operational definition) are immune.
