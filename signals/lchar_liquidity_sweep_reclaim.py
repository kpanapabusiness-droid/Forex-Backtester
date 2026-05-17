"""Arc 7 liquidity-sweep + reclaim long signal.

Signal definition (locked, copied verbatim from ARC_7_LIVE Step 1 dispatch):

  Long signal fires at close of bar t iff ALL of:
    1. swing_low_N = min(low[t-N..t-1])   N = 20 (strictly past bars)
    2. low[t]   <  swing_low_N             (price swept below past swing low)
    3. close[t] >  swing_low_N             (and reclaimed back above)
    4. swing_low_N − low[t] >= 0.25 × ATR(14)[t]   (sweep magnitude floor)
    5. close[t] > open[t]                  (bullish bar)
    6. (close[t] − swing_low_N) / (swing_low_N − low[t]) >= 0.5
                                            (reclaim strength ratio floor)
    7. >= 20 bars since last signal on this pair  (refractory)

  ATR(14) is Wilder-smoothed on 4H bars, causal (uses TR values 1..t only).

Diagnostic side-output: per Step 1 dispatch Open-Q #1, we also emit
`prefilter_pass` and `reclaim_ratio` for every bar where conditions 1-5 + 7
pass — i.e. the population the reclaim_ratio >= 0.5 filter (condition 6) is
applied to. Refractory under condition 7 uses prior full-signal fires (the
natural "without filter 6" interpretation: every gate other than 6 is at its
real setting).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Locked parameters (do not reinterpret; mirror in configs/wfo_l_arc_7.yaml).
SWING_LOOKBACK_N: int = 20
ATR_PERIOD: int = 14
SWEEP_MAGNITUDE_ATR_MIN: float = 0.25
RECLAIM_RATIO_MIN: float = 0.5
REFRACTORY_BARS: int = 20


def _wilder_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> np.ndarray:
    """Causal Wilder ATR(period). Uses TR values <= t only."""
    high = df["high"].astype(float).to_numpy()
    low = df["low"].astype(float).to_numpy()
    close = df["close"].astype(float).to_numpy()
    n = len(df)
    if n == 0:
        return np.array([], dtype=float)
    prev_close = np.empty(n, dtype=float)
    prev_close[0] = np.nan
    prev_close[1:] = close[:-1]
    tr = np.maximum.reduce(
        [
            high - low,
            np.abs(high - prev_close),
            np.abs(low - prev_close),
        ]
    )
    tr[0] = high[0] - low[0]
    atr = np.full(n, np.nan, dtype=float)
    if n < period:
        return atr
    atr[period - 1] = float(np.mean(tr[:period]))
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def compute_signal(
    df_4h: pd.DataFrame,
    *,
    n_lookback: int = SWING_LOOKBACK_N,
    atr_period: int = ATR_PERIOD,
    sweep_min_atr: float = SWEEP_MAGNITUDE_ATR_MIN,
    reclaim_min: float = RECLAIM_RATIO_MIN,
    refractory: int = REFRACTORY_BARS,
    signal_col: str = "signal",
) -> pd.DataFrame:
    """Compute Arc 7 signal on a chronologically-sorted 4H DataFrame.

    Returns a copy of ``df_4h`` with these added columns:
      - signal             bool — all 7 conditions met
      - prefilter_pass     bool — conditions 1-5 + 7 met (pre-filter set)
      - reclaim_ratio      float — (close-sw)/(sw-low) where 1-5 pass; NaN else
      - swing_low_N        float — min(low[t-N..t-1])
      - sweep_atr          float — (swing_low_N - low[t]) / ATR(14)[t]
      - atr14              float — causal Wilder ATR(14)
    """
    if df_4h.empty:
        out = df_4h.copy()
        for col, dtype in [
            (signal_col, bool),
            ("prefilter_pass", bool),
            ("reclaim_ratio", float),
            ("swing_low_N", float),
            ("sweep_atr", float),
            ("atr14", float),
        ]:
            out[col] = pd.Series(dtype=dtype)
        return out

    df = df_4h.reset_index(drop=True).copy()
    n = len(df)
    low = df["low"].astype(float).to_numpy()
    close = df["close"].astype(float).to_numpy()
    open_ = df["open"].astype(float).to_numpy()

    atr = _wilder_atr(df, atr_period)
    # Trailing min of past n_lookback bars, strictly past (shift 1).
    swing_low = (
        pd.Series(low).shift(1).rolling(n_lookback, min_periods=n_lookback).min().to_numpy()
    )

    signal = np.zeros(n, dtype=bool)
    prefilter = np.zeros(n, dtype=bool)
    reclaim_ratio = np.full(n, np.nan, dtype=float)
    sweep_atr_arr = np.full(n, np.nan, dtype=float)

    last_signal_t: int = -(10**9)

    for t in range(n):
        sw = swing_low[t]
        a = atr[t]
        if not np.isfinite(sw) or not np.isfinite(a) or a <= 0:
            continue
        if not (low[t] < sw):
            continue                         # cond 2
        if not (close[t] > sw):
            continue                         # cond 3
        sweep_mag = sw - low[t]              # > 0 by cond 2
        if not (sweep_mag >= sweep_min_atr * a):
            continue                         # cond 4
        if not (close[t] > open_[t]):
            continue                         # cond 5

        # Conditions 1-5 pass; compute reclaim ratio + sweep magnitude in ATR.
        denom = sweep_mag
        r = (close[t] - sw) / denom if denom > 0 else np.nan
        reclaim_ratio[t] = r
        sweep_atr_arr[t] = sweep_mag / a

        # Condition 7: refractory uses prior FULL-signal fires.
        if (t - last_signal_t) < refractory:
            continue
        prefilter[t] = True

        # Condition 6: reclaim strength.
        if r >= reclaim_min:
            signal[t] = True
            last_signal_t = t

    out = df_4h.reset_index(drop=True).copy()
    out[signal_col] = signal
    out["prefilter_pass"] = prefilter
    out["reclaim_ratio"] = reclaim_ratio
    out["swing_low_N"] = swing_low
    out["sweep_atr"] = sweep_atr_arr
    out["atr14"] = atr
    return out
