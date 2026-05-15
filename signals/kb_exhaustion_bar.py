"""
KB Exhaustion Bar signal — Phase KC.

The signal bar closes COUNTER to the upcoming trend direction, near its extreme,
with significant body size.  Entry fires at the NEXT bar's open.

Long signal  (precedes a long trend):
    1. close < open                                  — bearish bar
    2. abs(close - open) / ATR(14) >= body_threshold — substantial body
    3. (close - low) / (high - low) <= cp_max        — closed near low

Short signal (precedes a short trend):
    1. close > open                                  — bullish bar
    2. abs(close - open) / ATR(14) >= body_threshold — substantial body
    3. (close - low) / (high - low) >= cp_min        — closed near high

ATR(14) uses Wilder smoothing (EWM alpha=1/14, adjust=False).
No lookahead.  Signal at bar N close — entry at bar N+1 open.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── ATR helper ────────────────────────────────────────────────────────────────

def _wilder_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    ATR via Wilder smoothing (EWM, alpha=1/period, adjust=False).
    Returns a Series aligned to df.index, NaN for first (period-1) bars.
    """
    hi = df["high"].astype(float)
    lo = df["low"].astype(float)
    cl = df["close"].astype(float)
    prev_cl = cl.shift(1)

    tr = pd.concat(
        [hi - lo, (hi - prev_cl).abs(), (lo - prev_cl).abs()], axis=1
    ).max(axis=1)

    # Wilder smoothing: alpha = 1/period.  min_periods forces NaN during warmup.
    atr = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    # Align NaN for first (period-1) rows to match Wilder convention
    atr.iloc[: period - 1] = np.nan
    return atr


# ── Signal function ───────────────────────────────────────────────────────────

def kb_exhaustion_bar(
    df: pd.DataFrame,
    *,
    signal_col: str = "c1_signal",
    long_body_threshold: float = 0.5,
    short_body_threshold: float = 0.6,
    long_close_position_max: float = 0.24,
    short_close_position_min: float = 0.77,
    atr_period: int = 14,
    **kwargs,
) -> pd.DataFrame:
    """
    KB exhaustion bar signal.

    Parameters
    ----------
    long_body_threshold       Body size (abs(c-o)/ATR) >= this for a long signal.
    short_body_threshold      Body size >= this for a short signal.
    long_close_position_max   Close position <= this for a long signal (bar closed near low).
    short_close_position_min  Close position >= this for a short signal (bar closed near high).
    atr_period                ATR lookback period (Wilder smoothing).

    Returns
    -------
    df with df[signal_col] in {-1, 0, +1}.
      +1 = long signal  (bearish exhaustion bar precedes a long trend)
      -1 = short signal (bullish exhaustion bar precedes a short trend)
       0 = no signal
    """
    df = df.copy()

    o  = df["open"].astype(float)
    h  = df["high"].astype(float)
    lo = df["low"].astype(float)
    c  = df["close"].astype(float)

    atr = _wilder_atr(df, atr_period)

    bar_range = (h - lo).replace(0.0, np.nan)
    body      = (c - o).abs()
    close_pos = (c - lo) / bar_range
    body_atr  = body / atr.replace(0.0, np.nan)

    # Long signal: bearish bar, large body, closed in bottom cp_max of range
    long_mask = (
        (c < o)
        & (body_atr >= long_body_threshold)
        & (close_pos <= long_close_position_max)
        & atr.notna()
    )

    # Short signal: bullish bar, large body, closed in top (1-cp_min) of range
    short_mask = (
        (c > o)
        & (body_atr >= short_body_threshold)
        & (close_pos >= short_close_position_min)
        & atr.notna()
    )

    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_mask]  =  1
    sig[short_mask] = -1

    # Validate: no NaNs, values only in {-1, 0, +1}
    assert sig.isna().sum() == 0, "kb_exhaustion_bar: NaN in signal column"
    assert set(sig.unique()).issubset({-1, 0, 1}), "kb_exhaustion_bar: unexpected signal values"

    df[signal_col] = sig
    return df
