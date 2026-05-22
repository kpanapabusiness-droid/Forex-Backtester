"""Shared helpers for v3.0 feature producers.

Conventions:
- All producers consume a ``pair_df`` indexed by UTC DatetimeIndex with
  the canonical schema from ``core.data.histdata_loader.M1_COLUMNS``.
- All producers return a Series aligned to ``pair_df.index``.
- Lookahead safety: every producer that uses a rolling window calls
  ``shift(1)`` on the input before computing so the value at time t
  only depends on bars closed strictly before t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def wilder_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """Wilder's ATR via exponential smoothing on the true range.

    Returns a Series aligned to ``close.index``. NaN for the first
    ``period - 1`` bars (insufficient history).
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def kijun(high: pd.Series, low: pd.Series, period: int = 26) -> pd.Series:
    """Ichimoku Kijun-sen: (highest_high + lowest_low) / 2 over ``period`` bars."""
    return (
        high.rolling(window=period, min_periods=period).max()
        + low.rolling(window=period, min_periods=period).min()
    ) / 2.0


def mid_close(df: pd.DataFrame) -> pd.Series:
    """Per-bar mid-close from bid+ask: (close_bid + close_ask) / 2."""
    return (df["close_bid"] + df["close_ask"]) / 2.0


def mid_high(df: pd.DataFrame) -> pd.Series:
    return (df["high_bid"] + df["high_ask"]) / 2.0


def mid_low(df: pd.DataFrame) -> pd.Series:
    return (df["low_bid"] + df["low_ask"]) / 2.0


def percentile_rank_in_window(series: pd.Series, window: int) -> pd.Series:
    """Trailing percentile rank of the current value within the prior ``window``
    bars (excluding the current bar — strictly prior).

    Returns NaN for the first ``window`` bars.
    """
    shifted = series.shift(1)

    def _rank(arr: np.ndarray) -> float:
        if len(arr) == 0:
            return float("nan")
        cur = arr[-1]
        # Use the *prior* window for the rank base; current bar is the last one
        base = arr[:-1]
        if len(base) == 0 or np.isnan(cur):
            return float("nan")
        # Drop NaN from base
        base = base[~np.isnan(base)]
        if len(base) == 0:
            return float("nan")
        return float(np.mean(base <= cur))

    return shifted.rolling(window=window, min_periods=window).apply(_rank, raw=True)


def pip_size_for_pair(pair: str) -> float:
    """0.01 for JPY-quoted pairs, 0.0001 otherwise.

    Same convention as ``core.utils.get_pip_size`` for the v3 layer.
    """
    return 0.01 if pair.endswith("JPY") else 0.0001
