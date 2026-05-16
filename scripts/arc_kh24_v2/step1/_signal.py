"""Bare kb_exhaustion_bar signal evaluator for the KH-24 v2.0 self-test.

Implements conditions C1-C6, C8, C9 (long-only) per docs/KH24_SYSTEM_LOCK.md.
C7 (volume) is permanently disabled. C1-C3 are evaluated via
signals.kb_exhaustion_bar (gate-locked, do NOT modify); C4-C6 and C8/C9 are
layered on top here.

D1 alignment uses a one-day lag (pre-shifted date + merge_asof backward),
so each 4H bar at calendar day T sees only D1 data from day T-1 or earlier.
Never same-day, never forward-fill.

Read-only across the data series. Pure function: same inputs → same outputs.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

from signals.kb_exhaustion_bar import _wilder_atr, kb_exhaustion_bar


class SignalParams(NamedTuple):
    atr_period: int = 14
    kijun_period: int = 26
    d1_atr_period: int = 14
    d1_kijun_period: int = 26
    long_body_threshold: float = 0.5
    long_close_position_max: float = 0.24
    c5_distance_cap_atr: float = 1.0
    c6_depth_bars: int = 10
    c6_depth_threshold: float = 0.5
    c9_d1_distance_cap_atr: float = 1.0


def _kijun(df: pd.DataFrame, period: int) -> np.ndarray:
    """Ichimoku Kijun-sen: (highest high + lowest low) / 2 over `period` bars.

    NOT a moving average. NaN for the first (period - 1) bars.
    """
    hi_max = df["high"].rolling(window=period, min_periods=period).max()
    lo_min = df["low"].rolling(window=period, min_periods=period).min()
    return ((hi_max + lo_min) / 2.0).values.astype(float)


def _build_d1_lag1_arrays(
    df_4h: pd.DataFrame, df_d1: pd.DataFrame, params: SignalParams
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return per-4H-bar arrays aligned to the previous calendar day's D1 bar.

    Returns
    -------
    d1_close_lag1   : D1 close on day strictly < 4H bar's calendar day
    d1_kijun_lag1   : D1 Kijun(d1_kijun_period) on that same lag-1 D1 bar
    d1_atr_lag1     : D1 Wilder ATR(d1_atr_period) on that same lag-1 D1 bar
    d1_date_lag1    : the lag-1 D1 calendar date (datetime64[ns])

    Method: shift the 4H normalised date back one calendar day, then merge_asof
    backward against the D1 series. This is strict prior — never same-day.
    """
    n = len(df_4h)
    d1 = df_d1.copy()
    d1["date_norm"] = pd.to_datetime(d1["date"]).dt.normalize()
    d1 = d1.sort_values("date_norm").drop_duplicates("date_norm").reset_index(drop=True)
    d1["d1_close"] = d1["close"].astype(float)
    d1["d1_kijun"] = _kijun(d1, params.d1_kijun_period)
    d1["d1_atr"] = _wilder_atr(d1, params.d1_atr_period).values.astype(float)
    d1_compact = d1[["date_norm", "d1_close", "d1_kijun", "d1_atr"]].rename(
        columns={"date_norm": "date"}
    )

    # Shift 4H calendar date back one day; merge_asof backward gives the latest
    # D1 row whose date <= (4H_date - 1 day). Strict prior, never same-day.
    shifted = pd.DataFrame(
        {
            "date": pd.to_datetime(df_4h["date"]).dt.normalize() - pd.Timedelta(days=1),
            "_idx": np.arange(n, dtype=np.int64),
        }
    ).sort_values("date")

    merged = pd.merge_asof(shifted, d1_compact, on="date", direction="backward")
    merged = merged.sort_values("_idx").reset_index(drop=True)

    return (
        merged["d1_close"].values.astype(float),
        merged["d1_kijun"].values.astype(float),
        merged["d1_atr"].values.astype(float),
        merged["date"].values.astype("datetime64[ns]"),
    )


def evaluate_bare_signal(
    df_4h: pd.DataFrame,
    df_d1: pd.DataFrame,
    params: SignalParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the bare kb_exhaustion_bar signal (C1-C6, C8, C9) per 4H bar.

    Returns
    -------
    sig_mask        : boolean array, True where all conditions pass.
    atr_4h          : 4H Wilder ATR(period) — pinned values per bar.
    d1_date_lag1    : the D1 calendar date used for C8/C9 per 4H bar (datetime64[ns]).
                      Useful for the D1 lag-1 invariant test.
    fired_per_step  : dict-like with per-condition surviving counts for the report.
                      (Returned via a separate helper.)
    """
    df = df_4h.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # C1-C3 from gate-locked kb_exhaustion_bar (long signal = +1).
    df = kb_exhaustion_bar(
        df,
        signal_col="_c123",
        long_body_threshold=params.long_body_threshold,
        long_close_position_max=params.long_close_position_max,
        atr_period=params.atr_period,
    )
    c123 = df["_c123"].values == 1

    atr_4h = _wilder_atr(df, params.atr_period).values.astype(float)
    kijun_4h = _kijun(df, params.kijun_period)
    c = df["close"].values.astype(float)

    n = len(df)
    sig = np.zeros(n, dtype=bool)

    d1_close_lag1, d1_kijun_lag1, d1_atr_lag1, d1_date_lag1 = _build_d1_lag1_arrays(
        df, df_d1, params
    )

    warm_floor = max(
        params.atr_period,
        params.kijun_period,
        params.c6_depth_bars,
    )

    # Vectorised body of C4, C5, C6 (4H-only); evaluate each row individually
    # to keep the read-discipline obvious and the test surface small.
    for i in range(warm_floor, n):
        if not c123[i]:
            continue
        a = atr_4h[i]
        if not np.isfinite(a) or a <= 0:
            continue
        k = kijun_4h[i]
        if not np.isfinite(k):
            continue

        # C4: close > 4H Kijun
        if c[i] <= k:
            continue
        # C5: close <= 4H Kijun + 1.0 * ATR
        if c[i] > k + params.c5_distance_cap_atr * a:
            continue
        # C6: (close - close[N-10]) / ATR <= -0.5
        if i < params.c6_depth_bars:
            continue
        if (c[i] - c[i - params.c6_depth_bars]) / a > -params.c6_depth_threshold:
            continue

        # C8: prev D1 close > prev D1 Kijun (one-day lag)
        d1c = d1_close_lag1[i]
        d1k = d1_kijun_lag1[i]
        if not (np.isfinite(d1c) and np.isfinite(d1k)):
            continue
        if not (d1c > d1k):
            continue

        # C9: prev D1 close <= prev D1 Kijun + 1.0 * prev D1 ATR
        d1a = d1_atr_lag1[i]
        if not (np.isfinite(d1a) and d1a > 0):
            continue
        if d1c > d1k + params.c9_d1_distance_cap_atr * d1a:
            continue

        sig[i] = True

    return sig, atr_4h, d1_date_lag1
