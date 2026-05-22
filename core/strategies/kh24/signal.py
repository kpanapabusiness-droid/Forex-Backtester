"""KH-24 signal evaluator on v3 bid+ask schema.

Implements conditions C1-C6, C8, C9 per ARC_HISTORY.md and
``scripts/arc_kh24_v2/step1/_signal.py`` (the MT5-schema reference). C7
(volume) is permanently disabled per ``CLAUDE.md``'s eliminated list.

Conditions (long only):

    C1: close < open                              — bearish exhaustion bar
    C2: |close − open| / ATR(14) ≥ body_threshold — substantial body
    C3: (close − low) / (high − low) ≤ cp_max     — closed near low
    C4: close > 4H Kijun(26)                      — above the 4H baseline
    C5: close ≤ 4H Kijun + 1.0 × ATR              — not too extended
    C6: (close − close[N-10]) / ATR ≤ -0.5        — significant 10-bar drop
    C7: DISABLED                                   — volume gate, eliminated
    C8: prev D1 close > prev D1 Kijun(26)         — D1 regime up (lag-1)
    C9: prev D1 close ≤ prev D1 Kijun + 1.0×D1 ATR — D1 not too extended

Inputs use **bid-side single OHLC** (close_bid, high_bid, low_bid,
open_bid) to match the deployed MT5 EA, which reads single-side OHLC
from ``CopyRates`` (MT5 returns the bid-side OHLC by broker
convention). Earlier (pre-PR-E.1.6) v3 ports used mid-OHLC; PR-E.1.6
corrects this per ``docs/dispatches/kh24_ea_full_diff.md`` Section A.

D1 alignment uses the one-day-lag rule: each H4 bar at calendar day T
sees only D1 data from day T-1 or earlier (per L_PROTOCOL §1
non-negotiable on the D1 lag).

Causal lineage: clean — every input is strictly prior to the signal
bar's close, and the bar-N+1-open entry uses ``open_ask`` from PR-B's
fill primitives.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.features._helpers import kijun, wilder_atr


@dataclass(frozen=True)
class KH24SignalParams:
    """Locked KH-24 signal parameters (matches deployed EA)."""

    atr_period: int = 14
    kijun_period: int = 26
    d1_atr_period: int = 14
    d1_kijun_period: int = 26
    # C2/C3 body & close-position thresholds (long only — short side disabled)
    long_body_threshold: float = 0.5
    long_close_position_max: float = 0.24
    # C5 distance cap: close must be ≤ Kijun + cap × ATR
    c5_distance_cap_atr: float = 1.0
    # C6 depth: trailing N-bar drop expressed in ATRs
    c6_depth_bars: int = 10
    c6_depth_threshold: float = 0.5
    # C9 D1 distance cap
    c9_d1_distance_cap_atr: float = 1.0


@dataclass(frozen=True)
class KH24SignalResult:
    """Per-bar signal output.

    ``signal_mask`` is True where all enabled conditions pass.
    ``atr_h4`` and ``kijun_h4`` are returned for the SL / kijun-exit
    callers to reuse (avoid recomputing).
    """

    signal_mask: np.ndarray  # shape (n,) bool
    atr_h4: np.ndarray  # shape (n,) float
    kijun_h4: np.ndarray  # shape (n,) float
    d1_close_lag1: np.ndarray  # shape (n,) float — D1 close from prior day
    d1_kijun_lag1: np.ndarray  # shape (n,) float
    d1_atr_lag1: np.ndarray  # shape (n,) float


def _build_d1_lag1_arrays(
    df_h4: pd.DataFrame, df_d1: pd.DataFrame, params: KH24SignalParams
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align D1 close/Kijun/ATR onto H4 bars using the one-day-lag rule.

    Returns ``(d1_close_lag1, d1_kijun_lag1, d1_atr_lag1)`` — each
    aligned to ``df_h4.index``. Each H4 bar at calendar day T uses the
    D1 bar from day T-1 or earlier.

    Mirrors ``scripts/arc_kh24_v2/step1/_signal._build_d1_lag1_arrays``
    on v3 BID-side single OHLC (matches EA's CopyRates convention).
    """
    d1 = pd.DataFrame(index=df_d1.index.copy())
    d1["d1_close"] = df_d1["close_bid"].values
    d1["d1_kijun"] = kijun(
        df_d1["high_bid"], df_d1["low_bid"], period=params.d1_kijun_period
    ).values
    d1["d1_atr"] = wilder_atr(
        df_d1["high_bid"], df_d1["low_bid"], df_d1["close_bid"], period=params.d1_atr_period
    ).values
    d1["_date"] = d1.index.normalize()
    d1 = d1.drop_duplicates(subset=["_date"], keep="last").reset_index(drop=True)

    # Shift each H4 bar's calendar date back one day; merge_asof backward
    # gives the latest D1 row whose date ≤ H4_date − 1 day.
    shifted = pd.DataFrame(
        {
            "_date": df_h4.index.normalize() - pd.Timedelta(days=1),
            "_idx": np.arange(len(df_h4), dtype=np.int64),
        }
    ).sort_values("_date")
    merged = pd.merge_asof(
        shifted, d1[["_date", "d1_close", "d1_kijun", "d1_atr"]], on="_date", direction="backward"
    )
    merged = merged.sort_values("_idx").reset_index(drop=True)

    return (
        merged["d1_close"].values.astype(float),
        merged["d1_kijun"].values.astype(float),
        merged["d1_atr"].values.astype(float),
    )


def evaluate_kh24_signal(
    df_h4: pd.DataFrame, df_d1: pd.DataFrame, params: KH24SignalParams | None = None
) -> KH24SignalResult:
    """Evaluate the KH-24 signal on a single pair's H4 + D1 frames.

    Parameters
    ----------
    df_h4 : pd.DataFrame
        H4 bid+ask OHLC frame (from ``core.data.aggregator.aggregate(...,
        "H4")``), DatetimeIndex (UTC), columns per
        ``core.data.histdata_loader.M1_COLUMNS``.
    df_d1 : pd.DataFrame
        D1 bid+ask OHLC frame for the same pair.
    params : KH24SignalParams | None
        Locked defaults match the deployed EA; override only for tests
        / probes.

    Returns
    -------
    KH24SignalResult
        Signal mask + the auxiliary series (ATR, Kijun, lag-1 D1) the
        rest of the strategy needs.
    """
    params = params or KH24SignalParams()
    if len(df_h4) == 0:
        empty = np.zeros(0, dtype=bool)
        empty_f = np.zeros(0, dtype=float)
        return KH24SignalResult(
            signal_mask=empty,
            atr_h4=empty_f,
            kijun_h4=empty_f,
            d1_close_lag1=empty_f,
            d1_kijun_lag1=empty_f,
            d1_atr_lag1=empty_f,
        )

    open_bid = df_h4["open_bid"].values
    high_bid = df_h4["high_bid"].values
    low_bid = df_h4["low_bid"].values
    close_bid = df_h4["close_bid"].values

    atr_h4 = wilder_atr(
        df_h4["high_bid"], df_h4["low_bid"], df_h4["close_bid"], period=params.atr_period
    ).values.astype(float)
    kijun_h4 = kijun(df_h4["high_bid"], df_h4["low_bid"], period=params.kijun_period).values.astype(
        float
    )

    d1_close_lag1, d1_kijun_lag1, d1_atr_lag1 = _build_d1_lag1_arrays(df_h4, df_d1, params)

    n = len(df_h4)
    sig = np.zeros(n, dtype=bool)
    warm = max(params.atr_period, params.kijun_period, params.c6_depth_bars)

    bar_range = high_bid - low_bid
    body = np.abs(close_bid - open_bid)
    # close_position is undefined when range is 0 — guard with safe-divide.
    with np.errstate(divide="ignore", invalid="ignore"):
        close_pos = np.where(bar_range > 0, (close_bid - low_bid) / bar_range, np.nan)
        body_atr = np.where(atr_h4 > 0, body / atr_h4, np.nan)

    for i in range(warm, n):
        a = atr_h4[i]
        if not np.isfinite(a) or a <= 0:
            continue
        k = kijun_h4[i]
        if not np.isfinite(k):
            continue

        # C1: bearish bar
        if not (close_bid[i] < open_bid[i]):
            continue
        # C2: substantial body
        if not np.isfinite(body_atr[i]) or body_atr[i] < params.long_body_threshold:
            continue
        # C3: closed near low
        if not np.isfinite(close_pos[i]) or close_pos[i] > params.long_close_position_max:
            continue
        # C4: close > 4H Kijun
        if not (close_bid[i] > k):
            continue
        # C5: close ≤ Kijun + 1.0 × ATR
        if close_bid[i] > k + params.c5_distance_cap_atr * a:
            continue
        # C6: 10-bar drop ≥ 0.5 × ATR
        if i < params.c6_depth_bars:
            continue
        depth = (close_bid[i] - close_bid[i - params.c6_depth_bars]) / a
        if depth > -params.c6_depth_threshold:
            continue
        # C8: prev D1 close > prev D1 Kijun
        d1c = d1_close_lag1[i]
        d1k = d1_kijun_lag1[i]
        if not (np.isfinite(d1c) and np.isfinite(d1k)):
            continue
        if not (d1c > d1k):
            continue
        # C9: prev D1 close ≤ prev D1 Kijun + 1.0 × prev D1 ATR
        d1a = d1_atr_lag1[i]
        if not (np.isfinite(d1a) and d1a > 0):
            continue
        if d1c > d1k + params.c9_d1_distance_cap_atr * d1a:
            continue

        sig[i] = True

    return KH24SignalResult(
        signal_mask=sig,
        atr_h4=atr_h4,
        kijun_h4=kijun_h4,
        d1_close_lag1=d1_close_lag1,
        d1_kijun_lag1=d1_kijun_lag1,
        d1_atr_lag1=d1_atr_lag1,
    )
