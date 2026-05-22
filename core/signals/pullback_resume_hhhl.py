"""Arc 8 signal: pullback-and-resume in an HH/HL uptrend (long).

Spec: ``docs/archive/signal_specs/signal_spec_pullback_resume_hhhl_long_v0.1.md``

Mechanics (locked at arc-open):

  Swing definitions (3-bar local extreme, confirmation lag = 3 bars):
    - swing_high(k) = high[k] > max(high[k-3..k-1]) AND high[k] > max(high[k+1..k+3])
    - swing_low(k)  = low[k]  < min(low[k-3..k-1])  AND low[k]  < min(low[k+1..k+3])

  Trend established in window t-30..t-1:
    - >= 2 swing-highs in window, strictly ascending  (HH)
    - >= 2 swing-lows  in window, strictly ascending  (HL)
    - Right-edge constraint: most recent identifiable swing-high at most bar
      t-4 (confirmation lag). Same for swing-lows.

  Pullback:
    close[t-1] <= most_recent_swing_high - 0.5 * ATR(14)[t-1]

  Resume trigger at bar t (close):
    close[t] > open[t]                  (bullish close)
    close[t] > high[t-1]                (breaks prior bar high)
    (close[t] - low[t]) / (high[t] - low[t]) >= 0.5   (close in upper half)

  Spacing:
    >= 20 bars since last signal on this pair.

Causal lineage (Step 6 audit fodder):
  - Swing detection uses k+1..k+3 lookahead WITHIN THE DETECTION WINDOW only;
    no bar k > t-4 enters trigger evaluation at bar t. Right-edge audit
    (mandatory at Step 1) verifies this.
  - ATR(14) is shift(1) Wilder on mid OHLC (strictly prior bars).
  - All trigger fields (close[t], open[t], high[t-1], low[t]) read at bar
    close — no forward leakage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PullbackResumeParams:
    """Locked parameters per signal spec v0.1."""

    swing_lookback: int = 3                # k-3..k-1 and k+1..k+3
    trend_window_bars: int = 30            # t-30..t-1
    right_edge_lag: int = 4                # most recent swing at most bar t-right_edge_lag
    pullback_atr_mult: float = 0.5         # close[t-1] <= swing_high - 0.5*ATR
    upper_half_threshold: float = 0.5      # (close - low) / (high - low) >= 0.5
    spacing_bars: int = 20                 # >= 20 bars since last signal
    atr_period: int = 14                   # Wilder ATR


@dataclass(frozen=True)
class PullbackResumeSignalResult:
    """Per-bar evaluation output.

    All series share ``pair_df.index``. ``signal_mask`` is True on bars
    where a trigger fires after spacing enforcement. ``atr_h4`` is the
    Wilder ATR(14) computed on shift(1) mid OHLC (strictly prior bars).

    Diagnostic fields preserved for the Step 1 integrity report:

    - ``trend_ok`` — HH/HL trend established in window
    - ``pullback_ok`` — pullback bar passed the ATR-relative gate
    - ``resume_ok`` — trigger conditions met at bar t (raw, pre-spacing)
    - ``most_recent_swing_high_bar`` — index of the swing-high used at
      each evaluation (NaN if none); used by the right-edge audit.
    - ``most_recent_swing_high_price`` — the swing-high price used in
      the pullback gate (NaN if no eligible swing).
    """

    signal_mask: pd.Series
    atr_h4: pd.Series
    trend_ok: pd.Series
    pullback_ok: pd.Series
    resume_ok: pd.Series
    most_recent_swing_high_bar: pd.Series
    most_recent_swing_high_price: pd.Series


# ── helpers ──────────────────────────────────────────────────────────


def _wilder_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Wilder's ATR. Returns NaN for the first ``period`` bars."""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    prev_close = np.empty(n)
    prev_close[0] = np.nan
    prev_close[1:] = close[:-1]
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close),
    ])
    atr = np.full(n, np.nan)
    # Seed with simple average of first ``period`` true ranges
    seed = np.nanmean(tr[1:period + 1])
    atr[period] = seed
    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _is_swing(arr: np.ndarray, side: str, lookback: int) -> np.ndarray:
    """Vectorised 3-bar local extreme.

    side='high': arr[k] > all of arr[k-lookback..k-1] AND arr[k+1..k+lookback]
    side='low':  arr[k] < all of arr[k-lookback..k-1] AND arr[k+1..k+lookback]

    Uses k+1..k+lookback forward bars — caller responsible for not using
    detected swings at bar k when k > t - (lookback + 1).
    """
    n = len(arr)
    is_swing = np.zeros(n, dtype=bool)
    if n < 2 * lookback + 1:
        return is_swing
    cmp = (lambda a, b: a > b) if side == "high" else (lambda a, b: a < b)
    for k in range(lookback, n - lookback):
        v = arr[k]
        ok = True
        for j in range(1, lookback + 1):
            if not cmp(v, arr[k - j]) or not cmp(v, arr[k + j]):
                ok = False
                break
        is_swing[k] = ok
    return is_swing


# ── public API ───────────────────────────────────────────────────────


def evaluate_pullback_resume_hhhl_signal(
    pair_df: pd.DataFrame,
    params: PullbackResumeParams | None = None,
) -> PullbackResumeSignalResult:
    """Evaluate the PR-HHHL trigger on one pair's H4 series.

    Parameters
    ----------
    pair_df : DataFrame
        UTC-indexed H4 bid+ask OHLC frame from
        ``core.data.histdata_loader`` / ``core.data.aggregator``. Required
        columns: open_bid, high_bid, low_bid, close_bid, open_ask, high_ask,
        low_ask, close_ask.
    params : PullbackResumeParams | None
        Defaults to spec-locked values.

    Returns
    -------
    PullbackResumeSignalResult
    """
    p = params or PullbackResumeParams()
    n = len(pair_df)
    idx = pair_df.index

    # Trigger evaluation uses bid OHLC (MT5 chart convention). ATR on mid.
    open_bid = pair_df["open_bid"].to_numpy(dtype=float)
    high_bid = pair_df["high_bid"].to_numpy(dtype=float)
    low_bid = pair_df["low_bid"].to_numpy(dtype=float)
    close_bid = pair_df["close_bid"].to_numpy(dtype=float)
    high_mid = (pair_df["high_bid"].to_numpy(dtype=float) + pair_df["high_ask"].to_numpy(dtype=float)) / 2.0
    low_mid = (pair_df["low_bid"].to_numpy(dtype=float) + pair_df["low_ask"].to_numpy(dtype=float)) / 2.0
    close_mid = (pair_df["close_bid"].to_numpy(dtype=float) + pair_df["close_ask"].to_numpy(dtype=float)) / 2.0

    # ATR(14) on mid, shifted 1 — value at bar t reflects bars closing at t-1.
    atr_raw = _wilder_atr(high_mid, low_mid, close_mid, p.atr_period)
    atr_shift1 = np.empty(n)
    atr_shift1[0] = np.nan
    atr_shift1[1:] = atr_raw[:-1]

    # Swing detection on bid OHLC (chart-visible).
    is_swing_high = _is_swing(high_bid, "high", p.swing_lookback)
    is_swing_low = _is_swing(low_bid, "low", p.swing_lookback)

    trend_ok = np.zeros(n, dtype=bool)
    pullback_ok = np.zeros(n, dtype=bool)
    resume_ok = np.zeros(n, dtype=bool)
    most_recent_sh_bar = np.full(n, np.nan)
    most_recent_sh_price = np.full(n, np.nan)

    min_t = p.trend_window_bars + p.right_edge_lag  # need t-30 lookback + right-edge lag

    # Pre-compute lists of (bar_idx, price) for swing-highs / swing-lows for
    # fast windowed scans.
    sh_bars = np.where(is_swing_high)[0]
    sl_bars = np.where(is_swing_low)[0]
    sh_prices = high_bid[sh_bars]
    sl_prices = low_bid[sl_bars]

    # Vectorised would be possible but the per-trade work is small; this is
    # 28 pairs × ~26k bars = ~730k iterations. Inner ops are constant-time
    # via np.searchsorted.
    for t in range(min_t, n):
        win_lo = t - p.trend_window_bars              # t-30 inclusive
        win_hi = t - p.right_edge_lag                 # t-4 inclusive
        # Swings strictly within [win_lo, win_hi]
        lo_idx = np.searchsorted(sh_bars, win_lo, side="left")
        hi_idx = np.searchsorted(sh_bars, win_hi + 1, side="left")
        sh_in_win = sh_bars[lo_idx:hi_idx]
        sh_p_in_win = sh_prices[lo_idx:hi_idx]
        if len(sh_in_win) < 2:
            continue
        lo_idx_l = np.searchsorted(sl_bars, win_lo, side="left")
        hi_idx_l = np.searchsorted(sl_bars, win_hi + 1, side="left")
        sl_in_win = sl_bars[lo_idx_l:hi_idx_l]
        sl_p_in_win = sl_prices[lo_idx_l:hi_idx_l]
        if len(sl_in_win) < 2:
            continue
        # Strictly ascending HH
        if not np.all(np.diff(sh_p_in_win) > 0):
            continue
        # Strictly ascending HL
        if not np.all(np.diff(sl_p_in_win) > 0):
            continue
        trend_ok[t] = True
        # Most recent swing-high in window
        last_sh_bar = int(sh_in_win[-1])
        last_sh_price = float(sh_p_in_win[-1])
        most_recent_sh_bar[t] = last_sh_bar
        most_recent_sh_price[t] = last_sh_price
        # Pullback gate at bar t-1
        atr_tm1 = atr_shift1[t - 1]  # ATR computed on bars up to t-2 — strictly prior
        if not np.isfinite(atr_tm1) or atr_tm1 <= 0:
            continue
        if close_bid[t - 1] > last_sh_price - p.pullback_atr_mult * atr_tm1:
            continue
        pullback_ok[t] = True
        # Resume trigger at bar t
        if not (close_bid[t] > open_bid[t]):
            continue
        if not (close_bid[t] > high_bid[t - 1]):
            continue
        rng = high_bid[t] - low_bid[t]
        if rng <= 0:
            continue
        if (close_bid[t] - low_bid[t]) / rng < p.upper_half_threshold:
            continue
        resume_ok[t] = True

    # Spacing — keep only signals >= spacing_bars after the last accepted one.
    signal_mask = np.zeros(n, dtype=bool)
    last_accepted = -10_000
    for t in np.where(resume_ok)[0]:
        if (t - last_accepted) >= p.spacing_bars:
            signal_mask[t] = True
            last_accepted = t

    return PullbackResumeSignalResult(
        signal_mask=pd.Series(signal_mask, index=idx, name="signal_mask"),
        atr_h4=pd.Series(atr_shift1, index=idx, name="atr_14_shift1"),
        trend_ok=pd.Series(trend_ok, index=idx, name="trend_ok"),
        pullback_ok=pd.Series(pullback_ok, index=idx, name="pullback_ok"),
        resume_ok=pd.Series(resume_ok, index=idx, name="resume_ok"),
        most_recent_swing_high_bar=pd.Series(most_recent_sh_bar, index=idx, name="most_recent_sh_bar"),
        most_recent_swing_high_price=pd.Series(most_recent_sh_price, index=idx, name="most_recent_sh_price"),
    )
