"""Arc 11 — swing-high breakout in trend, long.

Signal spec (locked verbatim from
docs/signal_spec_swing_high_breakout_trend_long_v0.1.md / Downloads spec):

  Swing definitions (3-bar local extreme):
    swing-high at k iff high[k] > max(high[k-3..k-1]) AND high[k] > max(high[k+1..k+3])
    swing-low  at k iff low[k]  < min(low[k-3..k-1])  AND low[k]  < min(low[k+1..k+3])

  Trend filter (structural, no MA — same convention as Arc 9):
    1a. Identify swing-lows in window t-30..t-1 (only those with k+3 <= t-1,
        i.e. k <= t-4 — right-edge auditable).
    1b. Require >= 1 such swing-low exists.
    1c. Require close[t-1] > min(swing_low values in window).

  Reference swing-high:
    2a. Identify swing-highs in window t-20..t-1 with k <= t-4 (right-edge
        auditable).
    2b. H_ref = most recent identifiable swing-high.
    2c. Require H_ref exists (signal else None).

  Break trigger at bar t (long signal):
    3a. close[t] > H_ref + 0.10 * ATR(14)_4H[t]      (decisive break w/ buffer)
    3b. close[t] > open[t]                            (bullish close)
    3c. (close[t] - low[t]) / (high[t] - low[t]) >= 0.5   (close upper half)

  Spacing & entry:
    4a. >= 20 bars since last signal on this pair (refractory)
    4b. Entry: bar t+1 open (next-open per SPREAD_SEMANTICS_LOCK)

ATR(14) is Wilder-smoothed on 4H bars, causal (uses TR values 1..t only).

Right-edge audit (mandatory per dispatch + spec): both swing-high (H_ref)
and swing-low (trend filter) use k+1..k+3 lookahead within the DETECTION
WINDOW only — the t-4 right edge guarantees no future-bar leakage at
trigger evaluation time. step1_backtest audits this directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Locked parameters (mirrored in configs/wfo_l_arc_11.yaml).
SWING_K: int = 3                          # 3-bar local extreme on each side
TREND_FILTER_LOOKBACK: int = 30           # swing-low search window: t-30..t-1
H_REF_LOOKBACK: int = 20                  # swing-high search window: t-20..t-1
RIGHT_EDGE_OFFSET: int = 4                # swing detectable iff k <= t - RIGHT_EDGE_OFFSET
ATR_PERIOD: int = 14
BREAK_BUFFER_ATR: float = 0.10            # close[t] > H_ref + 0.10 * ATR
CLOSE_UPPER_HALF_MIN: float = 0.5         # (c-l)/(h-l) >= 0.5
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


def _detect_swings(
    high: np.ndarray, low: np.ndarray, k: int = SWING_K
) -> tuple[np.ndarray, np.ndarray]:
    """Return (swing_high_mask, swing_low_mask) per bar.

    swing-high at position i (i in [k, n-k-1]) iff:
        high[i] > max(high[i-k..i-1]) AND high[i] > max(high[i+1..i+k]).
    Same for swing-low with low + min.

    Vectorised via pandas rolling. Boundary bars (i < k or i > n-k-1) are False.
    """
    n = high.size
    sh = np.zeros(n, dtype=bool)
    sl = np.zeros(n, dtype=bool)
    if n < 2 * k + 1:
        return sh, sl

    s_high = pd.Series(high)
    s_low = pd.Series(low)

    # Left side: max(high[i-k..i-1]) → rolling(k).max() shifted by 1 forward.
    left_max_high = s_high.rolling(k, min_periods=k).max().shift(1).to_numpy()
    left_min_low = s_low.rolling(k, min_periods=k).min().shift(1).to_numpy()

    # Right side: max(high[i+1..i+k]). Equivalently, on the reversed series
    # the left side is max(high[i+1..i+k]); use shift(-k) + rolling forward.
    # Easiest: reverse, rolling, reverse back.
    right_max_high = (
        s_high[::-1].rolling(k, min_periods=k).max().shift(1).to_numpy()[::-1]
    )
    right_min_low = (
        s_low[::-1].rolling(k, min_periods=k).min().shift(1).to_numpy()[::-1]
    )

    for i in range(k, n - k):
        lh = left_max_high[i]
        rh = right_max_high[i]
        if np.isfinite(lh) and np.isfinite(rh):
            if high[i] > lh and high[i] > rh:
                sh[i] = True
        ll = left_min_low[i]
        rl = right_min_low[i]
        if np.isfinite(ll) and np.isfinite(rl):
            if low[i] < ll and low[i] < rl:
                sl[i] = True
    return sh, sl


def compute_signal(
    df_4h: pd.DataFrame,
    *,
    signal_col: str = "signal",
) -> pd.DataFrame:
    """Compute Arc 11 SHB signal on a chronologically-sorted 4H DataFrame.

    Returns a copy of df_4h with these added columns:
      - signal              bool — all gating conditions met
      - prefilter_pass      bool — trend filter PASS + H_ref exists + bullish
                                   close + decisive break (i.e. all except
                                   close-in-upper-half + refractory). Used for
                                   the dispatch's diagnostic surface.
      - h_ref               float — most recent identifiable swing-high in window
      - h_ref_bar_offset    int   — bars between H_ref bar and signal bar t (NaN
                                    if no H_ref)
      - break_magnitude_atr float — (close[t] - H_ref) / ATR(14)[t]; NaN if no H_ref
      - close_position      float — (close[t] - low[t]) / (high[t] - low[t])
      - trend_filter_swing_low float — min(swing-low values in window t-30..t-4); NaN if none
      - atr14               float — causal Wilder ATR(14)

    Note: swing detection uses k+1..k+3 lookahead within the detection window
    only. The t-4 right-edge constraint (k <= t-4) ensures no future-bar
    information leaks into trigger evaluation at bar t.
    """
    if df_4h.empty:
        out = df_4h.copy()
        out[signal_col] = pd.Series(dtype=bool)
        for col, dtype in [
            ("prefilter_pass", bool),
            ("h_ref", float),
            ("h_ref_bar_offset", float),
            ("break_magnitude_atr", float),
            ("close_position", float),
            ("trend_filter_swing_low", float),
            ("atr14", float),
        ]:
            out[col] = pd.Series(dtype=dtype)
        return out

    df = df_4h.reset_index(drop=True).copy()
    n = len(df)
    high = df["high"].astype(float).to_numpy()
    low = df["low"].astype(float).to_numpy()
    close = df["close"].astype(float).to_numpy()
    open_ = df["open"].astype(float).to_numpy()

    atr = _wilder_atr(df, ATR_PERIOD)
    sh_mask, sl_mask = _detect_swings(high, low, SWING_K)

    sh_positions = np.where(sh_mask)[0]
    sl_positions = np.where(sl_mask)[0]

    signal = np.zeros(n, dtype=bool)
    prefilter = np.zeros(n, dtype=bool)
    h_ref_arr = np.full(n, np.nan, dtype=float)
    h_ref_offset = np.full(n, np.nan, dtype=float)
    break_mag = np.full(n, np.nan, dtype=float)
    close_pos = np.full(n, np.nan, dtype=float)
    tf_low = np.full(n, np.nan, dtype=float)

    last_signal_t: int = -(10 ** 9)

    # Pointer indices for efficient window queries (positions are sorted).
    # For each t, swing positions in (t-W..t-4] are a contiguous slice.
    for t in range(n):
        rng = high[t] - low[t]
        if rng > 0:
            cp = (close[t] - low[t]) / rng
        else:
            cp = np.nan
        close_pos[t] = cp

        a = atr[t]
        if not np.isfinite(a) or a <= 0:
            continue

        # Right-edge: only swings with k <= t - RIGHT_EDGE_OFFSET are identifiable.
        right_edge = t - RIGHT_EDGE_OFFSET
        if right_edge < SWING_K:
            continue

        # Trend filter window: swing-lows with t - TREND_FILTER_LOOKBACK <= k <= right_edge.
        tf_lo = t - TREND_FILTER_LOOKBACK
        # Boolean filter (signal evaluation is not in the hot path; n*sl_count small).
        sl_in_window = sl_positions[(sl_positions >= tf_lo) & (sl_positions <= right_edge)]
        if sl_in_window.size == 0:
            continue
        sl_vals = low[sl_in_window]
        sl_min = float(sl_vals.min())
        tf_low[t] = sl_min
        # close[t-1] > min(sl_values).
        if t < 1:
            continue
        if not (close[t - 1] > sl_min):
            continue

        # H_ref window: swing-highs with t - H_REF_LOOKBACK <= k <= right_edge.
        h_lo = t - H_REF_LOOKBACK
        sh_in_window = sh_positions[(sh_positions >= h_lo) & (sh_positions <= right_edge)]
        if sh_in_window.size == 0:
            continue
        # Most recent identifiable swing-high.
        h_ref_pos = int(sh_in_window.max())
        h_ref_val = float(high[h_ref_pos])
        h_ref_arr[t] = h_ref_val
        h_ref_offset[t] = float(t - h_ref_pos)
        break_mag[t] = (close[t] - h_ref_val) / a

        # Trigger 3a: close[t] > H_ref + 0.10 * ATR.
        if not (close[t] > h_ref_val + BREAK_BUFFER_ATR * a):
            continue
        # Trigger 3b: close[t] > open[t].
        if not (close[t] > open_[t]):
            continue

        # Pre-filter (informational for co-fire/diagnostics): all conditions
        # except close-upper-half + refractory have passed.
        prefilter[t] = True

        # Trigger 3c: close in upper half.
        if not (np.isfinite(cp) and cp >= CLOSE_UPPER_HALF_MIN):
            continue

        # Refractory: >= 20 bars since last full-signal fire on this pair.
        if (t - last_signal_t) < REFRACTORY_BARS:
            continue

        signal[t] = True
        last_signal_t = t

    out = df_4h.reset_index(drop=True).copy()
    out[signal_col] = signal
    out["prefilter_pass"] = prefilter
    out["h_ref"] = h_ref_arr
    out["h_ref_bar_offset"] = h_ref_offset
    out["break_magnitude_atr"] = break_mag
    out["close_position"] = close_pos
    out["trend_filter_swing_low"] = tf_low
    out["atr14"] = atr
    return out
