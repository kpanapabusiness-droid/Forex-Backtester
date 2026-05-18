"""Arc 9 — signal_inside_bar_break_trend_long_v0.1.

Locked trigger per docs/signal_spec_inside_bar_break_trend_long_v0.1.md.

Direction: long only.
TF: 4H.
Family: Trend continuation (compression-and-break).

Swing-low definition (3-bar local low):
    swing_low at bar k iff
        low[k] < min(low[k-3..k-1]) AND
        low[k] < min(low[k+1..k+3])

Trigger at bar t (signal computed at bar t close, entry at bar t+1 open):
  1. Trend filter (no MA, pure structural):
       - Identify all swing-lows in window t-30..t-1.
       - Right-edge constraint: most recent swing-low at most bar t-4
         (i.e. k+3 <= t-1 for the validating window).
       - Require >= 1 swing-low in window.
       - Require close[t-1] > min(swing_lows in window).
  2. Inside bar at t-1 (strict nest in mother bar t-2):
       - high[t-1] < high[t-2]
       - low[t-1]  > low[t-2]
  3. Break trigger at bar t:
       - close[t] > high[t-1]
       - close[t] > open[t]
  4. Spacing: >= REFRACTORY_BARS since last fired signal on this pair.

Lookahead: swing-low identification uses k+1..k+3, but the right-edge
constraint k <= t-4 (=> k+3 <= t-1) guarantees ALL bars consumed by the
swing-low validation are STRICTLY before bar t. No future-bar leak.

Computes per-bar:
    signal             0/1, gated by trend + inside-bar + break + spacing
    prefilter_pass     0/1, conds 1-3 met (pre-spacing)
    swing_low_used     reference swing-low level (= min of qualifying swing-lows)
    n_swing_lows       count of qualifying swing-lows in window
    most_recent_sl_idx index of the most recent qualifying swing-low (or -1)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Locked parameters.
TREND_WINDOW: int = 30           # search window t-30..t-1 inclusive
SWING_HALF: int = 3              # k-3..k-1 backward, k+1..k+3 forward
SWING_RIGHT_EDGE_LAG: int = 4    # most recent swing-low at most bar t-4
REFRACTORY_BARS: int = 20        # spacing between fires (signal bars)


def _find_swing_lows(low: np.ndarray, half: int = SWING_HALF) -> np.ndarray:
    """Return boolean array — True iff bar k is a 3-bar swing-low.

    Definition: low[k] strictly less than min of low[k-3..k-1] AND
    strictly less than min of low[k+1..k+3]. The first `half` and last
    `half` bars cannot be swing-lows (insufficient lookback / lookahead).
    """
    n = low.shape[0]
    is_sw = np.zeros(n, dtype=bool)
    if n < 2 * half + 1:
        return is_sw
    for k in range(half, n - half):
        left_min = np.min(low[k - half:k])
        right_min = np.min(low[k + 1:k + half + 1])
        if low[k] < left_min and low[k] < right_min:
            is_sw[k] = True
    return is_sw


def compute_signal(
    df: pd.DataFrame,
    signal_col: str = "signal",
) -> pd.DataFrame:
    """Compute the IB-trend long signal on a 4H OHLC DataFrame.

    Returns a copy of `df` with columns added:
        signal              0/1 final (post spacing)
        prefilter_pass      0/1 — trend + inside-bar + break conds met
                              (spacing not yet applied; used by Step 1
                              diagnostics)
        swing_low_used      min swing-low in t-30..t-4 window (NaN if none)
        n_swing_lows        count of qualifying swing-lows in window
        most_recent_sl_idx  index of the most recent qualifying swing-low,
                              -1 if none
        ib_passed           0/1 — inside-bar condition met at t-1 vs t-2
        break_passed        0/1 — break condition met at bar t

    Note: signal column is computed pair-by-pair in the caller. This
    function assumes df is a single pair, sorted ascending by date.
    """
    out = df.reset_index(drop=True).copy()
    n = len(out)

    high = out["high"].astype(float).to_numpy()
    low = out["low"].astype(float).to_numpy()
    close = out["close"].astype(float).to_numpy()
    op = out["open"].astype(float).to_numpy()

    is_sw_array = _find_swing_lows(low, half=SWING_HALF)

    sig = np.zeros(n, dtype=np.int8)
    prefilter = np.zeros(n, dtype=np.int8)
    swing_low_used = np.full(n, np.nan, dtype=float)
    n_sl = np.zeros(n, dtype=np.int32)
    most_recent_sl = np.full(n, -1, dtype=np.int64)
    ib_passed = np.zeros(n, dtype=np.int8)
    break_passed = np.zeros(n, dtype=np.int8)

    last_signal_idx: int = -10_000_000  # sentinel "no prior signal"

    # Earliest valid t requires:
    #   t-2 >= 0  (mother bar exists)
    #   t-1 >= 0  (inside bar exists)
    #   t-30 >= 0 (trend window full)
    #   right-edge swing-low requires bar k where k <= t-4 and k-3 >= 0
    #     => smallest k = SWING_HALF = 3, requires t >= 7
    # Combined: t >= TREND_WINDOW (=30) is the binding constraint.
    earliest_t = TREND_WINDOW
    for t in range(earliest_t, n):
        # --- Cond 2: inside bar at t-1 (strict nest in mother bar t-2) ---
        h_im1 = high[t - 1]
        l_im1 = low[t - 1]
        h_im2 = high[t - 2]
        l_im2 = low[t - 2]
        ib_ok = (h_im1 < h_im2) and (l_im1 > l_im2)
        ib_passed[t] = int(ib_ok)

        # --- Cond 3: break trigger at bar t ---
        br_ok = (close[t] > h_im1) and (close[t] > op[t])
        break_passed[t] = int(br_ok)

        # --- Cond 1: trend filter via swing-lows in window t-30..t-1 ---
        # Only consider swing-lows at bars k with k <= t - SWING_RIGHT_EDGE_LAG (=4).
        # The validation k+1..k+3 then stays <= t-1 strictly past.
        window_lo = max(0, t - TREND_WINDOW)
        window_hi_excl = t - SWING_RIGHT_EDGE_LAG + 1  # exclusive
        # Guard: window_hi_excl might fall below window_lo for very small t.
        if window_hi_excl <= window_lo:
            qualifying_sl: np.ndarray = np.empty(0, dtype=np.int64)
        else:
            seg = is_sw_array[window_lo:window_hi_excl]
            rel_idx = np.where(seg)[0]
            qualifying_sl = rel_idx + window_lo
        n_qual = int(qualifying_sl.size)
        n_sl[t] = n_qual
        if n_qual == 0:
            most_recent_sl[t] = -1
            continue
        most_recent_sl[t] = int(qualifying_sl[-1])  # highest index
        sl_min = float(low[qualifying_sl].min())
        swing_low_used[t] = sl_min

        # Hold-above check: close[t-1] above min swing-low chain.
        trend_ok = close[t - 1] > sl_min

        prefilter_ok = bool(ib_ok and br_ok and trend_ok)
        prefilter[t] = int(prefilter_ok)

        if not prefilter_ok:
            continue

        # --- Cond 4: spacing (>= REFRACTORY_BARS bars since last fire) ---
        if (t - last_signal_idx) < REFRACTORY_BARS:
            continue

        sig[t] = 1
        last_signal_idx = t

    out[signal_col] = sig.astype(int)
    out["prefilter_pass"] = prefilter.astype(int)
    out["swing_low_used"] = swing_low_used
    out["n_swing_lows"] = n_sl.astype(int)
    out["most_recent_sl_idx"] = most_recent_sl
    out["ib_passed"] = ib_passed.astype(int)
    out["break_passed"] = break_passed.astype(int)
    return out
