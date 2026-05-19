"""Arc 10 D1 swing-low rejection long signal (DLR).

Signal definition (locked, verbatim per signal_spec_d1_swing_low_rejection_long_v0.1.md):

D1 anchor identification (one-day lag enforced — KH-24 convention):

  1. D1 swing-low at day d:  low[d] < min(low[d-3..d-1])  AND
                              low[d] < min(low[d+1..d+3])
  2. Most recent identifiable D1 swing-low at 4H bar t = L_1.
     Search window: D1 bars closing strictly before 4H bar t's open.
     Right-edge constraint: most recent identifiable L_1 at most
     D1[d_t − 4] where d_t is the D1 bar containing 4H bar t.
     (Required because confirming a swing-low at d needs d+3 to be known,
     so the latest confirmable d is d_t − 4.)
  3. Prior D1 swing-low = L_0 (next-most-recent before L_1)
  4. D1 HL structure: both L_1 and L_0 exist within last 30 D1 bars at
     4H bar t, AND L_1 > L_0 (strictly ascending)
  5. L_1 freshness: D1 bar containing L_1 not older than 20 D1 bars at
     4H bar t

4H test/reject (bar t = signal bar):

  6. Test (proximity):  low[t] <= L_1 + 0.25 * ATR(14)_4H[t]
  7. Reject (close back above):  close[t] > L_1 + 0.10 * ATR(14)_4H[t]
  8. Trigger-bar geometry:
       close[t] > open[t]                                       (bullish)
       (close[t] - low[t]) / (high[t] - low[t]) >= 0.6          (upper 40%)

Spacing & entry:

  9. >= 20 4H bars since last full signal on this pair (refractory)
  10. Entry: bar t+1 open (handled by step1 backtester, not this module)

D1 lag implementation (matches KH-24 convention in
scripts/phase_kgl_v2_4h_wfo.py): D1 bars are filtered to those closing
strictly before each 4H bar via a backward asof-style join on
pre-shifted 4H dates. Same-day D1 close is NOT available at 4H bar t.

NaN-perturbation invariance: NaN-ing D1 row at d_t (the D1 day
containing 4H bar t) MUST leave Arc 10 signal output unchanged for
any bar t. The signal references only D1 bars d <= d_t - 4 for
swing-low identification (and the D1 dates strictly < bar-t-open for
the d_t lookup itself); D1[d_t] is structurally unread.

The module accepts the D1 dataframe directly to keep alignment
explicit (no global state, fully testable). The step1 backtester
joins the 4H bars to per-bar D1 metadata before invoking compute_signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Locked parameters (mirror in configs/wfo_l_arc_10.yaml).
D1_SWING_WINDOW_K: int = 3                # k bars on each side for D1 swing-low
D1_RIGHT_EDGE_OFFSET: int = 4             # L_1 must be at most D1[d_t - 4]
D1_STRUCTURE_LOOKBACK_BARS: int = 30      # L_1 and L_0 within last 30 D1 bars
D1_L1_FRESHNESS_MAX_BARS: int = 20        # L_1 not older than 20 D1 bars
ATR_PERIOD_4H: int = 14
PROXIMITY_ATR_MULT: float = 0.25          # cond 6: low[t] <= L_1 + 0.25*ATR
REJECT_BUFFER_ATR_MULT: float = 0.10      # cond 7: close[t] > L_1 + 0.10*ATR
UPPER_FRACTION_MIN: float = 0.6           # cond 8b
REFRACTORY_BARS_4H: int = 20              # cond 9


def wilder_atr(df: pd.DataFrame, period: int = ATR_PERIOD_4H) -> np.ndarray:
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


def compute_d1_swing_low_flags(
    df_d1: pd.DataFrame, k: int = D1_SWING_WINDOW_K
) -> np.ndarray:
    """Mark each D1 bar d as a swing-low iff
    low[d] < min(low[d-k..d-1]) AND low[d] < min(low[d+1..d+k]).

    Bars within k of either edge are False (cannot be confirmed).
    NaN low at any position in the 2k+1 window also yields False.
    """
    low = df_d1["low"].astype(float).to_numpy()
    n = len(low)
    flags = np.zeros(n, dtype=bool)
    if n < 2 * k + 1:
        return flags
    for d in range(k, n - k):
        center = low[d]
        if not np.isfinite(center):
            continue
        past = low[d - k : d]
        fut = low[d + 1 : d + k + 1]
        if np.any(~np.isfinite(past)) or np.any(~np.isfinite(fut)):
            continue
        if center < past.min() and center < fut.min():
            flags[d] = True
    return flags


def _date_to_d1_index(
    bar_dates_4h: np.ndarray, d1_dates: np.ndarray
) -> np.ndarray:
    """For each 4H bar date, return the index of the D1 bar that contains it
    (d_t). Equivalent to merge_asof(direction='backward') on the date floor.

    Returns -1 for 4H bars whose date predates all D1 bars.
    """
    d1_ts = pd.to_datetime(d1_dates)
    d1_norm = d1_ts.normalize().to_numpy()
    bar_norm = pd.to_datetime(bar_dates_4h).normalize().to_numpy()
    # searchsorted right-1: largest d1_idx with d1_norm[d1_idx] <= bar_norm[i]
    idx = np.searchsorted(d1_norm, bar_norm, side="right") - 1
    # Bound: -1 stays as -1 (out of left edge)
    return idx.astype(int)


def compute_signal(
    df_4h: pd.DataFrame,
    df_d1: pd.DataFrame,
    *,
    k_swing: int = D1_SWING_WINDOW_K,
    right_edge_offset: int = D1_RIGHT_EDGE_OFFSET,
    structure_lookback: int = D1_STRUCTURE_LOOKBACK_BARS,
    l1_freshness_max: int = D1_L1_FRESHNESS_MAX_BARS,
    atr_period: int = ATR_PERIOD_4H,
    proximity_mult: float = PROXIMITY_ATR_MULT,
    reject_buffer_mult: float = REJECT_BUFFER_ATR_MULT,
    upper_fraction_min: float = UPPER_FRACTION_MIN,
    refractory_bars: int = REFRACTORY_BARS_4H,
    signal_col: str = "signal",
) -> pd.DataFrame:
    """Compute the Arc 10 DLR long signal on a chronological 4H DataFrame
    with a companion D1 DataFrame.

    Returns a copy of df_4h with these added columns:
      - signal              bool, True at the signal bar
      - prefilter_pass      bool, conditions 1-8 met (pre-refractory)
      - L1_value            float, D1 swing-low value at the signal bar (NaN if absent)
      - L0_value            float, prior D1 swing-low (NaN if absent)
      - L1_age_d1_bars      int,   d_t - L1_idx (NaN if absent)
      - L0_age_d1_bars      int,   d_t - L0_idx (NaN if absent)
      - L1_to_atr_proximity float, (low[t] - L1) / ATR
      - reject_buffer_atr   float, (close[t] - L1) / ATR
      - upper_fraction      float, (close - low) / (high - low)
      - d_t_idx             int,   D1 index of the D1 bar containing 4H bar t (-1 if none)
      - d_for_l1_search_max int,   d_t - right_edge_offset (-1 if d_t < offset)
      - atr14               float
    """
    if df_4h.empty:
        out = df_4h.copy()
        for col, dtype in [
            (signal_col, bool),
            ("prefilter_pass", bool),
            ("L1_value", float),
            ("L0_value", float),
            ("L1_age_d1_bars", float),
            ("L0_age_d1_bars", float),
            ("L1_to_atr_proximity", float),
            ("reject_buffer_atr", float),
            ("upper_fraction", float),
            ("d_t_idx", int),
            ("d_for_l1_search_max", int),
            ("atr14", float),
        ]:
            out[col] = pd.Series(dtype=dtype)
        return out

    df = df_4h.reset_index(drop=True).copy()
    n = len(df)
    low_4h = df["low"].astype(float).to_numpy()
    high_4h = df["high"].astype(float).to_numpy()
    close_4h = df["close"].astype(float).to_numpy()
    open_4h = df["open"].astype(float).to_numpy()
    bar_dates = df["date"].to_numpy()

    atr = wilder_atr(df, atr_period)

    df_d1_sorted = df_d1.sort_values("date").reset_index(drop=True)
    d1_low_arr = df_d1_sorted["low"].astype(float).to_numpy()
    d1_dates_arr = df_d1_sorted["date"].to_numpy()
    d1_swing_flags = compute_d1_swing_low_flags(df_d1_sorted, k=k_swing)
    d1_swing_indices = np.where(d1_swing_flags)[0]

    d_t_for_each_4h = _date_to_d1_index(bar_dates, d1_dates_arr)

    signal = np.zeros(n, dtype=bool)
    prefilter = np.zeros(n, dtype=bool)
    L1_value = np.full(n, np.nan, dtype=float)
    L0_value = np.full(n, np.nan, dtype=float)
    L1_age = np.full(n, np.nan, dtype=float)
    L0_age = np.full(n, np.nan, dtype=float)
    proximity = np.full(n, np.nan, dtype=float)
    reject_buf = np.full(n, np.nan, dtype=float)
    upper_frac = np.full(n, np.nan, dtype=float)
    d_search_max_arr = np.full(n, -1, dtype=int)

    last_signal_t: int = -(10**9)

    for t in range(n):
        a = atr[t]
        if not np.isfinite(a) or a <= 0:
            continue

        d_t = int(d_t_for_each_4h[t])
        if d_t < 0:
            continue
        d_search_max = d_t - right_edge_offset
        d_search_max_arr[t] = d_search_max
        if d_search_max < 0:
            continue

        # Most recent identifiable D1 swing-low with idx <= d_search_max.
        # Use searchsorted on the pre-computed swing indices.
        pos = np.searchsorted(d1_swing_indices, d_search_max, side="right")
        if pos < 1:
            continue
        l1_idx = int(d1_swing_indices[pos - 1])
        if pos < 2:
            continue
        l0_idx = int(d1_swing_indices[pos - 2])

        l1_val = float(d1_low_arr[l1_idx])
        l0_val = float(d1_low_arr[l0_idx])
        l1_age_bars = d_t - l1_idx
        l0_age_bars = d_t - l0_idx

        L1_value[t] = l1_val
        L0_value[t] = l0_val
        L1_age[t] = l1_age_bars
        L0_age[t] = l0_age_bars

        # Cond 4: D1 HL structure (both within structure_lookback, L_1 > L_0).
        if l1_age_bars > structure_lookback:
            continue
        if l0_age_bars > structure_lookback:
            continue
        if not (l1_val > l0_val):
            continue
        # Cond 5: L_1 freshness.
        if l1_age_bars > l1_freshness_max:
            continue

        # 4H bar t metrics.
        lo = low_4h[t]
        cl = close_4h[t]
        hi = high_4h[t]
        op = open_4h[t]
        if not (np.isfinite(lo) and np.isfinite(cl) and np.isfinite(hi) and np.isfinite(op)):
            continue
        proximity[t] = (lo - l1_val) / a
        reject_buf[t] = (cl - l1_val) / a
        rng = hi - lo
        upper_frac[t] = (cl - lo) / rng if rng > 0 else 0.5

        # Cond 6: proximity.
        if not (lo <= l1_val + proximity_mult * a):
            continue
        # Cond 7: reject buffer.
        if not (cl > l1_val + reject_buffer_mult * a):
            continue
        # Cond 8: trigger-bar geometry.
        if not (cl > op):
            continue
        if rng <= 0:
            continue
        if not ((cl - lo) / rng >= upper_fraction_min):
            continue

        # Conditions 1-8 pass.
        prefilter[t] = True

        # Cond 9: refractory.
        if (t - last_signal_t) < refractory_bars:
            continue
        signal[t] = True
        last_signal_t = t

    out = df_4h.reset_index(drop=True).copy()
    out[signal_col] = signal
    out["prefilter_pass"] = prefilter
    out["L1_value"] = L1_value
    out["L0_value"] = L0_value
    out["L1_age_d1_bars"] = L1_age
    out["L0_age_d1_bars"] = L0_age
    out["L1_to_atr_proximity"] = proximity
    out["reject_buffer_atr"] = reject_buf
    out["upper_fraction"] = upper_frac
    out["d_t_idx"] = d_t_for_each_4h
    out["d_for_l1_search_max"] = d_search_max_arr
    out["atr14"] = atr
    return out
