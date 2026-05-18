"""Arc 8 pullback-and-resume in HH/HL uptrend (PR-HHHL), long.

Signal definition (verbatim from docs/signal_spec_pullback_resume_hhhl_long_v0.1.md):

  Swing definitions (3-bar local extreme):
    swing-high at bar k iff high[k] > max(high[k-3..k-1]) AND
                            high[k] > max(high[k+1..k+3])
    swing-low at bar k iff low[k] < min(low[k-3..k-1]) AND
                           low[k] < min(low[k+1..k+3])

  Long signal fires at close of bar t iff ALL of:
    1. Trend established in window t-30..t-1:
       a. ≥ 2 swing-highs in window, strictly ascending (HH)
       b. ≥ 2 swing-lows in window, strictly ascending (HL)
       c. Right-edge: most recent identifiable swing-high at most bar t-4;
          same for swing-lows. (Equivalent: a swing at bar k requires
          k+3 ≤ t-1, so k ≤ t-4. The window is t-30..t-1 but only swings
          with k ≤ t-4 are identifiable from bars strictly < t.)
    2. Pullback: close[t-1] ≤ most_recent_swing_high − 0.5 × ATR(14)[t-1]
    3. Resume trigger at bar t:
       a. close[t] > open[t]                            (bullish close)
       b. close[t] > high[t-1]                          (breaks prior high)
       c. (close[t] − low[t]) / (high[t] − low[t]) ≥ 0.5 (close in upper half)
    4. Spacing: ≥ 20 bars since last signal on this pair

  ATR(14) is Wilder-smoothed on 4H bars, causal (uses TR values 1..t only).

Side-output features (computed for every signal-firing bar; NaN otherwise):

  num_higher_highs           int — count of swing-highs in window (≥ 2)
  num_higher_lows            int — count of swing-lows in window (≥ 2)
  most_recent_sh_price       float — price of most recent identifiable SH
  most_recent_sh_age         int — bars between most-recent SH and bar t
  most_recent_sl_price       float — price of most recent identifiable SL
  most_recent_sl_age         int — bars between most-recent SL and bar t
  hh_range_atr               float — (highest SH − lowest SH) / ATR(14)[t]
  hl_range_atr               float — (highest SL − lowest SL) / ATR(14)[t]
  pullback_depth_atr         float — (sh − close[t-1]) / ATR(14)[t-1]
  trigger_body_atr           float — (close[t] − open[t]) / ATR(14)[t]
  trigger_close_pos          float — (close[t] − low[t]) / (high[t] − low[t])
  trigger_break_size_atr     float — (close[t] − high[t-1]) / ATR(14)[t]

These features become part of trades_all.csv at Step 1 (entry-time features
for Step 4 Pipeline E predictability). All are computed strictly from bars
≤ t, no future-bar dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

# Locked parameters (mirror in configs/wfo_l_arc_8.yaml).
SWING_LOOKBACK_BARS: int = 3
TREND_WINDOW_BARS: int = 30
RIGHT_EDGE_GAP: int = 4
MIN_SWING_HIGHS: int = 2
MIN_SWING_LOWS: int = 2
PULLBACK_DEPTH_ATR_MIN: float = 0.5
TRIGGER_CLOSE_POS_MIN: float = 0.5
ATR_PERIOD: int = 14
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
    series: np.ndarray, kind: str, lookback: int = SWING_LOOKBACK_BARS
) -> np.ndarray:
    """Detect 3-bar swing extremes.

    Returns boolean array; entry True iff bar k is a swing extreme of the given
    kind (high or low) under the locked 3-bar definition:

      swing-high at k iff series[k] > max(series[k-lookback..k-1])
                          AND series[k] > max(series[k+1..k+lookback])
      swing-low  at k iff series[k] < min(series[k-lookback..k-1])
                          AND series[k] < min(series[k+1..k+lookback])

    Bars with insufficient surrounding bars (k < lookback or k > n-1-lookback)
    are False — never identifiable as a swing.

    Strict inequality used both sides (avoids degenerate plateaus from
    counting as swings). This is the documented locked semantics.
    """
    n = len(series)
    swing = np.zeros(n, dtype=bool)
    if n < 2 * lookback + 1:
        return swing
    if kind == "high":
        cmp_left = np.greater
        cmp_right = np.greater
        reduce_left = np.max
        reduce_right = np.max
    elif kind == "low":
        cmp_left = np.less
        cmp_right = np.less
        reduce_left = np.min
        reduce_right = np.min
    else:
        raise ValueError(f"kind must be 'high' or 'low', got {kind!r}")
    for k in range(lookback, n - lookback):
        v = series[k]
        left = reduce_left(series[k - lookback : k])
        right = reduce_right(series[k + 1 : k + 1 + lookback])
        if cmp_left(v, left) and cmp_right(v, right):
            swing[k] = True
    return swing


def _is_strictly_ascending(values: np.ndarray) -> bool:
    """True iff every value is strictly greater than the previous one."""
    if values.size < 2:
        return False
    return bool(np.all(np.diff(values) > 0))


@dataclass
class _SignalFeatures:
    num_higher_highs: int
    num_higher_lows: int
    most_recent_sh_price: float
    most_recent_sh_age: int
    most_recent_sl_price: float
    most_recent_sl_age: int
    hh_range_atr: float
    hl_range_atr: float
    pullback_depth_atr: float
    trigger_body_atr: float
    trigger_close_pos: float
    trigger_break_size_atr: float


def compute_signal(
    df_4h: pd.DataFrame,
    *,
    trend_window: int = TREND_WINDOW_BARS,
    right_edge_gap: int = RIGHT_EDGE_GAP,
    swing_lookback: int = SWING_LOOKBACK_BARS,
    min_highs: int = MIN_SWING_HIGHS,
    min_lows: int = MIN_SWING_LOWS,
    pullback_min_atr: float = PULLBACK_DEPTH_ATR_MIN,
    trigger_close_pos_min: float = TRIGGER_CLOSE_POS_MIN,
    atr_period: int = ATR_PERIOD,
    refractory: int = REFRACTORY_BARS,
    signal_col: str = "signal",
) -> pd.DataFrame:
    """Compute PR-HHHL signal on a chronologically-sorted 4H DataFrame.

    Returns a copy of ``df_4h`` with added columns:
      - signal                bool — all conditions met
      - atr14                 float — causal Wilder ATR(14)
      - is_swing_high         bool — 3-bar swing high at bar k
      - is_swing_low          bool — 3-bar swing low at bar k
      - num_higher_highs      int
      - num_higher_lows       int
      - most_recent_sh_price  float
      - most_recent_sh_age    int (bars between SH and signal bar t)
      - most_recent_sl_price  float
      - most_recent_sl_age    int
      - hh_range_atr          float
      - hl_range_atr          float
      - pullback_depth_atr    float
      - trigger_body_atr      float
      - trigger_close_pos     float
      - trigger_break_size_atr float
    """
    if df_4h.empty:
        out = df_4h.copy()
        for col, dtype in [
            (signal_col, bool),
            ("atr14", float),
            ("is_swing_high", bool),
            ("is_swing_low", bool),
            ("num_higher_highs", float),
            ("num_higher_lows", float),
            ("most_recent_sh_price", float),
            ("most_recent_sh_age", float),
            ("most_recent_sl_price", float),
            ("most_recent_sl_age", float),
            ("hh_range_atr", float),
            ("hl_range_atr", float),
            ("pullback_depth_atr", float),
            ("trigger_body_atr", float),
            ("trigger_close_pos", float),
            ("trigger_break_size_atr", float),
        ]:
            out[col] = pd.Series(dtype=dtype)
        return out

    df = df_4h.reset_index(drop=True).copy()
    n = len(df)
    open_ = df["open"].astype(float).to_numpy()
    high = df["high"].astype(float).to_numpy()
    low = df["low"].astype(float).to_numpy()
    close = df["close"].astype(float).to_numpy()

    atr = _wilder_atr(df, atr_period)
    is_sh = _detect_swings(high, kind="high", lookback=swing_lookback)
    is_sl = _detect_swings(low, kind="low", lookback=swing_lookback)

    signal = np.zeros(n, dtype=bool)
    feat: List[_SignalFeatures] = [
        _SignalFeatures(
            num_higher_highs=0,
            num_higher_lows=0,
            most_recent_sh_price=np.nan,
            most_recent_sh_age=-1,
            most_recent_sl_price=np.nan,
            most_recent_sl_age=-1,
            hh_range_atr=np.nan,
            hl_range_atr=np.nan,
            pullback_depth_atr=np.nan,
            trigger_body_atr=np.nan,
            trigger_close_pos=np.nan,
            trigger_break_size_atr=np.nan,
        )
        for _ in range(n)
    ]

    last_signal_t: int = -(10**9)

    # Right-edge: latest identifiable swing at bar t has k ≤ t - right_edge_gap.
    # Trend window: bars [t - trend_window, t - 1].
    # Effective swing search window: [t - trend_window, t - right_edge_gap].
    for t in range(n):
        if t < trend_window:
            continue
        atr_t = atr[t]
        atr_tm1 = atr[t - 1]
        if (
            not np.isfinite(atr_t)
            or not np.isfinite(atr_tm1)
            or atr_t <= 0
            or atr_tm1 <= 0
        ):
            continue

        win_start = t - trend_window
        win_end = t - right_edge_gap  # inclusive upper

        # Collect swing positions / prices in window (chronological order).
        sh_positions = [k for k in range(win_start, win_end + 1) if is_sh[k]]
        sl_positions = [k for k in range(win_start, win_end + 1) if is_sl[k]]

        if len(sh_positions) < min_highs or len(sl_positions) < min_lows:
            continue

        sh_prices = np.array([high[k] for k in sh_positions], dtype=float)
        sl_prices = np.array([low[k] for k in sl_positions], dtype=float)

        # HH / HL structure: strictly ascending sequences.
        if not _is_strictly_ascending(sh_prices):
            continue
        if not _is_strictly_ascending(sl_prices):
            continue

        # Most recent swings (last in chronological order).
        recent_sh_pos = sh_positions[-1]
        recent_sl_pos = sl_positions[-1]
        recent_sh_price = float(sh_prices[-1])
        recent_sl_price = float(sl_prices[-1])

        # Pullback at bar t-1.
        if not (close[t - 1] <= recent_sh_price - pullback_min_atr * atr_tm1):
            continue

        # Resume trigger at bar t.
        if not (close[t] > open_[t]):
            continue
        if not (close[t] > high[t - 1]):
            continue
        bar_range = high[t] - low[t]
        if bar_range <= 0:
            continue
        close_pos = (close[t] - low[t]) / bar_range
        if not (close_pos >= trigger_close_pos_min):
            continue

        # Spacing (refractory).
        if (t - last_signal_t) < refractory:
            continue

        # All conditions pass — record signal + features.
        signal[t] = True
        last_signal_t = t

        feat[t] = _SignalFeatures(
            num_higher_highs=len(sh_positions),
            num_higher_lows=len(sl_positions),
            most_recent_sh_price=recent_sh_price,
            most_recent_sh_age=t - recent_sh_pos,
            most_recent_sl_price=recent_sl_price,
            most_recent_sl_age=t - recent_sl_pos,
            hh_range_atr=float((sh_prices.max() - sh_prices.min()) / atr_t),
            hl_range_atr=float((sl_prices.max() - sl_prices.min()) / atr_t),
            pullback_depth_atr=float((recent_sh_price - close[t - 1]) / atr_tm1),
            trigger_body_atr=float((close[t] - open_[t]) / atr_t),
            trigger_close_pos=float(close_pos),
            trigger_break_size_atr=float((close[t] - high[t - 1]) / atr_t),
        )

    out = df_4h.reset_index(drop=True).copy()
    out[signal_col] = signal
    out["atr14"] = atr
    out["is_swing_high"] = is_sh
    out["is_swing_low"] = is_sl
    out["num_higher_highs"] = np.array(
        [f.num_higher_highs if signal[i] else np.nan for i, f in enumerate(feat)],
        dtype=float,
    )
    out["num_higher_lows"] = np.array(
        [f.num_higher_lows if signal[i] else np.nan for i, f in enumerate(feat)],
        dtype=float,
    )
    out["most_recent_sh_price"] = np.array(
        [f.most_recent_sh_price for f in feat], dtype=float
    )
    out["most_recent_sh_age"] = np.array(
        [f.most_recent_sh_age if signal[i] else np.nan for i, f in enumerate(feat)],
        dtype=float,
    )
    out["most_recent_sl_price"] = np.array(
        [f.most_recent_sl_price for f in feat], dtype=float
    )
    out["most_recent_sl_age"] = np.array(
        [f.most_recent_sl_age if signal[i] else np.nan for i, f in enumerate(feat)],
        dtype=float,
    )
    out["hh_range_atr"] = np.array([f.hh_range_atr for f in feat], dtype=float)
    out["hl_range_atr"] = np.array([f.hl_range_atr for f in feat], dtype=float)
    out["pullback_depth_atr"] = np.array(
        [f.pullback_depth_atr for f in feat], dtype=float
    )
    out["trigger_body_atr"] = np.array(
        [f.trigger_body_atr for f in feat], dtype=float
    )
    out["trigger_close_pos"] = np.array(
        [f.trigger_close_pos for f in feat], dtype=float
    )
    out["trigger_break_size_atr"] = np.array(
        [f.trigger_break_size_atr for f in feat], dtype=float
    )
    return out
