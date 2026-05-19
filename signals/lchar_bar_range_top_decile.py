"""LCHAR bar-range top-decile (neg) signal — registry Entry 4.

Trial ID: TRIAL__univariate_extreme__bar_range_top_decile__neg__h_001

Definition (mirrors scripts/lchar/run_layer4.py — canonical L4 source):

  At each 1H bar N close, the signal fires if:
    1. bar_range_N = high_N - low_N exceeds the p90 quantile of the prior
       100 1H bars' bar_range (bars N-100 .. N-1, strictly excluding N).
    2. close_N < open_N  (the `neg` sub-spec).

  Trailing top decile is computed via
    threshold = series.shift(1).rolling(100, min_periods=100).quantile(0.9)
  and the mask is the STRICT inequality series > threshold (mirrors L4
  trailing_top_decile — NaN cells yield False under `>`). The bar's own
  value is excluded from its own threshold (no lookahead).

  Direction sub-spec at L4 is `neg` (bar_sign == -1, i.e. close < open).
  At the arc level (L_ARC_PROTOCOL v2.1.1 §1.16 long-only baseline) every
  firing is a long signal regardless of the `neg` sign filter on the
  signal bar; the sub-spec is a SELECTION criterion on the signal bar,
  not the trade direction.

No lookahead. Signal observed at bar N close → entry at bar N+1 open.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ATR_PERIOD: int = 14  # informational; SL distance uses Wilder ATR computed by the engine
TRAILING_WINDOW: int = 100
TOP_DECILE_QUANTILE: float = 0.90


def _trailing_top_decile_strict(series: pd.Series, window: int, q: float) -> np.ndarray:
    """Mirrors run_layer4.trailing_top_decile (strict `>`).

    Returns bool[len(series)]: True where series[t] > q-quantile of
    series[t-window:t] (excluding t). False where the trailing window is
    incomplete or threshold is NaN.
    """
    threshold = series.shift(1).rolling(window, min_periods=window).quantile(q)
    raw = series.to_numpy() > threshold.to_numpy()
    return raw.astype(bool)


def compute_signal(
    df_1h: pd.DataFrame,
    *,
    trailing_window: int = TRAILING_WINDOW,
    top_decile_quantile: float = TOP_DECILE_QUANTILE,
    signal_col: str = "signal",
) -> pd.DataFrame:
    """Compute the bar_range_top_decile (neg) signal per Entry 4 of the registry.

    Parameters
    ----------
    df_1h : DataFrame
        1H bars with columns ['date', 'open', 'high', 'low', 'close', ...].
        `date` parsed as pd.Timestamp. Sorted ascending; no duplicates.
    trailing_window, top_decile_quantile
        Canonical L4 parameters (do not override without an explicit
        cross-arc calibration phase).
    signal_col
        Name of the output column (long signal, values in {0, 1}).

    Returns
    -------
    pd.DataFrame
        Copy of df_1h with a new int column `signal_col` ∈ {0, 1}. `1` =
        long signal fires on this bar's close (entry at next bar's open).
    """
    df = df_1h.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    high = df["high"].astype(float).to_numpy()
    low = df["low"].astype(float).to_numpy()
    open_ = df["open"].astype(float).to_numpy()
    close = df["close"].astype(float).to_numpy()

    bar_range = pd.Series(high - low)
    top_decile_mask = _trailing_top_decile_strict(bar_range, trailing_window, top_decile_quantile)

    # `neg` sub-spec: close < open on bar N (bar_sign == -1 in L4 terms).
    bar_sign_neg = close < open_

    sig = top_decile_mask & bar_sign_neg

    df[signal_col] = sig.astype(int)

    assert df[signal_col].isna().sum() == 0
    assert set(df[signal_col].unique()).issubset({0, 1})
    return df
