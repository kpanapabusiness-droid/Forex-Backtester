"""LCHAR D1 ATR top-decile signal — registry Entry 3.

Trial ID: TRIAL__volatility_regime__d1_atr_top_decile__any__h_120

Definition (mirrors scripts/lchar/run_layer4.py — canonical L4 source):

  At each 1H bar N close, the signal fires if the D1 ATR(14) of the
  most-recently-completed D1 bar (one-day lag) lies in the top decile of the
  trailing 100 D1 bars (excluding that D1 bar itself) for the pair.

  L2 most-recently-completed lookback convention: each 1H bar at calendar date
  D maps to the D1 row at index (D1-index-of-date-D) - 1 — i.e. the D1 bar
  BEFORE the one whose date contains the 1H bar. Identical to KH-24's one-day
  D1 alignment rule. NaN where the lookup is out of range (very early bars).

  ATR(14) is the SMA of the True Range over 14 D1 bars (mirrors L4's
  `compute_atr` — `.rolling(14).mean()`, NOT Wilder smoothing). The L4
  implementation is the canonical source of truth for the signal definition;
  the SL distance calculation uses a separate Wilder-smoothed 1H ATR(14) per
  the Arc 3 config.

  Trailing top decile: `series.shift(1).rolling(100, min_periods=100).quantile(0.9)`;
  the bar's own value is excluded from its own threshold (no lookahead).

  Direction sub-spec at L4 is `any`. At the arc level (Arc 3, long-only
  baseline per L_ARC_PROTOCOL v2.0 §1.16) every firing is a long signal.

No lookahead. Signal observed at bar N close → entry at bar N+1 open.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Canonical L4 parameters (locked, identical to configs/lchar/layer4.yaml).
ATR_PERIOD: int = 14
TRAILING_WINDOW: int = 100
TOP_DECILE_QUANTILE: float = 0.90


def _compute_d1_atr_sma(df_d1: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    """ATR(period) on D1 bars — SMA of True Range (mirrors run_layer4.compute_atr).

    Not Wilder. Matches the L4 atlas canonical source.
    """
    h = df_d1["high"].astype(float).to_numpy()
    lo = df_d1["low"].astype(float).to_numpy()
    c = df_d1["close"].astype(float).to_numpy()
    pc = np.concatenate(([np.nan], c[:-1]))
    tr = np.nanmax(np.column_stack([h - lo, np.abs(h - pc), np.abs(lo - pc)]), axis=1)
    return pd.Series(tr, index=df_d1.index).rolling(period, min_periods=period).mean()


def _trailing_top_decile(series: pd.Series, window: int, q: float) -> np.ndarray:
    """Mirrors run_layer4.trailing_top_decile.

    Returns bool[len(series)]: True where series[t] > q-quantile of
    series[t-window:t] (excluding t). False where the trailing window is
    incomplete or threshold is NaN.
    """
    threshold = series.shift(1).rolling(window, min_periods=window).quantile(q)
    raw = series.to_numpy() > threshold.to_numpy()
    # `>` with NaN yields False — explicit cast keeps boolean dtype.
    return raw.astype(bool)


def _lookback_d1_to_1h(df_1h: pd.DataFrame, df_d1: pd.DataFrame, d1_mask: np.ndarray) -> np.ndarray:
    """Mirrors run_layer4.lookback_d1_to_lower.

    For each 1H bar, return the D1 mask value at the D1 strictly before the D1
    bar containing it (one-day lag). False where the lookup is out of range.
    """
    floor_d1 = df_1h["date"].dt.normalize()
    idx_d1 = pd.Series(np.arange(len(df_d1), dtype=np.int64), index=df_d1["date"])
    contain = floor_d1.map(idx_d1).to_numpy(dtype=float)
    valid = ~np.isnan(contain)
    contain_int = np.where(valid, contain, 0).astype(np.int64)
    mr_idx = contain_int - 1
    in_range = valid & (mr_idx >= 0)
    out = np.zeros(len(df_1h), dtype=bool)
    out[in_range] = d1_mask[mr_idx[in_range]]
    return out


def compute_signal(
    df_1h: pd.DataFrame,
    df_d1: pd.DataFrame,
    *,
    atr_period: int = ATR_PERIOD,
    trailing_window: int = TRAILING_WINDOW,
    top_decile_quantile: float = TOP_DECILE_QUANTILE,
    signal_col: str = "signal",
) -> pd.DataFrame:
    """Compute the d1_atr_top_decile signal per Entry 3 of LCHAR_TOPN_REGISTRY.

    Parameters
    ----------
    df_1h : DataFrame
        1H bars with columns ['date', 'open', 'high', 'low', 'close', ...].
        `date` parsed as pd.Timestamp. Sorted ascending.
    df_d1 : DataFrame
        D1 bars for the same pair with the same column convention. Sorted
        ascending.
    atr_period, trailing_window, top_decile_quantile
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
    df_1h = df_1h.copy()
    df_d1 = df_d1.copy()

    # Ensure date columns are datetime and sorted.
    df_1h["date"] = pd.to_datetime(df_1h["date"], errors="coerce")
    df_d1["date"] = pd.to_datetime(df_d1["date"], errors="coerce")
    df_1h = df_1h.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    df_d1 = df_d1.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # 1. D1 ATR(14) SMA + trailing top-decile mask (per pair, no lookahead).
    d1_atr = _compute_d1_atr_sma(df_d1, atr_period)
    d1_top_mask = _trailing_top_decile(d1_atr, trailing_window, top_decile_quantile)

    # 2. Map D1 mask onto 1H index via one-day-lag lookback.
    sig_1h = _lookback_d1_to_1h(df_1h, df_d1, d1_top_mask)

    df_1h[signal_col] = sig_1h.astype(int)

    # Invariants (sanity).
    assert df_1h[signal_col].isna().sum() == 0
    assert set(df_1h[signal_col].unique()).issubset({0, 1})
    return df_1h
