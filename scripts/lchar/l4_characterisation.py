"""L4 univariate-extreme descriptive characterisation pipeline (research-mode).

Produces a per-signal feature matrix and aggregated reports over the 2021-01-01
through 2025-12-31 signal window. Disposition is descriptive only — no gate,
no filter derivation, no system construction. L6.0 §9 no-filter-rescue applies.

Reuses `core.signals.l4_univariate_extreme._compute_signals` verbatim so the
signal-firing bar set is byte-identical to Arc 1's over the overlap window.
Trade-outcome execution semantics also match Arc 1 (entry N+1 open, exit N+2
open or SL hit, 2.0 × ATR(14)_1H, 1% reset-floor sizing, spread floor applied).

Public entrypoint: `run_characterisation(config_path)`.
"""

from __future__ import annotations

import csv
import hashlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.signals.l4_univariate_extreme import (  # noqa: E402
    _compute_signals,
    _wilder_atr,
    _build_quote_to_usd_table,
    _quote_to_usd_at,
    _CCY_TO_USD_HELPER,
    _USD_QUOTE_PAIRS,
    _USD_BASE_PAIRS,
    _pip_size,
    TIME_COL,
)
from core.spread_floor import (  # noqa: E402
    SpreadFloorState,
    apply_spread_floor_to_pips,
    format_startup_log,
    load_spread_floor,
    STATE_CFG_KEY,
)
from validators_config import load_and_validate_config  # noqa: E402

PAIRS_DEFAULT: Tuple[str, ...] = (
    "AUD_CAD", "AUD_CHF", "AUD_JPY", "AUD_NZD", "AUD_USD", "CAD_CHF", "CAD_JPY", "CHF_JPY",
    "EUR_AUD", "EUR_CAD", "EUR_CHF", "EUR_GBP", "EUR_JPY", "EUR_NZD", "EUR_USD", "GBP_AUD",
    "GBP_CAD", "GBP_CHF", "GBP_JPY", "GBP_NZD", "GBP_USD", "NZD_CAD", "NZD_CHF", "NZD_JPY",
    "NZD_USD", "USD_CAD", "USD_CHF", "USD_JPY",
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_pair_tf_csv(pair: str, tf_dir: str, data_load_start: pd.Timestamp) -> pd.DataFrame:
    path = REPO_ROOT / tf_dir / f"{pair}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Data file missing: {path}")
    df = pd.read_csv(path)
    if TIME_COL not in df.columns:
        raise ValueError(f"{path} missing '{TIME_COL}' column")
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    df = df[df[TIME_COL] >= data_load_start].reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Generic vectorised feature primitives
# ---------------------------------------------------------------------------


def _wilder_ema(series: pd.Series, period: int) -> pd.Series:
    """Wilder smoothing (recursive). Used for ATR calc; not for EMA(20)/EMA(50)."""
    arr = series.to_numpy(dtype=float, copy=True)
    out = np.full_like(arr, np.nan, dtype=float)
    if len(arr) < period:
        return pd.Series(out, index=series.index)
    out[period - 1] = np.mean(arr[:period])
    for i in range(period, len(arr)):
        out[i] = (out[i - 1] * (period - 1) + arr[i]) / period
    return pd.Series(out, index=series.index)


def _ema(series: pd.Series, span: int) -> pd.Series:
    """Standard EMA, adjust=False (recursive)."""
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def _kijun(df: pd.DataFrame, period: int = 26) -> pd.Series:
    """Kijun = (rolling-N high + rolling-N low) / 2 over last N bars including current."""
    return (df["high"].rolling(period, min_periods=period).max()
            + df["low"].rolling(period, min_periods=period).min()) / 2.0


def _slope_5(arr: np.ndarray, window: int = 5) -> np.ndarray:
    """OLS slope over rolling-`window` of `arr`. NaN until window full.

    Slope computed against x = 0..window-1 (unit spacing). Fully vectorised.
    """
    n = len(arr)
    out = np.full(n, np.nan, dtype=float)
    if n < window:
        return out
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    x_dev = x - x_mean
    x_dev_sq_sum = float(np.sum(x_dev * x_dev))
    # Use a strided window via stride tricks for speed.
    for i in range(window - 1, n):
        y = arr[i - window + 1: i + 1]
        if not np.all(np.isfinite(y)):
            continue
        y_mean = y.mean()
        cov = float(np.sum(x_dev * (y - y_mean)))
        out[i] = cov / x_dev_sq_sum if x_dev_sq_sum > 0 else np.nan
    return out


def _rolling_apply_acf1(arr: np.ndarray, window: int) -> np.ndarray:
    """Lag-1 autocorrelation over a trailing window. Strict-trailing (uses values up to and including bar i)."""
    n = len(arr)
    out = np.full(n, np.nan, dtype=float)
    for i in range(window - 1, n):
        y = arr[i - window + 1: i + 1]
        if not np.all(np.isfinite(y)):
            continue
        m = y.mean()
        c = y - m
        var = float(np.sum(c * c))
        if var == 0.0:
            continue
        cov = float(np.sum(c[:-1] * c[1:]))
        out[i] = cov / var
    return out


def _signed_run_length(open_arr: np.ndarray, close_arr: np.ndarray) -> np.ndarray:
    """For each bar i, return signed count of consecutive same-direction bars
    PRIOR TO i (bars i-1, i-2, ... going back).
    +K = K up bars (close > open) preceded i.  -K = K down bars preceded i.  0 = bar i-1 was flat.
    """
    n = len(open_arr)
    out = np.zeros(n, dtype=float)
    # Direction at each bar: +1 / -1 / 0
    dir_ = np.where(close_arr > open_arr, 1, np.where(close_arr < open_arr, -1, 0)).astype(int)
    for i in range(1, n):
        d = dir_[i - 1]
        if d == 0:
            out[i] = 0
            continue
        # Count back while same sign
        count = 0
        j = i - 1
        while j >= 0 and dir_[j] == d:
            count += 1
            j -= 1
        out[i] = count * d
    out[0] = 0
    return out


# ---------------------------------------------------------------------------
# 1H per-pair feature pipeline
# ---------------------------------------------------------------------------


def _compute_h1_features(
    df_h1: pd.DataFrame,
    pair: str,
    feat_cfg: Dict[str, Any],
    sig_cfg: Dict[str, Any],
    spread_state: SpreadFloorState,
    cfg: dict,
) -> pd.DataFrame:
    """Full 1H feature frame for a single pair. Returns df_h1 + many feature columns.

    All features are rolling/derived from bar close[i] using only data ≤ i (or strictly < i
    where noted). The signal-bar identifier `signal_fired` is also computed via
    `_compute_signals` for the same lookback / direction filter as Arc 1.
    """
    # Reuse Arc-1 _compute_signals so log_return, abs_log_return, threshold,
    # atr_at_signal, signal_fired all match the Arc 1 trade set bar-for-bar.
    df = _compute_signals(
        df_h1,
        pair=pair,
        lookback=int(sig_cfg["lookback_bars"]),
        threshold_q=float(sig_cfg["threshold_quantile"]),
        direction_filter=str(sig_cfg["direction_filter"]),
        atr_period=14,
    ).copy()
    n = len(df)
    open_a = df["open"].to_numpy(float)
    high_a = df["high"].to_numpy(float)
    low_a = df["low"].to_numpy(float)
    close_a = df["close"].to_numpy(float)
    log_ret = df["log_return"].to_numpy(float)
    abs_log_ret = df["abs_log_return"].to_numpy(float)
    atr_a = df["atr_at_signal"].to_numpy(float)

    # Cumulative log returns over bars [N-K, N-1] strict — pre-signal context.
    # log_return.rolling(K).sum() at bar i covers [i-K+1..i]; we shift(1) to land at [i-K..i-1].
    log_ret_s = pd.Series(log_ret)
    df["cum_logret_1h_3"] = log_ret_s.rolling(3, min_periods=3).sum().shift(1).to_numpy()
    df["cum_logret_1h_6"] = log_ret_s.rolling(6, min_periods=6).sum().shift(1).to_numpy()
    df["cum_logret_1h_10"] = log_ret_s.rolling(10, min_periods=10).sum().shift(1).to_numpy()
    df["cum_logret_1h_12"] = log_ret_s.rolling(12, min_periods=12).sum().shift(1).to_numpy()
    df["cum_logret_1h_24"] = log_ret_s.rolling(24, min_periods=24).sum().shift(1).to_numpy()

    # Run-length of same-direction bars going BACKWARDS from N (using bars before N).
    df["run_length_into_signal"] = _signed_run_length(open_a, close_a)

    # ATR(14) — already from _compute_signals; just alias.
    df["atr_1h_at_n"] = atr_a

    # ATR regime: atr_at_n / median(atr, last 50 bars including N).
    atr_s = pd.Series(atr_a)
    atr_med50 = atr_s.rolling(int(feat_cfg["atr_regime_window"]), min_periods=int(feat_cfg["atr_regime_window"])).median()
    with np.errstate(invalid="ignore", divide="ignore"):
        df["atr_1h_regime"] = atr_a / atr_med50.to_numpy()
    # ATR slope over last 5 bars: linear regression slope on last 5 ATR values ending at N.
    df["atr_1h_slope_5"] = _slope_5(atr_a, window=5)

    # Range position 20: close[N] within trailing 20-bar high-low range INCLUDING N.
    high_s = pd.Series(high_a)
    low_s = pd.Series(low_a)
    rp_window = int(feat_cfg["range_position_window_1h"])
    h_max = high_s.rolling(rp_window, min_periods=rp_window).max()
    l_min = low_s.rolling(rp_window, min_periods=rp_window).min()
    rng = (h_max - l_min).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        df["range_position_20"] = (close_a - l_min.to_numpy()) / rng

    # ACF1 over last 20 log returns ending at N-1 (strict-trailing — uses [N-19..N-1])
    # We compute over log_ret series shifted by 1.
    log_ret_shifted = log_ret_s.shift(1).to_numpy()
    df["acf1_returns_20"] = _rolling_apply_acf1(log_ret_shifted, int(feat_cfg["acf_window"]))

    # Range expansion: mean(range, last 5) / mean(range, last 20). Using bars [N-4..N] / [N-19..N].
    bar_range = (high_s - low_s).to_numpy()
    bar_range_s = pd.Series(bar_range)
    rng_short = bar_range_s.rolling(int(feat_cfg["range_expansion_short_window"]), min_periods=int(feat_cfg["range_expansion_short_window"])).mean()
    rng_long = bar_range_s.rolling(int(feat_cfg["range_expansion_long_window"]), min_periods=int(feat_cfg["range_expansion_long_window"])).mean()
    with np.errstate(invalid="ignore", divide="ignore"):
        df["range_expansion_5"] = rng_short.to_numpy() / rng_long.to_numpy()

    # Realized vol 24h, 120h: sqrt(sum_sq_logret over last K bars including N) / median over last 500 such windows.
    log_ret_sq_s = pd.Series(np.where(np.isfinite(log_ret), log_ret ** 2, np.nan))
    rv_short_window = int(feat_cfg["realized_vol_short_bars"])
    rv_long_window = int(feat_cfg["realized_vol_long_bars"])
    rv_norm_window = int(feat_cfg["realized_vol_norm_window"])
    rv_short = log_ret_sq_s.rolling(rv_short_window, min_periods=rv_short_window).sum().pow(0.5)
    rv_long = log_ret_sq_s.rolling(rv_long_window, min_periods=rv_long_window).sum().pow(0.5)
    rv_short_med = rv_short.rolling(rv_norm_window, min_periods=rv_norm_window).median()
    rv_long_med = rv_long.rolling(rv_norm_window, min_periods=rv_norm_window).median()
    with np.errstate(invalid="ignore", divide="ignore"):
        df["realized_vol_24h"] = rv_short.to_numpy() / rv_short_med.to_numpy()
        df["realized_vol_120h"] = rv_long.to_numpy() / rv_long_med.to_numpy()

    # Signal-bar properties.
    with np.errstate(invalid="ignore", divide="ignore"):
        df["bar_size_atr"] = (high_a - low_a) / atr_a
        df["bar_body_atr"] = np.abs(close_a - open_a) / atr_a
        bar_height = high_a - low_a
        df["close_position_in_bar"] = np.where(bar_height > 0, (close_a - low_a) / bar_height, np.nan)
    # signal_zscore_100: |log_return[N]| z-score against trailing-100 std of |log_return| (strict — exclude N).
    abs_lr_s = pd.Series(abs_log_ret)
    abs_lr_shifted = abs_lr_s.shift(1)
    abs_lr_mean100 = abs_lr_shifted.rolling(100, min_periods=100).mean()
    abs_lr_std100 = abs_lr_shifted.rolling(100, min_periods=100).std(ddof=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        df["signal_zscore_100"] = (abs_log_ret - abs_lr_mean100.to_numpy()) / abs_lr_std100.to_numpy()

    # Spread at signal: from data 'spread' column (MT5 points, divide by points_per_pip = 10).
    points_per_pip = 10.0
    if "spread" in df.columns:
        sp_pips = df["spread"].astype(float).to_numpy() / points_per_pip
    else:
        sp_pips = np.zeros(n, dtype=float)
    floor_pips = spread_state.floors_pips.get(pair, 0.0)
    with np.errstate(invalid="ignore"):
        floored_mask = (sp_pips < floor_pips) & spread_state.enabled
    eff_pips = np.where(floored_mask, floor_pips, sp_pips)
    df["spread_at_signal_pips"] = eff_pips
    df["spread_floored_at_signal"] = floored_mask

    # 1H volume features.
    if "tick_volume" in df.columns:
        vol_a = df["tick_volume"].astype(float).to_numpy()
    else:
        vol_a = np.zeros(n, dtype=float)
    # Treat zero as NaN for null-handling diagnostics.
    vol_a = np.where(vol_a == 0, np.nan, vol_a)
    df["volume_1h_at_n"] = vol_a
    vol_s = pd.Series(vol_a)
    vol_med_w = int(feat_cfg["volume_median_window_1h"])
    vol_z_w = int(feat_cfg["volume_zscore_window_1h"])
    df["volume_1h_median_50"] = vol_s.rolling(vol_med_w, min_periods=1).median().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        df["volume_1h_ratio"] = df["volume_1h_at_n"].to_numpy() / df["volume_1h_median_50"].to_numpy()
    vol_shifted = vol_s.shift(1)
    vol_mean100 = vol_shifted.rolling(vol_z_w, min_periods=vol_z_w).mean()
    vol_std100 = vol_shifted.rolling(vol_z_w, min_periods=vol_z_w).std(ddof=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        df["volume_1h_zscore_100"] = (df["volume_1h_at_n"].to_numpy() - vol_mean100.to_numpy()) / vol_std100.to_numpy()

    # 1H baseline distances (signed, in ATR units).
    kijun_h1 = _kijun(df, period=26)
    ema20_h1 = _ema(df["close"], span=20)
    ema50_h1 = _ema(df["close"], span=50)
    with np.errstate(invalid="ignore", divide="ignore"):
        df["dist_to_kijun_1h_atr"] = (close_a - kijun_h1.to_numpy()) / atr_a
        df["dist_to_ema20_1h_atr"] = (close_a - ema20_h1.to_numpy()) / atr_a
        df["dist_to_ema50_1h_atr"] = (close_a - ema50_h1.to_numpy()) / atr_a
        # EMA50 1H slope (used for h1_trend_label) — (ema50[N] - ema50[N-5]) / atr_at_n.
        ema50_a = ema50_h1.to_numpy()
        ema50_shift5 = pd.Series(ema50_a).shift(5).to_numpy()
        df["ema50_1h_slope_atr"] = (ema50_a - ema50_shift5) / atr_a

    # Spread-floor liquidity proxy: pct of last 100 1H bars where the floor was applied.
    floored_int = floored_mask.astype(float)
    df["pair_floor_rate_100h"] = pd.Series(floored_int).rolling(100, min_periods=100).mean().to_numpy()

    # Weekend-gap context: bars_to_weekend_close, bars_since_weekend_open.
    # Define a gap as a bar-time delta > 2 hours (typical 1H gap is 1h).
    ts = df[TIME_COL].to_numpy()
    ts_pd = pd.Series(df[TIME_COL])
    diff_hr = ts_pd.diff().dt.total_seconds() / 3600.0
    is_first_after_gap = np.asarray((diff_hr > 2.0).to_numpy()).copy()  # NaN at idx 0 → False
    is_first_after_gap[0] = True  # Treat dataset start as "after a gap"
    is_last_before_gap = np.zeros(n, dtype=bool)
    if n >= 2:
        is_last_before_gap[:-1] = is_first_after_gap[1:]
    is_last_before_gap[-1] = True

    bars_since_open = np.zeros(n, dtype=float)
    bars_to_close = np.zeros(n, dtype=float)
    counter = 0
    for i in range(n):
        if is_first_after_gap[i]:
            counter = 0
        bars_since_open[i] = counter
        counter += 1
    counter = 0
    for i in range(n - 1, -1, -1):
        if is_last_before_gap[i]:
            counter = 0
        bars_to_close[i] = counter
        counter += 1
    df["bars_since_weekend_open"] = bars_since_open
    df["bars_to_weekend_close"] = bars_to_close

    return df


# ---------------------------------------------------------------------------
# 4H / D1 / W1 per-pair feature pipelines (computed once on full series; merged in via asof at signal time)
# ---------------------------------------------------------------------------


def _compute_h4_features(df_h4: pd.DataFrame, feat_cfg: Dict[str, Any]) -> pd.DataFrame:
    """4H feature frame. Adds `close_time_4h = time + 4h` for asof joins."""
    df = df_h4.sort_values(TIME_COL).reset_index(drop=True).copy()
    open_a = df["open"].to_numpy(float)
    high_a = df["high"].to_numpy(float)
    low_a = df["low"].to_numpy(float)
    close_a = df["close"].to_numpy(float)

    prev_close = np.empty(len(df), dtype=float)
    prev_close[0] = np.nan
    prev_close[1:] = close_a[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret = np.log(close_a / prev_close)

    log_ret_s = pd.Series(log_ret)
    df["cum_logret_4h_3"] = log_ret_s.rolling(3, min_periods=3).sum().to_numpy()
    df["cum_logret_4h_6"] = log_ret_s.rolling(6, min_periods=6).sum().to_numpy()
    df["cum_logret_4h_12"] = log_ret_s.rolling(12, min_periods=12).sum().to_numpy()

    atr_4h = _wilder_atr(df, period=14).to_numpy()
    df["atr_4h"] = atr_4h
    atr_med = pd.Series(atr_4h).rolling(int(feat_cfg["atr_regime_window"]), min_periods=int(feat_cfg["atr_regime_window"])).median().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        df["atr_4h_regime"] = atr_4h / atr_med

    rp_window = int(feat_cfg["range_position_window_1h"])  # reuse 20 for 4H
    high_s = pd.Series(high_a)
    low_s = pd.Series(low_a)
    h_max = high_s.rolling(rp_window, min_periods=rp_window).max().to_numpy()
    l_min = low_s.rolling(rp_window, min_periods=rp_window).min().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        df["range_position_4h_20"] = (close_a - l_min) / (h_max - l_min)

    kijun_4h = _kijun(df, period=26).to_numpy()
    ema20_4h = _ema(df["close"], span=20).to_numpy()
    ema50_4h = _ema(df["close"], span=50).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        df["dist_to_kijun_4h_atr"] = (close_a - kijun_4h) / atr_4h
        df["dist_to_ema20_4h_atr"] = (close_a - ema20_4h) / atr_4h
        df["dist_to_ema50_4h_atr"] = (close_a - ema50_4h) / atr_4h
        ema50_shift5 = pd.Series(ema50_4h).shift(5).to_numpy()
        df["ema50_4h_slope_atr"] = (ema50_4h - ema50_shift5) / atr_4h

    if "tick_volume" in df.columns:
        vol_a = df["tick_volume"].astype(float).to_numpy()
    else:
        vol_a = np.zeros(len(df), dtype=float)
    vol_a = np.where(vol_a == 0, np.nan, vol_a)
    df["volume_4h_at_lag1"] = vol_a
    vol_s = pd.Series(vol_a)
    vol_med_w = int(feat_cfg["volume_median_window_4h"])
    vol_z_w = int(feat_cfg["volume_zscore_window_4h"])
    df["volume_4h_median_50"] = vol_s.rolling(vol_med_w, min_periods=1).median().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        df["volume_4h_ratio"] = df["volume_4h_at_lag1"].to_numpy() / df["volume_4h_median_50"].to_numpy()
    vol_shifted = vol_s.shift(1)
    vol_mean = vol_shifted.rolling(vol_z_w, min_periods=vol_z_w).mean().to_numpy()
    vol_std = vol_shifted.rolling(vol_z_w, min_periods=vol_z_w).std(ddof=0).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        df["volume_4h_zscore_100"] = (df["volume_4h_at_lag1"].to_numpy() - vol_mean) / vol_std

    df["close_time_4h"] = df[TIME_COL] + pd.Timedelta(hours=4)
    return df


def _compute_d1_features(df_d1: pd.DataFrame, feat_cfg: Dict[str, Any]) -> pd.DataFrame:
    df = df_d1.sort_values(TIME_COL).reset_index(drop=True).copy()
    open_a = df["open"].to_numpy(float)
    high_a = df["high"].to_numpy(float)
    low_a = df["low"].to_numpy(float)
    close_a = df["close"].to_numpy(float)

    prev_close = np.empty(len(df), dtype=float)
    prev_close[0] = np.nan
    prev_close[1:] = close_a[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret = np.log(close_a / prev_close)

    log_ret_s = pd.Series(log_ret)
    df["cum_logret_d1_3"] = log_ret_s.rolling(3, min_periods=3).sum().to_numpy()
    df["cum_logret_d1_5"] = log_ret_s.rolling(5, min_periods=5).sum().to_numpy()
    df["cum_logret_d1_10"] = log_ret_s.rolling(10, min_periods=10).sum().to_numpy()

    atr_d1 = _wilder_atr(df, period=14).to_numpy()
    df["atr_d1"] = atr_d1
    atr_med = pd.Series(atr_d1).rolling(int(feat_cfg["atr_regime_window"]), min_periods=int(feat_cfg["atr_regime_window"])).median().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        df["atr_d1_regime"] = atr_d1 / atr_med

    kijun_d1 = _kijun(df, period=26).to_numpy()
    ema20_d1 = _ema(df["close"], span=20).to_numpy()
    ema50_d1 = _ema(df["close"], span=50).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        df["dist_to_kijun_d1_atr"] = (close_a - kijun_d1) / atr_d1
        df["dist_to_ema20_d1_atr"] = (close_a - ema20_d1) / atr_d1
        df["dist_to_ema50_d1_atr"] = (close_a - ema50_d1) / atr_d1
        ema20_shift5 = pd.Series(ema20_d1).shift(5).to_numpy()
        ema50_shift5 = pd.Series(ema50_d1).shift(5).to_numpy()
        df["ema20_d1_slope_atr"] = (ema20_d1 - ema20_shift5) / atr_d1
        df["ema50_d1_slope_atr"] = (ema50_d1 - ema50_shift5) / atr_d1
        # Prior day's high/low at bar B = high[B-1] / low[B-1]
        prev_high = pd.Series(high_a).shift(1).to_numpy()
        prev_low = pd.Series(low_a).shift(1).to_numpy()
        df["dist_to_prior_day_high_atr"] = (close_a - prev_high) / atr_d1
        df["dist_to_prior_day_low_atr"] = (close_a - prev_low) / atr_d1

    if "tick_volume" in df.columns:
        vol_a = df["tick_volume"].astype(float).to_numpy()
    else:
        vol_a = np.zeros(len(df), dtype=float)
    vol_a = np.where(vol_a == 0, np.nan, vol_a)
    df["volume_d1_at_lag1"] = vol_a
    vol_s = pd.Series(vol_a)
    vol_med_w = int(feat_cfg["volume_median_window_d1"])
    vol_z_w = int(feat_cfg["volume_zscore_window_d1"])
    df["volume_d1_median_50"] = vol_s.rolling(vol_med_w, min_periods=1).median().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        df["volume_d1_ratio"] = df["volume_d1_at_lag1"].to_numpy() / df["volume_d1_median_50"].to_numpy()
    vol_shifted = vol_s.shift(1)
    vol_mean = vol_shifted.rolling(vol_z_w, min_periods=vol_z_w).mean().to_numpy()
    vol_std = vol_shifted.rolling(vol_z_w, min_periods=vol_z_w).std(ddof=0).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        df["volume_d1_zscore_100"] = (df["volume_d1_at_lag1"].to_numpy() - vol_mean) / vol_std

    return df


def _compute_w1_features(df_w1: pd.DataFrame, feat_cfg: Dict[str, Any]) -> pd.DataFrame:
    df = df_w1.sort_values(TIME_COL).reset_index(drop=True).copy()
    high_a = df["high"].to_numpy(float)
    low_a = df["low"].to_numpy(float)
    close_a = df["close"].to_numpy(float)

    prev_close = np.empty(len(df), dtype=float)
    prev_close[0] = np.nan
    prev_close[1:] = close_a[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret = np.log(close_a / prev_close)

    log_ret_s = pd.Series(log_ret)
    df["cum_logret_w1_3"] = log_ret_s.rolling(3, min_periods=3).sum().to_numpy()
    df["cum_logret_w1_5"] = log_ret_s.rolling(5, min_periods=5).sum().to_numpy()

    # W1 has no "ATR(14)_W1" feature in the spec — but distances are W1 ATR units.
    # Use Wilder ATR(14) on W1 for that normalization.
    atr_w1 = _wilder_atr(df, period=14).to_numpy()

    ema8_w1 = _ema(df["close"], span=8).to_numpy()
    ema20_w1 = _ema(df["close"], span=20).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        df["dist_to_ema8_w1_atr"] = (close_a - ema8_w1) / atr_w1
        df["dist_to_ema20_w1_atr"] = (close_a - ema20_w1) / atr_w1
        ema8_shift5 = pd.Series(ema8_w1).shift(5).to_numpy()
        ema20_shift5 = pd.Series(ema20_w1).shift(5).to_numpy()
        df["ema8_w1_slope_atr"] = (ema8_w1 - ema8_shift5) / atr_w1
        df["ema20_w1_slope_atr"] = (ema20_w1 - ema20_shift5) / atr_w1

    if "tick_volume" in df.columns:
        vol_a = df["tick_volume"].astype(float).to_numpy()
    else:
        vol_a = np.zeros(len(df), dtype=float)
    vol_a = np.where(vol_a == 0, np.nan, vol_a)
    df["volume_w1_at_lag1"] = vol_a
    vol_s = pd.Series(vol_a)
    vol_med_w = int(feat_cfg["volume_median_window_w1"])
    df["volume_w1_median_20"] = vol_s.rolling(vol_med_w, min_periods=1).median().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        df["volume_w1_ratio"] = df["volume_w1_at_lag1"].to_numpy() / df["volume_w1_median_20"].to_numpy()

    return df


# ---------------------------------------------------------------------------
# Lag-1 lookup helpers
# ---------------------------------------------------------------------------


def _weekstart(ts: pd.Timestamp) -> pd.Timestamp:
    """Sunday of the week containing ts (00:00:00, normalised)."""
    days_since_sunday = (ts.dayofweek + 1) % 7  # Sun=6 → 0; Mon=0 → 1; ... Sat=5 → 6
    return (ts.normalize() - pd.Timedelta(days=int(days_since_sunday)))


def _h4_lag1_idx(close_time_arr: np.ndarray, t_n: pd.Timestamp) -> int:
    """Largest index in close_time_arr where close_time ≤ t_n. -1 if none."""
    pos = np.searchsorted(close_time_arr, np.datetime64(t_n.to_datetime64()), side="right") - 1
    return int(pos)


def _d1_lag1_idx(d1_dates_arr: np.ndarray, t_n: pd.Timestamp) -> int:
    """Largest index in d1_dates_arr where d1_date < date(t_n). -1 if none."""
    target = np.datetime64(pd.Timestamp(t_n.normalize()).to_datetime64())
    pos = np.searchsorted(d1_dates_arr, target, side="left") - 1
    return int(pos)


def _w1_lag1_idx(w1_dates_arr: np.ndarray, t_n: pd.Timestamp) -> int:
    """Largest index in w1_dates_arr where w1_date < weekstart(t_n). -1 if none."""
    weekstart = _weekstart(t_n)
    target = np.datetime64(pd.Timestamp(weekstart).to_datetime64())
    pos = np.searchsorted(w1_dates_arr, target, side="left") - 1
    return int(pos)


# ---------------------------------------------------------------------------
# Trade-outcome (Arc-1-equivalent execution)
# ---------------------------------------------------------------------------


@dataclass
class TradeOutcome:
    entry_price: float
    exit_price: float
    sl_price: float
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    exit_reason: str  # "sl" or "time"
    gross_r: float
    net_r: float
    spread_cost_r: float
    mfe_held_atr: float
    mae_held_atr: float
    spread_pips_entry: float
    spread_pips_exit: float
    spread_floored_at_entry: bool
    spread_floored_at_exit: bool
    position_size_units: float


def _trade_outcome(
    pair: str,
    sig_idx: int,
    df_h1: pd.DataFrame,
    atr_at_n: float,
    risk_usd: float,
    cfg: dict,
    spread_state: SpreadFloorState,
    quote_to_usd: Dict[str, pd.Series],
    bar_offset: int,
    sl_atr_mult: float,
) -> Optional[TradeOutcome]:
    """Replicate Arc-1 _execute_signals for a single LONG signal.

    Returns None if the signal can't be executed (no next bar, ATR unavailable, etc.).
    """
    n = len(df_h1)
    entry_idx = sig_idx + bar_offset
    time_exit_idx = entry_idx + 1
    if time_exit_idx >= n:
        return None

    sig_row = df_h1.iloc[sig_idx]
    entry_row = df_h1.iloc[entry_idx]
    time_exit_row = df_h1.iloc[time_exit_idx]
    if not math.isfinite(atr_at_n) or atr_at_n <= 0:
        return None

    direction_int = 1  # all L4 trades are long per L6.0 §6 sign rule

    # Resolve spread at entry.
    sp_entry_raw_pips = float(entry_row.get("spread", 0.0)) / 10.0 if entry_row.get("spread") is not None else 0.0
    if pd.isna(sp_entry_raw_pips):
        sp_entry_raw_pips = 0.0
    pre_apps = spread_state.n_applications
    sp_entry_eff_pips = apply_spread_floor_to_pips(cfg, pair, sp_entry_raw_pips)
    floored_e = spread_state.n_applications > pre_apps

    entry_mid = float(entry_row["open"])
    pip = _pip_size(pair)
    entry_fill = entry_mid + direction_int * (sp_entry_eff_pips * pip) / 2.0

    sl_distance_price = sl_atr_mult * atr_at_n
    sl_price = entry_fill - direction_int * sl_distance_price

    sig_ts = pd.Timestamp(sig_row[TIME_COL])
    quote_to_usd_rate = _quote_to_usd_at(pair, sig_ts, quote_to_usd)
    denom = sl_distance_price * quote_to_usd_rate
    if denom <= 0:
        return None
    position_size_units = risk_usd / denom

    entry_low = float(entry_row["low"])
    entry_high = float(entry_row["high"])
    sl_hit = entry_low <= sl_price  # long-only

    if sl_hit:
        # Intrabar exit at sl_price; spread from entry-bar.
        pre_apps = spread_state.n_applications
        sp_exit_raw = float(entry_row.get("spread", 0.0)) / 10.0 if entry_row.get("spread") is not None else 0.0
        if pd.isna(sp_exit_raw):
            sp_exit_raw = 0.0
        sp_exit_eff_pips = apply_spread_floor_to_pips(cfg, pair, sp_exit_raw)
        floored_x = spread_state.n_applications > pre_apps
        exit_fill = sl_price - direction_int * (sp_exit_eff_pips * pip) / 2.0
        exit_reason = "sl"
        exit_ts = pd.Timestamp(entry_row[TIME_COL])
    else:
        pre_apps = spread_state.n_applications
        sp_exit_raw = float(time_exit_row.get("spread", 0.0)) / 10.0 if time_exit_row.get("spread") is not None else 0.0
        if pd.isna(sp_exit_raw):
            sp_exit_raw = 0.0
        sp_exit_eff_pips = apply_spread_floor_to_pips(cfg, pair, sp_exit_raw)
        floored_x = spread_state.n_applications > pre_apps
        exit_mid = float(time_exit_row["open"])
        exit_fill = exit_mid - direction_int * (sp_exit_eff_pips * pip) / 2.0
        exit_reason = "time"
        exit_ts = pd.Timestamp(time_exit_row[TIME_COL])

    # Net PnL (matches Arc 1)
    net_pnl_per_unit = direction_int * (exit_fill - entry_fill)
    net_pnl_usd = net_pnl_per_unit * position_size_units * quote_to_usd_rate
    net_r = net_pnl_usd / risk_usd if risk_usd > 0 else 0.0

    # Gross PnL (mid-mid) — same exit-bar mid for time, or sl_price for SL hit
    if sl_hit:
        gross_exit_price = sl_price
    else:
        gross_exit_price = float(time_exit_row["open"])
    gross_pnl_per_unit = direction_int * (gross_exit_price - entry_mid)
    gross_pnl_usd = gross_pnl_per_unit * position_size_units * quote_to_usd_rate
    gross_r = gross_pnl_usd / risk_usd if risk_usd > 0 else 0.0
    spread_cost_r = gross_r - net_r

    # MFE/MAE during the held bar (bar N+1) in ATR units (price units)
    mae_price = max(0.0, entry_fill - entry_low)
    mfe_price = max(0.0, entry_high - entry_fill)
    mae_atr = mae_price / atr_at_n if atr_at_n > 0 else math.nan
    mfe_atr = mfe_price / atr_at_n if atr_at_n > 0 else math.nan

    return TradeOutcome(
        entry_price=entry_fill,
        exit_price=exit_fill,
        sl_price=sl_price,
        entry_ts=pd.Timestamp(entry_row[TIME_COL]),
        exit_ts=exit_ts,
        exit_reason=exit_reason,
        gross_r=gross_r,
        net_r=net_r,
        spread_cost_r=spread_cost_r,
        mfe_held_atr=mfe_atr,
        mae_held_atr=mae_atr,
        spread_pips_entry=sp_entry_eff_pips,
        spread_pips_exit=sp_exit_eff_pips,
        spread_floored_at_entry=bool(floored_e),
        spread_floored_at_exit=bool(floored_x),
        position_size_units=position_size_units,
    )


# ---------------------------------------------------------------------------
# Forward-horizon outcome
# ---------------------------------------------------------------------------


@dataclass
class FwdOutcome:
    by_h: Dict[int, Tuple[float, float, float]]  # h → (fwd_logret, fwd_mfe_atr, fwd_mae_atr)
    bars_to_plus_1atr: float
    bars_to_plus_2atr: float
    bars_to_minus_1atr: float
    bars_to_minus_2atr: float


def _forward_outcomes(
    df_h1: pd.DataFrame,
    sig_idx: int,
    bar_offset: int,
    atr_at_n: float,
    horizons: List[int],
    cap_bars: int = 240,
) -> FwdOutcome:
    n = len(df_h1)
    entry_idx = sig_idx + bar_offset
    if entry_idx >= n:
        return FwdOutcome(
            by_h={H: (math.nan, math.nan, math.nan) for H in horizons},
            bars_to_plus_1atr=math.nan,
            bars_to_plus_2atr=math.nan,
            bars_to_minus_1atr=math.nan,
            bars_to_minus_2atr=math.nan,
        )
    entry_price = float(df_h1.iloc[entry_idx]["open"])

    by_h: Dict[int, Tuple[float, float, float]] = {}
    high_a = df_h1["high"].to_numpy(float)
    low_a = df_h1["low"].to_numpy(float)
    close_a = df_h1["close"].to_numpy(float)

    for H in horizons:
        end_idx = entry_idx + H
        if end_idx >= n:
            by_h[H] = (math.nan, math.nan, math.nan)
            continue
        # log return entry -> close[end_idx]
        if entry_price > 0 and close_a[end_idx] > 0:
            fwd_lr = float(np.log(close_a[end_idx] / entry_price))
        else:
            fwd_lr = math.nan
        # MFE/MAE over [entry_idx, end_idx]
        slc_high = high_a[entry_idx: end_idx + 1]
        slc_low = low_a[entry_idx: end_idx + 1]
        mfe_price = float(np.max(slc_high)) - entry_price
        mae_price = entry_price - float(np.min(slc_low))
        if atr_at_n > 0 and math.isfinite(atr_at_n):
            mfe_atr = mfe_price / atr_at_n
            mae_atr = mae_price / atr_at_n
        else:
            mfe_atr = math.nan
            mae_atr = math.nan
        by_h[H] = (fwd_lr, mfe_atr, mae_atr)

    # Bars to +1 ATR / +2 ATR / -1 ATR / -2 ATR (capped at 240 bars from entry)
    cap_end = min(entry_idx + cap_bars, n - 1)
    plus_1 = entry_price + 1.0 * atr_at_n
    plus_2 = entry_price + 2.0 * atr_at_n
    minus_1 = entry_price - 1.0 * atr_at_n
    minus_2 = entry_price - 2.0 * atr_at_n
    bars_to: Dict[float, float] = {plus_1: math.nan, plus_2: math.nan, minus_1: math.nan, minus_2: math.nan}
    if math.isfinite(atr_at_n) and atr_at_n > 0:
        for j in range(entry_idx, cap_end + 1):
            bars_off = j - entry_idx
            if math.isnan(bars_to[plus_1]) and high_a[j] >= plus_1:
                bars_to[plus_1] = float(bars_off)
            if math.isnan(bars_to[plus_2]) and high_a[j] >= plus_2:
                bars_to[plus_2] = float(bars_off)
            if math.isnan(bars_to[minus_1]) and low_a[j] <= minus_1:
                bars_to[minus_1] = float(bars_off)
            if math.isnan(bars_to[minus_2]) and low_a[j] <= minus_2:
                bars_to[minus_2] = float(bars_off)
            if (not math.isnan(bars_to[plus_1])
                and not math.isnan(bars_to[plus_2])
                and not math.isnan(bars_to[minus_1])
                and not math.isnan(bars_to[minus_2])):
                break

    return FwdOutcome(
        by_h=by_h,
        bars_to_plus_1atr=bars_to[plus_1],
        bars_to_plus_2atr=bars_to[plus_2],
        bars_to_minus_1atr=bars_to[minus_1],
        bars_to_minus_2atr=bars_to[minus_2],
    )


# ---------------------------------------------------------------------------
# Cross-pair / portfolio context
# ---------------------------------------------------------------------------


def _build_currency_basket_returns(
    pair_h1: Dict[str, pd.DataFrame],
) -> Dict[str, pd.Series]:
    """Per-currency time-indexed series of "per-bar log return signed so positive=ccy strengthening".

    The result is a Series indexed by bar timestamp giving the AVERAGE per-bar
    signed return across all pairs involving that currency.
    """
    ccy_pairs: Dict[str, List[Tuple[str, int]]] = {"USD": [], "EUR": [], "GBP": [], "JPY": [], "AUD": [], "NZD": [], "CAD": [], "CHF": []}
    for pair in pair_h1:
        base, quote = pair.split("_")
        for ccy in (base, quote):
            sign = +1 if ccy == base else -1
            ccy_pairs.setdefault(ccy, []).append((pair, sign))

    # Build per-pair signed log_return series indexed by time.
    log_ret_by_pair: Dict[str, pd.Series] = {}
    for pair, df in pair_h1.items():
        # log_return is already on df from _compute_signals
        s = pd.Series(df["log_return"].to_numpy(), index=pd.DatetimeIndex(df[TIME_COL]))
        log_ret_by_pair[pair] = s

    out: Dict[str, pd.Series] = {}
    for ccy, plist in ccy_pairs.items():
        if not plist:
            continue
        contribs: List[pd.Series] = []
        for (pair, sign) in plist:
            contribs.append(sign * log_ret_by_pair[pair])
        # Outer-align across pairs by timestamp; take row-wise mean (NaN-aware).
        df_ccy = pd.concat(contribs, axis=1, keys=[p for p, _ in plist])
        out[ccy] = df_ccy.mean(axis=1, skipna=True)
    return out


def _basket_3h_at(ts: pd.Timestamp, basket_series: pd.Series) -> float:
    """Cumulative basket return over the 3 1H bars ending at ts (inclusive)."""
    idx = basket_series.index.searchsorted(ts, side="right") - 1
    if idx < 2:
        return math.nan
    # bars [idx-2, idx-1, idx]
    s = basket_series.iloc[idx - 2: idx + 1]
    if s.isna().all():
        return math.nan
    return float(s.sum(skipna=True))


# ---------------------------------------------------------------------------
# Session classifier
# ---------------------------------------------------------------------------


def _session_label(hour_utc: int, sessions_cfg: Dict[str, List[int]]) -> str:
    """Return the session name whose hour-window contains hour_utc.

    Each session block is [start_hour, end_hour] inclusive on both ends.
    If start <= end, normal range. If start > end, wraps over midnight.
    Order matters: more specific sessions (e.g., london_ny_overlap) should override
    less specific ones (e.g., london, ny). We rely on dictionary ordering.
    """
    # Specific (overlap) first; then session-exclusive others; else off_hours.
    for name, rng in sessions_cfg.items():
        if not rng:
            continue
        start, end = rng[0], rng[1]
        if start <= end:
            if start <= hour_utc <= end:
                return name
        else:
            if hour_utc >= start or hour_utc <= end:
                return name
    return "off_hours"


# ---------------------------------------------------------------------------
# Classification labels
# ---------------------------------------------------------------------------


def _trend_label(slope: float, up_min: float, down_max: float) -> str:
    if not math.isfinite(slope):
        return "unknown"
    if slope > up_min:
        return "up"
    if slope < down_max:
        return "down"
    return "flat"


def _mtf_alignment(d1: str, h4: str, h1: str) -> str:
    if d1 == "up" and h4 == "up" and h1 == "up":
        return "aligned_up"
    if d1 == "down" and h4 == "down" and h1 == "down":
        return "aligned_down"
    return "mixed"


def _structural_pattern(d1: str, h4: str, pre_momentum: str) -> str:
    """Decision tree per task brief.

    - reversal_at_uptrend_high: d1 up AND h4 up AND pre_momentum up
    - continuation_in_downtrend: d1 down AND h4 down AND pre_momentum down
    - pullback_in_uptrend: d1 up AND h4 up AND pre_momentum down
    - range_extreme: any of (d1, h4) is mixed/flat/unknown
    - mixed: anything else
    """
    if d1 in ("flat", "unknown", "mixed") or h4 in ("flat", "unknown", "mixed"):
        return "range_extreme"
    if d1 == "up" and h4 == "up":
        if pre_momentum == "up":
            return "reversal_at_uptrend_high"
        if pre_momentum == "down":
            return "pullback_in_uptrend"
        return "mixed"
    if d1 == "down" and h4 == "down":
        if pre_momentum == "down":
            return "continuation_in_downtrend"
        return "mixed"
    return "mixed"


# ---------------------------------------------------------------------------
# Arc 1 fold disposition
# ---------------------------------------------------------------------------


_ARC1_FOLDS = [
    (1, pd.Timestamp("2020-10-01"), pd.Timestamp("2021-07-01"), "catastrophic"),
    (2, pd.Timestamp("2021-07-01"), pd.Timestamp("2022-04-01"), "catastrophic"),
    (3, pd.Timestamp("2022-04-01"), pd.Timestamp("2023-01-01"), "profitable"),
    (4, pd.Timestamp("2023-01-01"), pd.Timestamp("2023-10-01"), "profitable"),
    (5, pd.Timestamp("2023-10-01"), pd.Timestamp("2024-07-01"), "mild_negative"),
    (6, pd.Timestamp("2024-07-01"), pd.Timestamp("2025-04-01"), "catastrophic"),
    (7, pd.Timestamp("2025-04-01"), pd.Timestamp("2026-01-01"), "mild_negative"),
]


def _arc1_fold(ts: pd.Timestamp) -> Tuple[int, str]:
    for fid, start, end, dispo in _ARC1_FOLDS:
        if start <= ts < end:
            return fid, dispo
    return -1, "out_of_fold"


# ---------------------------------------------------------------------------
# Output column order
# ---------------------------------------------------------------------------


FEATURE_COLUMNS: List[str] = [
    "pair", "signal_bar_ts", "fold_id", "arc1_fold_disposition",
    # Pre-signal 1H context
    "cum_logret_1h_3", "cum_logret_1h_6", "cum_logret_1h_10", "cum_logret_1h_12", "cum_logret_1h_24",
    "run_length_into_signal",
    "atr_1h_at_n", "atr_1h_regime", "atr_1h_slope_5",
    "range_position_20", "acf1_returns_20", "range_expansion_5",
    "realized_vol_24h", "realized_vol_120h",
    # Signal-bar properties
    "bar_size_atr", "bar_body_atr", "close_position_in_bar",
    "signal_zscore_100", "spread_at_signal_pips", "spread_floored_at_signal",
    # 1H volume
    "volume_1h_at_n", "volume_1h_median_50", "volume_1h_ratio", "volume_1h_zscore_100",
    # 1H baseline distances
    "dist_to_kijun_1h_atr", "dist_to_ema20_1h_atr", "dist_to_ema50_1h_atr",
    "ema50_1h_slope_atr",
    # 4H lag-1 features
    "ts_4h_used",
    "cum_logret_4h_3", "cum_logret_4h_6", "cum_logret_4h_12",
    "atr_4h", "atr_4h_regime", "range_position_4h_20",
    "dist_to_kijun_4h_atr", "dist_to_ema20_4h_atr", "dist_to_ema50_4h_atr",
    "ema50_4h_slope_atr",
    "volume_4h_at_lag1", "volume_4h_median_50", "volume_4h_ratio", "volume_4h_zscore_100",
    # D1 lag-1 features
    "ts_d1_used",
    "cum_logret_d1_3", "cum_logret_d1_5", "cum_logret_d1_10",
    "atr_d1", "atr_d1_regime",
    "dist_to_kijun_d1_atr", "dist_to_ema20_d1_atr", "dist_to_ema50_d1_atr",
    "ema20_d1_slope_atr", "ema50_d1_slope_atr",
    "dist_to_prior_day_high_atr", "dist_to_prior_day_low_atr",
    "volume_d1_at_lag1", "volume_d1_median_50", "volume_d1_ratio", "volume_d1_zscore_100",
    # W1 lag-1 features
    "ts_w1_used",
    "cum_logret_w1_3", "cum_logret_w1_5",
    "dist_to_ema8_w1_atr", "dist_to_ema20_w1_atr",
    "ema8_w1_slope_atr", "ema20_w1_slope_atr",
    "volume_w1_at_lag1", "volume_w1_median_20", "volume_w1_ratio",
    # Time / session / liquidity
    "hour_utc", "dow", "session",
    "bars_to_weekend_close", "bars_since_weekend_open",
    "pair_floor_rate_100h",
    # Cross-pair / portfolio context
    "concurrent_signals_same_bar", "concurrent_signals_within_3h",
    "usd_basket_3h", "eur_basket_3h", "jpy_basket_3h", "gbp_basket_3h",
    "time_since_last_signal_same_pair_bars", "cluster_label",
    # Classification labels
    "d1_trend_label", "h4_trend_label", "h1_trend_label",
    "mtf_alignment", "pre_momentum_label", "structural_pattern",
    # Trade-level outcome
    "entry_price", "exit_price", "sl_price",
    "entry_ts", "exit_ts", "exit_reason",
    "gross_r", "net_r", "spread_cost_r",
    "mfe_held_atr", "mae_held_atr",
    # Forward-horizon outcomes
    "fwd_logret_h1", "fwd_logret_h6", "fwd_logret_h24", "fwd_logret_h72", "fwd_logret_h120", "fwd_logret_h240",
    "fwd_mfe_h1_atr", "fwd_mfe_h6_atr", "fwd_mfe_h24_atr", "fwd_mfe_h72_atr", "fwd_mfe_h120_atr", "fwd_mfe_h240_atr",
    "fwd_mae_h1_atr", "fwd_mae_h6_atr", "fwd_mae_h24_atr", "fwd_mae_h72_atr", "fwd_mae_h120_atr", "fwd_mae_h240_atr",
    "bars_to_plus_1atr_capped_240h", "bars_to_plus_2atr_capped_240h",
    "bars_to_minus_1atr_capped_240h", "bars_to_minus_2atr_capped_240h",
]


# ---------------------------------------------------------------------------
# CSV value formatting (deterministic)
# ---------------------------------------------------------------------------


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (pd.Timestamp,)):
        return v.isoformat()
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, (np.bool_,)):
        return "True" if bool(v) else "False"
    if isinstance(v, float):
        if math.isnan(v):
            return ""
        if not math.isfinite(v):
            return ""
        return f"{v:.10g}"
    if isinstance(v, (np.floating,)):
        f = float(v)
        if math.isnan(f) or not math.isfinite(f):
            return ""
        return f"{f:.10g}"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return str(v)


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def run_characterisation(config_path: str | Path) -> Dict[str, Any]:
    """Run the L4 characterisation pipeline; return a manifest of artefacts.

    Manifest keys:
      'features_csv': str (absolute path)
      'output_dir': str (absolute path)
      'n_signals_in_window': int
      'n_lookahead_assertion_failures': int
      'pair_signal_counts': Dict[str, int]
      'arc1_overlap_count_check': Dict[str, int] (Arc-1 overlap-window count vs ours)
    """
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = (REPO_ROOT / cfg_path).resolve()
    cfg = load_and_validate_config(str(cfg_path))

    sig_cfg = cfg["signal"]
    win_cfg = cfg["window"]
    exec_cfg = cfg["execution"]
    char_cfg = cfg["characterisation"]
    feat_cfg = char_cfg["features"]
    sessions_cfg = char_cfg["sessions"]
    horizons: List[int] = [int(h) for h in char_cfg["forward_horizons"]]
    pairs: List[str] = list(cfg["pairs"])

    output_dir = Path(char_cfg["output_dir"])
    if not output_dir.is_absolute():
        output_dir = (REPO_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sig_window_start = pd.Timestamp(win_cfg["signal_start"])
    sig_window_end = pd.Timestamp(win_cfg["signal_end"])
    data_load_start = pd.Timestamp(win_cfg["data_load_start"])

    # Spread floor (with hash check)
    spread_state = load_spread_floor(cfg)
    cfg[STATE_CFG_KEY] = spread_state
    cfg.setdefault("spreads", {})
    cfg["spreads"]["enabled"] = True
    cfg["spreads"]["points_per_pip"] = 10.0
    print(format_startup_log(spread_state))

    # Load all 4 TFs per pair
    pair_h1: Dict[str, pd.DataFrame] = {}
    pair_h4: Dict[str, pd.DataFrame] = {}
    pair_d1: Dict[str, pd.DataFrame] = {}
    pair_w1: Dict[str, pd.DataFrame] = {}
    for pair in pairs:
        pair_h1[pair] = _load_pair_tf_csv(pair, char_cfg["data_paths"]["h1"], data_load_start)
        pair_h4[pair] = _load_pair_tf_csv(pair, char_cfg["data_paths"]["h4"], data_load_start)
        pair_d1[pair] = _load_pair_tf_csv(pair, char_cfg["data_paths"]["d1"], data_load_start)
        pair_w1[pair] = _load_pair_tf_csv(pair, char_cfg["data_paths"]["w1"], data_load_start)

    # FX conversion table (for trade outcome USD-PnL)
    quote_to_usd = _build_quote_to_usd_table(pair_h1)

    # Compute 1H feature frame for each pair (this also computes signal_fired)
    print(f"Computing 1H features for {len(pairs)} pairs ...")
    h1_features: Dict[str, pd.DataFrame] = {}
    for i, pair in enumerate(pairs, 1):
        h1_features[pair] = _compute_h1_features(pair_h1[pair], pair, feat_cfg, sig_cfg, spread_state, cfg)
        print(f"  [{i}/{len(pairs)}] {pair}: {len(h1_features[pair])} 1H bars, "
              f"{int(h1_features[pair]['signal_fired'].sum())} signals (full data)")

    # Compute 4H/D1/W1 feature frames
    print(f"Computing 4H/D1/W1 features for {len(pairs)} pairs ...")
    h4_features: Dict[str, pd.DataFrame] = {}
    d1_features: Dict[str, pd.DataFrame] = {}
    w1_features: Dict[str, pd.DataFrame] = {}
    for pair in pairs:
        h4_features[pair] = _compute_h4_features(pair_h4[pair], feat_cfg)
        d1_features[pair] = _compute_d1_features(pair_d1[pair], feat_cfg)
        w1_features[pair] = _compute_w1_features(pair_w1[pair], feat_cfg)

    # Currency basket signed-return per-bar series
    print("Computing currency-basket per-bar series ...")
    ccy_baskets = _build_currency_basket_returns(h1_features)

    # Build a global signal index for cross-pair concurrent counts
    print("Building global signal index for concurrent counts ...")
    all_signal_ts: List[pd.Timestamp] = []
    all_signal_pair: List[str] = []
    for pair in pairs:
        df = h1_features[pair]
        sig_mask = df["signal_fired"].to_numpy()
        ts = df[TIME_COL].to_numpy()
        for i in range(len(df)):
            if sig_mask[i]:
                t = pd.Timestamp(ts[i])
                if sig_window_start <= t <= sig_window_end:
                    all_signal_ts.append(t)
                    all_signal_pair.append(pair)
    sig_ts_arr = np.array(all_signal_ts, dtype="datetime64[ns]")
    sig_pair_arr = np.array(all_signal_pair, dtype=object)
    # For "same-bar concurrent" lookup: sort by timestamp
    sort_idx = np.argsort(sig_ts_arr)
    sig_ts_sorted = sig_ts_arr[sort_idx]
    sig_pair_sorted = sig_pair_arr[sort_idx]

    # Iterate over signals per pair, in chronological order (to also compute time_since_last_signal)
    print("Iterating signals; building feature rows ...")
    rows: List[Dict[str, Any]] = []
    pair_signal_counts: Dict[str, int] = {p: 0 for p in pairs}
    n_lookahead_failures = 0
    last_signal_idx_per_pair: Dict[str, int] = {}

    risk_usd = float(exec_cfg["starting_balance"]) * float(exec_cfg["pct_per_trade"])
    bar_offset = int(exec_cfg["bar_offset"])
    sl_atr_mult = float(exec_cfg["sl_atr_mult"])

    concurrent_window_hours = int(char_cfg["concurrent_window_hours"])

    for pair in pairs:
        df = h1_features[pair]
        df_h4 = h4_features[pair]
        df_d1 = d1_features[pair]
        df_w1 = w1_features[pair]
        h4_close_time_arr = df_h4["close_time_4h"].to_numpy(dtype="datetime64[ns]")
        d1_dates_arr = df_d1[TIME_COL].dt.normalize().to_numpy(dtype="datetime64[ns]")
        w1_dates_arr = df_w1[TIME_COL].dt.normalize().to_numpy(dtype="datetime64[ns]")

        sig_mask = df["signal_fired"].to_numpy()
        ts_h1 = df[TIME_COL].to_numpy()

        signal_indices = np.where(sig_mask)[0]
        # Filter to signal window
        signal_indices = [int(i) for i in signal_indices
                          if sig_window_start <= pd.Timestamp(ts_h1[i]) <= sig_window_end]
        pair_signal_counts[pair] = len(signal_indices)

        for sig_idx in signal_indices:
            t_n = pd.Timestamp(ts_h1[sig_idx])
            row = {"pair": pair, "signal_bar_ts": t_n}

            # Arc 1 fold + disposition
            fid, dispo = _arc1_fold(t_n)
            row["fold_id"] = fid if fid > 0 else "out_of_fold"
            row["arc1_fold_disposition"] = dispo

            sig_row = df.iloc[sig_idx]
            atr_at_n = float(sig_row["atr_1h_at_n"])

            # 1H context features (already computed columns)
            for col in (
                "cum_logret_1h_3", "cum_logret_1h_6", "cum_logret_1h_10",
                "cum_logret_1h_12", "cum_logret_1h_24",
                "run_length_into_signal", "atr_1h_at_n", "atr_1h_regime", "atr_1h_slope_5",
                "range_position_20", "acf1_returns_20", "range_expansion_5",
                "realized_vol_24h", "realized_vol_120h",
                "bar_size_atr", "bar_body_atr", "close_position_in_bar",
                "signal_zscore_100", "spread_at_signal_pips", "spread_floored_at_signal",
                "volume_1h_at_n", "volume_1h_median_50", "volume_1h_ratio", "volume_1h_zscore_100",
                "dist_to_kijun_1h_atr", "dist_to_ema20_1h_atr", "dist_to_ema50_1h_atr",
                "ema50_1h_slope_atr",
                "pair_floor_rate_100h",
                "bars_to_weekend_close", "bars_since_weekend_open",
            ):
                row[col] = sig_row[col]

            # Lag-1 lookups + assertions
            h4_idx = _h4_lag1_idx(h4_close_time_arr, t_n)
            d1_idx = _d1_lag1_idx(d1_dates_arr, t_n)
            w1_idx = _w1_lag1_idx(w1_dates_arr, t_n)

            ts_4h_used = pd.Timestamp(df_h4.iloc[h4_idx][TIME_COL]) if h4_idx >= 0 else pd.NaT
            ts_d1_used = pd.Timestamp(df_d1.iloc[d1_idx][TIME_COL]) if d1_idx >= 0 else pd.NaT
            ts_w1_used = pd.Timestamp(df_w1.iloc[w1_idx][TIME_COL]) if w1_idx >= 0 else pd.NaT

            # Runtime lag-1 assertions
            try:
                if pd.notna(ts_4h_used):
                    # ts_4h_used is the 4H bar OPEN time; close = open + 4h must be ≤ t_n
                    assert (ts_4h_used + pd.Timedelta(hours=4)) <= t_n, \
                        f"4H lookahead at {pair} {t_n}: ts_4h_used={ts_4h_used}"
                    assert ts_4h_used <= t_n, f"4H bar-time lookahead at {pair} {t_n}"
                if pd.notna(ts_d1_used):
                    assert ts_d1_used < t_n, f"D1 lookahead at {pair} {t_n}: ts_d1_used={ts_d1_used}"
                    assert ts_d1_used.date() < t_n.date(), \
                        f"D1 same-day at {pair} {t_n}: ts_d1_used.date={ts_d1_used.date()}"
                if pd.notna(ts_w1_used):
                    assert ts_w1_used < t_n, f"W1 lookahead at {pair} {t_n}"
                    assert ts_w1_used < _weekstart(t_n), \
                        f"W1 same-week at {pair} {t_n}: weekstart={_weekstart(t_n)}, ts_w1_used={ts_w1_used}"
            except AssertionError as e:
                n_lookahead_failures += 1
                raise RuntimeError(f"Lag-1 assertion failed: {e}")

            row["ts_4h_used"] = ts_4h_used
            row["ts_d1_used"] = ts_d1_used
            row["ts_w1_used"] = ts_w1_used

            # 4H features
            if h4_idx >= 0:
                h4_row = df_h4.iloc[h4_idx]
                for col in (
                    "cum_logret_4h_3", "cum_logret_4h_6", "cum_logret_4h_12",
                    "atr_4h", "atr_4h_regime", "range_position_4h_20",
                    "dist_to_kijun_4h_atr", "dist_to_ema20_4h_atr", "dist_to_ema50_4h_atr",
                    "ema50_4h_slope_atr",
                    "volume_4h_at_lag1", "volume_4h_median_50", "volume_4h_ratio", "volume_4h_zscore_100",
                ):
                    row[col] = h4_row[col]
            else:
                for col in (
                    "cum_logret_4h_3", "cum_logret_4h_6", "cum_logret_4h_12",
                    "atr_4h", "atr_4h_regime", "range_position_4h_20",
                    "dist_to_kijun_4h_atr", "dist_to_ema20_4h_atr", "dist_to_ema50_4h_atr",
                    "ema50_4h_slope_atr",
                    "volume_4h_at_lag1", "volume_4h_median_50", "volume_4h_ratio", "volume_4h_zscore_100",
                ):
                    row[col] = math.nan

            # D1 features
            if d1_idx >= 0:
                d1_row = df_d1.iloc[d1_idx]
                for col in (
                    "cum_logret_d1_3", "cum_logret_d1_5", "cum_logret_d1_10",
                    "atr_d1", "atr_d1_regime",
                    "dist_to_kijun_d1_atr", "dist_to_ema20_d1_atr", "dist_to_ema50_d1_atr",
                    "ema20_d1_slope_atr", "ema50_d1_slope_atr",
                    "dist_to_prior_day_high_atr", "dist_to_prior_day_low_atr",
                    "volume_d1_at_lag1", "volume_d1_median_50", "volume_d1_ratio", "volume_d1_zscore_100",
                ):
                    row[col] = d1_row[col]
            else:
                for col in (
                    "cum_logret_d1_3", "cum_logret_d1_5", "cum_logret_d1_10",
                    "atr_d1", "atr_d1_regime",
                    "dist_to_kijun_d1_atr", "dist_to_ema20_d1_atr", "dist_to_ema50_d1_atr",
                    "ema20_d1_slope_atr", "ema50_d1_slope_atr",
                    "dist_to_prior_day_high_atr", "dist_to_prior_day_low_atr",
                    "volume_d1_at_lag1", "volume_d1_median_50", "volume_d1_ratio", "volume_d1_zscore_100",
                ):
                    row[col] = math.nan

            # W1 features
            if w1_idx >= 0:
                w1_row = df_w1.iloc[w1_idx]
                for col in (
                    "cum_logret_w1_3", "cum_logret_w1_5",
                    "dist_to_ema8_w1_atr", "dist_to_ema20_w1_atr",
                    "ema8_w1_slope_atr", "ema20_w1_slope_atr",
                    "volume_w1_at_lag1", "volume_w1_median_20", "volume_w1_ratio",
                ):
                    row[col] = w1_row[col]
            else:
                for col in (
                    "cum_logret_w1_3", "cum_logret_w1_5",
                    "dist_to_ema8_w1_atr", "dist_to_ema20_w1_atr",
                    "ema8_w1_slope_atr", "ema20_w1_slope_atr",
                    "volume_w1_at_lag1", "volume_w1_median_20", "volume_w1_ratio",
                ):
                    row[col] = math.nan

            # Time / session
            row["hour_utc"] = int(t_n.hour)
            row["dow"] = int(t_n.dayofweek)
            row["session"] = _session_label(int(t_n.hour), sessions_cfg)

            # Cross-pair / portfolio
            # concurrent_signals_same_bar: count other pairs with signal at exact T_N
            mask_same_bar = (sig_ts_sorted == np.datetime64(t_n.to_datetime64()))
            same_bar_pairs = sig_pair_sorted[mask_same_bar]
            row["concurrent_signals_same_bar"] = int(len(same_bar_pairs)) - 1  # exclude self
            # concurrent_signals_within_3h: signals on any pair within [T_N-3h, T_N+3h], excluding self
            t_lo = np.datetime64((t_n - pd.Timedelta(hours=concurrent_window_hours)).to_datetime64())
            t_hi = np.datetime64((t_n + pd.Timedelta(hours=concurrent_window_hours)).to_datetime64())
            lo_idx = np.searchsorted(sig_ts_sorted, t_lo, side="left")
            hi_idx = np.searchsorted(sig_ts_sorted, t_hi, side="right")
            window_count = int(hi_idx - lo_idx)
            # Exclude self (count of same pair at same bar):
            self_in_window = int(np.sum(
                (sig_pair_sorted[lo_idx:hi_idx] == pair)
                & (sig_ts_sorted[lo_idx:hi_idx] == np.datetime64(t_n.to_datetime64()))
            ))
            row["concurrent_signals_within_3h"] = window_count - self_in_window

            for ccy_label, ccy_key in (("usd", "USD"), ("eur", "EUR"), ("jpy", "JPY"), ("gbp", "GBP")):
                row[f"{ccy_label}_basket_3h"] = _basket_3h_at(t_n, ccy_baskets[ccy_key]) if ccy_key in ccy_baskets else math.nan

            if pair in last_signal_idx_per_pair:
                row["time_since_last_signal_same_pair_bars"] = float(sig_idx - last_signal_idx_per_pair[pair])
            else:
                row["time_since_last_signal_same_pair_bars"] = math.nan
            last_signal_idx_per_pair[pair] = sig_idx
            tssl = row["time_since_last_signal_same_pair_bars"]
            row["cluster_label"] = "cluster" if (math.isfinite(tssl) and tssl <= 6) else "isolated"

            # Classification labels
            up_min = float(char_cfg["trend_label_thresholds"]["up_min"])
            down_max = float(char_cfg["trend_label_thresholds"]["down_max"])
            d1_slope = row["ema20_d1_slope_atr"]
            h4_slope = row["ema50_4h_slope_atr"]
            h1_slope = row["ema50_1h_slope_atr"]
            d1_lab = _trend_label(float(d1_slope) if pd.notna(d1_slope) else math.nan, up_min, down_max)
            h4_lab = _trend_label(float(h4_slope) if pd.notna(h4_slope) else math.nan, up_min, down_max)
            h1_lab = _trend_label(float(h1_slope) if pd.notna(h1_slope) else math.nan, up_min, down_max)
            row["d1_trend_label"] = d1_lab
            row["h4_trend_label"] = h4_lab
            row["h1_trend_label"] = h1_lab
            row["mtf_alignment"] = _mtf_alignment(d1_lab, h4_lab, h1_lab)
            pm_field = char_cfg["pre_momentum_label"]["field"]
            pm_up = float(char_cfg["pre_momentum_label"]["up_min"])
            pm_down = float(char_cfg["pre_momentum_label"]["down_max"])
            pm_val = row.get(pm_field)
            if pm_val is None or (isinstance(pm_val, float) and not math.isfinite(pm_val)):
                pm_lab = "unknown"
            else:
                pm_v = float(pm_val)
                if pm_v > pm_up:
                    pm_lab = "up"
                elif pm_v < pm_down:
                    pm_lab = "down"
                else:
                    pm_lab = "flat"
            row["pre_momentum_label"] = pm_lab
            row["structural_pattern"] = _structural_pattern(d1_lab, h4_lab, pm_lab)

            # Trade outcome (Arc 1 semantics, on a per-trade basis — no cross-trade equity tracking;
            # this is descriptive, not a backtest. risk_usd is constant 1% of starting_balance.)
            outcome = _trade_outcome(
                pair, sig_idx, df, atr_at_n, risk_usd, cfg,
                spread_state, quote_to_usd, bar_offset, sl_atr_mult,
            )
            if outcome is None:
                for col in (
                    "entry_price", "exit_price", "sl_price",
                    "entry_ts", "exit_ts", "exit_reason",
                    "gross_r", "net_r", "spread_cost_r",
                    "mfe_held_atr", "mae_held_atr",
                ):
                    row[col] = math.nan if col != "exit_reason" else ""
            else:
                row["entry_price"] = outcome.entry_price
                row["exit_price"] = outcome.exit_price
                row["sl_price"] = outcome.sl_price
                row["entry_ts"] = outcome.entry_ts
                row["exit_ts"] = outcome.exit_ts
                row["exit_reason"] = outcome.exit_reason
                row["gross_r"] = outcome.gross_r
                row["net_r"] = outcome.net_r
                row["spread_cost_r"] = outcome.spread_cost_r
                row["mfe_held_atr"] = outcome.mfe_held_atr
                row["mae_held_atr"] = outcome.mae_held_atr

            # Forward-horizon outcomes
            fwd = _forward_outcomes(df, sig_idx, bar_offset, atr_at_n, horizons, cap_bars=240)
            for H in horizons:
                fr, fmfe, fmae = fwd.by_h[H]
                row[f"fwd_logret_h{H}"] = fr
                row[f"fwd_mfe_h{H}_atr"] = fmfe
                row[f"fwd_mae_h{H}_atr"] = fmae
            row["bars_to_plus_1atr_capped_240h"] = fwd.bars_to_plus_1atr
            row["bars_to_plus_2atr_capped_240h"] = fwd.bars_to_plus_2atr
            row["bars_to_minus_1atr_capped_240h"] = fwd.bars_to_minus_1atr
            row["bars_to_minus_2atr_capped_240h"] = fwd.bars_to_minus_2atr

            rows.append(row)

    print(f"  total signals in window: {len(rows)}")

    # Sort rows by (pair, signal_bar_ts) for determinism
    rows.sort(key=lambda r: (r["pair"], r["signal_bar_ts"]))

    # Write signals_features.csv
    features_csv = output_dir / "signals_features.csv"
    with features_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(FEATURE_COLUMNS)
        for r in rows:
            w.writerow([_fmt(r.get(c)) for c in FEATURE_COLUMNS])
    print(f"Wrote {features_csv}")

    return {
        "features_csv": str(features_csv),
        "output_dir": str(output_dir),
        "n_signals_in_window": len(rows),
        "n_lookahead_assertion_failures": n_lookahead_failures,
        "pair_signal_counts": pair_signal_counts,
    }


__all__ = ["run_characterisation", "FEATURE_COLUMNS"]
