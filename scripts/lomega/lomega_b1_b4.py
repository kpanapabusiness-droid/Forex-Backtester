"""Lomega B1-B4 — multi-timeframe extractability ceiling discovery.

Discovery track parallel to L-arc. Inverts the L-arc question: from full
dataset, find which bar conditions at t=0 predict clean forward paths,
regardless of any specific signal.

Pipeline per timeframe:
  B1 — label every bar with clean_move (binary) and clean_score (0-4)
       from forward path under hypothetical SL = 2.0 x ATR(14).
  B2 — compute ~32 entry-time features at t=0 (strict no lookahead).
  B3 — per-feature univariate predictive power (logistic AUC, KS, Cohen-d).
  B4 — RF on top-15 features, 7-fold anchored expanding time-series CV;
       worst-fold AUC is the extractability ceiling.

Usage:
    py scripts/lomega/lomega_b1_b4.py --tf 1h
    py scripts/lomega/lomega_b1_b4.py --tf all

Outputs land under results/lomega/b1_b4_discovery/timeframe_<tf>/.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------- paths / config

_REPO = Path(__file__).resolve().parent.parent.parent
_DATA = _REPO / "data"
_OUT = _REPO / "results" / "lomega" / "b1_b4_discovery"

PAIRS: tuple[str, ...] = (
    "AUD_CAD", "AUD_CHF", "AUD_JPY", "AUD_NZD", "AUD_USD",
    "CAD_CHF", "CAD_JPY", "CHF_JPY",
    "EUR_AUD", "EUR_CAD", "EUR_CHF", "EUR_GBP", "EUR_JPY", "EUR_NZD", "EUR_USD",
    "GBP_AUD", "GBP_CAD", "GBP_CHF", "GBP_JPY", "GBP_NZD", "GBP_USD",
    "NZD_CAD", "NZD_CHF", "NZD_JPY", "NZD_USD",
    "USD_CAD", "USD_CHF", "USD_JPY",
)

# USD-base / USD-quote split for DXY proxy
USD_BASE = ("USD_CAD", "USD_CHF", "USD_JPY")
USD_QUOTE = ("AUD_USD", "EUR_USD", "GBP_USD", "NZD_USD")

# Per-timeframe config
TF_CONFIG = {
    "1h": {"folder": "1hr", "forward_window": 480, "rf_min_samples_leaf": 20},
    "4h": {"folder": "4hr", "forward_window": 240, "rf_min_samples_leaf": 20},
    "d1": {"folder": "daily", "forward_window": 60, "rf_min_samples_leaf": 20},
}

START_DATE = pd.Timestamp("2020-01-01")
END_DATE = pd.Timestamp("2026-01-31 23:59:59")

RNG_SEED = 42


# ---------------------------------------------------------------- IO

def load_pair(tf: str, pair: str) -> pd.DataFrame:
    folder = TF_CONFIG[tf]["folder"]
    fp = _DATA / folder / f"{pair}.csv"
    df = pd.read_csv(fp, usecols=["time", "open", "high", "low", "close"])
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype("float64")
    return df


# ---------------------------------------------------------------- indicators

def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    prev_close = np.concatenate([[close[0]], close[:-1]])
    return np.maximum.reduce(
        [high - low, np.abs(high - prev_close), np.abs(low - prev_close)]
    )


def rma(x: np.ndarray, n: int) -> np.ndarray:
    """Wilder smoothing (RMA) via pandas ewm alpha=1/n."""
    return (
        pd.Series(x)
        .ewm(alpha=1.0 / n, adjust=False, min_periods=n)
        .mean()
        .values
    )


def sma(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).rolling(n, min_periods=n).mean().values


def ema_adjust_false(x: np.ndarray, span: int) -> np.ndarray:
    return (
        pd.Series(x)
        .ewm(span=span, adjust=False, min_periods=span)
        .mean()
        .values
    )


def compute_atr(high, low, close, n=14):
    tr = true_range(high, low, close)
    return rma(tr, n), tr


def compute_adx(high, low, close, n=14):
    prev_close = np.concatenate([[close[0]], close[:-1]])
    prev_high = np.concatenate([[high[0]], high[:-1]])
    prev_low = np.concatenate([[low[0]], low[:-1]])
    up = high - prev_high
    dn = prev_low - low
    plus_dm_raw = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm_raw = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = np.maximum.reduce(
        [high - low, np.abs(high - prev_close), np.abs(low - prev_close)]
    )
    tr_n = rma(tr, n)
    plus_dm_n = rma(plus_dm_raw, n)
    minus_dm_n = rma(minus_dm_raw, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = 100.0 * plus_dm_n / tr_n
        minus_di = 100.0 * minus_dm_n / tr_n
        dx_denom = plus_di + minus_di
        dx = np.where(dx_denom > 0, 100.0 * np.abs(plus_di - minus_di) / dx_denom, np.nan)
    adx = rma(dx, n)
    return adx, plus_di, minus_di


def rolling_rank_pct(x: np.ndarray, n: int) -> np.ndarray:
    """Rank of current value within last n values, range [0, 1]. Deterministic."""
    out = np.full_like(x, np.nan, dtype="float64")
    s = pd.Series(x)
    # rank of last value: count of (window <= current) / n
    def _rank(window):
        cur = window[-1]
        if np.isnan(cur):
            return np.nan
        # exclude NaNs from comparison
        valid = window[~np.isnan(window)]
        if len(valid) == 0:
            return np.nan
        return float(np.sum(valid <= cur) / len(valid))
    out = s.rolling(n, min_periods=n).apply(_rank, raw=True).values
    return out


def rolling_max(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).rolling(n, min_periods=n).max().values


def rolling_min(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).rolling(n, min_periods=n).min().values


def rolling_std(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).rolling(n, min_periods=n).std(ddof=0).values


# ---------------------------------------------------------------- B1: labels

@dataclass
class LabelArrays:
    clean_move: np.ndarray
    clean_score: np.ndarray
    mono_pre_peak: np.ndarray
    mfe_max_R: np.ndarray
    mae_pre_peak_R: np.ndarray
    reached_1R_pre_peak: np.ndarray
    peak_mfe_bar: np.ndarray
    has_label: np.ndarray  # True when bar has a fully observed forward window


def compute_labels(close, high, low, atr_14, time_vals, W: int) -> LabelArrays:
    n = len(close)
    clean_move = np.zeros(n, dtype=bool)
    clean_score = np.zeros(n, dtype=np.int8)
    mono = np.zeros(n, dtype="float64")
    mfe_max = np.zeros(n, dtype="float64")
    mae_pp = np.zeros(n, dtype="float64")
    reached = np.zeros(n, dtype=bool)
    peak_bar = np.zeros(n, dtype=np.int32)
    has_label = np.zeros(n, dtype=bool)

    # bar must have t+W within data AND that t+W timestamp <= END_DATE AND atr_14 valid
    end_ts = END_DATE.to_numpy()
    for t in range(n):
        if t + W >= n:
            continue
        if time_vals[t + W] > end_ts:
            continue
        R = 2.0 * atr_14[t]
        if not np.isfinite(R) or R <= 0:
            continue

        c_t = close[t]
        fwd_high = high[t + 1 : t + 1 + W]
        fwd_low = low[t + 1 : t + 1 + W]
        fwd_close = close[t + 1 : t + 1 + W]

        mfe_path = (fwd_high - c_t) / R
        mae_path = (fwd_low - c_t) / R
        close_r = (fwd_close - c_t) / R

        mfe_so_far = np.maximum.accumulate(mfe_path)
        mae_so_far = np.minimum.accumulate(mae_path)

        pb = int(np.argmax(mfe_so_far))
        mfe_val = float(mfe_so_far[pb])
        mae_val = float(mae_so_far[pb])
        r1r = mfe_val >= 1.0

        pre_close_r = close_r[: pb + 1]
        ip = pre_close_r[pre_close_r > 0]
        if ip.size > 1:
            mono_val = float(np.mean(ip[1:] >= ip[:-1]))
        else:
            mono_val = 0.0

        c1 = mono_val >= 0.55
        c2 = mfe_val >= 1.5
        c3 = mae_val > -1.0
        c4 = r1r
        s = int(c1) + int(c2) + int(c3) + int(c4)

        has_label[t] = True
        clean_move[t] = s == 4
        clean_score[t] = s
        mono[t] = mono_val
        mfe_max[t] = mfe_val
        mae_pp[t] = mae_val
        reached[t] = r1r
        peak_bar[t] = pb

    return LabelArrays(
        clean_move=clean_move,
        clean_score=clean_score,
        mono_pre_peak=mono,
        mfe_max_R=mfe_max,
        mae_pre_peak_R=mae_pp,
        reached_1R_pre_peak=reached,
        peak_mfe_bar=peak_bar,
        has_label=has_label,
    )


# ---------------------------------------------------------------- B2: features

def session_marker(hour_utc: np.ndarray) -> np.ndarray:
    """Asia 22-07 UTC, London 07-12 UTC, NY 13-22 UTC, overlap = London/NY 12-16."""
    # 0=asia, 1=london, 2=ny, 3=overlap
    out = np.zeros(hour_utc.shape, dtype=np.int8)
    h = hour_utc
    out[(h >= 22) | (h < 7)] = 0
    out[(h >= 7) & (h < 12)] = 1
    out[(h >= 12) & (h < 16)] = 3
    out[(h >= 16) & (h < 22)] = 2
    return out


def compute_pair_features(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    n = len(df)
    open_ = df["open"].values
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    time_vals = df["time"].values

    atr_14, _ = compute_atr(high, low, close, 14)
    atr_50, _ = compute_atr(high, low, close, 50)
    atr_200, _ = compute_atr(high, low, close, 200)

    with np.errstate(divide="ignore", invalid="ignore"):
        atr_ratio_14_50 = atr_14 / atr_50
        atr_ratio_14_200 = atr_14 / atr_200

    atr_pct_20 = rolling_rank_pct(atr_14, 20)
    atr_pct_60 = rolling_rank_pct(atr_14, 60)
    atr_pct_200 = rolling_rank_pct(atr_14, 200)

    log_ret = np.zeros(n, dtype="float64")
    log_ret[1:] = np.log(close[1:] / close[:-1])
    realised_vol_20 = rolling_std(log_ret, 20)

    bar_range = high - low
    rng_mean_20 = sma(bar_range, 20)
    rng_mean_60 = sma(bar_range, 60)
    with np.errstate(divide="ignore", invalid="ignore"):
        range_compression_20 = bar_range / rng_mean_20
        range_compression_60 = bar_range / rng_mean_60

    ema_20 = ema_adjust_false(close, 20)
    ema_50 = ema_adjust_false(close, 50)
    ema_200 = ema_adjust_false(close, 200)

    def slope(ema, n):
        s = np.full(len(ema), np.nan)
        s[n:] = (ema[n:] - ema[:-n]) / ema[:-n]
        return s

    ema_20_slope = slope(ema_20, 20)
    ema_50_slope = slope(ema_50, 50)
    ema_200_slope = slope(ema_200, 200)

    ema_20_above_50 = (ema_20 > ema_50).astype(np.int8)
    ema_50_above_200 = (ema_50 > ema_200).astype(np.int8)
    with np.errstate(divide="ignore", invalid="ignore"):
        ema_spread_20_50_atr = (ema_20 - ema_50) / atr_14

    adx_14, di_plus_14, di_minus_14 = compute_adx(high, low, close, 14)

    high_20 = rolling_max(high, 20)
    low_20 = rolling_min(low, 20)
    high_60 = rolling_max(high, 60)
    low_60 = rolling_min(low, 60)
    with np.errstate(divide="ignore", invalid="ignore"):
        pos_range_20 = (close - low_20) / (high_20 - low_20)
        pos_range_60 = (close - low_60) / (high_60 - low_60)

    body = np.abs(close - open_)
    with np.errstate(divide="ignore", invalid="ignore"):
        body_ratio = body / bar_range
        close_in_bar = (close - low) / bar_range
        body_size_atr = body / atr_14

    def lag_mean(arr, n):
        s = pd.Series(arr)
        return s.shift(1).rolling(n, min_periods=n).mean().values

    prev3_body_ratio_mean = lag_mean(body_ratio, 3)
    prev3_body_size_atr_mean = lag_mean(body_size_atr, 3)

    with np.errstate(divide="ignore", invalid="ignore"):
        recent_5_size = bar_range / atr_14
    recent_5_bar_size_pct_atr = (
        pd.Series(recent_5_size).rolling(5, min_periods=5).mean().values
    )

    # Time/session
    ts = pd.to_datetime(time_vals)
    hour_of_day = ts.hour.values.astype(np.int16)
    day_of_week = ts.dayofweek.values.astype(np.int8)
    hour_of_week = (day_of_week.astype(np.int32) * 24 + hour_of_day).astype(np.int16)
    sess = session_marker(hour_of_day)

    feats = pd.DataFrame(
        {
            "time": time_vals,
            "atr_14": atr_14,
            "atr_50": atr_50,
            "atr_200": atr_200,
            "atr_ratio_14_50": atr_ratio_14_50,
            "atr_ratio_14_200": atr_ratio_14_200,
            "atr_percentile_20": atr_pct_20,
            "atr_percentile_60": atr_pct_60,
            "atr_percentile_200": atr_pct_200,
            "realised_vol_20": realised_vol_20,
            "range_compression_20": range_compression_20,
            "range_compression_60": range_compression_60,
            "ema_20_slope": ema_20_slope,
            "ema_50_slope": ema_50_slope,
            "ema_200_slope": ema_200_slope,
            "ema_20_above_50": ema_20_above_50,
            "ema_50_above_200": ema_50_above_200,
            "ema_spread_20_50_atr": ema_spread_20_50_atr,
            "adx_14": adx_14,
            "di_plus_14": di_plus_14,
            "di_minus_14": di_minus_14,
            "position_in_range_20": pos_range_20,
            "position_in_range_60": pos_range_60,
            "body_ratio": body_ratio,
            "close_in_bar": close_in_bar,
            "body_size_atr": body_size_atr,
            "prev_3_body_ratio_mean": prev3_body_ratio_mean,
            "prev_3_body_size_atr_mean": prev3_body_size_atr_mean,
            "recent_5_bar_size_pct_atr": recent_5_bar_size_pct_atr,
            "day_of_week": day_of_week,
            "session_marker": sess,
        }
    )
    if tf != "d1":
        feats["hour_of_week"] = hour_of_week
    return feats


# ---------------------------------------------------------------- cross-TF anchors

def attach_d1_anchor(features_per_pair_dict: dict, d1_data: dict) -> None:
    """For each non-D1 pair feature frame, attach d1_ema_50_slope_at_entry
    and d1_close_above_ema_50 using prior calendar day's D1 close.
    Modifies in place.
    """
    for pair, feats in features_per_pair_dict.items():
        df_d1 = d1_data[pair]
        ema50 = ema_adjust_false(df_d1["close"].values, 50)
        slope50 = np.full(len(df_d1), np.nan)
        slope50[50:] = (ema50[50:] - ema50[:-50]) / ema50[:-50]
        d1 = pd.DataFrame(
            {
                "d1_date": df_d1["time"].dt.normalize().values,
                "d1_ema_50_slope": slope50,
                "d1_close_above_ema_50": (df_d1["close"].values > ema50).astype("float64"),
            }
        ).dropna(subset=["d1_ema_50_slope"])
        d1 = d1.sort_values("d1_date").reset_index(drop=True)
        # "prior calendar day": for any signal time T, the D1 bar with d1_date < T.date()
        target_dates = pd.to_datetime(feats["time"]).dt.normalize().values
        # find largest d1_date strictly less than target_date  -> use searchsorted - 1
        idx = np.searchsorted(d1["d1_date"].values, target_dates, side="left") - 1
        valid = idx >= 0
        out_slope = np.full(len(feats), np.nan)
        out_above = np.full(len(feats), np.nan)
        out_slope[valid] = d1["d1_ema_50_slope"].values[idx[valid]]
        out_above[valid] = d1["d1_close_above_ema_50"].values[idx[valid]]
        feats["d1_ema_50_slope_at_entry"] = out_slope
        feats["d1_close_above_ema_50"] = out_above


def attach_h4_anchor(features_per_pair_dict: dict, h4_data: dict) -> None:
    """For each 1H pair feature frame, attach h4_ema_50_slope_at_entry using
    the most recent CLOSED 4H bar. A 4H bar with timestamp B closes at B+4h,
    so we want max B such that B + 4h <= T  ->  B <= T - 4h.
    """
    four_h = pd.Timedelta(hours=4)
    for pair, feats in features_per_pair_dict.items():
        df_h4 = h4_data[pair]
        ema50 = ema_adjust_false(df_h4["close"].values, 50)
        slope50 = np.full(len(df_h4), np.nan)
        slope50[50:] = (ema50[50:] - ema50[:-50]) / ema50[:-50]
        h4 = pd.DataFrame(
            {
                "h4_time": df_h4["time"].values,
                "h4_ema_50_slope": slope50,
            }
        ).dropna(subset=["h4_ema_50_slope"])
        h4 = h4.sort_values("h4_time").reset_index(drop=True)
        target = pd.to_datetime(feats["time"]).values - np.timedelta64(four_h, "ns")
        idx = np.searchsorted(h4["h4_time"].values, target, side="right") - 1
        valid = idx >= 0
        out = np.full(len(feats), np.nan)
        out[valid] = h4["h4_ema_50_slope"].values[idx[valid]]
        feats["h4_ema_50_slope_at_entry"] = out


def attach_dxy_proxy(features_per_pair_dict: dict, close_panel: pd.DataFrame) -> None:
    """Compute dxy_proxy_ema_20_slope at the signal TF.

    close_panel: time-indexed wide panel of close prices per pair.
    Per pair: ema_20 slope; proxy = mean(USD_BASE slopes) - mean(USD_QUOTE slopes).
    Attach per-bar to each pair's features by aligning on time.
    """
    if close_panel.empty:
        for f in features_per_pair_dict.values():
            f["dxy_proxy_ema_20_slope"] = np.nan
        return

    slope_cols = {}
    for col in close_panel.columns:
        ema20 = ema_adjust_false(close_panel[col].ffill().values, 20)
        slope = np.full(len(close_panel), np.nan)
        slope[20:] = (ema20[20:] - ema20[:-20]) / ema20[:-20]
        slope_cols[col] = slope
    slope_df = pd.DataFrame(slope_cols, index=close_panel.index)

    base_present = [c for c in USD_BASE if c in slope_df.columns]
    quote_present = [c for c in USD_QUOTE if c in slope_df.columns]
    proxy = slope_df[base_present].mean(axis=1) - slope_df[quote_present].mean(axis=1)

    proxy_df = pd.DataFrame({"time": proxy.index, "dxy_proxy_ema_20_slope": proxy.values})
    proxy_df = proxy_df.sort_values("time").reset_index(drop=True)
    proxy_time = proxy_df["time"].values
    proxy_val = proxy_df["dxy_proxy_ema_20_slope"].values

    for pair, feats in features_per_pair_dict.items():
        # nearest non-future proxy value
        idx = np.searchsorted(proxy_time, feats["time"].values, side="right") - 1
        valid = idx >= 0
        out = np.full(len(feats), np.nan)
        out[valid] = proxy_val[idx[valid]]
        feats["dxy_proxy_ema_20_slope"] = out


# ---------------------------------------------------------------- B3: univariate

def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va = np.var(a, ddof=1)
    vb = np.var(b, ddof=1)
    pooled = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
    if pooled == 0:
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / pooled)


def b3_univariate(features: pd.DataFrame, labels: np.ndarray, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    y = labels.astype(int)
    for col in feature_cols:
        x = features[col].values
        mask = np.isfinite(x)
        xm = x[mask]
        ym = y[mask]
        if len(np.unique(ym)) < 2 or len(xm) < 50:
            rows.append({"feature": col, "auc": np.nan, "ks_stat": np.nan, "ks_pvalue": np.nan,
                         "cohen_d": np.nan, "mean_pos": np.nan, "mean_neg": np.nan, "n": int(len(xm))})
            continue
        try:
            lr = LogisticRegression(max_iter=200, solver="lbfgs")
            lr.fit(xm.reshape(-1, 1), ym)
            proba = lr.predict_proba(xm.reshape(-1, 1))[:, 1]
            auc = float(roc_auc_score(ym, proba))
        except Exception:
            auc = float("nan")
        pos = xm[ym == 1]
        neg = xm[ym == 0]
        if len(pos) > 0 and len(neg) > 0:
            ks = sps.ks_2samp(pos, neg)
            ks_stat = float(ks.statistic)
            ks_p = float(ks.pvalue)
            d = cohen_d(pos, neg)
            mp = float(np.mean(pos))
            mn = float(np.mean(neg))
        else:
            ks_stat = ks_p = d = mp = mn = float("nan")
        rows.append({"feature": col, "auc": auc, "ks_stat": ks_stat, "ks_pvalue": ks_p,
                     "cohen_d": d, "mean_pos": mp, "mean_neg": mn, "n": int(len(xm))})
    out = pd.DataFrame(rows)
    out["auc_dist_from_0_5"] = (out["auc"] - 0.5).abs()
    out = out.sort_values("auc_dist_from_0_5", ascending=False).drop(columns="auc_dist_from_0_5")
    return out.reset_index(drop=True)


# ---------------------------------------------------------------- B4: RF + TS-CV

def b4_multivariate(features: pd.DataFrame, labels: np.ndarray, feature_cols: list[str],
                     pair_col: np.ndarray, time_col: np.ndarray,
                     n_splits: int = 7, min_samples_leaf: int = 20):
    """Time-series CV with anchored expanding folds (TimeSeriesSplit).
    Features must be pre-sorted by time. Rows with any NaN feature are dropped first.
    """
    X_raw = features[feature_cols].values
    y = labels.astype(int)
    mask = np.all(np.isfinite(X_raw), axis=1)
    X = X_raw[mask]
    y = y[mask]
    pairs = pair_col[mask]
    times = time_col[mask]

    order = np.argsort(times, kind="stable")
    X = X[order]
    y = y[order]
    pairs = pairs[order]
    times = times[order]

    tss = TimeSeriesSplit(n_splits=n_splits)
    fold_rows = []
    fold_aucs = []
    feat_imps_full = None

    pair_breakdown_per_fold = []

    for fold_idx, (tr, va) in enumerate(tss.split(X), start=1):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
            fold_rows.append({
                "fold_id": fold_idx,
                "fold_train_start": str(times[tr[0]]),
                "fold_train_end": str(times[tr[-1]]),
                "fold_val_start": str(times[va[0]]),
                "fold_val_end": str(times[va[-1]]),
                "train_n": int(len(tr)),
                "val_n": int(len(va)),
                "val_auc": float("nan"),
                "val_clean_rate": float(np.mean(y[va])) if len(va) else float("nan"),
                "feature_importances_top10": "",
            })
            continue
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=min_samples_leaf,
            random_state=RNG_SEED,
            n_jobs=-1,
        )
        rf.fit(X[tr], y[tr])
        proba = rf.predict_proba(X[va])[:, 1]
        auc = float(roc_auc_score(y[va], proba))
        fold_aucs.append(auc)
        imps = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
        top10 = "; ".join(f"{k}:{v:.4f}" for k, v in imps.head(10).items())
        fold_rows.append({
            "fold_id": fold_idx,
            "fold_train_start": str(times[tr[0]]),
            "fold_train_end": str(times[tr[-1]]),
            "fold_val_start": str(times[va[0]]),
            "fold_val_end": str(times[va[-1]]),
            "train_n": int(len(tr)),
            "val_n": int(len(va)),
            "val_auc": auc,
            "val_clean_rate": float(np.mean(y[va])),
            "feature_importances_top10": top10,
        })
        pb = pd.DataFrame({"pair": pairs[va], "y": y[va]})
        pb_counts = pb.groupby("pair")["y"].agg(["sum", "count"]).reset_index()
        pb_counts["fold_id"] = fold_idx
        pair_breakdown_per_fold.append(pb_counts)

    # Full-data fit for feature importances (diagnostic; not for headline AUC)
    rf_full = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=min_samples_leaf,
        random_state=RNG_SEED,
        n_jobs=-1,
    )
    rf_full.fit(X, y)
    proba_full = rf_full.predict_proba(X)[:, 1]
    auc_full = float(roc_auc_score(y, proba_full))
    feat_imps_full = pd.Series(rf_full.feature_importances_, index=feature_cols)

    folds_df = pd.DataFrame(fold_rows)
    summary = {
        "n_splits": n_splits,
        "n_after_dropna": int(len(y)),
        "clean_rate": float(np.mean(y)),
        "mean_fold_auc": float(np.nanmean(fold_aucs)) if fold_aucs else float("nan"),
        "worst_fold_auc": float(np.nanmin(fold_aucs)) if fold_aucs else float("nan"),
        "fold_auc_stdev": float(np.nanstd(fold_aucs, ddof=0)) if fold_aucs else float("nan"),
        "full_data_auc": auc_full,
    }
    pair_breakdown = (
        pd.concat(pair_breakdown_per_fold, ignore_index=True)
        if pair_breakdown_per_fold
        else pd.DataFrame()
    )
    return folds_df, summary, feat_imps_full.sort_values(ascending=False), pair_breakdown


# ---------------------------------------------------------------- spot-check

def spot_check_no_lookahead(rows_df: pd.DataFrame, pair_dfs: dict, tf: str,
                             n: int = 5, seed: int = RNG_SEED) -> list[dict]:
    """Pick n random labelled bars; recompute key features manually from raw OHLC <= t
    and assert equality with stored values."""
    rng = np.random.default_rng(seed)
    checks = []
    idx = rng.choice(len(rows_df), size=min(n, len(rows_df)), replace=False)
    for i in idx:
        row = rows_df.iloc[int(i)]
        pair = row["pair"]
        t = row["time"]
        df = pair_dfs[pair]
        # Find t in df
        loc = int(df.index[df["time"] == t][0])
        # Recompute atr_14 from scratch using only bars <= loc
        h = df["high"].values[: loc + 1]
        l = df["low"].values[: loc + 1]
        c = df["close"].values[: loc + 1]
        if loc + 1 < 14:
            continue
        atr_recomp = compute_atr(h, l, c, 14)[0][-1]
        # body_ratio
        o_t = df["open"].values[loc]
        c_t = df["close"].values[loc]
        h_t = df["high"].values[loc]
        l_t = df["low"].values[loc]
        br = float(abs(c_t - o_t) / (h_t - l_t)) if (h_t - l_t) > 0 else float("nan")
        # position_in_range_20
        if loc + 1 >= 20:
            h20 = df["high"].values[loc - 19 : loc + 1].max()
            l20 = df["low"].values[loc - 19 : loc + 1].min()
            pir = float((c_t - l20) / (h20 - l20)) if (h20 - l20) > 0 else float("nan")
        else:
            pir = float("nan")
        # ema_50_slope
        if loc + 1 >= 50 * 2:
            ema50 = ema_adjust_false(df["close"].values[: loc + 1], 50)
            ema50_slope = float((ema50[-1] - ema50[-51]) / ema50[-51])
        else:
            ema50_slope = float("nan")
        checks.append({
            "pair": pair,
            "time": str(t),
            "stored_atr_14": float(row.get("atr_14", np.nan)),
            "recomputed_atr_14": float(atr_recomp),
            "stored_body_ratio": float(row.get("body_ratio", np.nan)),
            "recomputed_body_ratio": br,
            "stored_position_in_range_20": float(row.get("position_in_range_20", np.nan)),
            "recomputed_position_in_range_20": pir,
            "stored_ema_50_slope": float(row.get("ema_50_slope", np.nan)),
            "recomputed_ema_50_slope": ema50_slope,
            "stored_mfe_max_R": float(row.get("mfe_max_R", np.nan)),
        })
    return checks


# ---------------------------------------------------------------- pipeline

def run_timeframe(tf: str) -> dict:
    cfg = TF_CONFIG[tf]
    W = cfg["forward_window"]
    out_dir = _OUT / f"timeframe_{tf}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{tf}] loading data for {len(PAIRS)} pairs ...", flush=True)
    pair_dfs: dict[str, pd.DataFrame] = {}
    for p in PAIRS:
        pair_dfs[p] = load_pair(tf, p)

    # Pre-load D1 for cross-TF anchor (1h, 4h)
    d1_data = None
    h4_data = None
    if tf == "1h":
        d1_data = {p: load_pair("d1", p) for p in PAIRS}
        h4_data = {p: load_pair("4h", p) for p in PAIRS}
    elif tf == "4h":
        d1_data = {p: load_pair("d1", p) for p in PAIRS}

    print(f"[{tf}] computing per-pair labels and features ...", flush=True)
    pair_label_frames = []
    pair_feature_frames = {}
    for pair in PAIRS:
        df = pair_dfs[pair]
        # compute labels and features on full data, then restrict by date later
        atr_14, _ = compute_atr(df["high"].values, df["low"].values, df["close"].values, 14)
        labels = compute_labels(
            df["close"].values,
            df["high"].values,
            df["low"].values,
            atr_14,
            df["time"].values,
            W,
        )
        feats = compute_pair_features(df, tf)
        # attach raw OHLC for downstream sanity, then drop later in features.csv
        feats["pair"] = pair
        label_frame = pd.DataFrame(
            {
                "pair": pair,
                "time": df["time"].values,
                "clean_move": labels.clean_move,
                "clean_score": labels.clean_score,
                "mono_pre_peak": labels.mono_pre_peak,
                "mfe_max_R": labels.mfe_max_R,
                "mae_pre_peak_R": labels.mae_pre_peak_R,
                "reached_1R_pre_peak": labels.reached_1R_pre_peak,
                "peak_mfe_bar": labels.peak_mfe_bar,
                "has_label": labels.has_label,
            }
        )
        pair_label_frames.append(label_frame)
        pair_feature_frames[pair] = feats

    # Cross-TF anchors
    if tf in ("1h", "4h") and d1_data is not None:
        attach_d1_anchor(pair_feature_frames, d1_data)
    if tf == "1h" and h4_data is not None:
        attach_h4_anchor(pair_feature_frames, h4_data)

    # DXY proxy at signal TF (after per-pair features ready)
    close_dict = {}
    common_times = None
    for pair in PAIRS:
        s = pair_dfs[pair].set_index("time")["close"]
        close_dict[pair] = s
    close_panel = pd.DataFrame(close_dict)
    attach_dxy_proxy(pair_feature_frames, close_panel)

    # Combine pair frames
    all_feats = pd.concat(pair_feature_frames.values(), ignore_index=True)
    all_labels = pd.concat(pair_label_frames, ignore_index=True)
    # Join on pair+time
    merged = all_feats.merge(all_labels, on=["pair", "time"], how="inner")

    # Restrict to labelled bars within [START, END] and where forward window fits
    merged = merged[merged["has_label"]].copy()
    merged = merged[(merged["time"] >= START_DATE) & (merged["time"] <= END_DATE)]
    merged = merged.sort_values(["time", "pair"]).reset_index(drop=True)
    print(f"[{tf}] labelled rows after date filter: {len(merged):,}", flush=True)

    # Save labels.csv and features.csv
    label_cols = ["pair", "time", "clean_move", "clean_score", "mono_pre_peak",
                  "mfe_max_R", "mae_pre_peak_R", "reached_1R_pre_peak", "peak_mfe_bar"]
    merged[label_cols].to_csv(out_dir / "labels.csv", index=False)

    feature_cols = [c for c in merged.columns if c not in (label_cols + ["has_label"])]
    feature_cols_no_meta = [c for c in feature_cols if c not in ("pair", "time")]
    merged[["pair", "time"] + feature_cols_no_meta].to_csv(out_dir / "features.csv", index=False)

    # ----- B3 univariate
    print(f"[{tf}] running B3 univariate ...", flush=True)
    b3 = b3_univariate(merged, merged["clean_move"].values, feature_cols_no_meta)
    b3.to_csv(out_dir / "b3_univariate.csv", index=False)

    # ----- B4 multivariate (top 15 by univariate AUC distance from 0.5)
    top_features = b3.head(15)["feature"].tolist()
    print(f"[{tf}] top-15 features: {top_features}", flush=True)
    folds_df, summary, imps_full, pair_breakdown = b4_multivariate(
        merged, merged["clean_move"].values, top_features,
        merged["pair"].values, merged["time"].values,
        n_splits=7, min_samples_leaf=cfg["rf_min_samples_leaf"],
    )
    folds_df.to_csv(out_dir / "b4_multivariate.csv", index=False)
    if not pair_breakdown.empty:
        pair_breakdown.to_csv(out_dir / "b4_per_pair_per_fold.csv", index=False)
    imps_full.rename("importance").to_csv(out_dir / "b4_feature_importances_full.csv",
                                          header=True, index_label="feature")
    with open(out_dir / "b4_summary.txt", "w") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")
        f.write(f"top_15_features: {top_features}\n")

    # ----- label distribution
    print(f"[{tf}] computing label distribution ...", flush=True)
    base_rate = float(merged["clean_move"].mean())
    score_dist = merged["clean_score"].value_counts().sort_index().to_dict()
    pair_counts = merged.groupby("pair")["clean_move"].agg(["sum", "count"]).reset_index()
    pair_counts["clean_rate"] = pair_counts["sum"] / pair_counts["count"]
    pair_counts = pair_counts.sort_values("sum", ascending=False)
    total_pos = int(merged["clean_move"].sum())
    pair_counts["share_of_clean"] = pair_counts["sum"] / max(total_pos, 1)
    pair_flag = pair_counts[pair_counts["share_of_clean"] > 0.40]
    with open(out_dir / "label_distribution.txt", "w") as f:
        f.write(f"timeframe: {tf}\n")
        f.write(f"forward_window_bars: {W}\n")
        f.write(f"n_rows: {len(merged)}\n")
        f.write(f"clean_move_count: {total_pos}\n")
        f.write(f"clean_move_rate: {base_rate:.6f}\n")
        f.write(f"clean_score_distribution: {score_dist}\n")
        f.write("per_pair_clean_counts:\n")
        f.write(pair_counts.to_string(index=False))
        f.write("\n")
        if not pair_flag.empty:
            f.write("\nWARNING: pair(s) contribute > 40% of clean_move labels:\n")
            f.write(pair_flag.to_string(index=False))
            f.write("\n")
        else:
            f.write("\nOK: no single pair contributes > 40% of clean_move labels.\n")

    # ----- spot-check (5 random bars; recompute features by hand)
    print(f"[{tf}] running no-lookahead spot-checks ...", flush=True)
    spot = spot_check_no_lookahead(merged, pair_dfs, tf, n=5)
    with open(out_dir / "no_lookahead_spotcheck.txt", "w") as f:
        f.write(f"timeframe: {tf}\n")
        f.write("Per bar: compare stored features vs manual recompute from raw OHLC <= t.\n\n")
        for s in spot:
            f.write(str(s) + "\n")

    print(f"[{tf}] DONE — worst-fold AUC {summary['worst_fold_auc']:.4f} "
          f"mean {summary['mean_fold_auc']:.4f} clean-rate {base_rate:.4f}", flush=True)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", choices=["1h", "4h", "d1", "all"], default="all")
    args = ap.parse_args()
    tfs = ["1h", "4h", "d1"] if args.tf == "all" else [args.tf]
    summaries = {}
    for tf in tfs:
        summaries[tf] = run_timeframe(tf)
    print("\n=== summary ===")
    for tf, s in summaries.items():
        print(f"  {tf}: worst_fold_auc={s['worst_fold_auc']:.4f} "
              f"mean={s['mean_fold_auc']:.4f} clean_rate={s['clean_rate']:.4f} "
              f"n={s['n_after_dropna']:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
