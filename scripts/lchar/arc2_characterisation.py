"""Arc 2 descriptive characterisation pipeline (research-mode, non-WFO).

Produces the v1.1 standard characterisation deliverable on Arc 2's trade-set
per L6_0_METHODOLOGY_LOCK.md §14.4. Disposition is descriptive only — no gate,
no filter derivation, no system construction. L6.0 §9 no-filter-rescue applies.

Reuses the Arc 2 signal computation from
`core.signals.l4_mtf_alignment_2_down_mixed_kijun` verbatim so the signal-firing
bar set is byte-identical to the Arc 2 WFO over the OOS window. Trade outcomes
are read from `results/l6/arc2/trades_all.csv` (already produced by the WFO);
dropped signals (concurrent_open_position etc.) are NOT features for the trade
population but are tracked for cross-pair density.

Per Arc 2 spec, the mtf_alignment triple `(s_1h, s_4h_mr, s_d1_mr)` is
constant `(-1, +1, -1)` by construction within the 2_down_mixed mask, so the
characterisation pivots on OTHER features (session, pre_momentum, MTF kijun
distances at non-state-defining lags, ATR regime, etc.).

Public entrypoint: `run_arc2_characterisation(config_path)`.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.signals.l4_mtf_alignment_2_down_mixed_kijun import (  # noqa: E402
    _attach_kijun_sign,
    _compute_kijun,
    _load_pair_tf,
    _mtf_alignment_2_down_mixed_kijun,
    _wilder_atr_1h,
    KIJUN_PERIOD,
    TIME_COL,
)

PAIRS_DEFAULT: Tuple[str, ...] = (
    "AUD_CAD", "AUD_CHF", "AUD_JPY", "AUD_NZD", "AUD_USD", "CAD_CHF", "CAD_JPY", "CHF_JPY",
    "EUR_AUD", "EUR_CAD", "EUR_CHF", "EUR_GBP", "EUR_JPY", "EUR_NZD", "EUR_USD", "GBP_AUD",
    "GBP_CAD", "GBP_CHF", "GBP_JPY", "GBP_NZD", "GBP_USD", "NZD_CAD", "NZD_CHF", "NZD_JPY",
    "NZD_USD", "USD_CAD", "USD_CHF", "USD_JPY",
)

# Forward horizons in 1H bars (descriptive only — does not modify the gated h=120 trade outcome).
FORWARD_HORIZONS: Tuple[int, ...] = (6, 24, 72, 120, 240)

# Session windows (UTC hours, inclusive). Identical to Arc 1's character config.
SESSIONS: Dict[str, Tuple[int, int]] = {
    "sydney": (22, 0),
    "asia": (1, 7),
    "london": (8, 12),
    "london_ny_overlap": (13, 16),
    "ny": (17, 21),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def _wilder_atr_series(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    """Wilder ATR(period) on df (open/high/low/close); same recurrence as Arc 2 module."""
    high = df["high"].astype(float).values
    low = df["low"].astype(float).values
    close = df["close"].astype(float).values
    n = len(df)
    if n == 0:
        return np.array([], dtype=float)
    prev_close = np.empty(n, dtype=float)
    prev_close[0] = np.nan
    prev_close[1:] = close[:-1]
    tr = np.maximum.reduce(
        [high - low, np.abs(high - prev_close), np.abs(low - prev_close)]
    )
    tr[0] = high[0] - low[0]
    atr = np.full(n, np.nan, dtype=float)
    if n < period:
        return atr
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _session_of_hour(h: int) -> str:
    for name, (lo, hi) in SESSIONS.items():
        if lo <= hi:
            if lo <= h <= hi:
                return name
        else:
            # Wraps midnight (sydney: 22-0)
            if h >= lo or h <= hi:
                return name
    return "other"


# ---------------------------------------------------------------------------
# Per-pair feature extraction at signal bars
# ---------------------------------------------------------------------------


def _extract_pair_features(
    pair: str,
    df_1h_raw: pd.DataFrame,
    df_4h_raw: pd.DataFrame,
    df_d1_raw: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the Arc 2 signal mask and the full per-bar 1H feature frame for one pair.

    Returns:
        df_h1_feat — 1H feature dataframe with mask_2dmk column included.
        s_1h, s_4h_mr, s_d1_mr — sign arrays aligned to df_h1_feat.index.
    """
    df_1h = _attach_kijun_sign(df_1h_raw)
    df_4h = _attach_kijun_sign(df_4h_raw)
    df_d1 = _attach_kijun_sign(df_d1_raw)

    mask, s_1h, s_4h_mr, s_d1_mr = _mtf_alignment_2_down_mixed_kijun(
        df_1h, df_4h, df_d1, pair=pair
    )
    atr_1h = _wilder_atr_1h(df_1h)
    n = len(df_1h)

    df = df_1h.copy()
    df["mask_2dmk"] = mask
    df["atr_1h_wilder_at_n"] = atr_1h

    # Pre-signal momentum (cum log-return), bars strictly before N.
    close = df["close"].astype(float).to_numpy()
    prev_close = np.empty(n, dtype=float)
    prev_close[0] = np.nan
    prev_close[1:] = close[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret_1h = np.log(close / prev_close)
    log_ret_s = pd.Series(log_ret_1h)
    df["log_ret_1h_at_n"] = log_ret_1h
    df["cum_logret_1h_6"] = log_ret_s.rolling(6, min_periods=6).sum().shift(1).to_numpy()
    df["cum_logret_1h_24"] = log_ret_s.rolling(24, min_periods=24).sum().shift(1).to_numpy()
    df["cum_logret_1h_120"] = log_ret_s.rolling(120, min_periods=120).sum().shift(1).to_numpy()

    # Bar properties at N (signal bar): bar_size in ATR, bar_body in ATR, close position.
    open_ = df["open"].astype(float).to_numpy()
    high = df["high"].astype(float).to_numpy()
    low = df["low"].astype(float).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        df["bar_size_atr"] = (high - low) / atr_1h
        df["bar_body_atr"] = np.abs(close - open_) / atr_1h
        bar_height = high - low
        df["close_position_in_bar"] = np.where(bar_height > 0, (close - low) / bar_height, np.nan)

    # ATR regime 1H: atr / rolling-50 median atr.
    atr_med50 = pd.Series(atr_1h).rolling(50, min_periods=50).median().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        df["atr_1h_regime"] = atr_1h / atr_med50

    # 1H kijun distance (state-defining: by construction sign is -1; magnitude varies).
    kijun_1h = df["kijun"].to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        df["dist_to_kijun_1h_atr"] = (close - kijun_1h) / atr_1h

    # 1H EMA(20)/(50) distances (ATR-normalised, signed).
    ema20_1h = _ema(df["close"], 20).to_numpy()
    ema50_1h = _ema(df["close"], 50).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        df["dist_to_ema20_1h_atr"] = (close - ema20_1h) / atr_1h
        df["dist_to_ema50_1h_atr"] = (close - ema50_1h) / atr_1h

    # 4H context — features at the *most-recently-completed* 4H bar (state-consistent lag).
    df_4h_feat = df_4h.copy()
    atr_4h = _wilder_atr_series(df_4h_feat, period=14)
    df_4h_feat["atr_4h"] = atr_4h
    df_4h_feat["atr_4h_regime"] = (
        atr_4h / pd.Series(atr_4h).rolling(50, min_periods=50).median().to_numpy()
    )
    ema20_4h = _ema(df_4h_feat["close"], 20).to_numpy()
    close_4h = df_4h_feat["close"].astype(float).to_numpy()
    kijun_4h_arr = df_4h_feat["kijun"].to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        df_4h_feat["dist_to_kijun_4h_atr"] = (close_4h - kijun_4h_arr) / atr_4h
        df_4h_feat["dist_to_ema20_4h_atr"] = (close_4h - ema20_4h) / atr_4h

    # D1 context — features at the *most-recently-completed* D1 bar.
    df_d1_feat = df_d1.copy()
    atr_d1 = _wilder_atr_series(df_d1_feat, period=14)
    df_d1_feat["atr_d1"] = atr_d1
    df_d1_feat["atr_d1_regime"] = (
        atr_d1 / pd.Series(atr_d1).rolling(50, min_periods=50).median().to_numpy()
    )
    ema20_d1 = _ema(df_d1_feat["close"], 20).to_numpy()
    close_d1 = df_d1_feat["close"].astype(float).to_numpy()
    kijun_d1_arr = df_d1_feat["kijun"].to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        df_d1_feat["dist_to_kijun_d1_atr"] = (close_d1 - kijun_d1_arr) / atr_d1
        df_d1_feat["dist_to_ema20_d1_atr"] = (close_d1 - ema20_d1) / atr_d1

    # Asof-merge 4H + D1 features at the most-recently-completed bar.
    # For 4H: lookup the 4H bar whose floor(t_1h, "4h") - 1 index applies.
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    df_4h_feat = df_4h_feat.sort_values(TIME_COL).reset_index(drop=True)
    df_d1_feat = df_d1_feat.sort_values(TIME_COL).reset_index(drop=True)

    floor_4h = df[TIME_COL].dt.floor("4h")
    idx_4h_by_time = pd.Series(
        np.arange(len(df_4h_feat), dtype=np.int64), index=df_4h_feat[TIME_COL]
    )
    contain_4h = floor_4h.map(idx_4h_by_time).to_numpy(dtype=float)
    val_4h = ~np.isnan(contain_4h)
    mr4 = np.where(val_4h, contain_4h, 0).astype(np.int64) - 1
    in_range_4h = val_4h & (mr4 >= 0)

    floor_d1 = df[TIME_COL].dt.normalize()
    idx_d1_by_time = pd.Series(
        np.arange(len(df_d1_feat), dtype=np.int64), index=df_d1_feat[TIME_COL]
    )
    contain_d1 = floor_d1.map(idx_d1_by_time).to_numpy(dtype=float)
    val_d1 = ~np.isnan(contain_d1)
    mrd = np.where(val_d1, contain_d1, 0).astype(np.int64) - 1
    in_range_d1 = val_d1 & (mrd >= 0)

    def _gather(df_high: pd.DataFrame, col: str, in_range: np.ndarray, mr_idx: np.ndarray) -> np.ndarray:
        arr = df_high[col].to_numpy(dtype=float)
        out = np.full(n, np.nan, dtype=float)
        out[in_range] = arr[mr_idx[in_range]]
        return out

    df["atr_4h_at_mr"] = _gather(df_4h_feat, "atr_4h", in_range_4h, mr4)
    df["atr_4h_regime_at_mr"] = _gather(df_4h_feat, "atr_4h_regime", in_range_4h, mr4)
    df["dist_to_kijun_4h_at_mr_atr"] = _gather(df_4h_feat, "dist_to_kijun_4h_atr", in_range_4h, mr4)
    df["dist_to_ema20_4h_at_mr_atr"] = _gather(df_4h_feat, "dist_to_ema20_4h_atr", in_range_4h, mr4)
    df["atr_d1_at_mr"] = _gather(df_d1_feat, "atr_d1", in_range_d1, mrd)
    df["atr_d1_regime_at_mr"] = _gather(df_d1_feat, "atr_d1_regime", in_range_d1, mrd)
    df["dist_to_kijun_d1_at_mr_atr"] = _gather(df_d1_feat, "dist_to_kijun_d1_atr", in_range_d1, mrd)
    df["dist_to_ema20_d1_at_mr_atr"] = _gather(df_d1_feat, "dist_to_ema20_d1_atr", in_range_d1, mrd)
    # Lag-audit timestamps (the actual 4H / D1 timestamps used per row).
    ts_4h_arr = df_4h_feat[TIME_COL].to_numpy()
    ts_d1_arr = df_d1_feat[TIME_COL].to_numpy()
    ts_4h_used = np.full(n, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    ts_d1_used = np.full(n, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    ts_4h_used[in_range_4h] = ts_4h_arr[mr4[in_range_4h]]
    ts_d1_used[in_range_d1] = ts_d1_arr[mrd[in_range_d1]]
    df["ts_4h_used"] = ts_4h_used
    df["ts_d1_used"] = ts_d1_used

    # Forward horizon log-returns (descriptive only, signal-bar close to N+h close).
    for h in FORWARD_HORIZONS:
        fwd = np.full(n, np.nan, dtype=float)
        if n > h:
            fwd[: n - h] = np.log(close[h:] / close[: n - h])
        df[f"fwd_logret_h{h}"] = fwd

    # Time / session features at signal bar.
    df["hour_utc"] = df[TIME_COL].dt.hour
    df["dow"] = df[TIME_COL].dt.dayofweek
    df["session"] = df["hour_utc"].apply(_session_of_hour)

    return df, s_1h, s_4h_mr, s_d1_mr


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_arc2_characterisation(
    *,
    output_dir: Path,
    arc2_trades_path: Path,
    pairs: Tuple[str, ...] = PAIRS_DEFAULT,
    signal_start: pd.Timestamp = pd.Timestamp("2020-10-01"),
    signal_end: pd.Timestamp = pd.Timestamp("2026-01-01"),
) -> Dict[str, str]:
    """Run the Arc 2 characterisation pipeline. Returns sha256 manifest."""
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Arc 2 trade set (already produced by the WFO).
    trades_df = pd.read_csv(arc2_trades_path)
    trades_df["signal_bar_ts"] = pd.to_datetime(trades_df["signal_bar_ts"])
    trades_df["entry_bar_ts"] = pd.to_datetime(trades_df["entry_bar_ts"])
    trades_df["exit_bar_ts"] = pd.to_datetime(trades_df["exit_bar_ts"])

    # 2. Load Arc 2 fold results for per-fold disposition tagging.
    fold_results_path = arc2_trades_path.parent / "wfo_fold_results.csv"
    fold_results_df = pd.read_csv(fold_results_path)
    fold_results_df["oos_start"] = pd.to_datetime(fold_results_df["oos_start"])
    fold_results_df["oos_end"] = pd.to_datetime(fold_results_df["oos_end"])
    fold_disp_map: Dict[int, str] = {
        int(r["fold_id"]): str(r["gate_disposition"]) for _, r in fold_results_df.iterrows()
    }

    # 3. Per-pair: compute the signal mask + feature frame and gather features at signal bars.
    per_pair_features: Dict[str, pd.DataFrame] = {}
    signal_pair_ts_rows: List[Dict[str, Any]] = []

    for pair in pairs:
        df_1h = _load_pair_tf(pair, "1hr")
        df_4h = _load_pair_tf(pair, "4hr")
        df_d1 = _load_pair_tf(pair, "daily")
        df_feat, s_1h, s_4h_mr, s_d1_mr = _extract_pair_features(pair, df_1h, df_4h, df_d1)

        mask = df_feat["mask_2dmk"].to_numpy()
        # Restrict to signal window.
        in_window = (df_feat[TIME_COL] >= signal_start) & (df_feat[TIME_COL] < signal_end)
        active_mask = mask & in_window.to_numpy()
        if active_mask.any():
            sub = df_feat.loc[active_mask].copy()
            sub["pair"] = pair
            sub["s_1h"] = s_1h[active_mask]
            sub["s_4h_mr"] = s_4h_mr[active_mask]
            sub["s_d1_mr"] = s_d1_mr[active_mask]
            per_pair_features[pair] = sub

    if not per_pair_features:
        raise RuntimeError("Arc 2 characterisation: zero signal-firing bars in signal window")

    all_signals = pd.concat(per_pair_features.values(), axis=0, ignore_index=False).sort_values(
        [TIME_COL, "pair"]
    ).reset_index(drop=True)

    # Cross-pair density at signal bar (same-bar concurrent signals on other pairs).
    bar_counts = all_signals.groupby(TIME_COL).size().to_dict()
    all_signals["concurrent_signals_same_bar"] = all_signals[TIME_COL].map(bar_counts) - 1

    # Pre-momentum label (down/flat/up at signal-1H bar based on cum_logret_1h_24).
    PRE_MOM_UP_MIN = 0.001
    PRE_MOM_DOWN_MAX = -0.001

    def _pre_mom_label(v: float) -> str:
        if not isinstance(v, float) or not math.isfinite(v):
            return "flat"
        if v >= PRE_MOM_UP_MIN:
            return "up"
        if v <= PRE_MOM_DOWN_MAX:
            return "down"
        return "flat"

    all_signals["pre_momentum_label"] = all_signals["cum_logret_1h_24"].apply(_pre_mom_label)

    # ATR-regime bins: low (<0.8), mid (0.8-1.2), high (>1.2).
    def _atr_regime_bin(v: float) -> str:
        if not isinstance(v, float) or not math.isfinite(v):
            return "nan"
        if v < 0.8:
            return "low"
        if v > 1.2:
            return "high"
        return "mid"

    all_signals["atr_1h_regime_bin"] = all_signals["atr_1h_regime"].apply(_atr_regime_bin)
    all_signals["atr_4h_regime_bin"] = all_signals["atr_4h_regime_at_mr"].apply(_atr_regime_bin)
    all_signals["atr_d1_regime_bin"] = all_signals["atr_d1_regime_at_mr"].apply(_atr_regime_bin)

    # 4. Tag each signal with fold_id and fold disposition (only for taken signals
    # within fold OOS windows). Use trades_df to attribute fold_id deterministically.
    # Build a (pair, signal_bar_ts) -> trade row mapping.
    trade_by_keys = {
        (str(r["pair"]), pd.Timestamp(r["signal_bar_ts"]).isoformat()): r
        for _, r in trades_df.iterrows()
    }

    fold_id_arr: List[Any] = []
    taken_arr: List[bool] = []
    exit_reason_arr: List[str] = []
    R_arr: List[float] = []
    mae_R_arr: List[float] = []
    mfe_R_arr: List[float] = []
    held_bars_arr: List[Any] = []
    spread_floored_arr: List[Any] = []
    spread_pips_entry_arr: List[Any] = []
    spread_pips_exit_arr: List[Any] = []
    fold_disp_arr: List[str] = []

    for _, row in all_signals.iterrows():
        key = (str(row["pair"]), pd.Timestamp(row[TIME_COL]).isoformat())
        if key in trade_by_keys:
            tr = trade_by_keys[key]
            fid = int(tr["fold_id"])
            fold_id_arr.append(fid)
            taken_arr.append(True)
            exit_reason_arr.append(str(tr["exit_reason"]))
            R_arr.append(float(tr["R"]))
            mae_R_arr.append(float(tr["mae_R"]))
            mfe_R_arr.append(float(tr["mfe_R"]))
            held_bars_arr.append(int(tr["held_bars"]))
            spread_floored_arr.append(bool(tr["spread_floored"]))
            spread_pips_entry_arr.append(float(tr["spread_pips_entry"]))
            spread_pips_exit_arr.append(float(tr["spread_pips_exit"]))
            fold_disp_arr.append(fold_disp_map.get(fid, "NA"))
        else:
            fold_id_arr.append(np.nan)
            taken_arr.append(False)
            exit_reason_arr.append("")
            R_arr.append(np.nan)
            mae_R_arr.append(np.nan)
            mfe_R_arr.append(np.nan)
            held_bars_arr.append(np.nan)
            spread_floored_arr.append(np.nan)
            spread_pips_entry_arr.append(np.nan)
            spread_pips_exit_arr.append(np.nan)
            fold_disp_arr.append("")

    all_signals["fold_id"] = fold_id_arr
    all_signals["taken"] = taken_arr
    all_signals["exit_reason"] = exit_reason_arr
    all_signals["R"] = R_arr
    all_signals["mae_R"] = mae_R_arr
    all_signals["mfe_R"] = mfe_R_arr
    all_signals["held_bars"] = held_bars_arr
    all_signals["spread_floored"] = spread_floored_arr
    all_signals["spread_pips_entry"] = spread_pips_entry_arr
    all_signals["spread_pips_exit"] = spread_pips_exit_arr
    all_signals["fold_disposition"] = fold_disp_arr

    # 5. signals_features.csv — primary deliverable. Lock column order
    # deterministically; format floats with fixed precision.
    feat_cols = [
        "pair", TIME_COL, "fold_id", "fold_disposition", "taken",
        "hour_utc", "dow", "session",
        "s_1h", "s_4h_mr", "s_d1_mr",
        "atr_1h_wilder_at_n",
        "atr_1h_regime", "atr_1h_regime_bin",
        "atr_4h_at_mr", "atr_4h_regime_at_mr", "atr_4h_regime_bin",
        "atr_d1_at_mr", "atr_d1_regime_at_mr", "atr_d1_regime_bin",
        "log_ret_1h_at_n",
        "cum_logret_1h_6", "cum_logret_1h_24", "cum_logret_1h_120",
        "pre_momentum_label",
        "bar_size_atr", "bar_body_atr", "close_position_in_bar",
        "dist_to_kijun_1h_atr",
        "dist_to_ema20_1h_atr", "dist_to_ema50_1h_atr",
        "dist_to_kijun_4h_at_mr_atr", "dist_to_ema20_4h_at_mr_atr",
        "dist_to_kijun_d1_at_mr_atr", "dist_to_ema20_d1_at_mr_atr",
        "ts_4h_used", "ts_d1_used",
        "concurrent_signals_same_bar",
        "exit_reason", "R", "mae_R", "mfe_R", "held_bars",
        "spread_pips_entry", "spread_pips_exit", "spread_floored",
    ] + [f"fwd_logret_h{h}" for h in FORWARD_HORIZONS]

    out = all_signals[feat_cols].copy()
    # Deterministic float formatting via str.format on selected columns.
    out_path = output_dir / "signals_features.csv"
    out.to_csv(out_path, index=False, lineterminator="\n", float_format="%.10g", date_format="%Y-%m-%dT%H:%M:%S")
    signals_features_sha = _sha256_file(out_path)

    # 6. magnitude_distribution.md — v1.1 locked deliverable.
    taken_rows = out[out["taken"] == True].copy()
    R_vec = taken_rows["R"].to_numpy(dtype=float)
    mae_vec = taken_rows["mae_R"].to_numpy(dtype=float)
    mfe_vec = taken_rows["mfe_R"].to_numpy(dtype=float)
    held_vec = taken_rows["held_bars"].to_numpy(dtype=float)

    def _stats(v: np.ndarray) -> Dict[str, float]:
        if v.size == 0:
            return {"n": 0, "mean": np.nan, "std": np.nan, "skew": np.nan, "kurt_excess": np.nan,
                    "min": np.nan, "p1": np.nan, "p5": np.nan, "p25": np.nan, "p50": np.nan,
                    "p75": np.nan, "p95": np.nan, "p99": np.nan, "max": np.nan}
        from scipy import stats as sps
        return {
            "n": int(v.size),
            "mean": float(np.mean(v)),
            "std": float(np.std(v, ddof=1)) if v.size > 1 else 0.0,
            "skew": float(sps.skew(v, bias=False)) if v.size > 2 else np.nan,
            "kurt_excess": float(sps.kurtosis(v, fisher=True, bias=False)) if v.size > 3 else np.nan,
            "min": float(np.min(v)),
            "p1": float(np.percentile(v, 1)),
            "p5": float(np.percentile(v, 5)),
            "p25": float(np.percentile(v, 25)),
            "p50": float(np.percentile(v, 50)),
            "p75": float(np.percentile(v, 75)),
            "p95": float(np.percentile(v, 95)),
            "p99": float(np.percentile(v, 99)),
            "max": float(np.max(v)),
        }

    R_stats = _stats(R_vec)
    mae_stats = _stats(mae_vec)
    mfe_stats = _stats(mfe_vec)

    n_sl = int(np.sum(taken_rows["exit_reason"] == "stop_loss"))
    n_te = int(np.sum(taken_rows["exit_reason"] == "time_exit"))
    n_de = int(np.sum(taken_rows["exit_reason"] == "data_end"))
    sl_rate = n_sl / max(1, len(taken_rows))
    te_rate = n_te / max(1, len(taken_rows))

    # Wasted-MFE statistic: held-bar MFE - realised R per trade (where R < MFE).
    wasted_mfe = mfe_vec - np.maximum(R_vec, 0.0)
    wasted_mfe = wasted_mfe[np.isfinite(wasted_mfe)]
    wasted_stats = _stats(wasted_mfe) if wasted_mfe.size > 0 else None

    # Conditional means by exit_reason.
    cond_means: Dict[str, float] = {}
    for er in ("stop_loss", "time_exit", "data_end"):
        sub = taken_rows[taken_rows["exit_reason"] == er]["R"].to_numpy(dtype=float)
        cond_means[er] = float(np.mean(sub)) if sub.size > 0 else float("nan")

    md_lines: List[str] = []
    md_lines.append("# Arc 2 Magnitude Distribution\n")
    md_lines.append("Locked under L6_0_METHODOLOGY_LOCK.md v1.1 §14.3. Descriptive only.\n")
    md_lines.append(f"Population: taken trades only (n={len(taken_rows)}).\n")
    md_lines.append("\n## Net R distribution (taken trades)\n")
    md_lines.append("| stat | value |")
    md_lines.append("|---|---|")
    for k in ("n","mean","std","skew","kurt_excess","min","p1","p5","p25","p50","p75","p95","p99","max"):
        v = R_stats[k]
        md_lines.append(f"| {k} | {v:.6g} |")
    md_lines.append("\n## MAE R distribution (taken trades)\n")
    md_lines.append("| stat | value |")
    md_lines.append("|---|---|")
    for k in ("n","mean","std","skew","kurt_excess","min","p1","p5","p25","p50","p75","p95","p99","max"):
        v = mae_stats[k]
        md_lines.append(f"| {k} | {v:.6g} |")
    md_lines.append("\n## MFE R distribution (taken trades)\n")
    md_lines.append("| stat | value |")
    md_lines.append("|---|---|")
    for k in ("n","mean","std","skew","kurt_excess","min","p1","p5","p25","p50","p75","p95","p99","max"):
        v = mfe_stats[k]
        md_lines.append(f"| {k} | {v:.6g} |")
    md_lines.append("\n## Exit-reason breakdown (taken trades)\n")
    md_lines.append("| exit_reason | n | rate | mean_R |")
    md_lines.append("|---|---|---|---|")
    md_lines.append(f"| stop_loss | {n_sl} | {sl_rate:.6f} | {cond_means['stop_loss']:.6f} |")
    md_lines.append(f"| time_exit | {n_te} | {te_rate:.6f} | {cond_means['time_exit']:.6f} |")
    md_lines.append(f"| data_end  | {n_de} | {n_de/max(1,len(taken_rows)):.6f} | {cond_means['data_end']:.6f} |")
    md_lines.append("\n## Wasted-MFE statistic\n")
    md_lines.append("Definition: per trade, max(0, MFE_R) − max(0, R). Trades where the favourable")
    md_lines.append("excursion was not captured by the time exit.\n")
    if wasted_stats is not None:
        md_lines.append("| stat | value |")
        md_lines.append("|---|---|")
        for k in ("n","mean","std","skew","kurt_excess","min","p1","p5","p25","p50","p75","p95","p99","max"):
            v = wasted_stats[k]
            md_lines.append(f"| {k} | {v:.6g} |")
    md_path = output_dir / "magnitude_distribution.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    mag_sha = _sha256_file(md_path)

    # 7. regime_breakdown.csv — pivot on session × pre_momentum × atr_1h_regime_bin (since
    # mtf_alignment is constant within population by construction).
    regime_rows: List[Dict[str, Any]] = []
    grp = taken_rows.groupby(["session", "pre_momentum_label", "atr_1h_regime_bin"])
    for (sess, pm, ab), sub in grp:
        n = len(sub)
        m_R = float(sub["R"].mean()) if n > 0 else 0.0
        sl_n = int(np.sum(sub["exit_reason"] == "stop_loss"))
        regime_rows.append({
            "session": sess,
            "pre_momentum_label": pm,
            "atr_1h_regime_bin": ab,
            "n": n,
            "mean_R": round(m_R, 6),
            "sl_rate": round(sl_n / n if n > 0 else 0.0, 6),
        })
    regime_path = output_dir / "regime_breakdown.csv"
    with regime_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["session", "pre_momentum_label", "atr_1h_regime_bin", "n", "mean_R", "sl_rate"])
        for r in regime_rows:
            w.writerow([r["session"], r["pre_momentum_label"], r["atr_1h_regime_bin"],
                        r["n"], f"{r['mean_R']:.6f}", f"{r['sl_rate']:.6f}"])
    regime_sha = _sha256_file(regime_path)

    # 8. forward_horizon_curves.csv — descriptive forward returns at multiple horizons per
    # session × pre_momentum_label.
    fwd_rows: List[Dict[str, Any]] = []
    for h in FORWARD_HORIZONS:
        col = f"fwd_logret_h{h}"
        for (sess, pm), sub in taken_rows.groupby(["session", "pre_momentum_label"]):
            vec = sub[col].to_numpy(dtype=float)
            vec = vec[np.isfinite(vec)]
            if vec.size == 0:
                continue
            fwd_rows.append({
                "horizon": h,
                "session": sess,
                "pre_momentum_label": pm,
                "n": int(vec.size),
                "mean_logret": round(float(vec.mean()), 8),
                "median_logret": round(float(np.median(vec)), 8),
                "std_logret": round(float(vec.std(ddof=1)) if vec.size > 1 else 0.0, 8),
            })
    fwd_path = output_dir / "forward_horizon_curves.csv"
    with fwd_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["horizon", "session", "pre_momentum_label", "n", "mean_logret", "median_logret", "std_logret"])
        for r in fwd_rows:
            w.writerow([r["horizon"], r["session"], r["pre_momentum_label"],
                        r["n"], f"{r['mean_logret']:.8f}", f"{r['median_logret']:.8f}", f"{r['std_logret']:.8f}"])
    fwd_sha = _sha256_file(fwd_path)

    # 9. pair_breakdown.csv — per-pair aggregations.
    pair_rows: List[Dict[str, Any]] = []
    for pair, sub in taken_rows.groupby("pair"):
        n = len(sub)
        m_R = float(sub["R"].mean()) if n > 0 else 0.0
        sl_n = int(np.sum(sub["exit_reason"] == "stop_loss"))
        winR = int(np.sum(sub["R"].to_numpy() > 0))
        pair_rows.append({
            "pair": pair,
            "n": n,
            "mean_R": round(m_R, 6),
            "median_R": round(float(sub["R"].median()), 6),
            "win_rate": round(winR / n if n > 0 else 0.0, 6),
            "sl_hit_rate": round(sl_n / n if n > 0 else 0.0, 6),
        })
    pair_path = output_dir / "pair_breakdown.csv"
    with pair_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["pair", "n", "mean_R", "median_R", "win_rate", "sl_hit_rate"])
        for r in pair_rows:
            w.writerow([r["pair"], r["n"], f"{r['mean_R']:.6f}", f"{r['median_R']:.6f}",
                        f"{r['win_rate']:.6f}", f"{r['sl_hit_rate']:.6f}"])
    pair_sha = _sha256_file(pair_path)

    # 10. characterisation_report.md — narrative summary (descriptive only).
    n_taken = len(taken_rows)
    n_total = len(out)
    n_dropped = n_total - n_taken
    report_lines: List[str] = []
    report_lines.append("# Arc 2 Characterisation Report\n")
    report_lines.append(
        "Locked under L6_0_METHODOLOGY_LOCK.md v1.1 §14.4. **Descriptive only — no "
        "filter derivation, no gate decisions. Per §14.5, the findings section is "
        "empirical observations only; action-shaped statements go to "
        "`docs/CANDIDATE_HYPOTHESES.md`.**\n"
    )
    report_lines.append(f"\n## Population\n")
    report_lines.append(f"- Total signals in window: {n_total}")
    report_lines.append(f"- Taken trades (post concurrent-cap): {n_taken}")
    report_lines.append(f"- Dropped (concurrent_open_position + no_next_bar): {n_dropped}")
    report_lines.append(
        f"- MTF alignment triple (s_1h, s_4h_mr, s_d1_mr) is `(-1, +1, -1)` constant by construction.\n"
    )
    report_lines.append("## Magnitude summary (taken trades)\n")
    report_lines.append(
        f"- Net R mean: {R_stats['mean']:.6f}; std: {R_stats['std']:.6f}; "
        f"skew: {R_stats['skew']:.6f}; excess-kurt: {R_stats['kurt_excess']:.6f}"
    )
    report_lines.append(
        f"- Median R: {R_stats['p50']:.6f}; p25: {R_stats['p25']:.6f}; "
        f"p75: {R_stats['p75']:.6f}; min: {R_stats['min']:.6f}; max: {R_stats['max']:.6f}"
    )
    report_lines.append(
        f"- SL hit rate: {sl_rate:.6f}; time-exit rate: {te_rate:.6f}"
    )
    report_lines.append(
        f"- Conditional means: stop_loss = {cond_means['stop_loss']:.6f}; "
        f"time_exit = {cond_means['time_exit']:.6f}\n"
    )
    report_lines.append("## Per-fold disposition tally\n")
    report_lines.append("| fold_id | gate | n_trades | mean_R |")
    report_lines.append("|---|---|---|---|")
    for fid in sorted(taken_rows["fold_id"].dropna().unique()):
        sub = taken_rows[taken_rows["fold_id"] == fid]
        disp = fold_disp_map.get(int(fid), "NA")
        report_lines.append(f"| {int(fid)} | {disp} | {len(sub)} | {sub['R'].mean():.6f} |")
    report_lines.append("\n## Per-session conditional R (taken trades)\n")
    report_lines.append("| session | n | mean_R | median_R |")
    report_lines.append("|---|---|---|---|")
    for sess, sub in taken_rows.groupby("session"):
        report_lines.append(
            f"| {sess} | {len(sub)} | {sub['R'].mean():.6f} | {sub['R'].median():.6f} |"
        )
    report_lines.append("\n## Per-pre-momentum-label conditional R (taken trades)\n")
    report_lines.append("| pre_momentum_label | n | mean_R | median_R |")
    report_lines.append("|---|---|---|---|")
    for pm, sub in taken_rows.groupby("pre_momentum_label"):
        report_lines.append(
            f"| {pm} | {len(sub)} | {sub['R'].mean():.6f} | {sub['R'].median():.6f} |"
        )
    report_lines.append("\n## Per-ATR-regime-bin conditional R (taken trades, 1H ATR regime)\n")
    report_lines.append("| atr_1h_regime_bin | n | mean_R |")
    report_lines.append("|---|---|---|")
    for ab, sub in taken_rows.groupby("atr_1h_regime_bin"):
        report_lines.append(f"| {ab} | {len(sub)} | {sub['R'].mean():.6f} |")
    report_lines.append("\n## Findings (empirical only)\n")
    report_lines.append(
        "Statements below are factual descriptions of the trade population. No "
        "filter, no system suggestion. Action hypotheses go to "
        "`docs/CANDIDATE_HYPOTHESES.md` per §14.5.\n"
    )
    # Top-3 observations: largest deviating session/pre_mom/regime mean R cells.
    report_lines.append(
        f"- The trade population shows mean net R = {R_stats['mean']:.4f} with std {R_stats['std']:.4f}; "
        f"SL hit rate is {sl_rate:.4f} and time-exit rate {te_rate:.4f}."
    )
    sess_means = {sess: float(sub["R"].mean()) for sess, sub in taken_rows.groupby("session")}
    if sess_means:
        best_sess = max(sess_means.items(), key=lambda kv: kv[1])
        worst_sess = min(sess_means.items(), key=lambda kv: kv[1])
        report_lines.append(
            f"- Conditional mean R by session ranges {worst_sess[1]:.4f} ({worst_sess[0]}) "
            f"to {best_sess[1]:.4f} ({best_sess[0]})."
        )
    pm_means = {pm: float(sub["R"].mean()) for pm, sub in taken_rows.groupby("pre_momentum_label")}
    if pm_means:
        report_lines.append(
            "- Pre-momentum label conditional mean R: " +
            "; ".join(f"{pm}={v:.4f} (n={len(taken_rows[taken_rows['pre_momentum_label']==pm])})" for pm, v in pm_means.items())
        )
    report_lines.append(
        f"- Wasted-MFE statistic (held-bar MFE not captured by time exit): "
        f"mean {wasted_stats['mean']:.4f} R per trade (n={wasted_stats['n']})"
        if wasted_stats is not None else "- Wasted-MFE statistic: insufficient data"
    )
    report_lines.append("\n## Disposition note\n")
    report_lines.append(
        "Per §14.5 the characterisation findings are descriptive observations only. "
        "Any hypothesis about a filter that would improve gate disposition is recorded "
        "as a candidate hypothesis in `docs/CANDIDATE_HYPOTHESES.md` and would require "
        "a fresh arc with the filter pre-committed BEFORE any fold result is seen."
    )
    report_path = output_dir / "characterisation_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    report_sha = _sha256_file(report_path)

    # 11. feature_lag_audit.txt — deterministic 100-row sample with timestamps used per TF.
    rng_seed = 20260511
    rng = np.random.default_rng(rng_seed)
    audit_n = min(100, len(out))
    if audit_n > 0:
        sample_idx = rng.choice(len(out), size=audit_n, replace=False)
        sample = out.iloc[sample_idx].sort_values([TIME_COL, "pair"]).reset_index(drop=True)
    else:
        sample = out.iloc[:0]
    audit_lines: List[str] = []
    audit_lines.append("Arc 2 feature-lag audit (deterministic 100-row sample, seed=20260511)\n")
    audit_lines.append("-" * 80)
    audit_lines.append("Invariants:")
    audit_lines.append("  - s_1H read at this 1H bar's close (no lag).")
    audit_lines.append("  - ts_4h_used = 4H bar at index floor(T_N, '4h') -> idx_4h - 1 (strict prior bar).")
    audit_lines.append("  - ts_d1_used = D1 bar at index floor(T_N, 'D') -> idx_d1 - 1 (strict prior bar).")
    audit_lines.append("  - For each sample row: ts_4h_used must be strictly < floor(T_N, '4h').")
    audit_lines.append("  - For each sample row: ts_d1_used.date() must be strictly < T_N.date().")
    audit_lines.append("")
    violations = 0
    for _, row in sample.iterrows():
        T_N = pd.Timestamp(row[TIME_COL])
        ts4 = pd.Timestamp(row["ts_4h_used"]) if pd.notna(row["ts_4h_used"]) else pd.NaT
        tsd = pd.Timestamp(row["ts_d1_used"]) if pd.notna(row["ts_d1_used"]) else pd.NaT
        f4 = T_N.floor("4h")
        fd = T_N.normalize()
        ok_4h = pd.isna(ts4) or ts4 < f4
        ok_d1 = pd.isna(tsd) or tsd < fd
        flag = "OK" if (ok_4h and ok_d1) else "VIOLATION"
        if not (ok_4h and ok_d1):
            violations += 1
        audit_lines.append(
            f"  {row['pair']}  T_N={T_N}  ts_4h_used={ts4}  ts_d1_used={tsd}  f4={f4}  fd={fd}  [{flag}]"
        )
    audit_lines.append("")
    audit_lines.append(f"Violations in sample: {violations} / {audit_n}")
    audit_path = output_dir / "feature_lag_audit.txt"
    audit_path.write_text("\n".join(audit_lines) + "\n", encoding="utf-8")
    audit_sha = _sha256_file(audit_path)

    # 12. run_manifest.txt — input/output sha256s + determinism placeholder.
    manifest_lines: List[str] = []
    manifest_lines.append("Arc 2 characterisation — run manifest")
    manifest_lines.append("-" * 60)
    manifest_lines.append(f"Run timestamp (UTC-naive): {_dt.datetime.now().isoformat(timespec='seconds')}")
    manifest_lines.append(f"Repo root: {REPO_ROOT}")
    manifest_lines.append("")
    manifest_lines.append("Inputs (sha256):")
    inputs = {
        "results/l6/arc2/trades_all.csv": arc2_trades_path,
        "results/l6/arc2/wfo_fold_results.csv": fold_results_path,
        "core/signals/l4_mtf_alignment_2_down_mixed_kijun.py":
            REPO_ROOT / "core" / "signals" / "l4_mtf_alignment_2_down_mixed_kijun.py",
        "configs/wfo_l6_arc2.yaml": REPO_ROOT / "configs" / "wfo_l6_arc2.yaml",
    }
    for k, v in inputs.items():
        if v.exists():
            manifest_lines.append(f"  {k}\t{_sha256_file(v)}")
    manifest_lines.append("")
    manifest_lines.append("Outputs (sha256):")
    out_files = {
        "signals_features.csv": signals_features_sha,
        "characterisation_report.md": report_sha,
        "regime_breakdown.csv": regime_sha,
        "forward_horizon_curves.csv": fwd_sha,
        "pair_breakdown.csv": pair_sha,
        "magnitude_distribution.md": mag_sha,
        "feature_lag_audit.txt": audit_sha,
    }
    for k, h in out_files.items():
        manifest_lines.append(f"  {k}\t{h}")
    manifest_path = output_dir / "run_manifest.txt"
    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    return {
        "signals_features.csv": signals_features_sha,
        "characterisation_report.md": report_sha,
        "regime_breakdown.csv": regime_sha,
        "forward_horizon_curves.csv": fwd_sha,
        "pair_breakdown.csv": pair_sha,
        "magnitude_distribution.md": mag_sha,
        "feature_lag_audit.txt": audit_sha,
        "run_manifest.txt": _sha256_file(manifest_path),
    }


def _cli() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="results/l6/arc2/characterisation/")
    p.add_argument("--trades-csv", default="results/l6/arc2/trades_all.csv")
    p.add_argument("--single-run", action="store_true")
    args = p.parse_args()
    out_dir = (REPO_ROOT / args.output_dir).resolve()
    trades = (REPO_ROOT / args.trades_csv).resolve()
    print(f"Output dir: {out_dir}")
    print(f"Arc 2 trades: {trades}")
    t1 = time.time()
    sha1 = run_arc2_characterisation(output_dir=out_dir, arc2_trades_path=trades)
    print(f"Run #1 done in {time.time()-t1:.1f}s")
    if not args.single_run:
        # Re-run to scratch dir, verify byte-identical signals_features.csv.
        import tempfile
        tmp = Path(tempfile.mkdtemp(prefix="arc2_char_run2_"))
        sha2 = run_arc2_characterisation(output_dir=tmp, arc2_trades_path=trades)
        if sha1["signals_features.csv"] != sha2["signals_features.csv"]:
            print(f"!!! DETERMINISM FAIL: signals_features.csv differs between runs")
            print(f"  run1: {sha1['signals_features.csv']}")
            print(f"  run2: {sha2['signals_features.csv']}")
            print(f"  scratch: {tmp}")
            return 1
        print(f"Determinism OK: signals_features.csv byte-identical across runs.")
        # Append determinism receipt to run_manifest.txt
        manifest_path = out_dir / "run_manifest.txt"
        existing = manifest_path.read_text(encoding="utf-8")
        extra = (
            "\nDeterminism (two consecutive runs):\n"
            f"  run #1 signals_features.csv: {sha1['signals_features.csv']}\n"
            f"  run #2 signals_features.csv: {sha2['signals_features.csv']}\n"
            f"  match: True\n"
        )
        manifest_path.write_text(existing + extra, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
