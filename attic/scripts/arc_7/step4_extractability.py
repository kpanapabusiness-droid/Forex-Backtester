"""Arc 7 — Step 4 extractability investigation.

L_ARC_PROTOCOL.md v2.1.2 §§8, 10, 17. For each Step-3 surviving unit (c1, c3,
agg_c1_c3), train Pipeline E (4H entry-time features) and Pipeline D1 (daily
regime features, one-day-lagged) Random Forest classifiers predicting binary
success at the unit's selected SL. Compute mean ROC AUC across 5
TimeSeriesSplit folds, apply §8 gate (E ≥ 0.65 OR D1 ≥ 0.60), sweep
admission threshold t ∈ {0.30 .. 0.80} step 0.02 for the largest t with
exclusion ≤ 0.30.

Success label per unit: re-impose the unit's selected SL on the full §15a
bar path (held + forward-observation rows); success = 1 iff `final_r ≥ 1.0`
in the new R-frame. SL re-imposition routine is imported from
``scripts.arc_7.step3_capturability._eval_trade_at_sl`` for byte-identity
with Step 3.

Feature design notes (deliberate departure from dispatch's strict one-hot):
  - hour_of_day and day_of_week encoded as sin/cos pairs (4 features) rather
    than one-hot (29 features) — RF picks up cyclic patterns identically and
    the smallest unit (c1, n=185) cannot tolerate 76+ features.
  - pair_id remains integer-encoded (RF handles it natively); flagged below
    for pair-leak risk monitoring.
  - All other features per dispatch verbatim.

Models: RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42,
n_jobs=1). n_jobs=1 (not -1) to guarantee determinism on Windows runners.

Determinism: two consecutive runs produce byte-identical CSVs. RF with fixed
random_state + n_jobs=1 is deterministic; sklearn.inspection.permutation_importance
uses an explicit random_state.

Usage:
    py scripts/arc_7/step4_extractability.py -c configs/arc_7/step4.yaml
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Import Step 3 SL eval for byte-identical success labels.
from scripts.arc_7.step3_capturability import _eval_trade_at_sl  # noqa: E402

ATR_PERIOD = 14


# ============================================================
# Indicator helpers (causal)
# ============================================================


def _wilder_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> np.ndarray:
    hi = df["high"].astype(float).to_numpy()
    lo = df["low"].astype(float).to_numpy()
    cl = df["close"].astype(float).to_numpy()
    n = hi.size
    if n == 0:
        return np.array([], dtype=float)
    prev_cl = np.empty(n, dtype=float)
    prev_cl[0] = np.nan
    prev_cl[1:] = cl[:-1]
    tr = np.maximum.reduce([hi - lo, np.abs(hi - prev_cl), np.abs(lo - prev_cl)])
    tr[0] = hi[0] - lo[0]
    atr = np.full(n, np.nan, dtype=float)
    if n < period:
        return atr
    atr[period - 1] = float(np.mean(tr[:period]))
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _ema(series: np.ndarray, period: int) -> np.ndarray:
    out = np.full_like(series, np.nan, dtype=float)
    if series.size == 0:
        return out
    alpha = 2.0 / (period + 1.0)
    # First EMA value seeded with first observation (standard).
    out[0] = series[0]
    for i in range(1, series.size):
        if math.isnan(series[i]):
            out[i] = out[i - 1]
        else:
            out[i] = alpha * series[i] + (1.0 - alpha) * out[i - 1]
    return out


# ============================================================
# Per-pair cache
# ============================================================


@dataclass
class PerPairCache:
    df_4h: pd.DataFrame
    idx_by_ts: Dict[pd.Timestamp, int]
    atr_4h: np.ndarray
    ema20_4h: np.ndarray
    ema50_4h: np.ndarray
    ema200_4h: np.ndarray
    df_d1: pd.DataFrame
    d1_date_norm: np.ndarray   # normalized D1 dates (datetime64[ns])
    atr_d1: np.ndarray
    ema20_d1: np.ndarray
    ema50_d1: np.ndarray
    ema200_d1: np.ndarray


def _load_pair_tf(pair: str, tf_dir: str) -> pd.DataFrame:
    path = _REPO_ROOT / tf_dir / f"{pair}.csv"
    df = pd.read_csv(path)
    if "time" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def _build_pair_cache(pair: str, dir_4h: str, dir_d1: str) -> PerPairCache:
    df_4h = _load_pair_tf(pair, dir_4h)
    atr_4h = _wilder_atr(df_4h, ATR_PERIOD)
    close_4h = df_4h["close"].astype(float).to_numpy()
    ema20_4h = _ema(close_4h, 20)
    ema50_4h = _ema(close_4h, 50)
    ema200_4h = _ema(close_4h, 200)
    idx_by_ts = {pd.Timestamp(ts): i for i, ts in enumerate(df_4h["date"].to_numpy())}

    df_d1 = _load_pair_tf(pair, dir_d1)
    df_d1["date_norm"] = df_d1["date"].dt.normalize()
    df_d1 = df_d1.drop_duplicates(subset="date_norm").sort_values("date_norm").reset_index(drop=True)
    close_d1 = df_d1["close"].astype(float).to_numpy()
    atr_d1 = _wilder_atr(df_d1, ATR_PERIOD)
    ema20_d1 = _ema(close_d1, 20)
    ema50_d1 = _ema(close_d1, 50)
    ema200_d1 = _ema(close_d1, 200)
    d1_dates = df_d1["date_norm"].to_numpy()

    return PerPairCache(
        df_4h=df_4h, idx_by_ts=idx_by_ts, atr_4h=atr_4h,
        ema20_4h=ema20_4h, ema50_4h=ema50_4h, ema200_4h=ema200_4h,
        df_d1=df_d1, d1_date_norm=d1_dates, atr_d1=atr_d1,
        ema20_d1=ema20_d1, ema50_d1=ema50_d1, ema200_d1=ema200_d1,
    )


def _d1_lag1_idx(cache: PerPairCache, signal_ts: pd.Timestamp) -> int:
    """Index of the D1 bar at calendar_date - 1 (one-day lag). -1 if not present."""
    target = (pd.Timestamp(signal_ts).normalize() - pd.Timedelta(days=1)).to_datetime64()
    dates = pd.DatetimeIndex(cache.d1_date_norm)
    idx_arr = dates.get_indexer([target], method="ffill")
    return int(idx_arr[0])


# ============================================================
# Feature computation (Pipeline E)
# ============================================================


PIPELINE_E_FEATURES: List[str] = [
    # Volatility
    "atr14_4h_at_entry",
    "atr14_4h_at_entry_pct_of_close",
    # EMA dist + slope (4H)
    "ema20_4h_dist_atr",
    "ema50_4h_dist_atr",
    "ema200_4h_dist_atr",
    "ema20_slope_4h",
    "ema50_slope_4h",
    # Recent range
    "range_pct_atr_last_5_bars",
    "range_pct_atr_last_20_bars",
    # Bar shape
    "bar_body_pct",
    "upper_wick_pct",
    "lower_wick_pct",
    # Cyclic time
    "hour_sin", "hour_cos",
    "weekday_sin", "weekday_cos",
    # Pair (integer-coded)
    "pair_id_int",
    # Arc-7-specific
    "reclaim_strength_ratio",
    "sweep_magnitude_atr",
    "swing_low_distance_atr",
    "bars_since_last_signal_on_pair",
    "swing_low_age_bars",
]


def compute_pipeline_e_features(
    trades_df: pd.DataFrame,
    pair_caches: Dict[str, PerPairCache],
) -> pd.DataFrame:
    """Pipeline E entry-time features. trades_df must include `pair`, `signal_bar_time`,
    `entry_time`, `entry_price`, `swing_low_used`, `reclaim_strength_ratio`,
    `sweep_magnitude_atr`."""
    # pair_id mapping (deterministic alphabetical).
    pairs_sorted = sorted(trades_df["pair"].astype(str).unique())
    pair_to_int = {p: i for i, p in enumerate(pairs_sorted)}

    # bars_since_last_signal_on_pair: walk chronological per pair.
    bsl_lookup: Dict[Tuple[str, pd.Timestamp], int] = {}
    trades_sorted = trades_df.copy()
    trades_sorted["signal_bar_time"] = pd.to_datetime(trades_sorted["signal_bar_time"])
    trades_sorted = trades_sorted.sort_values(["pair", "signal_bar_time"]).reset_index(drop=True)
    for pair, g in trades_sorted.groupby("pair", sort=True):
        cache = pair_caches[pair]
        sig_times = g["signal_bar_time"].tolist()
        prev_idx = -1
        for sig in sig_times:
            cur_idx = cache.idx_by_ts.get(pd.Timestamp(sig), -1)
            if prev_idx < 0 or cur_idx < 0:
                bsl = -1
            else:
                bsl = cur_idx - prev_idx
            bsl_lookup[(pair, pd.Timestamp(sig))] = int(bsl)
            prev_idx = cur_idx

    rows: List[Dict[str, Any]] = []
    for _, t in trades_df.iterrows():
        pair = str(t["pair"])
        cache = pair_caches[pair]
        sig_ts = pd.Timestamp(t["signal_bar_time"])
        i = cache.idx_by_ts.get(sig_ts, -1)
        if i < 0:
            raise ValueError(f"signal_bar_time {sig_ts} not found in {pair} 4H data")

        o_t = float(cache.df_4h["open"].iloc[i])
        h_t = float(cache.df_4h["high"].iloc[i])
        l_t = float(cache.df_4h["low"].iloc[i])
        c_t = float(cache.df_4h["close"].iloc[i])
        rng_t = h_t - l_t
        atr_t = float(cache.atr_4h[i]) if not math.isnan(cache.atr_4h[i]) else float("nan")
        ema20 = cache.ema20_4h[i]
        ema50 = cache.ema50_4h[i]
        ema200 = cache.ema200_4h[i]

        # ATR
        atr14 = atr_t
        atr14_pct = atr_t / c_t if (c_t > 0 and atr_t > 0) else float("nan")

        # EMA dist / slope (in ATR units, signed: positive = close above EMA)
        def _dist(ema_val: float) -> float:
            return (c_t - ema_val) / atr_t if (atr_t > 0 and not math.isnan(ema_val)) else float("nan")
        d20 = _dist(ema20)
        d50 = _dist(ema50)
        d200 = _dist(ema200)

        def _slope(ema_arr: np.ndarray, lookback: int = 5) -> float:
            if i < lookback or math.isnan(ema_arr[i]) or math.isnan(ema_arr[i - lookback]):
                return float("nan")
            return (ema_arr[i] - ema_arr[i - lookback]) / lookback / atr_t if atr_t > 0 else float("nan")
        s20 = _slope(cache.ema20_4h)
        s50 = _slope(cache.ema50_4h)

        # Recent range
        def _range_pct_atr(lookback: int) -> float:
            if i < lookback or atr_t <= 0:
                return float("nan")
            highs = cache.df_4h["high"].iloc[i - lookback:i].astype(float).to_numpy()
            lows = cache.df_4h["low"].iloc[i - lookback:i].astype(float).to_numpy()
            return (float(highs.max()) - float(lows.min())) / atr_t
        r5 = _range_pct_atr(5)
        r20 = _range_pct_atr(20)

        # Bar shape
        body_pct = abs(c_t - o_t) / rng_t if rng_t > 0 else 0.0
        upper_wick = (h_t - max(o_t, c_t)) / rng_t if rng_t > 0 else 0.0
        lower_wick = (min(o_t, c_t) - l_t) / rng_t if rng_t > 0 else 0.0

        # Cyclic time
        hour = sig_ts.hour
        dow = sig_ts.weekday()
        h_sin = math.sin(2.0 * math.pi * hour / 24.0)
        h_cos = math.cos(2.0 * math.pi * hour / 24.0)
        dow_sin = math.sin(2.0 * math.pi * dow / 7.0)
        dow_cos = math.cos(2.0 * math.pi * dow / 7.0)

        # Arc-7 signal features
        reclaim = float(t.get("reclaim_strength_ratio", float("nan")))
        sweep_atr = float(t.get("sweep_magnitude_atr", float("nan")))
        entry_price = float(t["entry_price"])
        swing_low = float(t.get("swing_low_used", float("nan")))
        swing_low_dist = (entry_price - swing_low) / atr_t if (atr_t > 0 and not math.isnan(swing_low)) else float("nan")

        # bars_since_last_signal_on_pair
        bsl = bsl_lookup.get((pair, sig_ts), -1)

        # swing_low_age_bars: bars since swing_low was last touched (low == swing_low_used).
        # Look back up to 60 bars; cap at 60 if not found.
        age = 60
        if not math.isnan(swing_low):
            for j in range(i - 1, max(i - 61, -1), -1):
                if abs(float(cache.df_4h["low"].iloc[j]) - swing_low) < 1e-12:
                    age = i - j
                    break

        rows.append({
            "trade_id": int(t["trade_id"]),
            "pair": pair,
            "entry_time": pd.Timestamp(t["entry_time"]),
            "atr14_4h_at_entry": atr14,
            "atr14_4h_at_entry_pct_of_close": atr14_pct,
            "ema20_4h_dist_atr": d20,
            "ema50_4h_dist_atr": d50,
            "ema200_4h_dist_atr": d200,
            "ema20_slope_4h": s20,
            "ema50_slope_4h": s50,
            "range_pct_atr_last_5_bars": r5,
            "range_pct_atr_last_20_bars": r20,
            "bar_body_pct": body_pct,
            "upper_wick_pct": upper_wick,
            "lower_wick_pct": lower_wick,
            "hour_sin": h_sin, "hour_cos": h_cos,
            "weekday_sin": dow_sin, "weekday_cos": dow_cos,
            "pair_id_int": pair_to_int[pair],
            "reclaim_strength_ratio": reclaim,
            "sweep_magnitude_atr": sweep_atr,
            "swing_low_distance_atr": swing_low_dist,
            "bars_since_last_signal_on_pair": bsl,
            "swing_low_age_bars": age,
        })
    return pd.DataFrame(rows)


# ============================================================
# Feature computation (Pipeline D1)
# ============================================================


PIPELINE_D1_FEATURES: List[str] = [
    "ema20_d1_dist_atr",
    "ema50_d1_dist_atr",
    "ema200_d1_dist_atr",
    "ema20_slope_d1",
    "ema50_slope_d1",
    "atr14_d1_at_entry",
    "atr14_d1_at_entry_pct_of_close",
    "d1_trend_up", "d1_trend_down", "d1_trend_range",  # one-hot
    "d1_range_pct_atr_last_5_bars",
]


def compute_pipeline_d1_features(
    trades_df: pd.DataFrame,
    pair_caches: Dict[str, PerPairCache],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []
    for _, t in trades_df.iterrows():
        pair = str(t["pair"])
        cache = pair_caches[pair]
        sig_ts = pd.Timestamp(t["signal_bar_time"])
        d1_idx = _d1_lag1_idx(cache, sig_ts)
        if d1_idx < 0:
            row = {f: float("nan") for f in PIPELINE_D1_FEATURES}
        else:
            close_d1 = float(cache.df_d1["close"].iloc[d1_idx])
            atr_d1_val = float(cache.atr_d1[d1_idx]) if not math.isnan(cache.atr_d1[d1_idx]) else float("nan")
            e20 = float(cache.ema20_d1[d1_idx])
            e50 = float(cache.ema50_d1[d1_idx])
            e200 = float(cache.ema200_d1[d1_idx])

            def _dist(ev: float) -> float:
                return (close_d1 - ev) / atr_d1_val if (atr_d1_val > 0 and not math.isnan(ev)) else float("nan")
            d20 = _dist(e20)
            d50 = _dist(e50)
            d200 = _dist(e200)

            def _slope(ema_arr: np.ndarray, lookback: int = 5) -> float:
                if d1_idx < lookback or math.isnan(ema_arr[d1_idx]) or math.isnan(ema_arr[d1_idx - lookback]):
                    return float("nan")
                return (ema_arr[d1_idx] - ema_arr[d1_idx - lookback]) / lookback / atr_d1_val if atr_d1_val > 0 else float("nan")
            s20 = _slope(cache.ema20_d1)
            s50 = _slope(cache.ema50_d1)

            atr_pct = atr_d1_val / close_d1 if close_d1 > 0 else float("nan")

            # d1_trend_label from EMA stack.
            if not (math.isnan(e20) or math.isnan(e50) or math.isnan(e200)):
                if e20 > e50 > e200:
                    trend = "up"
                elif e20 < e50 < e200:
                    trend = "down"
                else:
                    trend = "range"
            else:
                trend = "range"

            # 5-bar range pct atr (D1)
            if d1_idx >= 5 and atr_d1_val > 0:
                hi5 = cache.df_d1["high"].iloc[d1_idx - 5:d1_idx].astype(float).to_numpy()
                lo5 = cache.df_d1["low"].iloc[d1_idx - 5:d1_idx].astype(float).to_numpy()
                r5_d1 = (float(hi5.max()) - float(lo5.min())) / atr_d1_val
            else:
                r5_d1 = float("nan")

            row = {
                "ema20_d1_dist_atr": d20,
                "ema50_d1_dist_atr": d50,
                "ema200_d1_dist_atr": d200,
                "ema20_slope_d1": s20,
                "ema50_slope_d1": s50,
                "atr14_d1_at_entry": atr_d1_val,
                "atr14_d1_at_entry_pct_of_close": atr_pct,
                "d1_trend_up": 1.0 if trend == "up" else 0.0,
                "d1_trend_down": 1.0 if trend == "down" else 0.0,
                "d1_trend_range": 1.0 if trend == "range" else 0.0,
                "d1_range_pct_atr_last_5_bars": r5_d1,
            }

        row["trade_id"] = int(t["trade_id"])
        rows.append(row)

        # Record audit info (used by D1 lag audit).
        d1_date_joined = str(cache.df_d1["date_norm"].iloc[d1_idx]) if d1_idx >= 0 else "N/A"
        audit_rows.append({
            "trade_id": int(t["trade_id"]),
            "pair": pair,
            "entry_time": str(pd.Timestamp(t["entry_time"])),
            "signal_bar_time": str(sig_ts),
            "expected_d1_date_max": str((sig_ts.normalize() - pd.Timedelta(days=1)).to_pydatetime()),
            "d1_date_joined": d1_date_joined,
            "lag_correct": (d1_idx >= 0 and
                            pd.Timestamp(cache.df_d1["date_norm"].iloc[d1_idx]) <=
                            (sig_ts.normalize() - pd.Timedelta(days=1))),
        })
    return pd.DataFrame(rows), pd.DataFrame(audit_rows)


# ============================================================
# Success label (re-impose SL, success = final_r ≥ 1.0)
# ============================================================


def _build_paths_index(paths_df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    out: Dict[int, pd.DataFrame] = {}
    paths_sorted = paths_df.sort_values(["trade_id", "bar_offset"], kind="mergesort")
    for tid, g in paths_sorted.groupby("trade_id", sort=True):
        out[int(tid)] = g.reset_index(drop=True)
    return out


def compute_success_labels(
    trade_ids: List[int],
    paths_index: Dict[int, pd.DataFrame],
    sl_atr_mult: float,
    original_sl_atr_mult: float,
) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for tid in trade_ids:
        path = paths_index[tid]
        te = _eval_trade_at_sl(path, sl_atr_mult, original_sl_atr_mult)
        out[tid] = 1 if te.final_r_new >= 1.0 else 0
    return out


# ============================================================
# Modeling
# ============================================================


@dataclass
class PipelineResult:
    unit_id: str
    pipeline: str
    n: int
    n_features: int
    base_success_rate: float
    fold_aucs: List[float]
    mean_auc: float
    std_auc: float
    fold_n_train: List[int]
    fold_n_test: List[int]
    fold_base_success: List[float]
    gate_threshold: float
    gate_pass: bool
    selected_t: Optional[float]
    exclusion_at_t: Optional[float]
    admission_at_t: Optional[float]
    realised_success_at_t: Optional[float]
    lift_at_t: Optional[float]
    class_weight_used: str
    top_gini: List[Tuple[str, float]] = field(default_factory=list)
    top_perm: List[Tuple[str, float, float]] = field(default_factory=list)
    threshold_sweep: List[Tuple[float, float, float, float, float]] = field(default_factory=list)


def _cv_aucs(
    X: pd.DataFrame, y: np.ndarray, model_kw: dict, class_weight_used: str, n_splits: int,
) -> Tuple[List[float], List[int], List[int], List[float]]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import TimeSeriesSplit

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_aucs: List[float] = []
    fold_n_train: List[int] = []
    fold_n_test: List[int] = []
    fold_base_succ: List[float] = []
    for train_idx, test_idx in tscv.split(X):
        X_tr = X.iloc[train_idx]
        y_tr = y[train_idx]
        X_te = X.iloc[test_idx]
        y_te = y[test_idx]
        fold_n_train.append(int(len(train_idx)))
        fold_n_test.append(int(len(test_idx)))
        fold_base_succ.append(float(y_te.mean()) if y_te.size > 0 else 0.0)
        # Need both classes in train + test.
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            fold_aucs.append(float("nan"))
            continue
        kw = dict(model_kw)
        if class_weight_used == "balanced":
            kw["class_weight"] = "balanced"
        clf = RandomForestClassifier(**kw)
        # Impute NaNs to feature medians from train fold for both train and test.
        med = X_tr.median(numeric_only=True)
        X_tr_f = X_tr.fillna(med)
        X_te_f = X_te.fillna(med)
        clf.fit(X_tr_f, y_tr)
        p_te = clf.predict_proba(X_te_f)[:, 1]
        try:
            auc = float(roc_auc_score(y_te, p_te))
        except ValueError:
            auc = float("nan")
        fold_aucs.append(auc)
    return fold_aucs, fold_n_train, fold_n_test, fold_base_succ


def _full_fit_and_threshold(
    X: pd.DataFrame, y: np.ndarray, feature_cols: List[str], model_kw: dict,
    class_weight_used: str, sweep_cfg: dict, perm_cfg: dict,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float],
           List[Tuple[float, float, float, float, float]],
           List[Tuple[str, float]],
           List[Tuple[str, float, float]]]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance

    kw = dict(model_kw)
    if class_weight_used == "balanced":
        kw["class_weight"] = "balanced"
    clf = RandomForestClassifier(**kw)
    med = X.median(numeric_only=True)
    X_f = X.fillna(med)
    clf.fit(X_f, y)
    p = clf.predict_proba(X_f)[:, 1]

    base_success = float(y.mean())
    t_min = float(sweep_cfg["t_min"])
    t_max = float(sweep_cfg["t_max"])
    t_step = float(sweep_cfg["t_step"])
    excl_max = float(sweep_cfg["exclusion_max"])

    sweep: List[Tuple[float, float, float, float, float]] = []
    selected_t: Optional[float] = None
    excl_at_t: Optional[float] = None
    adm_at_t: Optional[float] = None
    real_at_t: Optional[float] = None
    lift_at_t: Optional[float] = None
    t = t_min
    while t <= t_max + 1e-9:
        admit_mask = p >= t
        admission = float(admit_mask.mean())
        exclusion = 1.0 - admission
        if admit_mask.any():
            real_succ = float(y[admit_mask].mean())
        else:
            real_succ = 0.0
        lift = (real_succ / base_success) if base_success > 0 else 0.0
        sweep.append((float(t), float(admission), float(exclusion), float(real_succ), float(lift)))
        if exclusion <= excl_max:
            # take the LARGEST t with exclusion <= cap (= last satisfying iteration).
            selected_t = float(t)
            excl_at_t = float(exclusion)
            adm_at_t = float(admission)
            real_at_t = float(real_succ)
            lift_at_t = float(lift)
        t += t_step

    # Gini importances (top 20).
    gini = sorted(zip(feature_cols, clf.feature_importances_), key=lambda x: -x[1])[:20]
    gini = [(n, float(v)) for n, v in gini]

    # Permutation importance (deterministic).
    perm = permutation_importance(
        clf, X_f, y, n_repeats=int(perm_cfg["n_repeats"]),
        random_state=int(perm_cfg["random_state"]), n_jobs=1, scoring="roc_auc",
    )
    perm_list = sorted(
        zip(feature_cols, perm.importances_mean, perm.importances_std),
        key=lambda x: -x[1],
    )[:20]
    perm_list = [(n, float(m), float(s)) for n, m, s in perm_list]

    return selected_t, excl_at_t, adm_at_t, real_at_t, lift_at_t, sweep, gini, perm_list


def evaluate_pipeline(
    unit_id: str, pipeline: str, feature_df: pd.DataFrame, feature_cols: List[str],
    y_by_tid: Dict[int, int], cfg: dict,
) -> PipelineResult:
    n = len(feature_df)
    # Order by entry_time ascending.
    fdf = feature_df.sort_values("entry_time").reset_index(drop=True)
    y = np.array([y_by_tid[int(tid)] for tid in fdf["trade_id"]], dtype=int)
    X = fdf[feature_cols].copy()
    base = float(y.mean())

    # Class weight.
    minority = min(base, 1.0 - base)
    if minority < float(cfg["model"]["class_weight_balanced_threshold"]):
        class_weight_used = "balanced"
    else:
        class_weight_used = "none"

    model_kw = dict(
        n_estimators=int(cfg["model"]["n_estimators"]),
        max_depth=int(cfg["model"]["max_depth"]),
        random_state=int(cfg["model"]["random_state"]),
        n_jobs=int(cfg["model"]["n_jobs"]),
    )

    fold_aucs, n_tr, n_te, fold_base = _cv_aucs(
        X, y, model_kw, class_weight_used, int(cfg["cv"]["n_splits"]),
    )
    valid_aucs = [a for a in fold_aucs if not math.isnan(a)]
    mean_auc = float(np.mean(valid_aucs)) if valid_aucs else float("nan")
    std_auc = float(np.std(valid_aucs, ddof=1)) if len(valid_aucs) >= 2 else 0.0

    gate_threshold = (
        float(cfg["gates"]["pipeline_e_auc_min"]) if pipeline == "E"
        else float(cfg["gates"]["pipeline_d1_auc_min"])
    )
    gate_pass = (not math.isnan(mean_auc)) and (mean_auc >= gate_threshold)

    sel_t, excl, adm, real, lift, sweep, gini, perm = (
        None, None, None, None, None, [], [], []
    )
    if gate_pass:
        sel_t, excl, adm, real, lift, sweep, gini, perm = _full_fit_and_threshold(
            X, y, feature_cols, model_kw, class_weight_used,
            cfg["threshold_sweep"], cfg["permutation_importance"],
        )

    return PipelineResult(
        unit_id=unit_id, pipeline=pipeline, n=n, n_features=len(feature_cols),
        base_success_rate=base,
        fold_aucs=fold_aucs, mean_auc=mean_auc, std_auc=std_auc,
        fold_n_train=n_tr, fold_n_test=n_te, fold_base_success=fold_base,
        gate_threshold=gate_threshold, gate_pass=gate_pass,
        selected_t=sel_t, exclusion_at_t=excl, admission_at_t=adm,
        realised_success_at_t=real, lift_at_t=lift,
        class_weight_used=class_weight_used,
        top_gini=gini, top_perm=perm, threshold_sweep=sweep,
    )


# ============================================================
# Output writers
# ============================================================


def _fmt(x: Any) -> str:
    if x is None:
        return ""
    try:
        xf = float(x)
        if not math.isfinite(xf):
            return ""
    except Exception:
        return str(x)
    return f"{xf:.10g}"


def _file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_extractability_summary(out_path: Path, results: List[PipelineResult]) -> None:
    cols = [
        "unit_id", "pipeline", "n", "n_features", "base_success_rate",
        "mean_auc", "std_auc", "fold_aucs",
        "gate_threshold", "gate_pass",
        "selected_t", "exclusion_at_t", "admission_at_t",
        "realised_success_at_t", "lift_at_t", "class_weight_used",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        for r in results:
            w.writerow([
                r.unit_id, r.pipeline, r.n, r.n_features,
                _fmt(r.base_success_rate),
                _fmt(r.mean_auc), _fmt(r.std_auc),
                ";".join(_fmt(a) for a in r.fold_aucs),
                _fmt(r.gate_threshold), "1" if r.gate_pass else "0",
                _fmt(r.selected_t), _fmt(r.exclusion_at_t), _fmt(r.admission_at_t),
                _fmt(r.realised_success_at_t), _fmt(r.lift_at_t),
                r.class_weight_used,
            ])


def write_fold_aucs(out_path: Path, r: PipelineResult) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["fold_idx", "auc", "n_train", "n_test", "base_success_rate"])
        for i, (a, ntr, nte, b) in enumerate(zip(r.fold_aucs, r.fold_n_train, r.fold_n_test, r.fold_base_success)):
            w.writerow([i, _fmt(a), ntr, nte, _fmt(b)])


def write_feature_importance(out_path: Path, r: PipelineResult) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["rank", "feature_gini", "gini_importance", "feature_perm", "perm_importance_mean", "perm_importance_std"])
        for i in range(20):
            gini = r.top_gini[i] if i < len(r.top_gini) else ("", float("nan"))
            perm = r.top_perm[i] if i < len(r.top_perm) else ("", float("nan"), float("nan"))
            w.writerow([i, gini[0], _fmt(gini[1]), perm[0], _fmt(perm[1]), _fmt(perm[2])])


def write_threshold_sweep(out_path: Path, r: PipelineResult) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["t", "admission", "exclusion", "realised_success", "lift"])
        for row in r.threshold_sweep:
            w.writerow([_fmt(row[0]), _fmt(row[1]), _fmt(row[2]), _fmt(row[3]), _fmt(row[4])])


def write_routing(out_path: Path, results: List[PipelineResult]) -> None:
    cols = ["unit_id", "pipeline_E_pass", "pipeline_D1_pass", "routing_decision"]
    by_unit: Dict[str, Dict[str, PipelineResult]] = {}
    for r in results:
        by_unit.setdefault(r.unit_id, {})[r.pipeline] = r
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        for unit_id, plr in by_unit.items():
            e_pass = plr.get("E").gate_pass if "E" in plr else False
            d1_pass = plr.get("D1").gate_pass if "D1" in plr else False
            if e_pass and d1_pass:
                decision = "both"
            elif e_pass:
                decision = "E"
            elif d1_pass:
                decision = "D1"
            else:
                decision = "none"
            w.writerow([unit_id, "1" if e_pass else "0", "1" if d1_pass else "0", decision])


def write_d1_lag_audit(out_path: Path, audit_rows: List[Dict[str, Any]]) -> None:
    cols = ["trade_id", "pair", "entry_time", "signal_bar_time",
            "expected_d1_date_max", "d1_date_joined", "lag_correct"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        for row in audit_rows:
            w.writerow([
                row["trade_id"], row["pair"], row["entry_time"],
                row["signal_bar_time"], row["expected_d1_date_max"],
                row["d1_date_joined"], "1" if row["lag_correct"] else "0",
            ])


# ============================================================
# Summary doc
# ============================================================


def write_summary_md(
    out_path: Path, results: List[PipelineResult], audit_rows: List[Dict[str, Any]],
    cfg: dict, hashes1: Dict[str, str], hashes2: Optional[Dict[str, str]],
    det_gate: str, base_success_by_unit: Dict[str, float],
    success_count_by_unit: Dict[str, int], unit_n: Dict[str, int],
    class_weight_by_unit: Dict[str, str],
) -> None:
    lines: List[str] = []
    lines.append("# Arc 7 — Step 4 extractability summary")
    lines.append("")
    lines.append("Protocol: `L_ARC_PROTOCOL.md` v2.1.2 §§8, 10, 17")
    lines.append("")

    any_pass = any(r.gate_pass and r.selected_t is not None for r in results)
    lines.append("## Verdict")
    if any_pass:
        passing = [(r.unit_id, r.pipeline) for r in results if r.gate_pass and r.selected_t is not None]
        descr = "; ".join(f"{u}/{p}" for u, p in passing)
        lines.append(f"**PASS** — {len(passing)} (unit, pipeline) pair(s) clear §8 gate with valid admission t: {descr}.")
    else:
        lines.append("**FAIL — CLEAN-NULL at Step 4.** Zero (unit, pipeline) pairs clear §8 gate with a valid admission threshold.")
    lines.append("")

    # Pipeline summary table
    lines.append("## Pipeline summary")
    lines.append("")
    lines.append("| unit | pipeline | n | n_feat | mean AUC | std AUC | gate | pass? | selected t | exclusion | admission | lift | realised_success |")
    lines.append("|---|---|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|")
    for r in results:
        lines.append(
            f"| {r.unit_id} | {r.pipeline} | {r.n} | {r.n_features} "
            f"| {_fmt(r.mean_auc)} | {_fmt(r.std_auc)} | {r.gate_threshold:.2f} "
            f"| {'YES' if r.gate_pass else 'no'} "
            f"| {_fmt(r.selected_t) or '—'} | {_fmt(r.exclusion_at_t) or '—'} "
            f"| {_fmt(r.admission_at_t) or '—'} | {_fmt(r.lift_at_t) or '—'} "
            f"| {_fmt(r.realised_success_at_t) or '—'} |"
        )
    lines.append("")

    # Per-fold AUCs
    lines.append("## Per-fold AUCs")
    lines.append("")
    for r in results:
        lines.append(f"### {r.unit_id} / {r.pipeline}")
        lines.append("")
        lines.append("| fold | AUC | n_train | n_test | base_success |")
        lines.append("|---:|---:|---:|---:|---:|")
        for i, (a, ntr, nte, b) in enumerate(zip(r.fold_aucs, r.fold_n_train, r.fold_n_test, r.fold_base_success)):
            lines.append(f"| {i} | {_fmt(a)} | {ntr} | {nte} | {_fmt(b)} |")
        lines.append("")

    # Routing
    lines.append("## Routing per unit")
    lines.append("")
    lines.append("| unit | E pass | D1 pass | route(s) carried |")
    lines.append("|---|:---:|:---:|---|")
    by_unit: Dict[str, Dict[str, PipelineResult]] = {}
    for r in results:
        by_unit.setdefault(r.unit_id, {})[r.pipeline] = r
    for unit_id, plr in by_unit.items():
        e_pass = plr.get("E").gate_pass if "E" in plr else False
        d1_pass = plr.get("D1").gate_pass if "D1" in plr else False
        route = "both" if (e_pass and d1_pass) else ("E" if e_pass else ("D1" if d1_pass else "none"))
        lines.append(f"| {unit_id} | {'yes' if e_pass else 'no'} | {'yes' if d1_pass else 'no'} | {route} |")
    lines.append("")

    # D1 lag audit
    lines.append("## D1 lag audit")
    lines.append("")
    all_correct = all(r["lag_correct"] for r in audit_rows)
    lines.append(f"{len(audit_rows)} spot-checks performed (5 random trades × 3 units). All correct: **{'YES' if all_correct else 'NO'}**.")
    lines.append("")
    lines.append("| trade_id | pair | entry_time | signal_bar | expected ≤ | d1 joined | correct |")
    lines.append("|---:|---|---|---|---|---|:---:|")
    for r in audit_rows:
        lines.append(
            f"| {r['trade_id']} | {r['pair']} | {r['entry_time']} | {r['signal_bar_time']} "
            f"| {r['expected_d1_date_max']} | {r['d1_date_joined']} | {'YES' if r['lag_correct'] else 'NO'} |"
        )
    lines.append("")

    # Top features per surviving pipeline
    lines.append("## Top features per surviving (unit, pipeline)")
    lines.append("")
    survivors = [r for r in results if r.gate_pass]
    if not survivors:
        lines.append("_None surviving._")
    for r in survivors:
        lines.append(f"### {r.unit_id} / {r.pipeline}")
        lines.append("")
        lines.append("**Top 10 by gini:**")
        for n, v in r.top_gini[:10]:
            lines.append(f"- `{n}` = {v:.4f}")
        lines.append("")
        lines.append("**Top 10 by permutation:**")
        for n, m, s in r.top_perm[:10]:
            lines.append(f"- `{n}` = {m:.4f} (std {s:.4f})")
        lines.append("")

    # Class balance
    lines.append("## Class balance per unit")
    lines.append("")
    lines.append("| unit | n | success_count | success_rate | class_weight_used |")
    lines.append("|---|---:|---:|---:|---|")
    for unit_id in unit_n:
        n = unit_n[unit_id]
        sc = success_count_by_unit[unit_id]
        sr = base_success_by_unit[unit_id]
        cw = class_weight_by_unit[unit_id]
        lines.append(f"| {unit_id} | {n} | {sc} | {_fmt(sr)} | {cw} |")
    lines.append("")

    # Determinism
    lines.append("## Determinism")
    lines.append("")
    lines.append(f"**Gate: {det_gate}**")
    lines.append("")
    if hashes2 is not None:
        lines.append("| File | run 1 sha256 | run 2 sha256 | match |")
        lines.append("|---|---|---|:---:|")
        for name in sorted(hashes1.keys()):
            h1 = hashes1[name]
            h2 = hashes2.get(name, "(missing)")
            match = "YES" if h1 == h2 else "NO"
            lines.append(f"| `{name}` | `{h1[:16]}…` | `{h2[:16]}…` | {match} |")
    lines.append("")

    # Kill reasons
    lines.append("## Kill reasons (per non-passing unit × pipeline)")
    lines.append("")
    failures = [r for r in results if not (r.gate_pass and r.selected_t is not None)]
    if not failures:
        lines.append("None — all (unit, pipeline) pairs pass.")
    else:
        for r in failures:
            margin = r.mean_auc - r.gate_threshold if not math.isnan(r.mean_auc) else float("nan")
            note = ""
            if r.gate_pass and r.selected_t is None:
                note = " — AUC cleared gate, but no threshold t satisfies exclusion ≤ 0.30 (probabilities too compressed)."
            lines.append(
                f"- `{r.unit_id}/{r.pipeline}`: mean AUC {_fmt(r.mean_auc)} vs gate {r.gate_threshold:.2f} (margin {_fmt(margin)}).{note}"
            )
    lines.append("")

    # Files
    lines.append("## Files")
    lines.append("")
    out_dir = Path(cfg["output"]["results_dir"])
    for name in sorted(hashes1.keys()):
        lines.append(f"- `{out_dir}/{name}`")
    lines.append(f"- `{out_dir}/STEP4_SUMMARY.md`")
    lines.append("- `configs/arc_7/step4.yaml`")
    lines.append("- `scripts/arc_7/step4_extractability.py`")
    lines.append("")

    lines.append("## Step 4 commit")
    lines.append("hash: _pending_")
    lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================================================
# Driver
# ============================================================


def _run_once(cfg: dict) -> Tuple[Dict[str, str], Dict[str, Any]]:
    in_cfg = cfg["input"]
    out_cfg = cfg["output"]
    step1_dir = _REPO_ROOT / in_cfg["step1_dir"]
    step2_dir = _REPO_ROOT / in_cfg["step2_dir"]
    out_dir = _REPO_ROOT / out_cfg["results_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    trades_df = pd.read_csv(step1_dir / in_cfg["trades_csv"])
    paths_df = pd.read_csv(step1_dir / in_cfg["paths_csv"])
    clusters_df = pd.read_csv(step2_dir / in_cfg["clusters_csv"])

    paths_index = _build_paths_index(paths_df)

    # Pair caches.
    pairs = sorted(trades_df["pair"].astype(str).unique())
    pair_caches: Dict[str, PerPairCache] = {}
    for p in pairs:
        print(f"[arc_7 step4] caching {p}", file=sys.stderr)
        pair_caches[p] = _build_pair_cache(p, cfg["data"]["dir_4h"], cfg["data"]["dir_d1"])

    # Features.
    print("[arc_7 step4] computing pipeline E features", file=sys.stderr)
    e_features = compute_pipeline_e_features(trades_df, pair_caches)
    print("[arc_7 step4] computing pipeline D1 features (with lag audit)", file=sys.stderr)
    d1_features, d1_audit_full = compute_pipeline_d1_features(trades_df, pair_caches)

    # D1 lag audit subset: 5 random trades per unit × 3 units.
    rng = np.random.default_rng(int(cfg["d1_lag_audit"]["random_seed"]))
    audit_subset: List[Dict[str, Any]] = []
    for u in cfg["units"]:
        cid_set = set(int(c) for c in u["cluster_ids"])
        tids = sorted(clusters_df[clusters_df["cluster_id"].isin(cid_set)]["trade_id"].astype(int).tolist())
        if len(tids) < int(cfg["d1_lag_audit"]["n_random_trades_per_unit"]):
            chosen = tids
        else:
            chosen = sorted(rng.choice(tids, size=int(cfg["d1_lag_audit"]["n_random_trades_per_unit"]), replace=False).tolist())
        for tid in chosen:
            row = d1_audit_full[d1_audit_full["trade_id"] == tid].iloc[0].to_dict()
            row["unit_id"] = u["unit_id"]
            audit_subset.append(row)
    if not all(r["lag_correct"] for r in audit_subset):
        raise SystemExit("D1 lag audit FAILED — halt per dispatch (lookahead risk).")

    # Per-unit evaluation.
    results: List[PipelineResult] = []
    base_success_by_unit: Dict[str, float] = {}
    success_count_by_unit: Dict[str, int] = {}
    unit_n: Dict[str, int] = {}
    class_weight_by_unit: Dict[str, str] = {}

    for u in cfg["units"]:
        unit_id = str(u["unit_id"])
        cid_set = set(int(c) for c in u["cluster_ids"])
        sl_mult = float(u["selected_sl_atr_mult"])
        tids = sorted(clusters_df[clusters_df["cluster_id"].isin(cid_set)]["trade_id"].astype(int).tolist())
        y_by_tid = compute_success_labels(
            tids, paths_index, sl_mult, float(cfg["original_sl_atr_mult"])
        )
        unit_n[unit_id] = len(tids)
        success_count_by_unit[unit_id] = int(sum(y_by_tid.values()))
        base_success_by_unit[unit_id] = float(success_count_by_unit[unit_id] / max(len(tids), 1))

        # Pipeline E.
        e_sub = e_features[e_features["trade_id"].isin(tids)].copy()
        r_e = evaluate_pipeline(unit_id, "E", e_sub, PIPELINE_E_FEATURES, y_by_tid, cfg)
        # Pipeline D1.
        d1_sub = d1_features[d1_features["trade_id"].isin(tids)].copy()
        # D1 features need entry_time for ordering; merge from e_sub.
        d1_sub = d1_sub.merge(e_sub[["trade_id", "entry_time"]], on="trade_id", how="left")
        r_d1 = evaluate_pipeline(unit_id, "D1", d1_sub, PIPELINE_D1_FEATURES, y_by_tid, cfg)

        class_weight_by_unit[unit_id] = r_e.class_weight_used   # E and D1 use same logic
        results.append(r_e)
        results.append(r_d1)

    # Write outputs.
    out_cfg = cfg["output"]
    summary_path = out_dir / "extractability_summary.csv"
    write_extractability_summary(summary_path, results)
    routing_path = out_dir / "pipeline_routing.csv"
    write_routing(routing_path, results)
    audit_path = out_dir / "d1_lag_audit.csv"
    write_d1_lag_audit(audit_path, audit_subset)

    hashes: Dict[str, str] = {}
    hashes["extractability_summary.csv"] = _file_sha256(summary_path)
    hashes["pipeline_routing.csv"] = _file_sha256(routing_path)
    hashes["d1_lag_audit.csv"] = _file_sha256(audit_path)
    for r in results:
        fa_path = out_dir / f"fold_aucs_{r.unit_id}_{r.pipeline}.csv"
        write_fold_aucs(fa_path, r)
        hashes[fa_path.name] = _file_sha256(fa_path)
        if r.gate_pass:
            fi_path = out_dir / f"feature_importance_{r.unit_id}_{r.pipeline}.csv"
            write_feature_importance(fi_path, r)
            hashes[fi_path.name] = _file_sha256(fi_path)
            ts_path = out_dir / f"threshold_sweep_{r.unit_id}_{r.pipeline}.csv"
            write_threshold_sweep(ts_path, r)
            hashes[ts_path.name] = _file_sha256(ts_path)

    ctx = {
        "results": results,
        "audit_subset": audit_subset,
        "base_success_by_unit": base_success_by_unit,
        "success_count_by_unit": success_count_by_unit,
        "unit_n": unit_n,
        "class_weight_by_unit": class_weight_by_unit,
        "out_dir": out_dir,
    }
    return hashes, ctx


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Arc 7 Step 4 — extractability.")
    ap.add_argument("-c", "--config", type=Path, default=Path("configs/arc_7/step4.yaml"))
    args = ap.parse_args(argv)
    cfg_path = args.config
    if not cfg_path.is_absolute():
        cfg_path = (_REPO_ROOT / cfg_path).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    print("[arc_7 step4] === RUN 1 ===", file=sys.stderr)
    hashes1, ctx = _run_once(cfg)
    hashes2 = None
    if bool(cfg["output"].get("determinism_check", True)):
        print("[arc_7 step4] === RUN 2 (determinism) ===", file=sys.stderr)
        hashes2, _ = _run_once(cfg)

    if hashes2 is None:
        det_gate = "N/A"
    else:
        det_gate = "PASS" if all(hashes1[k] == hashes2.get(k) for k in hashes1) else "FAIL"

    summary_path = ctx["out_dir"] / cfg["output"]["summary_md"]
    write_summary_md(
        summary_path, ctx["results"], ctx["audit_subset"], cfg, hashes1, hashes2, det_gate,
        ctx["base_success_by_unit"], ctx["success_count_by_unit"], ctx["unit_n"],
        ctx["class_weight_by_unit"],
    )

    n_pass = sum(1 for r in ctx["results"] if r.gate_pass and r.selected_t is not None)
    print(f"[arc_7 step4] DONE. survivors={n_pass}, det={det_gate}", file=sys.stderr)
    return 0 if n_pass > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
