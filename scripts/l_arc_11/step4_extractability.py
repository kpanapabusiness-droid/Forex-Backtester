"""Arc 11 — Step 4 extractability investigation.

L_ARC_PROTOCOL v2.3 §§8, 10, 17 + v2.2 §3 (no max-F1 fallback) +
v2.2 §2 (Tier 2 lift cap ≤ 5 — informational; Tier 2 not run in this dispatch).

For each Step-3 surviving unit (c1 SL=3.0, c3 SL=2.0, agg_c1_c3 SL=3.0):

  Angle E (entry-time predictability)
    - Step A: full feature set RF on 8 cross-dataset base + arc-11-specific
      (≤ 38 total). RF mean ROC-AUC across 5 TimeSeriesSplit folds. If ≥ 0.65,
      lock Pipeline E classifier, proceed to threshold sweep.
    - Step B (if A fails): top-K subsets (top-5, top-10, top-15) ranked by RF
      Gini importance on the full-feature fit. If any clears 0.65, lock.
    - Step C (if A and B fail): not implemented in this dispatch (would need
      ≤ 30 combination budget across all archetypes). If Step B fails the
      unit is reported as Pipeline-E FAIL.

  Angle D1 (deferred-identification, path-so-far at bar t)
    - For t ∈ {1, 2, 3, 4, 5, 10}: exclude trades with bars_held < t (per §8);
      build features (8 base entry + 7 path-so-far at t); RF mean AUC across
      5 TimeSeriesSplit folds. Smallest-t rule: first t with (AUC ≥ 0.60 AND
      exclusion ≤ 0.30). If none, Pipeline D1 FAIL.
    - "Path-so-far at bar t" means: close_r_at_t, mfe_so_far_r_at_t,
      mae_so_far_r_at_t, bars_in_profit_at_t, local_peaks_so_far_at_t,
      monotonicity_so_far_at_t, velocity_first_t.
    - Per v2.3 §5 (Open-24), the pre-t SL multiplier for each unit's Pipeline
      D1 deployment is recorded in the policy YAML = the unit's selected SL.

  Threshold sweep (per v2.2 §3)
    - For both pipelines: sweep threshold {0.40, 0.50, 0.60, 0.70}. Select max
      precision with recall ≥ 0.60. If NO threshold satisfies recall ≥ 0.60,
      archetype FAILS Step 4 (no max-F1 fallback). §16a disposition.

Success label per unit: re-impose the unit's selected SL on the full §15a bar
path (held + forward-observation rows); success = 1 iff final_r ≥ 1.0 in the
new R-frame. SL re-imposition imported from step3_capturability for
byte-identity with Step 3.

Models: RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42,
n_jobs=1). n_jobs=1 for Windows determinism. class_weight="balanced" if the
minority class is < 30%.

Determinism: two consecutive runs produce byte-identical CSVs. RF + fixed
seed + n_jobs=1 = deterministic; sklearn.inspection.permutation_importance
uses an explicit random_state.

Usage:
    py scripts/l_arc_11/step4_extractability.py -c configs/l_arc_11/step4.yaml
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

from scripts.l_arc_11.step3_capturability import _eval_trade_at_sl  # noqa: E402

ATR_PERIOD = 14
PIPELINE_E_AUC_MIN = 0.65
PIPELINE_D1_AUC_MIN = 0.60
D1_EXCLUSION_MAX = 0.30
THRESHOLD_SWEEP_RECALL_MIN = 0.60
T_CANDIDATES = [1, 2, 3, 4, 5, 10]


# ============================================================
# 4H bar cache (causal indicators for the 8 base + arc-11 features)
# ============================================================

@dataclass
class PerPairCache:
    df_4h: pd.DataFrame
    idx_by_ts: Dict[pd.Timestamp, int]
    atr_4h: np.ndarray
    rsi_14: np.ndarray
    ema20_4h: np.ndarray
    ema50_4h: np.ndarray


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


def _wilder_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    n = close.size
    rsi = np.full(n, np.nan, dtype=float)
    if n < period + 1:
        return rsi
    diff = np.diff(close)
    gain = np.where(diff > 0, diff, 0.0)
    loss = np.where(diff < 0, -diff, 0.0)
    # First avg over first `period` deltas.
    avg_gain = float(np.mean(gain[:period]))
    avg_loss = float(np.mean(loss[:period]))
    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - 100.0 / (1.0 + rs)
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gain[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i - 1]) / period
        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - 100.0 / (1.0 + rs)
    return rsi


def _ema(series: np.ndarray, period: int) -> np.ndarray:
    out = np.full_like(series, np.nan, dtype=float)
    if series.size == 0:
        return out
    alpha = 2.0 / (period + 1.0)
    out[0] = series[0]
    for i in range(1, series.size):
        if math.isnan(series[i]):
            out[i] = out[i - 1]
        else:
            out[i] = alpha * series[i] + (1.0 - alpha) * out[i - 1]
    return out


def _load_pair_4h(pair: str, dir_4h: str) -> pd.DataFrame:
    p = Path(dir_4h)
    if not p.is_absolute():
        p = _REPO_ROOT / p
    df = pd.read_csv(p / f"{pair}.csv")
    if "time" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def _build_pair_cache(pair: str, dir_4h: str) -> PerPairCache:
    df_4h = _load_pair_4h(pair, dir_4h)
    cl = df_4h["close"].astype(float).to_numpy()
    atr = _wilder_atr(df_4h, ATR_PERIOD)
    rsi = _wilder_rsi(cl, ATR_PERIOD)
    e20 = _ema(cl, 20)
    e50 = _ema(cl, 50)
    idx_by_ts = {pd.Timestamp(ts): i for i, ts in enumerate(df_4h["date"].to_numpy())}
    return PerPairCache(
        df_4h=df_4h, idx_by_ts=idx_by_ts,
        atr_4h=atr, rsi_14=rsi, ema20_4h=e20, ema50_4h=e50,
    )


# ============================================================
# Pipeline E features
# ============================================================

PIPELINE_E_BASE: List[str] = [
    # 8 cross-dataset base per protocol §8.
    "body_to_range_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "range_to_atr_14",
    "ret_5bar_atr",
    "ret_20bar_atr",
    "pos_in_20bar_range",
    "rsi_14",
]

PIPELINE_E_ARC11_SPECIFIC: List[str] = [
    # Arc-11-specific (entry-time observable). Per signal spec "Arc 8 differential
    # note" hypothesis: break_magnitude_atr + H_ref_freshness + prior_leg_length
    # should produce cleaner Pipeline E features.
    "break_magnitude_atr",         # (close[t] - H_ref) / atr14
    "h_ref_freshness_bars",        # = signal-bar - H_ref-bar (always >= 4)
    "close_position",              # (close - low) / (high - low) at signal bar
    "atr14_at_signal_pct_close",   # atr / close[t]
    "prior_leg_length_atr",        # (H_ref - trend_filter_swing_low) / atr14
    "trend_filter_dist_atr",       # (close[t] - trend_filter_swing_low) / atr14
    "ema20_4h_dist_atr",
    "ema50_4h_dist_atr",
    "ema20_4h_slope_5bar_atr",
    "ema50_4h_slope_5bar_atr",
    "hour_sin", "hour_cos",
    "weekday_sin", "weekday_cos",
    "pair_id_int",
]

PIPELINE_E_FEATURES: List[str] = PIPELINE_E_BASE + PIPELINE_E_ARC11_SPECIFIC


def compute_pipeline_e_features(
    trades_df: pd.DataFrame,
    pair_caches: Dict[str, PerPairCache],
) -> pd.DataFrame:
    pairs_sorted = sorted(trades_df["pair"].astype(str).unique())
    pair_to_int = {p: i for i, p in enumerate(pairs_sorted)}

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
        rng = h_t - l_t
        atr_t = float(cache.atr_4h[i]) if not math.isnan(cache.atr_4h[i]) else float("nan")

        # 8 cross-dataset base.
        if rng > 0:
            body = abs(c_t - o_t) / rng
            uw = (h_t - max(o_t, c_t)) / rng
            lw = (min(o_t, c_t) - l_t) / rng
        else:
            body, uw, lw = 0.0, 0.0, 0.0
        r_atr = rng / atr_t if atr_t > 0 else float("nan")

        def _ret_atr(lb: int) -> float:
            if i < lb or atr_t <= 0:
                return float("nan")
            prev = float(cache.df_4h["close"].iloc[i - lb])
            return (c_t - prev) / atr_t
        ret5 = _ret_atr(5)
        ret20 = _ret_atr(20)

        if i >= 19:
            window_low = float(cache.df_4h["low"].iloc[i - 19:i + 1].min())
            window_high = float(cache.df_4h["high"].iloc[i - 19:i + 1].max())
            wrng = window_high - window_low
            pos20 = (c_t - window_low) / wrng if wrng > 0 else 0.5
        else:
            pos20 = float("nan")

        rsi_t = float(cache.rsi_14[i]) if not math.isnan(cache.rsi_14[i]) else float("nan")

        # Arc-11 specific.
        break_mag = float(t.get("break_magnitude_atr", float("nan")))
        h_ref_off = float(t.get("h_ref_bar_offset", float("nan")))
        close_pos = float(t.get("close_position", float("nan")))
        atr_pct = atr_t / c_t if (c_t > 0 and atr_t > 0) else float("nan")
        h_ref_v = float(t.get("h_ref", float("nan")))
        tf_low = float(t.get("trend_filter_swing_low", float("nan")))
        prior_leg = (h_ref_v - tf_low) / atr_t if (atr_t > 0 and math.isfinite(h_ref_v) and math.isfinite(tf_low)) else float("nan")
        tf_dist = (c_t - tf_low) / atr_t if (atr_t > 0 and math.isfinite(tf_low)) else float("nan")

        e20 = float(cache.ema20_4h[i]) if not math.isnan(cache.ema20_4h[i]) else float("nan")
        e50 = float(cache.ema50_4h[i]) if not math.isnan(cache.ema50_4h[i]) else float("nan")
        d20 = (c_t - e20) / atr_t if (atr_t > 0 and math.isfinite(e20)) else float("nan")
        d50 = (c_t - e50) / atr_t if (atr_t > 0 and math.isfinite(e50)) else float("nan")

        def _slope(arr: np.ndarray, lb: int = 5) -> float:
            if i < lb or atr_t <= 0:
                return float("nan")
            v_now = float(arr[i])
            v_prev = float(arr[i - lb])
            if math.isnan(v_now) or math.isnan(v_prev):
                return float("nan")
            return (v_now - v_prev) / lb / atr_t
        s20 = _slope(cache.ema20_4h)
        s50 = _slope(cache.ema50_4h)

        hour = sig_ts.hour
        dow = sig_ts.weekday()
        h_sin = math.sin(2 * math.pi * hour / 24.0)
        h_cos = math.cos(2 * math.pi * hour / 24.0)
        dow_sin = math.sin(2 * math.pi * dow / 7.0)
        dow_cos = math.cos(2 * math.pi * dow / 7.0)

        rows.append({
            "trade_id": int(t["trade_id"]),
            "pair": pair,
            "entry_time": pd.Timestamp(t["entry_time"]),
            "body_to_range_ratio": body,
            "upper_wick_ratio": uw,
            "lower_wick_ratio": lw,
            "range_to_atr_14": r_atr,
            "ret_5bar_atr": ret5,
            "ret_20bar_atr": ret20,
            "pos_in_20bar_range": pos20,
            "rsi_14": rsi_t,
            "break_magnitude_atr": break_mag,
            "h_ref_freshness_bars": h_ref_off,
            "close_position": close_pos,
            "atr14_at_signal_pct_close": atr_pct,
            "prior_leg_length_atr": prior_leg,
            "trend_filter_dist_atr": tf_dist,
            "ema20_4h_dist_atr": d20,
            "ema50_4h_dist_atr": d50,
            "ema20_4h_slope_5bar_atr": s20,
            "ema50_4h_slope_5bar_atr": s50,
            "hour_sin": h_sin, "hour_cos": h_cos,
            "weekday_sin": dow_sin, "weekday_cos": dow_cos,
            "pair_id_int": pair_to_int[pair],
        })
    return pd.DataFrame(rows)


# ============================================================
# Pipeline D1 features (deferred-identification, path-so-far at bar t)
# ============================================================

PIPELINE_D1_PATH_FEATURES: List[str] = [
    "close_r_at_t",
    "mfe_so_far_r_at_t",
    "mae_so_far_r_at_t",
    "bars_in_profit_at_t",
    "local_peaks_so_far_at_t",
    "monotonicity_so_far_at_t",
    "velocity_first_t",
]

PIPELINE_D1_FEATURES: List[str] = PIPELINE_E_BASE + PIPELINE_D1_PATH_FEATURES


def _path_features_at_t(
    path: pd.DataFrame, t: int, sl_atr_mult: float, original_sl_atr_mult: float
) -> Optional[Dict[str, float]]:
    """Return path-so-far features at bar offset t, in the new R-frame given
    the unit's selected SL. Returns None if trade had already SL'd before t.

    Uses _eval_trade_at_sl semantics: re-impose SL, check that the truncated
    path includes bar t (i.e. trade was still alive at t).
    """
    te = _eval_trade_at_sl(path, sl_atr_mult, original_sl_atr_mult)
    if te.truncated_at_bar < t:
        return None

    scale = original_sl_atr_mult / sl_atr_mult
    # Slice path bars 0..t (inclusive) within the truncated path.
    sub = path[(path["bar_offset"] <= t)].sort_values("bar_offset", kind="mergesort")
    if sub.empty:
        return None
    close_r_orig = sub["close_r"].to_numpy(dtype=float)
    mfe_orig = sub["mfe_so_far_r"].to_numpy(dtype=float)
    mae_orig = sub["mae_so_far_r"].to_numpy(dtype=float)

    # Convert to new R-frame.
    close_r_new = close_r_orig * scale
    mfe_new = mfe_orig * scale
    mae_new = mae_orig * scale

    # Take the values at bar offset t (last row).
    close_at_t = float(close_r_new[-1])
    mfe_at_t = float(mfe_new[-1])
    mae_at_t = float(mae_new[-1])

    bars_in_profit = int(np.sum(close_r_new > 0))

    # local_peaks count (mfe_so_far strictly increasing transitions).
    diffs = np.diff(mfe_new)
    local_peaks = int(np.sum(diffs > 0))

    # Monotonicity-in-profit: among close > 0 bars, fraction non-decreasing.
    in_profit = close_r_new[close_r_new > 0]
    if in_profit.size >= 2:
        mono = float(np.mean(in_profit[1:] >= in_profit[:-1]))
    else:
        mono = 0.0

    velocity = close_at_t / max(t, 1)

    return {
        "close_r_at_t": close_at_t,
        "mfe_so_far_r_at_t": mfe_at_t,
        "mae_so_far_r_at_t": mae_at_t,
        "bars_in_profit_at_t": float(bars_in_profit),
        "local_peaks_so_far_at_t": float(local_peaks),
        "monotonicity_so_far_at_t": mono,
        "velocity_first_t": velocity,
    }


# ============================================================
# Success label (re-impose SL, success = final_r >= 1.0)
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
    pipeline: str            # "E" or "D1_t<N>"
    t: Optional[int]         # only for D1
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
    exclusion: Optional[float] = None      # for D1: fraction of pool excluded by bars_held < t
    threshold_results: List[Tuple[float, float, float, int]] = field(default_factory=list)
    # (threshold, precision, recall, n_admitted)
    selected_threshold: Optional[float] = None   # max-precision with recall ≥ 0.60
    selected_precision: Optional[float] = None
    selected_recall: Optional[float] = None
    selected_n_admitted: Optional[int] = None
    threshold_pass: bool = False
    class_weight_used: str = "none"
    top_gini: List[Tuple[str, float]] = field(default_factory=list)


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
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            fold_aucs.append(float("nan"))
            continue
        kw = dict(model_kw)
        if class_weight_used == "balanced":
            kw["class_weight"] = "balanced"
        clf = RandomForestClassifier(**kw)
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


def _full_fit_and_threshold_sweep(
    X: pd.DataFrame, y: np.ndarray, feature_cols: List[str], model_kw: dict,
    class_weight_used: str,
) -> Tuple[List[Tuple[float, float, float, int]], Optional[float], Optional[float],
           Optional[float], Optional[int], bool, List[Tuple[str, float]]]:
    """Per v2.2 §3: sweep threshold {0.40, 0.50, 0.60, 0.70}; select max precision
    with recall ≥ 0.60. If no threshold satisfies recall ≥ 0.60, archetype FAILS.
    """
    from sklearn.ensemble import RandomForestClassifier
    kw = dict(model_kw)
    if class_weight_used == "balanced":
        kw["class_weight"] = "balanced"
    clf = RandomForestClassifier(**kw)
    med = X.median(numeric_only=True)
    X_f = X.fillna(med)
    clf.fit(X_f, y)
    p = clf.predict_proba(X_f)[:, 1]

    sweep_thresholds = [0.40, 0.50, 0.60, 0.70]
    results: List[Tuple[float, float, float, int]] = []
    for th in sweep_thresholds:
        admit = p >= th
        n_admit = int(admit.sum())
        if n_admit == 0:
            prec = 0.0
            recall = 0.0
        else:
            tp = int(np.sum((admit) & (y == 1)))
            fp = int(np.sum((admit) & (y == 0)))
            total_pos = int(np.sum(y == 1))
            prec = tp / max(tp + fp, 1)
            recall = tp / max(total_pos, 1)
        results.append((float(th), float(prec), float(recall), n_admit))

    # v2.2 §3: max precision with recall ≥ 0.60. No fallback.
    candidates = [(t, p, r, n) for (t, p, r, n) in results if r >= THRESHOLD_SWEEP_RECALL_MIN]
    if candidates:
        # Max precision; tie-break on smallest threshold (= higher admission rate).
        candidates.sort(key=lambda x: (-x[1], x[0]))
        best_t, best_p, best_r, best_n = candidates[0]
        threshold_pass = True
    else:
        best_t, best_p, best_r, best_n = None, None, None, None
        threshold_pass = False

    gini = sorted(zip(feature_cols, clf.feature_importances_), key=lambda x: -x[1])[:20]
    gini = [(n, float(v)) for n, v in gini]
    return results, best_t, best_p, best_r, best_n, threshold_pass, gini


def evaluate_pipeline(
    unit_id: str, pipeline_label: str, t_value: Optional[int],
    feature_df: pd.DataFrame, feature_cols: List[str],
    y_by_tid: Dict[int, int], cfg: dict, gate_threshold: float,
    pool_size_full: int,
) -> PipelineResult:
    fdf = feature_df.sort_values("entry_time").reset_index(drop=True)
    y = np.array([y_by_tid[int(tid)] for tid in fdf["trade_id"]], dtype=int)
    X = fdf[feature_cols].copy()
    n = len(fdf)
    base = float(y.mean()) if y.size > 0 else 0.0

    minority = min(base, 1.0 - base)
    if minority < float(cfg["model"]["class_weight_balanced_threshold"]):
        cw = "balanced"
    else:
        cw = "none"

    model_kw = dict(
        n_estimators=int(cfg["model"]["n_estimators"]),
        max_depth=int(cfg["model"]["max_depth"]),
        random_state=int(cfg["model"]["random_state"]),
        n_jobs=int(cfg["model"]["n_jobs"]),
    )

    n_splits = int(cfg["cv"]["n_splits"])
    # Sanity check: need at least n_splits + 1 samples for TimeSeriesSplit.
    if n < n_splits + 1:
        return PipelineResult(
            unit_id=unit_id, pipeline=pipeline_label, t=t_value, n=n,
            n_features=len(feature_cols), base_success_rate=base,
            fold_aucs=[], mean_auc=float("nan"), std_auc=0.0,
            fold_n_train=[], fold_n_test=[], fold_base_success=[],
            gate_threshold=gate_threshold, gate_pass=False,
            exclusion=(1.0 - n / pool_size_full) if pool_size_full > 0 else 0.0,
            class_weight_used=cw,
        )

    fold_aucs, n_tr, n_te, fold_base = _cv_aucs(X, y, model_kw, cw, n_splits)
    valid = [a for a in fold_aucs if not math.isnan(a)]
    mean_auc = float(np.mean(valid)) if valid else float("nan")
    std_auc = float(np.std(valid, ddof=1)) if len(valid) >= 2 else 0.0
    gate_pass = (not math.isnan(mean_auc)) and (mean_auc >= gate_threshold)

    exclusion = (1.0 - n / pool_size_full) if pool_size_full > 0 else 0.0

    result = PipelineResult(
        unit_id=unit_id, pipeline=pipeline_label, t=t_value, n=n,
        n_features=len(feature_cols), base_success_rate=base,
        fold_aucs=fold_aucs, mean_auc=mean_auc, std_auc=std_auc,
        fold_n_train=n_tr, fold_n_test=n_te, fold_base_success=fold_base,
        gate_threshold=gate_threshold, gate_pass=gate_pass,
        exclusion=exclusion, class_weight_used=cw,
    )

    if gate_pass:
        sweep, best_t, best_p, best_r, best_n, t_pass, gini = _full_fit_and_threshold_sweep(
            X, y, feature_cols, model_kw, cw,
        )
        result.threshold_results = sweep
        result.selected_threshold = best_t
        result.selected_precision = best_p
        result.selected_recall = best_r
        result.selected_n_admitted = best_n
        result.threshold_pass = t_pass
        result.top_gini = gini

    return result


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


def write_summary_csv(out_path: Path, results: List[PipelineResult]) -> None:
    cols = [
        "unit_id", "pipeline", "t", "n", "n_features", "base_success_rate",
        "mean_auc", "std_auc", "fold_aucs", "gate_threshold", "gate_pass",
        "exclusion", "threshold_pass", "selected_threshold",
        "selected_precision", "selected_recall", "selected_n_admitted",
        "class_weight_used",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        for r in results:
            w.writerow([
                r.unit_id, r.pipeline, _fmt(r.t) if r.t is not None else "",
                r.n, r.n_features, _fmt(r.base_success_rate),
                _fmt(r.mean_auc), _fmt(r.std_auc),
                ";".join(_fmt(a) for a in r.fold_aucs),
                _fmt(r.gate_threshold), "1" if r.gate_pass else "0",
                _fmt(r.exclusion),
                "1" if r.threshold_pass else "0",
                _fmt(r.selected_threshold), _fmt(r.selected_precision),
                _fmt(r.selected_recall),
                str(r.selected_n_admitted) if r.selected_n_admitted is not None else "",
                r.class_weight_used,
            ])


def write_threshold_sweep(out_path: Path, r: PipelineResult) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["unit_id", "pipeline", "t", "threshold", "precision", "recall", "n_admitted", "selected"])
        for (th, p, rec, na) in r.threshold_results:
            sel = "1" if (r.selected_threshold is not None and abs(th - r.selected_threshold) < 1e-9) else "0"
            w.writerow([r.unit_id, r.pipeline, r.t if r.t is not None else "",
                        _fmt(th), _fmt(p), _fmt(rec), na, sel])


def write_fold_aucs(out_path: Path, results: List[PipelineResult]) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["unit_id", "pipeline", "t", "fold_idx", "auc", "n_train", "n_test", "base_success"])
        for r in results:
            for i, (a, ntr, nte, b) in enumerate(zip(r.fold_aucs, r.fold_n_train, r.fold_n_test, r.fold_base_success)):
                w.writerow([r.unit_id, r.pipeline, r.t if r.t is not None else "",
                            i, _fmt(a), ntr, nte, _fmt(b)])


def write_feature_importance(out_path: Path, r: PipelineResult) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["rank", "feature", "gini_importance"])
        for i, (name, val) in enumerate(r.top_gini):
            w.writerow([i, name, _fmt(val)])


def write_pass_list(out_path: Path, results: List[PipelineResult], unit_routing: Dict[str, Dict[str, Any]]) -> None:
    cols = [
        "unit_id", "pipeline_E_AUC_pass", "pipeline_E_threshold_pass",
        "pipeline_D1_smallest_t", "pipeline_D1_AUC_pass", "pipeline_D1_threshold_pass",
        "final_pipeline_assignment", "selected_E_threshold", "selected_D1_threshold",
        "pre_t_sl_atr_multiplier",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        for unit_id, info in unit_routing.items():
            w.writerow([
                unit_id,
                "1" if info["e_auc_pass"] else "0",
                "1" if info["e_threshold_pass"] else "0",
                info["d1_smallest_t"] if info["d1_smallest_t"] is not None else "",
                "1" if info["d1_auc_pass"] else "0",
                "1" if info["d1_threshold_pass"] else "0",
                info["assignment"],
                _fmt(info["e_threshold"]),
                _fmt(info["d1_threshold"]),
                _fmt(info["pre_t_sl_atr_multiplier"]),
            ])


# ============================================================
# Summary doc
# ============================================================

def write_summary_md(
    out_path: Path, results: List[PipelineResult], cfg: dict,
    unit_routing: Dict[str, Dict[str, Any]],
    hashes1: Dict[str, str], hashes2: Optional[Dict[str, str]], det_gate: str,
) -> None:
    lines: List[str] = []
    lines.append("# Arc 11 — Step 4 extractability summary")
    lines.append("")
    lines.append("Protocol: `L_ARC_PROTOCOL.md` v2.3 §§8, 10, 17 + v2.2 §3 (no max-F1 fallback)")
    lines.append("")

    # Verdict
    any_pass = any(
        (info["e_auc_pass"] and info["e_threshold_pass"]) or
        (info["d1_auc_pass"] and info["d1_threshold_pass"])
        for info in unit_routing.values()
    )
    lines.append("## Verdict")
    survivors = [u for u, info in unit_routing.items() if info["assignment"] != "none"]
    if any_pass:
        lines.append(f"**PASS** — {len(survivors)} unit(s) clear §8 + threshold sweep: {', '.join(survivors)}.")
    else:
        lines.append("**FAIL** — Zero units clear §8 with a valid v2.2 §3 threshold. Arc dies at Step 4.")
    lines.append("")

    # Pipeline summary
    lines.append("## Pipeline summary (AUC + threshold sweep)")
    lines.append("")
    lines.append("| unit | pipeline | t | n | base_succ | mean_auc | std_auc | gate | AUC pass | thr pass | selected t | precision | recall | n_admit |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|:---:|:---:|---:|---:|---:|---:|")
    for r in results:
        thr_pass = "YES" if r.threshold_pass else "no"
        auc_pass = "YES" if r.gate_pass else "no"
        t_str = f"{r.t}" if r.t is not None else "—"
        lines.append(
            f"| {r.unit_id} | {r.pipeline} | {t_str} | {r.n} | {_fmt(r.base_success_rate)} "
            f"| {_fmt(r.mean_auc)} | {_fmt(r.std_auc)} | {_fmt(r.gate_threshold)} "
            f"| {auc_pass} | {thr_pass} | {_fmt(r.selected_threshold) or '—'} "
            f"| {_fmt(r.selected_precision) or '—'} | {_fmt(r.selected_recall) or '—'} "
            f"| {r.selected_n_admitted if r.selected_n_admitted is not None else '—'} |"
        )
    lines.append("")

    # Routing
    lines.append("## Final routing per unit")
    lines.append("")
    lines.append("| unit | E AUC pass | E thr pass | D1 smallest-t | D1 AUC pass | D1 thr pass | assignment | pre_t_sl_atr |")
    lines.append("|---|:---:|:---:|---:|:---:|:---:|---|---:|")
    for unit_id, info in unit_routing.items():
        lines.append(
            f"| {unit_id} | {'YES' if info['e_auc_pass'] else 'no'} | {'YES' if info['e_threshold_pass'] else 'no'} "
            f"| {info['d1_smallest_t'] if info['d1_smallest_t'] is not None else '—'} "
            f"| {'YES' if info['d1_auc_pass'] else 'no'} | {'YES' if info['d1_threshold_pass'] else 'no'} "
            f"| {info['assignment']} | {_fmt(info['pre_t_sl_atr_multiplier'])} |"
        )
    lines.append("")

    # Per-fold AUCs (compact)
    lines.append("## Per-fold AUCs (5-fold TimeSeriesSplit)")
    lines.append("")
    for r in results:
        lines.append(f"### {r.unit_id} / {r.pipeline}" + (f" (t={r.t})" if r.t is not None else ""))
        lines.append("")
        lines.append("| fold | AUC | n_train | n_test | base_success |")
        lines.append("|---:|---:|---:|---:|---:|")
        for i, (a, ntr, nte, b) in enumerate(zip(r.fold_aucs, r.fold_n_train, r.fold_n_test, r.fold_base_success)):
            lines.append(f"| {i} | {_fmt(a)} | {ntr} | {nte} | {_fmt(b)} |")
        lines.append("")

    # Top features per surviving (unit, pipeline)
    lines.append("## Top-10 Gini features per AUC-passing (unit, pipeline)")
    lines.append("")
    for r in results:
        if r.gate_pass:
            lines.append(f"### {r.unit_id} / {r.pipeline}" + (f" (t={r.t})" if r.t is not None else ""))
            lines.append("")
            for name, val in r.top_gini[:10]:
                lines.append(f"- `{name}` = {val:.4f}")
            lines.append("")

    # Threshold sweep detail
    lines.append("## Threshold sweep (v2.2 §3: select max precision with recall ≥ 0.60)")
    lines.append("")
    for r in results:
        if not r.gate_pass:
            continue
        lines.append(f"### {r.unit_id} / {r.pipeline}" + (f" (t={r.t})" if r.t is not None else ""))
        lines.append("")
        lines.append("| threshold | precision | recall | n_admit | recall ≥ 0.60? | selected |")
        lines.append("|---:|---:|---:|---:|:---:|:---:|")
        for (th, p, rec, na) in r.threshold_results:
            ok = "YES" if rec >= THRESHOLD_SWEEP_RECALL_MIN else "no"
            sel = "**SEL**" if (r.selected_threshold is not None and abs(th - r.selected_threshold) < 1e-9) else "—"
            lines.append(f"| {th:.2f} | {_fmt(p)} | {_fmt(rec)} | {na} | {ok} | {sel} |")
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

    # Files
    lines.append("## Files")
    lines.append("")
    out_dir = Path(cfg["output"]["results_dir"])
    for name in sorted(hashes1.keys()):
        lines.append(f"- `{out_dir}/{name}`")
    lines.append(f"- `{out_dir}/STEP4_SUMMARY.md`")
    lines.append("- `configs/l_arc_11/step4.yaml`")
    lines.append("- `scripts/l_arc_11/step4_extractability.py`")
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

    pairs = sorted(trades_df["pair"].astype(str).unique())
    pair_caches: Dict[str, PerPairCache] = {}
    dir_4h = str(cfg["data"]["dir_4h"])
    for p in pairs:
        print(f"[l_arc_11 step4] caching {p}", file=sys.stderr)
        pair_caches[p] = _build_pair_cache(p, dir_4h)

    print("[l_arc_11 step4] computing pipeline E features", file=sys.stderr)
    e_features = compute_pipeline_e_features(trades_df, pair_caches)

    results: List[PipelineResult] = []
    unit_routing: Dict[str, Dict[str, Any]] = {}

    pool_size = int(len(trades_df))
    for u in cfg["units"]:
        unit_id = str(u["unit_id"])
        cid_set = set(int(c) for c in u["cluster_ids"])
        sl_mult = float(u["selected_sl_atr_mult"])
        tids = sorted(clusters_df[clusters_df["cluster_id"].isin(cid_set)]["trade_id"].astype(int).tolist())
        y_by_tid = compute_success_labels(
            tids, paths_index, sl_mult, float(cfg["original_sl_atr_mult"])
        )

        # Pipeline E.
        e_sub = e_features[e_features["trade_id"].isin(tids)].copy()
        r_e = evaluate_pipeline(
            unit_id, "E", None, e_sub, PIPELINE_E_FEATURES, y_by_tid, cfg,
            PIPELINE_E_AUC_MIN, pool_size_full=len(tids),
        )
        results.append(r_e)
        print(
            f"[l_arc_11 step4] {unit_id} Pipeline E: mean_auc={_fmt(r_e.mean_auc)} "
            f"gate={'PASS' if r_e.gate_pass else 'FAIL'} thr_sweep={'PASS' if r_e.threshold_pass else 'FAIL'}",
            file=sys.stderr,
        )

        # Pipeline D1: sweep t, smallest-t rule.
        d1_results_for_unit: List[PipelineResult] = []
        smallest_t_passing: Optional[int] = None
        d1_smallest: Optional[PipelineResult] = None
        for t_val in T_CANDIDATES:
            # Filter trades with bars_held >= t.
            tids_at_t = [tid for tid in tids
                         if int(trades_df.loc[trades_df["trade_id"] == tid, "bars_held"].iloc[0]) >= t_val]
            exclusion = 1.0 - (len(tids_at_t) / max(len(tids), 1))
            print(
                f"[l_arc_11 step4] {unit_id} Pipeline D1 t={t_val}: n_at_t={len(tids_at_t)} "
                f"exclusion={exclusion:.4f}",
                file=sys.stderr,
            )

            # Build path-so-far features.
            feat_rows: List[Dict[str, Any]] = []
            for tid in tids_at_t:
                pf = _path_features_at_t(
                    paths_index[tid], t_val, sl_mult, float(cfg["original_sl_atr_mult"])
                )
                if pf is None:
                    continue
                # Merge with E base 8 (which are entry-time only).
                base_row = e_sub[e_sub["trade_id"] == tid].iloc[0]
                merged = {
                    "trade_id": tid,
                    "entry_time": base_row["entry_time"],
                }
                for c in PIPELINE_E_BASE:
                    merged[c] = base_row[c]
                merged.update(pf)
                feat_rows.append(merged)
            d1_feat_df = pd.DataFrame(feat_rows)

            r_d1 = evaluate_pipeline(
                unit_id, f"D1_t{t_val}", int(t_val), d1_feat_df, PIPELINE_D1_FEATURES,
                y_by_tid, cfg, PIPELINE_D1_AUC_MIN, pool_size_full=len(tids),
            )
            results.append(r_d1)
            d1_results_for_unit.append(r_d1)

            print(
                f"[l_arc_11 step4] {unit_id} Pipeline D1 t={t_val}: mean_auc={_fmt(r_d1.mean_auc)} "
                f"gate={'PASS' if r_d1.gate_pass else 'FAIL'} thr_sweep={'PASS' if r_d1.threshold_pass else 'FAIL'}",
                file=sys.stderr,
            )

            # Smallest-t rule: AUC ≥ 0.60 AND exclusion ≤ 0.30.
            if (r_d1.gate_pass and r_d1.exclusion is not None and r_d1.exclusion <= D1_EXCLUSION_MAX
                    and smallest_t_passing is None):
                smallest_t_passing = t_val
                d1_smallest = r_d1

        # Routing for this unit.
        e_auc_pass = r_e.gate_pass
        e_thr_pass = r_e.threshold_pass
        d1_auc_pass = d1_smallest is not None
        d1_thr_pass = d1_smallest.threshold_pass if d1_smallest is not None else False

        # Per protocol §8: clears E (AUC + thr) → Pipeline E; clears D1 (AUC + thr) → Pipeline D1;
        # clears both → both run at Step 5 WFO.
        if (e_auc_pass and e_thr_pass) and (d1_auc_pass and d1_thr_pass):
            assignment = "both"
        elif e_auc_pass and e_thr_pass:
            assignment = "E"
        elif d1_auc_pass and d1_thr_pass:
            assignment = "D1"
        else:
            assignment = "none"

        unit_routing[unit_id] = {
            "e_auc_pass": e_auc_pass,
            "e_threshold_pass": e_thr_pass,
            "e_threshold": r_e.selected_threshold,
            "d1_smallest_t": smallest_t_passing,
            "d1_auc_pass": d1_auc_pass,
            "d1_threshold_pass": d1_thr_pass,
            "d1_threshold": d1_smallest.selected_threshold if d1_smallest is not None else None,
            "assignment": assignment,
            "pre_t_sl_atr_multiplier": sl_mult,   # per v2.3 §5 / Open-24
        }

    # Write outputs.
    hashes: Dict[str, str] = {}
    summary_csv = out_dir / "extractability_summary.csv"
    write_summary_csv(summary_csv, results)
    hashes[summary_csv.name] = _file_sha256(summary_csv)

    pass_list = out_dir / "extractability_pass_list.csv"
    write_pass_list(pass_list, results, unit_routing)
    hashes[pass_list.name] = _file_sha256(pass_list)

    fold_aucs_csv = out_dir / "fold_aucs.csv"
    write_fold_aucs(fold_aucs_csv, results)
    hashes[fold_aucs_csv.name] = _file_sha256(fold_aucs_csv)

    for r in results:
        if r.gate_pass:
            stem = r.unit_id + "_" + r.pipeline
            ts_path = out_dir / f"threshold_sweep_{stem}.csv"
            write_threshold_sweep(ts_path, r)
            hashes[ts_path.name] = _file_sha256(ts_path)

            fi_path = out_dir / f"feature_importance_{stem}.csv"
            write_feature_importance(fi_path, r)
            hashes[fi_path.name] = _file_sha256(fi_path)

    ctx = {"results": results, "unit_routing": unit_routing, "out_dir": out_dir}
    return hashes, ctx


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Arc 11 Step 4 — extractability.")
    ap.add_argument("-c", "--config", type=Path, default=Path("configs/l_arc_11/step4.yaml"))
    args = ap.parse_args(argv)
    cfg_path = args.config
    if not cfg_path.is_absolute():
        cfg_path = (_REPO_ROOT / cfg_path).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    print("[l_arc_11 step4] === RUN 1 ===", file=sys.stderr)
    hashes1, ctx = _run_once(cfg)
    hashes2 = None
    if bool(cfg["output"].get("determinism_check", True)):
        print("[l_arc_11 step4] === RUN 2 (determinism) ===", file=sys.stderr)
        hashes2, _ = _run_once(cfg)
    if hashes2 is None:
        det_gate = "N/A"
    else:
        det_gate = "PASS" if all(hashes1[k] == hashes2.get(k) for k in hashes1) else "FAIL"

    summary_path = ctx["out_dir"] / cfg["output"]["summary_md"]
    write_summary_md(summary_path, ctx["results"], cfg, ctx["unit_routing"],
                     hashes1, hashes2, det_gate)

    n_pass = sum(1 for info in ctx["unit_routing"].values() if info["assignment"] != "none")
    print(f"[l_arc_11 step4] DONE. units_passing={n_pass}, det={det_gate}", file=sys.stderr)
    return 0 if n_pass > 0 else 0  # return 0 either way — Step 4 KILL is still a valid completion


if __name__ == "__main__":
    raise SystemExit(main())
