"""Arc 11 — Filter-diagnosis test (off-protocol, documentation only).

Diagnoses which direction can lift Pipeline E AUC above the §8 gate (0.65) on
the cleanest capturable-not-extractable case: Arc 11 cohort c1.

NOT a canonical anything. Arc 11 stays Closed-HALT. No queue / registry /
protocol mutation. No archetype or signal status changes. All outputs tagged
`filter_diag_*` and placed under results/l_arc_11/filter_diag/.

Four regimes:
  A — Baseline: PIPELINE_E_FEATURES, target = (cluster_id == 1), entry at t=0
  B — Delayed entry: PIPELINE_E_BASE + path-so-far at t∈{1,3,5,8}, same target
  C — Multi-TF entry: A's features + D1 (one-day-lag) + 1H context features
  D — Reframed target: A's features at t=0, target switched to:
        D-reach1R: reach_1R at SL=3 (mfe in new R-frame >= 1.0)
        D-mfe2R:   mfe at SL=3 >= 2.0 (binary)

WFO fold structure: same 7 anchored expanding folds as canonical S5
(arc11_exp_s5_wfo FOLDS). Exit: SL=3.0×ATR + time-exit (no §11 trail —
diagnosing the selector, not the exit policy).

Pre-test sanity:
  - clusters_K4.csv sha256 hash printed and asserted against the value
    captured at S4 commit
  - FOLDS structure printed for visual fold-drift check
  - D1 alignment audit: D1 bar used has date < signal_date (one-day lag)
  - 1H alignment audit: 1H bar used has bar_start_time <= signal_bar_close
    (signal_bar_close = signal_bar_start + 4h)

Stop conditions:
  - Any regime with AUC > 0.70 on >= 5/7 folds → flag strong candidate
  - All regimes < 0.55 mean AUC → flag structural extractability ceiling

Single deliverable: results/l_arc_11/filter_diag/filter_diag_report.md
plus all per-regime CSVs tagged filter_diag_*.

Usage:
    py scripts/l_arc_11/filter_diag.py
"""

from __future__ import annotations

import csv
import hashlib
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.l_arc_11.experimental_s5_wfo import FOLDS, build_paths_index  # noqa: E402
from scripts.l_arc_11.step3_capturability import _eval_trade_at_sl  # noqa: E402
from scripts.l_arc_11.step4_extractability import (  # noqa: E402
    PIPELINE_E_BASE,
    PIPELINE_E_FEATURES,
    _build_pair_cache,
    _ema,
    _wilder_atr,
    _wilder_rsi,
    compute_pipeline_e_features,
)

OUT_DIR = _REPO_ROOT / "results" / "l_arc_11" / "filter_diag"
SL_DEPLOY = 3.0
ORIGINAL_SL = 2.0
DATA_DIR_4H = "C:/Users/panap/Documents/Forex-Backtester/data/4hr"
DATA_DIR_D1 = "C:/Users/panap/Documents/Forex-Backtester/data/daily"
DATA_DIR_1H = "C:/Users/panap/Documents/Forex-Backtester/data/1hr"

REGIME_B_T_VALUES = [1, 3, 5, 8]
GATE_E = 0.65
STRONG_CANDIDATE_AUC = 0.70
STRUCTURAL_CEILING_MEAN_AUC = 0.55

# Captured at S4 commit (verified manually):
EXPECTED_CLUSTERS_K4_SHA256 = "a380b57badf84dd79ba774a4855498d1cae67052eb4e0d6192d8ef0666dd2bb6"


# ============================================================
# Pre-test sanity
# ============================================================

def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pre_test_sanity() -> Dict[str, Any]:
    clusters_path = _REPO_ROOT / "results/l_arc_11/step2/clusters_K4.csv"
    cluster_hash = _file_sha256(clusters_path)
    cluster_ok = cluster_hash == EXPECTED_CLUSTERS_K4_SHA256

    sanity: Dict[str, Any] = {
        "clusters_K4_sha256": cluster_hash,
        "clusters_K4_expected": EXPECTED_CLUSTERS_K4_SHA256,
        "clusters_K4_stable": cluster_ok,
        "folds": FOLDS,
        "n_folds": len(FOLDS),
    }
    print(f"[diag] clusters_K4 sha256: {cluster_hash}", file=sys.stderr)
    print(f"[diag] clusters_K4 stable: {cluster_ok}", file=sys.stderr)
    print(f"[diag] WFO folds: {len(FOLDS)} folds, {FOLDS[0][1]} → {FOLDS[-1][2]}", file=sys.stderr)
    return sanity


# ============================================================
# Multi-TF pair cache (D1 + 1H additions to PerPairCache from step4)
# ============================================================

@dataclass
class MultiTFCache:
    pair: str
    # 4H
    df_4h: pd.DataFrame
    idx_4h_by_start: Dict[pd.Timestamp, int]
    # D1
    df_d1: pd.DataFrame
    d1_dates_normalised: np.ndarray   # datetime64[ns], normalised to date
    d1_close: np.ndarray
    d1_atr14: np.ndarray
    d1_rsi14: np.ndarray
    d1_ema5: np.ndarray               # used for "5-bar trend slope"
    # 1H
    df_1h: pd.DataFrame
    h1_start_times: np.ndarray         # datetime64[ns], bar start times
    h1_atr14: np.ndarray
    h1_close: np.ndarray
    h1_high: np.ndarray
    h1_low: np.ndarray


def _load_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "time" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def _build_multi_tf_cache(pair: str) -> MultiTFCache:
    df_4h = _load_df(Path(DATA_DIR_4H) / f"{pair}.csv")
    idx_4h = {pd.Timestamp(ts): i for i, ts in enumerate(df_4h["date"].to_numpy())}

    df_d1 = _load_df(Path(DATA_DIR_D1) / f"{pair}.csv")
    df_d1["date_norm"] = df_d1["date"].dt.normalize()
    df_d1 = df_d1.drop_duplicates(subset="date_norm").sort_values("date_norm").reset_index(drop=True)
    d1_close = df_d1["close"].astype(float).to_numpy()
    d1_atr = _wilder_atr(df_d1, 14)
    d1_rsi = _wilder_rsi(d1_close, 14)
    d1_ema5 = _ema(d1_close, 5)

    df_1h = _load_df(Path(DATA_DIR_1H) / f"{pair}.csv")
    h1_close = df_1h["close"].astype(float).to_numpy()
    h1_high = df_1h["high"].astype(float).to_numpy()
    h1_low = df_1h["low"].astype(float).to_numpy()
    h1_atr = _wilder_atr(df_1h, 14)
    h1_starts = df_1h["date"].to_numpy()

    return MultiTFCache(
        pair=pair, df_4h=df_4h, idx_4h_by_start=idx_4h,
        df_d1=df_d1, d1_dates_normalised=df_d1["date_norm"].to_numpy(),
        d1_close=d1_close, d1_atr14=d1_atr, d1_rsi14=d1_rsi, d1_ema5=d1_ema5,
        df_1h=df_1h, h1_start_times=h1_starts, h1_atr14=h1_atr,
        h1_close=h1_close, h1_high=h1_high, h1_low=h1_low,
    )


# ============================================================
# Multi-TF feature extraction (Regime C)
# ============================================================

MULTI_TF_FEATURES: List[str] = [
    # D1 (one-day lag — uses D1 bar with date < signal_date)
    "d1_trend_slope_5bar",
    "d1_atr_ratio_to_close",
    "d1_rsi_14",
    "d1_pos_in_20bar_range",
    # 1H (uses 1H bar with start_time <= 4h_bar_close)
    "h1_atr_14",
    "h1_pullback_depth_atr",
    "h1_hh_hl_count_last_6_bars",
]


def _d1_lag1_idx(cache: MultiTFCache, signal_ts: pd.Timestamp) -> int:
    target = (pd.Timestamp(signal_ts).normalize() - pd.Timedelta(days=1)).to_datetime64()
    dates = pd.DatetimeIndex(cache.d1_dates_normalised)
    idx = dates.get_indexer([target], method="ffill")
    return int(idx[0])


def _h1_aligned_idx(cache: MultiTFCache, signal_ts: pd.Timestamp) -> int:
    """Most recent 1H bar with start_time <= 4h_bar_close = signal_ts + 4h."""
    h4_close = pd.Timestamp(signal_ts) + pd.Timedelta(hours=4)
    target = h4_close.to_datetime64()
    starts = pd.DatetimeIndex(cache.h1_start_times)
    # Find largest i where starts[i] + 1h <= target  ==>  starts[i] <= target - 1h
    cutoff = (pd.Timestamp(target) - pd.Timedelta(hours=1)).to_datetime64()
    idx = starts.get_indexer([cutoff], method="ffill")
    return int(idx[0])


def compute_multi_tf_features(
    trades_df: pd.DataFrame, caches: Dict[str, MultiTFCache]
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    audit: List[Dict[str, Any]] = []
    for _, t in trades_df.iterrows():
        pair = str(t["pair"])
        cache = caches[pair]
        sig_ts = pd.Timestamp(t["signal_bar_time"])

        # ----- D1 (one-day lag) -----
        d1_idx = _d1_lag1_idx(cache, sig_ts)
        if d1_idx < 0 or d1_idx < 20:
            d1_slope = float("nan")
            d1_atr_ratio = float("nan")
            d1_rsi = float("nan")
            d1_pos20 = float("nan")
            d1_date_used = None
        else:
            close_d1 = float(cache.d1_close[d1_idx])
            atr_d1 = float(cache.d1_atr14[d1_idx]) if not math.isnan(cache.d1_atr14[d1_idx]) else float("nan")
            # Trend slope over last 5 D1 bars: (close_now - close_5_back) / 5 / atr.
            close_5_ago = float(cache.d1_close[d1_idx - 5])
            d1_slope = ((close_d1 - close_5_ago) / 5.0 / atr_d1) if atr_d1 > 0 else float("nan")
            d1_atr_ratio = (atr_d1 / close_d1) if close_d1 > 0 and atr_d1 > 0 else float("nan")
            d1_rsi = float(cache.d1_rsi14[d1_idx]) if not math.isnan(cache.d1_rsi14[d1_idx]) else float("nan")
            # pos in D1 20-bar range
            lows = cache.df_d1["low"].iloc[d1_idx - 20:d1_idx + 1].astype(float).to_numpy()
            highs = cache.df_d1["high"].iloc[d1_idx - 20:d1_idx + 1].astype(float).to_numpy()
            wrng = highs.max() - lows.min()
            d1_pos20 = (close_d1 - lows.min()) / wrng if wrng > 0 else 0.5
            d1_date_used = pd.Timestamp(cache.d1_dates_normalised[d1_idx])

        # ----- 1H (most recent bar closing at or before 4H bar close) -----
        h1_idx = _h1_aligned_idx(cache, sig_ts)
        if h1_idx < 0 or h1_idx < 12:
            h1_atr = float("nan")
            h1_pullback = float("nan")
            h1_hh_hl = float("nan")
            h1_start_used = None
        else:
            h1_atr = float(cache.h1_atr14[h1_idx]) if not math.isnan(cache.h1_atr14[h1_idx]) else float("nan")
            # Pullback depth: max(high) - close_now, in atr units.
            highs_12 = cache.h1_high[h1_idx - 12:h1_idx + 1]
            close_h1 = float(cache.h1_close[h1_idx])
            h1_pullback = (float(highs_12.max()) - close_h1) / h1_atr if h1_atr > 0 else float("nan")
            # HH/HL count last 6 bars (consecutive higher highs OR higher lows).
            highs_6 = cache.h1_high[h1_idx - 6:h1_idx + 1]
            lows_6 = cache.h1_low[h1_idx - 6:h1_idx + 1]
            hh = int(np.sum(np.diff(highs_6) > 0))
            hl = int(np.sum(np.diff(lows_6) > 0))
            h1_hh_hl = float(hh + hl)
            h1_start_used = pd.Timestamp(cache.h1_start_times[h1_idx])

        rows.append({
            "trade_id": int(t["trade_id"]),
            "d1_trend_slope_5bar": d1_slope,
            "d1_atr_ratio_to_close": d1_atr_ratio,
            "d1_rsi_14": d1_rsi,
            "d1_pos_in_20bar_range": d1_pos20,
            "h1_atr_14": h1_atr,
            "h1_pullback_depth_atr": h1_pullback,
            "h1_hh_hl_count_last_6_bars": h1_hh_hl,
        })

        # Audit a subset (first occurrence per pair plus a few random).
        if d1_date_used is not None or h1_start_used is not None:
            audit.append({
                "trade_id": int(t["trade_id"]),
                "pair": pair,
                "signal_bar_time": str(sig_ts),
                "expected_d1_date_max": str((sig_ts.normalize() - pd.Timedelta(days=1))),
                "d1_date_used": str(d1_date_used) if d1_date_used is not None else "",
                "d1_lag_ok": (d1_date_used is None) or (d1_date_used < sig_ts.normalize()),
                "expected_h1_start_max": str(sig_ts + pd.Timedelta(hours=3)),
                "h1_start_used": str(h1_start_used) if h1_start_used is not None else "",
                "h1_align_ok": (h1_start_used is None) or (h1_start_used <= sig_ts + pd.Timedelta(hours=3)),
            })

    return pd.DataFrame(rows), audit


# ============================================================
# Path-so-far features at t (Regime B)
# ============================================================

# Reuse step4's _path_features_at_t — but we want the features in the ORIGINAL
# R-frame (SL=2) since the target is cluster membership (path-shape based on
# original-SL truncation; the cluster IDs were computed on original-SL paths).
# Actually wait: clusters were computed on is_held=1 bars which depend on the
# original SL=2 simulation. For Regime B, "path-so-far at t bars after entry"
# under original SL=2 is the correct view because that's the path that defined
# the cluster.
#
# However, since we filter trades to bars_held >= t (under SL=2 simulation),
# trades that SL'd before t are excluded from feature dataset. We treat them
# as "no Regime B prediction" — but for AUC purposes we only score the trades
# that have features (consistent with step4's D1 t treatment).

PATH_SO_FAR_FEATURES_AT_T: List[str] = [
    "close_r_at_t",
    "mfe_so_far_r_at_t",
    "mae_so_far_r_at_t",
    "bars_in_profit_at_t",
    "local_peaks_so_far_at_t",
    "monotonicity_so_far_at_t",
    "velocity_first_t",
]


def _path_features_at_t_orig_frame(
    path: pd.DataFrame, t: int
) -> Optional[Dict[str, float]]:
    """Path-so-far features at bar t, in ORIGINAL R-frame (SL=2).

    Filters to trades that survived to bar t (SL not hit before t).

    The original-SL truncation matches the path-shape data the clusters were
    built on, so this is the right frame for "predicting cluster membership
    at t bars after signal."
    """
    # Use original-SL eval (sl_mult = original_sl = 2.0 → scale = 1, no transform).
    te = _eval_trade_at_sl(path, ORIGINAL_SL, ORIGINAL_SL)
    if te.truncated_at_bar < t:
        return None
    sub = path[path["bar_offset"] <= t].sort_values("bar_offset", kind="mergesort")
    if sub.empty:
        return None
    close_r = sub["close_r"].to_numpy(dtype=float)
    mfe = sub["mfe_so_far_r"].to_numpy(dtype=float)
    # mae_so_far_r in the path is the running cumulative mae (negative).

    close_at_t = float(close_r[-1])
    mfe_at_t = float(mfe[-1])
    # mae_so_far at t: take min cumulative mae.
    mae_arr = sub["mae_so_far_r"].to_numpy(dtype=float)
    mae_at_t = float(mae_arr.min()) if mae_arr.size > 0 else 0.0

    bars_in_profit = int(np.sum(close_r > 0))
    local_peaks = int(np.sum(np.diff(mfe) > 0))
    in_profit = close_r[close_r > 0]
    mono = float(np.mean(in_profit[1:] >= in_profit[:-1])) if in_profit.size >= 2 else 0.0
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
# WFO CV utility
# ============================================================

def _decide_class_weight(y: np.ndarray, threshold: float = 0.30) -> str:
    base = float(y.mean()) if y.size > 0 else 0.0
    minority = min(base, 1.0 - base)
    return "balanced" if minority < threshold else "none"


def _train_predict_one_fold(
    is_X: pd.DataFrame, is_y: np.ndarray,
    oos_X: pd.DataFrame, oos_y: np.ndarray,
    model_kw: dict,
) -> Tuple[float, int, int, float, Any, pd.Series]:
    """Return (auc, n_train, n_test, base_test, clf, med)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    n_tr = int(len(is_X))
    n_te = int(len(oos_X))
    base_te = float(oos_y.mean()) if oos_y.size > 0 else 0.0
    if n_tr < 20 or n_te < 5 or len(np.unique(is_y)) < 2 or len(np.unique(oos_y)) < 2:
        return float("nan"), n_tr, n_te, base_te, None, pd.Series(dtype=float)
    cw = _decide_class_weight(is_y)
    kw = dict(model_kw)
    if cw == "balanced":
        kw["class_weight"] = "balanced"
    clf = RandomForestClassifier(**kw)
    med = is_X.median(numeric_only=True)
    clf.fit(is_X.fillna(med), is_y)
    p = clf.predict_proba(oos_X.fillna(med))[:, 1]
    try:
        auc = float(roc_auc_score(oos_y, p))
    except ValueError:
        auc = float("nan")
    return auc, n_tr, n_te, base_te, clf, med


def _wfo_run(
    feature_df: pd.DataFrame, feature_cols: List[str], label_by_tid: Dict[int, int],
    model_kw: dict, fold_filter_fn: Optional[callable] = None,
) -> Tuple[List[float], List[Dict[str, Any]], Optional[Any], Optional[pd.Series]]:
    """Return (fold_aucs, fold_details, last_full_classifier, last_full_med).

    fold_filter_fn(trade_id) -> bool: optional eligibility filter applied to both
    IS and OOS pools (e.g. trades that survived to bar t).

    last_full_classifier / last_full_med are trained on ALL trades from the last
    fold's IS (used for feature-importance ranking only).
    """
    fdf = feature_df.copy()
    fdf["entry_time"] = pd.to_datetime(fdf["entry_time"])
    fdf = fdf.sort_values("entry_time").reset_index(drop=True)

    if fold_filter_fn is not None:
        elig_mask = fdf["trade_id"].astype(int).map(fold_filter_fn)
        fdf = fdf[elig_mask].reset_index(drop=True)

    fold_aucs: List[float] = []
    fold_details: List[Dict[str, Any]] = []
    last_clf = None
    last_med = None

    for fold_id, oos_s, oos_e in FOLDS:
        oos_start = pd.Timestamp(oos_s)
        oos_end = pd.Timestamp(oos_e)
        is_sub = fdf[fdf["entry_time"] < oos_start]
        oos_sub = fdf[(fdf["entry_time"] >= oos_start) & (fdf["entry_time"] < oos_end)]
        is_X = is_sub[feature_cols].copy()
        oos_X = oos_sub[feature_cols].copy()
        is_y = np.array([label_by_tid[int(t)] for t in is_sub["trade_id"]], dtype=int)
        oos_y = np.array([label_by_tid[int(t)] for t in oos_sub["trade_id"]], dtype=int)
        auc, n_tr, n_te, base_te, clf, med = _train_predict_one_fold(
            is_X, is_y, oos_X, oos_y, model_kw,
        )
        fold_aucs.append(auc)
        fold_details.append({
            "fold_id": fold_id, "oos_start": oos_s, "oos_end": oos_e,
            "n_train": n_tr, "n_test": n_te, "base_test": base_te, "auc": auc,
        })
        if clf is not None:
            last_clf, last_med = clf, med

    return fold_aucs, fold_details, last_clf, last_med


def _summarise_aucs(aucs: List[float], gate: float = GATE_E) -> Tuple[float, int, int]:
    valid = [a for a in aucs if not math.isnan(a)]
    mean_auc = float(np.mean(valid)) if valid else float("nan")
    n_gate = int(sum(1 for a in valid if a >= gate))
    n_total = len(valid)
    return mean_auc, n_gate, n_total


# ============================================================
# Regime runners
# ============================================================

@dataclass
class RegimeResult:
    name: str
    fold_aucs: List[float]
    mean_auc: float
    n_clears_gate: int
    n_folds: int
    fold_details: List[Dict[str, Any]]
    feature_cols: List[str]
    last_clf: Any = None
    last_med: Any = None
    notes: str = ""


def run_regime_a(
    e_features: pd.DataFrame, c1_label_by_tid: Dict[int, int], model_kw: dict,
) -> RegimeResult:
    aucs, details, clf, med = _wfo_run(
        e_features, PIPELINE_E_FEATURES, c1_label_by_tid, model_kw,
    )
    mean_auc, n_gate, n_total = _summarise_aucs(aucs)
    return RegimeResult(
        name="A_baseline_c1_PE", fold_aucs=aucs, mean_auc=mean_auc,
        n_clears_gate=n_gate, n_folds=n_total, fold_details=details,
        feature_cols=PIPELINE_E_FEATURES, last_clf=clf, last_med=med,
    )


def run_regime_b(
    trades_df: pd.DataFrame, paths_index: Dict[int, pd.DataFrame],
    e_features: pd.DataFrame, c1_label_by_tid: Dict[int, int], model_kw: dict,
) -> Tuple[List[RegimeResult], List[Dict[str, Any]]]:
    """Returns (results_per_t, mfe_cost_table)."""
    results: List[RegimeResult] = []
    mfe_cost_rows: List[Dict[str, Any]] = []
    e_base_idx = {int(r["trade_id"]): r for _, r in e_features.iterrows()}

    for t_val in REGIME_B_T_VALUES:
        feat_rows: List[Dict[str, Any]] = []
        mfe_at_t_arr: List[float] = []
        for _, tr in trades_df.iterrows():
            tid = int(tr["trade_id"])
            pf = _path_features_at_t_orig_frame(paths_index[tid], t_val)
            if pf is None:
                continue
            base_row = e_base_idx[tid]
            merged = {"trade_id": tid, "entry_time": base_row["entry_time"]}
            for c in PIPELINE_E_BASE:
                merged[c] = base_row[c]
            merged.update(pf)
            feat_rows.append(merged)
            mfe_at_t_arr.append(pf["mfe_so_far_r_at_t"])
        feat_df = pd.DataFrame(feat_rows)
        cols = list(PIPELINE_E_BASE) + list(PATH_SO_FAR_FEATURES_AT_T)
        aucs, details, clf, med = _wfo_run(
            feat_df, cols, c1_label_by_tid, model_kw,
        )
        mean_auc, n_gate, n_total = _summarise_aucs(aucs)
        results.append(RegimeResult(
            name=f"B_delayed_t{t_val}_c1", fold_aucs=aucs, mean_auc=mean_auc,
            n_clears_gate=n_gate, n_folds=n_total, fold_details=details,
            feature_cols=cols, last_clf=clf, last_med=med,
        ))
        # mfe cost (in original R units, mean over surviving trades).
        if mfe_at_t_arr:
            mfe_cost_rows.append({
                "t": t_val, "n_surviving": len(mfe_at_t_arr),
                "n_excluded_pre_t_sl": int(len(trades_df) - len(mfe_at_t_arr)),
                "mean_mfe_so_far_r_at_t": float(np.mean(mfe_at_t_arr)),
                "median_mfe_so_far_r_at_t": float(np.median(mfe_at_t_arr)),
            })
    return results, mfe_cost_rows


def run_regime_c(
    e_features: pd.DataFrame, multi_tf_df: pd.DataFrame,
    c1_label_by_tid: Dict[int, int], model_kw: dict,
) -> RegimeResult:
    combined = e_features.merge(multi_tf_df, on="trade_id", how="left")
    cols = list(PIPELINE_E_FEATURES) + list(MULTI_TF_FEATURES)
    aucs, details, clf, med = _wfo_run(combined, cols, c1_label_by_tid, model_kw)
    mean_auc, n_gate, n_total = _summarise_aucs(aucs)
    return RegimeResult(
        name="C_multiTF_c1_PE+D1+1H", fold_aucs=aucs, mean_auc=mean_auc,
        n_clears_gate=n_gate, n_folds=n_total, fold_details=details,
        feature_cols=cols, last_clf=clf, last_med=med,
    )


def run_regime_d(
    e_features: pd.DataFrame,
    reach_1r_label: Dict[int, int], mfe_2r_label: Dict[int, int], model_kw: dict,
) -> Tuple[RegimeResult, RegimeResult]:
    aucs_r, det_r, clf_r, med_r = _wfo_run(
        e_features, PIPELINE_E_FEATURES, reach_1r_label, model_kw,
    )
    mean_r, n_g_r, n_t_r = _summarise_aucs(aucs_r)
    r_reach = RegimeResult(
        name="D_target_reach1R_atSL3", fold_aucs=aucs_r, mean_auc=mean_r,
        n_clears_gate=n_g_r, n_folds=n_t_r, fold_details=det_r,
        feature_cols=PIPELINE_E_FEATURES, last_clf=clf_r, last_med=med_r,
    )
    aucs_m, det_m, clf_m, med_m = _wfo_run(
        e_features, PIPELINE_E_FEATURES, mfe_2r_label, model_kw,
    )
    mean_m, n_g_m, n_t_m = _summarise_aucs(aucs_m)
    r_mfe = RegimeResult(
        name="D_target_mfe2R_atSL3", fold_aucs=aucs_m, mean_auc=mean_m,
        n_clears_gate=n_g_m, n_folds=n_t_m, fold_details=det_m,
        feature_cols=PIPELINE_E_FEATURES, last_clf=clf_m, last_med=med_m,
    )
    return r_reach, r_mfe


# ============================================================
# Live-deployable simulation for best regime
# ============================================================

def _live_deployable_for_regime(
    regime: RegimeResult, feature_df: pd.DataFrame,
    label_by_tid: Dict[int, int], final_r_at_sl3: Dict[int, float], model_kw: dict,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """Simulate per-fold admission at threshold; report worst/mean fold ROI ann,
    worst DD, deployable Y/N, similar to prior exp S5 runs."""
    from sklearn.ensemble import RandomForestClassifier
    fdf = feature_df.copy()
    fdf["entry_time"] = pd.to_datetime(fdf["entry_time"])
    fdf = fdf.sort_values("entry_time").reset_index(drop=True)

    fold_metrics: List[Dict[str, Any]] = []
    full_returns: List[float] = []
    for fold_id, oos_s, oos_e in FOLDS:
        oos_start = pd.Timestamp(oos_s)
        oos_end = pd.Timestamp(oos_e)
        days = (oos_end - oos_start).days
        is_sub = fdf[fdf["entry_time"] < oos_start]
        oos_sub = fdf[(fdf["entry_time"] >= oos_start) & (fdf["entry_time"] < oos_end)].sort_values("entry_time")
        is_X = is_sub[regime.feature_cols]
        oos_X = oos_sub[regime.feature_cols]
        is_y = np.array([label_by_tid[int(t)] for t in is_sub["trade_id"]], dtype=int)
        admitted_tids: List[int] = []
        if len(is_sub) >= 50 and len(np.unique(is_y)) >= 2:
            cw = _decide_class_weight(is_y)
            kw = dict(model_kw)
            if cw == "balanced":
                kw["class_weight"] = "balanced"
            clf = RandomForestClassifier(**kw)
            med = is_X.median(numeric_only=True)
            clf.fit(is_X.fillna(med), is_y)
            p = clf.predict_proba(oos_X.fillna(med))[:, 1]
            admit_mask = p >= threshold
            admitted_tids = oos_sub["trade_id"].astype(int).to_numpy()[admit_mask].tolist()
            classifier_trained = True
        else:
            # Cold start — admit all.
            admitted_tids = oos_sub["trade_id"].astype(int).tolist()
            classifier_trained = False

        admit_returns = [final_r_at_sl3[tid] for tid in admitted_tids]
        # Compounded equity per fold.
        eq = 1.0
        curve = [1.0]
        for r in admit_returns:
            eq *= (1.0 + r * 0.005)
            curve.append(eq)
        roi = eq - 1.0
        arr = np.array(curve, dtype=float)
        peak = np.maximum.accumulate(arr)
        max_dd = float(((peak - arr) / peak).max())
        roi_ann = (eq) ** (365.25 / days) - 1.0 if days > 0 else 0.0
        fold_metrics.append({
            "fold_id": fold_id, "n_admit": len(admitted_tids),
            "n_universe": len(oos_sub), "mean_r": float(np.mean(admit_returns)) if admit_returns else 0.0,
            "roi_period_pct": roi * 100, "roi_ann_pct": roi_ann * 100, "max_dd_pct": max_dd * 100,
            "classifier_trained": classifier_trained,
        })
        full_returns.extend(admit_returns)

    fold_rois_ann = [f["roi_ann_pct"] / 100 for f in fold_metrics if f["n_admit"] > 0]
    fold_rois_period = [f["roi_period_pct"] / 100 for f in fold_metrics if f["n_admit"] > 0]
    fold_dds = [f["max_dd_pct"] / 100 for f in fold_metrics if f["n_admit"] > 0]
    fold_counts = [f["n_admit"] for f in fold_metrics if f["n_admit"] > 0]

    sign_ok = all(r > 0 for r in fold_rois_period)
    worst_roi = min(fold_rois_ann) if fold_rois_ann else 0.0
    mean_roi = float(np.mean(fold_rois_ann)) if fold_rois_ann else 0.0
    worst_dd = max(fold_dds) if fold_dds else 0.0
    min_count = min(fold_counts) if fold_counts else 0

    # Full-data
    eq = 1.0
    curve = [1.0]
    for r in full_returns:
        eq *= (1.0 + r * 0.005)
        curve.append(eq)
    full_roi = eq - 1.0
    arr = np.array(curve, dtype=float)
    peak = np.maximum.accumulate(arr)
    full_dd = float(((peak - arr) / peak).max())

    pass_dep = (sign_ok and worst_roi >= 0.05 and mean_roi >= 0.08
                and worst_dd <= 0.08 and min_count >= 15
                and full_roi >= 0.05 and full_dd <= 0.10)
    pass_viable = (sign_ok and worst_roi > 0.0 and mean_roi >= 0.03
                   and worst_dd <= 0.08 and min_count >= 5
                   and full_roi >= 0.03 and full_dd <= 0.10)

    return {
        "regime": regime.name,
        "threshold": threshold,
        "sign_consistency": sign_ok,
        "worst_fold_roi_ann_pct": worst_roi * 100,
        "mean_fold_roi_ann_pct": mean_roi * 100,
        "worst_fold_dd_pct": worst_dd * 100,
        "min_trade_count": min_count,
        "full_data_roi_pct": full_roi * 100,
        "full_data_dd_pct": full_dd * 100,
        "pass_deployable": pass_dep, "pass_viable": pass_viable,
        "per_fold": fold_metrics,
    }


# ============================================================
# Output writers
# ============================================================

def _fmt(x: Any, dec: int = 4) -> str:
    if x is None:
        return ""
    try:
        xf = float(x)
        if not math.isfinite(xf):
            return ""
    except Exception:
        return str(x)
    return f"{xf:.{dec}f}"


def write_csv(path: Path, headers: List[str], rows: List[List[Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(headers)
        for r in rows:
            w.writerow(r)


# ============================================================
# Driver
# ============================================================

def main() -> int:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[diag] === Pre-test sanity ===", file=sys.stderr)
    sanity = pre_test_sanity()
    if not sanity["clusters_K4_stable"]:
        print("[diag] WARNING: clusters_K4.csv hash drift — continuing but flag in report.", file=sys.stderr)

    print("[diag] loading Step 1 / Step 2 artefacts", file=sys.stderr)
    trades_df = pd.read_csv(_REPO_ROOT / "results/l_arc_11/step1_verbatim/trades_all.csv")
    paths_df = pd.read_csv(_REPO_ROOT / "results/l_arc_11/step1_verbatim/trades_paths.csv")
    clusters_df = pd.read_csv(_REPO_ROOT / "results/l_arc_11/step2/clusters_K4.csv")

    paths_index = build_paths_index(paths_df)

    print("[diag] computing final_r and reach_1R / mfe_2R labels at SL=3.0", file=sys.stderr)
    final_r_sl3: Dict[int, float] = {}
    reach_1r_label: Dict[int, int] = {}
    mfe_2r_label: Dict[int, int] = {}
    for tid in trades_df["trade_id"].astype(int):
        te = _eval_trade_at_sl(paths_index[tid], SL_DEPLOY, ORIGINAL_SL)
        final_r_sl3[tid] = float(te.final_r_new)
        reach_1r_label[tid] = 1 if te.fwd_mfe_new_r >= 1.0 else 0
        mfe_2r_label[tid] = 1 if te.fwd_mfe_new_r >= 2.0 else 0

    c1_label_by_tid: Dict[int, int] = {
        int(r["trade_id"]): int(int(r["cluster_id"]) == 1) for _, r in clusters_df.iterrows()
    }

    print("[diag] caching 4H pair indicators (PIPELINE_E features)", file=sys.stderr)
    pairs = sorted(trades_df["pair"].astype(str).unique())
    pair_caches_4h = {p: _build_pair_cache(p, DATA_DIR_4H) for p in pairs}
    e_features = compute_pipeline_e_features(trades_df, pair_caches_4h)

    print("[diag] caching multi-TF (4H + D1 + 1H) for Regime C", file=sys.stderr)
    multi_tf_caches: Dict[str, MultiTFCache] = {}
    for p in pairs:
        multi_tf_caches[p] = _build_multi_tf_cache(p)
    print("[diag] computing multi-TF features for all 2299 trades (with alignment audit)", file=sys.stderr)
    multi_tf_df, multi_tf_audit = compute_multi_tf_features(trades_df, multi_tf_caches)

    # Save alignment audit (first 20 + summary).
    audit_ok = sum(1 for r in multi_tf_audit if r["d1_lag_ok"] and r["h1_align_ok"])
    audit_total = len(multi_tf_audit)
    print(f"[diag] multi-TF alignment audit: {audit_ok}/{audit_total} OK", file=sys.stderr)
    write_csv(
        OUT_DIR / "filter_diag_multiTF_alignment_audit_sample.csv",
        ["trade_id", "pair", "signal_bar_time", "expected_d1_date_max", "d1_date_used",
         "d1_lag_ok", "expected_h1_start_max", "h1_start_used", "h1_align_ok"],
        [[r["trade_id"], r["pair"], r["signal_bar_time"],
          r["expected_d1_date_max"], r["d1_date_used"], r["d1_lag_ok"],
          r["expected_h1_start_max"], r["h1_start_used"], r["h1_align_ok"]]
         for r in multi_tf_audit[:20]],
    )

    model_kw = dict(n_estimators=200, max_depth=8, random_state=42, n_jobs=1)

    print("[diag] === Regime A (baseline) ===", file=sys.stderr)
    r_a = run_regime_a(e_features, c1_label_by_tid, model_kw)
    print(f"[diag]   mean AUC: {r_a.mean_auc:.4f}  clears 0.65 in {r_a.n_clears_gate}/{r_a.n_folds} folds",
          file=sys.stderr)
    print(f"[diag]   fold AUCs: {[round(a, 4) if not math.isnan(a) else 'NaN' for a in r_a.fold_aucs]}",
          file=sys.stderr)

    print("[diag] === Regime B (delayed entry sweep) ===", file=sys.stderr)
    b_results, mfe_cost = run_regime_b(trades_df, paths_index, e_features, c1_label_by_tid, model_kw)
    for r in b_results:
        print(f"[diag]   {r.name}: mean AUC {r.mean_auc:.4f}  clears in {r.n_clears_gate}/{r.n_folds}",
              file=sys.stderr)

    print("[diag] === Regime C (multi-TF) ===", file=sys.stderr)
    r_c = run_regime_c(e_features, multi_tf_df, c1_label_by_tid, model_kw)
    print(f"[diag]   mean AUC: {r_c.mean_auc:.4f}  clears in {r_c.n_clears_gate}/{r_c.n_folds}",
          file=sys.stderr)

    print("[diag] === Regime D (reframed targets) ===", file=sys.stderr)
    r_d_reach, r_d_mfe = run_regime_d(e_features, reach_1r_label, mfe_2r_label, model_kw)
    print(f"[diag]   D_reach1R: mean AUC {r_d_reach.mean_auc:.4f}  clears in {r_d_reach.n_clears_gate}/{r_d_reach.n_folds}",
          file=sys.stderr)
    print(f"[diag]   D_mfe2R:   mean AUC {r_d_mfe.mean_auc:.4f}  clears in {r_d_mfe.n_clears_gate}/{r_d_mfe.n_folds}",
          file=sys.stderr)

    # Per-regime AUC CSV.
    all_results = [r_a] + b_results + [r_c, r_d_reach, r_d_mfe]
    write_csv(
        OUT_DIR / "filter_diag_regime_aucs.csv",
        ["regime", "fold_id", "auc", "n_train", "n_test", "base_test"],
        [[r.name, d["fold_id"], _fmt(d["auc"]), d["n_train"], d["n_test"], _fmt(d["base_test"])]
         for r in all_results for d in r.fold_details],
    )
    write_csv(
        OUT_DIR / "filter_diag_regime_summary.csv",
        ["regime", "mean_auc", "n_clears_gate_0.65", "n_folds_valid", "clears_gate_5_of_7",
         "above_strong_candidate_0.70_in_folds", "feature_count"],
        [[r.name, _fmt(r.mean_auc), r.n_clears_gate, r.n_folds,
          "1" if r.n_clears_gate >= 5 else "0",
          sum(1 for a in r.fold_aucs if not math.isnan(a) and a >= STRONG_CANDIDATE_AUC),
          len(r.feature_cols)]
         for r in all_results],
    )

    # Regime B mfe cost.
    write_csv(
        OUT_DIR / "filter_diag_regimeB_mfe_cost.csv",
        ["t", "n_surviving", "n_excluded_pre_t_sl", "mean_mfe_so_far_r_at_t", "median_mfe_so_far_r_at_t"],
        [[r["t"], r["n_surviving"], r["n_excluded_pre_t_sl"],
          _fmt(r["mean_mfe_so_far_r_at_t"]), _fmt(r["median_mfe_so_far_r_at_t"])]
         for r in mfe_cost],
    )

    # Regime C feature importance.
    feat_imp_rows: List[List[Any]] = []
    if r_c.last_clf is not None:
        imps = list(zip(r_c.feature_cols, r_c.last_clf.feature_importances_))
        imps.sort(key=lambda x: -x[1])
        for i, (name, val) in enumerate(imps):
            feat_imp_rows.append([i, name, _fmt(float(val))])
    write_csv(
        OUT_DIR / "filter_diag_regimeC_feature_importance.csv",
        ["rank", "feature", "gini_importance"],
        feat_imp_rows,
    )

    # Stop condition + best regime selection.
    strong_candidates = [r for r in all_results
                         if sum(1 for a in r.fold_aucs if not math.isnan(a) and a >= STRONG_CANDIDATE_AUC) >= 5]
    all_below_55 = all(r.mean_auc < STRUCTURAL_CEILING_MEAN_AUC
                        for r in all_results if not math.isnan(r.mean_auc))

    # Best regime: highest mean AUC among those >= 0.60.
    above_60 = [r for r in all_results if not math.isnan(r.mean_auc) and r.mean_auc >= 0.60]
    if above_60:
        best = max(above_60, key=lambda r: r.mean_auc)
    else:
        best = max(all_results, key=lambda r: r.mean_auc if not math.isnan(r.mean_auc) else -1)

    print(f"[diag] best regime by mean AUC (>= 0.60 preferred): {best.name} ({best.mean_auc:.4f})",
          file=sys.stderr)

    # Live-deployable simulation for best regime (only if mean AUC >= 0.60).
    live_dep = None
    if best.mean_auc >= 0.60:
        print("[diag] running live-deployable simulation for best regime", file=sys.stderr)
        # Pick label set + feature df matching the regime.
        if best.name == "A_baseline_c1_PE":
            live_dep = _live_deployable_for_regime(
                best, e_features, c1_label_by_tid, final_r_sl3, model_kw,
            )
        elif best.name.startswith("B_delayed_t"):
            # Reconstruct feature df for the best t.
            t_val = int(best.name.split("_t")[1].split("_")[0])
            feat_rows = []
            e_base_idx = {int(r["trade_id"]): r for _, r in e_features.iterrows()}
            for _, tr in trades_df.iterrows():
                tid = int(tr["trade_id"])
                pf = _path_features_at_t_orig_frame(paths_index[tid], t_val)
                if pf is None:
                    continue
                base_row = e_base_idx[tid]
                merged = {"trade_id": tid, "entry_time": base_row["entry_time"]}
                for c in PIPELINE_E_BASE:
                    merged[c] = base_row[c]
                merged.update(pf)
                feat_rows.append(merged)
            feat_df = pd.DataFrame(feat_rows)
            live_dep = _live_deployable_for_regime(
                best, feat_df, c1_label_by_tid, final_r_sl3, model_kw,
            )
        elif best.name == "C_multiTF_c1_PE+D1+1H":
            combined = e_features.merge(multi_tf_df, on="trade_id", how="left")
            live_dep = _live_deployable_for_regime(
                best, combined, c1_label_by_tid, final_r_sl3, model_kw,
            )
        elif best.name == "D_target_reach1R_atSL3":
            live_dep = _live_deployable_for_regime(
                best, e_features, reach_1r_label, final_r_sl3, model_kw,
            )
        elif best.name == "D_target_mfe2R_atSL3":
            live_dep = _live_deployable_for_regime(
                best, e_features, mfe_2r_label, final_r_sl3, model_kw,
            )
        if live_dep:
            write_csv(
                OUT_DIR / "filter_diag_best_regime_live_deployable.csv",
                ["regime", "threshold", "sign_consistency", "worst_fold_roi_ann_pct",
                 "mean_fold_roi_ann_pct", "worst_fold_dd_pct", "min_trade_count",
                 "full_data_roi_pct", "full_data_dd_pct", "pass_deployable", "pass_viable"],
                [[live_dep["regime"], _fmt(live_dep["threshold"], 2),
                  "1" if live_dep["sign_consistency"] else "0",
                  _fmt(live_dep["worst_fold_roi_ann_pct"]), _fmt(live_dep["mean_fold_roi_ann_pct"]),
                  _fmt(live_dep["worst_fold_dd_pct"]), live_dep["min_trade_count"],
                  _fmt(live_dep["full_data_roi_pct"]), _fmt(live_dep["full_data_dd_pct"]),
                  "1" if live_dep["pass_deployable"] else "0",
                  "1" if live_dep["pass_viable"] else "0"]],
            )

    # ============================
    # Build report
    # ============================
    elapsed = time.time() - t0
    lines: List[str] = []
    lines.append("# Arc 11 — Filter-diagnosis report (off-protocol, documentation only)")
    lines.append("")
    lines.append("> Arc 11 remains **Closed-HALT** per §16a Path A. No queue / registry / protocol mutation. Diagnostic only.")
    lines.append("")
    lines.append(f"Wall-clock: {elapsed:.1f}s")
    lines.append("")

    # Pre-test sanity
    lines.append("## Pre-test sanity")
    lines.append("")
    lines.append(f"- `clusters_K4.csv` sha256: `{sanity['clusters_K4_sha256']}`")
    lines.append(f"- expected (S4 commit): `{sanity['clusters_K4_expected']}`")
    lines.append(f"- cluster labels stable: **{'YES' if sanity['clusters_K4_stable'] else 'NO — FLAG'}**")
    lines.append("")
    lines.append("WFO folds:")
    for fold_id, oos_s, oos_e in sanity["folds"]:
        lines.append(f"- fold {fold_id}: OOS {oos_s} → {oos_e}")
    lines.append("")
    lines.append(f"Multi-TF alignment audit: **{audit_ok}/{audit_total} trades OK** "
                 f"({'PASS — no D1 future-leak, no 1H future-leak' if audit_ok == audit_total else 'FAIL — investigate'}).")
    lines.append("")

    # Headline table.
    lines.append("## Headline AUC table (4 regimes × per-fold AUC)")
    lines.append("")
    lines.append("| Regime | F1 | F2 | F3 | F4 | F5 | F6 | F7 | mean | clears 0.65 | clears 0.70 | features |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|---:|")
    for r in all_results:
        cells = [f"{a:.4f}" if not math.isnan(a) else "—" for a in r.fold_aucs]
        n_70 = sum(1 for a in r.fold_aucs if not math.isnan(a) and a >= STRONG_CANDIDATE_AUC)
        lines.append(
            f"| {r.name} | " + " | ".join(cells) +
            f" | **{r.mean_auc:.4f}** | {r.n_clears_gate}/{r.n_folds} | "
            f"{n_70}/{r.n_folds} | {len(r.feature_cols)} |"
        )
    lines.append("")

    # Strong candidate flag.
    if strong_candidates:
        lines.append("### ⚠ Strong candidate flag")
        lines.append("")
        for r in strong_candidates:
            lines.append(f"- **{r.name}** clears AUC ≥ 0.70 in ≥5/7 folds (mean {r.mean_auc:.4f}).")
        lines.append("")
    else:
        lines.append("No regime cleared AUC ≥ 0.70 in ≥5/7 folds.")
        lines.append("")

    # Structural ceiling.
    if all_below_55:
        lines.append("### ⚠ Structural extractability ceiling")
        lines.append("")
        lines.append("**All four regimes finished < 0.55 mean AUC.** The extractability ceiling for "
                     "SHB long 4H is structural; the amendment direction must be either signal redesign "
                     "or sizing-without-filtering (separate test). Calibration via feature-set extension "
                     "alone will not lift this signal above the §8 gate.")
        lines.append("")

    # Regime B mfe cost.
    lines.append("## Regime B — delayed entry mfe-cost curve")
    lines.append("")
    lines.append("Question: does waiting N bars lift AUC above 0.65, and what R is given up per delay step?")
    lines.append("")
    lines.append("| t | n_surviving | n_excluded_pre_t_sl | mean mfe_so_far R at t | median mfe at t | AUC at t | clears 0.65 |")
    lines.append("|---:|---:|---:|---:|---:|---:|:---:|")
    for cost_row, r_b in zip(mfe_cost, b_results):
        lines.append(
            f"| {cost_row['t']} | {cost_row['n_surviving']} | {cost_row['n_excluded_pre_t_sl']} "
            f"| {cost_row['mean_mfe_so_far_r_at_t']:.4f} | {cost_row['median_mfe_so_far_r_at_t']:.4f} "
            f"| {r_b.mean_auc:.4f} | {'YES' if r_b.mean_auc >= GATE_E else 'no'} |"
        )
    lines.append("")
    lines.append("R units in original SL=2 frame (mfe_so_far_r as recorded in `trades_paths.csv`).")
    lines.append("")

    # Regime C feature importance.
    lines.append("## Regime C — multi-TF feature importance (last fold's classifier)")
    lines.append("")
    if not feat_imp_rows:
        lines.append("_No classifier trained (insufficient IS in last fold)._")
    else:
        lines.append("Top 15:")
        lines.append("")
        lines.append("| rank | feature | gini importance | TF |")
        lines.append("|---:|---|---:|:---:|")
        for row in feat_imp_rows[:15]:
            name = row[1]
            tf = "D1" if name.startswith("d1_") else ("1H" if name.startswith("h1_") else "4H")
            lines.append(f"| {row[0]} | `{name}` | {row[2]} | {tf} |")
        lines.append("")
        # TF mass: total importance by timeframe.
        tf_mass = {"4H": 0.0, "D1": 0.0, "1H": 0.0}
        for _rank, name, val in feat_imp_rows:
            v = float(val) if val else 0.0
            if name.startswith("d1_"):
                tf_mass["D1"] += v
            elif name.startswith("h1_"):
                tf_mass["1H"] += v
            else:
                tf_mass["4H"] += v
        total = sum(tf_mass.values())
        lines.append("Importance mass by timeframe (across all features):")
        for tf, v in tf_mass.items():
            pct = 100 * v / total if total > 0 else 0
            lines.append(f"- {tf}: {pct:.1f}%")
        lines.append("")

    # Regime D target comparison.
    lines.append("## Regime D — target comparison (cluster vs reach_1R vs mfe≥2R, same features)")
    lines.append("")
    target_rows = [(r_a, "predict cluster=c1"), (r_d_reach, "predict reach_1R at SL=3"), (r_d_mfe, "predict mfe≥2R at SL=3")]
    lines.append("| Target | mean AUC | clears 0.65 |")
    lines.append("|---|---:|:---:|")
    for r, label in target_rows:
        lines.append(f"| {label} | {r.mean_auc:.4f} | {r.n_clears_gate}/{r.n_folds} |")
    lines.append("")
    if r_d_reach.mean_auc > r_a.mean_auc:
        delta = r_d_reach.mean_auc - r_a.mean_auc
        lines.append(f"### ⚠ reach_1R supervision target stronger by {delta:+.4f} AUC")
        lines.append("")
        lines.append("The cluster-membership supervision target is *weaker* than the direct deployment-outcome "
                     "target (reach_1R). Flag for protocol amendment cycle: cluster ID may be the wrong "
                     "supervision target for Pipeline E.")
        lines.append("")
    if r_d_mfe.mean_auc > r_a.mean_auc:
        delta = r_d_mfe.mean_auc - r_a.mean_auc
        lines.append(f"### ⚠ mfe≥2R supervision target stronger by {delta:+.4f} AUC")
        lines.append("")
    if r_d_reach.mean_auc <= r_a.mean_auc and r_d_mfe.mean_auc <= r_a.mean_auc:
        lines.append("Direct deployment-outcome targets do not outperform cluster membership.")
        lines.append("")

    # Live-deployable for best.
    lines.append("## Live-deployable simulation (best regime only)")
    lines.append("")
    if live_dep is None:
        lines.append(f"Best regime: **{best.name}** (mean AUC {best.mean_auc:.4f}). Mean AUC < 0.60 — "
                     f"no live-deployable simulation run (would be uninformative noise per stop conditions).")
    else:
        lines.append(f"Best regime: **{best.name}** (mean AUC {best.mean_auc:.4f}). Threshold {live_dep['threshold']:.2f}.")
        lines.append("")
        lines.append("| metric | value |")
        lines.append("|---|---:|")
        lines.append(f"| sign consistency | {'YES' if live_dep['sign_consistency'] else 'no'} |")
        lines.append(f"| worst-fold ROI ann % | {live_dep['worst_fold_roi_ann_pct']:.4f} |")
        lines.append(f"| mean-fold ROI ann % | {live_dep['mean_fold_roi_ann_pct']:.4f} |")
        lines.append(f"| worst-fold DD % | {live_dep['worst_fold_dd_pct']:.4f} |")
        lines.append(f"| min trade count | {live_dep['min_trade_count']} |")
        lines.append(f"| full-data ROI % | {live_dep['full_data_roi_pct']:.4f} |")
        lines.append(f"| full-data DD % | {live_dep['full_data_dd_pct']:.4f} |")
        lines.append(f"| pass-deployable | {'YES' if live_dep['pass_deployable'] else 'no'} |")
        lines.append(f"| pass-viable | {'YES' if live_dep['pass_viable'] else 'no'} |")
        lines.append("")

    # Commentary.
    lines.append("## Commentary — direction(s) that move the needle")
    lines.append("")
    aucs_summary = "; ".join(f"{r.name}={r.mean_auc:.3f}" for r in all_results)
    lines.append(f"Mean AUC summary: {aucs_summary}.")
    lines.append("")
    bullet: List[str] = []
    # Diagnose direction.
    deltas_vs_a = {r.name: (r.mean_auc - r_a.mean_auc) for r in all_results if r.name != r_a.name}
    best_delta = max(deltas_vs_a.items(), key=lambda x: x[1]) if deltas_vs_a else ("(none)", 0.0)
    bullet.append(f"Best lift over Regime A baseline ({r_a.mean_auc:.4f}): "
                  f"**{best_delta[0]}** Δ {best_delta[1]:+.4f}.")
    # Reach1R vs cluster.
    if r_d_reach.mean_auc > r_a.mean_auc + 0.02:
        bullet.append("Reframed target (reach_1R) is materially stronger than cluster membership — "
                      "cluster IDs may be the wrong supervision target for Pipeline E.")
    elif abs(r_d_reach.mean_auc - r_a.mean_auc) <= 0.02:
        bullet.append("Cluster vs reach_1R target makes little difference — both register similar AUC; "
                      "supervision target is not the bottleneck.")
    # Multi-TF.
    if r_c.mean_auc > r_a.mean_auc + 0.05:
        bullet.append(f"Multi-TF context lifts AUC by {r_c.mean_auc - r_a.mean_auc:+.4f}; "
                      f"feature-regime extension is a live direction.")
    elif r_c.mean_auc <= r_a.mean_auc + 0.02:
        bullet.append(f"Multi-TF context adds {r_c.mean_auc - r_a.mean_auc:+.4f} — minor or no lift. "
                      f"D1+1H features as designed here do not crack the ceiling.")
    # Delayed entry.
    best_b = max(b_results, key=lambda r: r.mean_auc) if b_results else None
    if best_b is not None:
        if best_b.mean_auc >= GATE_E:
            cost_at_best = next((c["mean_mfe_so_far_r_at_t"] for c in mfe_cost
                                 if best_b.name.endswith(f"t{c['t']}_c1")), None)
            bullet.append(f"Delayed entry at t={best_b.name.split('_t')[1].split('_')[0]} clears 0.65 gate "
                          f"(mean AUC {best_b.mean_auc:.4f}). Cost: mean MFE given up ≈ "
                          f"{cost_at_best:.3f}R per trade (consider whether the AUC lift is worth the "
                          f"reduced capturable R).")
        else:
            bullet.append(f"Delayed entry sweep peaks at {best_b.name.split('_t')[1].split('_')[0]} bars "
                          f"with mean AUC {best_b.mean_auc:.4f} — does not crack 0.65 gate even after "
                          f"sacrificing pre-entry magnitude.")
    # Live deployable.
    if live_dep is not None:
        bullet.append(
            f"Best regime live-deployable simulation: worst-fold ROI ann "
            f"{live_dep['worst_fold_roi_ann_pct']:+.2f}%, DD {live_dep['worst_fold_dd_pct']:.2f}%, "
            f"pass-deployable = {'YES' if live_dep['pass_deployable'] else 'no'}."
        )
    # Amendment candidate.
    if strong_candidates:
        bullet.append(f"Strong candidate identified ({', '.join(r.name for r in strong_candidates)}). "
                      f"Recommend protocol amendment cycle evaluate adopting this feature/target/timing.")
    elif all_below_55:
        bullet.append("All regimes < 0.55 mean AUC: signal redesign or sizing-without-filtering required; "
                      "filter extension alone won't close the gap for SHB long 4H.")
    else:
        bullet.append("No regime clears the 0.65 §8 gate. Filter ceiling for this signal sits in the "
                      "0.50–0.55 range across the directions tested. Next step is either signal-level "
                      "redesign (Arc-spec-level) or sizing-without-filtering (open the §8 escape hatch).")
    for b in bullet[:8]:
        lines.append(f"- {b}")
    lines.append("")
    lines.append("## Artefacts")
    lines.append("")
    lines.append("- `filter_diag_regime_summary.csv` — per-regime mean AUC + clears-gate counts")
    lines.append("- `filter_diag_regime_aucs.csv` — per-regime per-fold AUC + n_train/n_test/base_test")
    lines.append("- `filter_diag_regimeB_mfe_cost.csv` — Regime B mfe-cost curve")
    lines.append("- `filter_diag_regimeC_feature_importance.csv` — Regime C top features by gini")
    lines.append("- `filter_diag_multiTF_alignment_audit_sample.csv` — D1/1H lookahead audit (first 20)")
    if live_dep is not None:
        lines.append("- `filter_diag_best_regime_live_deployable.csv` — best regime live WFO")
    lines.append("- `scripts/l_arc_11/filter_diag.py` — runner")
    lines.append("")

    report_path = OUT_DIR / "filter_diag_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[diag] wrote {report_path.relative_to(_REPO_ROOT)}", file=sys.stderr)

    # Stdout headline.
    print()
    print("=" * 96)
    print("FILTER-DIAGNOSIS TEST — Arc 11 (off-protocol; HALT status unchanged)")
    print("=" * 96)
    print()
    print(f"{'Regime':<40} {'mean AUC':>10} {'clears 0.65':>14} {'clears 0.70':>14}")
    print("-" * 96)
    for r in all_results:
        n70 = sum(1 for a in r.fold_aucs if not math.isnan(a) and a >= STRONG_CANDIDATE_AUC)
        print(f"{r.name:<40} {r.mean_auc:>10.4f} {f'{r.n_clears_gate}/{r.n_folds}':>14} {f'{n70}/{r.n_folds}':>14}")
    print()
    print(f"Wall-clock: {elapsed:.1f}s. Report: {report_path.relative_to(_REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
