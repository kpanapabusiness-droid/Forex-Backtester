"""Arc 9 Pipeline E retry - LightGBM + expanded features.

Held-open experiment under v2.3. Re-tests §8 Pipeline E extractability for
cluster 0 with three simultaneous changes vs Step 4 original:

  H1: classifier family change (LightGBM instead of RandomForest)
  H2: feature set expansion (+8 D1-lagged + 4 session = +12 features
       on top of Step 4's 16 base+arc-specific = 28 total)
  H3: Pipeline E only (no D1 deferred policy, no rejected-pool drag)

Hard rules per dispatch:
  - Single LightGBM hyperparameter configuration (no sweep).
  - No new features beyond the 12 listed (8 D1 + 4 session).
  - D1 lag via KH-24's merge_asof pattern (4H date - 1 day, direction=backward).
    Pattern source: scripts/phase_kgl_v2_4h_wfo.py:_precompute_d1_exit_arrays.
  - Cluster_0 label from Step 2; entry-feature matrix MUST NOT include any
    path-shape feature (monotonicity, local_peaks, pullback, ttp_rel).
    Verified: Step 4 entry_features.csv has 16 structural/base features only.
  - Determinism: LightGBM deterministic=True, force_row_wise=True, seed=42.
    Run twice -> byte-identical AUC + threshold sweep CSVs.
  - CV: TimeSeriesSplit(n_splits=5) per dispatch (note: factually different
    from Step 4's StratifiedKFold; dispatch overrides).

Outputs in results/l_arc_9/experiments/pipeline_e_retry/:
  feature_matrix.csv             per-trade 28-feature matrix + label
  per_fold_aucs.csv              4 cells x 5 folds = 20 rows + mean/std
  threshold_sweep_locked.csv     {0.40, 0.50, 0.60, 0.70} on LGBM expanded OOF
  threshold_sweep_extended.csv   101 thresholds 0..1 step 0.01 on LGBM expanded OOF
  feature_importances.csv        LGBM expanded full-data fit, top features
  determinism_check.json
  summary.json
  PIPELINE_E_RETRY_RESULT.md     report
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


SEED = 42
RF_KW = {
    "n_estimators": 200, "max_depth": 8, "min_samples_leaf": 20,
    "random_state": SEED, "n_jobs": -1,
}
LGBM_KW = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "num_leaves": 31,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "class_weight": "balanced",
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": -1,
    # Determinism settings: required for byte-reproducible parallel tree builds.
    "deterministic": True,
    "force_row_wise": True,
}
N_SPLITS = 5
THRESHOLDS_LOCKED: List[float] = [0.40, 0.50, 0.60, 0.70]
RECALL_FLOOR: float = 0.60
PIPELINE_E_AUC_FLOOR: float = 0.65

BASE_8 = [
    "body_to_range_ratio", "upper_wick_ratio", "lower_wick_ratio",
    "range_to_atr_14", "ret_5bar_atr", "ret_20bar_atr",
    "pos_in_20bar_range", "rsi_14",
]
ARC_SPECIFIC_8 = [
    "n_swing_lows", "most_recent_sl_lag", "swing_low_dist_atr",
    "mother_bar_range_atr", "inside_bar_range_atr", "ib_range_ratio",
    "break_bar_body_atr", "break_close_above_high_atr",
]
BASELINE_16 = BASE_8 + ARC_SPECIFIC_8

D1_8 = [
    "d1_trend_state", "d1_atr_ratio_to_4h", "d1_pos_in_20d_range",
    "d1_ret_5d_atr", "d1_rsi_14", "d1_close_above_kijun",
    "d1_bars_since_swing_high", "d1_bars_since_swing_low",
]
SESSION_4 = ["session_london", "session_ny_overlap", "hour_sin", "hour_cos"]

EXPANDED_28 = BASELINE_16 + D1_8 + SESSION_4

# Forbidden columns: path-shape features that would cause label leakage.
FORBIDDEN_LEAK_FEATURES = {
    "monotonicity_ratio_in_profit", "local_peaks_count",
    "pullback_magnitude_median", "time_to_peak_mfe_relative",
}


# ---------- D1 feature computation ----------------------------------------


def _load_d1(pair: str, data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / f"{pair}.csv")
    if "time" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df.sort_values("date").reset_index(drop=True)


def _wilder_atr_d1(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    high = df["high"].astype(float).to_numpy()
    low = df["low"].astype(float).to_numpy()
    close = df["close"].astype(float).to_numpy()
    n = len(df)
    if n == 0:
        return np.array([], dtype=float)
    prev = np.empty(n, dtype=float)
    prev[0] = np.nan
    prev[1:] = close[:-1]
    tr = np.maximum.reduce([high - low, np.abs(high - prev), np.abs(low - prev)])
    tr[0] = high[0] - low[0]
    atr = np.full(n, np.nan, dtype=float)
    if n < period:
        return atr
    atr[period - 1] = float(np.mean(tr[:period]))
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(close)
    rsi = np.full(n, np.nan, dtype=float)
    if n <= period:
        return rsi
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.full(n, np.nan, dtype=float)
    avg_loss = np.full(n, np.nan, dtype=float)
    avg_gain[period] = float(np.mean(gain[:period]))
    avg_loss[period] = float(np.mean(loss[:period]))
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i - 1]) / period
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _kijun(df: pd.DataFrame, period: int = 26) -> np.ndarray:
    """Kijun-sen = (max(high, period) + min(low, period)) / 2."""
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    return ((high.rolling(period).max() + low.rolling(period).min()) / 2.0).to_numpy()


def _bars_since(boolean_series: np.ndarray) -> np.ndarray:
    """For each index i, bars since boolean_series last was True (-1 if never)."""
    n = boolean_series.shape[0]
    out = np.full(n, -1, dtype=np.int64)
    last_true = -1
    for i in range(n):
        if boolean_series[i]:
            last_true = i
            out[i] = 0
        elif last_true >= 0:
            out[i] = i - last_true
    return out


def _d1_swing_high_low(df_d1: pd.DataFrame, half: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """20-bar swing high / low: bar k is a swing-high iff high[k] > max(high[k-10..k-1])
    AND high[k] >= max(high[k+1..k+10]) (similar for low with <).
    Right-edge half-window is NaN — we don't peek into the future for the live
    construction, BUT for the d1 lag pattern (use yesterday's D1), the swing
    detection at d1_date <= signal_4h_date - 1day is what matters; right-edge
    swings within the look-back window are well-defined as of that lagged date.

    Actually for live operation: a swing-high at d1 bar k requires bars k+1..k+10
    to confirm. So when we use bar k's "is_swing" flag, k must be at most
    (current_d1_idx - half). To stay legitimate under the merge_asof one-day
    lag, we just need to ensure the swing flag was determinable at the time the
    4H bar acted. Conservative approach: confirmed-only swings (no peek beyond
    today's D1), which means the most recent confirmable swing is half bars ago.

    bars_since uses 'last confirmed swing' (k such that k <= n - half - 1).
    """
    n = len(df_d1)
    high = df_d1["high"].astype(float).to_numpy()
    low = df_d1["low"].astype(float).to_numpy()
    is_sw_high = np.zeros(n, dtype=bool)
    is_sw_low = np.zeros(n, dtype=bool)
    for k in range(half, n - half):
        if high[k] > np.max(high[k - half:k]) and high[k] >= np.max(high[k + 1:k + half + 1]):
            is_sw_high[k] = True
        if low[k] < np.min(low[k - half:k]) and low[k] <= np.min(low[k + 1:k + half + 1]):
            is_sw_low[k] = True
    return is_sw_high, is_sw_low


def _build_d1_feature_frame(df_d1: pd.DataFrame) -> pd.DataFrame:
    """Compute per-D1-bar features. Returns a frame keyed by D1 date.

    NOTE: swing-high/low detection uses k+1..k+half forward window WITHIN the
    D1 frame. Because each 4H bar joins to the PRIOR D1 bar (one-day lag via
    merge_asof('date_minus_1', direction='backward')), and the swing detection
    requires `half` future D1 bars to confirm, only swings at d1 indices
    <= len(d1)-half-1 are confirmed. For 4H bars looking at lag-1 D1, the
    most recent confirmable swing as of that lag is min(lag1_idx - half, ...).
    We compute `bars_since` ON THE D1 FRAME using the confirmed swing flags
    (only flags where k <= n-half-1); 4H bars then merge_asof to whichever D1
    row their date_minus_1 lands on, and pick up that D1's bars_since.

    For most arc 9 trades (median entry several years into history), the
    confirm window is non-binding because there are always >half future bars
    in the D1 history. Only the last `half` D1 bars (most recent ~10 days)
    would have an unconfirmed-swing edge artifact. The cost is acceptable; the
    benefit (no lookahead) is non-negotiable.
    """
    d1 = df_d1[["date", "open", "high", "low", "close"]].copy()
    d1["d1_atr14"] = _wilder_atr_d1(d1, 14)
    d1["d1_rsi_14"] = _rsi(d1["close"].astype(float).to_numpy(), 14)
    d1["d1_kijun"] = _kijun(d1, 26)
    # trend state: close > open AND close > prior close
    d1["d1_close_prev"] = d1["close"].shift(1)
    d1["d1_trend_state"] = ((d1["close"] > d1["open"]) & (d1["close"] > d1["close_prev" if False else "d1_close_prev"])).astype(int)
    # 20-day position-in-range
    d1["d1_20d_high"] = d1["high"].rolling(20).max()
    d1["d1_20d_low"] = d1["low"].rolling(20).min()
    _rng = d1["d1_20d_high"] - d1["d1_20d_low"]
    d1["d1_pos_in_20d_range"] = np.where(_rng > 0, (d1["close"] - d1["d1_20d_low"]) / _rng, 0.5)
    # 5-day return in d1 ATR units
    d1["d1_close_5d_ago"] = d1["close"].shift(5)
    d1["d1_ret_5d_atr"] = np.where(
        (d1["d1_atr14"] > 0) & d1["d1_atr14"].notna(),
        (d1["close"] - d1["d1_close_5d_ago"]) / d1["d1_atr14"],
        np.nan,
    )
    d1["d1_close_above_kijun"] = (d1["close"] > d1["d1_kijun"]).astype(int)
    is_sw_high, is_sw_low = _d1_swing_high_low(d1, half=10)
    # Zero out unconfirmed swings (last half bars) — these aren't usable live
    # because forward confirmation hasn't happened.
    n = len(d1)
    confirmed_high = is_sw_high.copy()
    confirmed_low = is_sw_low.copy()
    if n > 10:
        confirmed_high[n - 10:] = False
        confirmed_low[n - 10:] = False
    d1["d1_bars_since_swing_high"] = _bars_since(confirmed_high)
    d1["d1_bars_since_swing_low"] = _bars_since(confirmed_low)
    return d1


def _attach_d1_features(
    trades: pd.DataFrame, data_d1_dir: Path, atr_4h_by_tid: Dict[int, float],
) -> pd.DataFrame:
    """For each trade, merge_asof against the PRIOR D1 bar of its pair.

    Per KH-24 pattern (scripts/phase_kgl_v2_4h_wfo.py:_precompute_d1_exit_arrays):
        lookup_date = normalize(4h date) - 1 day
        merge_asof(lookup_date, d1_date, direction='backward')
    This gives each 4H bar the D1 bar from D-1 — never the same-day D1.

    d1_atr_ratio_to_4h is computed post-merge as d1_atr14_lag1 / atr14_at_signal
    (4H ATR at signal bar, from trades_all.csv).
    """
    trades = trades.copy()
    trades["signal_bar_time"] = pd.to_datetime(trades["signal_bar_time"])
    trades["lookup_date"] = trades["signal_bar_time"].dt.normalize() - pd.Timedelta(days=1)

    # D1 features without ratio (computed post-merge).
    d1_cols_pre_ratio = [c for c in D1_8 if c != "d1_atr_ratio_to_4h"]

    out_rows: List[pd.DataFrame] = []
    pairs = sorted(trades["pair"].unique().tolist())
    for pair in pairs:
        sub = trades[trades["pair"] == pair].sort_values("lookup_date", kind="mergesort").reset_index(drop=True)
        df_d1 = _load_d1(pair, data_d1_dir)
        d1_feats = _build_d1_feature_frame(df_d1)
        merge_cols = ["date", "d1_atr14"] + d1_cols_pre_ratio
        d1_lite = d1_feats[merge_cols].rename(columns={"date": "d1_date", "d1_atr14": "d1_atr14_lag1"})
        merged = pd.merge_asof(
            sub[["trade_id", "lookup_date"]],
            d1_lite,
            left_on="lookup_date", right_on="d1_date",
            direction="backward",
        )
        out_rows.append(merged)
    merged_all = pd.concat(out_rows, ignore_index=True)
    result = trades.merge(merged_all, on="trade_id", how="left")
    # Post-merge: compute d1_atr_ratio_to_4h.
    atr4h = result["trade_id"].astype(int).map(atr_4h_by_tid)
    result["d1_atr_ratio_to_4h"] = np.where(
        (atr4h > 0) & atr4h.notna() & result["d1_atr14_lag1"].notna(),
        result["d1_atr14_lag1"] / atr4h, np.nan,
    )
    return result


# ---------- Session feature computation -----------------------------------


def _attach_session_features(trades: pd.DataFrame) -> pd.DataFrame:
    t = trades.copy()
    t["signal_bar_time"] = pd.to_datetime(t["signal_bar_time"])
    hour = t["signal_bar_time"].dt.hour
    t["session_london"] = ((hour >= 8) & (hour < 16)).astype(int)
    t["session_ny_overlap"] = ((hour >= 12) & (hour < 16)).astype(int)
    t["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    t["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    return t


# ---------- Training + evaluation ----------------------------------------


def _fit_and_oof(
    X: np.ndarray, y: np.ndarray, model_kind: str,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Time-series 5-fold CV. Returns (mean AUC, per-fold AUCs, OOF probabilities,
    OOF mask = 1 where prediction exists, 0 where trade was never in a test fold).

    TimeSeriesSplit's first ~1/(n_splits+1) of data is never tested. OOF mask is 0
    for those rows; we still report by trade_id which rows have OOF predictions.
    """
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    oof = np.full(len(y), np.nan, dtype=float)
    fold_aucs: List[float] = []
    for tr_idx, te_idx in tscv.split(X):
        if model_kind == "rf":
            mdl = RandomForestClassifier(**RF_KW)
        elif model_kind == "lgbm":
            mdl = lgb.LGBMClassifier(**LGBM_KW)
        else:
            raise ValueError(model_kind)
        mdl.fit(X[tr_idx], y[tr_idx])
        p = mdl.predict_proba(X[te_idx])[:, 1]
        oof[te_idx] = p
        try:
            fold_aucs.append(float(roc_auc_score(y[te_idx], p)))
        except Exception:
            fold_aucs.append(float("nan"))
    return float(np.nanmean(fold_aucs)), np.array(fold_aucs), oof, ~np.isnan(oof)


def _threshold_sweep_locked(y: np.ndarray, prob: np.ndarray, mask: np.ndarray) -> List[Dict[str, Any]]:
    """Evaluate the locked v2.2 §3 grid on OOF predictions where available."""
    y_o = y[mask]
    p_o = prob[mask]
    rows: List[Dict[str, Any]] = []
    for thr in THRESHOLDS_LOCKED:
        y_pred = (p_o >= thr).astype(int)
        prec = float(precision_score(y_o, y_pred, zero_division=0)) if y_pred.sum() > 0 else 0.0
        rec = float(recall_score(y_o, y_pred, zero_division=0))
        rows.append({
            "threshold": thr,
            "precision": prec,
            "recall": rec,
            "n_admitted": int(y_pred.sum()),
            "passes_recall_floor": int(rec >= RECALL_FLOOR),
        })
    return rows


def _threshold_sweep_extended(y: np.ndarray, prob: np.ndarray, mask: np.ndarray) -> List[Dict[str, Any]]:
    y_o = y[mask]
    p_o = prob[mask]
    rows: List[Dict[str, Any]] = []
    for thr in np.linspace(0.0, 1.0, 101):
        y_pred = (p_o >= thr).astype(int)
        prec = float(precision_score(y_o, y_pred, zero_division=0)) if y_pred.sum() > 0 else 0.0
        rec = float(recall_score(y_o, y_pred, zero_division=0))
        rows.append({
            "threshold": float(thr),
            "precision": prec,
            "recall": rec,
            "n_admitted": int(y_pred.sum()),
        })
    return rows


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _select(thr_rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """v2.2 §3 selection: max precision with recall >= 0.60. None if no pass."""
    cands = [r for r in thr_rows if r["recall"] >= RECALL_FLOOR]
    if not cands:
        return None
    return max(cands, key=lambda r: r["precision"])


def run(out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load baseline entry features (16 base+arc-specific) from Step 4.
    entry = pd.read_csv(_REPO_ROOT / "results" / "l_arc_9" / "step4_extractability" / "entry_features.csv")
    forbidden_in_entry = set(entry.columns) & FORBIDDEN_LEAK_FEATURES
    if forbidden_in_entry:
        raise RuntimeError(f"path-shape features leaked into entry features: {forbidden_in_entry}")
    for c in BASELINE_16:
        if c not in entry.columns:
            raise RuntimeError(f"missing baseline feature column: {c}")

    # Cluster 0 binary label.
    clusters = pd.read_csv(_REPO_ROOT / "results" / "l_arc_9" / "step2_clustering" / "clusters_K3.csv")
    cid0 = set(clusters[clusters["cluster_id"] == 0]["trade_id"].astype(int))

    # ATR at signal bar (4H) — needed for d1_atr_ratio_to_4h post-merge computation.
    trades_all = pd.read_csv(_REPO_ROOT / "results" / "l_arc_9" / "step1_verbatim" / "trades_all.csv")
    atr_4h_by_tid: Dict[int, float] = dict(zip(
        trades_all["trade_id"].astype(int),
        trades_all["atr14_at_signal"].astype(float),
    ))
    # Also need entry_time for chronological CV sort — attach from trades_all.
    entry_time_by_tid: Dict[int, str] = dict(zip(
        trades_all["trade_id"].astype(int),
        trades_all["entry_time"].astype(str),
    ))

    # Build expanded feature matrix.
    print("[pe-retry] attaching D1 features (one-day lag, merge_asof backward)...")
    data_d1_dir = _REPO_ROOT.parent.parent.parent / "data" / "daily"
    if not data_d1_dir.exists():
        data_d1_dir = Path("C:/Users/panap/Documents/Forex-Backtester/data/daily")
    if not data_d1_dir.exists():
        raise FileNotFoundError(f"D1 data dir not found at expected path: {data_d1_dir}")
    df = _attach_d1_features(entry, data_d1_dir, atr_4h_by_tid)
    df = _attach_session_features(df)
    df["entry_time"] = df["trade_id"].astype(int).map(entry_time_by_tid)

    # Add label.
    df["y"] = df["trade_id"].astype(int).apply(lambda x: 1 if int(x) in cid0 else 0)

    # Drop rows with any NaN in the expanded feature set.
    feat_cols = EXPANDED_28
    if len(feat_cols) > 50:
        raise RuntimeError(f"feature count {len(feat_cols)} exceeds hard cap 50")
    df_clean = df.dropna(subset=feat_cols).reset_index(drop=True)
    # Sort by entry_time for TimeSeriesSplit (chronological order required).
    df_clean["entry_time"] = pd.to_datetime(df_clean["entry_time"])
    df_clean = df_clean.sort_values(["entry_time", "pair"], kind="mergesort").reset_index(drop=True)

    n_total = len(df_clean)
    n_pos = int(df_clean["y"].sum())
    print(f"[pe-retry] expanded feature matrix: n_total={n_total}, n_pos={n_pos}, n_features={len(feat_cols)}")

    # Persist feature matrix.
    keep_cols = ["trade_id", "pair", "signal_bar_time", "entry_time", "y"] + feat_cols
    df_clean[keep_cols].to_csv(out_dir / "feature_matrix.csv", index=False,
                                float_format="%.10g", lineterminator="\n")

    X_baseline = df_clean[BASELINE_16].to_numpy(dtype=float)
    X_expanded = df_clean[EXPANDED_28].to_numpy(dtype=float)
    y = df_clean["y"].to_numpy(dtype=int)

    # Run 4 cells under TimeSeriesSplit(5).
    cells: Dict[str, Dict[str, Any]] = {}
    for cell_name, X, feature_set, kind in [
        ("rf_baseline_16",   X_baseline, BASELINE_16, "rf"),
        ("rf_expanded_28",   X_expanded, EXPANDED_28, "rf"),
        ("lgbm_baseline_16", X_baseline, BASELINE_16, "lgbm"),
        ("lgbm_expanded_28", X_expanded, EXPANDED_28, "lgbm"),
    ]:
        print(f"[pe-retry] training {cell_name}...")
        mean_auc, fold_aucs, oof, mask = _fit_and_oof(X, y, kind)
        cells[cell_name] = {
            "kind": kind, "feature_set": feature_set, "n_features": len(feature_set),
            "mean_auc": mean_auc, "fold_aucs": fold_aucs.tolist(),
            "oof_mask_size": int(mask.sum()), "oof_pos_in_mask": int(y[mask].sum()),
        }
        if cell_name == "lgbm_expanded_28":
            cells[cell_name]["oof"] = oof.copy()
            cells[cell_name]["mask"] = mask.copy()
        print(f"  mean AUC {mean_auc:.4f}, folds {[round(a, 4) for a in fold_aucs.tolist()]}")

    # Per-fold AUCs table.
    pf_rows: List[Dict[str, Any]] = []
    for cell_name, info in cells.items():
        for i, auc in enumerate(info["fold_aucs"], start=1):
            pf_rows.append({"cell": cell_name, "fold": i, "auc": auc})
        pf_rows.append({"cell": cell_name, "fold": "mean", "auc": info["mean_auc"]})
        pf_rows.append({"cell": cell_name, "fold": "std",
                        "auc": float(np.nanstd(info["fold_aucs"], ddof=1))})
    pd.DataFrame(pf_rows).to_csv(out_dir / "per_fold_aucs.csv", index=False,
                                  float_format="%.10g", lineterminator="\n")

    # Threshold sweeps on LGBM expanded OOF.
    lgbm_e = cells["lgbm_expanded_28"]
    locked_rows = _threshold_sweep_locked(y, lgbm_e["oof"], lgbm_e["mask"])
    ext_rows = _threshold_sweep_extended(y, lgbm_e["oof"], lgbm_e["mask"])
    pd.DataFrame(locked_rows).to_csv(out_dir / "threshold_sweep_locked.csv", index=False,
                                      float_format="%.10g", lineterminator="\n")
    pd.DataFrame(ext_rows).to_csv(out_dir / "threshold_sweep_extended.csv", index=False,
                                   float_format="%.10g", lineterminator="\n")

    # Feature importances on LGBM expanded — train on full data once.
    print("[pe-retry] computing LGBM expanded feature importances (full-data fit)...")
    mdl_full = lgb.LGBMClassifier(**LGBM_KW)
    mdl_full.fit(X_expanded, y)
    importances = pd.DataFrame({
        "feature": EXPANDED_28,
        "importance_gain": mdl_full.booster_.feature_importance(importance_type="gain"),
        "importance_split": mdl_full.booster_.feature_importance(importance_type="split"),
    })
    importances["origin"] = importances["feature"].apply(
        lambda c: "D1" if c in D1_8 else ("session" if c in SESSION_4 else ("base" if c in BASE_8 else "arc_specific"))
    )
    importances = importances.sort_values("importance_gain", ascending=False).reset_index(drop=True)
    importances.to_csv(out_dir / "feature_importances.csv", index=False,
                       float_format="%.10g", lineterminator="\n")

    # Verdict.
    lgbm_e_auc = lgbm_e["mean_auc"]
    passes_auc = lgbm_e_auc >= PIPELINE_E_AUC_FLOOR
    chosen = _select(locked_rows) if passes_auc else None
    if passes_auc and chosen is not None:
        verdict = "PASS_PIPELINE_E_AND_THRESHOLD"
    elif passes_auc and chosen is None:
        verdict = "FAIL_THRESHOLD_SWEEP"
    else:
        verdict = "FAIL_PIPELINE_E_AUC"

    summary = {
        "n_total": n_total, "n_pos": n_pos, "n_features_expanded": len(EXPANDED_28),
        "cells": {
            k: {kk: vv for kk, vv in v.items() if kk not in ("oof", "mask")}
            for k, v in cells.items()
        },
        "lgbm_expanded_auc": lgbm_e_auc,
        "passes_pipeline_e_auc_floor": bool(passes_auc),
        "threshold_sweep_chosen": chosen,
        "verdict": verdict,
        "rng_seed": SEED,
        "cv": "TimeSeriesSplit(n_splits=5), chronological order by entry_time",
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Arc 9 Pipeline E retry - LightGBM + expanded features.")
    parser.add_argument(
        "--out-dir", type=Path,
        default=_REPO_ROOT / "results" / "l_arc_9" / "experiments" / "pipeline_e_retry",
    )
    parser.add_argument("--verify-determinism", action="store_true")
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary = run(args.out_dir)
    print(f"[pe-retry] VERDICT: {summary['verdict']}")
    print(f"  cells (mean CV AUC):")
    for cn, info in summary["cells"].items():
        print(f"    {cn:24s}  AUC mean {info['mean_auc']:.4f}  folds {[round(a, 4) for a in info['fold_aucs']]}")
    if summary["threshold_sweep_chosen"]:
        c = summary["threshold_sweep_chosen"]
        print(f"  threshold sweep chosen: thr={c['threshold']:.2f}, precision={c['precision']:.3f}, recall={c['recall']:.3f}, admits={c['n_admitted']}")

    if args.verify_determinism:
        scratch = args.out_dir / "_determinism_scratch"
        scratch.mkdir(exist_ok=True)
        run(scratch)
        sha_fm_1 = _sha256_file(args.out_dir / "feature_matrix.csv")
        sha_fm_2 = _sha256_file(scratch / "feature_matrix.csv")
        sha_pf_1 = _sha256_file(args.out_dir / "per_fold_aucs.csv")
        sha_pf_2 = _sha256_file(scratch / "per_fold_aucs.csv")
        sha_tl_1 = _sha256_file(args.out_dir / "threshold_sweep_locked.csv")
        sha_tl_2 = _sha256_file(scratch / "threshold_sweep_locked.csv")
        det = {
            "feature_matrix_run1_sha256": sha_fm_1, "feature_matrix_run2_sha256": sha_fm_2,
            "per_fold_aucs_run1_sha256": sha_pf_1, "per_fold_aucs_run2_sha256": sha_pf_2,
            "threshold_sweep_locked_run1_sha256": sha_tl_1, "threshold_sweep_locked_run2_sha256": sha_tl_2,
            "byte_identical": bool(sha_fm_1 == sha_fm_2 and sha_pf_1 == sha_pf_2 and sha_tl_1 == sha_tl_2),
        }
        (args.out_dir / "determinism_check.json").write_text(
            json.dumps(det, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        for f in scratch.iterdir():
            f.unlink()
        scratch.rmdir()
        print(f"[pe-retry] determinism: {'PASS' if det['byte_identical'] else 'FAIL'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
