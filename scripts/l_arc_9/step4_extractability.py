"""Arc 9 - Step 4 extractability (L_ARC_PROTOCOL §8, v2.2 §3 no max-F1 fallback,
v2.2 §2 lift cap, v2.3 §4 pre_t_sl in policy YAML).

Reads:
    results/l_arc_9/step1_verbatim/{trades_all,trades_paths,prefilter_events}.csv
    results/l_arc_9/step2_clustering/{path_features,clusters_K{k},archetype_assignments}.csv
    results/l_arc_9/step3_capturability/{capturability_pass_list,archetype_summaries}.csv
    configs/wfo_l_arc_9.yaml
    4H pair data (path from config)

Produces in results/l_arc_9/step4_extractability/:
    entry_features.csv                 per-trade entry-time features (8 base + arc-specific)
    predictability_angle_E.csv         per archetype: LR AUC, RF AUC, RF importances
    predictability_angle_D1.csv        per archetype x t: RF AUC, exclusion rate
    threshold_sweep_E_<label>.csv      threshold sweep at chosen Pipeline E classifier
    threshold_sweep_D1_<label>.csv     threshold sweep at chosen Pipeline D1 classifier
    extractability_pass_list.csv       surviving archetypes + pipeline assignment
    archetype_<label>_E_classifier.joblib  (when E passes)
    archetype_<label>_E_filter.yaml        (when E passes)
    archetype_<label>_D1_classifier.joblib (when D1 passes)
    archetype_<label>_D1_policy.yaml       (when D1 passes; carries pre_t_sl_atr_multiplier)
    STEP4_SUMMARY.md

Per §8 + v2.2 §3:
- Pipeline E: 5-fold CV RF AUC >= 0.65 (logistic informational).
- Pipeline D1: smallest t in {1,2,3,4,5,10} with RF AUC >= 0.60 AND exclusion <= 30%.
- Threshold sweep {0.40, 0.50, 0.60, 0.70}: select max precision with recall >= 0.60.
  No threshold passes => archetype fails Step 4 (no max-F1 fallback).
- §15 sample-size floor: n_pos >= 50 AND n_total >= 200.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

E_AUC_FLOOR: float = 0.65
D1_AUC_FLOOR: float = 0.60
D1_EXCLUSION_CEILING: float = 0.30
N_POS_FLOOR: int = 50
N_TOTAL_FLOOR: int = 200
THRESHOLD_SWEEP: List[float] = [0.40, 0.50, 0.60, 0.70]
RECALL_FLOOR: float = 0.60
D1_T_VALUES: List[int] = [1, 2, 3, 4, 5, 10]

RF_KW = {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 20, "random_state": 42, "n_jobs": -1}
LR_KW = {"max_iter": 2000, "random_state": 42}
CV_KW = {"n_splits": 5, "shuffle": True, "random_state": 42}


def _load_pair_csv(pair: str, data_dir: Path) -> pd.DataFrame:
    fpath = data_dir / f"{pair}.csv"
    df = pd.read_csv(fpath)
    if "time" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def _slice_window(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    s = pd.Timestamp(start)
    e = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return df[(df["date"] >= s) & (df["date"] <= e)].reset_index(drop=True)


def _wilder_atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
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


def compute_entry_features(
    cfg: dict, trades_all: pd.DataFrame,
) -> pd.DataFrame:
    """Compute 8 base entry features + arc-specific features at signal bar t.

    Base 8: body_to_range_ratio, upper_wick_ratio, lower_wick_ratio, range_to_atr_14,
            ret_5bar_atr, ret_20bar_atr, pos_in_20bar_range, rsi_14.
    Arc-specific (from signal columns + structural):
            n_swing_lows, most_recent_sl_lag, swing_low_dist_atr,
            mother_bar_range_atr, inside_bar_range_atr, ib_range_ratio,
            break_bar_body_atr, break_close_above_high_atr.
    """
    data_4h_path = cfg["data"]["data_dirs"]["4H"]
    data_dir = Path(data_4h_path) if Path(data_4h_path).is_absolute() else _REPO_ROOT / data_4h_path
    date_start = str(cfg["data"]["date_start"])
    date_end = str(cfg["data"]["date_end"])

    pairs = sorted(trades_all["pair"].unique().tolist())
    cache: Dict[str, pd.DataFrame] = {}
    for pair in pairs:
        df_raw = _load_pair_csv(pair, data_dir)
        df = _slice_window(df_raw, date_start, date_end).reset_index(drop=True)
        df["atr14"] = _wilder_atr(df, 14)
        df["rsi14"] = _rsi(df["close"].astype(float).to_numpy(), 14)
        cache[pair] = df

    trades = trades_all.copy()
    trades["signal_bar_time"] = pd.to_datetime(trades["signal_bar_time"])
    out_rows: List[Dict[str, Any]] = []
    for _, row in trades.iterrows():
        pair = row["pair"]
        sig_t = row["signal_bar_time"]
        df = cache[pair]
        idx_arr = np.where(df["date"].to_numpy() == np.datetime64(sig_t))[0]
        if idx_arr.size == 0:
            continue
        t = int(idx_arr[0])
        n = len(df)
        if t < 21 or t >= n:  # need 20-bar lookback
            continue
        atr = float(df["atr14"].iloc[t]) if pd.notna(df["atr14"].iloc[t]) else float("nan")
        op_t = float(df["open"].iloc[t])
        hi_t = float(df["high"].iloc[t])
        lo_t = float(df["low"].iloc[t])
        cl_t = float(df["close"].iloc[t])
        rng_t = max(hi_t - lo_t, 1e-12)
        body = abs(cl_t - op_t)
        upper_wick = hi_t - max(cl_t, op_t)
        lower_wick = min(cl_t, op_t) - lo_t
        body_to_range = body / rng_t
        upper_wick_ratio = upper_wick / rng_t
        lower_wick_ratio = lower_wick / rng_t
        range_to_atr = rng_t / atr if atr > 0 else float("nan")
        cl_5_ago = float(df["close"].iloc[t - 5])
        cl_20_ago = float(df["close"].iloc[t - 20])
        ret_5bar_atr = (cl_t - cl_5_ago) / atr if atr > 0 else float("nan")
        ret_20bar_atr = (cl_t - cl_20_ago) / atr if atr > 0 else float("nan")
        win_lo = df["low"].iloc[t - 20:t].min()
        win_hi = df["high"].iloc[t - 20:t].max()
        pos_in_20bar_range = (cl_t - win_lo) / max(win_hi - win_lo, 1e-12)
        rsi14 = float(df["rsi14"].iloc[t]) if pd.notna(df["rsi14"].iloc[t]) else float("nan")

        # Arc-specific: structural features for IB-trend signal.
        hi_im1 = float(df["high"].iloc[t - 1])
        lo_im1 = float(df["low"].iloc[t - 1])
        hi_im2 = float(df["high"].iloc[t - 2])
        lo_im2 = float(df["low"].iloc[t - 2])
        mother_range = max(hi_im2 - lo_im2, 1e-12)
        inside_range = max(hi_im1 - lo_im1, 1e-12)
        mother_bar_range_atr = mother_range / atr if atr > 0 else float("nan")
        inside_bar_range_atr = inside_range / atr if atr > 0 else float("nan")
        ib_range_ratio = inside_range / mother_range
        break_bar_body_atr = body / atr if atr > 0 else float("nan")
        break_close_above_high_atr = (cl_t - hi_im1) / atr if atr > 0 else float("nan")
        swing_low_dist_atr = (
            (cl_t - float(row["swing_low_used"])) / atr
            if (atr > 0 and pd.notna(row["swing_low_used"])) else float("nan")
        )
        n_sl = int(row["n_swing_lows"])
        mr_lag = int(row["most_recent_sl_lag"])

        out_rows.append({
            "trade_id": int(row["trade_id"]),
            "pair": pair,
            "signal_bar_time": str(sig_t),
            # Base 8.
            "body_to_range_ratio": body_to_range,
            "upper_wick_ratio": upper_wick_ratio,
            "lower_wick_ratio": lower_wick_ratio,
            "range_to_atr_14": range_to_atr,
            "ret_5bar_atr": ret_5bar_atr,
            "ret_20bar_atr": ret_20bar_atr,
            "pos_in_20bar_range": pos_in_20bar_range,
            "rsi_14": rsi14,
            # Arc-specific.
            "n_swing_lows": n_sl,
            "most_recent_sl_lag": mr_lag,
            "swing_low_dist_atr": swing_low_dist_atr,
            "mother_bar_range_atr": mother_bar_range_atr,
            "inside_bar_range_atr": inside_bar_range_atr,
            "ib_range_ratio": ib_range_ratio,
            "break_bar_body_atr": break_bar_body_atr,
            "break_close_above_high_atr": break_close_above_high_atr,
        })
    return pd.DataFrame(out_rows)


BASE_8 = [
    "body_to_range_ratio", "upper_wick_ratio", "lower_wick_ratio",
    "range_to_atr_14", "ret_5bar_atr", "ret_20bar_atr",
    "pos_in_20bar_range", "rsi_14",
]
ARC_SPECIFIC = [
    "n_swing_lows", "most_recent_sl_lag", "swing_low_dist_atr",
    "mother_bar_range_atr", "inside_bar_range_atr", "ib_range_ratio",
    "break_bar_body_atr", "break_close_above_high_atr",
]
FULL_ENTRY_FEATURES = BASE_8 + ARC_SPECIFIC


def compute_path_so_far_features(
    trades_paths: pd.DataFrame, t: int,
) -> pd.DataFrame:
    """Per-trade path-so-far features at bar offset t."""
    held = trades_paths[(trades_paths["is_held"] == 1) & (trades_paths["bar_offset"] <= t)]
    held = held.sort_values(["trade_id", "bar_offset"], kind="mergesort")
    rows: List[Dict[str, Any]] = []
    for tid, sub in held.groupby("trade_id", sort=True):
        if sub["bar_offset"].max() < t:
            continue  # trade exited before bar t
        close_r = sub["close_r"].to_numpy(dtype=float)
        mfe_r = sub["mfe_so_far_r"].to_numpy(dtype=float)
        mae_r = sub["mae_so_far_r"].to_numpy(dtype=float)
        bars_in_profit = int((close_r > 0).sum())
        local_peaks_so_far = int(np.sum(mfe_r[1:] > mfe_r[:-1])) if mfe_r.size > 1 else 0
        in_profit = close_r[close_r > 0]
        mono_so_far = float(np.mean(in_profit[1:] >= in_profit[:-1])) if in_profit.size >= 2 else 0.0
        velocity = float(close_r[t] / max(t, 1)) if close_r.size > t else float(close_r[-1] / max(t, 1))
        rows.append({
            "trade_id": int(tid),
            f"close_r_at_t{t}": float(close_r[t]) if close_r.size > t else float(close_r[-1]),
            f"mfe_so_far_r_at_t{t}": float(mfe_r[t]) if mfe_r.size > t else float(mfe_r[-1]),
            f"mae_so_far_r_at_t{t}": float(mae_r[t]) if mae_r.size > t else float(mae_r[-1]),
            f"bars_in_profit_at_t{t}": bars_in_profit,
            f"local_peaks_so_far_at_t{t}": local_peaks_so_far,
            f"monotonicity_so_far_at_t{t}": mono_so_far,
            f"velocity_first_t{t}": velocity,
        })
    return pd.DataFrame(rows)


def _cv_auc(X: np.ndarray, y: np.ndarray, model_kind: str) -> Tuple[float, np.ndarray, np.ndarray]:
    """Return (mean CV AUC, per-fold AUCs, out-of-fold probabilities).

    model_kind in {"rf", "lr"}.
    """
    skf = StratifiedKFold(**CV_KW)
    fold_aucs: List[float] = []
    oof = np.zeros(len(y), dtype=float)
    for tr_idx, va_idx in skf.split(X, y):
        if model_kind == "rf":
            mdl = RandomForestClassifier(**RF_KW)
            mdl.fit(X[tr_idx], y[tr_idx])
            p = mdl.predict_proba(X[va_idx])[:, 1]
        else:
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X[tr_idx])
            mdl = LogisticRegression(**LR_KW)
            mdl.fit(Xs, y[tr_idx])
            p = mdl.predict_proba(scaler.transform(X[va_idx]))[:, 1]
        oof[va_idx] = p
        try:
            fold_aucs.append(float(roc_auc_score(y[va_idx], p)))
        except Exception:
            fold_aucs.append(float("nan"))
    return float(np.nanmean(fold_aucs)), np.array(fold_aucs), oof


def _threshold_sweep(
    y: np.ndarray, oof_prob: np.ndarray, thresholds: List[float],
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    for thr in thresholds:
        y_pred = (oof_prob >= thr).astype(int)
        if y_pred.sum() == 0:
            prec = 0.0
        else:
            prec = float(precision_score(y, y_pred, zero_division=0))
        rec = float(recall_score(y, y_pred, zero_division=0))
        tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
        rows.append({
            "threshold": thr,
            "precision": prec,
            "recall": rec,
            "n_admitted": int(y_pred.sum()),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "passes_recall_floor": int(rec >= RECALL_FLOOR),
        })
    # v2.2 §3: select max precision with recall >= 0.60. No max-F1 fallback.
    candidates = [r for r in rows if r["recall"] >= RECALL_FLOOR]
    if not candidates:
        return rows, None
    chosen = max(candidates, key=lambda r: r["precision"])
    return rows, chosen


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Arc 9 Step 4 extractability.")
    parser.add_argument("-c", "--config", type=Path, default=_REPO_ROOT / "configs" / "wfo_l_arc_9.yaml")
    parser.add_argument(
        "--step1-dir", type=Path, default=_REPO_ROOT / "results" / "l_arc_9" / "step1_verbatim",
    )
    parser.add_argument(
        "--step2-dir", type=Path, default=_REPO_ROOT / "results" / "l_arc_9" / "step2_clustering",
    )
    parser.add_argument(
        "--step3-dir", type=Path, default=_REPO_ROOT / "results" / "l_arc_9" / "step3_capturability",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=_REPO_ROOT / "results" / "l_arc_9" / "step4_extractability",
    )
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    trades_all = pd.read_csv(args.step1_dir / "trades_all.csv")
    trades_paths = pd.read_csv(args.step1_dir / "trades_paths.csv")
    clusters = pd.read_csv(args.step2_dir / "clusters_K3.csv")
    pass_list = pd.read_csv(args.step3_dir / "capturability_pass_list.csv")
    arch_summaries = pd.read_csv(args.step3_dir / "archetype_summaries.csv")

    # Build entry features once.
    features_path = args.out_dir / "entry_features.csv"
    if features_path.exists():
        entry_features = pd.read_csv(features_path)
        print(f"[step 4] loaded cached entry_features.csv ({len(entry_features)} rows)")
    else:
        print("[step 4] computing entry-time features (8 base + 8 arc-specific)...")
        entry_features = compute_entry_features(cfg, trades_all)
        entry_features.to_csv(features_path, index=False, float_format="%.10g", lineterminator="\n")
        print(f"[step 4] entry_features.csv written ({len(entry_features)} rows)")

    # Process each surviving (cluster, mode) row from capturability_pass_list.
    angle_E_rows: List[Dict[str, Any]] = []
    angle_D1_rows: List[Dict[str, Any]] = []
    extract_pass: List[Dict[str, Any]] = []
    summary_md_rows: List[str] = []

    for _, surv in pass_list.iterrows():
        label = str(surv["label_archetype_or_cluster"])
        cid = int(surv["cluster_id"])
        mode = str(surv["evaluation_mode"])
        sel_SL = float(surv["selected_SL_atr_mult"])
        pre_t_sl = float(surv["pre_t_sl_atr_multiplier"])
        # Build positive class.
        if mode == "individual":
            pos_trade_ids = set(clusters[clusters["cluster_id"] == cid]["trade_id"].astype(int))
        else:
            arch_label = str(surv["archetype_label"])
            arch_cluster_ids = set()
            arch_df = pd.read_csv(args.step2_dir / "archetype_assignments.csv")
            arch_cluster_ids = set(arch_df[arch_df["archetype_label"] == arch_label]["cluster_id"].astype(int))
            pos_trade_ids = set(clusters[clusters["cluster_id"].isin(arch_cluster_ids)]["trade_id"].astype(int))
        df = entry_features.copy()
        df["y"] = df["trade_id"].apply(lambda x: 1 if int(x) in pos_trade_ids else 0)
        df = df.dropna(subset=FULL_ENTRY_FEATURES).reset_index(drop=True)
        n_pos = int(df["y"].sum())
        n_total = int(len(df))
        n_neg = n_total - n_pos
        sample_ok = (n_pos >= N_POS_FLOOR and n_total >= N_TOTAL_FLOOR)

        # ---------------- Pipeline E ----------------
        e_status = "fail"
        e_threshold_info: Optional[Dict[str, Any]] = None
        e_rf_auc = float("nan")
        e_lr_auc = float("nan")
        e_fold_aucs = np.array([])
        e_oof: np.ndarray = np.array([])
        e_feature_importance: Dict[str, float] = {}
        if sample_ok:
            X = df[FULL_ENTRY_FEATURES].to_numpy(dtype=float)
            y = df["y"].to_numpy(dtype=int)
            e_rf_auc, e_fold_aucs, e_oof = _cv_auc(X, y, "rf")
            e_lr_auc, _, _ = _cv_auc(X, y, "lr")
            # Feature importances - train on full dataset.
            rf_full = RandomForestClassifier(**RF_KW)
            rf_full.fit(X, y)
            for f, w in sorted(zip(FULL_ENTRY_FEATURES, rf_full.feature_importances_),
                                key=lambda kv: -kv[1]):
                e_feature_importance[f] = float(w)
            angle_E_rows.append({
                "label": label,
                "cluster_id": cid,
                "n_total": n_total,
                "n_pos": n_pos,
                "n_neg": n_neg,
                "rf_auc_mean": e_rf_auc,
                "rf_auc_folds": list(e_fold_aucs.tolist()),
                "lr_auc_mean": e_lr_auc,
                "rf_logit_gap": e_rf_auc - e_lr_auc,
                "passes_e_floor": int(e_rf_auc >= E_AUC_FLOOR),
                "top_features": dict(list(e_feature_importance.items())[:10]),
            })
            if e_rf_auc >= E_AUC_FLOOR:
                # Threshold sweep on OOF probabilities.
                tsweep_rows, chosen = _threshold_sweep(y, e_oof, THRESHOLD_SWEEP)
                pd.DataFrame(tsweep_rows).to_csv(
                    args.out_dir / f"threshold_sweep_E_{label}.csv",
                    index=False, float_format="%.10g", lineterminator="\n",
                )
                if chosen is None:
                    e_status = "fail_threshold_sweep"
                else:
                    e_status = "pass"
                    e_threshold_info = chosen
                    # Persist artefact: classifier fit on FULL data + filter YAML.
                    joblib.dump(rf_full, args.out_dir / f"archetype_{label}_E_classifier.joblib")
                    (args.out_dir / f"archetype_{label}_E_filter.yaml").write_text(
                        yaml.safe_dump({
                            "label": label,
                            "cluster_id": cid,
                            "evaluation_mode": mode,
                            "pipeline": "E",
                            "feature_set": FULL_ENTRY_FEATURES,
                            "threshold": float(chosen["threshold"]),
                            "rf_auc_mean": float(e_rf_auc),
                            "rf_auc_folds": [float(x) for x in e_fold_aucs.tolist()],
                            "lr_auc_mean": float(e_lr_auc),
                            "precision_at_threshold": float(chosen["precision"]),
                            "recall_at_threshold": float(chosen["recall"]),
                            "selected_SL_atr_multiplier": float(sel_SL),
                            "rf_kwargs": RF_KW,
                            "feature_importances_top_10": dict(list(e_feature_importance.items())[:10]),
                        }, sort_keys=False),
                        encoding="utf-8",
                    )

        # ---------------- Pipeline D1 ----------------
        d1_status = "fail"
        d1_chosen_t: Optional[int] = None
        d1_threshold_info: Optional[Dict[str, Any]] = None
        d1_rf_auc = float("nan")
        d1_fold_aucs = np.array([])
        d1_exclusion = float("nan")
        d1_oof: np.ndarray = np.array([])
        d1_features: List[str] = []
        if sample_ok:
            # Sweep t in {1,2,3,4,5,10}; pick smallest t with AUC >= 0.60 AND exclusion <= 30%.
            d1_candidates: List[Dict[str, Any]] = []
            for t in D1_T_VALUES:
                psf = compute_path_so_far_features(trades_paths, t)
                # Restrict to trades surviving to bar t.
                merged = df.merge(psf, on="trade_id", how="inner")
                if len(merged) == 0:
                    continue
                excl = 1.0 - (len(merged) / len(df))
                d1_feature_cols = BASE_8 + ARC_SPECIFIC + [c for c in psf.columns if c != "trade_id"]
                merged = merged.dropna(subset=d1_feature_cols).reset_index(drop=True)
                if int(merged["y"].sum()) < 5 or len(merged) < 50:
                    angle_D1_rows.append({
                        "label": label, "t": t, "n_total_at_t": len(merged),
                        "n_pos_at_t": int(merged["y"].sum()),
                        "exclusion_rate": excl, "rf_auc_mean": float("nan"),
                        "rf_auc_folds": [],
                        "passes_d1_auc": 0, "passes_d1_exclusion": int(excl <= D1_EXCLUSION_CEILING),
                        "feature_set_size": len(d1_feature_cols),
                    })
                    continue
                Xt = merged[d1_feature_cols].to_numpy(dtype=float)
                yt = merged["y"].to_numpy(dtype=int)
                auc, fold_aucs, oof = _cv_auc(Xt, yt, "rf")
                d1_candidates.append({
                    "t": t,
                    "merged": merged,
                    "feature_cols": d1_feature_cols,
                    "auc": auc,
                    "fold_aucs": fold_aucs,
                    "oof": oof,
                    "exclusion": excl,
                })
                angle_D1_rows.append({
                    "label": label, "t": t,
                    "n_total_at_t": len(merged),
                    "n_pos_at_t": int(merged["y"].sum()),
                    "exclusion_rate": excl,
                    "rf_auc_mean": auc,
                    "rf_auc_folds": [float(x) for x in fold_aucs.tolist()],
                    "passes_d1_auc": int(auc >= D1_AUC_FLOOR),
                    "passes_d1_exclusion": int(excl <= D1_EXCLUSION_CEILING),
                    "feature_set_size": len(d1_feature_cols),
                })
            # Smallest-t selection.
            eligible = [c for c in d1_candidates if c["auc"] >= D1_AUC_FLOOR and c["exclusion"] <= D1_EXCLUSION_CEILING]
            eligible.sort(key=lambda c: c["t"])
            if eligible:
                chosen_c = eligible[0]
                d1_chosen_t = int(chosen_c["t"])
                d1_rf_auc = float(chosen_c["auc"])
                d1_fold_aucs = chosen_c["fold_aucs"]
                d1_exclusion = float(chosen_c["exclusion"])
                d1_oof = chosen_c["oof"]
                d1_features = list(chosen_c["feature_cols"])
                merged = chosen_c["merged"]
                # Threshold sweep.
                y_t = merged["y"].to_numpy(dtype=int)
                tsweep_rows, chosen = _threshold_sweep(y_t, d1_oof, THRESHOLD_SWEEP)
                pd.DataFrame(tsweep_rows).to_csv(
                    args.out_dir / f"threshold_sweep_D1_{label}.csv",
                    index=False, float_format="%.10g", lineterminator="\n",
                )
                if chosen is None:
                    d1_status = "fail_threshold_sweep"
                else:
                    d1_status = "pass"
                    d1_threshold_info = chosen
                    # Fit final D1 classifier on full eligible data.
                    Xt_full = merged[d1_features].to_numpy(dtype=float)
                    rf_d1 = RandomForestClassifier(**RF_KW)
                    rf_d1.fit(Xt_full, y_t)
                    joblib.dump(rf_d1, args.out_dir / f"archetype_{label}_D1_classifier.joblib")
                    (args.out_dir / f"archetype_{label}_D1_policy.yaml").write_text(
                        yaml.safe_dump({
                            "label": label,
                            "cluster_id": cid,
                            "evaluation_mode": mode,
                            "pipeline": "D1",
                            "t_bar": d1_chosen_t,
                            "feature_set": d1_features,
                            "threshold": float(chosen["threshold"]),
                            "rf_auc_mean": float(d1_rf_auc),
                            "rf_auc_folds": [float(x) for x in d1_fold_aucs.tolist()],
                            "exclusion_rate": float(d1_exclusion),
                            "precision_at_threshold": float(chosen["precision"]),
                            "recall_at_threshold": float(chosen["recall"]),
                            "selected_SL_atr_multiplier": float(sel_SL),
                            "pre_t_sl_atr_multiplier": float(pre_t_sl),
                            "rf_kwargs": RF_KW,
                        }, sort_keys=False),
                        encoding="utf-8",
                    )

        # ---------------- Disposition ----------------
        pipelines = []
        if e_status == "pass":
            pipelines.append("E")
        if d1_status == "pass":
            pipelines.append("D1")
        disposition = (
            "passes_E_and_D1" if len(pipelines) == 2 else
            "passes_E_only" if pipelines == ["E"] else
            "passes_D1_only" if pipelines == ["D1"] else
            "dies"
        )
        extract_pass.append({
            "label": label,
            "cluster_id": cid,
            "evaluation_mode": mode,
            "n_total": n_total,
            "n_pos": n_pos,
            "sample_ok": int(sample_ok),
            "e_rf_auc": e_rf_auc,
            "e_lr_auc": e_lr_auc,
            "e_status": e_status,
            "e_threshold": float(e_threshold_info["threshold"]) if e_threshold_info else float("nan"),
            "e_precision": float(e_threshold_info["precision"]) if e_threshold_info else float("nan"),
            "e_recall": float(e_threshold_info["recall"]) if e_threshold_info else float("nan"),
            "d1_chosen_t": d1_chosen_t if d1_chosen_t is not None else -1,
            "d1_rf_auc": d1_rf_auc,
            "d1_exclusion": d1_exclusion,
            "d1_status": d1_status,
            "d1_threshold": float(d1_threshold_info["threshold"]) if d1_threshold_info else float("nan"),
            "d1_precision": float(d1_threshold_info["precision"]) if d1_threshold_info else float("nan"),
            "d1_recall": float(d1_threshold_info["recall"]) if d1_threshold_info else float("nan"),
            "selected_SL_atr_multiplier": sel_SL,
            "pre_t_sl_atr_multiplier": pre_t_sl,
            "pipeline_assignment": disposition,
        })

    pd.DataFrame(angle_E_rows).to_csv(
        args.out_dir / "predictability_angle_E.csv", index=False,
        float_format="%.10g", lineterminator="\n",
    )
    pd.DataFrame(angle_D1_rows).to_csv(
        args.out_dir / "predictability_angle_D1.csv", index=False,
        float_format="%.10g", lineterminator="\n",
    )
    pd.DataFrame(extract_pass).to_csv(
        args.out_dir / "extractability_pass_list.csv", index=False,
        float_format="%.10g", lineterminator="\n",
    )

    # Markdown summary.
    md: List[str] = []
    md.append("# Arc 9 Step 4 - Extractability investigation + artefact production")
    md.append("")
    any_pass = any(r["pipeline_assignment"] != "dies" for r in extract_pass)
    md.append(f"Verdict: **{'PASS' if any_pass else 'FAIL'}**")
    md.append("")
    md.append(f"Pipeline E gate: RF AUC >= {E_AUC_FLOOR}. Pipeline D1 gate: RF AUC >= {D1_AUC_FLOOR} AND exclusion <= {D1_EXCLUSION_CEILING}.")
    md.append(f"Threshold sweep: {THRESHOLD_SWEEP}. Selection: max precision with recall >= {RECALL_FLOOR}. v2.2 §3: no max-F1 fallback.")
    md.append("")
    md.append("## Per-archetype results")
    md.append("")
    md.append("| label | mode | n_total | n_pos | E RF AUC | E LR AUC | E status | E thr | E prec | E rec | D1 t | D1 AUC | D1 excl | D1 status | D1 thr | D1 prec | D1 rec | pipeline | pre_t SL |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    def _fmt(v: float, fmt: str = ".3f") -> str:
        if v is None or (isinstance(v, float) and (np.isnan(v))):
            return "-"
        return f"{v:{fmt}}"

    for r in extract_pass:
        md.append(
            f"| {r['label']} | {r['evaluation_mode']} | {r['n_total']} | {r['n_pos']} | "
            f"{_fmt(r['e_rf_auc'])} | {_fmt(r['e_lr_auc'])} | {r['e_status']} | "
            f"{_fmt(r['e_threshold'], '.2f')} | "
            f"{_fmt(r['e_precision'])} | {_fmt(r['e_recall'])} | "
            f"{r['d1_chosen_t']} | {_fmt(r['d1_rf_auc'])} | {_fmt(r['d1_exclusion'])} | "
            f"{r['d1_status']} | {_fmt(r['d1_threshold'], '.2f')} | "
            f"{_fmt(r['d1_precision'])} | {_fmt(r['d1_recall'])} | "
            f"**{r['pipeline_assignment']}** | {r['pre_t_sl_atr_multiplier']:.1f} |"
        )
    md.append("")
    md.append("## Outputs")
    md.append("")
    md.append("- entry_features.csv")
    md.append("- predictability_angle_E.csv")
    md.append("- predictability_angle_D1.csv")
    md.append("- extractability_pass_list.csv")
    md.append("- threshold_sweep_E_<label>.csv (per surviving Pipeline E)")
    md.append("- threshold_sweep_D1_<label>.csv (per surviving Pipeline D1)")
    md.append("- archetype_<label>_E_classifier.joblib + _E_filter.yaml (per surviving E)")
    md.append("- archetype_<label>_D1_classifier.joblib + _D1_policy.yaml (per surviving D1)")
    (args.out_dir / "STEP4_SUMMARY.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"[step 4] {'PASS' if any_pass else 'FAIL'} - {len([r for r in extract_pass if r['pipeline_assignment']!='dies'])} archetype(s) extractable")
    for r in extract_pass:
        print(f"  {r['label']}: E={r['e_status']} (RF AUC {r['e_rf_auc']:.3f}), "
              f"D1={r['d1_status']} (t={r['d1_chosen_t']}, AUC {r['d1_rf_auc']:.3f}) "
              f"-> {r['pipeline_assignment']}")
    return 0 if any_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
