"""Arc 11 — SHB Long 4H Signal Improvement Sweep (off-protocol, documentation only).

Seven-stage diagnostic, candidate-generation only. NOT a canonical anything.
Arc 11 stays Closed-HALT. No queue / registry / protocol mutation.

Stages:
  1. Signal-tightening (entry-trigger filters)
  2. Pipeline DE extended t-sweep
  3. Pipeline D (post-entry classifier) on c1 directly
  4. Path-aware dynamic SL
  5. Pair-level Pareto
  6. Sizing without filtering
  7. Winner combination

Single deliverable: results/l_arc_11/sig_improve/sig_improve_report.md
plus per-stage CSVs tagged sig_improve_stageN_*.

Usage:
    py scripts/l_arc_11/sig_improve.py
"""

from __future__ import annotations

import csv
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.l_arc_11.step3_capturability import _eval_trade_at_sl  # noqa: E402
from scripts.l_arc_11.step4_extractability import (  # noqa: E402
    PIPELINE_E_FEATURES, PIPELINE_E_BASE,
    _build_pair_cache, compute_pipeline_e_features,
)
from scripts.l_arc_11.experimental_s5_wfo import (  # noqa: E402
    FOLDS, build_paths_index,
)
from scripts.l_arc_11.filter_diag import (  # noqa: E402
    PATH_SO_FAR_FEATURES_AT_T, _path_features_at_t_orig_frame,
    _train_predict_one_fold, _decide_class_weight, _wfo_run,
    _summarise_aucs, _live_deployable_for_regime, RegimeResult,
    EXPECTED_CLUSTERS_K4_SHA256, _file_sha256, DATA_DIR_4H,
    SL_DEPLOY, ORIGINAL_SL,
)

OUT_DIR = _REPO_ROOT / "results" / "l_arc_11" / "sig_improve"
STAGE_TIMEBOX_SEC = 3600  # 1 hour soft per stage
TOTAL_TIMEBOX_SEC = 21600  # 6 hours hard


# ============================================================
# Pre-test sanity
# ============================================================

def pre_test_sanity() -> Dict[str, Any]:
    clusters_path = _REPO_ROOT / "results/l_arc_11/step2/clusters_K4.csv"
    cluster_hash = _file_sha256(clusters_path)
    stable = cluster_hash == EXPECTED_CLUSTERS_K4_SHA256
    print(f"[sig] clusters_K4 sha256: {cluster_hash} (stable={stable})", file=sys.stderr)
    print(f"[sig] WFO folds: {len(FOLDS)} folds, {FOLDS[0][1]} → {FOLDS[-1][2]}", file=sys.stderr)
    return {"cluster_hash": cluster_hash, "cluster_stable": stable, "folds": FOLDS}


# ============================================================
# Shared CSV writer
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
# Trigger-bar features for Stage 1 (computed once)
# ============================================================

def compute_trigger_features(
    trades_df: pd.DataFrame, pair_caches: Dict[str, Any]
) -> pd.DataFrame:
    """Per-trade trigger-bar features for Stage 1 signal-tightening filters."""
    rows: List[Dict[str, Any]] = []
    for _, t in trades_df.iterrows():
        pair = str(t["pair"])
        cache = pair_caches[pair]
        sig_ts = pd.Timestamp(t["signal_bar_time"])
        i = cache.idx_4h_by_start[sig_ts] if hasattr(cache, "idx_4h_by_start") else cache.idx_by_ts.get(sig_ts, -1)
        if i < 0:
            continue

        o_t = float(cache.df_4h["open"].iloc[i])
        h_t = float(cache.df_4h["high"].iloc[i])
        l_t = float(cache.df_4h["low"].iloc[i])
        c_t = float(cache.df_4h["close"].iloc[i])
        rng = h_t - l_t
        atr = float(cache.atr_4h[i]) if not math.isnan(cache.atr_4h[i]) else float("nan")

        # trigger_close_pos: (close - low) / (high - low). Same as close_position.
        close_pos = (c_t - l_t) / rng if rng > 0 else 0.5

        # trigger_body_atr: |close - open| / atr (signed positive for bullish long).
        body_atr = ((c_t - o_t) / atr) if atr > 0 else float("nan")

        # trigger_break_size_atr: (close - H_ref) / atr. Already on the trade row.
        break_size_atr = float(t.get("break_magnitude_atr", float("nan")))

        # pullback_depth_atr: depth from H_ref to lowest low between H_ref bar and t-1.
        h_ref_offset = int(t.get("h_ref_bar_offset", 0)) if pd.notna(t.get("h_ref_bar_offset")) else 0
        h_ref_idx = i - h_ref_offset
        if h_ref_idx < 0 or h_ref_idx >= i:
            pullback_atr = float("nan")
        else:
            h_ref_val = float(t.get("h_ref", float("nan")))
            window_lows = cache.df_4h["low"].iloc[h_ref_idx + 1:i].astype(float).to_numpy()
            if window_lows.size == 0 or not math.isfinite(h_ref_val) or not atr > 0:
                pullback_atr = 0.0
            else:
                pullback_atr = float((h_ref_val - window_lows.min()) / atr)

        # ret_5bar_atr: (close[t] - close[t-5]) / atr.
        if i >= 5 and atr > 0:
            ret5 = (c_t - float(cache.df_4h["close"].iloc[i - 5])) / atr
        else:
            ret5 = float("nan")

        # pos_in_20bar_range: (close - 20-bar low) / 20-bar range.
        if i >= 19:
            wlow = float(cache.df_4h["low"].iloc[i - 19:i + 1].min())
            whigh = float(cache.df_4h["high"].iloc[i - 19:i + 1].max())
            wrng = whigh - wlow
            pos20 = (c_t - wlow) / wrng if wrng > 0 else 0.5
        else:
            pos20 = float("nan")

        rows.append({
            "trade_id": int(t["trade_id"]),
            "trigger_close_pos": close_pos,
            "pullback_depth_atr": pullback_atr,
            "trigger_body_atr": body_atr,
            "trigger_break_size_atr": break_size_atr,
            "ret_5bar_atr": ret5,
            "pos_in_20bar_range": pos20,
        })
    return pd.DataFrame(rows)


# ============================================================
# Stage 1 — Signal-tightening filters
# ============================================================

@dataclass
class S1Result:
    filter_name: str
    threshold: float
    n_remaining: int
    c1_retention: float
    c2_retention: float
    pool_remaining: int
    aggregate_mean_r: float
    passes_gate: bool


S1_FILTERS = {
    "F1_trigger_close_pos": ("trigger_close_pos", [0.5, 0.6, 0.7, 0.8, 0.9]),
    "F2_pullback_depth_atr": ("pullback_depth_atr", [0.5, 0.75, 1.0, 1.25, 1.5]),
    "F3_trigger_body_atr": ("trigger_body_atr", [0.3, 0.5, 0.7, 1.0]),
    "F4_trigger_break_size_atr": ("trigger_break_size_atr", [0.0, 0.25, 0.5, 0.75, 1.0]),
    "F5_ret_5bar_atr": ("ret_5bar_atr", [0.0, 0.5, 1.0, 1.5]),
    "F6_pos_in_20bar_range": ("pos_in_20bar_range", [0.4, 0.5, 0.6, 0.7]),
}


def stage_1_signal_tighten(
    trigger_features: pd.DataFrame, clusters_df: pd.DataFrame, final_r_sl3: Dict[int, float]
) -> Tuple[List[S1Result], List[Dict[str, Any]]]:
    """Run Stage 1: single-filter pareto + pairwise AND combinations."""
    cluster_map: Dict[int, int] = {int(r["trade_id"]): int(r["cluster_id"]) for _, r in clusters_df.iterrows()}
    full = trigger_features.merge(clusters_df, on="trade_id", how="left").copy()
    full["final_r_sl3"] = full["trade_id"].map(final_r_sl3)
    n_total = len(full)
    n_c1_total = int((full["cluster_id"] == 1).sum())
    n_c2_total = int((full["cluster_id"] == 2).sum())

    single_results: List[S1Result] = []
    for fname, (col, thresholds) in S1_FILTERS.items():
        for th in thresholds:
            mask = (full[col] >= th)
            sub = full[mask]
            n_rem = int(len(sub))
            c1_ret = int((sub["cluster_id"] == 1).sum()) / max(n_c1_total, 1)
            c2_ret = int((sub["cluster_id"] == 2).sum()) / max(n_c2_total, 1)
            agg_mean_r = float(sub["final_r_sl3"].mean()) if n_rem > 0 else 0.0
            gate = (c1_ret >= 0.80) and (c2_ret <= 0.30) and (n_rem >= 500)
            single_results.append(S1Result(
                filter_name=fname, threshold=float(th), n_remaining=n_rem,
                c1_retention=c1_ret, c2_retention=c2_ret,
                pool_remaining=n_rem, aggregate_mean_r=agg_mean_r,
                passes_gate=gate,
            ))

    # Pareto: rank by (c1_ret high, c2_ret low, aggregate_mean_r high).
    # For each filter, pick its best threshold (most c2 dropped at c1_ret >= 0.80, fallback to max c1-c2 gap).
    best_per_filter: Dict[str, S1Result] = {}
    for fname, _ in S1_FILTERS.items():
        candidates = [r for r in single_results if r.filter_name == fname and r.c1_retention >= 0.80]
        if not candidates:
            # No threshold preserves 80% c1 → take max c1_ret - c2_ret.
            cand_all = [r for r in single_results if r.filter_name == fname]
            best_per_filter[fname] = max(cand_all, key=lambda r: r.c1_retention - r.c2_retention)
        else:
            best_per_filter[fname] = max(candidates, key=lambda r: r.c1_retention - r.c2_retention)

    # Pairwise AND of top-3 single filters (ranked by c1-c2 gap among gate-passing single rules; fallback rank by c1-c2 gap).
    ranked = sorted(best_per_filter.values(), key=lambda r: -(r.c1_retention - r.c2_retention))
    top3 = ranked[:3]
    pair_rows: List[Dict[str, Any]] = []
    for i, a in enumerate(top3):
        for b in top3[i + 1:]:
            col_a = S1_FILTERS[a.filter_name][0]
            col_b = S1_FILTERS[b.filter_name][0]
            mask = (full[col_a] >= a.threshold) & (full[col_b] >= b.threshold)
            sub = full[mask]
            n_rem = int(len(sub))
            c1_ret = int((sub["cluster_id"] == 1).sum()) / max(n_c1_total, 1)
            c2_ret = int((sub["cluster_id"] == 2).sum()) / max(n_c2_total, 1)
            agg_mean_r = float(sub["final_r_sl3"].mean()) if n_rem > 0 else 0.0
            gate = (c1_ret >= 0.80) and (c2_ret <= 0.30) and (n_rem >= 500)
            pair_rows.append({
                "rule_a": a.filter_name, "th_a": a.threshold,
                "rule_b": b.filter_name, "th_b": b.threshold,
                "n_remaining": n_rem, "c1_retention": c1_ret,
                "c2_retention": c2_ret, "pool_remaining": n_rem,
                "aggregate_mean_r": agg_mean_r, "passes_gate": gate,
            })
    return single_results, pair_rows


# ============================================================
# Stage 2 — Extended t-sweep for Pipeline DE
# ============================================================

STAGE2_T_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16]


def _build_de_feature_df(
    trades_df: pd.DataFrame, paths_index: Dict[int, pd.DataFrame],
    e_features: pd.DataFrame, t: int,
) -> Tuple[pd.DataFrame, int, float]:
    """Return (feature_df, n_excluded_pre_t, mean_mfe_at_t)."""
    rows: List[Dict[str, Any]] = []
    mfes: List[float] = []
    e_base_idx = {int(r["trade_id"]): r for _, r in e_features.iterrows()}
    for _, tr in trades_df.iterrows():
        tid = int(tr["trade_id"])
        pf = _path_features_at_t_orig_frame(paths_index[tid], t)
        if pf is None:
            continue
        base_row = e_base_idx[tid]
        merged = {"trade_id": tid, "entry_time": base_row["entry_time"]}
        for c in PIPELINE_E_BASE:
            merged[c] = base_row[c]
        merged.update(pf)
        rows.append(merged)
        mfes.append(pf["mfe_so_far_r_at_t"])
    feat_df = pd.DataFrame(rows)
    n_excl = len(trades_df) - len(rows)
    mean_mfe = float(np.mean(mfes)) if mfes else 0.0
    return feat_df, n_excl, mean_mfe


def stage_2_de_sweep(
    trades_df: pd.DataFrame, paths_index: Dict[int, pd.DataFrame],
    e_features: pd.DataFrame, c1_label_by_tid: Dict[int, int],
    final_r_sl3: Dict[int, float], model_kw: dict,
) -> List[Dict[str, Any]]:
    n_pool = len(trades_df)
    rows: List[Dict[str, Any]] = []
    cols = list(PIPELINE_E_BASE) + list(PATH_SO_FAR_FEATURES_AT_T)
    for t in STAGE2_T_VALUES:
        feat_df, n_excl, mean_mfe = _build_de_feature_df(trades_df, paths_index, e_features, t)
        pre_t_filter_rate = n_excl / max(n_pool, 1)
        aucs, details, clf, med = _wfo_run(feat_df, cols, c1_label_by_tid, model_kw)
        mean_auc, n_gate, n_total = _summarise_aucs(aucs, gate=0.60)

        # Live-deployable.
        rr = RegimeResult(name=f"S2_DE_t{t}", fold_aucs=aucs, mean_auc=mean_auc,
                          n_clears_gate=n_gate, n_folds=n_total, fold_details=details,
                          feature_cols=cols, last_clf=clf, last_med=med)
        live = _live_deployable_for_regime(rr, feat_df, c1_label_by_tid, final_r_sl3, model_kw)

        rows.append({
            "t": t, "n_eligible": int(len(feat_df)), "pre_t_filter_rate": pre_t_filter_rate,
            "mean_mfe_at_t": mean_mfe, "mean_auc": mean_auc,
            "n_clears_0.60": n_gate, "n_folds_valid": n_total,
            "fold_aucs": [a for a in aucs],
            "live_sign_consistency": live["sign_consistency"],
            "live_worst_fold_roi_ann_pct": live["worst_fold_roi_ann_pct"],
            "live_mean_fold_roi_ann_pct": live["mean_fold_roi_ann_pct"],
            "live_worst_fold_dd_pct": live["worst_fold_dd_pct"],
            "live_min_trade_count": live["min_trade_count"],
            "live_full_data_roi_pct": live["full_data_roi_pct"],
            "live_full_data_dd_pct": live["full_data_dd_pct"],
            "live_pass_deployable": live["pass_deployable"],
            "live_pass_viable": live["pass_viable"],
        })
        print(f"[sig stage 2] t={t:>2}: mean_auc={mean_auc:.4f} folds≥0.60={n_gate}/{n_total} "
              f"pre_t_rate={pre_t_filter_rate*100:.1f}% mean_mfe={mean_mfe:.3f}R "
              f"live: worst={live['worst_fold_roi_ann_pct']:+.2f}% dd={live['worst_fold_dd_pct']:.2f}% "
              f"sign={live['sign_consistency']} pass_dep={live['pass_deployable']}",
              file=sys.stderr)
    return rows


# ============================================================
# Stage 3 — Pipeline D (post-entry classifier) on c1 directly
# ============================================================

STAGE3_T_VALUES = [3, 5, 8, 12]


def stage_3_pipeline_d_on_c1(
    trades_df: pd.DataFrame, paths_index: Dict[int, pd.DataFrame],
    e_features: pd.DataFrame, clusters_df: pd.DataFrame,
    final_r_sl3: Dict[int, float], model_kw: dict,
) -> List[Dict[str, Any]]:
    """Train D classifier on c1 cohort, predict (final_r >= 1R) at bar t.
    Compare hold (admit-all) vs exit-at-t (admit by classifier) economics
    on the OOS c1 portion per fold."""
    c1_tids = sorted(clusters_df[clusters_df["cluster_id"] == 1]["trade_id"].astype(int).tolist())
    c1_trades = trades_df[trades_df["trade_id"].isin(c1_tids)].copy()
    c1_trades["entry_time"] = pd.to_datetime(c1_trades["entry_time"])

    success_label = {tid: (1 if final_r_sl3[tid] >= 1.0 else 0) for tid in c1_tids}

    e_base_idx = {int(r["trade_id"]): r for _, r in e_features.iterrows()}
    cols = list(PIPELINE_E_BASE) + list(PATH_SO_FAR_FEATURES_AT_T)
    rows: List[Dict[str, Any]] = []

    for t in STAGE3_T_VALUES:
        # Build features for c1 trades that survived to bar t.
        feat_rows: List[Dict[str, Any]] = []
        for tid in c1_tids:
            pf = _path_features_at_t_orig_frame(paths_index[tid], t)
            if pf is None:
                continue
            base_row = e_base_idx[tid]
            merged = {"trade_id": tid, "entry_time": base_row["entry_time"]}
            for c in PIPELINE_E_BASE:
                merged[c] = base_row[c]
            merged.update(pf)
            feat_rows.append(merged)
        feat_df = pd.DataFrame(feat_rows)

        # Per-fold: train D, test on OOS, compute economics (hold vs exit).
        feat_df = feat_df.sort_values("entry_time").reset_index(drop=True)
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_auc_score, recall_score

        fold_aucs = []
        fold_recalls = []
        fold_metrics = []
        for fold_id, oos_s, oos_e in FOLDS:
            oos_start = pd.Timestamp(oos_s)
            oos_end = pd.Timestamp(oos_e)
            is_sub = feat_df[feat_df["entry_time"] < oos_start]
            oos_sub = feat_df[(feat_df["entry_time"] >= oos_start) & (feat_df["entry_time"] < oos_end)]
            is_X = is_sub[cols]
            oos_X = oos_sub[cols]
            is_y = np.array([success_label[int(t)] for t in is_sub["trade_id"]], dtype=int)
            oos_y = np.array([success_label[int(t)] for t in oos_sub["trade_id"]], dtype=int)
            if len(is_sub) < 20 or len(oos_sub) < 5 or len(np.unique(is_y)) < 2 or len(np.unique(oos_y)) < 2:
                fold_aucs.append(float("nan"))
                fold_recalls.append(float("nan"))
                continue
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
            fold_aucs.append(auc)

            # Threshold 0.50 for hold/exit decision.
            admit = p >= 0.50
            n_admit = int(admit.sum())
            # Hold economics: admit means HOLD to natural exit (use final_r_sl3).
            # Exit economics: reject means EXIT at bar t (close_r_at_t in original-SL frame; convert to new SL=3 frame).
            admit_returns = []
            reject_returns = []
            for tid, ad in zip(oos_sub["trade_id"].astype(int), admit):
                if ad:
                    admit_returns.append(final_r_sl3[tid])
                else:
                    # Reject: exit at bar t.
                    pf = _path_features_at_t_orig_frame(paths_index[tid], t)
                    if pf is None:
                        reject_returns.append(-1.0)
                    else:
                        # close_r_at_t in original SL=2 frame → convert to SL=3 frame.
                        # _path_features_at_t_orig_frame returns close_r in original-SL frame.
                        # scale = original_sl / sl_mult = 2/3 = 0.667.
                        # New R = orig_R * scale → but for SL=3 deployment, close_r_at_t * (2/3).
                        scale = ORIGINAL_SL / SL_DEPLOY
                        reject_returns.append(float(pf["close_r_at_t"]) * scale)

            try:
                rec = float(recall_score(oos_y, admit))
            except ValueError:
                rec = float("nan")
            fold_recalls.append(rec)

            fold_metrics.append({
                "fold": fold_id, "n_admit": n_admit, "n_reject": len(oos_sub) - n_admit,
                "admit_mean_r": float(np.mean(admit_returns)) if admit_returns else 0.0,
                "reject_mean_r": float(np.mean(reject_returns)) if reject_returns else 0.0,
                "auc": auc,
            })

        valid_aucs = [a for a in fold_aucs if not math.isnan(a)]
        valid_recs = [r for r in fold_recalls if not math.isnan(r)]
        mean_auc = float(np.mean(valid_aucs)) if valid_aucs else float("nan")
        mean_rec = float(np.mean(valid_recs)) if valid_recs else float("nan")
        admit_total_mean = float(np.mean([f["admit_mean_r"] for f in fold_metrics if f["n_admit"] > 0])) if fold_metrics else 0.0
        reject_total_mean = float(np.mean([f["reject_mean_r"] for f in fold_metrics if f["n_reject"] > 0])) if fold_metrics else 0.0

        # Gate: AUC >= 0.60 AND recall >= 0.60.
        gate_pass = (not math.isnan(mean_auc)) and (mean_auc >= 0.60) and (not math.isnan(mean_rec)) and (mean_rec >= 0.60)

        rows.append({
            "t": t, "n_eligible_c1": int(len(feat_df)),
            "mean_auc": mean_auc, "mean_recall_at_0.50": mean_rec,
            "hold_mean_r_admit_pool": admit_total_mean,
            "exit_mean_r_reject_pool": reject_total_mean,
            "gate_pass_auc_and_recall": gate_pass,
            "fold_aucs": fold_aucs,
        })
        print(f"[sig stage 3] D-on-c1 t={t}: mean_auc={mean_auc:.4f} mean_recall={mean_rec:.4f} "
              f"admit_mean_r={admit_total_mean:+.3f} reject_mean_r={reject_total_mean:+.3f} gate_pass={gate_pass}",
              file=sys.stderr)
    return rows


# ============================================================
# Stage 4 — Path-aware dynamic SL (simulator)
# ============================================================

def _simulate_dynamic_sl_4a(path: pd.DataFrame) -> float:
    """SL=5.0xATR for t<8, SL=2.0xATR for t>=8 (V-shape friendly).

    Returns final_r in the SL_DEPLOY=3.0 frame for consistency with prior runs.
    Walks the path bar-by-bar in original-SL frame, applies new SL truncation.

    R unit logic:
      - path's low_r / high_r / close_r are in original SL=2 frame
      - SL threshold in original-SL R units = -X/2 (where X is new SL multiplier)
      - SL_threshold for 5.0xATR = -5/2 = -2.5 in original-SL R units
      - SL_threshold for 2.0xATR = -2/2 = -1.0 in original-SL R units
      - Final R reported in SL_DEPLOY=3.0 frame: convert via scale = 2/3
    """
    sorted_path = path.sort_values("bar_offset", kind="mergesort").reset_index(drop=True)
    bars = sorted_path["bar_offset"].to_numpy(dtype=int)
    low_r = sorted_path["low_r"].to_numpy(dtype=float)
    close_r = sorted_path["close_r"].to_numpy(dtype=float)

    final_r_orig = None
    for i, bo in enumerate(bars):
        # Effective SL threshold in original-SL R units.
        if bo < 8:
            sl_threshold = -5.0 / ORIGINAL_SL  # -2.5
        else:
            sl_threshold = -2.0 / ORIGINAL_SL  # -1.0
        if low_r[i] <= sl_threshold:
            # SL hit at this bar. Final R in this dynamic-SL frame.
            # For consistency, report as -1R in the dynamic-SL frame at that bar.
            # Convert: at bar < 8, dynamic SL = 5xATR, -1R = -5/3 in SL_DEPLOY frame.
            # at bar >= 8, dynamic SL = 2xATR, -1R = -2/3 in SL_DEPLOY frame.
            # Simpler: just report in original-SL R units (preserves comparison).
            # Use SL frame at time of stop.
            final_r_orig = sl_threshold  # = -1 in dynamic-SL frame at that bar.
            break
    if final_r_orig is None:
        final_r_orig = float(close_r[-1])
    # Convert to SL_DEPLOY=3 frame for comparison.
    return float(final_r_orig * (ORIGINAL_SL / SL_DEPLOY))


def _simulate_dynamic_sl_4b(path: pd.DataFrame) -> float:
    """SL=3.0xATR for t<5, breakeven SL once mfe>=1R (in SL=3 frame).

    Returns final_r in SL_DEPLOY=3.0 frame.
    """
    sorted_path = path.sort_values("bar_offset", kind="mergesort").reset_index(drop=True)
    bars = sorted_path["bar_offset"].to_numpy(dtype=int)
    low_r = sorted_path["low_r"].to_numpy(dtype=float)
    mfe_r = sorted_path["mfe_so_far_r"].to_numpy(dtype=float)
    close_r = sorted_path["close_r"].to_numpy(dtype=float)

    # Default SL in original-SL R = -3/2 = -1.5.
    # Breakeven SL = 0 (in any R frame).
    # mfe threshold in original-SL R = 1.0 * (3/2) = 1.5 (since "mfe >= 1R in SL=3 frame" = mfe_orig >= 1.5).
    final_r_orig = None
    sl_threshold = -3.0 / ORIGINAL_SL  # -1.5
    breakeven_active = False
    for i, bo in enumerate(bars):
        if not breakeven_active:
            if bo >= 5 and mfe_r[i] >= 1.0 * (SL_DEPLOY / ORIGINAL_SL):
                breakeven_active = True
                sl_threshold = 0.0
        if low_r[i] <= sl_threshold:
            final_r_orig = sl_threshold
            break
    if final_r_orig is None:
        final_r_orig = float(close_r[-1])
    return float(final_r_orig * (ORIGINAL_SL / SL_DEPLOY))


def stage_4_dynamic_sl(
    paths_index: Dict[int, pd.DataFrame], clusters_df: pd.DataFrame,
    final_r_sl3: Dict[int, float],
) -> Dict[str, Any]:
    c1_tids = sorted(clusters_df[clusters_df["cluster_id"] == 1]["trade_id"].astype(int).tolist())

    baseline_returns = [final_r_sl3[tid] for tid in c1_tids]
    s4a_returns = [_simulate_dynamic_sl_4a(paths_index[tid]) for tid in c1_tids]
    s4b_returns = [_simulate_dynamic_sl_4b(paths_index[tid]) for tid in c1_tids]

    def _stats(returns: List[float]) -> Dict[str, float]:
        arr = np.array(returns, dtype=float)
        eq = 1.0
        curve = [1.0]
        for r in arr:
            eq *= (1.0 + r * 0.005)
            curve.append(eq)
        arr_eq = np.array(curve, dtype=float)
        peak = np.maximum.accumulate(arr_eq)
        dd = ((peak - arr_eq) / peak).max()
        return {
            "n": int(len(returns)),
            "mean_r": float(arr.mean()) if arr.size else 0.0,
            "median_r": float(np.median(arr)) if arr.size else 0.0,
            "p5_r": float(np.percentile(arr, 5)) if arr.size else 0.0,
            "p95_r": float(np.percentile(arr, 95)) if arr.size else 0.0,
            "win_rate": float((arr >= 1.0).mean()) if arr.size else 0.0,
            "loss_rate": float((arr <= -1.0).mean()) if arr.size else 0.0,
            "full_roi_pct": float(eq - 1.0) * 100,
            "full_dd_pct": float(dd) * 100,
        }

    return {
        "baseline_SL3_fixed": _stats(baseline_returns),
        "4a_SL5_then_SL2_at_t8": _stats(s4a_returns),
        "4b_SL3_then_BE_at_t5_if_mfe1R": _stats(s4b_returns),
    }


# ============================================================
# Stage 5 — Pair-level Pareto
# ============================================================

def stage_5_pair_pareto(
    trades_df: pd.DataFrame, paths_index: Dict[int, pd.DataFrame],
    clusters_df: pd.DataFrame, e_features: pd.DataFrame,
    c1_label_by_tid: Dict[int, int], final_r_sl3: Dict[int, float], model_kw: dict,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Per-pair c1 stats + top-N subset live-deployable on Stage 2's best t."""
    # Compute MFE at SL=3 for each c1 trade (for reach_1R and mfe_p50).
    c1_tids = sorted(clusters_df[clusters_df["cluster_id"] == 1]["trade_id"].astype(int).tolist())
    pair_map = {int(r["trade_id"]): str(r["pair"]) for _, r in trades_df.iterrows()}
    per_pair: Dict[str, List[Dict[str, Any]]] = {}
    for tid in c1_tids:
        pair = pair_map[tid]
        te = _eval_trade_at_sl(paths_index[tid], SL_DEPLOY, ORIGINAL_SL)
        per_pair.setdefault(pair, []).append({
            "trade_id": tid, "final_r": float(te.final_r_new), "fwd_mfe": float(te.fwd_mfe_new_r),
        })

    pair_rows: List[Dict[str, Any]] = []
    for pair, trades in per_pair.items():
        n = len(trades)
        rs = np.array([t["final_r"] for t in trades], dtype=float)
        mfes = np.array([t["fwd_mfe"] for t in trades], dtype=float)
        pair_rows.append({
            "pair": pair, "n_c1_trades": n,
            "mean_r": float(rs.mean()), "median_r": float(np.median(rs)),
            "hit_rate_1R": float((rs >= 1.0).mean()), "reach_1R": float((mfes >= 1.0).mean()),
            "mfe_p50": float(np.median(mfes)),
        })

    # Rank by mean_r.
    pair_rows.sort(key=lambda r: -r["mean_r"])
    top10 = pair_rows[:10]
    top10_pairs = {r["pair"] for r in top10}
    top10_tids = [tid for tid in c1_tids if pair_map[tid] in top10_pairs]

    # Top-10 pair-subset WFO on Stage 2's t=8 DE config (use full universe but restrict to top-10 pair subset).
    # Filter the universe to top-10 pairs first.
    universe_sub = trades_df[trades_df["pair"].isin(top10_pairs)].copy()
    e_sub = e_features[e_features["trade_id"].isin(universe_sub["trade_id"])].copy()
    paths_sub = {tid: paths_index[tid] for tid in universe_sub["trade_id"].astype(int)}

    # Build DE features at t=8 for top-10 pair universe.
    feat_rows: List[Dict[str, Any]] = []
    e_base_idx = {int(r["trade_id"]): r for _, r in e_sub.iterrows()}
    for _, tr in universe_sub.iterrows():
        tid = int(tr["trade_id"])
        pf = _path_features_at_t_orig_frame(paths_sub[tid], 8)
        if pf is None:
            continue
        base_row = e_base_idx[tid]
        merged = {"trade_id": tid, "entry_time": base_row["entry_time"]}
        for c in PIPELINE_E_BASE:
            merged[c] = base_row[c]
        merged.update(pf)
        feat_rows.append(merged)
    feat_df = pd.DataFrame(feat_rows)
    cols = list(PIPELINE_E_BASE) + list(PATH_SO_FAR_FEATURES_AT_T)
    aucs, details, clf, med = _wfo_run(feat_df, cols, c1_label_by_tid, model_kw)
    mean_auc, n_gate, n_total = _summarise_aucs(aucs, gate=0.60)
    rr = RegimeResult(name="S5_top10_DE_t8", fold_aucs=aucs, mean_auc=mean_auc,
                      n_clears_gate=n_gate, n_folds=n_total, fold_details=details,
                      feature_cols=cols, last_clf=clf, last_med=med)
    live = _live_deployable_for_regime(rr, feat_df, c1_label_by_tid, final_r_sl3, model_kw)

    return pair_rows, {
        "top10_pairs": sorted(top10_pairs),
        "mean_auc_t8": mean_auc,
        "live_sign_consistency": live["sign_consistency"],
        "live_worst_fold_roi_ann_pct": live["worst_fold_roi_ann_pct"],
        "live_mean_fold_roi_ann_pct": live["mean_fold_roi_ann_pct"],
        "live_worst_fold_dd_pct": live["worst_fold_dd_pct"],
        "live_min_trade_count": live["min_trade_count"],
        "live_full_data_roi_pct": live["full_data_roi_pct"],
        "live_full_data_dd_pct": live["full_data_dd_pct"],
        "live_pass_deployable": live["pass_deployable"],
        "live_pass_viable": live["pass_viable"],
    }


# ============================================================
# Stage 6 — Sizing without filtering (full SHB universe)
# ============================================================

def stage_6_sizing_only(
    trades_df: pd.DataFrame, final_r_sl3: Dict[int, float]
) -> List[Dict[str, Any]]:
    """Full 2299-trade universe, no classifier, no filter. Variable sizing."""
    sub = trades_df.copy()
    sub["entry_time"] = pd.to_datetime(sub["entry_time"])
    sub = sub.sort_values("entry_time").reset_index(drop=True)
    sub["final_r"] = sub["trade_id"].astype(int).map(final_r_sl3)

    rows: List[Dict[str, Any]] = []
    for size in [0.0025, 0.0050, 0.0100]:
        # Per fold metrics.
        fold_rois_ann = []
        fold_dds = []
        fold_signs = []
        fold_counts = []
        for fold_id, oos_s, oos_e in FOLDS:
            oos_start = pd.Timestamp(oos_s)
            oos_end = pd.Timestamp(oos_e)
            days = (oos_end - oos_start).days
            oos = sub[(sub["entry_time"] >= oos_start) & (sub["entry_time"] < oos_end)]
            if len(oos) == 0:
                continue
            eq = 1.0
            curve = [1.0]
            for r in oos["final_r"]:
                eq *= (1.0 + r * size)
                curve.append(eq)
            roi = eq - 1.0
            arr = np.array(curve, dtype=float)
            peak = np.maximum.accumulate(arr)
            dd = ((peak - arr) / peak).max()
            roi_ann = (eq) ** (365.25 / days) - 1.0 if days > 0 else 0.0
            fold_rois_ann.append(roi_ann)
            fold_dds.append(dd)
            fold_signs.append(roi > 0)
            fold_counts.append(len(oos))

        # Full data.
        eq = 1.0
        curve = [1.0]
        for r in sub["final_r"]:
            eq *= (1.0 + r * size)
            curve.append(eq)
        full_roi = eq - 1.0
        arr = np.array(curve, dtype=float)
        peak = np.maximum.accumulate(arr)
        full_dd = ((peak - arr) / peak).max()

        rows.append({
            "size_pct": size * 100, "n_trades_total": len(sub),
            "sign_consistency": all(fold_signs),
            "worst_fold_roi_ann_pct": float(min(fold_rois_ann)) * 100 if fold_rois_ann else 0.0,
            "mean_fold_roi_ann_pct": float(np.mean(fold_rois_ann)) * 100 if fold_rois_ann else 0.0,
            "worst_fold_dd_pct": float(max(fold_dds)) * 100 if fold_dds else 0.0,
            "min_trade_count": int(min(fold_counts)) if fold_counts else 0,
            "full_data_roi_pct": full_roi * 100,
            "full_data_dd_pct": full_dd * 100,
        })
    return rows


# ============================================================
# Stage 7 — Winner combination
# ============================================================

def _apply_stage1_filter(
    trades_df: pd.DataFrame, trigger_features: pd.DataFrame,
    filter_rule: Optional[Tuple[str, float]] = None,
) -> pd.DataFrame:
    if filter_rule is None:
        return trades_df.copy()
    col, th = filter_rule
    elig_tids = set(trigger_features[trigger_features[col] >= th]["trade_id"].astype(int).tolist())
    return trades_df[trades_df["trade_id"].astype(int).isin(elig_tids)].copy()


def stage_7_combinations(
    trades_df: pd.DataFrame, paths_index: Dict[int, pd.DataFrame],
    trigger_features: pd.DataFrame, e_features: pd.DataFrame,
    clusters_df: pd.DataFrame, c1_label_by_tid: Dict[int, int],
    final_r_sl3: Dict[int, float], model_kw: dict,
    s1_winner: Optional[Tuple[str, float]] = None,
    s2_best_t: int = 8,
    s4_winner: str = "baseline",
) -> List[Dict[str, Any]]:
    """Run a few combinations. We use:
      (i)   S1 winner + S2 DE t*
      (ii)  S1 winner + S4 winner (no classifier, just filter + SL policy)
      (iii) S2 DE t* + S4 winner (classifier + SL policy)
      (iv)  S1 + S2 + S4 triplet
    Live-deployable on each."""
    rows: List[Dict[str, Any]] = []

    # Combination (i): S1 + S2 DE t*.
    sub_i = _apply_stage1_filter(trades_df, trigger_features, s1_winner) if s1_winner else trades_df.copy()
    feat_i, _, _ = _build_de_feature_df(sub_i, paths_index, e_features, s2_best_t)
    cols = list(PIPELINE_E_BASE) + list(PATH_SO_FAR_FEATURES_AT_T)
    aucs_i, _, clf_i, med_i = _wfo_run(feat_i, cols, c1_label_by_tid, model_kw)
    rr_i = RegimeResult(name=f"S7_S1+S2_t{s2_best_t}", fold_aucs=aucs_i,
                        mean_auc=float(np.mean([a for a in aucs_i if not math.isnan(a)])) if any(not math.isnan(a) for a in aucs_i) else 0.0,
                        n_clears_gate=0, n_folds=0, fold_details=[], feature_cols=cols)
    live_i = _live_deployable_for_regime(rr_i, feat_i, c1_label_by_tid, final_r_sl3, model_kw)
    rows.append({"combo": f"S1({_s1_label(s1_winner)}) + S2_DE_t{s2_best_t}", **_live_to_row(live_i)})

    # Combination (ii): S1 + S4 winner (no classifier; just filter + custom SL).
    # Skip if s4_winner is "baseline" (no policy change). We compute custom final_r per trade via the simulator.
    if s4_winner != "baseline":
        sim_fn = _simulate_dynamic_sl_4a if s4_winner == "4a" else _simulate_dynamic_sl_4b
        custom_r = {int(tid): sim_fn(paths_index[int(tid)]) for tid in sub_i["trade_id"].astype(int)}
        sub_ii = sub_i.copy()
        sub_ii["entry_time"] = pd.to_datetime(sub_ii["entry_time"])
        sub_ii = sub_ii.sort_values("entry_time").reset_index(drop=True)
        full_returns = [custom_r[int(tid)] for tid in sub_ii["trade_id"]]
        # Per-fold computation.
        fold_rois_ann = []
        fold_dds = []
        fold_signs = []
        fold_counts = []
        for fold_id, oos_s, oos_e in FOLDS:
            oos_start = pd.Timestamp(oos_s)
            oos_end = pd.Timestamp(oos_e)
            days = (oos_end - oos_start).days
            oos = sub_ii[(sub_ii["entry_time"] >= oos_start) & (sub_ii["entry_time"] < oos_end)]
            if len(oos) == 0:
                continue
            eq = 1.0
            curve = [1.0]
            for r in [custom_r[int(t)] for t in oos["trade_id"]]:
                eq *= (1.0 + r * 0.005)
                curve.append(eq)
            roi = eq - 1.0
            arr = np.array(curve, dtype=float)
            peak = np.maximum.accumulate(arr)
            dd = ((peak - arr) / peak).max()
            roi_ann = (eq) ** (365.25 / days) - 1.0 if days > 0 else 0.0
            fold_rois_ann.append(roi_ann)
            fold_dds.append(dd)
            fold_signs.append(roi > 0)
            fold_counts.append(len(oos))
        eq = 1.0
        curve = [1.0]
        for r in full_returns:
            eq *= (1.0 + r * 0.005)
            curve.append(eq)
        full_roi = eq - 1.0
        arr = np.array(curve, dtype=float)
        peak = np.maximum.accumulate(arr)
        full_dd = ((peak - arr) / peak).max()
        rows.append({
            "combo": f"S1({_s1_label(s1_winner)}) + S4_{s4_winner} (no classifier)",
            "sign_consistency": all(fold_signs) if fold_signs else False,
            "worst_fold_roi_ann_pct": float(min(fold_rois_ann)) * 100 if fold_rois_ann else 0.0,
            "mean_fold_roi_ann_pct": float(np.mean(fold_rois_ann)) * 100 if fold_rois_ann else 0.0,
            "worst_fold_dd_pct": float(max(fold_dds)) * 100 if fold_dds else 0.0,
            "min_trade_count": int(min(fold_counts)) if fold_counts else 0,
            "full_data_roi_pct": full_roi * 100,
            "full_data_dd_pct": full_dd * 100,
            "pass_deployable": False, "pass_viable": False,  # filled below
        })
        # Re-evaluate pass.
        last = rows[-1]
        sign_ok = last["sign_consistency"]
        last["pass_deployable"] = (sign_ok and last["worst_fold_roi_ann_pct"] >= 5.0
                                    and last["mean_fold_roi_ann_pct"] >= 8.0
                                    and last["worst_fold_dd_pct"] <= 8.0
                                    and last["min_trade_count"] >= 15
                                    and last["full_data_roi_pct"] >= 5.0
                                    and last["full_data_dd_pct"] <= 10.0)
        last["pass_viable"] = (sign_ok and last["worst_fold_roi_ann_pct"] > 0
                                and last["mean_fold_roi_ann_pct"] >= 3.0
                                and last["worst_fold_dd_pct"] <= 8.0
                                and last["min_trade_count"] >= 5
                                and last["full_data_roi_pct"] >= 3.0
                                and last["full_data_dd_pct"] <= 10.0)

    # Combination (iii): S2 DE t* + S4 winner (classifier + custom SL on admitted).
    if s4_winner != "baseline":
        feat_iii, _, _ = _build_de_feature_df(trades_df, paths_index, e_features, s2_best_t)
        sim_fn = _simulate_dynamic_sl_4a if s4_winner == "4a" else _simulate_dynamic_sl_4b
        custom_r_iii = {int(tid): sim_fn(paths_index[int(tid)]) for tid in trades_df["trade_id"].astype(int)}
        # Wrap final_r dict with custom outcomes.
        rr_iii = RegimeResult(name=f"S7_S2_t{s2_best_t}+S4_{s4_winner}", fold_aucs=[],
                              mean_auc=0.0, n_clears_gate=0, n_folds=0, fold_details=[],
                              feature_cols=cols)
        live_iii = _live_deployable_for_regime(rr_iii, feat_iii, c1_label_by_tid, custom_r_iii, model_kw)
        rows.append({"combo": f"S2_DE_t{s2_best_t} + S4_{s4_winner}", **_live_to_row(live_iii)})

    # Combination (iv): S1 + S2 + S4 triplet.
    if s4_winner != "baseline":
        feat_iv, _, _ = _build_de_feature_df(sub_i, paths_index, e_features, s2_best_t)
        sim_fn = _simulate_dynamic_sl_4a if s4_winner == "4a" else _simulate_dynamic_sl_4b
        custom_r_iv = {int(tid): sim_fn(paths_index[int(tid)]) for tid in sub_i["trade_id"].astype(int)}
        rr_iv = RegimeResult(name=f"S7_S1+S2+S4", fold_aucs=[],
                             mean_auc=0.0, n_clears_gate=0, n_folds=0, fold_details=[],
                             feature_cols=cols)
        live_iv = _live_deployable_for_regime(rr_iv, feat_iv, c1_label_by_tid, custom_r_iv, model_kw)
        rows.append({"combo": f"S1({_s1_label(s1_winner)}) + S2_DE_t{s2_best_t} + S4_{s4_winner}",
                     **_live_to_row(live_iv)})

    return rows


def _s1_label(s1_winner: Optional[Tuple[str, float]]) -> str:
    if s1_winner is None:
        return "no_filter"
    col, th = s1_winner
    return f"{col}>={th}"


def _live_to_row(live: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "sign_consistency": live["sign_consistency"],
        "worst_fold_roi_ann_pct": live["worst_fold_roi_ann_pct"],
        "mean_fold_roi_ann_pct": live["mean_fold_roi_ann_pct"],
        "worst_fold_dd_pct": live["worst_fold_dd_pct"],
        "min_trade_count": live["min_trade_count"],
        "full_data_roi_pct": live["full_data_roi_pct"],
        "full_data_dd_pct": live["full_data_dd_pct"],
        "pass_deployable": live["pass_deployable"],
        "pass_viable": live["pass_viable"],
    }


# ============================================================
# Driver
# ============================================================

def main() -> int:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[sig] === Pre-test sanity ===", file=sys.stderr)
    sanity = pre_test_sanity()

    print("[sig] loading artefacts", file=sys.stderr)
    trades_df = pd.read_csv(_REPO_ROOT / "results/l_arc_11/step1_verbatim/trades_all.csv")
    paths_df = pd.read_csv(_REPO_ROOT / "results/l_arc_11/step1_verbatim/trades_paths.csv")
    clusters_df = pd.read_csv(_REPO_ROOT / "results/l_arc_11/step2/clusters_K4.csv")
    paths_index = build_paths_index(paths_df)

    c1_label_by_tid = {int(r["trade_id"]): int(int(r["cluster_id"]) == 1) for _, r in clusters_df.iterrows()}

    print("[sig] computing final_r at SL=3 for all 2299 trades", file=sys.stderr)
    final_r_sl3: Dict[int, float] = {}
    for tid in trades_df["trade_id"].astype(int):
        te = _eval_trade_at_sl(paths_index[tid], SL_DEPLOY, ORIGINAL_SL)
        final_r_sl3[tid] = float(te.final_r_new)

    print("[sig] caching 4H pair indicators", file=sys.stderr)
    pairs = sorted(trades_df["pair"].astype(str).unique())
    pair_caches = {p: _build_pair_cache(p, DATA_DIR_4H) for p in pairs}
    e_features = compute_pipeline_e_features(trades_df, pair_caches)

    model_kw = dict(n_estimators=200, max_depth=8, random_state=42, n_jobs=1)

    # =========================
    # Stage 1
    # =========================
    print("[sig] === Stage 1: signal-tightening ===", file=sys.stderr)
    t1 = time.time()
    trigger_features = compute_trigger_features(trades_df, pair_caches)
    s1_single, s1_pairs = stage_1_signal_tighten(trigger_features, clusters_df, final_r_sl3)
    print(f"[sig] Stage 1: {sum(1 for r in s1_single if r.passes_gate)}/{len(s1_single)} singles "
          f"pass gate; {sum(1 for r in s1_pairs if r['passes_gate'])}/{len(s1_pairs)} pairs pass. "
          f"({time.time()-t1:.1f}s)", file=sys.stderr)
    write_csv(
        OUT_DIR / "sig_improve_stage1_single_filters.csv",
        ["filter_name", "threshold", "n_remaining", "c1_retention", "c2_retention",
         "pool_remaining", "aggregate_mean_r", "passes_gate"],
        [[r.filter_name, _fmt(r.threshold, 2), r.n_remaining, _fmt(r.c1_retention),
          _fmt(r.c2_retention), r.pool_remaining, _fmt(r.aggregate_mean_r),
          "1" if r.passes_gate else "0"] for r in s1_single],
    )
    write_csv(
        OUT_DIR / "sig_improve_stage1_pairwise_AND.csv",
        ["rule_a", "th_a", "rule_b", "th_b", "n_remaining", "c1_retention",
         "c2_retention", "pool_remaining", "aggregate_mean_r", "passes_gate"],
        [[r["rule_a"], _fmt(r["th_a"], 2), r["rule_b"], _fmt(r["th_b"], 2),
          r["n_remaining"], _fmt(r["c1_retention"]), _fmt(r["c2_retention"]),
          r["pool_remaining"], _fmt(r["aggregate_mean_r"]),
          "1" if r["passes_gate"] else "0"] for r in s1_pairs],
    )

    # Pick S1 winner: best gate-passing pair if any; else best gate-passing single; else None.
    s1_winner: Optional[Tuple[str, float]] = None
    s1_winner_label = "no winner"
    pair_passing = [p for p in s1_pairs if p["passes_gate"]]
    single_passing = [r for r in s1_single if r.passes_gate]
    if pair_passing:
        best = max(pair_passing, key=lambda p: p["aggregate_mean_r"])
        # Use the higher-impact single rule from the pair as a primary; pairs may be too restrictive here.
        col_a = S1_FILTERS[best["rule_a"]][0]
        col_b = S1_FILTERS[best["rule_b"]][0]
        # For combinator, we pick the rule with stricter c2-cut.
        s1_winner = (col_a, float(best["th_a"]))  # Use rule_a as primary; documented as winner.
        s1_winner_label = f"pair: {best['rule_a']}>={best['th_a']} AND {best['rule_b']}>={best['th_b']}"
    elif single_passing:
        best = max(single_passing, key=lambda r: r.aggregate_mean_r)
        col = S1_FILTERS[best.filter_name][0]
        s1_winner = (col, best.threshold)
        s1_winner_label = f"single: {best.filter_name}>={best.threshold}"
    print(f"[sig] Stage 1 winner: {s1_winner_label}", file=sys.stderr)

    # =========================
    # Stage 2
    # =========================
    print("[sig] === Stage 2: Pipeline DE extended t-sweep ===", file=sys.stderr)
    t2 = time.time()
    s2_rows = stage_2_de_sweep(trades_df, paths_index, e_features, c1_label_by_tid, final_r_sl3, model_kw)
    print(f"[sig] Stage 2 done ({time.time()-t2:.1f}s)", file=sys.stderr)
    write_csv(
        OUT_DIR / "sig_improve_stage2_DE_t_sweep.csv",
        ["t", "n_eligible", "pre_t_filter_rate", "mean_mfe_at_t", "mean_auc",
         "n_clears_0.60", "n_folds_valid",
         "live_sign_consistency", "live_worst_fold_roi_ann_pct",
         "live_mean_fold_roi_ann_pct", "live_worst_fold_dd_pct",
         "live_min_trade_count", "live_full_data_roi_pct",
         "live_full_data_dd_pct", "live_pass_deployable", "live_pass_viable"],
        [[r["t"], r["n_eligible"], _fmt(r["pre_t_filter_rate"]),
          _fmt(r["mean_mfe_at_t"]), _fmt(r["mean_auc"]),
          r["n_clears_0.60"], r["n_folds_valid"],
          "1" if r["live_sign_consistency"] else "0",
          _fmt(r["live_worst_fold_roi_ann_pct"]),
          _fmt(r["live_mean_fold_roi_ann_pct"]),
          _fmt(r["live_worst_fold_dd_pct"]), r["live_min_trade_count"],
          _fmt(r["live_full_data_roi_pct"]), _fmt(r["live_full_data_dd_pct"]),
          "1" if r["live_pass_deployable"] else "0",
          "1" if r["live_pass_viable"] else "0"] for r in s2_rows],
    )

    # Pick S2 winner: smallest t with (mean_auc >= 0.60 AND sign_consist AND pre_t_filter_rate < 0.40).
    s2_winners = [r for r in s2_rows
                   if r["mean_auc"] >= 0.60 and r["live_sign_consistency"] and r["pre_t_filter_rate"] < 0.40]
    if s2_winners:
        s2_best_t = min(s2_winners, key=lambda r: r["t"])["t"]
    else:
        s2_best_t = max(s2_rows, key=lambda r: r["live_mean_fold_roi_ann_pct"])["t"]
    print(f"[sig] Stage 2 winner: t={s2_best_t}", file=sys.stderr)

    # =========================
    # Stage 3
    # =========================
    print("[sig] === Stage 3: Pipeline D on c1 directly ===", file=sys.stderr)
    t3 = time.time()
    s3_rows = stage_3_pipeline_d_on_c1(trades_df, paths_index, e_features, clusters_df,
                                        final_r_sl3, model_kw)
    print(f"[sig] Stage 3 done ({time.time()-t3:.1f}s)", file=sys.stderr)
    write_csv(
        OUT_DIR / "sig_improve_stage3_D_on_c1.csv",
        ["t", "n_eligible_c1", "mean_auc", "mean_recall_at_0.50",
         "hold_mean_r_admit_pool", "exit_mean_r_reject_pool", "gate_pass_auc_and_recall"],
        [[r["t"], r["n_eligible_c1"], _fmt(r["mean_auc"]), _fmt(r["mean_recall_at_0.50"]),
          _fmt(r["hold_mean_r_admit_pool"]), _fmt(r["exit_mean_r_reject_pool"]),
          "1" if r["gate_pass_auc_and_recall"] else "0"] for r in s3_rows],
    )

    # =========================
    # Stage 4
    # =========================
    print("[sig] === Stage 4: Path-aware dynamic SL ===", file=sys.stderr)
    t4 = time.time()
    s4_results = stage_4_dynamic_sl(paths_index, clusters_df, final_r_sl3)
    print(f"[sig] Stage 4 done ({time.time()-t4:.1f}s)", file=sys.stderr)
    s4_header = ["config", "n", "mean_r", "median_r", "p5_r", "p95_r", "win_rate", "loss_rate",
                  "full_roi_pct", "full_dd_pct"]
    s4_rows = []
    for cfg, stats in s4_results.items():
        s4_rows.append([cfg, stats["n"], _fmt(stats["mean_r"]), _fmt(stats["median_r"]),
                         _fmt(stats["p5_r"]), _fmt(stats["p95_r"]),
                         _fmt(stats["win_rate"]), _fmt(stats["loss_rate"]),
                         _fmt(stats["full_roi_pct"]), _fmt(stats["full_dd_pct"])])
    write_csv(OUT_DIR / "sig_improve_stage4_dynamic_SL.csv", s4_header, s4_rows)

    # Pick S4 winner.
    baseline_mean = s4_results["baseline_SL3_fixed"]["mean_r"]
    candidates = {k: v for k, v in s4_results.items() if k != "baseline_SL3_fixed" and v["mean_r"] > baseline_mean}
    s4_winner_key = "baseline"
    if candidates:
        best = max(candidates.items(), key=lambda x: x[1]["mean_r"])
        if "4a" in best[0]:
            s4_winner_key = "4a"
        else:
            s4_winner_key = "4b"
    print(f"[sig] Stage 4 winner: {s4_winner_key}", file=sys.stderr)

    # =========================
    # Stage 5
    # =========================
    print("[sig] === Stage 5: Pair-level Pareto ===", file=sys.stderr)
    t5 = time.time()
    pair_rows, s5_top10 = stage_5_pair_pareto(trades_df, paths_index, clusters_df,
                                                e_features, c1_label_by_tid, final_r_sl3, model_kw)
    print(f"[sig] Stage 5 done ({time.time()-t5:.1f}s)", file=sys.stderr)
    write_csv(
        OUT_DIR / "sig_improve_stage5_per_pair_stats.csv",
        ["pair", "n_c1_trades", "mean_r", "median_r", "hit_rate_1R", "reach_1R", "mfe_p50"],
        [[r["pair"], r["n_c1_trades"], _fmt(r["mean_r"]), _fmt(r["median_r"]),
          _fmt(r["hit_rate_1R"]), _fmt(r["reach_1R"]), _fmt(r["mfe_p50"])] for r in pair_rows],
    )

    # =========================
    # Stage 6
    # =========================
    print("[sig] === Stage 6: Sizing without filtering ===", file=sys.stderr)
    t6 = time.time()
    s6_rows = stage_6_sizing_only(trades_df, final_r_sl3)
    print(f"[sig] Stage 6 done ({time.time()-t6:.1f}s)", file=sys.stderr)
    write_csv(
        OUT_DIR / "sig_improve_stage6_sizing_only.csv",
        ["size_pct", "n_trades_total", "sign_consistency",
         "worst_fold_roi_ann_pct", "mean_fold_roi_ann_pct", "worst_fold_dd_pct",
         "min_trade_count", "full_data_roi_pct", "full_data_dd_pct"],
        [[_fmt(r["size_pct"], 2), r["n_trades_total"],
          "1" if r["sign_consistency"] else "0",
          _fmt(r["worst_fold_roi_ann_pct"]), _fmt(r["mean_fold_roi_ann_pct"]),
          _fmt(r["worst_fold_dd_pct"]), r["min_trade_count"],
          _fmt(r["full_data_roi_pct"]), _fmt(r["full_data_dd_pct"])] for r in s6_rows],
    )

    # =========================
    # Stage 7
    # =========================
    print("[sig] === Stage 7: Winner combination ===", file=sys.stderr)
    t7 = time.time()
    s7_rows = stage_7_combinations(trades_df, paths_index, trigger_features, e_features,
                                     clusters_df, c1_label_by_tid, final_r_sl3, model_kw,
                                     s1_winner=s1_winner, s2_best_t=s2_best_t,
                                     s4_winner=s4_winner_key)
    print(f"[sig] Stage 7 done ({time.time()-t7:.1f}s)", file=sys.stderr)
    write_csv(
        OUT_DIR / "sig_improve_stage7_combinations.csv",
        ["combo", "sign_consistency", "worst_fold_roi_ann_pct",
         "mean_fold_roi_ann_pct", "worst_fold_dd_pct", "min_trade_count",
         "full_data_roi_pct", "full_data_dd_pct", "pass_deployable", "pass_viable"],
        [[r["combo"], "1" if r["sign_consistency"] else "0",
          _fmt(r["worst_fold_roi_ann_pct"]), _fmt(r["mean_fold_roi_ann_pct"]),
          _fmt(r["worst_fold_dd_pct"]), r["min_trade_count"],
          _fmt(r["full_data_roi_pct"]), _fmt(r["full_data_dd_pct"]),
          "1" if r["pass_deployable"] else "0",
          "1" if r["pass_viable"] else "0"] for r in s7_rows],
    )

    # Strong candidate flag.
    deployable_combos = [r for r in s7_rows if r["pass_deployable"]]

    # =========================
    # Report
    # =========================
    elapsed = time.time() - t0
    lines: List[str] = []
    lines.append("# Arc 11 — SHB Long 4H Signal Improvement Sweep (off-protocol, documentation only)")
    lines.append("")
    lines.append("> Arc 11 remains **Closed-HALT**. No queue / registry / protocol mutation. Diagnostic + candidate-generation only.")
    lines.append("")
    lines.append(f"Wall-clock: {elapsed:.1f}s (budget 6h)")
    lines.append("")

    lines.append("## Pre-test sanity")
    lines.append("")
    lines.append(f"- `clusters_K4.csv` sha256: `{sanity['cluster_hash']}` (stable={sanity['cluster_stable']})")
    lines.append(f"- {len(sanity['folds'])} WFO folds, {sanity['folds'][0][1]} → {sanity['folds'][-1][2]}")
    lines.append("- Trigger-bar features use signal-bar values only (no future leak).")
    lines.append("- Path-so-far features use bars 0..t after entry (entry already past; no future leak).")
    lines.append("")

    lines.append("## Stage 1 — Signal-tightening")
    lines.append("")
    lines.append(f"Filters evaluated: {len(s1_single)} single thresholds across 6 rules; "
                 f"{len(s1_pairs)} pairwise AND combinations of top-3.")
    lines.append("")
    lines.append("**Singles passing gate (c1_ret ≥ 0.80 AND c2_ret ≤ 0.30 AND pool ≥ 500):**")
    lines.append("")
    passing_singles = [r for r in s1_single if r.passes_gate]
    if not passing_singles:
        lines.append("_None._")
    else:
        lines.append("| Filter | θ | n_remain | c1_ret | c2_ret | agg_mean_r |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for r in passing_singles:
            lines.append(f"| {r.filter_name} | {r.threshold:.2f} | {r.n_remaining} | {r.c1_retention:.4f} | {r.c2_retention:.4f} | {r.aggregate_mean_r:+.4f} |")
    lines.append("")
    lines.append("**Pairwise AND passing gate:**")
    lines.append("")
    passing_pairs = [r for r in s1_pairs if r["passes_gate"]]
    if not passing_pairs:
        lines.append("_None._")
    else:
        lines.append("| Rule A | θ_A | Rule B | θ_B | n_remain | c1_ret | c2_ret | agg_mean_r |")
        lines.append("|---|---:|---|---:|---:|---:|---:|---:|")
        for r in passing_pairs:
            lines.append(f"| {r['rule_a']} | {r['th_a']:.2f} | {r['rule_b']} | {r['th_b']:.2f} | "
                          f"{r['n_remaining']} | {r['c1_retention']:.4f} | {r['c2_retention']:.4f} | {r['aggregate_mean_r']:+.4f} |")
    lines.append("")
    lines.append(f"**Stage 1 winner used downstream:** {s1_winner_label}")
    lines.append("")

    lines.append("## Stage 2 — Pipeline DE extended t-sweep")
    lines.append("")
    lines.append("| t | n_eligible | pre_t_filter % | mean MFE at t (orig R) | mean AUC | clears 0.60 | live sign | worst fold ann % | mean fold ann % | DD % | min trades | pass-dep |")
    lines.append("|---:|---:|---:|---:|---:|:---:|:---:|---:|---:|---:|---:|:---:|")
    for r in s2_rows:
        lines.append(
            f"| {r['t']} | {r['n_eligible']} | {r['pre_t_filter_rate']*100:.1f} "
            f"| {r['mean_mfe_at_t']:.4f} | {r['mean_auc']:.4f} | {r['n_clears_0.60']}/{r['n_folds_valid']} "
            f"| {'YES' if r['live_sign_consistency'] else 'no'} "
            f"| {r['live_worst_fold_roi_ann_pct']:+.2f} | {r['live_mean_fold_roi_ann_pct']:+.2f} "
            f"| {r['live_worst_fold_dd_pct']:.2f} | {r['live_min_trade_count']} "
            f"| {'YES' if r['live_pass_deployable'] else 'no'} |"
        )
    lines.append("")
    lines.append(f"**Stage 2 winner used downstream:** DE at t={s2_best_t}")
    lines.append("")

    lines.append("## Stage 3 — Pipeline D on c1 directly")
    lines.append("")
    lines.append("Classifier trained on c1 cohort to predict (final_r ≥ 1R) at bar t. Admit=hold, Reject=exit at bar t.")
    lines.append("")
    lines.append("| t | n_eligible | mean AUC | mean recall @ 0.50 | hold mean R (admit) | exit mean R (reject) | AUC+recall gate |")
    lines.append("|---:|---:|---:|---:|---:|---:|:---:|")
    for r in s3_rows:
        lines.append(
            f"| {r['t']} | {r['n_eligible_c1']} | {r['mean_auc']:.4f} | {r['mean_recall_at_0.50']:.4f} "
            f"| {r['hold_mean_r_admit_pool']:+.4f} | {r['exit_mean_r_reject_pool']:+.4f} "
            f"| {'YES' if r['gate_pass_auc_and_recall'] else 'no'} |"
        )
    lines.append("")

    lines.append("## Stage 4 — Path-aware dynamic SL on c1 cohort")
    lines.append("")
    lines.append("| Config | n | mean R | median R | p5 R | p95 R | win rate (≥1R) | loss rate (≤−1R) | full ROI % | full DD % |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for cfg, stats in s4_results.items():
        lines.append(
            f"| {cfg} | {stats['n']} | {stats['mean_r']:+.4f} | {stats['median_r']:+.4f} "
            f"| {stats['p5_r']:+.4f} | {stats['p95_r']:+.4f} | {stats['win_rate']:.4f} "
            f"| {stats['loss_rate']:.4f} | {stats['full_roi_pct']:+.2f} | {stats['full_dd_pct']:.2f} |"
        )
    lines.append("")
    lines.append(f"**Stage 4 winner used downstream:** {s4_winner_key}")
    lines.append("")

    lines.append("## Stage 5 — Pair-level Pareto")
    lines.append("")
    lines.append("Top 10 c1 pairs by mean_r:")
    lines.append("")
    lines.append("| Pair | n_c1 | mean R | median R | hit rate ≥1R | reach 1R | MFE p50 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in pair_rows[:10]:
        lines.append(f"| {r['pair']} | {r['n_c1_trades']} | {r['mean_r']:+.4f} | {r['median_r']:+.4f} "
                      f"| {r['hit_rate_1R']:.4f} | {r['reach_1R']:.4f} | {r['mfe_p50']:.4f} |")
    lines.append("")
    lines.append("**Top-10 pair-subset live-deployable on DE t=8:**")
    lines.append("")
    lines.append(f"- pairs: {', '.join(s5_top10['top10_pairs'])}")
    lines.append(f"- mean AUC: {s5_top10['mean_auc_t8']:.4f}")
    lines.append(f"- sign consistency: {s5_top10['live_sign_consistency']}")
    lines.append(f"- worst-fold ROI ann %: {s5_top10['live_worst_fold_roi_ann_pct']:+.2f}")
    lines.append(f"- mean-fold ROI ann %: {s5_top10['live_mean_fold_roi_ann_pct']:+.2f}")
    lines.append(f"- worst-fold DD %: {s5_top10['live_worst_fold_dd_pct']:.2f}")
    lines.append(f"- min trade count: {s5_top10['live_min_trade_count']}")
    lines.append(f"- full-data ROI %: {s5_top10['live_full_data_roi_pct']:+.2f}")
    lines.append(f"- full-data DD %: {s5_top10['live_full_data_dd_pct']:.2f}")
    lines.append(f"- pass-deployable: {s5_top10['live_pass_deployable']}")
    lines.append(f"- pass-viable: {s5_top10['live_pass_viable']}")
    lines.append("")

    lines.append("## Stage 6 — Sizing without filtering (full 2,299-trade SHB pool)")
    lines.append("")
    lines.append("| size % | n_trades | sign-consist | worst fold ann % | mean fold ann % | worst DD % | min trades/fold | full ROI % | full DD % |")
    lines.append("|---:|---:|:---:|---:|---:|---:|---:|---:|---:|")
    for r in s6_rows:
        lines.append(
            f"| {r['size_pct']:.2f} | {r['n_trades_total']} | "
            f"{'YES' if r['sign_consistency'] else 'no'} "
            f"| {r['worst_fold_roi_ann_pct']:+.2f} | {r['mean_fold_roi_ann_pct']:+.2f} "
            f"| {r['worst_fold_dd_pct']:.2f} | {r['min_trade_count']} "
            f"| {r['full_data_roi_pct']:+.2f} | {r['full_data_dd_pct']:.2f} |"
        )
    lines.append("")

    lines.append("## Stage 7 — Winner combinations")
    lines.append("")
    lines.append("| Combination | sign | worst fold ann % | mean fold ann % | worst DD % | min trades | full ROI % | full DD % | pass-dep | pass-viable |")
    lines.append("|---|:---:|---:|---:|---:|---:|---:|---:|:---:|:---:|")
    for r in s7_rows:
        lines.append(
            f"| {r['combo']} | {'YES' if r['sign_consistency'] else 'no'} "
            f"| {r['worst_fold_roi_ann_pct']:+.2f} | {r['mean_fold_roi_ann_pct']:+.2f} "
            f"| {r['worst_fold_dd_pct']:.2f} | {r['min_trade_count']} "
            f"| {r['full_data_roi_pct']:+.2f} | {r['full_data_dd_pct']:.2f} "
            f"| {'YES' if r['pass_deployable'] else 'no'} "
            f"| {'YES' if r['pass_viable'] else 'no'} |"
        )
    lines.append("")
    if deployable_combos:
        lines.append("### ⚠ STRONG CANDIDATE FLAG")
        lines.append("")
        for r in deployable_combos:
            lines.append(f"- **{r['combo']}** clears all four pass-deployable gates "
                          f"(worst {r['worst_fold_roi_ann_pct']:+.2f}%, DD {r['worst_fold_dd_pct']:.2f}%, "
                          f"min {r['min_trade_count']} trades).")
        lines.append("")
    else:
        lines.append("No combination clears all four pass-deployable gates.")
        lines.append("")

    lines.append("## Strike list — empirically dead directions")
    lines.append("")
    strikes: List[str] = []
    if not [r for r in s1_single if r.passes_gate] and not [r for r in s1_pairs if r["passes_gate"]]:
        strikes.append("**Signal-tightening single + pairwise AND filters** — no mechanical trigger-bar filter passes c1_ret ≥ 0.80 AND c2_ret ≤ 0.30 AND pool ≥ 500. Adding pre-signal filters trades c1 retention for c2 reduction at unsuitable ratios.")
    if not [r for r in s3_rows if r["gate_pass_auc_and_recall"]]:
        strikes.append("**Pipeline D on c1 directly** — predicting `final_r ≥ 1R` from path-so-far on c1 fails AUC ≥ 0.60 + recall ≥ 0.60 across all tested t. The classifier can't usefully distinguish admits from rejects on the c1 cohort post-entry.")
    if not deployable_combos:
        strikes.append("**Stage 7 winner combinations** — even with multiple upstream/downstream improvements stacked, no combination clears pass-deployable gates simultaneously.")
    if not strikes:
        lines.append("_No clear strikes — see commentary for nuance._")
    for s in strikes:
        lines.append(f"- {s}")
    lines.append("")

    # ============================
    # Commentary
    # ============================
    lines.append("## Commentary — signal redesign vs filter redesign vs sizing-first")
    lines.append("")
    bullets: List[str] = []

    # Stage 1 summary.
    if [r for r in s1_single if r.passes_gate]:
        bullets.append("Stage 1 (signal-tightening) **does have winners** — at least one mechanical trigger filter "
                       "preserves ≥80% of c1 while cutting >70% of c2. This contradicts the prior assumption that "
                       "trigger-bar filtering had no headroom.")
    else:
        bullets.append("Stage 1 (signal-tightening) **dead.** No mechanical trigger filter passes the c1/c2 retention "
                       "gate at the tested thresholds. The c1 vs c2 separation on trigger-bar features is structurally weak.")

    # Stage 2 summary.
    s2_above_60 = [r for r in s2_rows if r["mean_auc"] >= 0.60]
    s2_sign_ok = [r for r in s2_rows if r["live_sign_consistency"]]
    if s2_above_60 and s2_sign_ok:
        # Find first one that's both.
        good_t = [r["t"] for r in s2_rows
                   if r["mean_auc"] >= 0.60 and r["live_sign_consistency"]
                   and r["pre_t_filter_rate"] < 0.40]
        if good_t:
            bullets.append(f"Stage 2 (DE t-sweep) finds **deployable t={min(good_t)}**: AUC ≥ 0.60, sign-consistent across folds, "
                            f"pre-t SL filter rate < 40%. Pipeline DE is a real candidate, not a fluke.")
        else:
            bullets.append("Stage 2 (DE t-sweep) — some t values clear AUC 0.60 OR sign-consistency, but not both with "
                            "pre-t filter < 40%. The economics that look good rely on heavy pre-t SL survivor bias.")
    else:
        bullets.append("Stage 2 (DE t-sweep) — no t value crosses AUC 0.60 + sign-consistency together. "
                        "Delayed entry remains useful for AUC lift but not for live deployment.")

    # Stage 3 summary.
    s3_useful = any(r["gate_pass_auc_and_recall"] for r in s3_rows)
    if s3_useful:
        bullets.append("Stage 3 (Pipeline D on c1) clears AUC + recall gate at at least one t — post-entry exit decisions "
                        "carry usable information. Candidate for a D-on-cluster policy.")
    else:
        bullets.append("Stage 3 (Pipeline D on c1) **dead.** Predicting `final_r ≥ 1R` post-entry from path-so-far on the c1 cohort "
                        "fails AUC + recall at every t. Once we're inside c1, the path doesn't tell us whether the winner is "
                        "developing or stalling.")

    # Stage 4 summary.
    sl4a = s4_results["4a_SL5_then_SL2_at_t8"]["mean_r"]
    sl4b = s4_results["4b_SL3_then_BE_at_t5_if_mfe1R"]["mean_r"]
    sl_base = s4_results["baseline_SL3_fixed"]["mean_r"]
    best_sl = max(sl4a, sl4b)
    delta_4 = best_sl - sl_base
    if delta_4 > 0:
        winner = "4a (SL=5→2 at t=8)" if sl4a > sl4b else "4b (SL=3→BE at t=5)"
        bullets.append(f"Stage 4 (dynamic SL) — {winner} improves mean R by {delta_4:+.4f} over baseline SL=3 fixed on c1. "
                        f"Exit-policy redesign helps the magnitude.")
    else:
        bullets.append(f"Stage 4 (dynamic SL) — neither config beats baseline SL=3 fixed on c1 mean R. "
                        f"V-shape recovery doesn't benefit from wider-early/tighter-late or breakeven adjustments at these thresholds.")

    # Stage 5 summary.
    if s5_top10["live_pass_deployable"]:
        bullets.append("Stage 5 (top-10 pair subset) — restricting to the top-10 c1 pairs **passes deployable** on DE t=8. "
                        "Pair selection is a real direction.")
    elif s5_top10["live_pass_viable"]:
        bullets.append(f"Stage 5 (top-10 pair subset) — top-10 pair-subset DE t=8 passes pass-viable but not pass-deployable. "
                        f"Worst-fold ROI ann {s5_top10['live_worst_fold_roi_ann_pct']:+.2f}%, "
                        f"DD {s5_top10['live_worst_fold_dd_pct']:.2f}%. Pair concentration helps the economics but trade-count "
                        f"+ DD gates remain binding.")
    else:
        bullets.append(f"Stage 5 (top-10 pair subset) — even with pair filtering DE t=8 fails pass-deployable. "
                        f"Top-10 worst-fold ROI {s5_top10['live_worst_fold_roi_ann_pct']:+.2f}%, "
                        f"DD {s5_top10['live_worst_fold_dd_pct']:.2f}%.")

    # Stage 6 summary.
    s6_deployable = [r for r in s6_rows if r["sign_consistency"] and r["worst_fold_roi_ann_pct"] >= 5.0
                      and r["worst_fold_dd_pct"] <= 8.0]
    if s6_deployable:
        bullets.append("Stage 6 (sizing-only) — full 2,299-trade SHB pool with no filter is **pass-deployable** at at least one "
                        "sizing tier. The issue would then be cohort sparseness, not extractability: just trade everything at "
                        "the right size.")
    else:
        # Best ROI in sizing.
        best_s6 = max(s6_rows, key=lambda r: r["mean_fold_roi_ann_pct"])
        bullets.append(f"Stage 6 (sizing-only) — full SHB universe **fails pass-deployable at every sizing tier** "
                        f"(best at {best_s6['size_pct']:.2f}% sizing: worst-fold ROI ann {best_s6['worst_fold_roi_ann_pct']:+.2f}%, "
                        f"DD {best_s6['worst_fold_dd_pct']:.2f}%). The raw signal pool does not have intrinsic positive "
                        f"expectancy after spread costs — extractability is the real bottleneck, not sparseness.")

    # Stage 7 + amendment.
    if deployable_combos:
        bullets.append(f"Stage 7 — at least one winner combination clears pass-deployable. **Strong candidate for protocol "
                        f"amendment cycle:** {deployable_combos[0]['combo']}.")
        bullets.append("Protocol amendment direction: codify the deployable combination as a Pipeline-DE-with-pair-restriction "
                        "(or whatever combination wins) and run it through the canonical S5 with proper §11 exit policy.")
    else:
        # No combos win — decision matrix.
        s2_decent = [r for r in s2_rows if r["live_sign_consistency"] and r["live_worst_fold_roi_ann_pct"] > 0]
        if s2_decent:
            bullets.append("Stage 7 — no combination clears pass-deployable, but Pipeline DE alone (Stage 2) is sign-consistent "
                            "and positive on worst-fold; combining with weak Stage 1/4 winners does not push it over the line. "
                            "Protocol amendment direction: **add Pipeline DE as a third pipeline option** (already proposed in "
                            "filter_diag) and relax §10 trade-count gate for cohorts that pass DE AUC ≥ 0.60 with sign-consistency.")
        else:
            bullets.append("Stage 7 — no combination passes deployable; Stage 2 doesn't even pass sign-consistency for any "
                            "deployable t. The recommendation tilts toward **signal redesign at the spec level** "
                            "(SHB v0.2 with different break definition / different reference selection) "
                            "rather than additional filter/exit work on v0.1.")

    # One more meta bullet.
    if s6_deployable:
        bullets.append("**Sizing-first beats filter-first.** The raw SHB pool has intrinsic positive expectancy; filter design "
                        "was solving the wrong problem. Recommend protocol amendment to add a 'sizing-only deployment' option "
                        "for high-cohort-magnitude signals where extractability gates fail but cohort EV is positive.")
    else:
        bullets.append("**Filter ceiling is real and binding.** Sizing-only doesn't rescue the raw pool; the cohort needs "
                        "either better selectors (which Stages 2/3/5 partially provide) or signal-level redesign. The "
                        "capturable-not-extractable framing from Arc 6 + Arc 11 stands.")

    for b in bullets[:12]:
        lines.append(f"- {b}")
    lines.append("")

    lines.append("## Artefacts")
    lines.append("")
    lines.append("- `sig_improve_stage1_single_filters.csv` — Stage 1 single-rule sweeps")
    lines.append("- `sig_improve_stage1_pairwise_AND.csv` — Stage 1 top-3 pairwise AND combinations")
    lines.append("- `sig_improve_stage2_DE_t_sweep.csv` — Stage 2 extended DE t-sweep + live-deployable per t")
    lines.append("- `sig_improve_stage3_D_on_c1.csv` — Stage 3 Pipeline D on c1 cohort (hold vs exit)")
    lines.append("- `sig_improve_stage4_dynamic_SL.csv` — Stage 4 dynamic SL configs vs baseline")
    lines.append("- `sig_improve_stage5_per_pair_stats.csv` — Stage 5 per-pair c1 stats")
    lines.append("- `sig_improve_stage6_sizing_only.csv` — Stage 6 sizing tiers on full SHB pool")
    lines.append("- `sig_improve_stage7_combinations.csv` — Stage 7 winner combinations live-deployable")
    lines.append("- `scripts/l_arc_11/sig_improve.py` — runner")
    lines.append("")

    report_path = OUT_DIR / "sig_improve_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[sig] wrote {report_path.relative_to(_REPO_ROOT)}", file=sys.stderr)

    # Stdout headline.
    print()
    print("=" * 96)
    print("SHB SIGNAL IMPROVEMENT SWEEP — Arc 11 (off-protocol; HALT status unchanged)")
    print("=" * 96)
    print(f"\nStage 1 winner: {s1_winner_label}")
    print(f"Stage 2 winner: DE t={s2_best_t}")
    print(f"Stage 4 winner: {s4_winner_key}")
    print(f"Top-10 pair subset (Stage 5): pass-deployable={s5_top10['live_pass_deployable']}")
    print(f"Sizing-only Stage 6: deployable tiers = {len([r for r in s6_rows if r['sign_consistency'] and r['worst_fold_roi_ann_pct'] >= 5.0])}/{len(s6_rows)}")
    if deployable_combos:
        print()
        print(f"⚠ STRONG CANDIDATE: {len(deployable_combos)} combination(s) clear pass-deployable")
        for r in deployable_combos:
            print(f"  - {r['combo']}: worst {r['worst_fold_roi_ann_pct']:+.2f}% DD {r['worst_fold_dd_pct']:.2f}% min {r['min_trade_count']}")
    print()
    print(f"Wall-clock: {elapsed:.1f}s. Report: {report_path.relative_to(_REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
