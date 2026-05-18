"""Arc 11 — Experimental Step 5 WFO (off-protocol, documentation only).

NOT a canonical S5 run. Arc 11 is Closed-HALT per §16a Path A regardless of
the outcome of this script. Outputs tagged `arc11_exp_s5_*` to keep separate
from any future canonical S5 artefacts.

Two runs:
  Run A — c1 raw (no admission filter) at SL=3.0×ATR
  Run B — agg_c1_c3 with Pipeline D1 t=5 classifier at SL=3.0×ATR

Method (intentionally simplified vs canonical §10):
  - 7 anchored expanding folds mirroring KH-24 OOS windows
    (2020-10 → 2026-01).
  - Per fold, OOS = trades whose entry_time falls in the fold window;
    IS = all trades with entry_time strictly before fold's OOS start.
  - Trade outcome = `final_r` from re-imposing SL=3.0 on the §15a bar path
    (via _eval_trade_at_sl, byte-identical to Step 3 / Step 4). Truncates at
    SL if hit; otherwise runs the recorded forward window to end (≤240 bars
    or end-of-data).
  - **Simplification vs §11 V-shape recovery exit policy:** uses fixed-SL +
    time-exit truncation; does NOT simulate the "standard trail after
    recovery confirmed" trail logic. Captures the magnitude truth of the
    cohort under the cluster's Step 3 selected SL but does NOT capture trail
    upside or trail-induced early exits. Flagged in the commentary.
  - Equity compounding: 0.5% risk per trade; equity *= (1 + final_r × 0.005)
    per trade chronologically. Per-fold ROI from compounded equity.
  - Annualised ROI: (1 + roi)^(365.25/fold_days) - 1.
  - DD: max drawdown on the OOS equity curve per fold.

Run B classifier per fold:
  - IS trades: bars_held ≥ 5 with entry_time strictly < fold OOS start;
    require ≥50 trades to train (else skip fold's classifier — fold reports
    raw-equivalent outcome).
  - Features: 8 cross-dataset base entry + 7 path-so-far at t=5 (in new
    R-frame at SL=3.0). Same feature set as Step 4 D1.
  - RF (200, max_depth 8, random_state 42, n_jobs 1, class_weight=balanced
    when minority < 30%).
  - Admission threshold = 0.50 (fixed midpoint of v2.2 §3 sweep grid; not
    selected by recall ≥ 0.60 rule — that rule never fires in Arc 11 S4).
  - Per OOS trade outcome:
      bars_held < 5      → pre-t SL hit, final_r = −1.0
      P(success) ≥ 0.50  → admit, use final_r at SL=3.0
      P(success) < 0.50  → reject, close at bar 5 → final_r = close_r_at_5
                                                    in new R-frame at SL=3.0

Gates checked (informational only; doesn't change HALT status):
  - Pass-deployable: worst-fold annualised ROI ≥ 5%, mean ≥ 8%, worst-fold
    DD ≤ 8%, all folds positive, ≥ 15 trades/fold, full-data ROI ≥ 5%,
    full-data DD ≤ 10%.
  - Pass-viable: worst-fold ROI > 0%, DD ≤ 8%, mean ≥ 3%, all folds positive,
    ≥ 5 trades/fold, full-data ROI ≥ 3%, full-data DD ≤ 10%.

Usage:
    py scripts/l_arc_11/experimental_s5_wfo.py
"""

from __future__ import annotations

import csv
import math
import sys
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
    PIPELINE_E_BASE,
    PIPELINE_D1_PATH_FEATURES,
    _build_pair_cache,
    compute_pipeline_e_features,
    _path_features_at_t,
)

RISK_PER_TRADE = 0.005       # 0.5% per L arc convention
SL_C1 = 3.0
SL_AGG = 3.0
ORIGINAL_SL = 2.0
T_FOR_B = 5
D1_ADMIT_THRESHOLD = 0.50
DATA_DIR_4H = "C:/Users/panap/Documents/Forex-Backtester/data/4hr"
EXP_S5_DIR = _REPO_ROOT / "results" / "l_arc_11" / "experimental_s5"

# 7 folds mirroring KH-24 OOS schedule, truncated to Arc 11's data window.
FOLDS = [
    (1, "2020-10-01", "2021-07-01"),
    (2, "2021-07-01", "2022-04-01"),
    (3, "2022-04-01", "2023-01-01"),
    (4, "2023-01-01", "2023-10-01"),
    (5, "2023-10-01", "2024-07-01"),
    (6, "2024-07-01", "2025-04-01"),
    (7, "2025-04-01", "2026-01-31"),
]

PASS_DEPLOYABLE_TRADE_FLOOR = 15
PASS_VIABLE_TRADE_FLOOR = 5


# ============================================================
# Per-trade SL=3 evaluation cache
# ============================================================

def build_paths_index(paths_df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    out: Dict[int, pd.DataFrame] = {}
    sorted_df = paths_df.sort_values(["trade_id", "bar_offset"], kind="mergesort")
    for tid, g in sorted_df.groupby("trade_id", sort=True):
        out[int(tid)] = g.reset_index(drop=True)
    return out


def compute_final_r_at_sl(
    paths_index: Dict[int, pd.DataFrame],
    trade_ids: List[int],
    sl_mult: float,
    original_sl: float = ORIGINAL_SL,
) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for tid in trade_ids:
        path = paths_index[tid]
        te = _eval_trade_at_sl(path, sl_mult, original_sl)
        out[tid] = float(te.final_r_new)
    return out


def compute_close_r_at_bar(
    paths_index: Dict[int, pd.DataFrame],
    trade_ids: List[int],
    bar_offset: int,
    sl_mult: float,
    original_sl: float = ORIGINAL_SL,
) -> Dict[int, Optional[float]]:
    """Return close_r at the given bar_offset, in the new R-frame at sl_mult,
    OR None if the trade SL'd before reaching that bar."""
    scale = original_sl / sl_mult
    out: Dict[int, Optional[float]] = {}
    for tid in trade_ids:
        path = paths_index[tid]
        te = _eval_trade_at_sl(path, sl_mult, original_sl)
        if te.truncated_at_bar < bar_offset:
            out[tid] = None
            continue
        # Get close_r at the requested bar.
        row = path[path["bar_offset"] == bar_offset]
        if row.empty:
            out[tid] = None
            continue
        close_orig = float(row["close_r"].iloc[0])
        out[tid] = close_orig * scale
    return out


# ============================================================
# Fold metrics
# ============================================================

@dataclass
class FoldMetrics:
    fold_id: int
    oos_start: str
    oos_end: str
    days: int
    n_trades: int
    n_admitted: int           # for Run B; equals n_trades for Run A
    n_rejected: int
    n_pre_t_losses: int       # for Run B; 0 for Run A
    mean_r: float
    sum_r: float
    roi_period: float         # equity end / start - 1
    roi_annualised: float
    max_dd: float
    final_r_mean_admit_only: float
    classifier_trained: bool  # for Run B
    n_is_trades: int          # for Run B classifier (IS for path-so-far at t=5)
    notes: str = ""


def _compounded_equity(returns: List[float], risk: float = RISK_PER_TRADE) -> Tuple[List[float], float]:
    eq = 1.0
    curve = [1.0]
    for r in returns:
        eq *= (1.0 + r * risk)
        curve.append(eq)
    return curve, eq - 1.0


def _max_dd_from_curve(curve: List[float]) -> float:
    if not curve:
        return 0.0
    arr = np.array(curve, dtype=float)
    peak = np.maximum.accumulate(arr)
    dd = (peak - arr) / peak
    return float(dd.max())


def _annualise(roi: float, days: int) -> float:
    if days <= 0:
        return 0.0
    eq = 1.0 + roi
    if eq <= 0:
        return -1.0
    return float(eq ** (365.25 / days) - 1.0)


# ============================================================
# Run A — c1 raw
# ============================================================

def run_a_c1_raw(
    trades_df: pd.DataFrame,
    paths_index: Dict[int, pd.DataFrame],
    cluster_assignments: pd.DataFrame,
) -> List[FoldMetrics]:
    c1_tids = sorted(cluster_assignments[cluster_assignments["cluster_id"] == 1]["trade_id"].astype(int).tolist())
    final_r_map = compute_final_r_at_sl(paths_index, c1_tids, SL_C1)
    sub = trades_df[trades_df["trade_id"].isin(c1_tids)].copy()
    sub["final_r_sl3"] = sub["trade_id"].map(final_r_map)
    sub["entry_time"] = pd.to_datetime(sub["entry_time"])
    sub = sub.sort_values("entry_time").reset_index(drop=True)

    fold_metrics: List[FoldMetrics] = []
    for fold_id, oos_s, oos_e in FOLDS:
        oos_start = pd.Timestamp(oos_s)
        oos_end = pd.Timestamp(oos_e)
        days = (oos_end - oos_start).days
        oos = sub[(sub["entry_time"] >= oos_start) & (sub["entry_time"] < oos_end)].copy()
        n = len(oos)
        if n == 0:
            fold_metrics.append(FoldMetrics(
                fold_id=fold_id, oos_start=oos_s, oos_end=oos_e, days=days,
                n_trades=0, n_admitted=0, n_rejected=0, n_pre_t_losses=0,
                mean_r=0.0, sum_r=0.0, roi_period=0.0, roi_annualised=0.0,
                max_dd=0.0, final_r_mean_admit_only=0.0,
                classifier_trained=False, n_is_trades=0,
                notes="no OOS trades in window",
            ))
            continue
        rs = oos["final_r_sl3"].tolist()
        curve, roi = _compounded_equity(rs)
        max_dd = _max_dd_from_curve(curve)
        mean_r = float(oos["final_r_sl3"].mean())
        sum_r = float(oos["final_r_sl3"].sum())
        fold_metrics.append(FoldMetrics(
            fold_id=fold_id, oos_start=oos_s, oos_end=oos_e, days=days,
            n_trades=n, n_admitted=n, n_rejected=0, n_pre_t_losses=0,
            mean_r=mean_r, sum_r=sum_r,
            roi_period=roi, roi_annualised=_annualise(roi, days),
            max_dd=max_dd, final_r_mean_admit_only=mean_r,
            classifier_trained=False, n_is_trades=0,
        ))
    return fold_metrics


# ============================================================
# Run B — agg_c1_c3 + D1 t=5 classifier
# ============================================================

def _build_agg_d1_feature_dataset(
    trade_ids: List[int],
    trades_df: pd.DataFrame,
    paths_index: Dict[int, pd.DataFrame],
    e_base_features: pd.DataFrame,
    sl_mult: float,
    t: int,
) -> Tuple[pd.DataFrame, Dict[int, int]]:
    """Build the D1-at-t feature dataset (8 base + 7 path-so-far) + success
    labels for trades that survived to bar t. Trades that SL'd before t are
    omitted from the feature set but their pre-t loss is handled in the
    per-trade outcome routine elsewhere.
    """
    rows: List[Dict[str, Any]] = []
    success_labels: Dict[int, int] = {}
    for tid in trade_ids:
        pf = _path_features_at_t(paths_index[tid], t, sl_mult, ORIGINAL_SL)
        if pf is None:
            continue
        base_row_q = e_base_features[e_base_features["trade_id"] == tid]
        if base_row_q.empty:
            continue
        base_row = base_row_q.iloc[0]
        merged = {"trade_id": int(tid), "entry_time": pd.Timestamp(base_row["entry_time"])}
        for c in PIPELINE_E_BASE:
            merged[c] = base_row[c]
        merged.update(pf)
        rows.append(merged)
        te = _eval_trade_at_sl(paths_index[tid], sl_mult, ORIGINAL_SL)
        success_labels[int(tid)] = 1 if te.final_r_new >= 1.0 else 0
    return pd.DataFrame(rows), success_labels


def _train_rf(
    X: pd.DataFrame, y: np.ndarray, model_kw: dict, class_weight_used: str
) -> Any:
    from sklearn.ensemble import RandomForestClassifier
    kw = dict(model_kw)
    if class_weight_used == "balanced":
        kw["class_weight"] = "balanced"
    clf = RandomForestClassifier(**kw)
    med = X.median(numeric_only=True)
    X_f = X.fillna(med)
    clf.fit(X_f, y)
    return clf, med


def run_b_agg_d1_t5(
    trades_df: pd.DataFrame,
    paths_index: Dict[int, pd.DataFrame],
    cluster_assignments: pd.DataFrame,
    e_base_features: pd.DataFrame,
) -> List[FoldMetrics]:
    agg_tids = sorted(
        cluster_assignments[cluster_assignments["cluster_id"].isin([1, 3])]["trade_id"].astype(int).tolist()
    )
    # Pre-compute final_r at SL=3.0 (used for admit outcomes + IS labels).
    final_r_map = compute_final_r_at_sl(paths_index, agg_tids, SL_AGG)
    # Pre-compute close_r at bar 5 in new R-frame (used for reject outcomes).
    close_at_5_map = compute_close_r_at_bar(paths_index, agg_tids, T_FOR_B, SL_AGG)

    # Build full feature dataset for all trades that survived to t=5.
    full_feat_df, _success_full = _build_agg_d1_feature_dataset(
        agg_tids, trades_df, paths_index, e_base_features, SL_AGG, T_FOR_B
    )
    if not full_feat_df.empty:
        full_feat_df["entry_time"] = pd.to_datetime(full_feat_df["entry_time"])

    # Map trade_id → entry_time, bars_held for fold partitioning.
    tr_lite = trades_df[["trade_id", "entry_time", "bars_held"]].copy()
    tr_lite["entry_time"] = pd.to_datetime(tr_lite["entry_time"])
    tr_idx = tr_lite.set_index("trade_id")

    feature_cols = list(PIPELINE_E_BASE) + list(PIPELINE_D1_PATH_FEATURES)
    model_kw = dict(n_estimators=200, max_depth=8, random_state=42, n_jobs=1)

    fold_metrics: List[FoldMetrics] = []
    for fold_id, oos_s, oos_e in FOLDS:
        oos_start = pd.Timestamp(oos_s)
        oos_end = pd.Timestamp(oos_e)
        days = (oos_end - oos_start).days

        # IS: trades with entry_time < oos_start and bars_held >= 5 (eligible for D1 evaluation).
        is_mask_q = (full_feat_df["entry_time"] < oos_start)
        is_feat = full_feat_df[is_mask_q].copy()
        is_tids = is_feat["trade_id"].astype(int).tolist()
        is_labels_map = {tid: (1 if final_r_map[tid] >= 1.0 else 0) for tid in is_tids}

        # OOS: trades with oos_start <= entry_time < oos_end (irrespective of bars_held).
        oos_tids = tr_lite[(tr_lite["entry_time"] >= oos_start) & (tr_lite["entry_time"] < oos_end)
                           & (tr_lite["trade_id"].isin(agg_tids))]["trade_id"].astype(int).tolist()

        n_total = len(oos_tids)
        if n_total == 0:
            fold_metrics.append(FoldMetrics(
                fold_id=fold_id, oos_start=oos_s, oos_end=oos_e, days=days,
                n_trades=0, n_admitted=0, n_rejected=0, n_pre_t_losses=0,
                mean_r=0.0, sum_r=0.0, roi_period=0.0, roi_annualised=0.0,
                max_dd=0.0, final_r_mean_admit_only=0.0,
                classifier_trained=False, n_is_trades=len(is_tids),
                notes="no OOS trades in window",
            ))
            continue

        # Train classifier on IS (if enough trades).
        classifier_trained = False
        clf = None
        med = None
        if len(is_tids) >= 50:
            y_is = np.array([is_labels_map[t] for t in is_tids], dtype=int)
            if len(np.unique(y_is)) >= 2:
                base = float(y_is.mean())
                minority = min(base, 1.0 - base)
                cw = "balanced" if minority < 0.30 else "none"
                clf, med = _train_rf(is_feat[feature_cols], y_is, model_kw, cw)
                classifier_trained = True

        # Build OOS feature subset (for trades that survived to t=5).
        oos_feat = full_feat_df[full_feat_df["trade_id"].isin(oos_tids)].copy()

        outcomes: List[Tuple[int, float, str]] = []   # (trade_id, final_r, reason)

        # Predict for OOS trades that survived to t=5.
        survived_oos_tids = oos_feat["trade_id"].astype(int).tolist()
        admit_decisions: Dict[int, bool] = {}
        if classifier_trained:
            X_oos = oos_feat[feature_cols].fillna(med)
            probs = clf.predict_proba(X_oos)[:, 1]
            for tid, p in zip(survived_oos_tids, probs):
                admit_decisions[int(tid)] = bool(p >= D1_ADMIT_THRESHOLD)
        else:
            # No classifier — admit all survivors (fall back to raw-equivalent for D1 survivors).
            for tid in survived_oos_tids:
                admit_decisions[int(tid)] = True

        # Walk OOS trades in chronological order.
        oos_meta = tr_lite[tr_lite["trade_id"].isin(oos_tids)].sort_values("entry_time")
        n_admit = 0
        n_reject = 0
        n_pre_t = 0
        for _, row in oos_meta.iterrows():
            tid = int(row["trade_id"])
            bh = int(row["bars_held"])
            if bh < T_FOR_B:
                # Pre-t SL loss.
                outcomes.append((tid, -1.0, "pre_t_sl"))
                n_pre_t += 1
                continue
            if admit_decisions.get(tid, True):
                fr = float(final_r_map[tid])
                outcomes.append((tid, fr, "admit"))
                n_admit += 1
            else:
                cv = close_at_5_map.get(tid)
                # If we got here, the trade survived to bar 5 (bars_held >= 5).
                # close_at_5_map should have a non-None value; fall back to 0 if not.
                fr = float(cv) if cv is not None else 0.0
                outcomes.append((tid, fr, "reject_t5"))
                n_reject += 1

        returns = [o[1] for o in outcomes]
        admit_returns = [o[1] for o in outcomes if o[2] == "admit"]
        curve, roi = _compounded_equity(returns)
        max_dd = _max_dd_from_curve(curve)
        mean_r = float(np.mean(returns)) if returns else 0.0
        sum_r = float(np.sum(returns)) if returns else 0.0
        admit_mean = float(np.mean(admit_returns)) if admit_returns else 0.0

        fold_metrics.append(FoldMetrics(
            fold_id=fold_id, oos_start=oos_s, oos_end=oos_e, days=days,
            n_trades=n_total, n_admitted=n_admit, n_rejected=n_reject,
            n_pre_t_losses=n_pre_t,
            mean_r=mean_r, sum_r=sum_r,
            roi_period=roi, roi_annualised=_annualise(roi, days),
            max_dd=max_dd, final_r_mean_admit_only=admit_mean,
            classifier_trained=classifier_trained,
            n_is_trades=len(is_tids),
            notes="" if classifier_trained else "classifier not trained (IS < 50); admitted-all fallback",
        ))
    return fold_metrics


# ============================================================
# Aggregate metrics + gates
# ============================================================

@dataclass
class RunAggregates:
    run_name: str
    fold_metrics: List[FoldMetrics]
    sign_consistency: bool
    worst_fold_roi_pct: float          # annualised
    mean_fold_roi_pct: float           # annualised
    worst_fold_dd_pct: float
    min_trade_count: int
    full_data_roi_pct: float           # period roi, not annualised
    full_data_annualised_roi_pct: float
    full_data_max_dd_pct: float
    full_data_total_days: int
    pass_deployable: bool
    pass_viable: bool
    notes: List[str] = field(default_factory=list)


def aggregate_run(run_name: str, fm: List[FoldMetrics], full_returns: List[float], full_days: int) -> RunAggregates:
    rois_ann = [f.roi_annualised for f in fm if f.n_trades > 0]
    rois_period = [f.roi_period for f in fm if f.n_trades > 0]
    dds = [f.max_dd for f in fm if f.n_trades > 0]
    counts = [f.n_trades for f in fm if f.n_trades > 0]

    sign_consistency = all(r > 0 for r in rois_period) if rois_period else False
    worst_roi_ann = float(min(rois_ann)) if rois_ann else 0.0
    mean_roi_ann = float(np.mean(rois_ann)) if rois_ann else 0.0
    worst_dd = float(max(dds)) if dds else 0.0
    min_count = int(min(counts)) if counts else 0

    full_curve, full_roi = _compounded_equity(full_returns)
    full_dd = _max_dd_from_curve(full_curve)
    full_ann = _annualise(full_roi, full_days)

    notes: List[str] = []
    pass_deployable = (
        sign_consistency
        and worst_roi_ann >= 0.05
        and mean_roi_ann >= 0.08
        and worst_dd <= 0.08
        and min_count >= PASS_DEPLOYABLE_TRADE_FLOOR
        and full_roi >= 0.05
        and full_dd <= 0.10
    )
    if not pass_deployable:
        if not sign_consistency:
            notes.append("sign_consistency: at least one fold non-positive")
        if worst_roi_ann < 0.05:
            notes.append(f"worst-fold annualised ROI {worst_roi_ann:.4f} < 0.05")
        if mean_roi_ann < 0.08:
            notes.append(f"mean-fold annualised ROI {mean_roi_ann:.4f} < 0.08")
        if worst_dd > 0.08:
            notes.append(f"worst-fold DD {worst_dd:.4f} > 0.08")
        if min_count < PASS_DEPLOYABLE_TRADE_FLOOR:
            notes.append(f"min trade count {min_count} < {PASS_DEPLOYABLE_TRADE_FLOOR}")
        if full_roi < 0.05:
            notes.append(f"full-data ROI {full_roi:.4f} < 0.05")
        if full_dd > 0.10:
            notes.append(f"full-data DD {full_dd:.4f} > 0.10")

    pass_viable = (
        sign_consistency
        and worst_roi_ann > 0.0
        and mean_roi_ann >= 0.03
        and worst_dd <= 0.08
        and min_count >= PASS_VIABLE_TRADE_FLOOR
        and full_roi >= 0.03
        and full_dd <= 0.10
    )

    return RunAggregates(
        run_name=run_name, fold_metrics=fm,
        sign_consistency=sign_consistency,
        worst_fold_roi_pct=worst_roi_ann * 100,
        mean_fold_roi_pct=mean_roi_ann * 100,
        worst_fold_dd_pct=worst_dd * 100,
        min_trade_count=min_count,
        full_data_roi_pct=full_roi * 100,
        full_data_annualised_roi_pct=full_ann * 100,
        full_data_max_dd_pct=full_dd * 100,
        full_data_total_days=full_days,
        pass_deployable=pass_deployable,
        pass_viable=pass_viable,
        notes=notes,
    )


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


def write_per_fold_csv(out_path: Path, run_name: str, fm: List[FoldMetrics]) -> None:
    cols = ["run", "fold_id", "oos_start", "oos_end", "days", "n_trades",
            "n_admitted", "n_rejected", "n_pre_t_losses", "mean_r", "sum_r",
            "roi_period_pct", "roi_annualised_pct", "max_dd_pct",
            "admit_only_final_r_mean", "classifier_trained", "n_is_trades", "notes"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        for fold in fm:
            w.writerow([
                run_name, fold.fold_id, fold.oos_start, fold.oos_end, fold.days,
                fold.n_trades, fold.n_admitted, fold.n_rejected, fold.n_pre_t_losses,
                _fmt(fold.mean_r), _fmt(fold.sum_r),
                _fmt(fold.roi_period * 100), _fmt(fold.roi_annualised * 100),
                _fmt(fold.max_dd * 100), _fmt(fold.final_r_mean_admit_only),
                "1" if fold.classifier_trained else "0",
                fold.n_is_trades,
                fold.notes,
            ])


def write_aggregate_csv(out_path: Path, runs: List[RunAggregates]) -> None:
    cols = ["run", "sign_consistency", "worst_fold_roi_ann_pct", "mean_fold_roi_ann_pct",
            "worst_fold_dd_pct", "min_trade_count", "full_data_roi_pct",
            "full_data_ann_roi_pct", "full_data_dd_pct", "full_data_days",
            "pass_deployable", "pass_viable", "fail_notes"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        for r in runs:
            w.writerow([
                r.run_name, "1" if r.sign_consistency else "0",
                _fmt(r.worst_fold_roi_pct, 4), _fmt(r.mean_fold_roi_pct, 4),
                _fmt(r.worst_fold_dd_pct, 4), r.min_trade_count,
                _fmt(r.full_data_roi_pct, 4), _fmt(r.full_data_annualised_roi_pct, 4),
                _fmt(r.full_data_max_dd_pct, 4), r.full_data_total_days,
                "1" if r.pass_deployable else "0", "1" if r.pass_viable else "0",
                "; ".join(r.notes),
            ])


# ============================================================
# Driver
# ============================================================

def main() -> int:
    EXP_S5_DIR.mkdir(parents=True, exist_ok=True)

    print("[exp_s5] loading Step 1 / Step 2 artefacts", file=sys.stderr)
    trades_df = pd.read_csv(_REPO_ROOT / "results/l_arc_11/step1_verbatim/trades_all.csv")
    paths_df = pd.read_csv(_REPO_ROOT / "results/l_arc_11/step1_verbatim/trades_paths.csv")
    clusters_df = pd.read_csv(_REPO_ROOT / "results/l_arc_11/step2/clusters_K4.csv")

    print("[exp_s5] building paths index", file=sys.stderr)
    paths_index = build_paths_index(paths_df)

    print("[exp_s5] caching 4H per-pair indicators for Pipeline E base features", file=sys.stderr)
    pairs = sorted(trades_df["pair"].astype(str).unique())
    pair_caches = {p: _build_pair_cache(p, DATA_DIR_4H) for p in pairs}

    print("[exp_s5] computing Pipeline E base + arc-11 features (Step 4 spec)", file=sys.stderr)
    e_base_features = compute_pipeline_e_features(trades_df, pair_caches)

    print("[exp_s5] === Run A: c1 raw at SL=3.0 ===", file=sys.stderr)
    a_folds = run_a_c1_raw(trades_df, paths_index, clusters_df)
    write_per_fold_csv(EXP_S5_DIR / "arc11_exp_s5_run_a_c1_raw_per_fold.csv", "A_c1_raw_sl3", a_folds)

    # Full-data sequence for Run A.
    c1_tids = sorted(clusters_df[clusters_df["cluster_id"] == 1]["trade_id"].astype(int).tolist())
    final_r_c1 = compute_final_r_at_sl(paths_index, c1_tids, SL_C1)
    c1_meta = trades_df[trades_df["trade_id"].isin(c1_tids)].copy()
    c1_meta["entry_time"] = pd.to_datetime(c1_meta["entry_time"])
    c1_meta = c1_meta.sort_values("entry_time")
    full_returns_a = [final_r_c1[int(t)] for t in c1_meta["trade_id"]]
    full_days_a = (pd.Timestamp(FOLDS[-1][2]) - pd.Timestamp(FOLDS[0][1])).days
    agg_a = aggregate_run("A_c1_raw_sl3", a_folds, full_returns_a, full_days_a)

    print("[exp_s5] === Run B: agg_c1_c3 + D1 t=5 classifier at SL=3.0 ===", file=sys.stderr)
    b_folds = run_b_agg_d1_t5(trades_df, paths_index, clusters_df, e_base_features)
    write_per_fold_csv(EXP_S5_DIR / "arc11_exp_s5_run_b_agg_d1_t5_per_fold.csv", "B_agg_c1c3_d1_t5_sl3", b_folds)

    # Full-data sequence for Run B (sum across folds).
    full_returns_b: List[float] = []
    for fold in b_folds:
        # Reconstruct returns from the metrics? We didn't keep them. Re-derive:
        # easiest is to re-walk OOS trades per fold here. We approximate full-data
        # ROI from per-fold sum_r preserving chronology by ordering folds.
        # For DD purposes we re-run quickly:
        pass

    # Cleaner: rebuild full-period returns by re-running outcome computation in one chronological pass
    # without folds, using the full-trained classifier on cumulative data.
    # Simpler approximation acceptable for this experimental run: chain per-fold returns.
    # Reconstruct per-fold returns sequence (need to actually re-walk, simplest = run the per-fold
    # logic and capture returns).
    agg_tids = sorted(clusters_df[clusters_df["cluster_id"].isin([1, 3])]["trade_id"].astype(int).tolist())
    final_r_agg = compute_final_r_at_sl(paths_index, agg_tids, SL_AGG)
    close_at_5_agg = compute_close_r_at_bar(paths_index, agg_tids, T_FOR_B, SL_AGG)
    tr_lite_agg = trades_df[trades_df["trade_id"].isin(agg_tids)][["trade_id", "entry_time", "bars_held"]].copy()
    tr_lite_agg["entry_time"] = pd.to_datetime(tr_lite_agg["entry_time"])

    # Build features once for all agg trades.
    full_feat_df, _ = _build_agg_d1_feature_dataset(
        agg_tids, trades_df, paths_index, e_base_features, SL_AGG, T_FOR_B
    )
    if not full_feat_df.empty:
        full_feat_df["entry_time"] = pd.to_datetime(full_feat_df["entry_time"])
    feature_cols = list(PIPELINE_E_BASE) + list(PIPELINE_D1_PATH_FEATURES)
    model_kw = dict(n_estimators=200, max_depth=8, random_state=42, n_jobs=1)

    full_returns_b = []
    for fold_id, oos_s, oos_e in FOLDS:
        oos_start = pd.Timestamp(oos_s)
        oos_end = pd.Timestamp(oos_e)
        is_feat_q = full_feat_df[full_feat_df["entry_time"] < oos_start].copy()
        is_tids_q = is_feat_q["trade_id"].astype(int).tolist()
        oos_meta_q = tr_lite_agg[(tr_lite_agg["entry_time"] >= oos_start) & (tr_lite_agg["entry_time"] < oos_end)].sort_values("entry_time")

        clf_q = None
        med_q = None
        if len(is_tids_q) >= 50:
            y_is = np.array([1 if final_r_agg[t] >= 1.0 else 0 for t in is_tids_q], dtype=int)
            if len(np.unique(y_is)) >= 2:
                base = float(y_is.mean())
                minority = min(base, 1.0 - base)
                cw = "balanced" if minority < 0.30 else "none"
                clf_q, med_q = _train_rf(is_feat_q[feature_cols], y_is, model_kw, cw)

        oos_feat_q = full_feat_df[full_feat_df["entry_time"].between(oos_start, oos_end, inclusive="left")].copy()
        admit_q: Dict[int, bool] = {}
        if clf_q is not None and not oos_feat_q.empty:
            X_q = oos_feat_q[feature_cols].fillna(med_q)
            probs_q = clf_q.predict_proba(X_q)[:, 1]
            for tid, p in zip(oos_feat_q["trade_id"].astype(int).tolist(), probs_q):
                admit_q[int(tid)] = bool(p >= D1_ADMIT_THRESHOLD)
        else:
            for tid in oos_feat_q["trade_id"].astype(int).tolist():
                admit_q[int(tid)] = True

        for _, row in oos_meta_q.iterrows():
            tid = int(row["trade_id"])
            bh = int(row["bars_held"])
            if bh < T_FOR_B:
                full_returns_b.append(-1.0)
                continue
            if admit_q.get(tid, True):
                full_returns_b.append(float(final_r_agg[tid]))
            else:
                cv = close_at_5_agg.get(tid)
                full_returns_b.append(float(cv) if cv is not None else 0.0)

    full_days_b = full_days_a
    agg_b = aggregate_run("B_agg_c1c3_d1_t5_sl3", b_folds, full_returns_b, full_days_b)

    write_aggregate_csv(EXP_S5_DIR / "arc11_exp_s5_aggregate.csv", [agg_a, agg_b])

    # Compact stdout summary table.
    print()
    print("=" * 88)
    print("EXPERIMENTAL S5 — Arc 11 (off-protocol; HALT status unchanged)")
    print("=" * 88)
    for r in (agg_a, agg_b):
        print()
        print(f"### {r.run_name}")
        print(f"  sign_consistency:         {r.sign_consistency}")
        print(f"  worst-fold ROI ann %:     {r.worst_fold_roi_pct:.4f}")
        print(f"  mean-fold ROI ann %:      {r.mean_fold_roi_pct:.4f}")
        print(f"  worst-fold DD %:          {r.worst_fold_dd_pct:.4f}")
        print(f"  min trade count:          {r.min_trade_count}")
        print(f"  full-data ROI %:          {r.full_data_roi_pct:.4f}")
        print(f"  full-data ann ROI %:      {r.full_data_annualised_roi_pct:.4f}")
        print(f"  full-data DD %:           {r.full_data_max_dd_pct:.4f}")
        print(f"  pass_deployable:          {r.pass_deployable}")
        print(f"  pass_viable:              {r.pass_viable}")
        if r.notes:
            print(f"  fail_notes:               {'; '.join(r.notes)}")
    print()
    print("=" * 88)
    print("Per-fold detail in:")
    print("  results/l_arc_11/experimental_s5/arc11_exp_s5_run_a_c1_raw_per_fold.csv")
    print("  results/l_arc_11/experimental_s5/arc11_exp_s5_run_b_agg_d1_t5_per_fold.csv")
    print("  results/l_arc_11/experimental_s5/arc11_exp_s5_aggregate.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
