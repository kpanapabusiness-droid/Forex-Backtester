"""Arc 11 — Experimental Step 5 WFO, NO-ORACLE (off-protocol, documentation only).

Removes the cluster-ID-at-entry oracle that was baked into the prior
arc11_exp_s5 runs A and B. Both runs here use the full 2,299-trade SHB
signal pool; admission is decided per-fold by a live-trained Pipeline E
classifier (Run C) or an E→D1 cascade (Run D), using only information
available at decision time.

NOT a canonical S5 run. Arc 11 stays Closed-HALT per §16a Path A regardless
of this script's outcome. No mutation to queue, ARC_11_LIVE.md, registry,
or protocol state.

Runs:
  Run C — Pipeline E only (live-deployable analogue of prior Run A)
    Per fold, train RF on IS trades with binary label `cluster_id == 1`
    using PIPELINE_E_FEATURES (S4 c1 E-classifier config). Admit if
    P(c1) >= threshold (sweep {0.30, 0.40, 0.50, 0.60, 0.70}; report
    threshold 0.50 baseline + the best operating point by §10 worst-fold
    ROI ann subject to DD <= 8%). Admitted trades use final_r at SL=3.0.
    Rejected trades = no entry (no P&L impact).

  Run D — Pipeline E → D1 cascade (live-deployable analogue of prior Run B)
    Stage 1 (E): per fold, train RF to predict P(cluster_id in {1, 3})
    using PIPELINE_E_FEATURES. Admit if P >= 0.50.
    Stage 2 (D1): for E-admitted OOS trades with bars_held >= 5, train
    RF on IS E-admitted survivors using PIPELINE_E_BASE + path-so-far at
    t=5 (SL=3.0 R-frame), label = final_r >= 1.0. Admit if P >= 0.50.
    Outcomes:
      bars_held < 5             -> pre-t SL loss, final_r = -1.0
      P(success) >= 0.50        -> admit, use final_r at SL=3.0
      P(success) <  0.50        -> reject at bar 5, final_r = close_r_at_5
    E-rejected trades = no entry (no P&L impact).

Same fold structure + simplifications as the prior runs:
  - 7 folds (KH-24 OOS schedule).
  - Fixed-SL + time-exit truncation; NO §11 V-shape recovery trail.
  - Risk 0.5%/trade, compounded equity per fold.
  - Trade outcome from _eval_trade_at_sl (byte-identical with S3 / S4).

Outputs (all tagged arc11_exp_s5_noOracle_*):
  arc11_exp_s5_noOracle_C_threshold_sweep.csv   - Run C metrics per threshold
  arc11_exp_s5_noOracle_C_per_fold.csv          - Run C per-fold detail (best + 0.50)
  arc11_exp_s5_noOracle_D_per_fold.csv          - Run D per-fold detail
  arc11_exp_s5_noOracle_D_stage_attrition.csv   - Run D n_entered/n_admit_E/n_admit_D1
  arc11_exp_s5_noOracle_NOTES.md                - human commentary
  comparison_table.csv                          - A, B, C, D in one table

Usage:
    py scripts/l_arc_11/experimental_s5_no_oracle_wfo.py
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

from scripts.l_arc_11.experimental_s5_wfo import (  # noqa: E402
    FOLDS, RISK_PER_TRADE, ORIGINAL_SL, T_FOR_B, EXP_S5_DIR,
    PASS_DEPLOYABLE_TRADE_FLOOR, PASS_VIABLE_TRADE_FLOOR,
    FoldMetrics, RunAggregates,
    build_paths_index, compute_final_r_at_sl, compute_close_r_at_bar,
    aggregate_run, _compounded_equity, _max_dd_from_curve, _annualise,
)
from scripts.l_arc_11.step4_extractability import (  # noqa: E402
    PIPELINE_E_FEATURES, PIPELINE_E_BASE, PIPELINE_D1_PATH_FEATURES,
    _build_pair_cache, compute_pipeline_e_features, _path_features_at_t,
)

SL_DEPLOY = 3.0
DATA_DIR_4H = "C:/Users/panap/Documents/Forex-Backtester/data/4hr"

THRESHOLD_SWEEP_C = [0.30, 0.40, 0.50, 0.60, 0.70]
THRESHOLD_E_RUN_D = 0.50
THRESHOLD_D1_RUN_D = 0.50

NO_ORACLE_DIR = EXP_S5_DIR  # share output dir with prior runs; file names differ


# ============================================================
# Helpers
# ============================================================

def _train_rf(X: pd.DataFrame, y: np.ndarray, model_kw: dict, class_weight_used: str
              ) -> Tuple[Any, pd.Series]:
    from sklearn.ensemble import RandomForestClassifier
    kw = dict(model_kw)
    if class_weight_used == "balanced":
        kw["class_weight"] = "balanced"
    clf = RandomForestClassifier(**kw)
    med = X.median(numeric_only=True)
    X_f = X.fillna(med)
    clf.fit(X_f, y)
    return clf, med


def _decide_class_weight(y: np.ndarray, threshold: float = 0.30) -> str:
    if y.size == 0:
        return "none"
    base = float(y.mean())
    minority = min(base, 1.0 - base)
    return "balanced" if minority < threshold else "none"


def _predict_proba(clf: Any, X: pd.DataFrame, med: pd.Series) -> np.ndarray:
    X_f = X.fillna(med)
    return clf.predict_proba(X_f)[:, 1]


# ============================================================
# Run C — Pipeline E only (predict c1 membership)
# ============================================================

@dataclass
class RunCFold:
    fold_id: int
    oos_start: str
    oos_end: str
    days: int
    n_universe: int           # total SHB trades in OOS window
    n_admit: int              # admitted by E
    admit_rate: float
    n_admit_true_c1: int      # of admitted, how many actually were c1
    precision_on_c1: float    # n_admit_true_c1 / n_admit
    mean_r_admit: float
    sum_r_admit: float
    roi_period: float
    roi_annualised: float
    max_dd: float
    classifier_trained: bool
    n_is_trades: int
    notes: str = ""


def _run_c_at_threshold(
    threshold: float,
    universe_trades: pd.DataFrame,            # has trade_id, pair, entry_time, cluster_id, final_r_sl3
    e_features: pd.DataFrame,                  # all trades
    model_kw: dict,
) -> Tuple[List[RunCFold], List[float]]:
    """Return (per-fold metrics, full-period returns chronologically)."""
    fold_metrics: List[RunCFold] = []
    full_returns: List[float] = []

    e_features = e_features.copy()
    e_features["entry_time"] = pd.to_datetime(e_features["entry_time"])
    universe_trades = universe_trades.copy()
    universe_trades["entry_time"] = pd.to_datetime(universe_trades["entry_time"])

    # Merge labels (cluster_id) + final_r_sl3 onto features for IS training convenience.
    feat_with_label = e_features.merge(
        universe_trades[["trade_id", "cluster_id", "final_r_sl3"]],
        on="trade_id", how="left",
    )

    for fold_id, oos_s, oos_e in FOLDS:
        oos_start = pd.Timestamp(oos_s)
        oos_end = pd.Timestamp(oos_e)
        days = (oos_end - oos_start).days

        is_mask = feat_with_label["entry_time"] < oos_start
        oos_mask = (feat_with_label["entry_time"] >= oos_start) & (feat_with_label["entry_time"] < oos_end)
        is_sub = feat_with_label[is_mask]
        oos_sub = feat_with_label[oos_mask].sort_values("entry_time")

        n_universe = len(oos_sub)
        if n_universe == 0:
            fold_metrics.append(RunCFold(
                fold_id=fold_id, oos_start=oos_s, oos_end=oos_e, days=days,
                n_universe=0, n_admit=0, admit_rate=0.0, n_admit_true_c1=0,
                precision_on_c1=0.0, mean_r_admit=0.0, sum_r_admit=0.0,
                roi_period=0.0, roi_annualised=0.0, max_dd=0.0,
                classifier_trained=False, n_is_trades=len(is_sub),
                notes="no OOS trades in window",
            ))
            continue

        # Train E classifier if enough IS.
        classifier_trained = False
        if len(is_sub) >= 50:
            y_is = (is_sub["cluster_id"].astype(int) == 1).to_numpy()
            if len(np.unique(y_is)) >= 2:
                cw = _decide_class_weight(y_is)
                clf, med = _train_rf(is_sub[PIPELINE_E_FEATURES], y_is, model_kw, cw)
                probs = _predict_proba(clf, oos_sub[PIPELINE_E_FEATURES], med)
                admit_mask = probs >= threshold
                classifier_trained = True
            else:
                admit_mask = np.zeros(len(oos_sub), dtype=bool)
        else:
            # No classifier: admit all (deployment "discovery mode").
            admit_mask = np.ones(len(oos_sub), dtype=bool)

        admitted = oos_sub[admit_mask]
        n_admit = int(admit_mask.sum())
        admit_rate = float(n_admit / max(n_universe, 1))
        n_admit_true_c1 = int((admitted["cluster_id"].astype(int) == 1).sum())
        prec = float(n_admit_true_c1 / max(n_admit, 1))

        admit_returns = admitted["final_r_sl3"].astype(float).tolist()
        curve, roi = _compounded_equity(admit_returns)
        max_dd = _max_dd_from_curve(curve)
        mean_r = float(admitted["final_r_sl3"].mean()) if n_admit > 0 else 0.0
        sum_r = float(admitted["final_r_sl3"].sum()) if n_admit > 0 else 0.0

        fold_metrics.append(RunCFold(
            fold_id=fold_id, oos_start=oos_s, oos_end=oos_e, days=days,
            n_universe=n_universe, n_admit=n_admit, admit_rate=admit_rate,
            n_admit_true_c1=n_admit_true_c1, precision_on_c1=prec,
            mean_r_admit=mean_r, sum_r_admit=sum_r,
            roi_period=roi, roi_annualised=_annualise(roi, days),
            max_dd=max_dd, classifier_trained=classifier_trained,
            n_is_trades=int(len(is_sub)),
            notes="" if classifier_trained else "no classifier (IS<50 or single class); admitted-all fallback",
        ))
        full_returns.extend(admit_returns)
    return fold_metrics, full_returns


def _runc_fold_to_foldmetrics(f: RunCFold) -> FoldMetrics:
    return FoldMetrics(
        fold_id=f.fold_id, oos_start=f.oos_start, oos_end=f.oos_end, days=f.days,
        n_trades=f.n_admit, n_admitted=f.n_admit, n_rejected=f.n_universe - f.n_admit,
        n_pre_t_losses=0,
        mean_r=f.mean_r_admit, sum_r=f.sum_r_admit,
        roi_period=f.roi_period, roi_annualised=f.roi_annualised,
        max_dd=f.max_dd, final_r_mean_admit_only=f.mean_r_admit,
        classifier_trained=f.classifier_trained, n_is_trades=f.n_is_trades,
        notes=f.notes,
    )


# ============================================================
# Run D — E → D1 cascade
# ============================================================

@dataclass
class RunDFold:
    fold_id: int
    oos_start: str
    oos_end: str
    days: int
    n_universe: int
    n_admit_e: int
    e_admit_rate: float
    n_admit_d1: int
    d1_admit_rate_within_e: float
    n_pre_t_losses: int
    n_e_admit_true_agg: int        # of E-admitted, how many actually in {c1, c3}
    e_precision_on_agg: float
    mean_r_overall: float           # across all E-admitted (pre-t + admit + reject)
    mean_r_admit_d1_only: float
    sum_r_overall: float
    roi_period: float
    roi_annualised: float
    max_dd: float
    e_classifier_trained: bool
    d1_classifier_trained: bool
    n_is_trades: int
    n_is_trades_d1_eligible: int
    notes: str = ""


def _run_d_full(
    universe_trades: pd.DataFrame,
    paths_index: Dict[int, pd.DataFrame],
    e_features: pd.DataFrame,
    model_kw: dict,
    t_e: float,
    t_d1: float,
) -> Tuple[List[RunDFold], List[float]]:
    fold_metrics: List[RunDFold] = []
    full_returns: List[float] = []

    e_features = e_features.copy()
    e_features["entry_time"] = pd.to_datetime(e_features["entry_time"])
    universe_trades = universe_trades.copy()
    universe_trades["entry_time"] = pd.to_datetime(universe_trades["entry_time"])

    # Pre-compute close_r at bar 5 in SL=3 R-frame for all trades.
    all_tids = universe_trades["trade_id"].astype(int).tolist()
    close_at_5_map = compute_close_r_at_bar(paths_index, all_tids, T_FOR_B, SL_DEPLOY)

    feat_with_label = e_features.merge(
        universe_trades[["trade_id", "cluster_id", "final_r_sl3", "bars_held"]],
        on="trade_id", how="left",
    )

    for fold_id, oos_s, oos_e in FOLDS:
        oos_start = pd.Timestamp(oos_s)
        oos_end = pd.Timestamp(oos_e)
        days = (oos_end - oos_start).days

        is_mask = feat_with_label["entry_time"] < oos_start
        oos_mask = (feat_with_label["entry_time"] >= oos_start) & (feat_with_label["entry_time"] < oos_end)
        is_sub = feat_with_label[is_mask].copy()
        oos_sub = feat_with_label[oos_mask].sort_values("entry_time").copy()

        n_universe = len(oos_sub)
        if n_universe == 0:
            fold_metrics.append(RunDFold(
                fold_id=fold_id, oos_start=oos_s, oos_end=oos_e, days=days,
                n_universe=0, n_admit_e=0, e_admit_rate=0.0,
                n_admit_d1=0, d1_admit_rate_within_e=0.0, n_pre_t_losses=0,
                n_e_admit_true_agg=0, e_precision_on_agg=0.0,
                mean_r_overall=0.0, mean_r_admit_d1_only=0.0, sum_r_overall=0.0,
                roi_period=0.0, roi_annualised=0.0, max_dd=0.0,
                e_classifier_trained=False, d1_classifier_trained=False,
                n_is_trades=len(is_sub), n_is_trades_d1_eligible=0,
                notes="no OOS trades in window",
            ))
            continue

        # ----- Stage 1: Pipeline E classifier (predict agg = c1 OR c3) -----
        e_trained = False
        if len(is_sub) >= 50:
            y_is_e = is_sub["cluster_id"].astype(int).isin([1, 3]).to_numpy()
            if len(np.unique(y_is_e)) >= 2:
                cw_e = _decide_class_weight(y_is_e)
                clf_e, med_e = _train_rf(is_sub[PIPELINE_E_FEATURES], y_is_e, model_kw, cw_e)
                probs_e = _predict_proba(clf_e, oos_sub[PIPELINE_E_FEATURES], med_e)
                e_admit_mask = probs_e >= t_e
                e_trained = True
            else:
                e_admit_mask = np.zeros(len(oos_sub), dtype=bool)
        else:
            # No classifier: admit all (discovery mode)
            e_admit_mask = np.ones(len(oos_sub), dtype=bool)

        e_admitted = oos_sub[e_admit_mask].copy()
        n_admit_e = int(e_admit_mask.sum())
        e_admit_rate = float(n_admit_e / max(n_universe, 1))
        n_e_admit_true_agg = int(e_admitted["cluster_id"].astype(int).isin([1, 3]).sum())
        e_precision = float(n_e_admit_true_agg / max(n_admit_e, 1))

        # ----- Stage 2: D1 classifier on E-admitted IS (path-so-far at bar 5) -----
        # Build IS dataset: trades that E would have admitted (same E classifier, predicted on IS)
        # AND survived to bar 5. Label = final_r_sl3 >= 1.0.
        if e_trained:
            # Re-predict on IS to identify deployment-distribution training set.
            probs_e_is = _predict_proba(clf_e, is_sub[PIPELINE_E_FEATURES], med_e)
            is_e_admit_mask = probs_e_is >= t_e
        else:
            is_e_admit_mask = np.ones(len(is_sub), dtype=bool)

        is_e_admitted = is_sub[is_e_admit_mask].copy()

        # Build D1 features for IS E-admitted trades that survived to bar 5.
        d1_feature_cols = list(PIPELINE_E_BASE) + list(PIPELINE_D1_PATH_FEATURES)
        d1_is_rows: List[Dict[str, Any]] = []
        d1_is_labels: List[int] = []
        for _, row in is_e_admitted.iterrows():
            tid = int(row["trade_id"])
            pf = _path_features_at_t(paths_index[tid], T_FOR_B, SL_DEPLOY, ORIGINAL_SL)
            if pf is None:
                continue
            base_vals = {c: row[c] for c in PIPELINE_E_BASE}
            merged = {"trade_id": tid, "entry_time": row["entry_time"]}
            merged.update(base_vals)
            merged.update(pf)
            d1_is_rows.append(merged)
            d1_is_labels.append(1 if float(row["final_r_sl3"]) >= 1.0 else 0)
        d1_is_df = pd.DataFrame(d1_is_rows)
        n_is_d1_eligible = len(d1_is_df)

        d1_trained = False
        clf_d1 = None
        med_d1 = None
        if n_is_d1_eligible >= 50:
            y_is_d1 = np.array(d1_is_labels, dtype=int)
            if len(np.unique(y_is_d1)) >= 2:
                cw_d1 = _decide_class_weight(y_is_d1)
                clf_d1, med_d1 = _train_rf(
                    d1_is_df[d1_feature_cols], y_is_d1, model_kw, cw_d1
                )
                d1_trained = True

        # Walk E-admitted OOS chronologically; apply pre-t SL / D1 decisions.
        e_admitted_sorted = e_admitted.sort_values("entry_time")
        outcomes: List[Tuple[int, float, str]] = []
        for _, row in e_admitted_sorted.iterrows():
            tid = int(row["trade_id"])
            bh = int(row["bars_held"])
            if bh < T_FOR_B:
                outcomes.append((tid, -1.0, "pre_t_sl"))
                continue
            # Build D1 features for this OOS trade.
            pf = _path_features_at_t(paths_index[tid], T_FOR_B, SL_DEPLOY, ORIGINAL_SL)
            if pf is None:
                # Shouldn't happen if bh >= T_FOR_B, but defensive.
                outcomes.append((tid, -1.0, "pre_t_sl_unexpected"))
                continue
            if d1_trained:
                base_vals = {c: row[c] for c in PIPELINE_E_BASE}
                feat_row = {**base_vals, **pf}
                X = pd.DataFrame([feat_row])[d1_feature_cols]
                p_succ = float(_predict_proba(clf_d1, X, med_d1)[0])
                if p_succ >= t_d1:
                    outcomes.append((tid, float(row["final_r_sl3"]), "d1_admit"))
                else:
                    cv = close_at_5_map.get(tid)
                    outcomes.append((tid, float(cv) if cv is not None else 0.0, "d1_reject"))
            else:
                # No D1 classifier: admit all survivors.
                outcomes.append((tid, float(row["final_r_sl3"]), "d1_admit_no_classifier"))

        n_admit_d1 = sum(1 for o in outcomes if "d1_admit" in o[2])
        n_reject_d1 = sum(1 for o in outcomes if o[2] == "d1_reject")
        n_pre_t = sum(1 for o in outcomes if o[2].startswith("pre_t_sl"))
        d1_admit_rate_within_e = float(n_admit_d1 / max(n_admit_e, 1))

        returns = [o[1] for o in outcomes]
        admit_returns = [o[1] for o in outcomes if "d1_admit" in o[2]]
        curve, roi = _compounded_equity(returns)
        max_dd = _max_dd_from_curve(curve)
        mean_r_overall = float(np.mean(returns)) if returns else 0.0
        mean_r_admit = float(np.mean(admit_returns)) if admit_returns else 0.0
        sum_r_overall = float(np.sum(returns)) if returns else 0.0

        fold_metrics.append(RunDFold(
            fold_id=fold_id, oos_start=oos_s, oos_end=oos_e, days=days,
            n_universe=n_universe, n_admit_e=n_admit_e, e_admit_rate=e_admit_rate,
            n_admit_d1=n_admit_d1, d1_admit_rate_within_e=d1_admit_rate_within_e,
            n_pre_t_losses=n_pre_t,
            n_e_admit_true_agg=n_e_admit_true_agg, e_precision_on_agg=e_precision,
            mean_r_overall=mean_r_overall, mean_r_admit_d1_only=mean_r_admit,
            sum_r_overall=sum_r_overall,
            roi_period=roi, roi_annualised=_annualise(roi, days),
            max_dd=max_dd,
            e_classifier_trained=e_trained, d1_classifier_trained=d1_trained,
            n_is_trades=int(len(is_sub)), n_is_trades_d1_eligible=int(n_is_d1_eligible),
        ))
        full_returns.extend(returns)
    return fold_metrics, full_returns


def _rund_fold_to_foldmetrics(f: RunDFold) -> FoldMetrics:
    return FoldMetrics(
        fold_id=f.fold_id, oos_start=f.oos_start, oos_end=f.oos_end, days=f.days,
        n_trades=f.n_admit_e, n_admitted=f.n_admit_d1,
        n_rejected=f.n_admit_e - f.n_admit_d1 - f.n_pre_t_losses,
        n_pre_t_losses=f.n_pre_t_losses,
        mean_r=f.mean_r_overall, sum_r=f.sum_r_overall,
        roi_period=f.roi_period, roi_annualised=f.roi_annualised,
        max_dd=f.max_dd, final_r_mean_admit_only=f.mean_r_admit_d1_only,
        classifier_trained=f.d1_classifier_trained,
        n_is_trades=f.n_is_trades,
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


def write_runc_threshold_sweep(out_path: Path, sweep: List[Tuple[float, RunAggregates]]) -> None:
    cols = ["threshold", "sign_consistency", "worst_fold_roi_ann_pct",
            "mean_fold_roi_ann_pct", "worst_fold_dd_pct", "min_trade_count",
            "full_data_roi_pct", "full_data_ann_roi_pct", "full_data_dd_pct",
            "pass_deployable", "pass_viable", "selected"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        # Pick "best" by max worst-fold ROI ann subject to DD <= 8%, among those
        # also pass-deployable on the other gates; fallback to max mean-fold ROI.
        eligible = [(t, agg) for t, agg in sweep if agg.worst_fold_dd_pct <= 8.0]
        if eligible:
            best_t = max(eligible, key=lambda x: x[1].worst_fold_roi_pct)[0]
        else:
            best_t = max(sweep, key=lambda x: x[1].mean_fold_roi_pct)[0]
        for t, agg in sweep:
            w.writerow([
                _fmt(t, 2), "1" if agg.sign_consistency else "0",
                _fmt(agg.worst_fold_roi_pct, 4), _fmt(agg.mean_fold_roi_pct, 4),
                _fmt(agg.worst_fold_dd_pct, 4), agg.min_trade_count,
                _fmt(agg.full_data_roi_pct, 4), _fmt(agg.full_data_annualised_roi_pct, 4),
                _fmt(agg.full_data_max_dd_pct, 4),
                "1" if agg.pass_deployable else "0",
                "1" if agg.pass_viable else "0",
                "1" if abs(t - best_t) < 1e-9 else "0",
            ])


def write_runc_per_fold(out_path: Path, label: str, folds: List[RunCFold]) -> None:
    cols = ["run", "fold_id", "oos_start", "oos_end", "days",
            "n_universe", "n_admit", "admit_rate", "n_admit_true_c1",
            "precision_on_c1", "mean_r_admit", "sum_r_admit",
            "roi_period_pct", "roi_annualised_pct", "max_dd_pct",
            "classifier_trained", "n_is_trades", "notes"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        for fd in folds:
            w.writerow([
                label, fd.fold_id, fd.oos_start, fd.oos_end, fd.days,
                fd.n_universe, fd.n_admit, _fmt(fd.admit_rate),
                fd.n_admit_true_c1, _fmt(fd.precision_on_c1),
                _fmt(fd.mean_r_admit), _fmt(fd.sum_r_admit),
                _fmt(fd.roi_period * 100), _fmt(fd.roi_annualised * 100),
                _fmt(fd.max_dd * 100),
                "1" if fd.classifier_trained else "0",
                fd.n_is_trades, fd.notes,
            ])


def write_rund_per_fold(out_path: Path, folds: List[RunDFold]) -> None:
    cols = ["run", "fold_id", "oos_start", "oos_end", "days",
            "n_universe", "n_admit_e", "e_admit_rate",
            "n_admit_d1", "d1_admit_rate_within_e", "n_pre_t_losses",
            "n_e_admit_true_agg", "e_precision_on_agg",
            "mean_r_overall", "mean_r_admit_d1_only", "sum_r_overall",
            "roi_period_pct", "roi_annualised_pct", "max_dd_pct",
            "e_classifier_trained", "d1_classifier_trained",
            "n_is_trades", "n_is_d1_eligible", "notes"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        for fd in folds:
            w.writerow([
                "D_e_cascade_d1_t5_sl3", fd.fold_id, fd.oos_start, fd.oos_end, fd.days,
                fd.n_universe, fd.n_admit_e, _fmt(fd.e_admit_rate),
                fd.n_admit_d1, _fmt(fd.d1_admit_rate_within_e), fd.n_pre_t_losses,
                fd.n_e_admit_true_agg, _fmt(fd.e_precision_on_agg),
                _fmt(fd.mean_r_overall), _fmt(fd.mean_r_admit_d1_only),
                _fmt(fd.sum_r_overall),
                _fmt(fd.roi_period * 100), _fmt(fd.roi_annualised * 100),
                _fmt(fd.max_dd * 100),
                "1" if fd.e_classifier_trained else "0",
                "1" if fd.d1_classifier_trained else "0",
                fd.n_is_trades, fd.n_is_trades_d1_eligible, fd.notes,
            ])


def write_rund_stage_attrition(out_path: Path, folds: List[RunDFold]) -> None:
    cols = ["fold_id", "n_universe", "n_admit_e", "e_admit_pct_of_universe",
            "n_pre_t_losses", "n_admit_d1", "d1_admit_pct_of_e_admit",
            "e_precision_on_agg", "e_classifier_trained", "d1_classifier_trained"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        for fd in folds:
            w.writerow([
                fd.fold_id, fd.n_universe, fd.n_admit_e,
                _fmt(fd.e_admit_rate * 100),
                fd.n_pre_t_losses, fd.n_admit_d1,
                _fmt(fd.d1_admit_rate_within_e * 100),
                _fmt(fd.e_precision_on_agg * 100),
                "1" if fd.e_classifier_trained else "0",
                "1" if fd.d1_classifier_trained else "0",
            ])


def write_comparison_table(out_path: Path, all_runs: List[Tuple[str, str, str, RunAggregates]]) -> None:
    """Each tuple: (run_label, description, oracle_assumption, RunAggregates)."""
    cols = ["run", "description", "oracle_assumption", "sign_consistency",
            "worst_fold_roi_ann_pct", "mean_fold_roi_ann_pct",
            "worst_fold_dd_pct", "min_trade_count", "full_data_roi_pct",
            "full_data_ann_roi_pct", "full_data_dd_pct",
            "pass_deployable", "pass_viable", "notes"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        for label, desc, oracle, agg in all_runs:
            w.writerow([
                label, desc, oracle,
                "1" if agg.sign_consistency else "0",
                _fmt(agg.worst_fold_roi_pct, 4), _fmt(agg.mean_fold_roi_pct, 4),
                _fmt(agg.worst_fold_dd_pct, 4), agg.min_trade_count,
                _fmt(agg.full_data_roi_pct, 4),
                _fmt(agg.full_data_annualised_roi_pct, 4),
                _fmt(agg.full_data_max_dd_pct, 4),
                "1" if agg.pass_deployable else "0",
                "1" if agg.pass_viable else "0",
                "; ".join(agg.notes),
            ])


# ============================================================
# Driver
# ============================================================

def main() -> int:
    NO_ORACLE_DIR.mkdir(parents=True, exist_ok=True)

    print("[exp_s5_noOracle] loading Step 1 / Step 2 artefacts", file=sys.stderr)
    trades_df = pd.read_csv(_REPO_ROOT / "results/l_arc_11/step1_verbatim/trades_all.csv")
    paths_df = pd.read_csv(_REPO_ROOT / "results/l_arc_11/step1_verbatim/trades_paths.csv")
    clusters_df = pd.read_csv(_REPO_ROOT / "results/l_arc_11/step2/clusters_K4.csv")

    paths_index = build_paths_index(paths_df)

    # Pre-compute final_r at SL=3.0 for all trades.
    all_tids = sorted(trades_df["trade_id"].astype(int).tolist())
    print("[exp_s5_noOracle] computing final_r at SL=3.0 for all 2299 trades", file=sys.stderr)
    final_r_sl3 = compute_final_r_at_sl(paths_index, all_tids, SL_DEPLOY)

    # Build universe table with cluster_id + final_r_sl3.
    universe = trades_df.merge(clusters_df, on="trade_id", how="left").copy()
    universe["final_r_sl3"] = universe["trade_id"].map(final_r_sl3)

    print("[exp_s5_noOracle] caching 4H per-pair indicators", file=sys.stderr)
    pairs = sorted(trades_df["pair"].astype(str).unique())
    pair_caches = {p: _build_pair_cache(p, DATA_DIR_4H) for p in pairs}

    print("[exp_s5_noOracle] computing Pipeline E features for all 2299 trades", file=sys.stderr)
    e_features = compute_pipeline_e_features(trades_df, pair_caches)

    model_kw = dict(n_estimators=200, max_depth=8, random_state=42, n_jobs=1)

    # ============================
    # Run C — sweep thresholds.
    # ============================
    print("[exp_s5_noOracle] === Run C — Pipeline E predicting c1; threshold sweep ===", file=sys.stderr)
    full_days = (pd.Timestamp(FOLDS[-1][2]) - pd.Timestamp(FOLDS[0][1])).days
    runc_sweep: List[Tuple[float, RunAggregates]] = []
    runc_folds_by_threshold: Dict[float, List[RunCFold]] = {}
    runc_full_returns_by_threshold: Dict[float, List[float]] = {}
    for t in THRESHOLD_SWEEP_C:
        print(f"[exp_s5_noOracle]   Run C threshold={t}", file=sys.stderr)
        folds, full_returns = _run_c_at_threshold(t, universe, e_features, model_kw)
        runc_folds_by_threshold[t] = folds
        runc_full_returns_by_threshold[t] = full_returns
        fold_metrics_compat = [_runc_fold_to_foldmetrics(f) for f in folds]
        agg = aggregate_run(f"C_E_only_c1_t{t}", fold_metrics_compat, full_returns, full_days)
        runc_sweep.append((t, agg))
        print(f"[exp_s5_noOracle]     worst_roi_ann={agg.worst_fold_roi_pct:.2f}% "
              f"mean={agg.mean_fold_roi_pct:.2f}% dd={agg.worst_fold_dd_pct:.2f}% "
              f"pass_dep={agg.pass_deployable}", file=sys.stderr)

    write_runc_threshold_sweep(
        NO_ORACLE_DIR / "arc11_exp_s5_noOracle_C_threshold_sweep.csv", runc_sweep
    )

    # Select best operating point: max worst-fold ROI ann subject to DD <= 8%.
    eligible = [(t, agg) for t, agg in runc_sweep if agg.worst_fold_dd_pct <= 8.0]
    if eligible:
        best_t, best_agg = max(eligible, key=lambda x: x[1].worst_fold_roi_pct)
    else:
        best_t, best_agg = max(runc_sweep, key=lambda x: x[1].mean_fold_roi_pct)
    print(f"[exp_s5_noOracle] Run C best operating point: threshold={best_t}", file=sys.stderr)

    # Write per-fold for both 0.50 baseline AND best threshold.
    runc_perfold_rows: List[RunCFold] = []
    baseline_folds = runc_folds_by_threshold[0.50]
    for f in baseline_folds:
        # Tag run label inline by changing the writer slightly — easier to just write twice.
        pass
    write_runc_per_fold(
        NO_ORACLE_DIR / "arc11_exp_s5_noOracle_C_per_fold.csv",
        "C_E_only_c1_t0.50_baseline", baseline_folds,
    )
    # Append best (if different from 0.50) to the same file? Cleaner: write two CSVs.
    if abs(best_t - 0.50) > 1e-9:
        write_runc_per_fold(
            NO_ORACLE_DIR / f"arc11_exp_s5_noOracle_C_per_fold_best_t{best_t}.csv",
            f"C_E_only_c1_t{best_t}_best", runc_folds_by_threshold[best_t],
        )

    # Pick which aggregate goes into comparison table: report best operating point.
    baseline_agg = next(agg for t, agg in runc_sweep if abs(t - 0.50) < 1e-9)

    # ============================
    # Run D — E -> D1 cascade.
    # ============================
    print("[exp_s5_noOracle] === Run D — E -> D1 t=5 cascade ===", file=sys.stderr)
    rund_folds, rund_full_returns = _run_d_full(
        universe, paths_index, e_features, model_kw,
        t_e=THRESHOLD_E_RUN_D, t_d1=THRESHOLD_D1_RUN_D,
    )
    rund_fold_compat = [_rund_fold_to_foldmetrics(f) for f in rund_folds]
    rund_agg = aggregate_run("D_E_cascade_d1_t5_sl3", rund_fold_compat, rund_full_returns, full_days)
    print(f"[exp_s5_noOracle]   Run D worst_roi_ann={rund_agg.worst_fold_roi_pct:.2f}% "
          f"mean={rund_agg.mean_fold_roi_pct:.2f}% dd={rund_agg.worst_fold_dd_pct:.2f}% "
          f"pass_dep={rund_agg.pass_deployable}", file=sys.stderr)

    write_rund_per_fold(NO_ORACLE_DIR / "arc11_exp_s5_noOracle_D_per_fold.csv", rund_folds)
    write_rund_stage_attrition(NO_ORACLE_DIR / "arc11_exp_s5_noOracle_D_stage_attrition.csv", rund_folds)

    # ============================
    # Comparison table — A, B, C (best), C (0.50), D.
    # ============================
    print("[exp_s5_noOracle] loading prior runs A/B from arc11_exp_s5_aggregate.csv", file=sys.stderr)
    prior_agg_path = NO_ORACLE_DIR / "arc11_exp_s5_aggregate.csv"
    prior = pd.read_csv(prior_agg_path)

    def _row_to_agg(row: pd.Series) -> RunAggregates:
        return RunAggregates(
            run_name=row["run"], fold_metrics=[],
            sign_consistency=bool(int(row["sign_consistency"])),
            worst_fold_roi_pct=float(row["worst_fold_roi_ann_pct"]),
            mean_fold_roi_pct=float(row["mean_fold_roi_ann_pct"]),
            worst_fold_dd_pct=float(row["worst_fold_dd_pct"]),
            min_trade_count=int(row["min_trade_count"]),
            full_data_roi_pct=float(row["full_data_roi_pct"]),
            full_data_annualised_roi_pct=float(row["full_data_ann_roi_pct"]),
            full_data_max_dd_pct=float(row["full_data_dd_pct"]),
            full_data_total_days=int(row["full_data_days"]),
            pass_deployable=bool(int(row["pass_deployable"])),
            pass_viable=bool(int(row["pass_viable"])),
            notes=[],
        )

    a_agg = _row_to_agg(prior.iloc[0])
    b_agg = _row_to_agg(prior.iloc[1])

    comparison_rows = [
        ("A_c1_raw_sl3", "c1 raw at SL=3 (NO admission)", "cluster-ID-at-entry oracle", a_agg),
        ("B_agg_c1c3_d1_t5_sl3", "agg_c1_c3 + D1 t=5 classifier at SL=3", "cluster-ID-at-entry oracle", b_agg),
        ("C_E_only_c1_t0.50_baseline", f"Live E classifier predicting c1, t=0.50 baseline",
         "NO oracle — live E classifier", baseline_agg),
    ]
    if abs(best_t - 0.50) > 1e-9:
        comparison_rows.append((
            f"C_E_only_c1_t{best_t}_best",
            f"Live E classifier predicting c1, t={best_t} best operating point",
            "NO oracle — live E classifier", best_agg,
        ))
    comparison_rows.append((
        "D_E_cascade_d1_t5_sl3",
        "Live E -> D1 t=5 cascade, both t=0.50",
        "NO oracle — live E + live D1 cascade", rund_agg,
    ))
    write_comparison_table(NO_ORACLE_DIR / "comparison_table.csv", comparison_rows)

    # ============================
    # Stdout summary.
    # ============================
    print()
    print("=" * 96)
    print("EXPERIMENTAL S5 NO-ORACLE — Arc 11 (off-protocol; HALT status unchanged)")
    print("=" * 96)
    print()
    print(f"{'Run':<42} {'worst%':>10} {'mean%':>10} {'DD%':>8} {'trades':>8} {'pass_dep':>10}")
    print("-" * 96)
    for label, desc, oracle, agg in comparison_rows:
        print(f"{label:<42} {agg.worst_fold_roi_pct:>10.2f} {agg.mean_fold_roi_pct:>10.2f} "
              f"{agg.worst_fold_dd_pct:>8.2f} {agg.min_trade_count:>8d} "
              f"{'YES' if agg.pass_deployable else 'no':>10}")
    print()
    print("Run C threshold sweep:")
    for t, agg in runc_sweep:
        marker = " (BEST)" if abs(t - best_t) < 1e-9 else (" (baseline)" if abs(t - 0.50) < 1e-9 else "")
        print(f"  t={t:.2f}: worst={agg.worst_fold_roi_pct:7.2f}%  mean={agg.mean_fold_roi_pct:7.2f}%  "
              f"DD={agg.worst_fold_dd_pct:6.2f}%  min_trades={agg.min_trade_count:4d}  "
              f"pass_dep={'YES' if agg.pass_deployable else 'no'}{marker}")
    print()
    print("Run D stage attrition (cumulative across folds):")
    total_universe = sum(f.n_universe for f in rund_folds)
    total_admit_e = sum(f.n_admit_e for f in rund_folds)
    total_admit_d1 = sum(f.n_admit_d1 for f in rund_folds)
    total_pre_t = sum(f.n_pre_t_losses for f in rund_folds)
    print(f"  universe ({total_universe}) -> E-admit ({total_admit_e}, "
          f"{100 * total_admit_e / max(total_universe, 1):.1f}%) "
          f"-> pre-t losses ({total_pre_t}) + D1-admit ({total_admit_d1}, "
          f"{100 * total_admit_d1 / max(total_admit_e, 1):.1f}% of E-admit)")
    print()
    print("Oracle premium (mean-fold ROI ann %, no-oracle minus oracle):")
    print(f"  c1 leg:  C(0.50)  {baseline_agg.mean_fold_roi_pct:7.2f}%  vs  A(oracle)  {a_agg.mean_fold_roi_pct:7.2f}%  "
          f"-> delta {baseline_agg.mean_fold_roi_pct - a_agg.mean_fold_roi_pct:+.2f} pp")
    print(f"  agg leg: D(0.50)  {rund_agg.mean_fold_roi_pct:7.2f}%  vs  B(oracle)  {b_agg.mean_fold_roi_pct:7.2f}%  "
          f"-> delta {rund_agg.mean_fold_roi_pct - b_agg.mean_fold_roi_pct:+.2f} pp")
    print()
    print("Artefacts:")
    print(f"  {NO_ORACLE_DIR.relative_to(_REPO_ROOT)}/arc11_exp_s5_noOracle_C_threshold_sweep.csv")
    print(f"  {NO_ORACLE_DIR.relative_to(_REPO_ROOT)}/arc11_exp_s5_noOracle_C_per_fold.csv")
    if abs(best_t - 0.50) > 1e-9:
        print(f"  {NO_ORACLE_DIR.relative_to(_REPO_ROOT)}/arc11_exp_s5_noOracle_C_per_fold_best_t{best_t}.csv")
    print(f"  {NO_ORACLE_DIR.relative_to(_REPO_ROOT)}/arc11_exp_s5_noOracle_D_per_fold.csv")
    print(f"  {NO_ORACLE_DIR.relative_to(_REPO_ROOT)}/arc11_exp_s5_noOracle_D_stage_attrition.csv")
    print(f"  {NO_ORACLE_DIR.relative_to(_REPO_ROOT)}/comparison_table.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
