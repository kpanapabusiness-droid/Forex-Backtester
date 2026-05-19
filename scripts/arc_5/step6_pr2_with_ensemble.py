"""Arc 5 — Step 5 re-eval + Step 6 risk sweep + tiered ensemble — all under PR 2.

Replaces the prior PR 1 Step 6 dispatch (halted mid-run). Under PR 2, admitted
trades use:
  - Archetype-specific SL after classifier admission at bar t=1
    (cluster 1: 3.0×ATR; cluster 3: 2.0×ATR)
  - §11 row 2 exit policy: MFE-lock at 1R + trail 0.75R from new high
  - time_exit cap at bar 120
  - Pre-t SL = 2.0×ATR (same as PR 1)

Rejected trades (classifier prob < threshold) close at market on bar 2 open with
spread cost (same as PR 1 close-at-market).

Three strategies evaluated in parallel:
  1. Cluster 1 alone (P_c1 >= 0.20 -> c1 PR 2, else reject)
  2. Cluster 3 alone (P_c3 >= 0.15 -> c3 PR 2, else reject)
  3. Ensemble: P_c1 >= 0.20 -> Tier A (c1 PR 2);
              P_c1 < 0.20 AND P_c3 >= 0.15 -> Tier B (c3 PR 2);
              else Tier C (reject = close at bar 2)
     Parsimony: c1 wins ties (trades that pass both classifiers go Tier A).

Per strategy:
  - Step 5 re-eval under PR 2: §9 gates on per-fold metrics at baseline 0.5% risk
  - Step 6 risk sweep: §10 disposition at each risk in
    {0.50, 0.40, 0.30, 0.25, 0.20, 0.18, 0.16, 0.15, 0.14, 0.12, 0.10}%

PR 2 simulator forked from scripts/l_arc_4/step5_stability.py::_simulate_one_trade
with time_exit=120 cap (was 240/cap_bind in Arc 4 SL-only).

Spread cost on exit: for SL/trail exits the simulator uses effective_sl as exit
price (no bid-fill adjustment — matches the Arc 4 simulator convention; small
approximation of < ~spread_pips/2 per exit). For time_exit and close-at-market
on bar 2 open, spread cost = spread_pips_used (entry spread proxy) × pip_size,
applied to long exits.

Eligibility: per the PR 1 dispatch's halt diagnostics, trades with bars_held <= 2
in the original engine pool (SL hit on bar 0 or 1 before classifier evaluation)
are routed to their actual final_r regardless of classifier verdict — Pipeline D1
cannot reject what already closed.

Usage:
  py scripts/arc_5/step6_pr2_with_ensemble.py
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.arc_5.step5_rerun_new_spreads import (  # noqa: E402
    BASE_ENTRY_FEATURES,
    CLUSTER_F9_THRESHOLD,
    CLUSTER_LABEL,
    CLUSTER_R_FRAME_ATR_MULT,
    WFO_FOLDS,
    _build_entry_features_for_pool,
    _build_path_tensor,
    _compute_t1_features,
    _file_sha256,
)

# ============================================================
# Paths + constants
# ============================================================

NEW_STEP1_DIR = _REPO_ROOT / "results" / "l_arc_5" / "step1_spread_v2"
BASELINE_STEP5_DIR = _REPO_ROOT / "results" / "l_arc_5" / "step5"
OUT_DIR = _REPO_ROOT / "results" / "l_arc_5" / "step6_pr2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Risk grid per prompt
RISK_GRID_PCT: List[float] = [0.50, 0.40, 0.30, 0.25, 0.20, 0.18, 0.16, 0.15, 0.14, 0.12, 0.10]

# §10 thresholds
DEPLOY_WORST_ROI_ANN_PCT = 5.0
DEPLOY_MEAN_ROI_ANN_PCT = 8.0
DEPLOY_TRADES_PER_FOLD = 15
VIABLE_WORST_ROI_PCT = 0.0
VIABLE_MEAN_ROI_ANN_PCT = 3.0
VIABLE_TRADES_PER_FOLD = 5
DD_CEILING_PCT = 8.0

# PR 2 §11 row 2 policy parameters
PRE_T_SL_ATR_MULT = 2.0
MFE_LOCK_R = 1.0
TRAIL_DISTANCE_R = 0.75
TIME_EXIT_BAR = 120  # close at bar 120's open if no SL/trail by then
PATH_BARS = 241


# ============================================================
# PR 2 simulator (forked from l_arc_4 step5_stability with time_exit=120)
# ============================================================


def _pip_size(pair: str) -> float:
    return 0.01 if pair.endswith("_JPY") else 0.0001


@dataclass
class SimResult:
    exit_bar: int
    exit_reason: str   # "archetype_sl" | "mfe_lock_breakeven" | "trail" | "time_exit"
    exit_price: float
    final_r: float
    mfe_locked_bar: int  # -1 if never
    peak_mfe_r: float


def _simulate_pr2_one_trade(
    highs: np.ndarray,
    lows: np.ndarray,
    opens: np.ndarray,
    entry_price: float,
    atr_signal: float,
    cluster_sl_mult: float,
    spread_pips: float,
    pip_size: float,
) -> SimResult:
    """Walk bars 0..TIME_EXIT_BAR applying §11 row 2 (MFE-lock + trail) under PR 2.

    Pre-t (bars 0-1): SL = entry - PRE_T_SL_ATR_MULT × ATR.
    Post-t (bar 1 end, if not already trailing): SL replaced with archetype SL
      (entry - cluster_sl_mult × ATR). Wider for c1 (3×ATR), same for c3 (2×ATR).
    Per-bar: MFE update -> MFE-lock check -> trail update -> effective_sl ->
             SL check vs bar low -> end-of-bar ratchet.
    Time exit: if no SL/trail by bar TIME_EXIT_BAR-1, close at bar TIME_EXIT_BAR
               open with spread cost.

    R-frame: 1R = cluster_sl_mult × ATR (post-t archetype SL distance).
    final_r is in R units of that frame.

    Spread on exit: for SL/trail exits, exit_price = effective_sl (no bid adjustment;
      matches l_arc_4 simulator convention). For time_exit, exit_price =
      open(bar_time_exit) - spread_pips/2 × pip_size (long bid fill, proxying
      bar-time-exit spread with entry spread).
    """
    R = cluster_sl_mult * atr_signal
    pre_t_sl_price = entry_price - PRE_T_SL_ATR_MULT * atr_signal
    post_t_sl_price = entry_price - R

    current_sl_price = pre_t_sl_price
    mfe_locked = False
    trail_active = False
    trail_stop_price = -math.inf
    mfe_r = 0.0
    mfe_locked_bar = -1
    peak_mfe_r = 0.0

    n_bars = min(len(highs), TIME_EXIT_BAR)
    for t in range(n_bars):
        bar_high = float(highs[t])
        bar_low = float(lows[t])

        # Update MFE from bar high.
        bar_mfe_r = (bar_high - entry_price) / R
        new_mfe_high = bar_mfe_r > mfe_r
        if new_mfe_high:
            mfe_r = bar_mfe_r
        peak_mfe_r = max(peak_mfe_r, mfe_r)

        # MFE lock at 1R.
        if not mfe_locked and mfe_r >= MFE_LOCK_R:
            mfe_locked = True
            trail_active = True
            mfe_locked_bar = t
            trail_stop_price = entry_price + (mfe_r - TRAIL_DISTANCE_R) * R
        elif trail_active and new_mfe_high:
            new_trail = entry_price + (mfe_r - TRAIL_DISTANCE_R) * R
            if new_trail > trail_stop_price:
                trail_stop_price = new_trail

        # Effective SL.
        if trail_active:
            effective_sl = max(current_sl_price, trail_stop_price)
        else:
            effective_sl = current_sl_price

        # SL check.
        if bar_low <= effective_sl:
            exit_price = effective_sl
            if trail_active and effective_sl == trail_stop_price and trail_stop_price > current_sl_price:
                exit_reason = "trail"
            elif trail_active and effective_sl == entry_price:
                exit_reason = "mfe_lock_breakeven"
            else:
                exit_reason = "archetype_sl"
            final_r = (exit_price - entry_price) / R
            return SimResult(t, exit_reason, exit_price, final_r, mfe_locked_bar, peak_mfe_r)

        # End-of-bar ratchet of static SL.
        if trail_active and trail_stop_price > current_sl_price:
            current_sl_price = trail_stop_price

        # Post-t SL replacement at end of bar 1 (D1 mechanic), if not already trailing.
        if t == 1 and not trail_active:
            current_sl_price = post_t_sl_price

    # Time exit at bar TIME_EXIT_BAR open with spread cost.
    if len(opens) > TIME_EXIT_BAR:
        exit_mid = float(opens[TIME_EXIT_BAR])
    else:
        # Path doesn't extend to bar 120; use last available open.
        exit_mid = float(opens[-1])
    exit_price = exit_mid - (spread_pips / 2.0) * pip_size
    final_r = (exit_price - entry_price) / R
    return SimResult(TIME_EXIT_BAR, "time_exit", exit_price, final_r, mfe_locked_bar, peak_mfe_r)


def _compute_close_at_market_r(
    bar2_opens: np.ndarray,
    entry_prices: np.ndarray,
    sl_distance_price: np.ndarray,  # SL distance in PRICE units (for R denom)
    spread_pips: np.ndarray,
    pip_sizes: np.ndarray,
) -> np.ndarray:
    """Close at bar 2 open with spread cost. R denominated to sl_distance_price."""
    spread_price_half = spread_pips / 2.0 * pip_sizes
    pnl_price = bar2_opens - entry_prices - spread_price_half
    return pnl_price / sl_distance_price


# ============================================================
# Metrics
# ============================================================


def _compounded_per_fold_metrics(r_chrono: np.ndarray, risk_pct: float, oos_days: int) -> Dict[str, float]:
    n = len(r_chrono)
    if n == 0:
        return {"fold_roi_pct": 0.0, "fold_roi_ann_pct": 0.0, "fold_max_dd_pct": 0.0, "fold_terminal_equity": 1.0}
    r_scaled = r_chrono * (risk_pct / 100.0)
    eq = np.cumprod(1.0 + r_scaled)
    eq_with_start = np.concatenate([[1.0], eq])
    peak = np.maximum.accumulate(eq_with_start)
    dd = (peak - eq_with_start) / peak
    return {
        "fold_roi_pct": float((eq[-1] - 1.0) * 100.0),
        "fold_roi_ann_pct": float((eq[-1] - 1.0) * 100.0 * (365.0 / max(oos_days, 1))),
        "fold_max_dd_pct": float(np.max(dd) * 100.0),
        "fold_terminal_equity": float(eq[-1]),
    }


def _simple_sum_metrics(r_chrono: np.ndarray, risk_pct: float, oos_days: int) -> Dict[str, float]:
    """Step-5-style additive cum-sum for comparison."""
    n = len(r_chrono)
    if n == 0:
        return {"fold_roi_pct": 0.0, "fold_max_dd_pct": 0.0}
    r_scaled = r_chrono * (risk_pct / 100.0)
    eq = np.cumsum(r_scaled) * 100.0
    peak = np.maximum.accumulate(eq)
    return {
        "fold_roi_pct": float(eq[-1]),
        "fold_max_dd_pct": float(np.max(peak - eq)),
    }


def _check_deploy(per_fold: List[Dict], full_roi_pct: float, full_dd_pct: float, n_admit_per_fold: List[int]) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if not per_fold:
        return False, ["no folds with results"]
    worst_roi_ann = min(f["fold_roi_ann_pct"] for f in per_fold)
    mean_roi_ann = float(np.mean([f["fold_roi_ann_pct"] for f in per_fold]))
    worst_dd = max(f["fold_max_dd_pct"] for f in per_fold)
    min_trades = min(n_admit_per_fold)
    if worst_roi_ann < DEPLOY_WORST_ROI_ANN_PCT:
        reasons.append(f"worst-fold ann ROI {worst_roi_ann:.2f}% < {DEPLOY_WORST_ROI_ANN_PCT}%")
    if mean_roi_ann < DEPLOY_MEAN_ROI_ANN_PCT:
        reasons.append(f"mean-fold ann ROI {mean_roi_ann:.2f}% < {DEPLOY_MEAN_ROI_ANN_PCT}%")
    if worst_dd > DD_CEILING_PCT:
        reasons.append(f"worst-fold DD {worst_dd:.2f}% > {DD_CEILING_PCT}%")
    if min_trades < DEPLOY_TRADES_PER_FOLD:
        reasons.append(f"min trades/fold {min_trades} < {DEPLOY_TRADES_PER_FOLD}")
    return (len(reasons) == 0, reasons)


def _check_viable(per_fold: List[Dict], full_roi_pct: float, full_dd_pct: float, n_admit_per_fold: List[int]) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if not per_fold:
        return False, ["no folds with results"]
    worst_roi = min(f["fold_roi_pct"] for f in per_fold)
    mean_roi_ann = float(np.mean([f["fold_roi_ann_pct"] for f in per_fold]))
    worst_dd = max(f["fold_max_dd_pct"] for f in per_fold)
    min_trades = min(n_admit_per_fold)
    if worst_roi <= VIABLE_WORST_ROI_PCT:
        reasons.append(f"worst-fold ROI {worst_roi:.2f}% <= {VIABLE_WORST_ROI_PCT}%")
    if mean_roi_ann < VIABLE_MEAN_ROI_ANN_PCT:
        reasons.append(f"mean-fold ann ROI {mean_roi_ann:.2f}% < {VIABLE_MEAN_ROI_ANN_PCT}%")
    if worst_dd > DD_CEILING_PCT:
        reasons.append(f"worst-fold DD {worst_dd:.2f}% > {DD_CEILING_PCT}%")
    if min_trades < VIABLE_TRADES_PER_FOLD:
        reasons.append(f"min trades/fold {min_trades} < {VIABLE_TRADES_PER_FOLD}")
    return (len(reasons) == 0, reasons)


def _py(o):
    if isinstance(o, dict):
        return {k: _py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_py(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return _py(o.tolist())
    return o


# ============================================================
# Driver
# ============================================================


def main() -> int:
    t0 = time.time()
    rng_grid = list(RISK_GRID_PCT)

    print("[step6_pr2] loading inputs", file=sys.stderr)
    new_trades = pd.read_csv(NEW_STEP1_DIR / "trades_all.csv")
    new_trades_sorted = new_trades.sort_values("trade_id").reset_index(drop=True)
    new_paths = pd.read_csv(NEW_STEP1_DIR / "trades_paths.csv")
    n_total = len(new_trades_sorted)
    print(f"  pool: {n_total} trades", file=sys.stderr)

    print("[step6_pr2] building path tensor + entry features", file=sys.stderr)
    path_tensor = _build_path_tensor(new_paths, n_total)
    entry_feats = _build_entry_features_for_pool(new_trades_sorted)
    entry_sorted = entry_feats.sort_values("trade_id").reset_index(drop=True)
    if list(entry_sorted["trade_id"].astype(int)) != list(new_trades_sorted["trade_id"].astype(int)):
        raise RuntimeError("entry_features trade_id ordering mismatch")
    X_entry = entry_sorted[BASE_ENTRY_FEATURES].to_numpy(dtype=float)

    pairs_arr = new_trades_sorted["pair"].to_numpy()
    pip_sizes = np.array([_pip_size(p) for p in pairs_arr], dtype=float)
    entry_prices = new_trades_sorted["entry_price"].to_numpy(dtype=float)
    atr_14 = new_trades_sorted["atr_14_at_signal"].to_numpy(dtype=float)
    bars_held = new_trades_sorted["bars_held"].to_numpy(dtype=int)
    final_r_pool = new_trades_sorted["final_r"].to_numpy(dtype=float)
    spread_pips = new_trades_sorted["spread_pips_used"].to_numpy(dtype=float)
    entry_times = pd.to_datetime(new_trades_sorted["entry_time"])
    trade_ids_arr = new_trades_sorted["trade_id"].to_numpy(dtype=int)

    # ===== Close-at-market R (cluster-independent in price terms, but R denom depends
    # on cluster_sl_mult). Compute per cluster.
    print("[step6_pr2] computing close-at-market R (per cluster R-frame)", file=sys.stderr)
    bar2_open = path_tensor[:, 2, 0]
    cam_r_by_cluster: Dict[int, np.ndarray] = {}
    for cid, sl_mult in CLUSTER_R_FRAME_ATR_MULT.items():
        sl_dist_price = sl_mult * atr_14  # R-frame for this cluster
        cam_r = _compute_close_at_market_r(bar2_open, entry_prices, sl_dist_price, spread_pips, pip_sizes)
        cam_r_by_cluster[cid] = cam_r
    # Survivor mask (bars_held >= 3 in original pool, classifier can route)
    survived = bars_held >= 3
    # For early-exit trades (bars_held <= 2), R under cluster X = (engine final_r) / sl_mult_X / 2.0
    # The engine's final_r is denominated to 2.0×ATR. To re-express in cluster R-frame:
    # cluster_final_r = engine_final_r × (2.0 / cluster_sl_mult)
    early_exit_r_by_cluster: Dict[int, np.ndarray] = {}
    for cid, sl_mult in CLUSTER_R_FRAME_ATR_MULT.items():
        early_exit_r_by_cluster[cid] = final_r_pool * (2.0 / sl_mult)

    # ===== PR 2 simulator: per cluster, simulate every trade =====
    print("[step6_pr2] running PR 2 simulator per cluster", file=sys.stderr)
    sim_results_by_cluster: Dict[int, List[SimResult]] = {}
    for cid, sl_mult in CLUSTER_R_FRAME_ATR_MULT.items():
        print(f"  cluster {cid}: simulating {n_total} trades under R={sl_mult}×ATR", file=sys.stderr)
        sims: List[SimResult] = []
        for i in range(n_total):
            sims.append(_simulate_pr2_one_trade(
                highs=path_tensor[i, :, 1],
                lows=path_tensor[i, :, 2],
                opens=path_tensor[i, :, 0],
                entry_price=float(entry_prices[i]),
                atr_signal=float(atr_14[i]),
                cluster_sl_mult=sl_mult,
                spread_pips=float(spread_pips[i]),
                pip_size=float(pip_sizes[i]),
            ))
        sim_results_by_cluster[cid] = sims

    # Build per-trade R + exit-reason arrays per cluster (admitted scenario).
    pr2_final_r: Dict[int, np.ndarray] = {}
    pr2_exit_reason: Dict[int, np.ndarray] = {}
    pr2_exit_bar: Dict[int, np.ndarray] = {}
    pr2_mfe_locked_bar: Dict[int, np.ndarray] = {}
    pr2_peak_mfe_r: Dict[int, np.ndarray] = {}
    for cid in (1, 3):
        sims = sim_results_by_cluster[cid]
        pr2_final_r[cid] = np.array([s.final_r for s in sims], dtype=float)
        pr2_exit_reason[cid] = np.array([s.exit_reason for s in sims], dtype=object)
        pr2_exit_bar[cid] = np.array([s.exit_bar for s in sims], dtype=int)
        pr2_mfe_locked_bar[cid] = np.array([s.mfe_locked_bar for s in sims], dtype=int)
        pr2_peak_mfe_r[cid] = np.array([s.peak_mfe_r for s in sims], dtype=float)

    # ===== Per-cluster classifier scores per fold =====
    print("[step6_pr2] scoring classifiers per fold per cluster", file=sys.stderr)
    proba_by_cluster: Dict[int, Dict[int, np.ndarray]] = {1: {}, 3: {}}
    fold_oos_idx_by_fold: Dict[int, np.ndarray] = {}
    fold_clf_sha: Dict[Tuple[int, int], str] = {}
    for fidx, (_is_start, _is_end, oos_start, oos_end) in enumerate(WFO_FOLDS, start=1):
        oos_start_ts = pd.Timestamp(oos_start)
        oos_end_ts = pd.Timestamp(oos_end)
        mask = (entry_times >= oos_start_ts) & (entry_times < oos_end_ts)
        oos_idx = np.where(mask)[0]
        fold_oos_idx_by_fold[fidx] = oos_idx
        for cid in (1, 3):
            sl_mult = CLUSTER_R_FRAME_ATR_MULT[cid]
            t1_feats = _compute_t1_features(path_tensor[oos_idx], entry_prices[oos_idx], atr_14[oos_idx], sl_mult)
            X_full = np.concatenate([X_entry[oos_idx], t1_feats], axis=1)
            for j in range(X_full.shape[1]):
                col = X_full[:, j]
                if np.isnan(col).any():
                    col = np.where(np.isnan(col), np.nanmedian(col), col)
                    X_full[:, j] = col
            clf_path = BASELINE_STEP5_DIR / f"cluster_{cid}" / "per_fold_classifiers" / f"fold_{fidx}_classifier.joblib"
            fold_clf_sha[(cid, fidx)] = _file_sha256(clf_path)
            clf = joblib.load(clf_path)
            proba_by_cluster[cid][fidx] = clf.predict_proba(X_full)[:, 1]
    print(f"  classifier sha256 captured for {len(fold_clf_sha)} (cluster, fold) pairs", file=sys.stderr)

    # ===== Per-strategy per-fold r-vectors =====
    @dataclass
    class FoldStratData:
        fold: int
        oos_start: pd.Timestamp
        oos_end: pd.Timestamp
        oos_days: int
        n_oos: int
        n_admit: int
        admit_rate: float
        n_target_admit_pos: int  # admits where final_r > 0 (win rate metric)
        r_chrono: np.ndarray
        is_admit_chrono: np.ndarray
        entry_time_chrono: np.ndarray
        trade_id_chrono: np.ndarray
        exit_reason_chrono: np.ndarray
        tier_chrono: np.ndarray  # 'A', 'B', 'C' for ensemble; 'A' = admit for cluster strat
        clf_sha_used: List[str]

    def _build_fold_strat_data_cluster(cid: int) -> Dict[int, FoldStratData]:
        """For a cluster-alone strategy."""
        threshold = CLUSTER_F9_THRESHOLD[cid]
        cam_r = cam_r_by_cluster[cid]
        early_r = early_exit_r_by_cluster[cid]
        out: Dict[int, FoldStratData] = {}
        for fidx, (_, _, oos_start, oos_end) in enumerate(WFO_FOLDS, start=1):
            oos_start_ts = pd.Timestamp(oos_start)
            oos_end_ts = pd.Timestamp(oos_end)
            oos_days = int((oos_end_ts - oos_start_ts).total_seconds() / 86400)
            oos_idx = fold_oos_idx_by_fold[fidx]
            proba = proba_by_cluster[cid][fidx]
            admit_mask = proba >= threshold
            r = np.empty(len(oos_idx), dtype=float)
            exit_reason = np.empty(len(oos_idx), dtype=object)
            for k, tid_idx in enumerate(oos_idx):
                if bars_held[tid_idx] <= 2:
                    # Trade closed before classifier eval -> engine R
                    r[k] = early_r[tid_idx]
                    exit_reason[k] = "early_engine_sl"
                elif admit_mask[k]:
                    # Admitted: PR 2 simulated outcome
                    r[k] = pr2_final_r[cid][tid_idx]
                    exit_reason[k] = str(pr2_exit_reason[cid][tid_idx])
                else:
                    # Rejected: close at bar 2
                    r[k] = cam_r[tid_idx]
                    exit_reason[k] = "close_at_market_bar2"
            # Chrono sort
            et_oos = entry_times.iloc[oos_idx].to_numpy()
            order = np.argsort(et_oos, kind="mergesort")
            r_chrono = r[order]
            is_admit_chrono = (admit_mask & (bars_held[oos_idx] >= 3))[order]
            et_chrono = et_oos[order]
            tid_chrono = trade_ids_arr[oos_idx][order]
            er_chrono = exit_reason[order]
            tier_chrono = np.where(is_admit_chrono, "A", "C")
            n_admit = int(is_admit_chrono.sum())
            admit_rate = n_admit / len(oos_idx) if len(oos_idx) > 0 else 0.0
            # Win-rate over admits = fraction with final_r > 0 within admits
            admit_idx_global = oos_idx[admit_mask & (bars_held[oos_idx] >= 3)]
            n_target_admit_pos = int((pr2_final_r[cid][admit_idx_global] > 0).sum())
            out[fidx] = FoldStratData(
                fold=fidx, oos_start=oos_start_ts, oos_end=oos_end_ts, oos_days=oos_days,
                n_oos=len(oos_idx), n_admit=n_admit, admit_rate=admit_rate,
                n_target_admit_pos=n_target_admit_pos, r_chrono=r_chrono,
                is_admit_chrono=is_admit_chrono, entry_time_chrono=et_chrono,
                trade_id_chrono=tid_chrono, exit_reason_chrono=er_chrono,
                tier_chrono=tier_chrono, clf_sha_used=[fold_clf_sha[(cid, fidx)]],
            )
        return out

    def _build_fold_strat_data_ensemble() -> Dict[int, FoldStratData]:
        """For the tiered ensemble strategy."""
        t1 = CLUSTER_F9_THRESHOLD[1]
        t3 = CLUSTER_F9_THRESHOLD[3]
        cam_r_1 = cam_r_by_cluster[1]
        cam_r_3 = cam_r_by_cluster[3]
        early_r_1 = early_exit_r_by_cluster[1]
        early_r_3 = early_exit_r_by_cluster[3]
        out: Dict[int, FoldStratData] = {}
        for fidx, (_, _, oos_start, oos_end) in enumerate(WFO_FOLDS, start=1):
            oos_start_ts = pd.Timestamp(oos_start)
            oos_end_ts = pd.Timestamp(oos_end)
            oos_days = int((oos_end_ts - oos_start_ts).total_seconds() / 86400)
            oos_idx = fold_oos_idx_by_fold[fidx]
            proba_1 = proba_by_cluster[1][fidx]
            proba_3 = proba_by_cluster[3][fidx]
            tier = np.empty(len(oos_idx), dtype=object)
            r = np.empty(len(oos_idx), dtype=float)
            exit_reason = np.empty(len(oos_idx), dtype=object)
            for k, tid_idx in enumerate(oos_idx):
                if proba_1[k] >= t1:
                    # Tier A -> cluster 1 PR 2
                    tier[k] = "A"
                    if bars_held[tid_idx] <= 2:
                        r[k] = early_r_1[tid_idx]
                        exit_reason[k] = "early_engine_sl"
                    else:
                        r[k] = pr2_final_r[1][tid_idx]
                        exit_reason[k] = str(pr2_exit_reason[1][tid_idx])
                elif proba_3[k] >= t3:
                    # Tier B -> cluster 3 PR 2
                    tier[k] = "B"
                    if bars_held[tid_idx] <= 2:
                        r[k] = early_r_3[tid_idx]
                        exit_reason[k] = "early_engine_sl"
                    else:
                        r[k] = pr2_final_r[3][tid_idx]
                        exit_reason[k] = str(pr2_exit_reason[3][tid_idx])
                else:
                    # Tier C -> close at market (use c3 R-frame for consistency; small)
                    tier[k] = "C"
                    r[k] = cam_r_3[tid_idx]
                    exit_reason[k] = "close_at_market_bar2"
            # Admit set = A + B (C is "rejected")
            admit_mask = np.isin(tier, ["A", "B"])
            # Chrono sort
            et_oos = entry_times.iloc[oos_idx].to_numpy()
            order = np.argsort(et_oos, kind="mergesort")
            r_chrono = r[order]
            is_admit_chrono = admit_mask[order]
            tier_chrono = tier[order]
            et_chrono = et_oos[order]
            tid_chrono = trade_ids_arr[oos_idx][order]
            er_chrono = exit_reason[order]
            n_admit = int(admit_mask.sum())
            admit_rate = n_admit / len(oos_idx) if len(oos_idx) > 0 else 0.0
            # Win rate = positive among admits (using their own R-frame)
            admit_global = oos_idx[admit_mask]
            n_target_admit_pos = int((r[admit_mask] > 0).sum())
            out[fidx] = FoldStratData(
                fold=fidx, oos_start=oos_start_ts, oos_end=oos_end_ts, oos_days=oos_days,
                n_oos=len(oos_idx), n_admit=n_admit, admit_rate=admit_rate,
                n_target_admit_pos=n_target_admit_pos, r_chrono=r_chrono,
                is_admit_chrono=is_admit_chrono, entry_time_chrono=et_chrono,
                trade_id_chrono=tid_chrono, exit_reason_chrono=er_chrono,
                tier_chrono=tier_chrono, clf_sha_used=[fold_clf_sha[(1, fidx)], fold_clf_sha[(3, fidx)]],
            )
        return out

    strategies: Dict[str, Dict[int, FoldStratData]] = {
        "cluster_1": _build_fold_strat_data_cluster(1),
        "cluster_3": _build_fold_strat_data_cluster(3),
        "ensemble": _build_fold_strat_data_ensemble(),
    }

    for sname, sd in strategies.items():
        print(f"[step6_pr2] {sname} per-fold admit counts:", file=sys.stderr)
        for fidx in range(1, 8):
            f = sd[fidx]
            print(f"  fold {fidx}: n_oos={f.n_oos}, n_admit={f.n_admit} ({f.admit_rate*100:.2f}%)", file=sys.stderr)

    # ===== Per-strategy: §9 + Step 6 risk sweep =====
    ship_candidates: Dict[str, Dict] = {}
    summary_rows: List[Dict] = []
    pr1_vs_pr2_rows: List[Dict] = []

    for sname, sd in strategies.items():
        strat_dir = OUT_DIR / sname
        strat_dir.mkdir(parents=True, exist_ok=True)

        # ===== Step 5 re-eval (§9) at baseline 0.5% risk =====
        per_fold_baseline: List[Dict] = []
        for fidx in range(1, 8):
            f = sd[fidx]
            m = _compounded_per_fold_metrics(f.r_chrono, 0.5, f.oos_days)
            ss = _simple_sum_metrics(f.r_chrono, 0.5, f.oos_days)
            per_fold_baseline.append({
                "fold": f.fold,
                "oos_start": f.oos_start.isoformat(),
                "oos_end": f.oos_end.isoformat(),
                "oos_days": f.oos_days,
                "n_oos": f.n_oos,
                "n_admit": f.n_admit,
                "admit_rate": f.admit_rate,
                "win_rate_admits": (f.n_target_admit_pos / f.n_admit) if f.n_admit > 0 else 0.0,
                "compounded_fold_roi_pct": m["fold_roi_pct"],
                "compounded_fold_roi_ann_pct": m["fold_roi_ann_pct"],
                "compounded_fold_max_dd_pct": m["fold_max_dd_pct"],
                "simple_sum_fold_roi_pct": ss["fold_roi_pct"],
                "simple_sum_fold_max_dd_pct": ss["fold_max_dd_pct"],
                "mean_r_admits_only": float(f.r_chrono[f.is_admit_chrono].mean()) if f.n_admit > 0 else 0.0,
                "mean_r_full_oos": float(f.r_chrono.mean()),
            })
        pd.DataFrame(per_fold_baseline).to_csv(strat_dir / "fold_stability_pr2.csv", index=False)

        # §9 on full OOS chrono (admits + rejects) — compounded
        sign_consistency = all(
            (sd[fidx].r_chrono[sd[fidx].is_admit_chrono].mean() if sd[fidx].n_admit > 0 else 0.0) > 0
            for fidx in range(1, 8)
        )
        admit_rates = [sd[fidx].admit_rate for fidx in range(1, 8) if sd[fidx].admit_rate > 0]
        size_ratio = max(admit_rates) / min(admit_rates) if admit_rates else float("inf")
        size_pass = size_ratio <= 3.0
        dd_values = [per_fold_baseline[fidx - 1]["compounded_fold_max_dd_pct"] for fidx in range(1, 8)]
        dd_med = float(np.median(dd_values))
        dd_max = float(np.max(dd_values))
        dd_ratio = dd_max / dd_med if dd_med > 0 else float("inf")
        dd_pass = dd_ratio <= 2.0
        overall_pass = sign_consistency and size_pass and dd_pass
        gate_rows = [
            {"gate": "A_sign_consistency", "spec": "admit-set mean(final_r) > 0 in every fold",
             "result": "PASS" if sign_consistency else "FAIL", "value": "all positive" if sign_consistency else "fails"},
            {"gate": "B_size_variance", "spec": "max(admit_rate)/min(admit_rate) <= 3.0",
             "result": "PASS" if size_pass else "FAIL",
             "value": f"ratio={size_ratio:.6f}; min={min(admit_rates):.6f}; max={max(admit_rates):.6f}" if admit_rates else "no admits"},
            {"gate": "C_dd_ceiling", "spec": "worst-fold compounded DD <= 2.0 x median-fold compounded DD",
             "result": "PASS" if dd_pass else "FAIL",
             "value": f"max={dd_max:.6f}; median={dd_med:.6f}; ratio={dd_ratio:.6f}"},
            {"gate": "overall_verdict",
             "spec": "A AND B AND C",
             "result": "PASS" if overall_pass else "FAIL",
             "value": "all gates pass" if overall_pass else "at least one gate fails"},
        ]
        pd.DataFrame(gate_rows).to_csv(strat_dir / "gate_check_pr2.csv", index=False)

        # ===== Step 6 risk sweep =====
        sweep_rows: List[Dict] = []
        best_tier = "FAIL"
        best_risk: Optional[float] = None
        best_metrics: Optional[Dict] = None
        best_per_fold: Optional[List[Dict]] = None
        tier_rank_map = {"FAIL": 0, "VIABLE": 1, "DEPLOYABLE": 2}
        best_key: Optional[Tuple[int, float, float]] = None  # (tier_rank, worst_roi_ann, smaller_risk_better)

        for risk_pct in rng_grid:
            r_bps = int(round(risk_pct * 100))
            per_fold: List[Dict] = []
            cross_eq = 1.0
            for fidx in range(1, 8):
                f = sd[fidx]
                m = _compounded_per_fold_metrics(f.r_chrono, risk_pct, f.oos_days)
                per_fold.append({
                    "fold": fidx,
                    "oos_days": f.oos_days,
                    "n_oos": f.n_oos,
                    "n_admit": f.n_admit,
                    "admit_rate": f.admit_rate,
                    "win_rate_admits": (f.n_target_admit_pos / f.n_admit) if f.n_admit > 0 else 0.0,
                    "fold_roi_pct": m["fold_roi_pct"],
                    "fold_roi_ann_pct": m["fold_roi_ann_pct"],
                    "fold_max_dd_pct": m["fold_max_dd_pct"],
                    "fold_terminal_equity": m["fold_terminal_equity"],
                })
                # Cross-fold compounded
                r_scaled = f.r_chrono * (risk_pct / 100.0)
                for r in r_scaled:
                    cross_eq *= (1.0 + r)
            full_roi_pct = (cross_eq - 1.0) * 100.0
            # Cross-fold DD
            cross_curve = [1.0]
            running = 1.0
            for fidx in range(1, 8):
                r_scaled = sd[fidx].r_chrono * (risk_pct / 100.0)
                for r in r_scaled:
                    running *= (1.0 + r)
                    cross_curve.append(running)
            cross_arr = np.asarray(cross_curve)
            peak = np.maximum.accumulate(cross_arr)
            full_dd_pct = float(np.max((peak - cross_arr) / peak) * 100.0)

            n_admit_per_fold = [sd[fidx].n_admit for fidx in range(1, 8)]
            worst_roi_ann = float(min(f["fold_roi_ann_pct"] for f in per_fold))
            worst_roi = float(min(f["fold_roi_pct"] for f in per_fold))
            mean_roi_ann = float(np.mean([f["fold_roi_ann_pct"] for f in per_fold]))
            mean_fold_dd = float(np.mean([f["fold_max_dd_pct"] for f in per_fold]))
            worst_dd = float(max(f["fold_max_dd_pct"] for f in per_fold))
            ok_d, deploy_reasons = _check_deploy(per_fold, full_roi_pct, full_dd_pct, n_admit_per_fold)
            ok_v, viable_reasons = _check_viable(per_fold, full_roi_pct, full_dd_pct, n_admit_per_fold)
            if ok_d:
                tier = "DEPLOYABLE"
                fail = ""
            elif ok_v:
                tier = "VIABLE"
                fail = " | ".join(deploy_reasons)
            else:
                tier = "FAIL"
                fail = " | ".join(viable_reasons)
            sweep_rows.append({
                "strategy": sname,
                "risk_pct": risk_pct,
                "risk_bps": r_bps,
                "n_admit_total": int(sum(n_admit_per_fold)),
                "n_admit_min_per_fold": int(min(n_admit_per_fold)),
                "worst_fold_roi_ann_pct": worst_roi_ann,
                "worst_fold_roi_pct": worst_roi,
                "mean_fold_roi_ann_pct": mean_roi_ann,
                "worst_fold_max_dd_pct": worst_dd,
                "mean_fold_max_dd_pct": mean_fold_dd,
                "full_data_compounded_roi_pct": full_roi_pct,
                "full_data_compounded_max_dd_pct": full_dd_pct,
                "tier": tier,
                "tier_rank": tier_rank_map[tier],
                "fail_reasons": fail,
            })

            tr = tier_rank_map[tier]
            # Best: highest tier, then smallest risk, then highest worst_roi_ann
            cand_key = (tr, -risk_pct, worst_roi_ann)
            if best_key is None or tr > tier_rank_map[best_tier] or (tr == tier_rank_map[best_tier] and cand_key > best_key):
                best_tier = tier
                best_risk = risk_pct
                best_metrics = {
                    "worst_fold_roi_ann_pct": worst_roi_ann,
                    "worst_fold_roi_pct": worst_roi,
                    "mean_fold_roi_ann_pct": mean_roi_ann,
                    "worst_fold_max_dd_pct": worst_dd,
                    "mean_fold_max_dd_pct": mean_fold_dd,
                    "full_data_compounded_roi_pct": full_roi_pct,
                    "full_data_compounded_max_dd_pct": full_dd_pct,
                    "n_admit_total": int(sum(n_admit_per_fold)),
                    "n_admit_min_per_fold": int(min(n_admit_per_fold)),
                }
                best_per_fold = per_fold
                best_key = cand_key

            pd.DataFrame(per_fold).to_csv(strat_dir / f"per_fold_wfo_{sname}_risk_{r_bps}bps.csv", index=False)

        pd.DataFrame(sweep_rows).to_csv(strat_dir / "risk_sweep_pr2.csv", index=False)

        # ===== Equity curve at best risk =====
        if best_risk is not None:
            best_bps = int(round(best_risk * 100))
            curves: List[Dict] = []
            cross_eq = 1.0
            step = 0
            for fidx in range(1, 8):
                f = sd[fidx]
                r_scaled = f.r_chrono * (best_risk / 100.0)
                fold_eq = 1.0
                for i, r in enumerate(r_scaled):
                    fold_eq *= (1.0 + r)
                    cross_eq *= (1.0 + r)
                    step += 1
                    curves.append({
                        "fold": fidx,
                        "global_step": step,
                        "trade_id": int(f.trade_id_chrono[i]),
                        "entry_time": pd.Timestamp(f.entry_time_chrono[i]).isoformat(),
                        "tier": str(f.tier_chrono[i]),
                        "is_admit": int(f.is_admit_chrono[i]),
                        "trade_r": float(f.r_chrono[i]),
                        "exit_reason": str(f.exit_reason_chrono[i]),
                        "fold_equity": float(fold_eq),
                        "cross_fold_equity": float(cross_eq),
                    })
            pd.DataFrame(curves).to_csv(strat_dir / f"equity_curve_{sname}_risk_{best_bps}bps.csv", index=False)

        cfg_yaml = _py({
            "strategy": sname,
            "best_risk_pct": best_risk,
            "tier": best_tier,
            "metrics": best_metrics,
            "section_9_overall_pass": overall_pass,
            "section_9_sign_consistency": sign_consistency,
            "section_9_size_ratio": size_ratio,
            "section_9_dd_ratio": dd_ratio,
            "section_9_dd_max_pct": dd_max,
            "section_9_dd_median_pct": dd_med,
            "classifier_sha256s_used": list({s for fidx in range(1, 8) for s in sd[fidx].clf_sha_used}),
        })
        (strat_dir / "best_risk_config.yaml").write_text(yaml.safe_dump(cfg_yaml, sort_keys=False))

        summary_rows.append({
            "strategy": sname,
            "best_risk_pct": best_risk,
            "tier": best_tier,
            "section_9_pass": overall_pass,
            "section_9_dd_ratio": dd_ratio,
            **(best_metrics or {}),
        })
        ship_candidates[sname] = {
            "tier": best_tier,
            "best_risk_pct": best_risk,
            "metrics": best_metrics,
            "section_9_pass": overall_pass,
        }
        print(f"  {sname}: best risk={best_risk}% tier={best_tier} worst-ROI={best_metrics['worst_fold_roi_ann_pct']:.2f}% worst-DD={best_metrics['worst_fold_max_dd_pct']:.2f}% §9_pass={overall_pass}", file=sys.stderr)

    # ===== Strategy comparison summary =====
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "strategy_comparison_summary.csv", index=False)

    # ===== PR 1 vs PR 2 comparison (per cluster) =====
    # Load prior PR 1 numbers from step5_spread_v2/
    for cid in (1, 3):
        sd = strategies[f"cluster_{cid}"]
        pr1_path = _REPO_ROOT / "results" / "l_arc_5" / "step5_spread_v2" / f"cluster_{cid}" / f"fold_stability_new_spreads.csv"
        pr1 = pd.read_csv(pr1_path)
        for fidx in range(1, 8):
            f = sd[fidx]
            m = _compounded_per_fold_metrics(f.r_chrono, 0.5, f.oos_days)
            pr1_row = pr1[pr1["fold"] == fidx].iloc[0]
            pr1_vs_pr2_rows.append({
                "cluster_id": cid,
                "fold": fidx,
                "pr1_admit_rate": float(pr1_row["admit_rate"]),
                "pr2_admit_rate": float(f.admit_rate),
                "pr1_mean_r_admits": float(pr1_row["final_r_mean"]),
                "pr2_mean_r_admits": float(f.r_chrono[f.is_admit_chrono].mean()) if f.n_admit > 0 else 0.0,
                "pr1_fold_roi_ann_pct_simple_sum": float(pr1_row["fold_roi_pct_annualised"]),
                "pr2_fold_roi_ann_pct_compounded": m["fold_roi_ann_pct"],
                "pr1_fold_max_dd_pct_simple_sum": float(pr1_row["fold_max_dd_pct"]),
                "pr2_fold_max_dd_pct_compounded": m["fold_max_dd_pct"],
            })
    pd.DataFrame(pr1_vs_pr2_rows).to_csv(OUT_DIR / "pr1_vs_pr2_per_cluster.csv", index=False)

    # ===== Exit reason breakdown per cluster strategy =====
    for cid in (1, 3):
        sd = strategies[f"cluster_{cid}"]
        all_exits: List[str] = []
        for fidx in range(1, 8):
            f = sd[fidx]
            # only count admits
            for i in range(len(f.r_chrono)):
                if f.is_admit_chrono[i]:
                    all_exits.append(str(f.exit_reason_chrono[i]))
        counts = pd.Series(all_exits).value_counts()
        rows = [{"cluster_id": cid, "exit_reason": k, "count": int(v), "pct": float(v / max(len(all_exits), 1) * 100)}
                for k, v in counts.items()]
        pd.DataFrame(rows).to_csv(OUT_DIR / f"cluster_{cid}" / f"exit_reason_breakdown_{cid}.csv", index=False)

    # Ensemble tier breakdown
    sd_ens = strategies["ensemble"]
    tier_rows = []
    for tier_letter in ("A", "B", "C"):
        all_r: List[float] = []
        per_fold_means: List[float] = []
        per_fold_counts: List[int] = []
        for fidx in range(1, 8):
            f = sd_ens[fidx]
            mask = (f.tier_chrono == tier_letter)
            if mask.sum() > 0:
                per_fold_means.append(float(f.r_chrono[mask].mean()))
                per_fold_counts.append(int(mask.sum()))
                all_r.extend(f.r_chrono[mask].tolist())
            else:
                per_fold_means.append(0.0)
                per_fold_counts.append(0)
        total_r = float(np.sum(all_r))
        tier_rows.append({
            "tier": tier_letter,
            "tier_label": {"A": "P_c1>=0.20 -> c1 PR2", "B": "P_c1<0.20 & P_c3>=0.15 -> c3 PR2", "C": "rejected -> close@market"}[tier_letter],
            "total_trades": int(len(all_r)),
            "mean_r": float(np.mean(all_r)) if all_r else 0.0,
            "worst_fold_mean_r": float(min(per_fold_means)),
            "best_fold_mean_r": float(max(per_fold_means)),
            "sum_r_total": total_r,
            "per_fold_counts_csv": ";".join(str(c) for c in per_fold_counts),
        })
    pd.DataFrame(tier_rows).to_csv(OUT_DIR / "ensemble" / "tier_breakdown.csv", index=False)

    # ===== Ship decision =====
    tier_rank_map = {"FAIL": 0, "VIABLE": 1, "DEPLOYABLE": 2}
    ranks = {s: tier_rank_map[ship_candidates[s]["tier"]] for s in ship_candidates}
    if max(ranks.values()) == 0:
        ship_pick = None
        rationale = "No strategy reaches PASS-VIABLE or higher. Arc 5 closes Step 6 FAIL."
    else:
        max_tier_rank = max(ranks.values())
        eligible = [s for s, r in ranks.items() if r == max_tier_rank]
        eligible.sort(key=lambda s: -(ship_candidates[s]["metrics"]["worst_fold_roi_ann_pct"]))
        ship_pick = eligible[0]
        tier_name = ship_candidates[ship_pick]["tier"]
        rationale = (
            f"Highest §10 tier reached: {tier_name}. Eligible: {eligible}. "
            f"Tie-broken on worst-fold annualised ROI: {ship_pick} wins at "
            f"{ship_candidates[ship_pick]['metrics']['worst_fold_roi_ann_pct']:.2f}%."
        )

    ship_yaml = _py({
        "ship_pick_strategy": ship_pick,
        "rationale": rationale,
        "strategies": ship_candidates,
        "protocol_version": "L_ARC_PROTOCOL v2.1.1 §10 (PR 2 mechanics)",
        "spread_regime": "2026-05-17 per-pair p50 (HistData 2024-2025)",
        "exit_policy": "§11 row 2 Stepwise climber (MFE-lock at 1R + trail 0.75R from new high, time_exit=120)",
        "rejected_trade_handling": "close-at-market on bar 2 open with spread cost (proxy: entry-bar spread)",
        "skip_to_step_6_caveat": (
            "Baseline classifiers + F9 thresholds reused (no retraining). "
            "Cluster labels frozen via Phase 2 cluster_assignment.csv. "
            "Decision-quality, not publication-quality."
        ),
    })
    (OUT_DIR / "ship_decision_pr2.yaml").write_text(yaml.safe_dump(ship_yaml, sort_keys=False))

    print(f"[step6_pr2] DONE in {time.time() - t0:.1f}s; ship_pick={ship_pick}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
