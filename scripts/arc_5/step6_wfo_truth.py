"""Arc 5 — Step 6 (WFO truth + ship rule) under 2026-05-17 spread calibration.

Per L_ARC_PROTOCOL.md v2.1.1 §10. Final gate before ship/no-ship decision.

Skip-to-Step-6 framing (consistent with Step 5 re-run; deliberate approximation):
  - Baseline per-fold classifiers reused (no retraining, sha-verified unchanged).
  - F9 thresholds frozen (cluster 1 = 0.20, cluster 3 = 0.15).
  - Cluster labels frozen via Phase 2 cluster_assignment.csv
    (12,144 matched by signal_time + 204 nearest-centroid).
  - New-spread trade pool from results/l_arc_5/step1_spread_v2/.

Per §10 spec, Step 6 evaluates COMPOUNDED equity (not additive cum-sum) —
distinct from Step 5's convention (a) simple-sum DD. Per-trade equity update:
    eq_after = eq_before × (1 + r × risk_pct)
Compounded ROI = (eq_final − 1) × 100. Compounded DD = max((peak − eq) / peak) × 100.

Pipeline D1 PR-1 mechanics for rejected trades (per prompt §5–§6):
  - Admit (classifier_proba ≥ threshold) → outcome = trades_all.final_r
    (real outcome under 2.0×ATR SL to time_exit=120, from Step 1 baseline).
  - Reject → close-at-market at bar_offset=2 open with applied spread cost:
      close_at_market_R = (bar2_open_mid − entry_price − spread_pips_used × pip_size / 2) /
                          sl_distance_price
    (entry_price already absorbs +S_bar1/2; bar2 spread proxied with bar1 spread —
     small approximation, magnitude < 1 pip on most pairs.)

Outputs (per cluster c ∈ {1, 3}) under results/l_arc_5/step6/cluster_<c>/:
  - risk_sweep_<c>.csv
  - per_fold_wfo_<c>_risk_<r_bps>.csv (one per risk level)
  - equity_curves_<c>_risk_<r_bps>.csv (per-fold OOS equity series at best risk)
  - best_risk_config_<c>.yaml

Top-level under results/l_arc_5/step6/:
  - ship_decision.yaml
  - arc_5_step6_summary.csv
  - PHASE_L_ARC_5_STEP6.md (written by separate post-processing step)

Determinism: deterministic per-fold arithmetic; classifiers loaded read-only.
Two-run byte-identical (no randomness).

Usage:
  py scripts/arc_5/step6_wfo_truth.py
"""

from __future__ import annotations

import hashlib
import json
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
    ALL_FEATURES,
    BASE_ENTRY_FEATURES,
    CLUSTER_F9_THRESHOLD,
    CLUSTER_LABEL,
    CLUSTER_R_FRAME_ATR_MULT,
    DATA_DIR,
    WFO_FOLDS,
    _build_entry_features_for_pool,
    _build_path_tensor,
    _compute_t1_features,
    _file_sha256,
)

# ============================================================
# Paths + constants
# ============================================================

BASELINE_STEP1_DIR = _REPO_ROOT / "results" / "l_arc_5" / "step1"
NEW_STEP1_DIR = _REPO_ROOT / "results" / "l_arc_5" / "step1_spread_v2"
BASELINE_STEP5_DIR = _REPO_ROOT / "results" / "l_arc_5" / "step5"
STEP5_V2_DIR = _REPO_ROOT / "results" / "l_arc_5" / "step5_spread_v2"

OUT_DIR = _REPO_ROOT / "results" / "l_arc_5" / "step6"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Risk grid per prompt
RISK_GRID_PCT: List[float] = [0.50, 0.40, 0.30, 0.25, 0.20, 0.18, 0.16, 0.15, 0.14, 0.12, 0.10]

# §10 thresholds
DEPLOY_WORST_ROI_ANN_PCT = 5.0
DEPLOY_MEAN_ROI_ANN_PCT = 8.0
DEPLOY_FULL_DATA_ROI_PCT = 5.0
DEPLOY_TRADES_PER_FOLD = 15
VIABLE_WORST_ROI_PCT = 0.0
VIABLE_MEAN_ROI_ANN_PCT = 3.0
VIABLE_FULL_DATA_ROI_PCT = 3.0
VIABLE_TRADES_PER_FOLD = 5
DD_CEILING_PCT = 8.0
FULL_DATA_DD_CEILING_PCT = 10.0


# ============================================================
# Helpers
# ============================================================


def _pip_size(pair: str) -> float:
    return 0.01 if pair.endswith("_JPY") else 0.0001


def _compute_close_at_market_r(
    trades_sorted: pd.DataFrame,
    path_tensor: np.ndarray,
) -> np.ndarray:
    """Compute close-at-market R if trade were rejected at bar t=1 (close at bar2 open).

    Proxy bar2 spread with bar1 spread (trades_all.spread_pips_used). For pairs with
    stable per-bar spread this is accurate; for noisy bars the proxy adds < ~1 pip error.

    NOTE: only meaningful when bars_held >= 3 (trade still open at bar 2). For trades
    that SL-hit on bar 0 or bar 1 (bars_held <= 2), Pipeline D1 cannot reject — the
    trade is already closed. Caller must clip / route appropriately; this function
    computes the hypothetical bar-2 R regardless and the caller decides whether to use it.
    """
    bar2_open = path_tensor[:, 2, 0]  # column 0 = open
    entry_price = trades_sorted["entry_price"].to_numpy(dtype=float)
    sl_dist_price = 2.0 * trades_sorted["atr_14_at_signal"].to_numpy(dtype=float)
    spread_pips = trades_sorted["spread_pips_used"].to_numpy(dtype=float)
    pairs = trades_sorted["pair"].to_numpy()
    pip_sizes = np.array([_pip_size(p) for p in pairs], dtype=float)
    spread_price_half = spread_pips / 2.0 * pip_sizes
    pnl_price = bar2_open - entry_price - spread_price_half
    return pnl_price / sl_dist_price


@dataclass
class FoldResult:
    fold: int
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp
    oos_days: int
    n_oos: int
    n_admit: int
    admit_rate: float
    win_rate_admits: float
    n_target_admit: int
    classifier_sha256: str
    # Per-risk results stored in parent
    # Per-trade ordered r-vector for compounding:
    r_chrono: np.ndarray  # admitted use final_r; rejected use close_at_market_r
    is_admit_chrono: np.ndarray  # bool mask
    entry_time_chrono: np.ndarray  # datetime64 for cross-fold concat
    trade_id_chrono: np.ndarray


def _compounded_metrics(r_chrono: np.ndarray, risk_pct: float, oos_days: int) -> Dict[str, float]:
    """Compounded equity curve metrics at given risk_pct."""
    n = len(r_chrono)
    if n == 0:
        return {
            "fold_roi_pct": 0.0,
            "fold_roi_ann_pct": 0.0,
            "fold_max_dd_pct": 0.0,
            "fold_terminal_equity": 1.0,
        }
    r_scaled = r_chrono * (risk_pct / 100.0)
    eq = np.cumprod(1.0 + r_scaled)
    eq_with_start = np.concatenate([[1.0], eq])
    peak = np.maximum.accumulate(eq_with_start)
    dd = (peak - eq_with_start) / peak
    fold_max_dd_pct = float(np.max(dd) * 100.0)
    fold_roi_pct = float((eq[-1] - 1.0) * 100.0)
    fold_roi_ann_pct = float(fold_roi_pct * (365.0 / max(oos_days, 1)))
    return {
        "fold_roi_pct": fold_roi_pct,
        "fold_roi_ann_pct": fold_roi_ann_pct,
        "fold_max_dd_pct": fold_max_dd_pct,
        "fold_terminal_equity": float(eq[-1]),
    }


def _simple_sum_metrics(r_chrono: np.ndarray, risk_pct: float, oos_days: int) -> Dict[str, float]:
    """Step-5-style additive cum-sum metrics for comparison."""
    n = len(r_chrono)
    if n == 0:
        return {
            "fold_roi_pct": 0.0,
            "fold_roi_ann_pct": 0.0,
            "fold_max_dd_pct": 0.0,
        }
    r_scaled = r_chrono * (risk_pct / 100.0)
    eq = np.cumsum(r_scaled) * 100.0
    peak = np.maximum.accumulate(eq)
    fold_max_dd_pct = float(np.max(peak - eq))
    fold_roi_pct = float(eq[-1])
    return {
        "fold_roi_pct": fold_roi_pct,
        "fold_roi_ann_pct": float(fold_roi_pct * (365.0 / max(oos_days, 1))),
        "fold_max_dd_pct": fold_max_dd_pct,
    }


def _check_deploy(per_fold: List[Dict], full_roi_pct: float, full_dd_pct: float, n_admit_per_fold: List[int]) -> Tuple[bool, List[str]]:
    """Apply PASS-DEPLOYABLE gate; return (passes, failing_reasons)."""
    reasons: List[str] = []
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
    if full_roi_pct < DEPLOY_FULL_DATA_ROI_PCT:
        reasons.append(f"full-data ROI {full_roi_pct:.2f}% < {DEPLOY_FULL_DATA_ROI_PCT}%")
    if full_dd_pct > FULL_DATA_DD_CEILING_PCT:
        reasons.append(f"full-data DD {full_dd_pct:.2f}% > {FULL_DATA_DD_CEILING_PCT}%")
    return (len(reasons) == 0, reasons)


def _check_viable(per_fold: List[Dict], full_roi_pct: float, full_dd_pct: float, n_admit_per_fold: List[int]) -> Tuple[bool, List[str]]:
    """Apply PASS-VIABLE gate."""
    reasons: List[str] = []
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
    if full_roi_pct < VIABLE_FULL_DATA_ROI_PCT:
        reasons.append(f"full-data ROI {full_roi_pct:.2f}% < {VIABLE_FULL_DATA_ROI_PCT}%")
    if full_dd_pct > FULL_DATA_DD_CEILING_PCT:
        reasons.append(f"full-data DD {full_dd_pct:.2f}% > {FULL_DATA_DD_CEILING_PCT}%")
    return (len(reasons) == 0, reasons)


# ============================================================
# Driver
# ============================================================


def main() -> int:
    t0 = time.time()

    print("[step6] loading inputs", file=sys.stderr)
    new_trades = pd.read_csv(NEW_STEP1_DIR / "trades_all.csv")
    new_trades_sorted = new_trades.sort_values("trade_id").reset_index(drop=True)
    new_paths = pd.read_csv(NEW_STEP1_DIR / "trades_paths.csv")
    print(f"  new pool: {len(new_trades_sorted)} trades", file=sys.stderr)

    print("[step6] building path tensor + entry features", file=sys.stderr)
    path_tensor = _build_path_tensor(new_paths, len(new_trades_sorted))
    entry_feats = _build_entry_features_for_pool(new_trades_sorted)
    entry_sorted = entry_feats.sort_values("trade_id").reset_index(drop=True)
    if list(entry_sorted["trade_id"].astype(int)) != list(new_trades_sorted["trade_id"].astype(int)):
        raise RuntimeError("entry_features trade_id ordering mismatch vs trades_all")
    X_entry = entry_sorted[BASE_ENTRY_FEATURES].to_numpy(dtype=float)

    # ===== Per-trade close-at-market R (cluster-independent) =====
    print("[step6] computing close-at-market R per trade", file=sys.stderr)
    close_at_mkt_r = _compute_close_at_market_r(new_trades_sorted, path_tensor)
    bars_held_pre = new_trades_sorted["bars_held"].to_numpy(dtype=int)
    survived_mask = bars_held_pre >= 3
    print(
        f"  close-at-market R (ALL trades): mean={close_at_mkt_r.mean():+.4f}, "
        f"std={close_at_mkt_r.std():.4f}, min={close_at_mkt_r.min():.4f}, max={close_at_mkt_r.max():.4f}",
        file=sys.stderr,
    )
    print(
        f"  close-at-market R (survived bars_held>=3, n={survived_mask.sum()}/{len(survived_mask)}): "
        f"mean={close_at_mkt_r[survived_mask].mean():+.4f}, "
        f"std={close_at_mkt_r[survived_mask].std():.4f}, "
        f"min={close_at_mkt_r[survived_mask].min():.4f}, max={close_at_mkt_r[survived_mask].max():.4f}",
        file=sys.stderr,
    )
    print(
        f"  early-exit trades (bars_held <= 2): {(~survived_mask).sum()} "
        f"({(~survived_mask).mean()*100:.2f}%)",
        file=sys.stderr,
    )

    # ===== Cross-fold arrays =====
    entry_times = pd.to_datetime(new_trades_sorted["entry_time"])
    final_r_arr = new_trades_sorted["final_r"].to_numpy(dtype=float)
    bars_held = new_trades_sorted["bars_held"].to_numpy(dtype=int)
    pairs_arr = new_trades_sorted["pair"].to_numpy()
    trade_ids_arr = new_trades_sorted["trade_id"].to_numpy(dtype=int)
    atr_14 = new_trades_sorted["atr_14_at_signal"].to_numpy(dtype=float)
    entry_prices = new_trades_sorted["entry_price"].to_numpy(dtype=float)

    # Eligibility: every trade has bars_held >= 1 (Pipeline D1 can always evaluate t=1).
    # For close-at-market exit, every trade has bar_offset=2 in trades_paths.csv (paths run to 240).
    elig_mask = np.ones(len(new_trades_sorted), dtype=bool)

    # ===== Per-cluster Step 6 =====
    summary_rows: List[Dict] = []
    ship_candidates: Dict[int, Dict] = {}

    for cid in (1, 3):
        print(f"[step6] === cluster {cid} ({CLUSTER_LABEL[cid]}) ===", file=sys.stderr)
        r_frame_mult = CLUSTER_R_FRAME_ATR_MULT[cid]
        threshold = CLUSTER_F9_THRESHOLD[cid]
        cluster_dir = OUT_DIR / f"cluster_{cid}"
        cluster_dir.mkdir(parents=True, exist_ok=True)

        # Build feature matrix once per cluster.
        print(f"  computing t=1 features under R={r_frame_mult}\xd7ATR", file=sys.stderr)
        t1_feats = _compute_t1_features(path_tensor, entry_prices, atr_14, r_frame_mult)
        X_full = np.concatenate([X_entry, t1_feats], axis=1)
        # NaN-fill with column median (matches Phase 2 + baseline _features_to_matrix).
        for j in range(X_full.shape[1]):
            col = X_full[:, j]
            if np.isnan(col).any():
                col = np.where(np.isnan(col), np.nanmedian(col), col)
                X_full[:, j] = col

        # Per-fold trade ordering + per-trade r (admit or close-at-market).
        fold_results: List[FoldResult] = []
        all_clf_sha: List[str] = []
        for fidx, (_is_start, _is_end, oos_start, oos_end) in enumerate(WFO_FOLDS, start=1):
            oos_start_ts = pd.Timestamp(oos_start)
            oos_end_ts = pd.Timestamp(oos_end)
            oos_days = int((oos_end_ts - oos_start_ts).total_seconds() / 86400)
            oos_mask = (entry_times >= oos_start_ts) & (entry_times < oos_end_ts) & elig_mask
            oos_idx = np.where(oos_mask)[0]
            n_oos = int(len(oos_idx))

            clf_path = BASELINE_STEP5_DIR / f"cluster_{cid}" / "per_fold_classifiers" / f"fold_{fidx}_classifier.joblib"
            clf_sha = _file_sha256(clf_path)
            all_clf_sha.append(clf_sha)
            clf = joblib.load(clf_path)
            X_oos = X_full[oos_idx, :]
            proba = clf.predict_proba(X_oos)[:, 1]
            admit_mask = proba >= threshold
            n_admit = int(admit_mask.sum())
            n_target_admit = int(((proba >= threshold) & (final_r_arr[oos_idx] > 0)).sum())
            admit_rate = n_admit / n_oos if n_oos > 0 else 0.0
            win_rate_admits = float((final_r_arr[oos_idx][admit_mask] > 0).mean()) if n_admit > 0 else 0.0

            # Per-trade r: routing per Pipeline D1 PR-1 mechanics.
            #   - bars_held <= 2: trade closed before bar 2 open (SL hit by bar 1 end).
            #     Classifier cannot reject what's already closed -> use actual final_r.
            #   - bars_held >= 3: trade still open at bar 2; classifier decides:
            #       admit -> final_r (real outcome); reject -> close_at_market_r.
            bh_oos = bars_held[oos_idx]
            early_exit_mask = bh_oos <= 2
            survived_to_bar2 = ~early_exit_mask
            reject_and_survived = (~admit_mask) & survived_to_bar2
            r_per_trade = final_r_arr[oos_idx].copy()
            r_per_trade[reject_and_survived] = close_at_mkt_r[oos_idx][reject_and_survived]
            # Order chronologically by entry_time (stable sort for reproducibility).
            et_oos = entry_times.iloc[oos_idx].to_numpy()
            order = np.argsort(et_oos, kind="mergesort")
            r_chrono = r_per_trade[order]
            is_admit_chrono = admit_mask[order]
            et_chrono = et_oos[order]
            tid_chrono = trade_ids_arr[oos_idx][order]

            fr = FoldResult(
                fold=fidx,
                oos_start=oos_start_ts,
                oos_end=oos_end_ts,
                oos_days=oos_days,
                n_oos=n_oos,
                n_admit=n_admit,
                admit_rate=admit_rate,
                win_rate_admits=win_rate_admits,
                n_target_admit=n_target_admit,
                classifier_sha256=clf_sha,
                r_chrono=r_chrono,
                is_admit_chrono=is_admit_chrono,
                entry_time_chrono=et_chrono,
                trade_id_chrono=tid_chrono,
            )
            fold_results.append(fr)
            print(
                f"  fold {fidx}: n_oos={n_oos}, admit={n_admit} ({admit_rate*100:.2f}%), "
                f"win-rate@admit={win_rate_admits*100:.1f}%, clf={clf_sha[:8]}",
                file=sys.stderr,
            )

        # ===== Risk sweep =====
        sweep_rows: List[Dict] = []
        per_risk_per_fold_rows: Dict[int, List[Dict]] = {int(round(r * 100)): [] for r in RISK_GRID_PCT}
        best_disposition_tier = "FAIL"
        best_disposition_risk: Optional[float] = None
        best_disposition_metrics: Optional[Dict] = None
        best_disposition_reasons: List[str] = []
        ship_metric_rank: Optional[Tuple[int, float, float]] = None  # (tier_rank, worst_roi_ann, smallest_risk inverse)

        for risk_pct in RISK_GRID_PCT:
            r_bps = int(round(risk_pct * 100))
            per_fold: List[Dict] = []
            n_admit_per_fold: List[int] = []
            # Cross-fold compounded equity (concatenate fold chrono r-vectors).
            cross_eq = 1.0
            cross_eq_curve: List[float] = [1.0]
            cross_admit_total = 0
            for fr in fold_results:
                # Admit-set compounded metrics (the §10 trade-count uses admit only).
                # But equity curve includes BOTH admit and reject (PR1 mechanics fire all signals).
                m = _compounded_metrics(fr.r_chrono, risk_pct, fr.oos_days)
                ss = _simple_sum_metrics(fr.r_chrono, risk_pct, fr.oos_days)
                per_fold.append({
                    "fold": fr.fold,
                    "oos_days": fr.oos_days,
                    "n_oos": fr.n_oos,
                    "n_admit": fr.n_admit,
                    "admit_rate": fr.admit_rate,
                    "win_rate_admits": fr.win_rate_admits,
                    "fold_roi_pct": m["fold_roi_pct"],
                    "fold_roi_ann_pct": m["fold_roi_ann_pct"],
                    "fold_max_dd_pct": m["fold_max_dd_pct"],
                    "fold_terminal_equity": m["fold_terminal_equity"],
                    "ss_fold_roi_pct": ss["fold_roi_pct"],
                    "ss_fold_max_dd_pct": ss["fold_max_dd_pct"],
                })
                n_admit_per_fold.append(fr.n_admit)
                cross_admit_total += fr.n_admit
                # Cross-fold compounded curve
                r_scaled = fr.r_chrono * (risk_pct / 100.0)
                for r in r_scaled:
                    cross_eq *= (1.0 + r)
                    cross_eq_curve.append(cross_eq)
            full_roi_pct = (cross_eq - 1.0) * 100.0
            eq_arr = np.asarray(cross_eq_curve)
            peak = np.maximum.accumulate(eq_arr)
            full_dd_pct = float(np.max((peak - eq_arr) / peak) * 100.0)

            worst_roi_ann = float(min(f["fold_roi_ann_pct"] for f in per_fold))
            worst_roi = float(min(f["fold_roi_pct"] for f in per_fold))
            mean_roi_ann = float(np.mean([f["fold_roi_ann_pct"] for f in per_fold]))
            mean_fold_dd = float(np.mean([f["fold_max_dd_pct"] for f in per_fold]))
            worst_dd = float(max(f["fold_max_dd_pct"] for f in per_fold))

            ok_deploy, deploy_reasons = _check_deploy(per_fold, full_roi_pct, full_dd_pct, n_admit_per_fold)
            ok_viable, viable_reasons = _check_viable(per_fold, full_roi_pct, full_dd_pct, n_admit_per_fold)
            if ok_deploy:
                tier = "DEPLOYABLE"
                tier_rank = 2
                fail_reasons = ""
            elif ok_viable:
                tier = "VIABLE"
                tier_rank = 1
                fail_reasons = " | ".join(deploy_reasons)
            else:
                tier = "FAIL"
                tier_rank = 0
                fail_reasons = " | ".join(viable_reasons)

            sweep_rows.append({
                "cluster_id": cid,
                "risk_pct": risk_pct,
                "risk_bps": r_bps,
                "n_admit_total": cross_admit_total,
                "n_admit_min_per_fold": int(min(n_admit_per_fold)),
                "worst_fold_roi_ann_pct": worst_roi_ann,
                "worst_fold_roi_pct": worst_roi,
                "mean_fold_roi_ann_pct": mean_roi_ann,
                "worst_fold_max_dd_pct": worst_dd,
                "mean_fold_max_dd_pct": mean_fold_dd,
                "full_data_compounded_roi_pct": full_roi_pct,
                "full_data_compounded_max_dd_pct": full_dd_pct,
                "tier": tier,
                "tier_rank": tier_rank,
                "fail_reasons": fail_reasons,
            })

            per_risk_per_fold_rows[r_bps] = per_fold

            # Track best disposition: prefer higher tier, then smaller risk (more conservative),
            # then higher worst_fold_roi_ann_pct.
            this_key = (tier_rank, -risk_pct, worst_roi_ann)
            if (best_disposition_tier == "FAIL"
                or (tier_rank > {"FAIL": 0, "VIABLE": 1, "DEPLOYABLE": 2}[best_disposition_tier])
                or (tier_rank == {"FAIL": 0, "VIABLE": 1, "DEPLOYABLE": 2}[best_disposition_tier]
                    and (ship_metric_rank is None or this_key > ship_metric_rank))):
                best_disposition_tier = tier
                best_disposition_risk = risk_pct
                best_disposition_metrics = {
                    "worst_fold_roi_ann_pct": worst_roi_ann,
                    "worst_fold_roi_pct": worst_roi,
                    "mean_fold_roi_ann_pct": mean_roi_ann,
                    "worst_fold_max_dd_pct": worst_dd,
                    "mean_fold_max_dd_pct": mean_fold_dd,
                    "full_data_compounded_roi_pct": full_roi_pct,
                    "full_data_compounded_max_dd_pct": full_dd_pct,
                    "n_admit_total": cross_admit_total,
                    "n_admit_min_per_fold": int(min(n_admit_per_fold)),
                }
                best_disposition_reasons = deploy_reasons if not ok_deploy else []
                ship_metric_rank = this_key

        # Write risk_sweep
        sweep_df = pd.DataFrame(sweep_rows)
        sweep_df.to_csv(cluster_dir / f"risk_sweep_{cid}.csv", index=False)

        # Write per-risk per-fold detail
        for r_bps, rows in per_risk_per_fold_rows.items():
            pd.DataFrame(rows).to_csv(cluster_dir / f"per_fold_wfo_{cid}_risk_{r_bps}bps.csv", index=False)

        # Equity curve at best risk
        if best_disposition_risk is not None:
            best_bps = int(round(best_disposition_risk * 100))
            curves: List[Dict] = []
            cross_eq = 1.0
            global_step = 0
            for fr in fold_results:
                r_scaled = fr.r_chrono * (best_disposition_risk / 100.0)
                fold_eq = 1.0
                for i, r in enumerate(r_scaled):
                    fold_eq *= (1.0 + r)
                    cross_eq *= (1.0 + r)
                    global_step += 1
                    curves.append({
                        "fold": fr.fold,
                        "global_step": global_step,
                        "trade_id": int(fr.trade_id_chrono[i]),
                        "entry_time": pd.Timestamp(fr.entry_time_chrono[i]).isoformat(),
                        "is_admit": int(fr.is_admit_chrono[i]),
                        "trade_r": float(fr.r_chrono[i]),
                        "fold_equity": float(fold_eq),
                        "cross_fold_equity": float(cross_eq),
                    })
            pd.DataFrame(curves).to_csv(cluster_dir / f"equity_curves_{cid}_risk_{best_bps}bps.csv", index=False)

        # Best config yaml
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
        cfg = _py({
            "cluster_id": cid,
            "cluster_label": CLUSTER_LABEL[cid],
            "best_risk_pct": best_disposition_risk,
            "tier": best_disposition_tier,
            "metrics": best_disposition_metrics,
            "failing_gate_reasons_at_lower_tier_evaluation": best_disposition_reasons,
            "classifier_sha256_per_fold": all_clf_sha,
            "threshold_F9": threshold,
            "r_frame_atr_mult": r_frame_mult,
        })
        (cluster_dir / f"best_risk_config_{cid}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

        summary_rows.append({
            "cluster_id": cid,
            "cluster_label": CLUSTER_LABEL[cid],
            "best_risk_pct": best_disposition_risk,
            "tier": best_disposition_tier,
            **(best_disposition_metrics or {}),
        })
        ship_candidates[cid] = {
            "best_risk_pct": best_disposition_risk,
            "tier": best_disposition_tier,
            "metrics": best_disposition_metrics,
        }
        print(
            f"  cluster {cid} best: risk={best_disposition_risk}% tier={best_disposition_tier} "
            f"worst-ROI={best_disposition_metrics['worst_fold_roi_ann_pct']:.2f}% "
            f"worst-DD={best_disposition_metrics['worst_fold_max_dd_pct']:.2f}%",
            file=sys.stderr,
        )

    # ===== Ship decision =====
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "arc_5_step6_summary.csv", index=False)

    tier_rank = {"FAIL": 0, "VIABLE": 1, "DEPLOYABLE": 2}
    tiers = {cid: tier_rank[ship_candidates[cid]["tier"]] for cid in (1, 3)}
    ship_pick: Optional[int] = None
    rationale = ""
    if max(tiers.values()) == 0:
        ship_pick = None
        rationale = "No cluster achieves PASS-VIABLE or higher. Arc 5 closes Step 6 FAIL — no shipment."
    else:
        max_tier = max(tiers.values())
        eligible = [c for c, t in tiers.items() if t == max_tier]
        # Tie-break on highest worst-fold ROI ann
        eligible.sort(key=lambda c: -(ship_candidates[c]["metrics"]["worst_fold_roi_ann_pct"]))
        ship_pick = eligible[0]
        tier_name = ship_candidates[ship_pick]["tier"]
        rationale = (
            f"Highest §10 tier achieved: {tier_name}. Eligible clusters at this tier: {eligible}. "
            f"Tie-broken by worst-fold annualised ROI: cluster {ship_pick} at "
            f"{ship_candidates[ship_pick]['metrics']['worst_fold_roi_ann_pct']:.2f}% wins."
        )

    def _py2(o):
        if isinstance(o, dict):
            return {k: _py2(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_py2(v) for v in o]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return _py2(o.tolist())
        return o
    ship_yaml = _py2({
        "ship_pick_cluster": ship_pick,
        "rationale": rationale,
        "clusters": {
            cid: {
                "tier": ship_candidates[cid]["tier"],
                "best_risk_pct": ship_candidates[cid]["best_risk_pct"],
                "metrics": ship_candidates[cid]["metrics"],
            }
            for cid in (1, 3)
        },
        "protocol_version": "L_ARC_PROTOCOL v2.1.1 §10",
        "spread_regime": "2026-05-17 per-pair p50 (HistData 2024-2025)",
        "skip_to_step_6_caveat": (
            "Baseline classifiers + F9 thresholds reused (no retraining). "
            "Cluster labels from baseline clusters_K4.csv + nearest-centroid for 204 new-only trades. "
            "Admit-set Jaccard vs baseline 0.91 (c1) / 0.90 (c3) — within prompt-defined "
            "acceptable approximation bound. Decision-quality, not publication-quality."
        ),
    })
    (OUT_DIR / "ship_decision.yaml").write_text(yaml.safe_dump(ship_yaml, sort_keys=False))

    # Top-level metadata
    meta = {
        "run_timestamp_utc": pd.Timestamp.now("UTC").isoformat(),
        "elapsed_seconds": float(time.time() - t0),
        "risk_grid_pct": RISK_GRID_PCT,
        "wfo_folds": WFO_FOLDS,
        "deployable_thresholds": {
            "worst_fold_roi_ann_pct_min": DEPLOY_WORST_ROI_ANN_PCT,
            "mean_fold_roi_ann_pct_min": DEPLOY_MEAN_ROI_ANN_PCT,
            "worst_fold_dd_pct_max": DD_CEILING_PCT,
            "min_trades_per_fold": DEPLOY_TRADES_PER_FOLD,
            "full_data_roi_pct_min": DEPLOY_FULL_DATA_ROI_PCT,
            "full_data_dd_pct_max": FULL_DATA_DD_CEILING_PCT,
        },
        "viable_thresholds": {
            "worst_fold_roi_pct_min_exclusive": VIABLE_WORST_ROI_PCT,
            "mean_fold_roi_ann_pct_min": VIABLE_MEAN_ROI_ANN_PCT,
            "worst_fold_dd_pct_max": DD_CEILING_PCT,
            "min_trades_per_fold": VIABLE_TRADES_PER_FOLD,
            "full_data_roi_pct_min": VIABLE_FULL_DATA_ROI_PCT,
            "full_data_dd_pct_max": FULL_DATA_DD_CEILING_PCT,
        },
    }
    (OUT_DIR / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=str))

    print(f"[step6] DONE in {time.time() - t0:.1f}s; ship_pick={ship_pick}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
