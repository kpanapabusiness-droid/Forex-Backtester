"""Arc 8 — Step 5 Walk-Forward Optimisation on c1 V-shape recovery (FG-weak).

L_ARC_PROTOCOL v2.3 §10 multi-pipeline ship rule. Runs WFO independently
on:
  1. (c1, Pipeline E) — RF classifier, 5 features at entry, threshold 0.70
  2. (c1, Pipeline D1) — RF classifier, 15 features at t=1, threshold 0.60

Both archetype label = "V-shape recovery (forward-geometry weak)".
Selected SL throughout: 4.0×ATR_4H (entry-anchored).
pre_t_sl_atr_multiplier (D1): 4.0 (engine PR feat/open-24-pre-t-sl-per-
archetype merged at 716ce84; consumed via core/d1_pipeline.py).

WFO windows: configs/wfo_kh24.yaml verbatim (7 anchored folds, IS expanding
from 2019-01-01, OOS 9mo each). Fold 1 IS window precedes Arc 8 data start
(c1 trades begin 2020-11) — fold 1 skipped as IS-empty. Folds 2-7 evaluated.

Two economics views computed for each fold (per protocol §10 + CLAUDE.md
cross-arc lesson from Arc 4 RERUN / Arc 5):
  - admit_only: classifier applied to c1 OOS trades only. Validates Step 4
    in different fold structure. NOT the §10 ship-gate basis.
  - full_pool: classifier applied to ALL Step 1 OOS trades. Reflects
    production deployment economics (we cannot pre-filter to c1 in
    real time — clusters are post-hoc). THIS is the §10 ship-gate basis.

Exit policy under this dispatch: PR 1 close-at-market path per protocol §13.
The §11 row 5 V-shape "standard trail" exit awaits PR 2 (per-archetype
exit policies). Under this WFO, all admitted trades run with default exit
(hard SL=4.0×ATR + time exit at bar 240). D1 rejects close at bar 2
mid (proxy for "next-bar open" close-at-market).

Position sizing: 0.5% risk per trade × starting balance 10000 USD with
reset-floor compounding (standard L arc convention).

Ship gates (v2.3 §10) checked per pipeline independently:
  - Worst-window max_dd ≤ 8%
  - Worst-window ROI > 0
  - Aggregate sharpe ≥ 0.8
  - n_oos_trades ≥ 30

Usage:
    py scripts/l_arc_8/step5_wfo.py -c configs/l_arc_8/step5.yaml
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
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

# Reuse Step 4 feature computation + SL re-evaluation.
from scripts.l_arc_8.step3_capturability import _eval_trade_at_sl  # noqa: E402
from scripts.l_arc_8.step4_extractability import (  # noqa: E402
    PIPELINE_E_BASE_FEATURES,
    _build_pair_cache,
    _impute_nans,
    _make_rf,
    compute_base_e_features,
    compute_d1_features_at_t,
)

# Locked from Step 4 closure (do not retune).
PIPELINE_E_LOCKED_FEATURES: List[str] = [
    "ret_5bar_atr",
    "pos_in_20bar_range",
    "pullback_depth_atr",
    "range_to_atr_14",
    "hl_range_atr",
]
PIPELINE_E_THRESHOLD: float = 0.70

PIPELINE_D1_T: int = 1
PIPELINE_D1_PATH_FEATURES: List[str] = [
    "close_r_at_t",
    "mfe_so_far_r_at_t",
    "mae_so_far_r_at_t",
    "bars_in_profit_at_t",
    "local_peaks_so_far_at_t",
    "monotonicity_so_far_at_t",
    "velocity_first_t",
]
PIPELINE_D1_THRESHOLD: float = 0.60

UNIT_SELECTED_SL: float = 4.0   # c1 SL=4.0×ATR
ORIGINAL_SL: float = 2.0        # Step 1 R-frame
SCALE_TO_UNIT: float = ORIGINAL_SL / UNIT_SELECTED_SL  # 0.5

STARTING_BALANCE: float = 10000.0
RISK_PCT: float = 0.005


# ============================================================
# Data loaders
# ============================================================


def _build_paths_index(paths_df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    out: Dict[int, pd.DataFrame] = {}
    paths_sorted = paths_df.sort_values(["trade_id", "bar_offset"], kind="mergesort")
    for tid, g in paths_sorted.groupby("trade_id", sort=True):
        out[int(tid)] = g.reset_index(drop=True)
    return out


# ============================================================
# PnL computation under unit SL
# ============================================================


def _pnl_admit(path: pd.DataFrame) -> Dict[str, Any]:
    """Compute final_r for a trade admitted at signal time (Pipeline E)
    or admitted at bar 1 (Pipeline D1) under SL=4.0×ATR, default exit
    (hard SL hit OR time exit at bar 240).
    """
    eval_ = _eval_trade_at_sl(path, UNIT_SELECTED_SL, ORIGINAL_SL)
    return {
        "admitted": True,
        "final_r": float(eval_.final_r_new),
        "exit_bar": int(eval_.truncated_at_bar),
        "sl_hit": bool(eval_.sl_hit),
    }


def _pnl_d1_reject(path: pd.DataFrame, t: int = PIPELINE_D1_T) -> Dict[str, Any]:
    """Pipeline D1 rejected at bar t — close at bar t+1 open (PR 1 close-at-market).

    Path data has close_r at each bar_offset (mid-bar close). The faithful
    approximation for "close at next-bar open" is bar t+1 close (since path
    data doesn't include opens). For t=1, exit_bar=2, exit price ≈ close_r[2].

    Returns the realised R in unit frame.
    """
    path_sorted = path.sort_values("bar_offset", kind="mergesort").reset_index(drop=True)
    exit_bar = t + 1
    if exit_bar >= len(path_sorted):
        # Edge case: trade has fewer than t+1 bars in path. Use last available.
        exit_bar = len(path_sorted) - 1
    exit_close_orig = float(path_sorted["close_r"].iloc[exit_bar])
    final_r_unit = exit_close_orig * SCALE_TO_UNIT
    # If SL would have hit during bars 0..t (before t+1 close), use SL.
    # SL threshold in original-R frame at unit SL = -1.0 / SCALE_TO_UNIT = -2.0
    sl_thresh_orig = -1.0 / SCALE_TO_UNIT
    low_r_to_exit = path_sorted["low_r"].iloc[: exit_bar + 1].to_numpy(dtype=float)
    if (low_r_to_exit <= sl_thresh_orig).any():
        # SL hit before close-at-market could execute → standard SL exit at -1R.
        sl_bar_offset = int(np.argmax(low_r_to_exit <= sl_thresh_orig))
        return {
            "admitted": False,
            "final_r": -1.0,
            "exit_bar": sl_bar_offset,
            "sl_hit": True,
            "rejected_close_at_market": False,
            "rejected_pre_empted_by_sl": True,
        }
    return {
        "admitted": False,
        "final_r": final_r_unit,
        "exit_bar": exit_bar,
        "sl_hit": False,
        "rejected_close_at_market": True,
        "rejected_pre_empted_by_sl": False,
    }


# ============================================================
# Per-fold metrics
# ============================================================


@dataclass
class _FoldResult:
    fold: int
    is_start: str
    is_end: str
    oos_start: str
    oos_end: str
    n_oos_signals: int          # all c1 OOS trades (before filter)
    n_admitted: int             # post-classifier admit (Pipeline E) or full trades (D1)
    n_d1_rejected: int          # D1-rejected (close at bar 2) — 0 for Pipeline E
    n_traded: int               # admitted + d1_rejected (D1 counts rejects as trades that opened)
    hit_rate: float
    mean_r: float
    std_r: float
    total_r: float
    sharpe_r: float
    starting_balance: float
    ending_balance: float
    roi_pct: float
    max_dd_pct: float
    max_dd_r: float
    n_pairs_active: int
    is_train_count: int
    skipped: bool = False
    skip_reason: str = ""
    per_trade_r: List[float] = field(default_factory=list)
    per_trade_balance: List[float] = field(default_factory=list)


def _equity_metrics(per_trade_r: List[float], starting_balance: float, risk_pct: float
                    ) -> Tuple[float, float, float, float, List[float]]:
    """Returns (ending_balance, roi_pct, max_dd_pct, max_dd_r, balance_curve)."""
    bal = starting_balance
    peak = bal
    max_dd_pct = 0.0
    bal_curve: List[float] = []
    max_dd_r = 0.0
    cum_r = 0.0
    peak_cum_r = 0.0
    for r in per_trade_r:
        dollar_risk = bal * risk_pct
        pnl = dollar_risk * r
        bal += pnl
        bal_curve.append(bal)
        peak = max(peak, bal)
        if peak > 0:
            dd_pct = (peak - bal) / peak * 100.0
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct
        cum_r += r
        peak_cum_r = max(peak_cum_r, cum_r)
        if (peak_cum_r - cum_r) > max_dd_r:
            max_dd_r = peak_cum_r - cum_r
    roi_pct = (bal - starting_balance) / starting_balance * 100.0
    return bal, roi_pct, max_dd_pct, max_dd_r, bal_curve


def _sharpe(per_trade_r: List[float]) -> float:
    if len(per_trade_r) < 2:
        return float("nan")
    arr = np.array(per_trade_r, dtype=float)
    std = float(arr.std(ddof=1))
    if std == 0:
        return float("nan")
    return float(arr.mean() / std)


# ============================================================
# Pipeline E WFO
# ============================================================


def run_pipeline_e_wfo(
    c1_trades: pd.DataFrame,
    base_e_df: pd.DataFrame,
    paths_index: Dict[int, pd.DataFrame],
    wfo_folds: List[Dict[str, Any]],
    threshold: float = PIPELINE_E_THRESHOLD,
    is_train_min: int = 30,
    oos_pool_trades: Optional[pd.DataFrame] = None,
    oos_pool_base_e: Optional[pd.DataFrame] = None,
) -> List[_FoldResult]:
    """Pipeline E: retrain RF per fold, admit if prob >= threshold,
    compute PnL under SL=4.0×ATR + default exit.

    If oos_pool_trades is given (full Step 1 pool), classifier is applied
    to that pool's OOS subset (production deployment economics). Otherwise
    classifier is applied to c1's OOS subset (admit-only economics).
    """
    results: List[_FoldResult] = []
    use_full_pool = oos_pool_trades is not None
    for fold_cfg in wfo_folds:
        fold_num = int(fold_cfg["fold"])
        is_start = pd.Timestamp(fold_cfg["is_start"])
        is_end = pd.Timestamp(fold_cfg["is_end"])
        oos_start = pd.Timestamp(fold_cfg["oos_start"])
        oos_end = pd.Timestamp(fold_cfg["oos_end"])

        is_mask = (c1_trades["entry_time"] >= is_start) & (c1_trades["entry_time"] < is_end)
        is_trades = c1_trades[is_mask].copy()
        if use_full_pool:
            oos_mask = (oos_pool_trades["entry_time"] >= oos_start) & (oos_pool_trades["entry_time"] < oos_end)
            oos_trades = oos_pool_trades[oos_mask].copy()
            oos_base_e_src = oos_pool_base_e
        else:
            oos_mask = (c1_trades["entry_time"] >= oos_start) & (c1_trades["entry_time"] < oos_end)
            oos_trades = c1_trades[oos_mask].copy()
            oos_base_e_src = base_e_df

        if len(is_trades) < is_train_min or len(oos_trades) == 0:
            results.append(
                _FoldResult(
                    fold=fold_num, is_start=str(is_start.date()), is_end=str(is_end.date()),
                    oos_start=str(oos_start.date()), oos_end=str(oos_end.date()),
                    n_oos_signals=int(len(oos_trades)),
                    n_admitted=0, n_d1_rejected=0, n_traded=0,
                    hit_rate=float("nan"), mean_r=float("nan"), std_r=float("nan"),
                    total_r=0.0, sharpe_r=float("nan"),
                    starting_balance=STARTING_BALANCE, ending_balance=STARTING_BALANCE,
                    roi_pct=0.0, max_dd_pct=0.0, max_dd_r=0.0, n_pairs_active=0,
                    is_train_count=int(len(is_trades)),
                    skipped=True,
                    skip_reason=(
                        f"IS train count {len(is_trades)} < {is_train_min} or OOS empty"
                    ),
                )
            )
            continue

        # Build labels under unit SL for IS.
        is_labels = []
        for _, tr in is_trades.iterrows():
            path = paths_index[int(tr["trade_id"])]
            ev = _eval_trade_at_sl(path, UNIT_SELECTED_SL, ORIGINAL_SL)
            is_labels.append(1 if ev.final_r_new >= 1.0 else 0)
        y_is = np.array(is_labels, dtype=int)

        # Features.
        is_feat = base_e_df.merge(
            is_trades[["trade_id"]], on="trade_id", how="inner"
        )
        is_feat = is_feat.set_index("trade_id").loc[is_trades["trade_id"].to_numpy()].reset_index()
        X_is = _impute_nans(is_feat[PIPELINE_E_LOCKED_FEATURES])

        if len(np.unique(y_is)) < 2:
            results.append(
                _FoldResult(
                    fold=fold_num, is_start=str(is_start.date()), is_end=str(is_end.date()),
                    oos_start=str(oos_start.date()), oos_end=str(oos_end.date()),
                    n_oos_signals=int(len(oos_trades)),
                    n_admitted=0, n_d1_rejected=0, n_traded=0,
                    hit_rate=float("nan"), mean_r=float("nan"), std_r=float("nan"),
                    total_r=0.0, sharpe_r=float("nan"),
                    starting_balance=STARTING_BALANCE, ending_balance=STARTING_BALANCE,
                    roi_pct=0.0, max_dd_pct=0.0, max_dd_r=0.0, n_pairs_active=0,
                    is_train_count=int(len(is_trades)),
                    skipped=True, skip_reason="IS single-class labels",
                )
            )
            continue

        # Train RF.
        clf = _make_rf()
        clf.fit(X_is, y_is)

        # Apply on OOS.
        oos_feat = oos_base_e_src.merge(
            oos_trades[["trade_id"]], on="trade_id", how="inner"
        )
        oos_feat = oos_feat.set_index("trade_id").loc[oos_trades["trade_id"].to_numpy()].reset_index()
        X_oos = _impute_nans(oos_feat[PIPELINE_E_LOCKED_FEATURES])
        probs = clf.predict_proba(X_oos)[:, 1]
        admit_mask = probs >= threshold

        per_trade_r: List[float] = []
        pairs_active = set()
        for ok, (_, tr) in zip(admit_mask, oos_trades.iterrows()):
            if not ok:
                continue
            path = paths_index[int(tr["trade_id"])]
            result = _pnl_admit(path)
            per_trade_r.append(result["final_r"])
            pairs_active.add(str(tr["pair"]))

        n_admitted = int(admit_mask.sum())
        n_oos_signals = int(len(oos_trades))
        if per_trade_r:
            arr = np.array(per_trade_r, dtype=float)
            mean_r = float(arr.mean())
            std_r = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
            total_r = float(arr.sum())
            hit_rate = float((arr > 0).mean())
        else:
            mean_r = 0.0
            std_r = 0.0
            total_r = 0.0
            hit_rate = 0.0

        ending_bal, roi_pct, max_dd_pct, max_dd_r, bal_curve = _equity_metrics(
            per_trade_r, STARTING_BALANCE, RISK_PCT
        )
        sharpe = _sharpe(per_trade_r)

        results.append(
            _FoldResult(
                fold=fold_num, is_start=str(is_start.date()), is_end=str(is_end.date()),
                oos_start=str(oos_start.date()), oos_end=str(oos_end.date()),
                n_oos_signals=n_oos_signals,
                n_admitted=n_admitted, n_d1_rejected=0, n_traded=n_admitted,
                hit_rate=hit_rate, mean_r=mean_r, std_r=std_r,
                total_r=total_r, sharpe_r=sharpe,
                starting_balance=STARTING_BALANCE, ending_balance=ending_bal,
                roi_pct=roi_pct, max_dd_pct=max_dd_pct, max_dd_r=max_dd_r,
                n_pairs_active=len(pairs_active),
                is_train_count=int(len(is_trades)),
                per_trade_r=per_trade_r, per_trade_balance=bal_curve,
            )
        )
    return results


# ============================================================
# Pipeline D1 WFO
# ============================================================


def run_pipeline_d1_wfo(
    c1_trades: pd.DataFrame,
    base_e_df: pd.DataFrame,
    paths_index: Dict[int, pd.DataFrame],
    wfo_folds: List[Dict[str, Any]],
    threshold: float = PIPELINE_D1_THRESHOLD,
    t: int = PIPELINE_D1_T,
    is_train_min: int = 30,
    oos_pool_trades: Optional[pd.DataFrame] = None,
    oos_pool_base_e: Optional[pd.DataFrame] = None,
) -> List[_FoldResult]:
    """Pipeline D1: every signal enters; at bar t classify; if prob >= threshold
    continue under default exit; else close at bar t+1 open (PR 1 path).

    Full-pool mode (oos_pool_trades given) applies the classifier to all
    Step 1 OOS trades, not just c1's OOS subset.
    """
    results: List[_FoldResult] = []
    use_full_pool = oos_pool_trades is not None
    for fold_cfg in wfo_folds:
        fold_num = int(fold_cfg["fold"])
        is_start = pd.Timestamp(fold_cfg["is_start"])
        is_end = pd.Timestamp(fold_cfg["is_end"])
        oos_start = pd.Timestamp(fold_cfg["oos_start"])
        oos_end = pd.Timestamp(fold_cfg["oos_end"])

        is_mask = (c1_trades["entry_time"] >= is_start) & (c1_trades["entry_time"] < is_end)
        is_trades = c1_trades[is_mask].copy()
        if use_full_pool:
            oos_mask = (oos_pool_trades["entry_time"] >= oos_start) & (oos_pool_trades["entry_time"] < oos_end)
            oos_trades = oos_pool_trades[oos_mask].copy()
            oos_base_e_src = oos_pool_base_e
        else:
            oos_mask = (c1_trades["entry_time"] >= oos_start) & (c1_trades["entry_time"] < oos_end)
            oos_trades = c1_trades[oos_mask].copy()
            oos_base_e_src = base_e_df

        if len(is_trades) < is_train_min or len(oos_trades) == 0:
            results.append(
                _FoldResult(
                    fold=fold_num, is_start=str(is_start.date()), is_end=str(is_end.date()),
                    oos_start=str(oos_start.date()), oos_end=str(oos_end.date()),
                    n_oos_signals=int(len(oos_trades)),
                    n_admitted=0, n_d1_rejected=0, n_traded=0,
                    hit_rate=float("nan"), mean_r=float("nan"), std_r=float("nan"),
                    total_r=0.0, sharpe_r=float("nan"),
                    starting_balance=STARTING_BALANCE, ending_balance=STARTING_BALANCE,
                    roi_pct=0.0, max_dd_pct=0.0, max_dd_r=0.0, n_pairs_active=0,
                    is_train_count=int(len(is_trades)),
                    skipped=True,
                    skip_reason=f"IS train count {len(is_trades)} < {is_train_min} or OOS empty",
                )
            )
            continue

        # IS labels under unit SL.
        is_labels = []
        for _, tr in is_trades.iterrows():
            path = paths_index[int(tr["trade_id"])]
            ev = _eval_trade_at_sl(path, UNIT_SELECTED_SL, ORIGINAL_SL)
            is_labels.append(1 if ev.final_r_new >= 1.0 else 0)
        y_is = np.array(is_labels, dtype=int)

        # D1 features at t for IS + OOS.
        is_d1_feat, is_eligible = compute_d1_features_at_t(
            is_trades, paths_index, t, UNIT_SELECTED_SL, ORIGINAL_SL
        )
        if not is_eligible.any() or len(np.unique(y_is[is_eligible])) < 2:
            results.append(
                _FoldResult(
                    fold=fold_num, is_start=str(is_start.date()), is_end=str(is_end.date()),
                    oos_start=str(oos_start.date()), oos_end=str(oos_end.date()),
                    n_oos_signals=int(len(oos_trades)),
                    n_admitted=0, n_d1_rejected=0, n_traded=0,
                    hit_rate=float("nan"), mean_r=float("nan"), std_r=float("nan"),
                    total_r=0.0, sharpe_r=float("nan"),
                    starting_balance=STARTING_BALANCE, ending_balance=STARTING_BALANCE,
                    roi_pct=0.0, max_dd_pct=0.0, max_dd_r=0.0, n_pairs_active=0,
                    is_train_count=int(len(is_trades)),
                    skipped=True, skip_reason="IS D1 eligibility or single-class",
                )
            )
            continue

        is_full = base_e_df.merge(is_d1_feat, on="trade_id")
        is_full = is_full.set_index("trade_id").loc[is_trades["trade_id"].to_numpy()].reset_index()
        all_d1_features = PIPELINE_E_BASE_FEATURES + PIPELINE_D1_PATH_FEATURES
        X_is_d1 = _impute_nans(is_full[all_d1_features].iloc[is_eligible])
        y_is_d1 = y_is[is_eligible]

        clf = _make_rf()
        clf.fit(X_is_d1, y_is_d1)

        # OOS: every signal enters; at bar t classify; admit/reject.
        oos_d1_feat, oos_eligible = compute_d1_features_at_t(
            oos_trades, paths_index, t, UNIT_SELECTED_SL, ORIGINAL_SL
        )
        oos_full = oos_base_e_src.merge(oos_d1_feat, on="trade_id")
        oos_full = oos_full.set_index("trade_id").loc[oos_trades["trade_id"].to_numpy()].reset_index()

        per_trade_r: List[float] = []
        per_trade_pair: List[str] = []
        per_trade_admitted: List[bool] = []
        n_admitted_count = 0
        n_rejected_close_at_market_count = 0
        for i, (_, tr) in enumerate(oos_trades.iterrows()):
            path = paths_index[int(tr["trade_id"])]
            if not oos_eligible[i]:
                # Trade exited before bar t (under unit SL=4.0×ATR). Could happen
                # in rare cases — apply SL outcome directly.
                ev = _eval_trade_at_sl(path, UNIT_SELECTED_SL, ORIGINAL_SL)
                per_trade_r.append(float(ev.final_r_new))
                per_trade_pair.append(str(tr["pair"]))
                per_trade_admitted.append(False)
                continue
            X_one = _impute_nans(oos_full[all_d1_features].iloc[[i]])
            prob = float(clf.predict_proba(X_one)[0, 1])
            if prob >= threshold:
                # Admit — run with default exit under unit SL.
                ev = _eval_trade_at_sl(path, UNIT_SELECTED_SL, ORIGINAL_SL)
                per_trade_r.append(float(ev.final_r_new))
                per_trade_pair.append(str(tr["pair"]))
                per_trade_admitted.append(True)
                n_admitted_count += 1
            else:
                # Reject — close-at-market at bar t+1.
                reject_result = _pnl_d1_reject(path, t=t)
                per_trade_r.append(float(reject_result["final_r"]))
                per_trade_pair.append(str(tr["pair"]))
                per_trade_admitted.append(False)
                n_rejected_close_at_market_count += 1

        n_oos_signals = int(len(oos_trades))
        n_traded = len(per_trade_r)
        if per_trade_r:
            arr = np.array(per_trade_r, dtype=float)
            mean_r = float(arr.mean())
            std_r = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
            total_r = float(arr.sum())
            hit_rate = float((arr > 0).mean())
        else:
            mean_r = std_r = total_r = hit_rate = 0.0

        ending_bal, roi_pct, max_dd_pct, max_dd_r, bal_curve = _equity_metrics(
            per_trade_r, STARTING_BALANCE, RISK_PCT
        )
        sharpe = _sharpe(per_trade_r)

        results.append(
            _FoldResult(
                fold=fold_num, is_start=str(is_start.date()), is_end=str(is_end.date()),
                oos_start=str(oos_start.date()), oos_end=str(oos_end.date()),
                n_oos_signals=n_oos_signals,
                n_admitted=n_admitted_count,
                n_d1_rejected=n_rejected_close_at_market_count,
                n_traded=n_traded,
                hit_rate=hit_rate, mean_r=mean_r, std_r=std_r,
                total_r=total_r, sharpe_r=sharpe,
                starting_balance=STARTING_BALANCE, ending_balance=ending_bal,
                roi_pct=roi_pct, max_dd_pct=max_dd_pct, max_dd_r=max_dd_r,
                n_pairs_active=len(set(per_trade_pair)),
                is_train_count=int(len(is_trades)),
                per_trade_r=per_trade_r, per_trade_balance=bal_curve,
            )
        )
    return results


# ============================================================
# Aggregate + ship gates
# ============================================================


@dataclass
class _AggregateResult:
    pipeline: str
    n_folds_run: int
    n_folds_skipped: int
    total_oos_signals: int
    total_traded: int
    aggregate_hit_rate: float
    aggregate_mean_r: float
    aggregate_total_r: float
    aggregate_sharpe: float
    worst_window_roi_pct: float
    worst_window_max_dd_pct: float
    final_balance: float
    aggregate_roi_pct: float
    aggregate_max_dd_pct: float
    time_coverage_pct: float
    n_pairs_active_total: int
    # Ship gates
    gate_worst_dd_le_8: bool
    gate_worst_roi_gt_0: bool
    gate_sharpe_ge_08: bool
    gate_n_trades_ge_30: bool
    ship_gate_pass: bool
    ship_gate_fails: List[str] = field(default_factory=list)


def aggregate_folds(
    fold_results: List[_FoldResult], pipeline_name: str,
    total_data_days: int,
) -> _AggregateResult:
    run_folds = [r for r in fold_results if not r.skipped]
    skipped_folds = [r for r in fold_results if r.skipped]
    n_run = len(run_folds)
    n_skipped = len(skipped_folds)

    total_traded = sum(r.n_traded for r in run_folds)
    total_oos = sum(r.n_oos_signals for r in run_folds)

    all_r: List[float] = []
    for r in run_folds:
        all_r.extend(r.per_trade_r)

    if all_r:
        arr = np.array(all_r, dtype=float)
        agg_hit = float((arr > 0).mean())
        agg_mean = float(arr.mean())
        agg_total = float(arr.sum())
        agg_sharpe = _sharpe(all_r)
    else:
        agg_hit = agg_mean = agg_total = 0.0
        agg_sharpe = float("nan")

    # Worst-window metrics across run folds.
    if run_folds:
        worst_roi = min(r.roi_pct for r in run_folds)
        worst_dd = max(r.max_dd_pct for r in run_folds)
    else:
        worst_roi = 0.0
        worst_dd = 0.0

    # Concatenated balance curve across folds (compounded).
    bal = STARTING_BALANCE
    peak = bal
    max_dd_pct = 0.0
    full_curve: List[float] = []
    for fold_r in run_folds:
        for r in fold_r.per_trade_r:
            dollar_risk = bal * RISK_PCT
            bal += dollar_risk * r
            full_curve.append(bal)
            peak = max(peak, bal)
            if peak > 0:
                dd = (peak - bal) / peak * 100.0
                if dd > max_dd_pct:
                    max_dd_pct = dd
    final_balance = bal
    agg_roi = (final_balance - STARTING_BALANCE) / STARTING_BALANCE * 100.0

    # Time coverage % = sum of OOS window days / total data window days.
    oos_days = 0
    for r in run_folds:
        os_ = pd.Timestamp(r.oos_start)
        oe_ = pd.Timestamp(r.oos_end)
        oos_days += (oe_ - os_).days
    time_cov = (oos_days / total_data_days * 100.0) if total_data_days > 0 else 0.0

    for fr in run_folds:
        # We rebuild pairs from per_trade_pair if stored... we didn't keep it
        # in _FoldResult outside loops, so use n_pairs_active aggregated.
        pass

    # Ship gates (v2.3 §10).
    gate_dd = worst_dd <= 8.0
    gate_roi = worst_roi > 0.0
    gate_sharpe = (not math.isnan(agg_sharpe)) and agg_sharpe >= 0.8
    gate_n = total_traded >= 30

    fails: List[str] = []
    if not gate_dd:
        fails.append(f"worst_window_max_dd {worst_dd:.2f}% > 8%")
    if not gate_roi:
        fails.append(f"worst_window_roi {worst_roi:.2f}% <= 0")
    if not gate_sharpe:
        fails.append(f"aggregate_sharpe {agg_sharpe:.4f} < 0.8")
    if not gate_n:
        fails.append(f"n_oos_trades {total_traded} < 30")
    ship_pass = gate_dd and gate_roi and gate_sharpe and gate_n

    return _AggregateResult(
        pipeline=pipeline_name,
        n_folds_run=n_run, n_folds_skipped=n_skipped,
        total_oos_signals=total_oos, total_traded=total_traded,
        aggregate_hit_rate=agg_hit, aggregate_mean_r=agg_mean,
        aggregate_total_r=agg_total, aggregate_sharpe=agg_sharpe,
        worst_window_roi_pct=worst_roi, worst_window_max_dd_pct=worst_dd,
        final_balance=final_balance, aggregate_roi_pct=agg_roi,
        aggregate_max_dd_pct=max_dd_pct, time_coverage_pct=time_cov,
        n_pairs_active_total=sum(set([fr.n_pairs_active for fr in run_folds])) if run_folds else 0,
        gate_worst_dd_le_8=gate_dd, gate_worst_roi_gt_0=gate_roi,
        gate_sharpe_ge_08=gate_sharpe, gate_n_trades_ge_30=gate_n,
        ship_gate_pass=ship_pass, ship_gate_fails=fails,
    )


# ============================================================
# Output writers
# ============================================================


def write_per_window_csv(out_path: Path, results: List[_FoldResult]) -> None:
    cols = [
        "fold", "is_start", "is_end", "oos_start", "oos_end",
        "is_train_count", "n_oos_signals", "n_admitted", "n_d1_rejected", "n_traded",
        "hit_rate", "mean_r", "std_r", "total_r", "sharpe_r",
        "starting_balance", "ending_balance", "roi_pct", "max_dd_pct", "max_dd_r",
        "n_pairs_active", "skipped", "skip_reason",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        for r in results:
            w.writerow([
                r.fold, r.is_start, r.is_end, r.oos_start, r.oos_end,
                r.is_train_count, r.n_oos_signals, r.n_admitted, r.n_d1_rejected, r.n_traded,
                f"{r.hit_rate:.6g}" if not math.isnan(r.hit_rate) else "",
                f"{r.mean_r:.6g}" if not math.isnan(r.mean_r) else "",
                f"{r.std_r:.6g}" if not math.isnan(r.std_r) else "",
                f"{r.total_r:.6g}",
                f"{r.sharpe_r:.6g}" if not math.isnan(r.sharpe_r) else "",
                f"{r.starting_balance:.6g}", f"{r.ending_balance:.6g}",
                f"{r.roi_pct:.6g}", f"{r.max_dd_pct:.6g}", f"{r.max_dd_r:.6g}",
                r.n_pairs_active, int(r.skipped), r.skip_reason,
            ])


def write_aggregate_json(out_path: Path, agg: _AggregateResult,
                         per_fold: List[_FoldResult]) -> None:
    payload = {
        "pipeline": agg.pipeline,
        "n_folds_run": agg.n_folds_run,
        "n_folds_skipped": agg.n_folds_skipped,
        "totals": {
            "oos_signals": agg.total_oos_signals,
            "traded": agg.total_traded,
            "hit_rate": agg.aggregate_hit_rate,
            "mean_r": agg.aggregate_mean_r,
            "total_r": agg.aggregate_total_r,
            "sharpe": agg.aggregate_sharpe,
        },
        "worst_window": {
            "roi_pct": agg.worst_window_roi_pct,
            "max_dd_pct": agg.worst_window_max_dd_pct,
        },
        "compounded_aggregate": {
            "starting_balance": STARTING_BALANCE,
            "final_balance": agg.final_balance,
            "roi_pct": agg.aggregate_roi_pct,
            "max_dd_pct": agg.aggregate_max_dd_pct,
            "time_coverage_pct": agg.time_coverage_pct,
        },
        "ship_gates": {
            "worst_dd_le_8": agg.gate_worst_dd_le_8,
            "worst_roi_gt_0": agg.gate_worst_roi_gt_0,
            "sharpe_ge_08": agg.gate_sharpe_ge_08,
            "n_trades_ge_30": agg.gate_n_trades_ge_30,
            "pass": agg.ship_gate_pass,
            "fails": agg.ship_gate_fails,
        },
        "per_fold": [
            {
                "fold": r.fold,
                "is_start": r.is_start, "is_end": r.is_end,
                "oos_start": r.oos_start, "oos_end": r.oos_end,
                "is_train_count": r.is_train_count,
                "n_oos_signals": r.n_oos_signals,
                "n_admitted": r.n_admitted,
                "n_d1_rejected": r.n_d1_rejected,
                "n_traded": r.n_traded,
                "hit_rate": r.hit_rate if not math.isnan(r.hit_rate) else None,
                "mean_r": r.mean_r if not math.isnan(r.mean_r) else None,
                "total_r": r.total_r,
                "sharpe_r": r.sharpe_r if not math.isnan(r.sharpe_r) else None,
                "ending_balance": r.ending_balance,
                "roi_pct": r.roi_pct,
                "max_dd_pct": r.max_dd_pct,
                "n_pairs_active": r.n_pairs_active,
                "skipped": r.skipped,
                "skip_reason": r.skip_reason,
            }
            for r in per_fold
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_equity_curve_png(out_path: Path, per_fold: List[_FoldResult], pipeline_name: str
                           ) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[step5_wfo] matplotlib unavailable, skipping equity curve: {e}",
              file=sys.stderr)
        return

    # Stitch per-fold compounded curve into one continuous chart.
    bal = STARTING_BALANCE
    balances: List[float] = [bal]
    fold_boundaries: List[Tuple[int, str]] = []
    for fr in per_fold:
        if fr.skipped:
            continue
        for r in fr.per_trade_r:
            bal += (bal * RISK_PCT) * r
            balances.append(bal)
        fold_boundaries.append((len(balances) - 1, f"fold {fr.fold} end"))

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(balances, color="#1f77b4", linewidth=1.4)
    ax.axhline(STARTING_BALANCE, color="black", linewidth=0.7, linestyle="--",
               label="starting balance")
    for x, label in fold_boundaries:
        ax.axvline(x, color="gray", linewidth=0.4, alpha=0.5)
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Balance ($)")
    ax.set_title(f"Arc 8 Step 5 — {pipeline_name} compounded equity curve (c1 V-shape recovery)")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, metadata={"Software": ""})
    plt.close(fig)


def _sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ============================================================
# Driver
# ============================================================


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Arc 8 Step 5 WFO.")
    ap.add_argument("-c", "--config", type=Path, default=Path("configs/l_arc_8/step5.yaml"))
    args = ap.parse_args(argv)
    cfg_path = args.config
    if not cfg_path.is_absolute():
        cfg_path = (_REPO_ROOT / cfg_path).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    out_dir = _REPO_ROOT / cfg["output"]["results_dir"]
    e_dir = out_dir / "pipeline_e"
    d1_dir = out_dir / "pipeline_d1"
    out_dir.mkdir(parents=True, exist_ok=True)
    e_dir.mkdir(parents=True, exist_ok=True)
    d1_dir.mkdir(parents=True, exist_ok=True)

    # Inputs.
    step1_dir = _REPO_ROOT / cfg["input"]["step1_dir"]
    step2_dir = _REPO_ROOT / cfg["input"]["step2_dir"]
    _REPO_ROOT / cfg["input"]["step4_dir"]
    trades_df = pd.read_csv(step1_dir / cfg["input"]["trades_csv"])
    paths_df = pd.read_csv(step1_dir / cfg["input"]["paths_csv"])
    clusters_df = pd.read_csv(step2_dir / cfg["input"]["clusters_csv"])

    trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"])
    trades_df["signal_bar_time"] = pd.to_datetime(trades_df["signal_bar_time"])

    # Filter to c1.
    c1_cluster_id = int(cfg["input"]["cluster_id"])
    c1_trade_ids = clusters_df[clusters_df["cluster_id"] == c1_cluster_id]["trade_id"].tolist()
    c1_trades = trades_df[trades_df["trade_id"].isin(c1_trade_ids)].copy()
    c1_trades = c1_trades.sort_values("entry_time", kind="mergesort").reset_index(drop=True)
    print(f"[step5_wfo] c1 pool: n={len(c1_trades)}, "
          f"range {c1_trades['entry_time'].min().date()} to {c1_trades['entry_time'].max().date()}",
          file=sys.stderr)

    paths_index = _build_paths_index(paths_df)

    # Build base E features. For the FULL Step 1 pool (used for full-pool
    # deployment economics) AND c1's subset (used for admit-only validation).
    pairs = sorted(trades_df["pair"].unique())
    dir_4h = cfg["data"]["dir_4h"]
    pair_caches = {p: _build_pair_cache(p, dir_4h) for p in pairs}
    arc8_entry_cols = [
        "num_higher_highs", "num_higher_lows", "most_recent_sh_age",
        "most_recent_sl_age", "hh_range_atr", "hl_range_atr",
        "pullback_depth_atr", "trigger_body_atr", "trigger_close_pos",
        "trigger_break_size_atr",
    ]
    print(f"[step5_wfo] Building base E features for full Step 1 pool (n={len(trades_df)})...",
          file=sys.stderr)
    base_e_full = compute_base_e_features(trades_df, pair_caches)
    base_e_full = base_e_full.merge(
        trades_df[["trade_id"] + arc8_entry_cols], on="trade_id", how="left"
    )
    # c1 subset of base_e_full (filter, do not re-compute).
    base_e_df = base_e_full[base_e_full["trade_id"].isin(c1_trades["trade_id"].to_numpy())].copy()

    # WFO folds.
    wfo_cfg = yaml.safe_load((_REPO_ROOT / cfg["input"]["wfo_kh24_yaml"]).read_text(encoding="utf-8"))
    wfo_folds = wfo_cfg["wfo"]["folds"]
    data_start = pd.Timestamp(wfo_cfg["wfo"]["start"])
    data_end = pd.Timestamp(wfo_cfg["wfo"]["end"])
    total_days = (data_end - data_start).days
    print(f"[step5_wfo] WFO folds: {len(wfo_folds)}; data window {data_start.date()} to {data_end.date()}",
          file=sys.stderr)

    # ===== Pipeline E WFO =====
    # admit_only view: c1 OOS trades only (Step 4 cross-validation under
    # different fold structure). NOT the ship-gate basis.
    print("[step5_wfo] === Pipeline E WFO (admit_only view: c1 OOS only) ===", file=sys.stderr)
    e_admit = run_pipeline_e_wfo(c1_trades, base_e_df, paths_index, wfo_folds)
    for r in e_admit:
        if r.skipped:
            print(f"  fold {r.fold}: SKIP — {r.skip_reason}", file=sys.stderr)
        else:
            print(f"  fold {r.fold}: oos_c1={r.n_oos_signals} admit={r.n_admitted} "
                  f"mean_r={r.mean_r:+.3f} total_r={r.total_r:+.2f} roi={r.roi_pct:+.2f}% "
                  f"max_dd={r.max_dd_pct:.2f}% hit={r.hit_rate:.3f}", file=sys.stderr)
    write_per_window_csv(e_dir / "wfo_results_per_window_admit_only.csv", e_admit)
    e_admit_agg = aggregate_folds(e_admit, "pipeline_e_admit_only", total_days)
    write_aggregate_json(e_dir / "wfo_aggregate_admit_only.json", e_admit_agg, e_admit)

    # full_pool view: classifier applied to ALL Step 1 OOS trades.
    # THIS is the §10 ship-gate basis (production deployment economics).
    print("[step5_wfo] === Pipeline E WFO (full_pool view: all Step 1 OOS) ===", file=sys.stderr)
    e_full = run_pipeline_e_wfo(
        c1_trades, base_e_df, paths_index, wfo_folds,
        oos_pool_trades=trades_df, oos_pool_base_e=base_e_full,
    )
    for r in e_full:
        if r.skipped:
            print(f"  fold {r.fold}: SKIP — {r.skip_reason}", file=sys.stderr)
        else:
            admit_pct = (r.n_admitted / r.n_oos_signals * 100.0) if r.n_oos_signals else 0.0
            print(f"  fold {r.fold}: oos_pool={r.n_oos_signals} admit={r.n_admitted} ({admit_pct:.1f}%) "
                  f"mean_r={r.mean_r:+.3f} total_r={r.total_r:+.2f} roi={r.roi_pct:+.2f}% "
                  f"max_dd={r.max_dd_pct:.2f}% hit={r.hit_rate:.3f}", file=sys.stderr)
    write_per_window_csv(e_dir / "wfo_results_per_window.csv", e_full)
    e_full_agg = aggregate_folds(e_full, "pipeline_e", total_days)
    write_aggregate_json(e_dir / "wfo_aggregate.json", e_full_agg, e_full)
    write_equity_curve_png(e_dir / "equity_curve.png", e_full, "Pipeline E (full-pool deployment)")

    # ===== Pipeline D1 WFO =====
    print("[step5_wfo] === Pipeline D1 WFO (admit_only view: c1 OOS only) ===", file=sys.stderr)
    d1_admit = run_pipeline_d1_wfo(c1_trades, base_e_df, paths_index, wfo_folds)
    for r in d1_admit:
        if r.skipped:
            print(f"  fold {r.fold}: SKIP — {r.skip_reason}", file=sys.stderr)
        else:
            print(f"  fold {r.fold}: oos_c1={r.n_oos_signals} admit={r.n_admitted} "
                  f"reject_cam={r.n_d1_rejected} traded={r.n_traded} "
                  f"mean_r={r.mean_r:+.3f} total_r={r.total_r:+.2f} roi={r.roi_pct:+.2f}% "
                  f"max_dd={r.max_dd_pct:.2f}% hit={r.hit_rate:.3f}", file=sys.stderr)
    write_per_window_csv(d1_dir / "wfo_results_per_window_admit_only.csv", d1_admit)
    d1_admit_agg = aggregate_folds(d1_admit, "pipeline_d1_admit_only", total_days)
    write_aggregate_json(d1_dir / "wfo_aggregate_admit_only.json", d1_admit_agg, d1_admit)

    print("[step5_wfo] === Pipeline D1 WFO (full_pool view: all Step 1 OOS) ===", file=sys.stderr)
    d1_full = run_pipeline_d1_wfo(
        c1_trades, base_e_df, paths_index, wfo_folds,
        oos_pool_trades=trades_df, oos_pool_base_e=base_e_full,
    )
    for r in d1_full:
        if r.skipped:
            print(f"  fold {r.fold}: SKIP — {r.skip_reason}", file=sys.stderr)
        else:
            admit_pct = (r.n_admitted / r.n_oos_signals * 100.0) if r.n_oos_signals else 0.0
            reject_pct = (r.n_d1_rejected / r.n_oos_signals * 100.0) if r.n_oos_signals else 0.0
            print(f"  fold {r.fold}: oos_pool={r.n_oos_signals} admit={r.n_admitted} ({admit_pct:.1f}%) "
                  f"reject_cam={r.n_d1_rejected} ({reject_pct:.1f}%) "
                  f"mean_r={r.mean_r:+.3f} total_r={r.total_r:+.2f} roi={r.roi_pct:+.2f}% "
                  f"max_dd={r.max_dd_pct:.2f}% hit={r.hit_rate:.3f}", file=sys.stderr)
    write_per_window_csv(d1_dir / "wfo_results_per_window.csv", d1_full)
    d1_full_agg = aggregate_folds(d1_full, "pipeline_d1", total_days)
    write_aggregate_json(d1_dir / "wfo_aggregate.json", d1_full_agg, d1_full)
    write_equity_curve_png(d1_dir / "equity_curve.png", d1_full, "Pipeline D1 (full-pool deployment)")

    # Ship-gate summary on full_pool view (per §10).
    print("", file=sys.stderr)
    print("=== STEP 5 SHIP-GATE SUMMARY (full-pool deployment economics per §10) ===", file=sys.stderr)
    for agg in (e_full_agg, d1_full_agg):
        print(f"{agg.pipeline}: ship_gate_pass={agg.ship_gate_pass}", file=sys.stderr)
        print(f"  worst_window_roi={agg.worst_window_roi_pct:+.2f}% (gate >0)", file=sys.stderr)
        print(f"  worst_window_max_dd={agg.worst_window_max_dd_pct:.2f}% (gate <=8%)", file=sys.stderr)
        print(f"  aggregate_sharpe={agg.aggregate_sharpe:.4f} (gate >=0.8)", file=sys.stderr)
        print(f"  n_traded={agg.total_traded} (gate >=30)", file=sys.stderr)
        if agg.ship_gate_fails:
            print(f"  fails: {'; '.join(agg.ship_gate_fails)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
