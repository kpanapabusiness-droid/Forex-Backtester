"""Arc 4 — Step 5B-refit risk sweep on cluster 1 (L_ARC_PROTOCOL v2.1.1 §9 supplement).

Purpose: redo the original 5B risk sweep on the post-leakage-fix refit data
produced at Step 5C (per_trade_simulated_refit_1.csv), under CONVENTION (b)
hourly mark-to-market equity DD, with the addition of DAILY DD distribution
analysis answering the 5ers 5%/day broker limit.

Scope:
  - Cluster 1 only (cluster 3 eliminated at Step 5).
  - F2-F7 only (F1 dropped — refit excluded F1 due to no training history).

Methodology:
  1. Per fold F2..F7: build the hourly MTM equity curve at 0.5% risk by
     summing each admitted trade's per-bar floating R (bars 0..exit_bar) and
     constant final_r (after exit) on a common 1H grid within the fold's
     OOS window.
  2. fold_max_dd_b_0.5pct = max(running_peak − equity) on the hourly curve.
  3. fold_roi_b_0.5pct = final_equity − initial_equity (= 0 at fold start).
  4. Daily DD per day = max intraday drop from start-of-day equity to
     intraday low (broker rule; 5ers daily limit 5%).
  5. Linear scaling (verified in original 5B) → all other risk levels.
  6. Per-risk aggregation + §10 gates incl. daily DD constraint.

DD scaling property used (per original 5B verification): convention-(a) DD
scales perfectly linearly with risk. Convention-(b) DD ALSO scales perfectly
linearly with risk because every contribution to the hourly equity curve is
linear in risk_pct.

Outputs (results/l_arc_4/step5b_refit/):
  - risk_sweep_refit_b_cluster_1.csv
  - fold_b_at_<risk_bps>_cluster_1.csv (per risk level)
  - daily_dd_distribution_at_<risk_bps>_cluster_1.csv (per risk level)
  - hourly_equity_curve_at_0.5pct_cluster_1.csv (reference for audit)
  - step5b_refit_diagnostics.md

Determinism: deterministic per-fold arithmetic; two-run byte-identical sha256.

Usage:
  py scripts/l_arc_4/step5b_refit_risk_sweep.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]


# ============================================================================
# Constants
# ============================================================================

RISK_GRID_PCT: Tuple[float, ...] = (0.50, 0.40, 0.30, 0.25, 0.22, 0.20, 0.18, 0.15, 0.10)
BASELINE_RISK_PCT: float = 0.50

PATH_BARS: int = 241

# §10 thresholds
DD_CEILING_PCT: float = 8.0
PASS_DEPLOY_WORST_ROI_ANN_PCT: float = 5.0
PASS_DEPLOY_MEAN_ROI_ANN_PCT: float = 8.0
PASS_VIABLE_WORST_ROI_PCT: float = 0.0
PASS_VIABLE_MEAN_ROI_ANN_PCT: float = 3.0

# 5ers daily DD limits
DAILY_DD_LIMIT_PCT: float = 5.0    # account closure threshold
DAILY_DD_TARGET_PCT: float = 4.0   # safety margin


# ============================================================================
# Fold loading
# ============================================================================


@dataclass
class WFOFold:
    fold: int
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp
    oos_days: int


def load_fold_defs(fold_defs_csv: Path) -> List[WFOFold]:
    df = pd.read_csv(fold_defs_csv)
    out: List[WFOFold] = []
    for _, r in df.iterrows():
        s = pd.Timestamp(r["oos_start"])
        e = pd.Timestamp(r["oos_end"])
        out.append(WFOFold(fold=int(r["fold"]), oos_start=s, oos_end=e, oos_days=(e - s).days))
    return out


# ============================================================================
# Hourly MTM equity curve per fold (at baseline risk)
# ============================================================================


def build_hourly_equity_per_fold(
    sim_trades: pd.DataFrame,
    folds: List[WFOFold],
    path_tensor_closes: np.ndarray,
    path_tensor_times: np.ndarray,
    trade_id_to_idx: Dict[int, int],
    risk_pct_baseline: float = BASELINE_RISK_PCT,
) -> Tuple[Dict[int, pd.Series], Dict[int, float], Dict[int, float]]:
    """Return per-fold:
      - hourly_equity: pd.Series indexed by 1H timestamps (in % of starting balance)
      - fold_max_dd: max running drawdown on hourly equity
      - fold_roi: end - start equity
    """
    risk_scale = risk_pct_baseline / 100.0  # fraction of balance
    hourly_equity_per_fold: Dict[int, pd.Series] = {}
    fold_max_dd: Dict[int, float] = {}
    fold_roi: Dict[int, float] = {}

    for f in folds:
        if f.fold == 1:
            # F1 has no refit admitted trades — skip (refit excludes F1 by design).
            continue
        f_trades = sim_trades[sim_trades["fold"] == f.fold]
        if f_trades.empty:
            hourly_equity_per_fold[f.fold] = pd.Series([], dtype=float)
            fold_max_dd[f.fold] = 0.0
            fold_roi[f.fold] = 0.0
            continue

        timeline = pd.date_range(
            start=f.oos_start, end=f.oos_end - pd.Timedelta(hours=1), freq="1h"
        )
        if len(timeline) == 0:
            hourly_equity_per_fold[f.fold] = pd.Series([], dtype=float)
            fold_max_dd[f.fold] = 0.0
            fold_roi[f.fold] = 0.0
            continue

        total_r = pd.Series(0.0, index=timeline)
        for _, row in f_trades.iterrows():
            tid = int(row["trade_id"])
            entry_price = float(row["entry_price"])
            cluster_R = float(row["cluster_R"])
            exit_bar = int(row["exit_bar"])
            final_r = float(row["final_r"])
            if tid not in trade_id_to_idx:
                continue
            t_idx = trade_id_to_idx[tid]
            bar_times = pd.to_datetime(path_tensor_times[t_idx])
            bar_closes = path_tensor_closes[t_idx]

            # Per-bar floating R during held window 0..exit_bar.
            held_r = (bar_closes[: exit_bar + 1] - entry_price) / cluster_R
            held_times = bar_times[: exit_bar + 1]
            # After exit_bar: constant final_r (closed position).
            after_times = bar_times[exit_bar + 1 :]
            if len(after_times) > 0:
                after_r = np.full(len(after_times), final_r)
                contrib = pd.Series(
                    np.concatenate([held_r, after_r]),
                    index=pd.DatetimeIndex(list(held_times) + list(after_times)),
                )
            else:
                contrib = pd.Series(held_r, index=pd.DatetimeIndex(held_times))

            contrib = contrib.sort_index()
            # Reindex onto fold's 1H grid; forward-fill from each bar close-of-bar
            # timestamp. Pre-entry hours: 0 (fillna).
            contrib_on_grid = contrib.reindex(timeline, method="ffill").fillna(0.0)
            total_r = total_r + contrib_on_grid

        # Convert to % equity: total_r is in R-units; PnL_pct = R × risk_pct.
        equity_pct = total_r * risk_scale * 100.0
        hourly_equity_per_fold[f.fold] = equity_pct
        running_peak = np.maximum.accumulate(equity_pct.to_numpy())
        dd = running_peak - equity_pct.to_numpy()
        fold_max_dd[f.fold] = float(dd.max()) if dd.size else 0.0
        fold_roi[f.fold] = float(equity_pct.iloc[-1]) - float(equity_pct.iloc[0]) if len(equity_pct) >= 2 else 0.0

    return hourly_equity_per_fold, fold_max_dd, fold_roi


# ============================================================================
# Daily DD analysis
# ============================================================================


def compute_daily_dds(hourly_equity: pd.Series) -> Tuple[List[Tuple[pd.Timestamp, float]], List[float]]:
    """Per calendar day, compute the intraday max drop from start-of-day equity
    to intraday minimum (broker-rule daily DD).

    Returns:
      - per_day: list of (day, daily_dd_pct) for every day with data
      - dds: list of daily_dd_pct values only (for distribution stats)
    """
    if hourly_equity.empty:
        return [], []
    eq = hourly_equity
    by_day = eq.groupby(eq.index.normalize())
    per_day: List[Tuple[pd.Timestamp, float]] = []
    dds: List[float] = []
    for day, group in by_day:
        if len(group) == 0:
            continue
        sod = float(group.iloc[0])     # start-of-day equity (first hour)
        intraday_min = float(group.min())
        daily_dd = max(0.0, sod - intraday_min)
        per_day.append((pd.Timestamp(day), daily_dd))
        dds.append(daily_dd)
    return per_day, dds


# ============================================================================
# Linear scaling helpers
# ============================================================================


def scale_dd_or_roi(value_at_baseline: float, risk_pct: float, baseline: float = BASELINE_RISK_PCT) -> float:
    return value_at_baseline * (risk_pct / baseline)


# ============================================================================
# CSV / output helpers
# ============================================================================


def _file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fmt_g(x: Any) -> Any:
    if x is None or (isinstance(x, float) and (math.isnan(x) or not math.isfinite(x))):
        return ""
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.10g}"
    return x


def _write_rows(rows: List[Dict[str, Any]], path: Path) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, na_rep="", lineterminator="\n")


# ============================================================================
# Single-run orchestration
# ============================================================================


def run_once(
    sim_csv: Path,
    paths_csv: Path,
    fold_defs_csv: Path,
    out_dir: Path,
) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    folds = load_fold_defs(fold_defs_csv)

    sim = pd.read_csv(sim_csv)
    sim["entry_ts"] = pd.to_datetime(sim["entry_ts"])
    sim = sim.sort_values("entry_ts").reset_index(drop=True)
    # Filter to admitted refit trades (F2..F7 by definition).
    sim = sim[sim["fold"].isin([f.fold for f in folds if f.fold >= 2])].reset_index(drop=True)
    if sim.empty:
        raise RuntimeError(f"{sim_csv.name} contains no F2..F7 admitted trades.")

    # Load trades_paths.csv (just close + timestamp + trade_id + bar_offset).
    print("[5b-refit] loading per-trade bar paths...", file=sys.stderr)
    t0 = time.time()
    paths_df = pd.read_csv(
        paths_csv, usecols=["trade_id", "bar_offset", "timestamp", "close"]
    )
    paths_df = paths_df.sort_values(["trade_id", "bar_offset"]).reset_index(drop=True)
    # Filter to admitted refit trades (use existing trade_ids in sim).
    admitted_tids = sim["trade_id"].astype(int).unique().tolist()
    paths_sub = paths_df[paths_df["trade_id"].isin(admitted_tids)].copy()
    paths_sub["timestamp"] = pd.to_datetime(paths_sub["timestamp"])
    paths_sub = paths_sub.sort_values(["trade_id", "bar_offset"]).reset_index(drop=True)
    n_adm = paths_sub["trade_id"].nunique()
    if len(paths_sub) != n_adm * PATH_BARS:
        raise ValueError(
            f"Per-trade path expansion mismatch: got {len(paths_sub)} rows, "
            f"expected {n_adm}×{PATH_BARS}"
        )
    closes_2d = paths_sub["close"].to_numpy(dtype=float).reshape(n_adm, PATH_BARS)
    times_2d = paths_sub["timestamp"].to_numpy().reshape(n_adm, PATH_BARS)
    tid_arr = paths_sub["trade_id"].to_numpy(dtype=int).reshape(n_adm, PATH_BARS)[:, 0]
    trade_id_to_idx = {int(t): i for i, t in enumerate(tid_arr)}
    print(f"[5b-refit] paths loaded in {time.time() - t0:.1f}s "
          f"({n_adm} admitted trades × {PATH_BARS} bars)", file=sys.stderr)

    # Build per-fold hourly equity at baseline risk.
    print("[5b-refit] building hourly MTM equity at 0.5% risk...", file=sys.stderr)
    t1 = time.time()
    hourly_equity_per_fold, fold_max_dd_baseline, fold_roi_baseline = build_hourly_equity_per_fold(
        sim, folds, closes_2d, times_2d, trade_id_to_idx, BASELINE_RISK_PCT
    )
    print(f"[5b-refit] equity curves built in {time.time() - t1:.1f}s", file=sys.stderr)

    # Persist hourly equity curve concatenated across folds for audit.
    rows_eq: List[Dict[str, Any]] = []
    for fold_id, eq in hourly_equity_per_fold.items():
        for ts, val in eq.items():
            rows_eq.append(
                {"fold": fold_id, "timestamp": pd.Timestamp(ts).isoformat(), "equity_pct": _fmt_g(float(val))}
            )
    eq_curve_path = out_dir / "hourly_equity_curve_at_0.5pct_cluster_1.csv"
    _write_rows(rows_eq, eq_curve_path)

    # Per-fold daily DD distribution at baseline (then scale linearly).
    daily_dds_per_fold_baseline: Dict[int, List[Tuple[pd.Timestamp, float]]] = {}
    for fold_id, eq in hourly_equity_per_fold.items():
        per_day, _ = compute_daily_dds(eq)
        daily_dds_per_fold_baseline[fold_id] = per_day

    sha_files: Dict[str, str] = {}
    sha_files[eq_curve_path.name] = _file_sha256(eq_curve_path)

    # Per fold OOS days lookup.
    fold_oos_days = {f.fold: f.oos_days for f in folds}
    annualisation_per_fold = {f.fold: 365.0 / f.oos_days for f in folds if f.fold >= 2}

    # Risk sweep — scale + aggregate per risk level.
    sweep_rows: List[Dict[str, Any]] = []
    for risk in RISK_GRID_PCT:
        factor = risk / BASELINE_RISK_PCT
        per_fold_rows: List[Dict[str, Any]] = []
        daily_dd_rows: List[Dict[str, Any]] = []
        per_fold_dd_at_risk: Dict[int, float] = {}
        per_fold_roi_at_risk: Dict[int, float] = {}
        per_fold_roi_ann_at_risk: Dict[int, float] = {}
        all_daily_dds_at_risk: List[float] = []
        worst_daily_per_fold: Dict[int, float] = {}

        for f in folds:
            if f.fold == 1:
                continue
            f_trades = sim[sim["fold"] == f.fold]
            n = int(len(f_trades))
            dd_baseline = fold_max_dd_baseline.get(f.fold, 0.0)
            roi_baseline = fold_roi_baseline.get(f.fold, 0.0)
            dd_at_risk = dd_baseline * factor
            roi_at_risk = roi_baseline * factor
            roi_ann_at_risk = roi_at_risk * annualisation_per_fold[f.fold]
            per_fold_dd_at_risk[f.fold] = dd_at_risk
            per_fold_roi_at_risk[f.fold] = roi_at_risk
            per_fold_roi_ann_at_risk[f.fold] = roi_ann_at_risk

            # Daily DD scaled.
            per_day_baseline = daily_dds_per_fold_baseline.get(f.fold, [])
            per_day_scaled = [(d, dd * factor) for d, dd in per_day_baseline]
            fold_daily_dds = [dd for _, dd in per_day_scaled]
            worst_daily_per_fold[f.fold] = float(max(fold_daily_dds)) if fold_daily_dds else 0.0
            all_daily_dds_at_risk.extend(fold_daily_dds)
            for d, dd in per_day_scaled:
                daily_dd_rows.append(
                    {
                        "risk_pct": risk,
                        "fold": f.fold,
                        "day": pd.Timestamp(d).date().isoformat(),
                        "daily_dd_pct": _fmt_g(dd),
                    }
                )

            mean_r_fold = (
                float(f_trades["final_r"].mean()) if not f_trades.empty else 0.0
            )
            std_r_fold = (
                float(f_trades["final_r"].std(ddof=1)) if len(f_trades) > 1 else 0.0
            )
            t_stat_fold = (
                float(mean_r_fold / (std_r_fold / math.sqrt(n))) if std_r_fold > 0 else 0.0
            )
            # Exit reason dist (cosmetic).
            exit_dist = (
                f_trades["exit_reason"].value_counts().to_dict() if not f_trades.empty else {}
            )
            per_fold_rows.append(
                {
                    "cluster_id": 1,
                    "risk_pct": risk,
                    "fold": f.fold,
                    "n": n,
                    "oos_days": fold_oos_days[f.fold],
                    "annualisation_factor": _fmt_g(annualisation_per_fold[f.fold]),
                    "mean_r": _fmt_g(mean_r_fold),
                    "std_r": _fmt_g(std_r_fold),
                    "t_stat": _fmt_g(t_stat_fold),
                    "fold_roi_b_pct": _fmt_g(roi_at_risk),
                    "fold_roi_b_ann_pct": _fmt_g(roi_ann_at_risk),
                    "fold_max_dd_b_pct": _fmt_g(dd_at_risk),
                    "worst_daily_dd_b_pct": _fmt_g(worst_daily_per_fold[f.fold]),
                    "exit_reason_dist": ";".join(f"{k}={int(v)}" for k, v in sorted(exit_dist.items())),
                }
            )

        # Aggregations.
        active_folds = [f.fold for f in folds if f.fold >= 2]
        worst_fold_dd = max(per_fold_dd_at_risk[fid] for fid in active_folds)
        mean_fold_dd = float(np.mean([per_fold_dd_at_risk[fid] for fid in active_folds]))
        worst_fold_roi = min(per_fold_roi_at_risk[fid] for fid in active_folds)
        worst_fold_roi_ann = min(per_fold_roi_ann_at_risk[fid] for fid in active_folds)
        mean_fold_roi = float(np.mean([per_fold_roi_at_risk[fid] for fid in active_folds]))
        mean_fold_roi_ann = float(np.mean([per_fold_roi_ann_at_risk[fid] for fid in active_folds]))

        worst_daily_dd_overall = max(worst_daily_per_fold.values()) if worst_daily_per_fold else 0.0
        if all_daily_dds_at_risk:
            daily_dd_p99 = float(np.percentile(all_daily_dds_at_risk, 99))
            daily_dd_p95 = float(np.percentile(all_daily_dds_at_risk, 95))
            daily_dd_p50 = float(np.percentile(all_daily_dds_at_risk, 50))
        else:
            daily_dd_p99 = daily_dd_p95 = daily_dd_p50 = 0.0

        passes_fold_dd = worst_fold_dd <= DD_CEILING_PCT
        passes_daily_dd_target = worst_daily_dd_overall <= DAILY_DD_TARGET_PCT
        passes_daily_dd_limit = worst_daily_dd_overall <= DAILY_DD_LIMIT_PCT

        passes_deployable = (
            worst_fold_roi_ann >= PASS_DEPLOY_WORST_ROI_ANN_PCT
            and mean_fold_roi_ann >= PASS_DEPLOY_MEAN_ROI_ANN_PCT
            and passes_fold_dd
            and daily_dd_p99 <= DAILY_DD_TARGET_PCT
            and passes_daily_dd_limit
        )
        passes_viable = (
            worst_fold_roi_ann > PASS_VIABLE_WORST_ROI_PCT
            and mean_fold_roi_ann >= PASS_VIABLE_MEAN_ROI_ANN_PCT
            and passes_fold_dd
            and passes_daily_dd_limit
        )

        sweep_rows.append(
            {
                "cluster_id": 1,
                "risk_pct": risk,
                "worst_fold_dd_b_pct": _fmt_g(worst_fold_dd),
                "worst_fold_roi_b_pct": _fmt_g(worst_fold_roi),
                "worst_fold_roi_ann_pct": _fmt_g(worst_fold_roi_ann),
                "mean_fold_dd_b_pct": _fmt_g(mean_fold_dd),
                "mean_fold_roi_b_pct": _fmt_g(mean_fold_roi),
                "mean_fold_roi_ann_pct": _fmt_g(mean_fold_roi_ann),
                "worst_daily_dd_b_pct": _fmt_g(worst_daily_dd_overall),
                "daily_dd_p99_b_pct": _fmt_g(daily_dd_p99),
                "daily_dd_p95_b_pct": _fmt_g(daily_dd_p95),
                "daily_dd_p50_b_pct": _fmt_g(daily_dd_p50),
                "passes_fold_dd_ceiling": int(passes_fold_dd),
                "passes_daily_dd_target": int(passes_daily_dd_target),
                "passes_daily_dd_limit": int(passes_daily_dd_limit),
                "passes_deployable": int(passes_deployable),
                "passes_viable": int(passes_viable),
            }
        )

        # Per-risk per-fold CSV.
        risk_bps = int(round(risk * 100))
        per_fold_path = out_dir / f"fold_b_at_{risk_bps}bps_cluster_1.csv"
        _write_rows(per_fold_rows, per_fold_path)
        sha_files[per_fold_path.name] = _file_sha256(per_fold_path)

        # Per-risk daily DD distribution CSV.
        dd_dist_path = out_dir / f"daily_dd_distribution_at_{risk_bps}bps_cluster_1.csv"
        _write_rows(daily_dd_rows, dd_dist_path)
        sha_files[dd_dist_path.name] = _file_sha256(dd_dist_path)

    sweep_path = out_dir / "risk_sweep_refit_b_cluster_1.csv"
    _write_rows(sweep_rows, sweep_path)
    sha_files[sweep_path.name] = _file_sha256(sweep_path)

    return sha_files


# ============================================================================
# Diagnostics writer
# ============================================================================


def write_diagnostics(
    out_path: Path,
    sweep_csv: Path,
    sha_run1: Dict[str, str],
    sha_run2: Optional[Dict[str, str]],
    determinism_gate: str,
    config_paths: Dict[str, str],
    config_shas: Dict[str, str],
) -> Tuple[str, Optional[float]]:
    sweep = pd.read_csv(sweep_csv)

    # Highest-risk PASS-DEPLOYABLE entry.
    dep_pass = sweep[sweep["passes_deployable"] == 1]
    viable_pass = sweep[sweep["passes_viable"] == 1]
    highest_deploy_risk = float(dep_pass["risk_pct"].max()) if not dep_pass.empty else None
    highest_viable_risk = float(viable_pass["risk_pct"].max()) if not viable_pass.empty else None
    recommended_risk = highest_deploy_risk if highest_deploy_risk is not None else highest_viable_risk

    # Identify which gate is binding at the recommended risk's boundary.
    if recommended_risk is None:
        headline = "FAIL — no risk level in grid passes all gates (fold DD, daily DD, ROI floors)"
    else:
        # Find the row just below the recommended (or the recommended itself) to identify
        # the binding constraint.
        rec_row = sweep[sweep["risk_pct"] == recommended_risk].iloc[0]
        sorted_above = sweep[sweep["risk_pct"] > recommended_risk].sort_values("risk_pct")
        if not sorted_above.empty:
            # First failing risk above the recommendation reveals the binding gate.
            first_fail = sorted_above.iloc[0]
            failing_gates: List[str] = []
            if not int(first_fail["passes_fold_dd_ceiling"]):
                failing_gates.append("fold DD ≤ 8%")
            if not int(first_fail["passes_daily_dd_target"]):
                failing_gates.append("daily DD p99 ≤ 4%")
            if not int(first_fail["passes_daily_dd_limit"]):
                failing_gates.append("worst daily DD ≤ 5%")
            if float(first_fail["worst_fold_roi_ann_pct"]) < PASS_DEPLOY_WORST_ROI_ANN_PCT:
                failing_gates.append("worst-fold ann ROI ≥ 5%")
            if float(first_fail["mean_fold_roi_ann_pct"]) < PASS_DEPLOY_MEAN_ROI_ANN_PCT:
                failing_gates.append("mean-fold ann ROI ≥ 8%")
            binding = ", ".join(failing_gates) if failing_gates else "all gates clear"
        else:
            binding = "no risk level above recommended in grid"
        headline = (
            f"Recommended risk = **{recommended_risk}%** "
            f"({'PASS-DEPLOYABLE' if highest_deploy_risk is not None and recommended_risk == highest_deploy_risk else 'PASS-VIABLE'}); "
            f"binding gate(s) above this level: {binding}. "
            f"Worst-fold annualised ROI {rec_row['worst_fold_roi_ann_pct']}%, "
            f"worst-fold DD (b) {rec_row['worst_fold_dd_b_pct']}%, "
            f"worst daily DD {rec_row['worst_daily_dd_b_pct']}%, "
            f"daily DD p99 {rec_row['daily_dd_p99_b_pct']}%."
        )

    lines: List[str] = []
    lines.append("# Arc 4 — Step 5B-refit risk sweep diagnostics")
    lines.append("")
    lines.append("Protocol: `L_ARC_PROTOCOL.md` v2.1.1 §9 supplement; §10 deploy/viable; "
                 "5ers daily DD limit (5%) / safety target (4%)")
    lines.append("Cluster: 1 (`stepwise_pullback_extended`, R = 3.0 × ATR)")
    lines.append("Source: `results/l_arc_4/step5c/per_trade_simulated_refit_1.csv` — per-fold "
                 "classifier refit (F1 excluded — structural leakage)")
    lines.append("Convention: **(b) hourly mark-to-market** — concurrent open-position floating PnL "
                 "aggregated across all 28 pairs on a common 1H grid per fold's OOS window. "
                 "Linear-in-risk scaling applied per risk grid level.")
    lines.append("")

    lines.append("## Headline")
    lines.append("")
    lines.append(headline)
    lines.append("")
    lines.append(f"Determinism: **{determinism_gate}**")
    lines.append("")

    # Risk-sweep table.
    lines.append("## Full risk-sweep table (convention (b), refit data, F2-F7)")
    lines.append("")
    lines.append(
        "| Risk% | worst fold DD% | worst fold ROI ann% | mean fold ROI ann% | "
        "worst daily DD% | daily DD p99% | daily DD p95% | fold DD ≤ 8% | "
        "daily ≤ 4% | daily ≤ 5% | pass-deploy | pass-viable |"
    )
    lines.append(
        "|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|:---:|:---:|"
    )
    for _, r in sweep.iterrows():
        lines.append(
            f"| {float(r['risk_pct']):.2f} | {float(r['worst_fold_dd_b_pct']):.3f} | "
            f"{float(r['worst_fold_roi_ann_pct']):.2f} | {float(r['mean_fold_roi_ann_pct']):.2f} | "
            f"{float(r['worst_daily_dd_b_pct']):.3f} | {float(r['daily_dd_p99_b_pct']):.3f} | "
            f"{float(r['daily_dd_p95_b_pct']):.3f} | "
            f"{'✓' if int(r['passes_fold_dd_ceiling']) else '✗'} | "
            f"{'✓' if int(r['passes_daily_dd_target']) else '✗'} | "
            f"{'✓' if int(r['passes_daily_dd_limit']) else '✗'} | "
            f"{'✓' if int(r['passes_deployable']) else '✗'} | "
            f"{'✓' if int(r['passes_viable']) else '✗'} |"
        )
    lines.append("")

    # Per-fold detail at recommended risk.
    if recommended_risk is not None:
        risk_bps = int(round(recommended_risk * 100))
        per_fold_csv = out_path.parent / f"fold_b_at_{risk_bps}bps_cluster_1.csv"
        if per_fold_csv.exists():
            pf = pd.read_csv(per_fold_csv)
            lines.append(f"## Per-fold detail at recommended risk = {recommended_risk}%")
            lines.append("")
            lines.append(
                "| F | n | mean_r | t-stat | fold_roi_b% | fold_roi_b_ann% | "
                "fold_dd_b% | worst_daily_dd% | exit_reason_dist |"
            )
            lines.append(
                "|---:|---:|---:|---:|---:|---:|---:|---:|---|"
            )
            for _, r in pf.iterrows():
                lines.append(
                    f"| F{int(r['fold'])} | {int(r['n'])} | {float(r['mean_r']):+.3f} | "
                    f"{float(r['t_stat']):+.2f} | {float(r['fold_roi_b_pct']):+.2f} | "
                    f"{float(r['fold_roi_b_ann_pct']):+.2f} | "
                    f"{float(r['fold_max_dd_b_pct']):.3f} | "
                    f"{float(r['worst_daily_dd_b_pct']):.3f} | {r['exit_reason_dist']} |"
                )
            lines.append("")

            # Daily DD distribution at recommended risk.
            dd_dist_csv = out_path.parent / f"daily_dd_distribution_at_{risk_bps}bps_cluster_1.csv"
            if dd_dist_csv.exists():
                dd_dist = pd.read_csv(dd_dist_csv)
                dds = dd_dist["daily_dd_pct"].dropna().to_numpy(dtype=float)
                lines.append(f"## Daily DD distribution at recommended risk = {recommended_risk}%")
                lines.append("")
                lines.append(f"- Total days observed (F2-F7): {len(dds)}")
                if len(dds) > 0:
                    lines.append(f"- Max daily DD: **{float(np.max(dds)):.3f}%**")
                    lines.append(f"- p99 daily DD: **{float(np.percentile(dds, 99)):.3f}%**")
                    lines.append(f"- p95 daily DD: {float(np.percentile(dds, 95)):.3f}%")
                    lines.append(f"- p90 daily DD: {float(np.percentile(dds, 90)):.3f}%")
                    lines.append(f"- p50 daily DD: {float(np.percentile(dds, 50)):.3f}%")
                    lines.append(f"- Days with daily DD > 4%: {int(np.sum(dds > 4.0))} "
                                 f"({float(100 * np.mean(dds > 4.0)):.2f}% of days)")
                    lines.append(f"- Days with daily DD > 5%: {int(np.sum(dds > 5.0))} "
                                 f"({float(100 * np.mean(dds > 5.0)):.2f}% of days) "
                                 f"(5ers account-closure threshold)")
                lines.append("")

    # Convention (b) vs (a) divergence note (informational, at recommended risk).
    if recommended_risk is not None:
        lines.append(
            f"## Convention (b) vs (a) divergence at recommended risk = {recommended_risk}%"
        )
        lines.append("")
        lines.append(
            "Original 5B reported convention-(a) DD inflation factors of +14% to +63% across "
            "F1-F7 (mean +28%). Convention (b) is the production-truth DD that captures "
            "concurrent floating-loss exposure across simultaneously-open positions; (a) "
            "treats each closed trade in isolation along entry_ts order, missing the "
            "cross-trade overlap. The 5B-refit recommended risk uses (b) directly — no "
            "conversion factor needed."
        )
        lines.append("")

    # Cross-arc observation.
    lines.append("## Cross-arc observation (informational)")
    lines.append("")
    lines.append(
        "The 5ers 5%/day daily DD limit binds against this signal class. The bar_range "
        "top-decile signal concentrates trade entries during high-volatility regime "
        "transitions, producing clusters of concurrently-open positions whose combined "
        "floating drawdown can exceed the daily limit before any individual position "
        "stops out. For deployment under 5ers, the deployable risk per trade is bounded "
        "below convention (a) by the daily DD constraint; this is a structural property "
        "of the signal, not of the protocol."
    )
    lines.append("")

    # Determinism.
    lines.append("## Determinism")
    lines.append("")
    lines.append("Two-run byte-identical sha256 over all written files:")
    lines.append("")
    lines.append("| File | Run 1 sha256 | Run 2 sha256 | Match |")
    lines.append("|---|---|---|---|")
    for fname in sorted(sha_run1.keys()):
        s1 = sha_run1[fname]
        s2 = sha_run2.get(fname) if sha_run2 else None
        match = "—" if s2 is None else ("PASS" if s1 == s2 else "FAIL")
        lines.append(f"| `{fname}` | `{s1[:16]}…` | `{(s2 or '—')[:16]}{'…' if s2 else ''}` | {match} |")
    lines.append("")
    lines.append(f"**Determinism: {determinism_gate}**")
    lines.append("")

    # Config sha256s.
    lines.append("## Input / config sha256s")
    lines.append("")
    for label, path in config_paths.items():
        lines.append(f"- `{label}` (`{path}`) — `{config_shas[label]}`")
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return headline, recommended_risk


# ============================================================================
# Main
# ============================================================================


def _env_dict() -> Dict[str, str]:
    return {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Arc 4 Step 5B-refit risk sweep — convention (b) MTM hourly equity, daily DD."
    )
    p.add_argument(
        "--sim-csv",
        type=Path,
        default=_REPO_ROOT / "results" / "l_arc_4" / "step5c" / "per_trade_simulated_refit_1.csv",
    )
    p.add_argument(
        "--paths-csv",
        type=Path,
        default=_REPO_ROOT / "results" / "l_arc_4" / "step1" / "trades_paths.csv",
    )
    p.add_argument(
        "--fold-defs-csv",
        type=Path,
        default=_REPO_ROOT / "results" / "l_arc_4" / "step5c" / "fold_definitions.csv",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO_ROOT / "results" / "l_arc_4" / "step5b_refit",
    )
    p.add_argument("--no-determinism-check", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    sim_csv = args.sim_csv.resolve()
    paths_csv = args.paths_csv.resolve()
    fold_defs_csv = args.fold_defs_csv.resolve()
    out_dir = args.out_dir.resolve()

    t0 = time.time()
    print("[5b-refit] === RUN 1 ===", file=sys.stderr)
    sha_run1 = run_once(sim_csv, paths_csv, fold_defs_csv, out_dir)
    print(f"[5b-refit] Run 1 done in {time.time() - t0:.1f}s", file=sys.stderr)

    sha_run2: Optional[Dict[str, str]] = None
    determinism_gate = "N/A"
    if not args.no_determinism_check:
        print("[5b-refit] === RUN 2 (determinism) ===", file=sys.stderr)
        t1 = time.time()
        sha_run2 = run_once(sim_csv, paths_csv, fold_defs_csv, out_dir)
        print(f"[5b-refit] Run 2 done in {time.time() - t1:.1f}s", file=sys.stderr)
        matched = all(
            sha_run1.get(k) == sha_run2.get(k) for k in sorted(set(sha_run1) | set(sha_run2))
        )
        determinism_gate = "PASS" if matched else "FAIL"

    config_paths = {
        "results/l_arc_4/step5c/per_trade_simulated_refit_1.csv": str(sim_csv.relative_to(_REPO_ROOT)),
        "results/l_arc_4/step1/trades_paths.csv": str(paths_csv.relative_to(_REPO_ROOT)),
        "results/l_arc_4/step5c/fold_definitions.csv": str(fold_defs_csv.relative_to(_REPO_ROOT)),
    }
    config_shas = {
        label: _file_sha256(_REPO_ROOT / Path(path)) for label, path in config_paths.items()
    }

    sweep_csv = out_dir / "risk_sweep_refit_b_cluster_1.csv"
    diag_path = out_dir / "step5b_refit_diagnostics.md"
    headline, rec_risk = write_diagnostics(
        diag_path,
        sweep_csv,
        sha_run1,
        sha_run2,
        determinism_gate,
        config_paths,
        config_shas,
    )

    print(f"[5b-refit] DONE recommended_risk={rec_risk} determinism={determinism_gate}",
          file=sys.stderr)
    print(f"[5b-refit] diagnostics → {diag_path}", file=sys.stderr)
    print(f"[5b-refit] headline: {headline[:200]}", file=sys.stderr)

    (out_dir / "step5b_refit_env.json").write_text(
        json.dumps(
            {
                "env": _env_dict(),
                "args": {k: str(v) for k, v in vars(args).items()},
                "recommended_risk_pct": rec_risk,
                "determinism_gate": determinism_gate,
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    return 0 if determinism_gate in ("PASS", "N/A") else 2


if __name__ == "__main__":
    raise SystemExit(main())
