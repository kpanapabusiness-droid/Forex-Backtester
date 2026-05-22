"""Arc 4 — Step 5B risk sweep on cluster 1 (L_ARC_PROTOCOL v2.1.1 §9 supplement).

Objective: find the lowest risk-per-trade level at which cluster 1's worst-fold
max DD comes under §10's 8% ceiling, and report worst-fold + mean-fold
annualised ROI at that risk. Also verify DD scales linearly with risk and
produce a convention-(b) (mark-to-market hourly equity) DD comparison at the
baseline 0.5% risk for transparency.

DD CONVENTION USED IN STEP 5 (confirmed by code inspection):
  Convention (a): max running drawdown over the sequence of CLOSED-trade PnLs
  (`final_r × risk × 100`) ordered by entry_ts within each fold. No
  concurrent-position floating-PnL aggregation.

  Implication: fold_max_dd from Step 5 MAY UNDERSTATE true DD because
  concurrent floating-loss exposure across simultaneously-open positions is
  not captured. This 5B run preserves convention (a) for the risk-sweep
  (so the linear-scaling argument holds), and adds convention (b) hourly
  mark-to-market DD per fold at 0.5% risk as a comparison only.

Inputs:
  - results/l_arc_4/step5/per_trade_simulated_1.csv (6,011 cluster-1 admitted trades)
  - results/l_arc_4/step1/trades_paths.csv (per-bar close prices for convention (b))
  - results/l_arc_4/step1/trades_all.csv (entry_price + atr per trade)
  - configs/wfo_kh24.yaml (fold OOS windows)

Outputs (results/l_arc_4/step5b/):
  - risk_sweep_cluster_1.csv
  - fold_roi_at_<risk_bps>_cluster_1.csv (one per risk level)
  - dd_convention_comparison.csv (at 0.5% only, since (a) was used in Step 5)
  - step5b_diagnostics.md

Determinism: deterministic per-fold arithmetic; no models, no randomness.
Two-run byte-identical for all outputs; both run sha256s logged.

DO NOT:
  - Re-simulate trades (use existing per_trade_simulated_1.csv).
  - Modify classifier / threshold / exit policy.
  - Touch cluster 3 (Step 5 eliminated it).
  - Address WFO leakage or pre-t SL bug — separate work items.

Usage:
  py scripts/l_arc_4/step5b_risk_sweep.py
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
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]

# ============================================================================
# Constants — risk grid, §10 thresholds
# ============================================================================

RISK_GRID_PCT: Tuple[float, ...] = (0.50, 0.40, 0.34, 0.30, 0.25, 0.20, 0.15, 0.10)
DD_CEILING_PCT: float = 8.0           # §10 worst-fold max DD ceiling

# §10 pass-deployable thresholds (annualised)
PASS_DEPLOY_WORST_ROI_ANN_PCT: float = 5.0
PASS_DEPLOY_MEAN_ROI_ANN_PCT: float = 8.0
PASS_DEPLOY_FULL_DATA_ROI_PCT: float = 5.0
PASS_DEPLOY_FULL_DATA_DD_PCT: float = 10.0

# §10 pass-viable (relaxed)
PASS_VIABLE_WORST_ROI_PCT: float = 0.0         # raw (sign), no annualisation requirement
PASS_VIABLE_MEAN_ROI_ANN_PCT: float = 3.0
PASS_VIABLE_FULL_DATA_ROI_PCT: float = 3.0
PASS_VIABLE_FULL_DATA_DD_PCT: float = 10.0

PATH_BARS: int = 241


# ============================================================================
# WFO fold structure
# ============================================================================


@dataclass
class WFOFold:
    fold: int
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp
    oos_days: int


def load_wfo_folds(wfo_cfg_path: Path) -> List[WFOFold]:
    cfg = yaml.safe_load(wfo_cfg_path.read_text(encoding="utf-8"))
    out: List[WFOFold] = []
    for entry in cfg["wfo"]["folds"]:
        s = pd.Timestamp(entry["oos_start"])
        e = pd.Timestamp(entry["oos_end"])
        out.append(WFOFold(fold=int(entry["fold"]), oos_start=s, oos_end=e, oos_days=(e - s).days))
    return out


# ============================================================================
# Convention (a) — Step 5 cum-sum DD, vectorised over risk grid
# ============================================================================


@dataclass
class FoldRiskMetrics:
    fold: int
    n: int
    risk_pct: float
    fold_roi_pct: float                # raw fold-window ROI (sum of final_r × risk%)
    fold_roi_annualised_pct: float
    fold_max_dd_pct: float
    oos_days: int
    annualisation_factor: float


def evaluate_fold_at_risk(
    fold_trades_sorted: pd.DataFrame, risk_pct: float, fold: WFOFold
) -> FoldRiskMetrics:
    """Convention (a): cum-sum of closed-trade R × risk%, then max running DD."""
    n = len(fold_trades_sorted)
    if n == 0:
        return FoldRiskMetrics(
            fold=fold.fold, n=0, risk_pct=risk_pct, fold_roi_pct=0.0,
            fold_roi_annualised_pct=0.0, fold_max_dd_pct=0.0,
            oos_days=fold.oos_days, annualisation_factor=365.0 / fold.oos_days,
        )
    rs = fold_trades_sorted["final_r"].to_numpy(dtype=float)
    per_trade_pct = rs * (risk_pct / 100.0) * 100.0   # = rs × risk_pct
    cum = np.cumsum(per_trade_pct)
    running_peak = np.maximum.accumulate(cum)
    dd = running_peak - cum
    max_dd = float(dd.max())
    fold_roi = float(cum[-1])
    factor = 365.0 / fold.oos_days
    fold_roi_ann = fold_roi * factor
    return FoldRiskMetrics(
        fold=fold.fold, n=n, risk_pct=risk_pct, fold_roi_pct=fold_roi,
        fold_roi_annualised_pct=fold_roi_ann, fold_max_dd_pct=max_dd,
        oos_days=fold.oos_days, annualisation_factor=factor,
    )


# ============================================================================
# Convention (b) — Mark-to-market hourly equity DD per fold, at 0.5% risk
# ============================================================================


def compute_convention_b_dd_per_fold(
    sim_trades: pd.DataFrame,
    folds: List[WFOFold],
    trades_all: pd.DataFrame,
    path_tensor_closes: np.ndarray,
    path_tensor_times: np.ndarray,
    trade_id_to_idx: Dict[int, int],
    risk_pct: float = 0.50,
) -> Dict[int, float]:
    """Build per-fold hourly mark-to-market equity series and return per-fold max DD.

    For each cluster-1 admitted trade T (already in `sim_trades`):
      - bars 0..exit_bar: floating R = (close_at_bar - entry_price) / cluster_R
      - exit_bar onwards: realized final_r (constant)
      - Before entry_ts or after fold's oos_end: not contributing within this fold
    """
    risk_scale = risk_pct / 100.0
    # Build per-trade contribution time series (using cluster_R, entry_price from sim_trades).
    per_fold_dd: Dict[int, float] = {}

    for f in folds:
        f_trades = sim_trades[sim_trades["fold"] == f.fold]
        if f_trades.empty:
            per_fold_dd[f.fold] = 0.0
            continue

        # Build common 1H timeline for the fold.
        timeline = pd.date_range(
            start=f.oos_start, end=f.oos_end - pd.Timedelta(hours=1), freq="1h"
        )
        if len(timeline) == 0:
            per_fold_dd[f.fold] = 0.0
            continue

        # Total equity at each hour = sum across trades of (floating or realized R)
        # × risk_scale (in fraction of account).
        total_equity = pd.Series(0.0, index=timeline)

        for _, row in f_trades.iterrows():
            tid = int(row["trade_id"])
            entry_price = float(row["entry_price"])
            cluster_R = float(row["cluster_R"])
            exit_bar = int(row["exit_bar"])
            final_r = float(row["final_r"])
            t_idx = trade_id_to_idx[tid]
            bar_times = path_tensor_times[t_idx]    # 241 timestamps (np.datetime64)
            bar_closes = path_tensor_closes[t_idx]  # 241 closes

            # Build a per-bar contribution series for this trade:
            # bars 0..exit_bar: floating_r (or realized at exit bar — we use
            # the bar-close-based floating R as a close-of-bar proxy; realized
            # final_r differs by spread / SL price vs bar close, small).
            # bars > exit_bar: constant final_r.
            held_r = (bar_closes[: exit_bar + 1] - entry_price) / cluster_R
            held_times = pd.to_datetime(bar_times[: exit_bar + 1])
            held_series = pd.Series(held_r, index=held_times)

            # After exit_bar: extend with final_r at exit_bar + 1's timestamp through
            # min(entry_ts + 240h, fold.oos_end).
            after_times = pd.to_datetime(bar_times[exit_bar + 1 :])
            if len(after_times) > 0:
                after_series = pd.Series(np.full(len(after_times), final_r), index=after_times)
                contrib = pd.concat([held_series, after_series])
            else:
                contrib = held_series

            # Reindex onto fold timeline using forward-fill from each bar's
            # close-of-bar timestamp. Trades not yet entered → 0; after path
            # window ends but within fold → keep final_r.
            contrib = contrib.sort_index()
            # Reindex to fold timeline; use method="ffill" but only for timestamps
            # at or after the first contribution timestamp.
            contrib_on_grid = contrib.reindex(timeline, method="ffill").fillna(0.0)
            # For timestamps before the first contribution (entry bar close), 0.0.
            # For timestamps after the last contribution (path-window cap),
            # ffill keeps final_r unchanged (good — trade stays closed).
            total_equity += contrib_on_grid

        # Scale by risk and convert to %.
        equity_pct = total_equity.to_numpy() * risk_scale * 100.0
        running_peak = np.maximum.accumulate(equity_pct)
        dd = running_peak - equity_pct
        per_fold_dd[f.fold] = float(dd.max())

    return per_fold_dd


# ============================================================================
# CSV writers (deterministic)
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
    trades_all_csv: Path,
    wfo_cfg_path: Path,
    out_dir: Path,
) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    folds = load_wfo_folds(wfo_cfg_path)

    sim = pd.read_csv(sim_csv)
    sim["entry_ts"] = pd.to_datetime(sim["entry_ts"])
    sim = sim.sort_values("entry_ts").reset_index(drop=True)

    # Per-fold ordered trade lists (matches Step 5: order by entry_ts).
    fold_trade_subsets: Dict[int, pd.DataFrame] = {}
    for f in folds:
        sub = sim[sim["fold"] == f.fold].sort_values("entry_ts").reset_index(drop=True)
        fold_trade_subsets[f.fold] = sub

    # ----- Convention (a) risk sweep -----
    sweep_rows: List[Dict[str, Any]] = []
    sha_files: Dict[str, str] = {}
    full_data_roi_at_risk: Dict[float, float] = {}
    full_data_dd_at_risk: Dict[float, float] = {}
    per_fold_dd_at_risk_a: Dict[float, Dict[int, float]] = {}

    for risk in RISK_GRID_PCT:
        per_fold_rows: List[Dict[str, Any]] = []
        per_fold_dd: Dict[int, float] = {}
        for f in folds:
            m = evaluate_fold_at_risk(fold_trade_subsets[f.fold], risk, f)
            per_fold_rows.append(
                {
                    "cluster_id": 1,
                    "risk_pct": risk,
                    "fold": m.fold,
                    "n": m.n,
                    "oos_days": m.oos_days,
                    "annualisation_factor": _fmt_g(m.annualisation_factor),
                    "fold_roi_pct": _fmt_g(m.fold_roi_pct),
                    "fold_roi_annualised_pct": _fmt_g(m.fold_roi_annualised_pct),
                    "fold_max_dd_pct": _fmt_g(m.fold_max_dd_pct),
                    "mean_r": _fmt_g(
                        fold_trade_subsets[f.fold]["final_r"].mean()
                        if not fold_trade_subsets[f.fold].empty
                        else 0.0
                    ),
                }
            )
            per_fold_dd[f.fold] = m.fold_max_dd_pct
        per_fold_dd_at_risk_a[risk] = per_fold_dd

        # Full-data ROI / DD: concatenate all folds in chronological order.
        all_trades = sim.sort_values("entry_ts").reset_index(drop=True)
        rs_all = all_trades["final_r"].to_numpy(dtype=float)
        per_trade_pct_all = rs_all * (risk / 100.0) * 100.0
        cum_all = np.cumsum(per_trade_pct_all)
        running_peak_all = np.maximum.accumulate(cum_all)
        dd_all = running_peak_all - cum_all
        full_roi = float(cum_all[-1]) if cum_all.size else 0.0
        full_dd = float(dd_all.max()) if dd_all.size else 0.0
        # Annualise full-data ROI by pool span.
        span_days = (all_trades["entry_ts"].iloc[-1] - all_trades["entry_ts"].iloc[0]).days
        full_roi_ann = full_roi * (365.0 / span_days) if span_days > 0 else 0.0
        full_data_roi_at_risk[risk] = full_roi_ann
        full_data_dd_at_risk[risk] = full_dd

        # Aggregate (worst-fold, mean-fold).
        worst_dd = max(per_fold_dd.values())
        non_empty_metrics = [
            evaluate_fold_at_risk(fold_trade_subsets[f.fold], risk, f) for f in folds
        ]
        non_empty_metrics = [m for m in non_empty_metrics if m.n > 0]
        worst_roi = min(m.fold_roi_pct for m in non_empty_metrics) if non_empty_metrics else 0.0
        worst_roi_ann = (
            min(m.fold_roi_annualised_pct for m in non_empty_metrics)
            if non_empty_metrics
            else 0.0
        )
        mean_roi = (
            float(np.mean([m.fold_roi_pct for m in non_empty_metrics]))
            if non_empty_metrics
            else 0.0
        )
        mean_roi_ann = (
            float(np.mean([m.fold_roi_annualised_pct for m in non_empty_metrics]))
            if non_empty_metrics
            else 0.0
        )

        passes_dd_ceiling = worst_dd <= DD_CEILING_PCT
        passes_deployable = (
            passes_dd_ceiling
            and worst_roi_ann >= PASS_DEPLOY_WORST_ROI_ANN_PCT
            and mean_roi_ann >= PASS_DEPLOY_MEAN_ROI_ANN_PCT
            and full_roi_ann >= PASS_DEPLOY_FULL_DATA_ROI_PCT
            and full_dd <= PASS_DEPLOY_FULL_DATA_DD_PCT
        )
        passes_viable = (
            worst_roi > PASS_VIABLE_WORST_ROI_PCT
            and worst_dd <= DD_CEILING_PCT
            and mean_roi_ann >= PASS_VIABLE_MEAN_ROI_ANN_PCT
            and full_roi_ann >= PASS_VIABLE_FULL_DATA_ROI_PCT
            and full_dd <= PASS_VIABLE_FULL_DATA_DD_PCT
        )
        sweep_rows.append(
            {
                "cluster_id": 1,
                "risk_pct": risk,
                "worst_fold_DD_pct": _fmt_g(worst_dd),
                "worst_fold_ROI_pct": _fmt_g(worst_roi),
                "worst_fold_ROI_annualised_pct": _fmt_g(worst_roi_ann),
                "mean_fold_ROI_pct": _fmt_g(mean_roi),
                "mean_fold_ROI_annualised_pct": _fmt_g(mean_roi_ann),
                "full_data_ROI_annualised_pct": _fmt_g(full_roi_ann),
                "full_data_max_DD_pct": _fmt_g(full_dd),
                "passes_dd_ceiling": int(passes_dd_ceiling),
                "passes_deployable": int(passes_deployable),
                "passes_viable": int(passes_viable),
            }
        )

        per_fold_path = out_dir / f"fold_roi_at_{int(round(risk * 100))}bps_cluster_1.csv"
        _write_rows(per_fold_rows, per_fold_path)
        sha_files[per_fold_path.name] = _file_sha256(per_fold_path)

    sweep_path = out_dir / "risk_sweep_cluster_1.csv"
    _write_rows(sweep_rows, sweep_path)
    sha_files[sweep_path.name] = _file_sha256(sweep_path)

    # ----- Convention (b) MTM equity DD at 0.5% risk -----
    print("[l_arc_4 step5b] Computing convention (b) MTM DD at 0.5% risk...", file=sys.stderr)
    paths_df = pd.read_csv(paths_csv, usecols=["trade_id", "bar_offset", "timestamp", "close"])
    paths_df = paths_df.sort_values(["trade_id", "bar_offset"]).reset_index(drop=True)
    # Per-trade 241-bar timeline of (timestamp, close). Build only for cluster-1 admitted trades.
    admitted_tids = sim["trade_id"].astype(int).unique().tolist()
    paths_sub = paths_df[paths_df["trade_id"].isin(admitted_tids)].copy()
    paths_sub["timestamp"] = pd.to_datetime(paths_sub["timestamp"])
    # Reshape into tensors (n_admitted, 241).
    # Stable order: by trade_id ascending.
    paths_sub = paths_sub.sort_values(["trade_id", "bar_offset"]).reset_index(drop=True)
    n_adm = len(admitted_tids)
    if len(paths_sub) != n_adm * PATH_BARS:
        raise ValueError(
            f"Per-trade path expansion mismatch: got {len(paths_sub)} rows, "
            f"expected {n_adm}×{PATH_BARS}"
        )
    closes_2d = paths_sub["close"].to_numpy(dtype=float).reshape(n_adm, PATH_BARS)
    times_2d = paths_sub["timestamp"].to_numpy().reshape(n_adm, PATH_BARS)
    tid_2d = paths_sub["trade_id"].to_numpy(dtype=int).reshape(n_adm, PATH_BARS)[:, 0]
    trade_id_to_idx = {int(t): i for i, t in enumerate(tid_2d)}

    trades_all = pd.read_csv(trades_all_csv, usecols=["trade_id", "entry_price", "atr_14_at_signal"])

    per_fold_dd_b = compute_convention_b_dd_per_fold(
        sim, folds, trades_all, closes_2d, times_2d, trade_id_to_idx, risk_pct=0.50
    )

    # Build dd_convention_comparison.csv.
    comp_rows: List[Dict[str, Any]] = []
    for f in folds:
        dd_a = per_fold_dd_at_risk_a[0.50][f.fold]
        dd_b = per_fold_dd_b[f.fold]
        comp_rows.append(
            {
                "fold": f.fold,
                "oos_start": str(f.oos_start.date()),
                "oos_end": str(f.oos_end.date()),
                "dd_convention_a_pct": _fmt_g(dd_a),
                "dd_convention_b_pct": _fmt_g(dd_b),
                "delta_pct": _fmt_g(dd_b - dd_a),
                "delta_relative": _fmt_g(((dd_b - dd_a) / dd_a) if dd_a > 0 else float("nan")),
            }
        )
    comp_path = out_dir / "dd_convention_comparison.csv"
    _write_rows(comp_rows, comp_path)
    sha_files[comp_path.name] = _file_sha256(comp_path)

    return sha_files


# ============================================================================
# Diagnostics writer
# ============================================================================


def write_diagnostics(
    out_path: Path,
    sha_run1: Dict[str, str],
    sha_run2: Optional[Dict[str, str]],
    determinism_gate: str,
    sweep_csv: Path,
    comp_csv: Path,
    config_paths: Dict[str, str],
    config_shas: Dict[str, str],
) -> str:
    sweep = pd.read_csv(sweep_csv)
    comp = pd.read_csv(comp_csv)

    # Identify lowest passing risk (DD ≤ 8%).
    passing_dd = sweep[sweep["passes_dd_ceiling"] == 1]
    passing_dep = sweep[sweep["passes_deployable"] == 1]
    lowest_dd = passing_dd["risk_pct"].min() if not passing_dd.empty else None
    highest_dep = passing_dep["risk_pct"].max() if not passing_dep.empty else None
    headline_disposition = (
        "PASS-DEPLOYABLE" if highest_dep is not None and lowest_dd is not None
        else "PARTIAL" if lowest_dd is not None else "FAIL"
    )

    # Linear scaling check.
    baseline = sweep[sweep["risk_pct"] == 0.50]
    if not baseline.empty:
        base_dd = float(baseline["worst_fold_DD_pct"].iloc[0])
        scaling_check_rows: List[Tuple[float, float, float]] = []
        for _, r in sweep.iterrows():
            expected = base_dd * (float(r["risk_pct"]) / 0.50)
            actual = float(r["worst_fold_DD_pct"])
            scaling_check_rows.append((float(r["risk_pct"]), expected, actual))
    else:
        scaling_check_rows = []

    lines: List[str] = []
    lines.append("# Arc 4 — Step 5B risk sweep on cluster 1 — diagnostics")
    lines.append("")
    lines.append("Protocol: `L_ARC_PROTOCOL.md` v2.1.1 §9 supplement; §10 deploy/viable thresholds")
    lines.append("Cluster: 1 (`stepwise_pullback_extended`, R = 3.0 × ATR; passed §9 at Step 5)")
    lines.append("Input: `results/l_arc_4/step5/per_trade_simulated_1.csv` (6,011 admitted trades)")
    lines.append("")

    # Headline.
    lines.append("## Headline")
    lines.append("")
    if lowest_dd is not None:
        lines.append(
            f"Lowest risk-per-trade passing §10's 8% worst-fold DD ceiling: "
            f"**{lowest_dd}% per trade**."
        )
    else:
        lines.append(
            "No risk level in the grid {0.50, 0.40, 0.34, 0.30, 0.25, 0.20, 0.15, 0.10}% "
            "passes the 8% worst-fold DD ceiling — cluster 1 cannot pass §10's deploy DD gate even at the smallest tested risk."
        )
    if highest_dep is not None:
        dep_row = sweep[sweep["risk_pct"] == highest_dep].iloc[0]
        lines.append(
            f"Highest risk-per-trade still PASS-DEPLOYABLE on all §10 gates "
            f"(DD ≤ 8%, worst-fold annualised ROI ≥ 5%, mean-fold annualised ROI ≥ 8%, "
            f"full-data ROI ann ≥ 5%, full-data DD ≤ 10%): **{highest_dep}% per trade** — "
            f"worst-fold ann ROI {dep_row['worst_fold_ROI_annualised_pct']}%, "
            f"mean-fold ann ROI {dep_row['mean_fold_ROI_annualised_pct']}%, "
            f"worst-fold DD {dep_row['worst_fold_DD_pct']}%."
        )
    lines.append(f"**Overall disposition (within risk-grid): {headline_disposition}**")
    lines.append(f"Determinism: **{determinism_gate}**.")
    lines.append("")

    # DD convention statement.
    lines.append("## DD convention statement")
    lines.append("")
    lines.append(
        "Step 5's `fold_max_dd_pct` was computed using **convention (a)**: "
        "max running drawdown over the cum-sum of CLOSED-trade PnLs "
        "(`final_r × risk × 100`) ordered by entry_ts within each fold. "
        "No concurrent-position floating-PnL aggregation."
    )
    lines.append("")
    lines.append(
        "**Caveat (per prompt)**: this convention may UNDERSTATE the true peak-to-trough drawdown because "
        "it does not aggregate floating losses across simultaneously-open positions. "
        "5B preserves convention (a) for the risk-sweep (so the linear-scaling argument holds; DD scales "
        "perfectly with risk under (a)), and reports convention (b) hourly mark-to-market equity DD per fold "
        "at 0.50% risk for comparison only."
    )
    lines.append("")
    lines.append(
        "**5B SELECTION RULE**: use convention (a) DD as the gate. Convention (b) is INFORMATIONAL only — "
        "do NOT change the selected risk level on the basis of (b) alone; flag for chat-level review."
    )
    lines.append("")

    # Annualisation table.
    lines.append("## Annualisation factor per fold")
    lines.append("")
    # Read one of the per-fold files to extract OOS days.
    sample_pf = pd.read_csv(out_path.parent / "fold_roi_at_50bps_cluster_1.csv")
    lines.append("| Fold | OOS days | Annualisation factor (365/days) |")
    lines.append("|---:|---:|---:|")
    for _, r in sample_pf.iterrows():
        lines.append(
            f"| F{int(r['fold'])} | {int(r['oos_days'])} | {float(r['annualisation_factor']):.4f} |"
        )
    lines.append("")

    # Risk-sweep table.
    lines.append("## Risk sweep — convention (a) DD ordering by entry_ts")
    lines.append("")
    lines.append(
        "| Risk% | worst_fold DD% | worst_fold ROI% | worst_fold ROI ann% | "
        "mean_fold ROI% | mean_fold ROI ann% | full-data ROI ann% | full-data DD% | "
        "DD ≤ 8% | pass-deploy | pass-viable |"
    )
    lines.append(
        "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|"
    )
    for _, r in sweep.iterrows():
        lines.append(
            f"| {float(r['risk_pct']):.2f} | {float(r['worst_fold_DD_pct']):.3f} | "
            f"{float(r['worst_fold_ROI_pct']):.2f} | {float(r['worst_fold_ROI_annualised_pct']):.2f} | "
            f"{float(r['mean_fold_ROI_pct']):.2f} | {float(r['mean_fold_ROI_annualised_pct']):.2f} | "
            f"{float(r['full_data_ROI_annualised_pct']):.2f} | {float(r['full_data_max_DD_pct']):.3f} | "
            f"{'✓' if int(r['passes_dd_ceiling']) else '✗'} | "
            f"{'✓' if int(r['passes_deployable']) else '✗'} | "
            f"{'✓' if int(r['passes_viable']) else '✗'} |"
        )
    lines.append("")

    # Linear-scaling sanity check.
    lines.append("## Linear DD scaling sanity check")
    lines.append("")
    lines.append("Under convention (a), DD scales perfectly linearly with risk%: every fold's DD multiplies by the risk-ratio.")
    lines.append("")
    lines.append("| Risk% | Expected DD% (= base × ratio) | Actual DD% | Δ |")
    lines.append("|---:|---:|---:|---:|")
    for risk, expected, actual in scaling_check_rows:
        delta = actual - expected
        lines.append(f"| {risk:.2f} | {expected:.6f} | {actual:.6f} | {delta:+.6e} |")
    lines.append("")
    deviations = [abs(a - e) > 1e-6 for _, e, a in scaling_check_rows]
    if any(deviations):
        lines.append("**Deviation detected** — DD does not scale linearly. Investigate.")
    else:
        lines.append("All deltas |≤ 1e-6%| — **DD scales linearly as expected** under convention (a).")
    lines.append("")

    # Convention (b) comparison.
    lines.append("## Convention (b) hourly mark-to-market DD comparison (at 0.50% risk)")
    lines.append("")
    lines.append(
        "For each fold, build an hourly equity curve summing across all admitted trades' floating R "
        "(bars 0..exit_bar) and realized final_r (after exit_bar), scaled by risk × 100. "
        "DD = max running drawdown on this hourly equity. Compare to convention (a) DD at the same risk."
    )
    lines.append("")
    lines.append("| Fold | OOS window | DD (a) % | DD (b) % | Δ pp | Δ relative |")
    lines.append("|---:|---|---:|---:|---:|---:|")
    for _, r in comp.iterrows():
        dd_a = float(r["dd_convention_a_pct"])
        dd_b = float(r["dd_convention_b_pct"])
        delta = float(r["delta_pct"])
        rel = r["delta_relative"]
        rel_str = f"{float(rel):+.2%}" if pd.notna(rel) and str(rel) != "" else "—"
        lines.append(
            f"| F{int(r['fold'])} | {r['oos_start']} → {r['oos_end']} | "
            f"{dd_a:.3f} | {dd_b:.3f} | {delta:+.3f} | {rel_str} |"
        )
    max_delta_pp = comp["delta_pct"].max()
    max_delta_rel = comp["delta_relative"].max()
    lines.append("")
    lines.append(
        f"Maximum absolute DD inflation under (b) vs (a): **{max_delta_pp:+.3f} pp** "
        f"(relative {(float(max_delta_rel) if pd.notna(max_delta_rel) else float('nan')):+.2%})."
    )
    lines.append(
        "Convention (b) IS expected to inflate DD because concurrent floating losses across "
        "simultaneously-open positions are not netted out by closed-trade timing. "
        "**This is informational** — the 5B risk-level selection uses convention (a)."
    )
    lines.append("")
    lines.append(
        "**Implication for chat review**: if convention (b) reports DD inflation > 50%, "
        "deploy-time DD may exceed the convention-(a)-derived risk-level selection. "
        "Future Step 5C / engine work item: implement convention (b) DD natively in the deployed engine "
        "for production-accurate DD tracking."
    )
    lines.append("")

    # Determinism log.
    lines.append("## Determinism")
    lines.append("")
    lines.append("Two-run byte-identical sha256 over all written CSVs:")
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

    # Input / config sha256s.
    lines.append("## Input / config sha256s")
    lines.append("")
    for label, path in config_paths.items():
        lines.append(f"- `{label}` (`{path}`) — `{config_shas[label]}`")
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return headline_disposition


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
        description="Arc 4 Step 5B risk sweep on cluster 1 (L_ARC_PROTOCOL v2.1.1 §9 supplement)."
    )
    p.add_argument(
        "--sim-csv",
        type=Path,
        default=_REPO_ROOT / "results" / "l_arc_4" / "step5" / "per_trade_simulated_1.csv",
    )
    p.add_argument(
        "--paths-csv",
        type=Path,
        default=_REPO_ROOT / "results" / "l_arc_4" / "step1" / "trades_paths.csv",
    )
    p.add_argument(
        "--trades-all-csv",
        type=Path,
        default=_REPO_ROOT / "results" / "l_arc_4" / "step1" / "trades_all.csv",
    )
    p.add_argument(
        "--wfo-config",
        type=Path,
        default=_REPO_ROOT / "configs" / "wfo_kh24.yaml",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO_ROOT / "results" / "l_arc_4" / "step5b",
    )
    p.add_argument("--no-determinism-check", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    sim_csv = args.sim_csv.resolve()
    paths_csv = args.paths_csv.resolve()
    trades_all_csv = args.trades_all_csv.resolve()
    wfo_cfg_path = args.wfo_config.resolve()
    out_dir = args.out_dir.resolve()

    t0 = time.time()
    print("[l_arc_4 step5b] === RUN 1 ===", file=sys.stderr)
    sha_run1 = run_once(sim_csv, paths_csv, trades_all_csv, wfo_cfg_path, out_dir)
    print(f"[l_arc_4 step5b] Run 1 done in {time.time() - t0:.1f}s", file=sys.stderr)

    sha_run2: Optional[Dict[str, str]] = None
    determinism_gate = "N/A"
    if not args.no_determinism_check:
        print("[l_arc_4 step5b] === RUN 2 (determinism) ===", file=sys.stderr)
        t1 = time.time()
        sha_run2 = run_once(sim_csv, paths_csv, trades_all_csv, wfo_cfg_path, out_dir)
        print(f"[l_arc_4 step5b] Run 2 done in {time.time() - t1:.1f}s", file=sys.stderr)
        matched = all(
            sha_run1.get(k) == sha_run2.get(k) for k in sorted(set(sha_run1) | set(sha_run2))
        )
        determinism_gate = "PASS" if matched else "FAIL"

    config_paths = {
        "configs/wfo_kh24.yaml": str(wfo_cfg_path.relative_to(_REPO_ROOT)),
        "results/l_arc_4/step1/trades_all.csv": str(trades_all_csv.relative_to(_REPO_ROOT)),
        "results/l_arc_4/step1/trades_paths.csv": str(paths_csv.relative_to(_REPO_ROOT)),
        "results/l_arc_4/step5/per_trade_simulated_1.csv": str(sim_csv.relative_to(_REPO_ROOT)),
    }
    config_shas = {
        label: _file_sha256(_REPO_ROOT / Path(path)) for label, path in config_paths.items()
    }

    diag_path = out_dir / "step5b_diagnostics.md"
    headline = write_diagnostics(
        diag_path,
        sha_run1,
        sha_run2,
        determinism_gate,
        sweep_csv=out_dir / "risk_sweep_cluster_1.csv",
        comp_csv=out_dir / "dd_convention_comparison.csv",
        config_paths=config_paths,
        config_shas=config_shas,
    )

    print(
        f"[l_arc_4 step5b] DONE disposition={headline} determinism={determinism_gate}",
        file=sys.stderr,
    )
    print(f"[l_arc_4 step5b] diagnostics → {diag_path}", file=sys.stderr)

    (out_dir / "step5b_env.json").write_text(
        json.dumps(
            {
                "env": _env_dict(),
                "args": {k: str(v) for k, v in vars(args).items()},
                "headline_disposition": headline,
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
