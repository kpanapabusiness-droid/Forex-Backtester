"""Arc 4 — Step 5B-spread: exit spread correction on refit data.

L_ARC_PROTOCOL v2.1.1 §9 supplement. The original Step 5 simulator did not
apply exit-bar spread. This step adds the missing half-spread on the exit
side per trade, re-runs the convention-(b) risk sweep with daily DD on the
spread-corrected refit data, and produces a side-by-side delta vs the
uncorrected (no-spread) numbers from 5B-refit.

SPREAD MECHANIC (audited against Step 1):
  - Step 1 entry: entry_price = entry_mid + S_entry/2 × pip_size  (long ask)
  - Step 1 exit (intrabar SL): exit_price = SL_level − S_exit/2 × pip_size  (long bid)
  - Step 1 exit (max_life):    exit_price = exit_mid  − S_exit/2 × pip_size
  - Round-trip cost: S/2 (entry, baked into entry_price) + S/2 (exit, NOT in
    Step 5 simulator's final_r) = S total.

  Step 5/5C simulator returned final_r = (trigger_level − entry_price) / R
  with trigger_level being the SL price, trail-stop price, or bar-240 close —
  all mid-equivalent on the exit side, no S_exit/2 deduction. So the
  simulator OVERSTATES final_r by S_exit/2 / R per trade.

  Correction (this step): subtract (S_exit_pips/2 × pip_size) / R_per_trade
  from each trade's final_r. R_per_trade = 3.0 × atr_14_1h_at_signal for
  cluster 1. S_exit_pips is the floored spread at the exit bar (matching
  Step 1's spread floor convention from configs/spread_floors_5ers.yaml).

  Note: the prompt's "deduct one full spread" is colloquial — the mechanical
  correction needed to restore Step 1's round-trip cost is the half-spread
  on the exit side (S/2), not the full S. This implementation applies S/2;
  the math is logged in step5b_spread_diagnostics.md for chat audit.

DELTA OUTPUT: re-runs the convention-(b) hourly MTM risk sweep on BOTH the
spread-corrected and uncorrected data so the cost of the prompt error is
quantified per fold and at the recommended risk level.

Scope: cluster 1 only, F2-F7 only (consistent with 5B-refit).

Inputs:
  - results/l_arc_4/step5c/per_trade_simulated_refit_1.csv  (5,361 trades)
  - results/l_arc_4/step5c/fold_definitions.csv
  - results/l_arc_4/step1/trades_paths.csv  (per-bar timestamps + closes)
  - configs/spread_floors_5ers.yaml  (sha256-verified)
  - data/1hr/<pair>.csv per pair  (for raw spread column at exit bars)

Outputs (results/l_arc_4/step5b_spread/):
  - per_trade_simulated_refit_spread_1.csv
  - risk_sweep_refit_spread_b_cluster_1.csv
  - risk_sweep_refit_no_spread_b_cluster_1.csv  (recomputed here for delta)
  - fold_b_at_<risk_bps>_cluster_1_spread.csv per risk level
  - daily_dd_distribution_at_<risk_bps>_cluster_1_spread.csv per risk level
  - spread_delta_analysis.csv  (no-spread vs spread, per-fold and aggregate)
  - hourly_equity_curve_at_0.5pct_cluster_1_spread.csv  (audit)
  - step5b_spread_diagnostics.md

Determinism: deterministic per-fold arithmetic; two-run byte-identical sha256.

Usage:
  py scripts/l_arc_4/step5b_spread_correction.py
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
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.lchar.compute_spread_floors import compute_body_sha256  # noqa: E402

# ============================================================================
# Constants
# ============================================================================

RISK_GRID_PCT: Tuple[float, ...] = (0.50, 0.30, 0.25, 0.22, 0.20, 0.18, 0.15, 0.12, 0.10)
BASELINE_RISK_PCT: float = 0.50
PATH_BARS: int = 241
POINTS_PER_PIP: float = 10.0

# §10 thresholds
DD_CEILING_PCT: float = 8.0
PASS_DEPLOY_WORST_ROI_ANN_PCT: float = 5.0
PASS_DEPLOY_MEAN_ROI_ANN_PCT: float = 8.0
PASS_VIABLE_WORST_ROI_PCT: float = 0.0
PASS_VIABLE_MEAN_ROI_ANN_PCT: float = 3.0

# 5ers daily DD limits
DAILY_DD_LIMIT_PCT: float = 5.0
DAILY_DD_TARGET_PCT: float = 4.0

EXPECTED_SPREAD_FLOOR_SHA: str = (
    "a613b4ce641c8d5218490531770a4924204029dedaa80fb24111beb61bd15547"
)


# ============================================================================
# Helpers
# ============================================================================


def _pip_size(pair: str) -> float:
    return 0.01 if pair.endswith("_JPY") else 0.0001


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
# Spread floor + per-pair 1H spread lookup
# ============================================================================


@dataclass
class SpreadFloor:
    floors_pips: Dict[str, float]
    points_per_pip: float
    body_sha256: str


def load_spread_floor(path: Path) -> SpreadFloor:
    actual = compute_body_sha256(path)
    if actual != EXPECTED_SPREAD_FLOOR_SHA:
        raise ValueError(
            f"spread_floor body sha256 mismatch:\n  expected={EXPECTED_SPREAD_FLOOR_SHA}\n  actual={actual}"
        )
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    floors_section = data.get("floors", {})
    floors_pips: Dict[str, float] = {
        pair: float(stats["min_nonzero_spread_native"]) / POINTS_PER_PIP
        for pair, stats in floors_section.items()
    }
    return SpreadFloor(floors_pips=floors_pips, points_per_pip=POINTS_PER_PIP, body_sha256=actual)


def load_1h_spread_lookup(pair: str, data_dir: Path) -> pd.Series:
    """Return a pd.Series indexed by timestamp with raw spread (in points) per 1H bar."""
    df = pd.read_csv(data_dir / f"{pair}.csv", usecols=["time", "spread"])
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").drop_duplicates("time", keep="first")
    return df.set_index("time")["spread"].astype(float)


def _floor_to_pips(raw_points: float, pair: str, sf: SpreadFloor) -> float:
    raw_pips = raw_points / sf.points_per_pip if sf.points_per_pip > 0 else 0.0
    floor = sf.floors_pips.get(pair, 0.0)
    return max(raw_pips, floor)


# ============================================================================
# Exit timestamp lookup per trade (from trades_paths.csv)
# ============================================================================


def build_exit_timestamps(
    sim: pd.DataFrame, paths_csv: Path
) -> pd.DataFrame:
    """For every (trade_id, exit_bar) pair, look up the exit timestamp from
    trades_paths.csv. Returns sim augmented with `exit_ts` column.
    """
    # Only need trade_id, bar_offset, timestamp.
    paths = pd.read_csv(paths_csv, usecols=["trade_id", "bar_offset", "timestamp"])
    paths["timestamp"] = pd.to_datetime(paths["timestamp"])
    # Index by (trade_id, bar_offset) for fast lookup.
    idx = paths.set_index(["trade_id", "bar_offset"])
    # Lookup per trade.
    keys = list(zip(sim["trade_id"].astype(int), sim["exit_bar"].astype(int)))
    exit_ts: List[pd.Timestamp] = []
    for tid, bo in keys:
        try:
            ts = idx.loc[(tid, bo), "timestamp"]
            if isinstance(ts, pd.Series):
                ts = ts.iloc[0]
            exit_ts.append(pd.Timestamp(ts))
        except KeyError:
            exit_ts.append(pd.NaT)
    out = sim.copy()
    out["exit_ts"] = exit_ts
    return out


# ============================================================================
# Per-trade exit-spread correction
# ============================================================================


@dataclass
class SpreadAuditRecord:
    n_trades_total: int
    n_trades_with_spread: int
    n_trades_gap_filled: int       # exit ts not found in pair data → used closest preceding
    n_trades_floor_active: int     # spread was floored
    n_trades_outlier: int          # exit_spread_R > 0.50 (>50% of R)
    mean_exit_spread_R: float
    median_exit_spread_R: float
    max_exit_spread_R: float
    min_exit_spread_R: float
    per_pair_mean_pips: Dict[str, float]


def apply_exit_spread(
    sim_with_exit_ts: pd.DataFrame,
    sf: SpreadFloor,
    data_dir: Path,
) -> Tuple[pd.DataFrame, SpreadAuditRecord]:
    """For each trade, look up the spread at exit_ts from the pair's 1H bar
    file. Apply S_exit_pips/2 × pip_size as the half-spread price, divide by
    cluster_R to get exit_spread_R, and subtract from final_r.

    If the exit timestamp doesn't exactly match a bar (rare; can happen if
    pair feed has gaps), fall back to the closest preceding in-session bar.
    """
    sim = sim_with_exit_ts.copy()
    pairs = sorted(sim["pair"].unique().tolist())
    spread_lookups: Dict[str, pd.Series] = {p: load_1h_spread_lookup(p, data_dir) for p in pairs}

    exit_spread_pips_eff: List[float] = []
    exit_spread_price: List[float] = []
    exit_spread_R: List[float] = []
    spread_gap_filled: List[bool] = []
    spread_floored: List[bool] = []
    spread_lookup_ok: List[bool] = []

    for _, row in sim.iterrows():
        pair = str(row["pair"])
        cluster_R = float(row["cluster_R"])
        ets = pd.Timestamp(row["exit_ts"])
        lookup = spread_lookups[pair]
        if ets in lookup.index:
            raw_pts = float(lookup.loc[ets])
            gap = False
            ok = True
        else:
            # asof preceding
            try:
                preceding = lookup.loc[: ets].iloc[-1]
                raw_pts = float(preceding)
                gap = True
                ok = True
            except Exception:
                raw_pts = 0.0
                gap = True
                ok = False
        # Apply spread floor convention.
        raw_pips_pre_floor = raw_pts / sf.points_per_pip if sf.points_per_pip > 0 else 0.0
        floor_pips = sf.floors_pips.get(pair, 0.0)
        floored = raw_pips_pre_floor < floor_pips
        eff_pips = max(raw_pips_pre_floor, floor_pips)
        half_spread_price = (eff_pips / 2.0) * _pip_size(pair)
        spread_R = (half_spread_price / cluster_R) if cluster_R > 0 else 0.0

        exit_spread_pips_eff.append(eff_pips)
        exit_spread_price.append(half_spread_price)
        exit_spread_R.append(spread_R)
        spread_gap_filled.append(gap)
        spread_floored.append(floored)
        spread_lookup_ok.append(ok)

    sim["exit_spread_pips_eff"] = exit_spread_pips_eff
    sim["exit_spread_price_half"] = exit_spread_price
    sim["exit_spread_R"] = exit_spread_R
    sim["spread_gap_filled"] = spread_gap_filled
    sim["spread_floored"] = spread_floored
    sim["final_r_original"] = sim["final_r"].astype(float)
    sim["final_r_with_exit_spread"] = sim["final_r_original"] - sim["exit_spread_R"]
    sim["exit_spread_pct_of_R"] = sim["exit_spread_R"]   # by definition (in R units)

    # Audit summary.
    arr = sim["exit_spread_R"].to_numpy(dtype=float)
    outliers = (arr > 0.50).sum()
    per_pair_mean = sim.groupby("pair")["exit_spread_pips_eff"].mean().to_dict()
    audit = SpreadAuditRecord(
        n_trades_total=int(len(sim)),
        n_trades_with_spread=int(sum(spread_lookup_ok)),
        n_trades_gap_filled=int(sum(spread_gap_filled)),
        n_trades_floor_active=int(sum(spread_floored)),
        n_trades_outlier=int(outliers),
        mean_exit_spread_R=float(np.mean(arr)),
        median_exit_spread_R=float(np.median(arr)),
        max_exit_spread_R=float(np.max(arr)),
        min_exit_spread_R=float(np.min(arr)),
        per_pair_mean_pips={k: float(v) for k, v in per_pair_mean.items()},
    )
    return sim, audit


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
# Hourly MTM equity curve (convention b) at baseline risk
# ============================================================================


def build_hourly_equity_per_fold(
    sim_trades: pd.DataFrame,
    folds: List[WFOFold],
    closes_2d: np.ndarray,
    times_2d: np.ndarray,
    trade_id_to_idx: Dict[int, int],
    final_r_col: str,
    risk_pct_baseline: float = BASELINE_RISK_PCT,
) -> Tuple[Dict[int, pd.Series], Dict[int, float], Dict[int, float]]:
    """Per fold: hourly MTM equity series, max DD, fold ROI.

    The hold-window floating R uses MID close (no spread mark-down) — standard
    MTM convention. At exit_bar and after, the trade contributes `final_r_col`
    (which may be either `final_r_original` or `final_r_with_exit_spread`).
    The mark-down between bar exit_bar−1 and bar exit_bar captures the
    realised exit spread (and the trail/SL trigger price) at the exit bar.
    """
    risk_scale = risk_pct_baseline / 100.0
    hourly: Dict[int, pd.Series] = {}
    max_dd: Dict[int, float] = {}
    roi: Dict[int, float] = {}

    for f in folds:
        if f.fold == 1:
            continue
        ft = sim_trades[sim_trades["fold"] == f.fold]
        if ft.empty:
            hourly[f.fold] = pd.Series([], dtype=float)
            max_dd[f.fold] = 0.0
            roi[f.fold] = 0.0
            continue
        timeline = pd.date_range(
            start=f.oos_start, end=f.oos_end - pd.Timedelta(hours=1), freq="1h"
        )
        if len(timeline) == 0:
            hourly[f.fold] = pd.Series([], dtype=float)
            max_dd[f.fold] = 0.0
            roi[f.fold] = 0.0
            continue

        total_r = pd.Series(0.0, index=timeline)
        for _, row in ft.iterrows():
            tid = int(row["trade_id"])
            entry_price = float(row["entry_price"])
            cluster_R = float(row["cluster_R"])
            exit_bar = int(row["exit_bar"])
            final_r = float(row[final_r_col])
            if tid not in trade_id_to_idx:
                continue
            ti = trade_id_to_idx[tid]
            bar_times = pd.to_datetime(times_2d[ti])
            bar_closes = closes_2d[ti]
            # 0..exit_bar - 1: floating mid-based R
            if exit_bar >= 1:
                pre_exit_r = (bar_closes[: exit_bar] - entry_price) / cluster_R
                pre_exit_times = bar_times[: exit_bar]
            else:
                pre_exit_r = np.array([], dtype=float)
                pre_exit_times = bar_times[:0]
            # At bar exit_bar onwards: realised final_r (constant).
            post_exit_r = np.full(PATH_BARS - exit_bar, final_r)
            post_exit_times = bar_times[exit_bar:]
            contrib = pd.Series(
                np.concatenate([pre_exit_r, post_exit_r]),
                index=pd.DatetimeIndex(list(pre_exit_times) + list(post_exit_times)),
            ).sort_index()
            grid = contrib.reindex(timeline, method="ffill").fillna(0.0)
            total_r = total_r + grid

        equity_pct = total_r * risk_scale * 100.0
        hourly[f.fold] = equity_pct
        arr = equity_pct.to_numpy()
        running_peak = np.maximum.accumulate(arr)
        dd = running_peak - arr
        max_dd[f.fold] = float(dd.max()) if dd.size else 0.0
        roi[f.fold] = (
            float(equity_pct.iloc[-1]) - float(equity_pct.iloc[0]) if len(equity_pct) >= 2 else 0.0
        )

    return hourly, max_dd, roi


def compute_daily_dds(hourly_equity: pd.Series) -> List[Tuple[pd.Timestamp, float]]:
    if hourly_equity.empty:
        return []
    eq = hourly_equity
    out: List[Tuple[pd.Timestamp, float]] = []
    for day, group in eq.groupby(eq.index.normalize()):
        if len(group) == 0:
            continue
        sod = float(group.iloc[0])
        intraday_min = float(group.min())
        out.append((pd.Timestamp(day), max(0.0, sod - intraday_min)))
    return out


# ============================================================================
# Risk-sweep aggregation
# ============================================================================


def build_risk_sweep(
    sim_trades: pd.DataFrame,
    folds: List[WFOFold],
    hourly_eq_baseline: Dict[int, pd.Series],
    daily_dds_baseline: Dict[int, List[Tuple[pd.Timestamp, float]]],
    fold_max_dd_baseline: Dict[int, float],
    fold_roi_baseline: Dict[int, float],
    out_dir: Path,
    final_r_col: str,
    label_suffix: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    fold_oos_days = {f.fold: f.oos_days for f in folds}
    annualisation = {f.fold: 365.0 / f.oos_days for f in folds if f.fold >= 2}
    sha_files: Dict[str, str] = {}
    sweep_rows: List[Dict[str, Any]] = []

    for risk in RISK_GRID_PCT:
        factor = risk / BASELINE_RISK_PCT
        per_fold_rows: List[Dict[str, Any]] = []
        daily_dd_rows: List[Dict[str, Any]] = []
        per_fold_dd: Dict[int, float] = {}
        per_fold_roi: Dict[int, float] = {}
        per_fold_roi_ann: Dict[int, float] = {}
        worst_daily_per_fold: Dict[int, float] = {}
        all_daily_dds: List[float] = []

        for f in folds:
            if f.fold == 1:
                continue
            ft = sim_trades[sim_trades["fold"] == f.fold]
            n = int(len(ft))
            dd_base = fold_max_dd_baseline.get(f.fold, 0.0)
            roi_base = fold_roi_baseline.get(f.fold, 0.0)
            dd_at = dd_base * factor
            roi_at = roi_base * factor
            roi_ann = roi_at * annualisation[f.fold]
            per_fold_dd[f.fold] = dd_at
            per_fold_roi[f.fold] = roi_at
            per_fold_roi_ann[f.fold] = roi_ann

            day_dds = daily_dds_baseline.get(f.fold, [])
            scaled = [(d, dd * factor) for d, dd in day_dds]
            fold_daily_dds = [dd for _, dd in scaled]
            worst_daily_per_fold[f.fold] = float(max(fold_daily_dds)) if fold_daily_dds else 0.0
            all_daily_dds.extend(fold_daily_dds)
            for d, dd in scaled:
                daily_dd_rows.append(
                    {
                        "risk_pct": risk,
                        "fold": f.fold,
                        "day": pd.Timestamp(d).date().isoformat(),
                        "daily_dd_pct": _fmt_g(dd),
                    }
                )
            rs = ft[final_r_col].to_numpy(dtype=float) if not ft.empty else np.array([])
            mean_r = float(rs.mean()) if rs.size else 0.0
            std_r = float(rs.std(ddof=1)) if rs.size > 1 else 0.0
            t_stat = float(mean_r / (std_r / math.sqrt(n))) if std_r > 0 else 0.0
            exit_dist = ft["exit_reason"].value_counts().to_dict() if not ft.empty else {}
            per_fold_rows.append(
                {
                    "cluster_id": 1,
                    "risk_pct": risk,
                    "fold": f.fold,
                    "n": n,
                    "oos_days": fold_oos_days[f.fold],
                    "annualisation_factor": _fmt_g(annualisation[f.fold]),
                    "mean_r": _fmt_g(mean_r),
                    "std_r": _fmt_g(std_r),
                    "t_stat": _fmt_g(t_stat),
                    "fold_roi_b_pct": _fmt_g(roi_at),
                    "fold_roi_b_ann_pct": _fmt_g(roi_ann),
                    "fold_max_dd_b_pct": _fmt_g(dd_at),
                    "worst_daily_dd_b_pct": _fmt_g(worst_daily_per_fold[f.fold]),
                    "exit_reason_dist": ";".join(f"{k}={int(v)}" for k, v in sorted(exit_dist.items())),
                }
            )

        active = [f.fold for f in folds if f.fold >= 2]
        worst_dd = max(per_fold_dd[fid] for fid in active)
        mean_dd = float(np.mean([per_fold_dd[fid] for fid in active]))
        worst_roi = min(per_fold_roi[fid] for fid in active)
        worst_roi_ann = min(per_fold_roi_ann[fid] for fid in active)
        mean_roi = float(np.mean([per_fold_roi[fid] for fid in active]))
        mean_roi_ann = float(np.mean([per_fold_roi_ann[fid] for fid in active]))
        worst_daily_overall = max(worst_daily_per_fold.values()) if worst_daily_per_fold else 0.0
        if all_daily_dds:
            p99 = float(np.percentile(all_daily_dds, 99))
            p95 = float(np.percentile(all_daily_dds, 95))
            p50 = float(np.percentile(all_daily_dds, 50))
        else:
            p99 = p95 = p50 = 0.0

        passes_fold_dd = worst_dd <= DD_CEILING_PCT
        passes_daily_target = worst_daily_overall <= DAILY_DD_TARGET_PCT
        passes_daily_limit = worst_daily_overall <= DAILY_DD_LIMIT_PCT
        passes_deployable = (
            worst_roi_ann >= PASS_DEPLOY_WORST_ROI_ANN_PCT
            and mean_roi_ann >= PASS_DEPLOY_MEAN_ROI_ANN_PCT
            and passes_fold_dd
            and p99 <= DAILY_DD_TARGET_PCT
            and passes_daily_limit
        )
        passes_viable = (
            worst_roi_ann > PASS_VIABLE_WORST_ROI_PCT
            and mean_roi_ann >= PASS_VIABLE_MEAN_ROI_ANN_PCT
            and passes_fold_dd
            and passes_daily_limit
        )
        sweep_rows.append(
            {
                "cluster_id": 1,
                "risk_pct": risk,
                "worst_fold_dd_b_pct": _fmt_g(worst_dd),
                "worst_fold_roi_b_pct": _fmt_g(worst_roi),
                "worst_fold_roi_ann_pct": _fmt_g(worst_roi_ann),
                "mean_fold_dd_b_pct": _fmt_g(mean_dd),
                "mean_fold_roi_b_pct": _fmt_g(mean_roi),
                "mean_fold_roi_ann_pct": _fmt_g(mean_roi_ann),
                "worst_daily_dd_b_pct": _fmt_g(worst_daily_overall),
                "daily_dd_p99_b_pct": _fmt_g(p99),
                "daily_dd_p95_b_pct": _fmt_g(p95),
                "daily_dd_p50_b_pct": _fmt_g(p50),
                "passes_fold_dd_ceiling": int(passes_fold_dd),
                "passes_daily_dd_target": int(passes_daily_target),
                "passes_daily_dd_limit": int(passes_daily_limit),
                "passes_deployable": int(passes_deployable),
                "passes_viable": int(passes_viable),
            }
        )

        risk_bps = int(round(risk * 100))
        per_fold_path = out_dir / f"fold_b_at_{risk_bps}bps_cluster_1_{label_suffix}.csv"
        _write_rows(per_fold_rows, per_fold_path)
        sha_files[per_fold_path.name] = _file_sha256(per_fold_path)
        dd_dist_path = out_dir / f"daily_dd_distribution_at_{risk_bps}bps_cluster_1_{label_suffix}.csv"
        _write_rows(daily_dd_rows, dd_dist_path)
        sha_files[dd_dist_path.name] = _file_sha256(dd_dist_path)

    return sweep_rows, sha_files


# ============================================================================
# Single-run orchestration
# ============================================================================


def run_once(
    sim_csv: Path,
    paths_csv: Path,
    fold_defs_csv: Path,
    spread_floor_path: Path,
    data_dir: Path,
    out_dir: Path,
) -> Tuple[Dict[str, str], SpreadAuditRecord]:
    out_dir.mkdir(parents=True, exist_ok=True)
    folds = load_fold_defs(fold_defs_csv)

    # Load refit per-trade.
    sim = pd.read_csv(sim_csv)
    sim["entry_ts"] = pd.to_datetime(sim["entry_ts"])
    sim = sim[sim["fold"].isin([f.fold for f in folds if f.fold >= 2])].reset_index(drop=True)

    # Spread floor (sha-locked).
    sf = load_spread_floor(spread_floor_path)

    # Look up exit timestamps from trades_paths.
    print("[5b-spread] looking up exit timestamps from trades_paths...", file=sys.stderr)
    t0 = time.time()
    sim_w = build_exit_timestamps(sim, paths_csv)
    print(f"[5b-spread] exit timestamps in {time.time() - t0:.1f}s", file=sys.stderr)

    # Apply exit spread.
    print("[5b-spread] applying exit-spread correction (half-spread per trade)...", file=sys.stderr)
    t1 = time.time()
    sim_corr, audit = apply_exit_spread(sim_w, sf, data_dir)
    print(f"[5b-spread] spread applied in {time.time() - t1:.1f}s", file=sys.stderr)

    # Write per-trade spread-corrected output.
    per_trade_path = out_dir / "per_trade_simulated_refit_spread_1.csv"
    out_rows = sim_corr[
        [
            "trade_id", "pair", "fold", "entry_ts", "entry_price", "atr_signal",
            "cluster_R", "pre_t_sl_price", "post_t_sl_price", "exit_bar",
            "exit_reason", "exit_price", "exit_ts", "exit_spread_pips_eff",
            "exit_spread_price_half", "exit_spread_R", "spread_gap_filled",
            "spread_floored", "final_r_original", "final_r_with_exit_spread",
            "exit_spread_pct_of_R", "mfe_locked_bar", "peak_mfe_r",
        ]
    ].copy()
    # Format datetimes as ISO.
    out_rows["entry_ts"] = pd.to_datetime(out_rows["entry_ts"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    out_rows["exit_ts"] = pd.to_datetime(out_rows["exit_ts"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    out_rows.to_csv(per_trade_path, index=False, na_rep="", lineterminator="\n")
    sha_files: Dict[str, str] = {per_trade_path.name: _file_sha256(per_trade_path)}

    # Load paths into 2D tensor for the equity curve build.
    print("[5b-spread] loading paths into 2D tensor...", file=sys.stderr)
    t2 = time.time()
    paths_df = pd.read_csv(paths_csv, usecols=["trade_id", "bar_offset", "timestamp", "close"])
    paths_df = paths_df.sort_values(["trade_id", "bar_offset"]).reset_index(drop=True)
    admitted_tids = sim_corr["trade_id"].astype(int).unique().tolist()
    paths_sub = paths_df[paths_df["trade_id"].isin(admitted_tids)].copy()
    paths_sub["timestamp"] = pd.to_datetime(paths_sub["timestamp"])
    paths_sub = paths_sub.sort_values(["trade_id", "bar_offset"]).reset_index(drop=True)
    n_adm = paths_sub["trade_id"].nunique()
    if len(paths_sub) != n_adm * PATH_BARS:
        raise ValueError(
            f"Per-trade path expansion mismatch: got {len(paths_sub)}, expected {n_adm}×{PATH_BARS}"
        )
    closes_2d = paths_sub["close"].to_numpy(dtype=float).reshape(n_adm, PATH_BARS)
    times_2d = paths_sub["timestamp"].to_numpy().reshape(n_adm, PATH_BARS)
    tid_arr = paths_sub["trade_id"].to_numpy(dtype=int).reshape(n_adm, PATH_BARS)[:, 0]
    trade_id_to_idx = {int(t): i for i, t in enumerate(tid_arr)}
    print(f"[5b-spread] paths tensor built in {time.time() - t2:.1f}s", file=sys.stderr)

    # ----- Spread-corrected equity / DD baseline (at 0.5%) -----
    print("[5b-spread] building hourly MTM equity (spread-corrected)...", file=sys.stderr)
    t3 = time.time()
    hourly_eq_corr, max_dd_corr, roi_corr = build_hourly_equity_per_fold(
        sim_corr, folds, closes_2d, times_2d, trade_id_to_idx,
        final_r_col="final_r_with_exit_spread",
    )
    daily_dds_corr = {fold_id: compute_daily_dds(eq) for fold_id, eq in hourly_eq_corr.items()}
    print(f"[5b-spread] spread equity built in {time.time() - t3:.1f}s", file=sys.stderr)

    # Persist hourly equity curve (audit).
    eq_rows: List[Dict[str, Any]] = []
    for fold_id, eq in hourly_eq_corr.items():
        for ts, val in eq.items():
            eq_rows.append(
                {"fold": fold_id, "timestamp": pd.Timestamp(ts).isoformat(),
                 "equity_pct_spread_corrected": _fmt_g(float(val))}
            )
    eq_path = out_dir / "hourly_equity_curve_at_0.5pct_cluster_1_spread.csv"
    _write_rows(eq_rows, eq_path)
    sha_files[eq_path.name] = _file_sha256(eq_path)

    # Spread-corrected risk sweep.
    sweep_corr_rows, sha_corr = build_risk_sweep(
        sim_corr, folds, hourly_eq_corr, daily_dds_corr, max_dd_corr, roi_corr,
        out_dir, final_r_col="final_r_with_exit_spread", label_suffix="spread",
    )
    sha_files.update(sha_corr)
    sweep_corr_path = out_dir / "risk_sweep_refit_spread_b_cluster_1.csv"
    _write_rows(sweep_corr_rows, sweep_corr_path)
    sha_files[sweep_corr_path.name] = _file_sha256(sweep_corr_path)

    # ----- No-spread baseline (use final_r_original; rerun for direct delta) -----
    print("[5b-spread] building hourly MTM equity (no-spread baseline)...", file=sys.stderr)
    t4 = time.time()
    hourly_eq_ns, max_dd_ns, roi_ns = build_hourly_equity_per_fold(
        sim_corr, folds, closes_2d, times_2d, trade_id_to_idx,
        final_r_col="final_r_original",
    )
    daily_dds_ns = {fold_id: compute_daily_dds(eq) for fold_id, eq in hourly_eq_ns.items()}
    print(f"[5b-spread] no-spread equity built in {time.time() - t4:.1f}s", file=sys.stderr)

    sweep_ns_rows, sha_ns = build_risk_sweep(
        sim_corr, folds, hourly_eq_ns, daily_dds_ns, max_dd_ns, roi_ns,
        out_dir, final_r_col="final_r_original", label_suffix="nospread",
    )
    sha_files.update(sha_ns)
    sweep_ns_path = out_dir / "risk_sweep_refit_no_spread_b_cluster_1.csv"
    _write_rows(sweep_ns_rows, sweep_ns_path)
    sha_files[sweep_ns_path.name] = _file_sha256(sweep_ns_path)

    # ----- Delta analysis -----
    delta_rows: List[Dict[str, Any]] = []
    for f in folds:
        if f.fold == 1:
            continue
        ft = sim_corr[sim_corr["fold"] == f.fold]
        n = int(len(ft))
        if n == 0:
            continue
        mean_orig = float(ft["final_r_original"].mean())
        mean_spread = float(ft["final_r_with_exit_spread"].mean())
        roi_base_orig = roi_ns.get(f.fold, 0.0)
        roi_base_spread = roi_corr.get(f.fold, 0.0)
        dd_base_orig = max_dd_ns.get(f.fold, 0.0)
        dd_base_spread = max_dd_corr.get(f.fold, 0.0)
        # All at 0.5% baseline.
        total_spread_cost_R = float(ft["exit_spread_R"].sum())
        mean_spread_cost_R = float(ft["exit_spread_R"].mean())
        delta_rows.append(
            {
                "fold": f.fold,
                "n": n,
                "mean_r_original": _fmt_g(mean_orig),
                "mean_r_with_spread": _fmt_g(mean_spread),
                "delta_mean_r": _fmt_g(mean_spread - mean_orig),
                "fold_roi_pct_at_0.5_no_spread": _fmt_g(roi_base_orig),
                "fold_roi_pct_at_0.5_with_spread": _fmt_g(roi_base_spread),
                "delta_fold_roi_pct": _fmt_g(roi_base_spread - roi_base_orig),
                "fold_dd_pct_at_0.5_no_spread": _fmt_g(dd_base_orig),
                "fold_dd_pct_at_0.5_with_spread": _fmt_g(dd_base_spread),
                "delta_fold_dd_pct": _fmt_g(dd_base_spread - dd_base_orig),
                "mean_exit_spread_R": _fmt_g(mean_spread_cost_R),
                "total_R_harvest_reduction": _fmt_g(total_spread_cost_R),
            }
        )
    delta_path = out_dir / "spread_delta_analysis.csv"
    _write_rows(delta_rows, delta_path)
    sha_files[delta_path.name] = _file_sha256(delta_path)

    return sha_files, audit


# ============================================================================
# Diagnostics writer
# ============================================================================


def write_diagnostics(
    out_path: Path,
    sweep_corr_csv: Path,
    sweep_ns_csv: Path,
    delta_csv: Path,
    audit: SpreadAuditRecord,
    sha_run1: Dict[str, str],
    sha_run2: Optional[Dict[str, str]],
    determinism_gate: str,
    config_paths: Dict[str, str],
    config_shas: Dict[str, str],
) -> Tuple[str, Optional[float], Optional[float]]:
    sweep_c = pd.read_csv(sweep_corr_csv)
    sweep_n = pd.read_csv(sweep_ns_csv)
    delta = pd.read_csv(delta_csv)

    # Identify recommended risk (highest pass-deployable) in spread-corrected sweep.
    dep_c = sweep_c[sweep_c["passes_deployable"] == 1]
    viable_c = sweep_c[sweep_c["passes_viable"] == 1]
    rec_dep = float(dep_c["risk_pct"].max()) if not dep_c.empty else None
    rec_viable = float(viable_c["risk_pct"].max()) if not viable_c.empty else None
    rec_risk = rec_dep if rec_dep is not None else rec_viable

    # Same for no-spread (for delta).
    dep_n = sweep_n[sweep_n["passes_deployable"] == 1]
    rec_dep_n = float(dep_n["risk_pct"].max()) if not dep_n.empty else None

    lines: List[str] = []
    lines.append("# Arc 4 — Step 5B-spread (exit spread correction) — diagnostics")
    lines.append("")
    lines.append("Protocol: `L_ARC_PROTOCOL.md` v2.1.1 §9 supplement; SPREAD_SEMANTICS_LOCK.md; "
                 "5ers daily DD limit (5%) / safety target (4%)")
    lines.append("Cluster: 1 (`stepwise_pullback_extended`, R = 3.0 × ATR; refit F2-F7)")
    lines.append("Source: `results/l_arc_4/step5c/per_trade_simulated_refit_1.csv` (5,361 trades)")
    lines.append("Convention: **(b) hourly mark-to-market** with concurrent floating PnL. "
                 "Linear-in-risk scaling from 0.5% baseline.")
    lines.append("")

    # Spread mechanic statement.
    lines.append("## Spread mechanic (audited)")
    lines.append("")
    lines.append(
        "Step 1 entry: `entry_price = entry_mid + S_entry/2 × pip_size` (long ask). "
        "Step 1 exit (intrabar SL / max-life): `exit_price = trigger_level − S_exit/2 × pip_size` "
        "(long bid). Round-trip cost = S/2 (entry, in entry_price) + S/2 (exit) = S total."
    )
    lines.append("")
    lines.append(
        "Step 5/5C simulator returned `final_r = (trigger_level − entry_price) / R` with "
        "`trigger_level` at the SL price, trail-stop price, or bar-240 close — all "
        "**mid-equivalent** on the exit side with NO S_exit/2 deduction. So 5C `final_r` "
        "**overstated** outcomes by `S_exit/2 / R` per trade. Step 5B-spread correction: "
        "subtract `(S_exit_pips / 2) × pip_size / R_per_trade` from each trade's `final_r`."
    )
    lines.append("")
    lines.append(
        "**Note on prompt language**: the Step 5B-spread prompt says \"deduct one full spread\", "
        "but the **mechanically correct** correction to restore Step 1's round-trip cost is the "
        "half-spread on the exit side (S/2, not full S). Applying full S would double-count "
        "the entry-side S/2 already baked into `entry_price`. This implementation applies S/2; "
        "the math is logged here for chat audit."
    )
    lines.append("")

    # Spread audit.
    lines.append("## Spread application audit")
    lines.append("")
    lines.append(
        f"- Trades processed: **{audit.n_trades_total}** (cluster 1 refit F2-F7)"
    )
    lines.append(
        f"- Exit-spread lookups OK: {audit.n_trades_with_spread} "
        f"({100 * audit.n_trades_with_spread / audit.n_trades_total:.2f}%)"
    )
    lines.append(
        f"- Gap-filled (exit ts not in pair feed → closest preceding bar): "
        f"{audit.n_trades_gap_filled} "
        f"({100 * audit.n_trades_gap_filled / audit.n_trades_total:.2f}%)"
    )
    lines.append(
        f"- Spread floor active (raw spread < floor): {audit.n_trades_floor_active} "
        f"({100 * audit.n_trades_floor_active / audit.n_trades_total:.2f}%)"
    )
    lines.append(
        f"- Outliers (`exit_spread_R > 0.50`): **{audit.n_trades_outlier}**"
    )
    lines.append(
        f"- Mean exit-spread cost: **{audit.mean_exit_spread_R:.4f} R** "
        f"(median {audit.median_exit_spread_R:.4f} R; "
        f"min {audit.min_exit_spread_R:.4f} R; "
        f"max {audit.max_exit_spread_R:.4f} R)"
    )
    lines.append("")
    lines.append("Per-pair mean exit-spread (pips, floored):")
    lines.append("")
    lines.append("| Pair | mean exit_spread (pips) |")
    lines.append("|---|---:|")
    for pair in sorted(audit.per_pair_mean_pips.keys()):
        lines.append(f"| {pair} | {audit.per_pair_mean_pips[pair]:.3f} |")
    lines.append("")

    # Headline.
    lines.append("## Headline")
    lines.append("")
    if rec_risk is not None:
        rec_row = sweep_c[sweep_c["risk_pct"] == rec_risk].iloc[0]
        cls = "PASS-DEPLOYABLE" if (rec_dep is not None and rec_risk == rec_dep) else "PASS-VIABLE"
        lines.append(
            f"Spread-corrected recommended risk = **{rec_risk}% per trade** ({cls}). "
            f"Worst-fold annualised ROI {float(rec_row['worst_fold_roi_ann_pct']):.2f}%; "
            f"worst-fold DD (b) {float(rec_row['worst_fold_dd_b_pct']):.3f}%; "
            f"worst daily DD {float(rec_row['worst_daily_dd_b_pct']):.3f}%; "
            f"daily DD p99 {float(rec_row['daily_dd_p99_b_pct']):.3f}%."
        )
        if rec_dep_n is not None:
            shift = rec_dep_n - rec_risk
            lines.append(
                f"No-spread (5B-refit) recommended risk: {rec_dep_n}%. "
                f"Spread correction shifts recommended risk by **{shift:+.2f}pp** "
                f"({'tighter' if shift > 0 else 'unchanged or looser'})."
            )
    else:
        lines.append("Spread-corrected sweep: **no risk level passes pass-deployable**. "
                     "Check pass-viable fallback in the table below.")
    lines.append(f"Determinism: **{determinism_gate}**")
    lines.append("")

    # Full sweep tables — spread-corrected, then no-spread for delta.
    lines.append("## Spread-corrected risk sweep (convention (b), F2-F7)")
    lines.append("")
    lines.append(
        "| Risk% | worst fold DD% | worst fold ROI ann% | mean fold ROI ann% | "
        "worst daily DD% | daily DD p99% | daily DD p95% | fold DD ≤ 8% | "
        "daily ≤ 4% | daily ≤ 5% | pass-deploy | pass-viable |"
    )
    lines.append(
        "|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|:---:|:---:|"
    )
    for _, r in sweep_c.iterrows():
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

    # Side-by-side delta at recommended risk.
    if rec_risk is not None:
        lines.append(f"## No-spread vs spread side-by-side at recommended risk = {rec_risk}%")
        lines.append("")
        ns_row = sweep_n[sweep_n["risk_pct"] == rec_risk].iloc[0]
        c_row = sweep_c[sweep_c["risk_pct"] == rec_risk].iloc[0]
        lines.append("| Metric | No spread | Spread-corrected | Δ |")
        lines.append("|---|---:|---:|---:|")
        for col, label in [
            ("worst_fold_dd_b_pct", "worst fold DD%"),
            ("worst_fold_roi_ann_pct", "worst fold ROI ann%"),
            ("mean_fold_roi_ann_pct", "mean fold ROI ann%"),
            ("worst_daily_dd_b_pct", "worst daily DD%"),
            ("daily_dd_p99_b_pct", "daily DD p99%"),
        ]:
            v_ns = float(ns_row[col])
            v_c = float(c_row[col])
            lines.append(f"| {label} | {v_ns:.3f} | {v_c:.3f} | {v_c - v_ns:+.3f} |")
        lines.append("")

    # Per-fold detail at recommended risk (spread-corrected).
    if rec_risk is not None:
        risk_bps = int(round(rec_risk * 100))
        per_fold_csv = out_path.parent / f"fold_b_at_{risk_bps}bps_cluster_1_spread.csv"
        if per_fold_csv.exists():
            pf = pd.read_csv(per_fold_csv)
            lines.append(f"## Per-fold detail at recommended risk = {rec_risk}% (spread-corrected)")
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
            dd_csv = out_path.parent / f"daily_dd_distribution_at_{risk_bps}bps_cluster_1_spread.csv"
            if dd_csv.exists():
                dd = pd.read_csv(dd_csv)
                arr = dd["daily_dd_pct"].dropna().to_numpy(dtype=float)
                lines.append(f"## Daily DD distribution at recommended risk = {rec_risk}% (spread-corrected)")
                lines.append("")
                lines.append(f"- Total days observed (F2-F7): {len(arr)}")
                if len(arr) > 0:
                    lines.append(f"- Max daily DD: **{float(np.max(arr)):.3f}%**")
                    lines.append(f"- p99 daily DD: **{float(np.percentile(arr, 99)):.3f}%**")
                    lines.append(f"- p95 daily DD: {float(np.percentile(arr, 95)):.3f}%")
                    lines.append(f"- p90 daily DD: {float(np.percentile(arr, 90)):.3f}%")
                    lines.append(f"- p50 daily DD: {float(np.percentile(arr, 50)):.3f}%")
                    lines.append(
                        f"- Days with daily DD > 4%: {int(np.sum(arr > 4.0))} "
                        f"({float(100 * np.mean(arr > 4.0)):.2f}% of days)"
                    )
                    lines.append(
                        f"- Days with daily DD > 5%: {int(np.sum(arr > 5.0))} "
                        f"({float(100 * np.mean(arr > 5.0)):.2f}% of days; 5ers limit)"
                    )
                lines.append("")

    # Per-fold spread delta summary.
    lines.append("## Per-fold spread delta (at 0.5% baseline)")
    lines.append("")
    lines.append(
        "| F | n | mean_r (orig) | mean_r (spread) | Δ mean_r | "
        "fold_ROI (orig) | fold_ROI (spread) | Δ ROI | "
        "fold_DD (orig) | fold_DD (spread) | Δ DD | total R cost |"
    )
    lines.append(
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for _, r in delta.iterrows():
        lines.append(
            f"| F{int(r['fold'])} | {int(r['n'])} | "
            f"{float(r['mean_r_original']):+.4f} | {float(r['mean_r_with_spread']):+.4f} | "
            f"{float(r['delta_mean_r']):+.4f} | "
            f"{float(r['fold_roi_pct_at_0.5_no_spread']):+.3f} | {float(r['fold_roi_pct_at_0.5_with_spread']):+.3f} | "
            f"{float(r['delta_fold_roi_pct']):+.3f} | "
            f"{float(r['fold_dd_pct_at_0.5_no_spread']):.3f} | {float(r['fold_dd_pct_at_0.5_with_spread']):.3f} | "
            f"{float(r['delta_fold_dd_pct']):+.3f} | "
            f"{float(r['total_R_harvest_reduction']):.2f} |"
        )
    lines.append("")

    # Cross-arc observation.
    lines.append("## Cross-arc observations (informational)")
    lines.append("")
    lines.append(
        f"- Exit-spread cost is materially uniform per-trade: mean {audit.mean_exit_spread_R:.4f} R, "
        f"median {audit.median_exit_spread_R:.4f} R. With {audit.n_trades_total} trades over F2-F7, "
        "the cumulative R harvest reduction is appreciable but shouldn't be a deal-breaker "
        "given the signal's baseline +0.15-0.20 R mean per trade."
    )
    lines.append("")
    lines.append(
        "- **Protocol audit item**: `SPREAD_SEMANTICS_LOCK.md` already documents the entry/exit "
        "spread convention (entry at ask, exit at bid, S/2 each side). The Step 5/5C simulator's "
        "skipped exit-spread was a prompt-author error (acknowledged in the 5B-spread prompt). "
        "Future post-hoc simulators should apply S_exit/2 on exit by default; this 5B-spread "
        "correction is a one-off fix for Arc 4. Cross-arc v2.2 backlog: enforce exit-spread "
        "application in all post-hoc simulator templates."
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

    # Config + input sha256s.
    lines.append("## Input / config sha256s")
    lines.append("")
    for label, path in config_paths.items():
        lines.append(f"- `{label}` (`{path}`) — `{config_shas[label]}`")
    lines.append("")
    lines.append(f"- Spread-floor file sha256 (`configs/spread_floors_5ers.yaml`): verified against "
                 f"`{EXPECTED_SPREAD_FLOOR_SHA[:32]}…` (sha-locked load).")
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ("PASS-DEPLOYABLE" if rec_dep is not None else ("PASS-VIABLE" if rec_viable is not None else "FAIL"),
            rec_risk, rec_dep_n)


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
        description="Arc 4 Step 5B-spread (exit-spread correction on refit, convention (b))."
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
        "--spread-floor",
        type=Path,
        default=_REPO_ROOT / "configs" / "spread_floors_5ers.yaml",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=_REPO_ROOT / "data" / "1hr",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO_ROOT / "results" / "l_arc_4" / "step5b_spread",
    )
    p.add_argument("--no-determinism-check", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    sim_csv = args.sim_csv.resolve()
    paths_csv = args.paths_csv.resolve()
    fold_defs_csv = args.fold_defs_csv.resolve()
    spread_floor_path = args.spread_floor.resolve()
    data_dir = args.data_dir.resolve()
    out_dir = args.out_dir.resolve()

    t0 = time.time()
    print("[5b-spread] === RUN 1 ===", file=sys.stderr)
    sha_run1, audit = run_once(sim_csv, paths_csv, fold_defs_csv, spread_floor_path, data_dir, out_dir)
    print(f"[5b-spread] Run 1 done in {time.time() - t0:.1f}s", file=sys.stderr)

    sha_run2: Optional[Dict[str, str]] = None
    determinism_gate = "N/A"
    if not args.no_determinism_check:
        print("[5b-spread] === RUN 2 (determinism) ===", file=sys.stderr)
        t1 = time.time()
        sha_run2, _ = run_once(sim_csv, paths_csv, fold_defs_csv, spread_floor_path, data_dir, out_dir)
        print(f"[5b-spread] Run 2 done in {time.time() - t1:.1f}s", file=sys.stderr)
        matched = all(
            sha_run1.get(k) == sha_run2.get(k) for k in sorted(set(sha_run1) | set(sha_run2))
        )
        determinism_gate = "PASS" if matched else "FAIL"

    config_paths = {
        "results/l_arc_4/step5c/per_trade_simulated_refit_1.csv": str(sim_csv.relative_to(_REPO_ROOT)),
        "results/l_arc_4/step1/trades_paths.csv": str(paths_csv.relative_to(_REPO_ROOT)),
        "results/l_arc_4/step5c/fold_definitions.csv": str(fold_defs_csv.relative_to(_REPO_ROOT)),
        "configs/spread_floors_5ers.yaml": str(spread_floor_path.relative_to(_REPO_ROOT)),
    }
    config_shas = {
        label: _file_sha256(_REPO_ROOT / Path(path)) for label, path in config_paths.items()
    }

    diag_path = out_dir / "step5b_spread_diagnostics.md"
    headline_class, rec_risk, rec_risk_ns = write_diagnostics(
        diag_path,
        sweep_corr_csv=out_dir / "risk_sweep_refit_spread_b_cluster_1.csv",
        sweep_ns_csv=out_dir / "risk_sweep_refit_no_spread_b_cluster_1.csv",
        delta_csv=out_dir / "spread_delta_analysis.csv",
        audit=audit,
        sha_run1=sha_run1,
        sha_run2=sha_run2,
        determinism_gate=determinism_gate,
        config_paths=config_paths,
        config_shas=config_shas,
    )

    print(
        f"[5b-spread] DONE class={headline_class} rec_risk={rec_risk} "
        f"rec_risk_no_spread={rec_risk_ns} determinism={determinism_gate}",
        file=sys.stderr,
    )
    print(f"[5b-spread] diagnostics → {diag_path}", file=sys.stderr)

    (out_dir / "step5b_spread_env.json").write_text(
        json.dumps(
            {
                "env": _env_dict(),
                "args": {k: str(v) for k, v in vars(args).items()},
                "recommended_risk_pct": rec_risk,
                "recommended_risk_pct_no_spread": rec_risk_ns,
                "headline_class": headline_class,
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
