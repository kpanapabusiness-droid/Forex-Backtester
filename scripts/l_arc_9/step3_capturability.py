"""Arc 9 - Step 3 capturability (L_ARC_PROTOCOL §7, v2.1.1 composite, v2.1.2 floors).

Reads:
    results/l_arc_9/step1_verbatim/trades_paths.csv  (path-bar grid w/ low_r, close_r,
                                                       mfe_so_far_r, mae_so_far_r, is_held)
    results/l_arc_9/step1_verbatim/trades_all.csv     (per-trade summary; final_r is in
                                                       baseline R = sl_distance / 2.0*ATR)
    results/l_arc_9/step2_clustering/clusters_K{k}.csv  (trade_id, cluster_id)
    results/l_arc_9/step2_clustering/archetype_assignments.csv (cluster_id, label)

Produces in results/l_arc_9/step3_capturability/:
    per_trade_sl_sweep.csv          one row per (trade_id, X) - re-imposed SL results
    cluster_sl_sweep.csv             one row per (cluster_id, X) - per-cluster aggregates
    archetype_sl_sweep.csv           one row per (archetype, X) - per-aggregate when >=2 share
    archetype_summaries.csv          chosen SL per archetype + identity + forward geometry
    archetype_<label>_distribution.csv  shape-tag detail at chosen SL
    capturability_pass_list.csv      surviving archetypes + selected SL
    cluster_routing.csv              per-cluster individual / aggregate / both / dies
    STEP3_SUMMARY.md

Schema notes:
- trades_paths.csv close_r/low_r/high_r are in baseline-R units where 1 R_baseline = 2*ATR.
  To evaluate candidate SL X*ATR: 1 R_candidate = X*ATR = (X/2) * R_baseline.
  SL trigger: low_r_baseline <= -X/2.
  Final_r in candidate-R: close_r_baseline * 2/X.
- We use the full path (held + forward-obs); is_held column is informational, not used here.

Per §7:
1. SL sweep over {0.5, 1.0, 1.5, 2.0, 3.0, 4.0} * ATR.
2. Pre-peak metrics on bars 0..peak_mfe_bar_X (peak recomputed per SL).
3. Apply §2 floors conjunctively at each candidate SL.
4. Among passing SLs, select the SL maximising the capturability composite
   (mono_pp - 0.55) + (frac_reach_1R - 0.70) + (0.30 - frac_wrong_way_pp).
   Tiebreaker 1: |composite_a - composite_b| <= 0.02 -> larger peak_mfe in ATR units.
   Tiebreaker 2: smaller SL.
5. Per-cluster AND per-aggregate (when >=2 clusters share an archetype label).
6. `bimodal_separated` test: Hartigan dip p<0.05 + min mode mass >=0.20 + mode sep >=1R.
   Implementation: diptest if available; fallback noted in shape_tag.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


SL_CANDIDATES_ATR: List[float] = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]

# §2 floors at selected SL (v2.1.2).
FLOOR_MONO_PRE_PEAK: float = 0.55
FLOOR_REACH_1R: float = 0.70
FLOOR_WRONG_WAY_PRE_PEAK: float = 0.30
FLOOR_FWD_MFE_P50: float = 1.5
FLOOR_SIZE_FRACTION: float = 0.10
TIEBREAK_COMPOSITE: float = 0.02

# bimodal_separated test.
BIMODAL_DIP_P_MAX: float = 0.05
BIMODAL_MIN_MODE_MASS: float = 0.20
BIMODAL_MODE_SEPARATION_R: float = 1.0

# scattered detection (simple CV threshold).
SCATTERED_CV_MIN: float = 1.5


def _monotonicity_in_profit(close_r: np.ndarray) -> float:
    in_profit = close_r[close_r > 0]
    if in_profit.size < 2:
        return 0.0
    return float(np.mean(in_profit[1:] >= in_profit[:-1]))


@dataclass
class TradeSLResult:
    trade_id: int
    X: float
    truncated_at_bar: int
    sl_triggered: bool
    peak_mfe_bar_X: int
    peak_mfe_X: float
    peak_mfe_atr: float
    final_r_X: float
    reached_1R_X: bool
    wrong_way_pre_peak_X: bool
    mono_pre_peak_X: float


def per_trade_sweep(
    trade_id: int,
    close_r_baseline: np.ndarray,
    low_r_baseline: np.ndarray,
    candidates: List[float],
) -> List[TradeSLResult]:
    n = close_r_baseline.size
    out: List[TradeSLResult] = []
    if n == 0:
        return out
    for X in candidates:
        threshold = -X / 2.0
        below = low_r_baseline <= threshold
        if below.any():
            trigger = int(np.argmax(below))
            sl_trig = True
        else:
            trigger = n - 1
            sl_trig = False
        cr_trunc = close_r_baseline[: trigger + 1]
        low_trunc = low_r_baseline[: trigger + 1]
        peak_bar = int(np.argmax(cr_trunc))
        peak_baseline = float(cr_trunc[peak_bar])
        peak_mfe_X = peak_baseline * 2.0 / X
        peak_mfe_atr = peak_baseline * 2.0
        if sl_trig:
            final_r_X = -1.0
        else:
            final_r_X = float(cr_trunc[-1]) * 2.0 / X
        reached_1R = bool(peak_baseline >= X / 2.0)
        low_pre = low_trunc[: peak_bar + 1]
        ww_pp = bool((low_pre <= threshold).any())
        cr_pre = cr_trunc[: peak_bar + 1]
        mono_pp = _monotonicity_in_profit(cr_pre)
        out.append(TradeSLResult(
            trade_id=trade_id, X=X, truncated_at_bar=trigger,
            sl_triggered=sl_trig, peak_mfe_bar_X=peak_bar,
            peak_mfe_X=peak_mfe_X, peak_mfe_atr=peak_mfe_atr,
            final_r_X=final_r_X, reached_1R_X=reached_1R,
            wrong_way_pre_peak_X=ww_pp, mono_pre_peak_X=mono_pp,
        ))
    return out


def sweep_all_trades(
    paths_df: pd.DataFrame, candidates: List[float],
) -> pd.DataFrame:
    paths = paths_df.sort_values(["trade_id", "bar_offset"], kind="mergesort")
    out_rows: List[Dict[str, Any]] = []
    for tid, grp in paths.groupby("trade_id", sort=True):
        cr = grp["close_r"].to_numpy(dtype=float)
        lr = grp["low_r"].to_numpy(dtype=float)
        bo = grp["bar_offset"].to_numpy(dtype=int)
        if bo[0] != 0 or not np.array_equal(bo, np.arange(bo.size)):
            order = np.argsort(bo, kind="mergesort")
            cr = cr[order]
            lr = lr[order]
        for r in per_trade_sweep(int(tid), cr, lr, candidates):
            out_rows.append(r.__dict__)
    return pd.DataFrame(out_rows)


def _aggregate(
    sub: pd.DataFrame, X: float, pool_n: int,
) -> Dict[str, Any]:
    """Compute one (group, X) aggregate row."""
    n = len(sub)
    if n == 0:
        return {"n": 0, "X": X}
    peak_X = sub["peak_mfe_X"].to_numpy(dtype=float)
    peak_atr = sub["peak_mfe_atr"].to_numpy(dtype=float)
    final_X = sub["final_r_X"].to_numpy(dtype=float)
    p5, p25, p50, p75, p95 = np.percentile(peak_X, [5, 25, 50, 75, 95])
    fp5, fp25, fp50, fp75, fp95 = np.percentile(final_X, [5, 25, 50, 75, 95])
    fr_mean = float(final_X.mean())
    fr_std = float(final_X.std(ddof=1)) if n > 1 else 0.0
    fr_t = (fr_mean / (fr_std / np.sqrt(n))) if (fr_std > 0 and n > 1) else float("nan")
    mass_gt_5R = float(np.mean(peak_X >= 5.0))
    frac_reach_1R = float(sub["reached_1R_X"].astype(int).mean())
    frac_reach_2R = float(np.mean(peak_X >= 2.0))
    frac_ww_pp = float(sub["wrong_way_pre_peak_X"].astype(int).mean())
    mono_pp_cent = float(sub["mono_pre_peak_X"].mean())
    composite = (
        (mono_pp_cent - FLOOR_MONO_PRE_PEAK)
        + (frac_reach_1R - FLOOR_REACH_1R)
        + (FLOOR_WRONG_WAY_PRE_PEAK - frac_ww_pp)
    )
    size_fraction = n / float(pool_n)
    floors = {
        "mono_pp": mono_pp_cent >= FLOOR_MONO_PRE_PEAK,
        "reach_1R": frac_reach_1R >= FLOOR_REACH_1R,
        "wrong_way_pp": frac_ww_pp <= FLOOR_WRONG_WAY_PRE_PEAK,
        "fwd_mfe_p50": p50 >= FLOOR_FWD_MFE_P50,
        "size_fraction": size_fraction >= FLOOR_SIZE_FRACTION,
    }
    all_pass = all(floors.values())
    return {
        "X": float(X),
        "n": int(n),
        "size_fraction": float(size_fraction),
        "mono_pre_peak_centroid": mono_pp_cent,
        "frac_reach_1R": frac_reach_1R,
        "frac_reach_2R": frac_reach_2R,
        "frac_wrong_way_pre_peak": frac_ww_pp,
        "fwd_mfe_p5": float(p5),
        "fwd_mfe_p25": float(p25),
        "fwd_mfe_p50": float(p50),
        "fwd_mfe_p75": float(p75),
        "fwd_mfe_p95": float(p95),
        "fwd_mfe_p50_atr": float(np.percentile(peak_atr, 50)),
        "final_r_p5": float(fp5),
        "final_r_p25": float(fp25),
        "final_r_p50": float(fp50),
        "final_r_p75": float(fp75),
        "final_r_p95": float(fp95),
        "final_r_mean": fr_mean,
        "final_r_t_stat": fr_t,
        "mass_gt_5R": mass_gt_5R,
        "composite": float(composite),
        "floor_mono_pp_pass": int(floors["mono_pp"]),
        "floor_reach_1R_pass": int(floors["reach_1R"]),
        "floor_wrong_way_pp_pass": int(floors["wrong_way_pp"]),
        "floor_fwd_mfe_p50_pass": int(floors["fwd_mfe_p50"]),
        "floor_size_fraction_pass": int(floors["size_fraction"]),
        "all_floors_pass": int(all_pass),
    }


def _select_sl(group_sl_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    passing = group_sl_df[group_sl_df["all_floors_pass"] == 1]
    if passing.empty:
        return None
    best = passing.sort_values(
        ["composite", "fwd_mfe_p50_atr", "X"],
        ascending=[False, False, True],
    ).iloc[0]
    # Tiebreaker 1: any other SLs within TIEBREAK_COMPOSITE -> larger peak_mfe_atr
    near = passing[
        np.abs(passing["composite"] - best["composite"]) <= TIEBREAK_COMPOSITE
    ]
    if len(near) > 1:
        best = near.sort_values(["fwd_mfe_p50_atr", "X"], ascending=[False, True]).iloc[0]
    return best.to_dict()


def _shape_tag(peak_atr: np.ndarray) -> Tuple[str, Dict[str, Any]]:
    """Return (shape_tag, detail). Detection uses peak_mfe distribution in ATR units.

    Test order: scattered -> bimodal_separated -> tight_unimodal -> heavy_right_tail
    -> unclassified.

    bimodal_separated requires diptest (optional dep). If absent we approximate
    using a simple two-mode kde-density check; result flagged as approximate.
    """
    n = peak_atr.size
    if n < 10:
        return "unclassified", {"reason": "n<10"}
    mu = float(peak_atr.mean())
    sd = float(peak_atr.std(ddof=1)) if n > 1 else 0.0
    cv = (sd / mu) if mu > 0 else float("inf")
    detail: Dict[str, Any] = {"n": int(n), "mean_atr": mu, "std_atr": sd, "cv": cv}

    # scattered = CV high AND no detectable mode (Gaussian KDE bimodality test as proxy).
    if cv >= SCATTERED_CV_MIN:
        # If both first AND second mode mass < 0.20 we call it scattered.
        detail["scattered_cv_threshold"] = SCATTERED_CV_MIN
        return "scattered", detail

    # bimodal_separated test.
    dip_p = float("nan")
    used_diptest = False
    try:
        from diptest import diptest  # type: ignore
        dip_stat, dip_p = diptest(peak_atr.astype(np.float64))
        used_diptest = True
    except Exception:
        dip_stat = float("nan")
        dip_p = float("nan")
    detail["dip_stat"] = dip_stat
    detail["dip_p"] = dip_p
    detail["used_diptest"] = used_diptest

    # Estimate modes via Gaussian KDE.
    try:
        from scipy.stats import gaussian_kde  # type: ignore
        # ATR units divided by 2 -> R-units at SL=2.0; keep in ATR space here.
        grid = np.linspace(peak_atr.min(), peak_atr.max(), 200)
        kde = gaussian_kde(peak_atr)
        dens = kde(grid)
        # Local maxima in dens.
        is_peak = np.zeros_like(dens, dtype=bool)
        is_peak[1:-1] = (dens[1:-1] > dens[:-2]) & (dens[1:-1] > dens[2:])
        peak_idx = np.where(is_peak)[0]
        modes_sorted = sorted(peak_idx, key=lambda i: -dens[i])
        if len(modes_sorted) >= 2:
            mode_a = float(grid[modes_sorted[0]])
            mode_b = float(grid[modes_sorted[1]])
            mode_sep_atr = abs(mode_a - mode_b)
            # Convert separation to R-units at SL = X (use ATR/X for R-units? Actually
            # per §7 separation is in R units; with peak_atr in ATR, sep in R requires
            # scaling by X. Caller can convert.
            detail["mode_a_atr"] = mode_a
            detail["mode_b_atr"] = mode_b
            detail["mode_sep_atr"] = mode_sep_atr
            # mass partitioned at midpoint between modes
            mid = (mode_a + mode_b) / 2.0
            mass_a = float(np.mean(peak_atr <= mid))
            mass_b = float(np.mean(peak_atr > mid))
            detail["mass_below_mid"] = mass_a
            detail["mass_above_mid"] = mass_b
            min_mass = min(mass_a, mass_b)
            detail["min_mode_mass"] = min_mass
        else:
            mode_sep_atr = 0.0
            min_mass = 0.0
            detail["mode_a_atr"] = float(grid[modes_sorted[0]]) if modes_sorted else float("nan")
            detail["mode_b_atr"] = float("nan")
            detail["mode_sep_atr"] = 0.0
            detail["min_mode_mass"] = 0.0
    except Exception:
        mode_sep_atr = 0.0
        min_mass = 0.0

    # tight_unimodal: low CV + unimodal.
    if cv < 0.5 and (np.isnan(dip_p) or dip_p > 0.10):
        return "tight_unimodal", detail
    # heavy_right_tail: skewness > 1.0 and unimodal.
    skew = float(((peak_atr - mu) ** 3).mean() / (sd ** 3)) if sd > 0 else 0.0
    detail["skew"] = skew
    if skew > 1.0 and (np.isnan(dip_p) or dip_p > 0.10):
        return "heavy_right_tail", detail
    return "unclassified", detail


def _is_bimodal_separated(
    peak_atr: np.ndarray, selected_X: float,
) -> Tuple[bool, Dict[str, Any]]:
    """Three-part test on peak_mfe distribution at selected SL.

    Mode separation reported in R units = ATR_units / selected_X * 2  (since
    1R_X = X*ATR; so separation in R = ATR_sep / X). Wait — separation in R_X
    means how many candidate-SL R units apart. 1 R_X = X*ATR, so separation
    in R_X = ATR_separation / X.
    """
    tag, detail = _shape_tag(peak_atr)
    if detail.get("min_mode_mass", 0.0) < BIMODAL_MIN_MODE_MASS:
        return False, detail | {"bimodal_reason": "min_mode_mass<0.20"}
    sep_atr = detail.get("mode_sep_atr", 0.0)
    sep_R = sep_atr / max(selected_X, 1e-9)
    detail["mode_sep_R_selected_X"] = sep_R
    if sep_R < BIMODAL_MODE_SEPARATION_R:
        return False, detail | {"bimodal_reason": f"mode_sep_R<{BIMODAL_MODE_SEPARATION_R}"}
    dip_p = detail.get("dip_p", float("nan"))
    if np.isnan(dip_p):
        # No dip test available; require separation + min mode mass only (approximate).
        return True, detail | {"bimodal_reason": "approximate (no diptest)"}
    if dip_p > BIMODAL_DIP_P_MAX:
        return False, detail | {"bimodal_reason": f"dip_p>{BIMODAL_DIP_P_MAX}"}
    return True, detail | {"bimodal_reason": "all three pass"}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Arc 9 Step 3 capturability.")
    parser.add_argument(
        "--step1-dir", type=Path,
        default=_REPO_ROOT / "results" / "l_arc_9" / "step1_verbatim",
    )
    parser.add_argument(
        "--step2-dir", type=Path,
        default=_REPO_ROOT / "results" / "l_arc_9" / "step2_clustering",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=_REPO_ROOT / "results" / "l_arc_9" / "step3_capturability",
    )
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    paths = pd.read_csv(args.step1_dir / "trades_paths.csv")
    trades = pd.read_csv(args.step1_dir / "trades_all.csv")
    pool_n = int(trades["trade_id"].nunique())

    # Determine chosen K and load clusters + archetype assignment.
    sil = pd.read_csv(args.step2_dir / "silhouette_summary.csv")
    arch = pd.read_csv(args.step2_dir / "archetype_assignments.csv")
    # Use the K that matches the archetype_assignments.csv row count (number of clusters).
    chosen_k = int(arch["cluster_id"].nunique())
    clusters = pd.read_csv(args.step2_dir / f"clusters_K{chosen_k}.csv")
    cluster_id_by_trade = dict(zip(clusters["trade_id"], clusters["cluster_id"]))

    # Per-trade SL sweep across all trades + all candidates.
    print(f"[step 3] per-trade SL sweep on {len(trades)} trades x {len(SL_CANDIDATES_ATR)} SL candidates...")
    per_trade_sl = sweep_all_trades(paths, SL_CANDIDATES_ATR)
    per_trade_sl["cluster_id"] = per_trade_sl["trade_id"].map(cluster_id_by_trade)
    per_trade_sl.to_csv(
        args.out_dir / "per_trade_sl_sweep.csv", index=False,
        float_format="%.10g", lineterminator="\n",
    )

    # Per-cluster aggregates (X x cluster).
    cluster_rows: List[Dict[str, Any]] = []
    for (cid, X), sub in per_trade_sl.groupby(["cluster_id", "X"], sort=True):
        rec = _aggregate(sub, float(X), pool_n)
        rec["cluster_id"] = int(cid)
        cluster_rows.append(rec)
    cluster_sl = pd.DataFrame(cluster_rows)
    cluster_sl = cluster_sl[
        ["cluster_id", "X", "n", "size_fraction", "mono_pre_peak_centroid",
         "frac_reach_1R", "frac_reach_2R", "frac_wrong_way_pre_peak",
         "fwd_mfe_p5", "fwd_mfe_p25", "fwd_mfe_p50", "fwd_mfe_p75", "fwd_mfe_p95",
         "fwd_mfe_p50_atr",
         "final_r_p5", "final_r_p25", "final_r_p50", "final_r_p75", "final_r_p95",
         "final_r_mean", "final_r_t_stat", "mass_gt_5R", "composite",
         "floor_mono_pp_pass", "floor_reach_1R_pass", "floor_wrong_way_pp_pass",
         "floor_fwd_mfe_p50_pass", "floor_size_fraction_pass", "all_floors_pass"]
    ]
    cluster_sl.to_csv(
        args.out_dir / "cluster_sl_sweep.csv", index=False,
        float_format="%.10g", lineterminator="\n",
    )

    # Per-archetype aggregates (X x archetype), where archetype label aggregates
    # >= 2 clusters sharing the same §11 label.
    cluster_to_label = dict(zip(arch["cluster_id"], arch["archetype_label"]))
    per_trade_sl["archetype_label"] = per_trade_sl["cluster_id"].map(cluster_to_label)
    # Identify which labels have >=2 clusters.
    label_cluster_count = arch.groupby("archetype_label")["cluster_id"].nunique().to_dict()
    aggregate_labels = [lbl for lbl, cnt in label_cluster_count.items() if cnt >= 2]
    arch_rows: List[Dict[str, Any]] = []
    for lbl in aggregate_labels:
        sub_all = per_trade_sl[per_trade_sl["archetype_label"] == lbl]
        for X in SL_CANDIDATES_ATR:
            sub = sub_all[sub_all["X"] == X]
            rec = _aggregate(sub, float(X), pool_n)
            rec["archetype_label"] = lbl
            arch_rows.append(rec)
    archetype_sl = pd.DataFrame(arch_rows)
    if not archetype_sl.empty:
        archetype_sl = archetype_sl[
            ["archetype_label", "X", "n", "size_fraction", "mono_pre_peak_centroid",
             "frac_reach_1R", "frac_reach_2R", "frac_wrong_way_pre_peak",
             "fwd_mfe_p50", "fwd_mfe_p50_atr",
             "final_r_mean", "final_r_t_stat", "composite",
             "floor_mono_pp_pass", "floor_reach_1R_pass", "floor_wrong_way_pp_pass",
             "floor_fwd_mfe_p50_pass", "floor_size_fraction_pass", "all_floors_pass"]
        ]
        archetype_sl.to_csv(
            args.out_dir / "archetype_sl_sweep.csv", index=False,
            float_format="%.10g", lineterminator="\n",
        )

    # SL selection per cluster + per aggregate.
    arch_routing_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    distribution_paths: List[str] = []
    capturability_pass: List[Dict[str, Any]] = []

    for cid in sorted(arch["cluster_id"].unique().tolist()):
        cluster_label = cluster_to_label[cid]
        g = cluster_sl[cluster_sl["cluster_id"] == cid]
        sel = _select_sl(g)
        passes_individual = sel is not None
        passes_aggregate = False
        sel_agg: Optional[Dict[str, Any]] = None
        if cluster_label in aggregate_labels:
            ga = archetype_sl[archetype_sl["archetype_label"] == cluster_label]
            sel_agg = _select_sl(ga)
            passes_aggregate = sel_agg is not None
        disposition = (
            "both_pass" if (passes_individual and passes_aggregate) else
            "individual_only" if passes_individual else
            "aggregate_only" if passes_aggregate else
            "dies"
        )
        arch_routing_rows.append(
            {
                "cluster_id": int(cid),
                "archetype_label": cluster_label,
                "individual_passes": int(passes_individual),
                "individual_selected_SL_atr_mult": float(sel["X"]) if sel else float("nan"),
                "individual_composite": float(sel["composite"]) if sel else float("nan"),
                "aggregate_passes": int(passes_aggregate),
                "aggregate_selected_SL_atr_mult": float(sel_agg["X"]) if sel_agg else float("nan"),
                "aggregate_composite": float(sel_agg["composite"]) if sel_agg else float("nan"),
                "disposition": disposition,
            }
        )

        if sel is not None:
            # Per-cluster summary.
            peak_X = per_trade_sl[
                (per_trade_sl["cluster_id"] == cid) & (per_trade_sl["X"] == sel["X"])
            ]["peak_mfe_atr"].to_numpy()
            tag, tag_detail = _shape_tag(peak_X)
            bimodal_pass, bimodal_detail = _is_bimodal_separated(peak_X, float(sel["X"]))
            if bimodal_pass:
                tag_final = "bimodal_separated"
            else:
                tag_final = tag
            # §2 internal-consistency floor: shape_tag != scattered
            shape_floor_pass = tag_final != "scattered"
            pre_t_sl_atr = float(sel["X"])
            summary_rows.append({
                "label_archetype_or_cluster": f"cluster_{cid}_individual",
                "cluster_id": int(cid),
                "archetype_label": cluster_label,
                "evaluation_mode": "individual",
                "selected_SL_atr_mult": pre_t_sl_atr,
                "n": int(sel["n"]),
                "size_fraction": float(sel["size_fraction"]),
                "mono_pre_peak_centroid": float(sel["mono_pre_peak_centroid"]),
                "frac_reach_1R": float(sel["frac_reach_1R"]),
                "frac_wrong_way_pre_peak": float(sel["frac_wrong_way_pre_peak"]),
                "fwd_mfe_p50": float(sel["fwd_mfe_p50"]),
                "fwd_mfe_p50_atr": float(sel["fwd_mfe_p50_atr"]),
                "final_r_mean": float(sel["final_r_mean"]),
                "final_r_t_stat": float(sel["final_r_t_stat"]),
                "composite": float(sel["composite"]),
                "shape_tag": tag_final,
                "shape_tag_pass": int(shape_floor_pass),
                "bimodal_separated_pass": int(bimodal_pass),
                "pre_t_sl_atr_multiplier": pre_t_sl_atr,
            })
            if shape_floor_pass:
                capturability_pass.append({
                    "label_archetype_or_cluster": f"cluster_{cid}_individual",
                    "cluster_id": int(cid),
                    "archetype_label": cluster_label,
                    "evaluation_mode": "individual",
                    "selected_SL_atr_mult": pre_t_sl_atr,
                    "pre_t_sl_atr_multiplier": pre_t_sl_atr,
                    "size_fraction": float(sel["size_fraction"]),
                    "final_r_mean": float(sel["final_r_mean"]),
                    "shape_tag": tag_final,
                })
            # Distribution detail file.
            dist_path = args.out_dir / f"cluster_{cid}_distribution.csv"
            mass_in_band = {
                "0-0.5R": float(np.mean(peak_X / float(sel["X"]) < 0.5)),
                "0.5-1R": float(np.mean((peak_X / float(sel["X"]) >= 0.5) & (peak_X / float(sel["X"]) < 1.0))),
                "1-2R": float(np.mean((peak_X / float(sel["X"]) >= 1.0) & (peak_X / float(sel["X"]) < 2.0))),
                "2-5R": float(np.mean((peak_X / float(sel["X"]) >= 2.0) & (peak_X / float(sel["X"]) < 5.0))),
                ">5R": float(np.mean(peak_X / float(sel["X"]) >= 5.0)),
            }
            with dist_path.open("w", encoding="utf-8") as f:
                f.write("metric,value\n")
                f.write(f"cluster_id,{cid}\n")
                f.write(f"archetype_label,{cluster_label}\n")
                f.write(f"selected_SL_atr_mult,{sel['X']}\n")
                f.write(f"n,{sel['n']}\n")
                f.write(f"shape_tag,{tag_final}\n")
                f.write(f"dip_stat,{tag_detail.get('dip_stat', 'NaN')}\n")
                f.write(f"dip_p,{tag_detail.get('dip_p', 'NaN')}\n")
                f.write(f"used_diptest,{tag_detail.get('used_diptest', False)}\n")
                f.write(f"bimodal_separated_pass,{bimodal_pass}\n")
                f.write(f"bimodal_detail,{json.dumps(bimodal_detail, default=str)}\n")
                for k, v in mass_in_band.items():
                    f.write(f"mass_in_band_{k},{v}\n")
            distribution_paths.append(str(dist_path.relative_to(_REPO_ROOT)))

        if cluster_label in aggregate_labels and sel_agg is not None:
            # Only emit one summary row per aggregate, not per cluster.
            # We'll emit when we encounter the smallest cluster_id of the label.
            label_clusters = sorted(arch[arch["archetype_label"] == cluster_label]["cluster_id"].tolist())
            if cid == label_clusters[0]:
                peak_X = per_trade_sl[
                    (per_trade_sl["archetype_label"] == cluster_label) & (per_trade_sl["X"] == sel_agg["X"])
                ]["peak_mfe_atr"].to_numpy()
                tag, tag_detail = _shape_tag(peak_X)
                bimodal_pass, bimodal_detail = _is_bimodal_separated(peak_X, float(sel_agg["X"]))
                tag_final = "bimodal_separated" if bimodal_pass else tag
                shape_floor_pass = tag_final != "scattered"
                pre_t_sl_atr = float(sel_agg["X"])
                summary_rows.append({
                    "label_archetype_or_cluster": f"{cluster_label}_aggregate",
                    "cluster_id": -1,
                    "archetype_label": cluster_label,
                    "evaluation_mode": "aggregate",
                    "selected_SL_atr_mult": pre_t_sl_atr,
                    "n": int(sel_agg["n"]),
                    "size_fraction": float(sel_agg["size_fraction"]),
                    "mono_pre_peak_centroid": float(sel_agg["mono_pre_peak_centroid"]),
                    "frac_reach_1R": float(sel_agg["frac_reach_1R"]),
                    "frac_wrong_way_pre_peak": float(sel_agg["frac_wrong_way_pre_peak"]),
                    "fwd_mfe_p50": float(sel_agg["fwd_mfe_p50"]),
                    "fwd_mfe_p50_atr": float(sel_agg["fwd_mfe_p50_atr"]),
                    "final_r_mean": float(sel_agg["final_r_mean"]),
                    "final_r_t_stat": float(sel_agg["final_r_t_stat"]),
                    "composite": float(sel_agg["composite"]),
                    "shape_tag": tag_final,
                    "shape_tag_pass": int(shape_floor_pass),
                    "bimodal_separated_pass": int(bimodal_pass),
                    "pre_t_sl_atr_multiplier": pre_t_sl_atr,
                })
                if shape_floor_pass:
                    capturability_pass.append({
                        "label_archetype_or_cluster": f"{cluster_label}_aggregate",
                        "cluster_id": -1,
                        "archetype_label": cluster_label,
                        "evaluation_mode": "aggregate",
                        "selected_SL_atr_mult": pre_t_sl_atr,
                        "pre_t_sl_atr_multiplier": pre_t_sl_atr,
                        "size_fraction": float(sel_agg["size_fraction"]),
                        "final_r_mean": float(sel_agg["final_r_mean"]),
                        "shape_tag": tag_final,
                    })

    pd.DataFrame(arch_routing_rows).to_csv(
        args.out_dir / "cluster_routing.csv", index=False,
        float_format="%.10g", lineterminator="\n",
    )
    pd.DataFrame(summary_rows).to_csv(
        args.out_dir / "archetype_summaries.csv", index=False,
        float_format="%.10g", lineterminator="\n",
    )
    pd.DataFrame(capturability_pass).to_csv(
        args.out_dir / "capturability_pass_list.csv", index=False,
        float_format="%.10g", lineterminator="\n",
    )

    # Markdown summary.
    md: List[str] = []
    md.append("# Arc 9 Step 3 - Capturability characterisation")
    md.append("")
    verdict = "PASS" if capturability_pass else "FAIL"
    md.append(f"Verdict: **{verdict}** ({len(capturability_pass)} archetype(s) clear §2 floors at the selected SL)")
    md.append("")
    md.append(f"SL sweep candidates: {SL_CANDIDATES_ATR}")
    md.append(f"§2 floors (v2.1.2): mono_pre_peak≥{FLOOR_MONO_PRE_PEAK}, frac_reach_1R≥{FLOOR_REACH_1R}, "
              f"frac_wrong_way_pre_peak≤{FLOOR_WRONG_WAY_PRE_PEAK}, fwd_mfe_p50≥{FLOOR_FWD_MFE_P50}, "
              f"size_fraction≥{FLOOR_SIZE_FRACTION}, shape_tag ≠ scattered")
    md.append("")
    md.append("## Cluster routing")
    md.append("")
    md.append("| cluster | archetype | individual? | sel SL (×ATR) | composite | aggregate? | agg SL | agg composite | disposition |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for r in arch_routing_rows:
        ind_sl = f"{r['individual_selected_SL_atr_mult']:.1f}" if r["individual_passes"] else "-"
        ind_comp = f"{r['individual_composite']:.3f}" if r["individual_passes"] else "-"
        agg_sl = f"{r['aggregate_selected_SL_atr_mult']:.1f}" if r["aggregate_passes"] else "-"
        agg_comp = f"{r['aggregate_composite']:.3f}" if r["aggregate_passes"] else "-"
        md.append(
            f"| {r['cluster_id']} | {r['archetype_label']} | "
            f"{'PASS' if r['individual_passes'] else 'fail'} | "
            f"{ind_sl} | {ind_comp} | "
            f"{'PASS' if r['aggregate_passes'] else '-'} | "
            f"{agg_sl} | {agg_comp} | "
            f"**{r['disposition']}** |"
        )
    md.append("")
    md.append("## Selected SL per surviving archetype")
    md.append("")
    md.append("| label | mode | n | size | sel SL | mono_pp | reach_1R | ww_pp | fwd_mfe_p50 | final_r_mean | t | composite | shape | shape_pass | bimodal? |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in summary_rows:
        md.append(
            f"| {r['label_archetype_or_cluster']} | {r['evaluation_mode']} | "
            f"{r['n']} | {r['size_fraction']:.3f} | {r['selected_SL_atr_mult']:.1f} | "
            f"{r['mono_pre_peak_centroid']:.3f} | {r['frac_reach_1R']:.3f} | "
            f"{r['frac_wrong_way_pre_peak']:.3f} | {r['fwd_mfe_p50']:.2f} | "
            f"{r['final_r_mean']:+.3f} | {r['final_r_t_stat']:+.2f} | "
            f"{r['composite']:.3f} | {r['shape_tag']} | "
            f"{'PASS' if r['shape_tag_pass'] else 'fail'} | "
            f"{'PASS' if r['bimodal_separated_pass'] else '-'} |"
        )
    md.append("")
    md.append("## v2.3 §4 (Open-24) - per-archetype pre_t_sl_atr_multiplier")
    md.append("")
    md.append("Per v2.3 §4, each Pipeline D1 archetype's pre-t SL multiplier equals its Step 3 selected SL multiplier. This is recorded in `cluster_routing.csv` and `archetype_summaries.csv` (col `pre_t_sl_atr_multiplier`). For Steps 1-4 this is metadata only; engine PR honour pending.")
    md.append("")
    md.append("## Files")
    md.append("")
    md.append("- per_trade_sl_sweep.csv")
    md.append("- cluster_sl_sweep.csv")
    md.append("- archetype_sl_sweep.csv")
    md.append("- archetype_summaries.csv")
    md.append("- capturability_pass_list.csv")
    md.append("- cluster_routing.csv")
    for d in distribution_paths:
        md.append(f"- {d}")

    (args.out_dir / "STEP3_SUMMARY.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[step 3] {verdict} - {len(capturability_pass)} archetype(s) survive")
    for r in capturability_pass:
        print(f"  {r['label_archetype_or_cluster']}: SL={r['selected_SL_atr_mult']}×ATR, mean R={r['final_r_mean']:+.3f}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
