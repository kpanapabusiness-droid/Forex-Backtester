"""Step 3 — per-cluster capturability per L_PROTOCOL §2 Step 3.

Per cluster (from Step 2):

  - reach rates P(MFE ≥ {1R, 2R, 3R})
  - MFE distribution percentiles (p25, p50, p75, p90)
  - wrong_way_pp (fraction hitting -1R MAE before +1R MFE)
  - time-to-peak distribution (p25, p50, p75)
  - mean R, p25 R, p50 R
  - SL multiplier sweep across {1.5, 2.0, 2.5, 3.0, 3.5, 4.0}×ATR with
    per-multiplier capturability composite — picks an arc-default SL
  - capturability composite at the selected SL
  - candidate-cluster flag: reach_1R ≥ 0.50 ∧ ww_pp ≤ 0.30 ∧ mfe_p50 ≥ 1.5R

The SL sweep operates on the per-trade path data Step 1 emitted at the
ARC's declared sl_atr_mult. Re-scaling to a hypothetical alternate
multiplier is done by inflating / deflating R-multiples by the ratio.
This is approximate (the SL would have fired at a different bar for a
different multiplier), but suitable for Step 3 ranking; Step 5 re-runs
real WFO at each candidate SL.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import hashlib
import numpy as np
import pandas as pd

from core.steps._shape_tags import (
    ClusterCentroid,
    assign_shape_tag,
)

SL_MULT_SWEEP = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
REACH_THRESHOLDS = (1.0, 2.0, 3.0)
COMPOSITE_W_REACH1 = 0.40
COMPOSITE_W_MFE_P50 = 0.40
COMPOSITE_W_1MINUS_WW = 0.20

CANDIDATE_REACH_1R_MIN = 0.50
CANDIDATE_WW_PP_MAX = 0.30
CANDIDATE_MFE_P50_MIN = 1.5


@dataclass(frozen=True)
class ClusterCapturability:
    """Per-cluster Step 3 metrics + verdict flag."""

    cluster_id: int
    n_trades: int
    shape_tag: str
    reach_1r: float
    reach_2r: float
    reach_3r: float
    mfe_p25: float
    mfe_p50: float
    mfe_p75: float
    mfe_p90: float
    wrong_way_pp: float
    ttp_p25: int
    ttp_p50: int
    ttp_p75: int
    mean_r: float
    final_r_p25: float
    final_r_p50: float
    sl_sweep: Mapping[float, float]  # multiplier -> composite at this SL
    selected_sl: float
    capturability_composite: float
    is_candidate: bool


@dataclass(frozen=True)
class Step3Result:
    """Output of :func:`run_step_3`."""

    per_cluster: tuple[ClusterCapturability, ...]
    capturability_csv: pd.DataFrame
    summary_md: str


def _cluster_metrics(
    trades_cluster: pd.DataFrame, paths_cluster: pd.DataFrame, declared_sl: float
) -> tuple[float, float, float, float, float, float, float, float, int, int, int, float, float, float, dict[float, float]]:
    """Compute the per-cluster numbers; returns a tuple in field order."""
    mfe = trades_cluster["mfe_r"].values
    mae = trades_cluster["mae_r"].values
    final_r = trades_cluster["final_r"].values
    bars_held = trades_cluster["bars_held"].values

    n = len(trades_cluster)
    reach_1r = float(np.mean(mfe >= 1.0)) if n else 0.0
    reach_2r = float(np.mean(mfe >= 2.0)) if n else 0.0
    reach_3r = float(np.mean(mfe >= 3.0)) if n else 0.0
    mfe_p25 = float(np.percentile(mfe, 25)) if n else 0.0
    mfe_p50 = float(np.percentile(mfe, 50)) if n else 0.0
    mfe_p75 = float(np.percentile(mfe, 75)) if n else 0.0
    mfe_p90 = float(np.percentile(mfe, 90)) if n else 0.0

    # wrong_way_pp: from paths, fraction of trades whose mae_so_far_r reached -1.0
    # before mfe_so_far_r reached +1.0
    ww_count = 0
    for tid, group in paths_cluster.groupby("trade_id", sort=True):
        group = group.sort_values("bar_offset")
        m_arr = group["mfe_so_far_r"].values
        ae_arr = group["mae_so_far_r"].values
        ww = False
        for i in range(len(group)):
            if ae_arr[i] <= -1.0:
                ww = True
                break
            if m_arr[i] >= 1.0:
                ww = False
                break
        if ww:
            ww_count += 1
    wrong_way_pp = ww_count / max(1, n)

    ttp_p25 = int(np.percentile(bars_held, 25)) if n else 0
    ttp_p50 = int(np.percentile(bars_held, 50)) if n else 0
    ttp_p75 = int(np.percentile(bars_held, 75)) if n else 0

    mean_r = float(np.mean(final_r)) if n else 0.0
    final_r_p25 = float(np.percentile(final_r, 25)) if n else 0.0
    final_r_p50 = float(np.percentile(final_r, 50)) if n else 0.0

    # SL sweep: per multiplier mult, scale R values by (declared_sl / mult)
    # (a larger SL produces smaller R; this is approximate as documented)
    sweep: dict[float, float] = {}
    for mult in SL_MULT_SWEEP:
        scale = declared_sl / mult
        scaled_mfe = mfe * scale
        scaled_reach_1 = float(np.mean(scaled_mfe >= 1.0)) if n else 0.0
        scaled_mfe_p50 = float(np.percentile(scaled_mfe, 50)) if n else 0.0
        # ww_pp doesn't rescale cleanly — keep at declared_sl value
        composite = (
            COMPOSITE_W_REACH1 * scaled_reach_1
            + COMPOSITE_W_MFE_P50 * (scaled_mfe_p50 / 3.0)  # normalise by 3R upper
            + COMPOSITE_W_1MINUS_WW * (1.0 - wrong_way_pp)
        )
        sweep[float(mult)] = float(composite)
    return (
        reach_1r, reach_2r, reach_3r,
        mfe_p25, mfe_p50, mfe_p75, mfe_p90,
        wrong_way_pp,
        ttp_p25, ttp_p50, ttp_p75,
        mean_r, final_r_p25, final_r_p50,
        sweep,
    )


def run_step_3(
    trades: pd.DataFrame,
    paths: pd.DataFrame,
    cluster_assignments: pd.DataFrame,
    *,
    declared_sl_mult: float = 2.0,
    cluster_centroids: Mapping[int, ClusterCentroid] | None = None,
) -> Step3Result:
    """Run Step 3 across every cluster in ``cluster_assignments``.

    ``declared_sl_mult`` is the SL multiplier Step 1 simulated at —
    the SL sweep rescales path-R values around this anchor.

    ``cluster_centroids`` from Step 2 is consulted for shape_tag
    assignment. If absent the per-cluster aggregates from this Step are
    used to derive the tag.
    """
    if "cluster_id" not in cluster_assignments.columns:
        raise ValueError("cluster_assignments must have cluster_id column")
    merged = trades.merge(
        cluster_assignments[["trade_id", "cluster_id"]],
        on="trade_id",
        how="inner",
    )
    cluster_ids = sorted(set(int(c) for c in merged["cluster_id"]))

    per_cluster: list[ClusterCapturability] = []
    csv_rows: list[dict] = []

    for cid in cluster_ids:
        sub_trades = merged[merged["cluster_id"] == cid]
        sub_path_tids = set(sub_trades["trade_id"].astype(int))
        sub_paths = paths[paths["trade_id"].isin(sub_path_tids)]
        m = _cluster_metrics(sub_trades, sub_paths, declared_sl_mult)
        (
            reach_1r, reach_2r, reach_3r,
            mfe_p25, mfe_p50, mfe_p75, mfe_p90,
            wrong_way_pp,
            ttp_p25, ttp_p50, ttp_p75,
            mean_r, final_r_p25, final_r_p50,
            sweep,
        ) = m

        # Pick SL from the sweep argmax
        selected_sl = max(sweep.items(), key=lambda kv: kv[1])[0]
        composite = sweep[selected_sl]

        if cluster_centroids is not None and cid in cluster_centroids:
            shape_tag = assign_shape_tag(cluster_centroids[cid])
        else:
            # Derive tag from per-cluster metrics
            implied_centroid = ClusterCentroid(
                cluster_id=cid,
                monotonicity=0.5,  # unknown without path-shape features here
                local_peaks=1.0,
                mfe_p50=mfe_p50,
                time_to_peak_rel=0.5,
                wrong_way_pp=wrong_way_pp,
            )
            shape_tag = assign_shape_tag(implied_centroid)

        is_candidate = (
            reach_1r >= CANDIDATE_REACH_1R_MIN
            and wrong_way_pp <= CANDIDATE_WW_PP_MAX
            and mfe_p50 >= CANDIDATE_MFE_P50_MIN
        )
        cap = ClusterCapturability(
            cluster_id=cid,
            n_trades=len(sub_trades),
            shape_tag=shape_tag,
            reach_1r=reach_1r,
            reach_2r=reach_2r,
            reach_3r=reach_3r,
            mfe_p25=mfe_p25,
            mfe_p50=mfe_p50,
            mfe_p75=mfe_p75,
            mfe_p90=mfe_p90,
            wrong_way_pp=wrong_way_pp,
            ttp_p25=ttp_p25,
            ttp_p50=ttp_p50,
            ttp_p75=ttp_p75,
            mean_r=mean_r,
            final_r_p25=final_r_p25,
            final_r_p50=final_r_p50,
            sl_sweep=sweep,
            selected_sl=selected_sl,
            capturability_composite=composite,
            is_candidate=is_candidate,
        )
        per_cluster.append(cap)
        csv_rows.append({
            "cluster_id": cid,
            "n_trades": cap.n_trades,
            "shape_tag": shape_tag,
            "reach_1r": reach_1r,
            "reach_2r": reach_2r,
            "reach_3r": reach_3r,
            "mfe_p25": mfe_p25,
            "mfe_p50": mfe_p50,
            "mfe_p75": mfe_p75,
            "mfe_p90": mfe_p90,
            "wrong_way_pp": wrong_way_pp,
            "ttp_p25": ttp_p25,
            "ttp_p50": ttp_p50,
            "ttp_p75": ttp_p75,
            "mean_r": mean_r,
            "final_r_p25": final_r_p25,
            "final_r_p50": final_r_p50,
            "selected_sl": selected_sl,
            "capturability_composite": composite,
            "is_candidate": is_candidate,
        })

    csv = pd.DataFrame(csv_rows)
    if len(csv) > 0:
        csv = csv.sort_values("capturability_composite", ascending=False).reset_index(drop=True)
    md = _render_summary_md(csv)
    return Step3Result(
        per_cluster=tuple(per_cluster),
        capturability_csv=csv,
        summary_md=md,
    )


def _render_summary_md(csv: pd.DataFrame) -> str:
    lines = ["# Step 3 — Capturability Summary", ""]
    if len(csv) == 0:
        lines.append("(no clusters to summarise)")
        return "\n".join(lines) + "\n"
    lines.append(
        "| cluster | n | tag | reach_1R | reach_2R | mfe_p50 | ww_pp | "
        "ttp_p50 | mean_R | sel_SL | composite | candidate |"
    )
    lines.append("|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|")
    for _, r in csv.iterrows():
        lines.append(
            f"| {int(r['cluster_id'])} | {int(r['n_trades'])} | {r['shape_tag']} | "
            f"{r['reach_1r']:.3f} | {r['reach_2r']:.3f} | {r['mfe_p50']:.3f} | "
            f"{r['wrong_way_pp']:.3f} | {int(r['ttp_p50'])} | {r['mean_r']:+.3f} | "
            f"{r['selected_sl']:.1f} | {r['capturability_composite']:.4f} | "
            f"{'✓' if r['is_candidate'] else ''} |"
        )
    return "\n".join(lines) + "\n"


def step_3_sha256(result: Step3Result) -> str:
    payload = result.capturability_csv.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = (
    "SL_MULT_SWEEP",
    "ClusterCapturability",
    "Step3Result",
    "run_step_3",
    "step_3_sha256",
)
