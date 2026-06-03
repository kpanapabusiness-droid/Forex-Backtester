"""Per-cluster forward geometry metrics for KH-24 v2.0 Step 3.

All percentiles / fractions / t-stats are computed from the per-trade
summary (`trades_all.csv`) or the per-bar paths (`trades_paths.csv`) joined
on the cluster assignment from Step 2 (`clusters_K5.csv`).

Two definitions of `frac_wrong_way` are surfaced — the protocol's §17
glossary doesn't pin it down precisely:

  Definition A (fallback): trade ended materially adverse, i.e.
      `final_r <= -0.5`.

  Definition B (intent-aligned): trade was directionally wrong from the
      outset, i.e. `mae_so_far_r <= -1.0` was hit before any
      `mfe_so_far_r > 0.5` was reached. Trades that never reach
      `mfe_so_far_r > 0.5` are counted as wrong-way regardless of MAE.

Both are reported in `archetype_summaries.csv`. The §2 evaluation uses
Definition B (matches the prompt's intent statement and KH-24's anchor
`frac_wrong_way = 0.04` — only Definition B can produce numbers that
small on a hard-SL-only design).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

PERCENTILES = (5, 25, 50, 75, 95)


class ClusterTradeData(NamedTuple):
    cluster_id: int
    trade_ids: list[str]
    trades_df: pd.DataFrame  # per-trade summary for this cluster
    paths_df: pd.DataFrame  # per-bar paths for this cluster's trades


def split_by_cluster(
    trades_df: pd.DataFrame, paths_df: pd.DataFrame, clusters_df: pd.DataFrame
) -> list[ClusterTradeData]:
    """Partition trades + paths by cluster_id.

    Returns one ClusterTradeData per unique cluster_id, sorted asc.
    """
    merged = trades_df.merge(clusters_df, on="trade_id", how="inner")
    out: list[ClusterTradeData] = []
    for cid, sub in merged.groupby("cluster_id", sort=True):
        tids = sub["trade_id"].tolist()
        paths_sub = paths_df[paths_df["trade_id"].isin(tids)].copy()
        out.append(
            ClusterTradeData(
                cluster_id=int(cid),
                trade_ids=tids,
                trades_df=sub.reset_index(drop=True),
                paths_df=paths_sub.reset_index(drop=True),
            )
        )
    return out


def _percentiles(x: np.ndarray, qs: tuple[int, ...] = PERCENTILES) -> dict[int, float]:
    if len(x) == 0:
        return {q: float("nan") for q in qs}
    pct = np.percentile(x, list(qs), method="linear")
    return {q: float(v) for q, v in zip(qs, pct)}


def _t_stat(x: np.ndarray) -> tuple[float, float, float]:
    """Return (mean, std (ddof=1), t-stat = mean / (std / sqrt(n)))."""
    n = len(x)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(x))
    if n == 1:
        return mean, 0.0, float("nan")
    std = float(np.std(x, ddof=1))
    if std == 0.0:
        return mean, 0.0, float("nan") if mean == 0.0 else float("inf") * np.sign(mean)
    return mean, std, mean / (std / np.sqrt(n))


def frac_wrong_way_definition_a(final_r: np.ndarray, threshold: float = -0.5) -> float:
    """Definition A — fraction of trades with final_r <= -0.5."""
    if len(final_r) == 0:
        return float("nan")
    return float(np.mean(final_r <= threshold))


def frac_wrong_way_definition_b(
    paths_df: pd.DataFrame, mfe_thresh: float = 0.5, mae_thresh: float = -1.0
) -> float:
    """Definition B — wrong-from-outset: never reached mfe > 0.5R, OR mae <= -1
    happened before the first bar with mfe > 0.5R.
    """
    if paths_df.empty:
        return float("nan")
    wrong = 0
    total = 0
    for _, group in paths_df.groupby("trade_id", sort=False):
        total += 1
        g = group.sort_values("bar_offset")
        mfe = g["mfe_so_far_r"].to_numpy(dtype=np.float64)
        mae = g["mae_so_far_r"].to_numpy(dtype=np.float64)
        # First bar where mfe > 0.5
        above = np.flatnonzero(mfe > mfe_thresh)
        if above.size == 0:
            wrong += 1
            continue
        first_profit_bar = int(above[0])
        # Was mae <= -1 anywhere BEFORE first_profit_bar?
        if first_profit_bar > 0 and (mae[:first_profit_bar] <= mae_thresh).any():
            wrong += 1
    return wrong / total if total > 0 else float("nan")


def pct_peak_and_collapse(
    trades_df: pd.DataFrame,
    features_df: pd.DataFrame,
    ttp_thresh: float = 0.30,
    collapse_ratio: float = 0.5,
) -> float:
    """Per §11 boundary-rule context: fraction of trades where
        (a) time_to_peak_mfe_relative <= ttp_thresh, AND
        (b) final_r <= peak_mfe_r * collapse_ratio.

    `peak_mfe_r` is the trade's `mfe_r` (max excursion over held window).
    """
    if trades_df.empty:
        return float("nan")
    merged = trades_df.merge(
        features_df[["trade_id", "time_to_peak_mfe_relative"]], on="trade_id", how="inner"
    )
    early = merged["time_to_peak_mfe_relative"].to_numpy(dtype=np.float64) <= ttp_thresh
    collapse = merged["final_r"].to_numpy(dtype=np.float64) <= collapse_ratio * merged[
        "mfe_r"
    ].to_numpy(dtype=np.float64)
    return float(np.mean(early & collapse))


def compute_cluster_forward_geometry(
    cluster_data: ClusterTradeData,
    features_df: pd.DataFrame,
    pool_size: int,
) -> dict:
    """All forward-geometry + identity metrics for a single cluster."""
    tdf = cluster_data.trades_df
    pdf = cluster_data.paths_df
    n = len(tdf)
    final_r = tdf["final_r"].to_numpy(dtype=np.float64)
    mfe_r = tdf["mfe_r"].to_numpy(dtype=np.float64)
    mae_r = tdf["mae_r"].to_numpy(dtype=np.float64)
    bars_held = tdf["bars_held"].to_numpy(dtype=np.int64)

    mean, std, tstat = _t_stat(final_r)
    pct_fr = _percentiles(final_r)
    pct_mfe = _percentiles(mfe_r)
    pct_mae = _percentiles(mae_r)
    pct_bh = _percentiles(bars_held.astype(np.float64))

    frac_1r = float(np.mean(mfe_r >= 1.0))
    frac_2r = float(np.mean(mfe_r >= 2.0))
    fww_a = frac_wrong_way_definition_a(final_r)
    fww_b = frac_wrong_way_definition_b(pdf)
    ppc = pct_peak_and_collapse(tdf, features_df)
    cap_240 = float(np.mean(bars_held >= 240))

    return {
        "cluster_id": cluster_data.cluster_id,
        "size_count": n,
        "size_fraction_of_pool": n / pool_size,
        "fwd_mfe_p5": pct_mfe[5],
        "fwd_mfe_p25": pct_mfe[25],
        "fwd_mfe_p50": pct_mfe[50],
        "fwd_mfe_p75": pct_mfe[75],
        "fwd_mfe_p95": pct_mfe[95],
        "final_r_mean": mean,
        "final_r_std": std,
        "final_r_t_stat": tstat,
        "final_r_p5": pct_fr[5],
        "final_r_p25": pct_fr[25],
        "final_r_p50": pct_fr[50],
        "final_r_p75": pct_fr[75],
        "final_r_p95": pct_fr[95],
        "mae_r_p5": pct_mae[5],
        "mae_r_p25": pct_mae[25],
        "mae_r_p50": pct_mae[50],
        "mae_r_p75": pct_mae[75],
        "mae_r_p95": pct_mae[95],
        "bars_held_p5": pct_bh[5],
        "bars_held_p25": pct_bh[25],
        "bars_held_p50": pct_bh[50],
        "bars_held_p75": pct_bh[75],
        "bars_held_p95": pct_bh[95],
        "frac_reach_1R": frac_1r,
        "frac_reach_2R": frac_2r,
        "frac_wrong_way_def_a": fww_a,
        "frac_wrong_way_def_b": fww_b,
        "pct_peak_and_collapse": ppc,
        "frac_cap_240": cap_240,
    }
