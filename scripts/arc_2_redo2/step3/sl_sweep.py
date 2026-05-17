"""Per-trade SL re-imposition + per-cluster aggregates (v2.1.1 §7).

The trades_paths.csv schema (v2.1.1 fork) carries:
  bar_offset, low, close, close_r, mfe_so_far_r, mae_so_far_r, is_held
where close_r/mfe_so_far_r/mae_so_far_r are R-normalised at BASELINE SL
(2.0 × ATR(14) at signal bar). To evaluate a candidate SL X × ATR we
re-impose by scanning the full path (held + forward-obs combined) and:

  1. SL trigger: first bar B where `low_r_baseline ≤ -X/2`. low_r_baseline
     = (low - entry_price) / sl_distance_baseline. If never triggers → B=240.
  2. Truncate path at B (inclusive). Compute peak_mfe_bar_X = first bar in
     [0, B] reaching max close_r (close-based mfe, matching CSV's
     mfe_so_far_r convention).
  3. Convert peak_mfe to candidate-SL R: peak_mfe_X = peak_mfe_baseline × 2/X.
     final_r_X = -1.0 if SL triggered, else close_r_at_bar240 × 2/X.
  4. Per-bar bools:
       - reached_1R_X = peak_mfe_baseline ≥ X/2  (i.e. peak ≥ 1 candidate-SL R)
       - wrong_way_pre_peak_X = low_r_baseline ≤ -X/2 at any bar in [0, peak_mfe_bar_X]
     mono_pre_peak_X = monotonicity_ratio_in_profit on close_r over [0, peak_mfe_bar_X].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TradeSLResult:
    trade_id: int
    X: float                      # candidate SL multiplier (×ATR)
    truncated_at_bar: int
    sl_triggered: bool
    peak_mfe_bar_X: int
    peak_mfe_X: float             # in candidate-SL R units
    peak_mfe_atr: float           # in ATR units (independent of X)
    final_r_X: float              # in candidate-SL R units
    reached_1R_X: bool
    wrong_way_pre_peak_X: bool
    mono_pre_peak_X: float


def _monotonicity_in_profit(close_r: np.ndarray) -> float:
    """Among bars where close_r > 0, fraction where close_r ≥ previous in-profit bar."""
    in_profit = close_r[close_r > 0]
    if in_profit.size < 2:
        return 0.0
    return float(np.mean(in_profit[1:] >= in_profit[:-1]))


def per_trade_sweep(
    trade_id: int,
    close_r_baseline: np.ndarray,   # full path: bar 0..240 in baseline R
    low_r_baseline: np.ndarray,     # full path: bar 0..240 in baseline R
    candidates_atr: List[float],
) -> List[TradeSLResult]:
    """Sweep candidate SLs on one trade. close_r and low_r are pre-aligned
    to bar_offset (positional). Returns one result per candidate.
    """
    n = close_r_baseline.size
    if n == 0:
        return []

    results: List[TradeSLResult] = []
    for X in candidates_atr:
        threshold = -X / 2.0   # baseline-R threshold for SL trigger
        below = low_r_baseline <= threshold
        if below.any():
            trigger_bar = int(np.argmax(below))
            sl_triggered = True
        else:
            trigger_bar = n - 1   # bar 240
            sl_triggered = False

        # Truncated path inclusive of trigger_bar.
        cr_trunc = close_r_baseline[: trigger_bar + 1]
        low_trunc = low_r_baseline[: trigger_bar + 1]

        # peak_mfe_bar_X = FIRST bar in truncated path where close_r reaches max.
        peak_mfe_bar_X = int(np.argmax(cr_trunc))
        peak_close_baseline = float(cr_trunc[peak_mfe_bar_X])
        peak_mfe_X = peak_close_baseline * 2.0 / X
        # peak_mfe in ATR units (for tiebreaker) is independent of X:
        # close_r_baseline = (close - entry) / (2 × ATR), so close_r_baseline × 2 = (close-entry)/ATR
        peak_mfe_atr = peak_close_baseline * 2.0

        if sl_triggered:
            final_r_X = -1.0
        else:
            # close_r at bar 240 (last bar in path) — use cr_trunc[-1].
            close_at_240_baseline = float(cr_trunc[-1])
            final_r_X = close_at_240_baseline * 2.0 / X

        reached_1R_X = bool(peak_close_baseline >= X / 2.0)

        # wrong_way_pre_peak_X: low_r ≤ -X/2 at ANY bar in [0, peak_mfe_bar_X]
        # peak_mfe_bar_X inclusive: per glossary "ON or before peak_mfe_bar".
        low_pre_peak = low_trunc[: peak_mfe_bar_X + 1]
        wrong_way_pre_peak_X = bool((low_pre_peak <= threshold).any())

        # mono_pre_peak_X: monotonicity on close_r in [0, peak_mfe_bar_X]
        cr_pre_peak = cr_trunc[: peak_mfe_bar_X + 1]
        mono_pre_peak_X = _monotonicity_in_profit(cr_pre_peak)

        results.append(
            TradeSLResult(
                trade_id=trade_id,
                X=X,
                truncated_at_bar=trigger_bar,
                sl_triggered=sl_triggered,
                peak_mfe_bar_X=peak_mfe_bar_X,
                peak_mfe_X=peak_mfe_X,
                peak_mfe_atr=peak_mfe_atr,
                final_r_X=final_r_X,
                reached_1R_X=reached_1R_X,
                wrong_way_pre_peak_X=wrong_way_pre_peak_X,
                mono_pre_peak_X=mono_pre_peak_X,
            )
        )
    return results


def sweep_all_trades(
    paths_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    candidates_atr: List[float],
) -> pd.DataFrame:
    """Run per_trade_sweep on every trade. Returns one row per (trade_id, X).

    paths_df columns expected: trade_id, bar_offset, low, close, close_r, is_held.
    trades_df columns expected: trade_id, entry_price, sl_price.
    """
    # Build per-trade lookup of sl_distance.
    t = trades_df[["trade_id", "entry_price", "sl_price"]].copy()
    t["sl_distance"] = t["entry_price"] - t["sl_price"]
    paths = paths_df.merge(t[["trade_id", "entry_price", "sl_distance"]], on="trade_id", how="left")
    paths["low_r_baseline"] = (paths["low"].astype(float) - paths["entry_price"]) / paths["sl_distance"]
    # close_r is already baseline R per step1 schema; keep as-is.
    paths = paths.sort_values(["trade_id", "bar_offset"], kind="mergesort")

    # Iterate trade groups via groupby — fast enough for 12k trades × 241 bars.
    out_rows: List[TradeSLResult] = []
    for tid, grp in paths.groupby("trade_id", sort=True):
        # Confirm contiguous bar_offset 0..N.
        bar_off = grp["bar_offset"].to_numpy(dtype=int)
        close_r = grp["close_r"].to_numpy(dtype=float)
        low_r = grp["low_r_baseline"].to_numpy(dtype=float)
        # Defensive: ensure sorted contiguous from 0.
        if bar_off[0] != 0 or not np.array_equal(bar_off, np.arange(bar_off.size)):
            # Re-sort and re-extract if non-contiguous (shouldn't happen).
            order = np.argsort(bar_off, kind="mergesort")
            close_r = close_r[order]
            low_r = low_r[order]
        results = per_trade_sweep(int(tid), close_r, low_r, candidates_atr)
        out_rows.extend(results)

    df = pd.DataFrame([r.__dict__ for r in out_rows])
    return df


def aggregate_per_cluster(
    per_trade_sl_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate per-trade SL results to per-(cluster, X) summary rows."""
    merged = per_trade_sl_df.merge(clusters_df, on="trade_id", how="left")
    if merged["cluster_id"].isna().any():
        raise ValueError("Per-trade results contain trade_ids missing from clusters_df")

    rows: List[Dict[str, float]] = []
    grp = merged.groupby(["cluster_id", "X"])
    for (cid, X), sub in grp:
        n = len(sub)
        if n == 0:
            continue
        peak_mfe_X = sub["peak_mfe_X"].to_numpy(dtype=float)
        peak_mfe_atr = sub["peak_mfe_atr"].to_numpy(dtype=float)
        final_r_X = sub["final_r_X"].to_numpy(dtype=float)

        p5, p25, p50, p75, p95 = np.percentile(peak_mfe_X, [5, 25, 50, 75, 95])
        fp5, fp25, fp50, fp75, fp95 = np.percentile(final_r_X, [5, 25, 50, 75, 95])
        fr_mean = float(final_r_X.mean())
        fr_std = float(final_r_X.std(ddof=1)) if n > 1 else 0.0
        fr_tstat = (fr_mean / (fr_std / np.sqrt(n))) if (fr_std > 0 and n > 1) else float("nan")

        mass_gt_5R = float(np.mean(peak_mfe_X >= 5.0))
        frac_reach_1R = float(sub["reached_1R_X"].astype(int).mean())
        frac_reach_2R = float(np.mean(peak_mfe_X >= 2.0))
        frac_wrong_way_pp = float(sub["wrong_way_pre_peak_X"].astype(int).mean())
        mono_pp_centroid = float(sub["mono_pre_peak_X"].mean())

        # Capturability composite (v2.1.1 §7)
        composite = (
            (mono_pp_centroid - 0.55)
            + (frac_reach_1R - 0.70)
            + (0.30 - frac_wrong_way_pp)
        )

        rows.append(
            {
                "cluster_id": int(cid),
                "X": float(X),
                "n": n,
                "mono_pre_peak_centroid": mono_pp_centroid,
                "frac_reach_1R": frac_reach_1R,
                "frac_reach_2R": frac_reach_2R,
                "frac_wrong_way_pre_peak": frac_wrong_way_pp,
                "fwd_mfe_p5": float(p5),
                "fwd_mfe_p25": float(p25),
                "fwd_mfe_p50": float(p50),
                "fwd_mfe_p75": float(p75),
                "fwd_mfe_p95": float(p95),
                "fwd_mfe_p50_atr": float(np.percentile(peak_mfe_atr, 50)),
                "final_r_p5": float(fp5),
                "final_r_p25": float(fp25),
                "final_r_p50": float(fp50),
                "final_r_p75": float(fp75),
                "final_r_p95": float(fp95),
                "final_r_mean": fr_mean,
                "final_r_t_stat": fr_tstat,
                "mass_gt_5R": mass_gt_5R,
                "composite": composite,
            }
        )
    return pd.DataFrame(rows).sort_values(["cluster_id", "X"]).reset_index(drop=True)
