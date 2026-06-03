"""Path-shape features for KH-24 v2.0 Step 2 — outcome-blind, per §6.

Four features per trade, computed strictly from `trades_paths.csv`:

  monotonicity_ratio_in_profit
      Among rows with close_r > 0, the fraction where close_r >= the
      previous in-profit row's close_r. Zero in-profit rows -> 0.

  local_peaks_count
      Number of rows where mfe_so_far_r is strictly greater than the
      previous row's mfe_so_far_r (new-MFE bars). bars_held=0 -> 0.

  pullback_magnitude_median
      For each consecutive pair of new-peak bars, the depth of the
      between-peaks dip:
          earlier_peak_mfe - min(close_r over [earlier_peak_offset,
                                              later_peak_offset - 1])
      Take the median across all such pairs. Operational fix per
      §6/Open-08 (literal definition on mfe_so_far_r is identically 0
      since mfe is non-decreasing). < 2 peaks -> 0.

  time_to_peak_mfe_relative
      bar_offset of the maximum mfe_so_far_r divided by max(bars_held, 1),
      capped at 1.0. Trade never in profit (max mfe_so_far_r <= 0) -> 0.

The functions consume EITHER the path rows for a single trade (as a
DataFrame) OR a list of dicts; they only read `bar_offset`, `close_r`,
and `mfe_so_far_r`. They never read `final_r`, `mae_r`, `exit_reason`,
or any other forward-knowledge quantity — outcome-blind by construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS: tuple[str, ...] = (
    "monotonicity_ratio_in_profit",
    "local_peaks_count",
    "pullback_magnitude_median",
    "time_to_peak_mfe_relative",
)


def _monotonicity_ratio_in_profit(close_r: np.ndarray) -> float:
    in_profit = close_r > 0
    if not in_profit.any():
        return 0.0
    seq = close_r[in_profit]
    if len(seq) <= 1:
        # One in-profit bar can't violate monotonicity; treat as 1.0 (every
        # in-profit bar is >= itself's "previous"). Spec gives no special-case
        # for n==1, but the natural reading of "fraction >= previous" treats
        # the first as trivially satisfied. Use n>1 denominator to avoid
        # degenerate 0/0; for n==1 return 1.0.
        return 1.0
    diffs = np.diff(seq)
    n_non_decreasing = int((diffs >= 0).sum())
    return float(n_non_decreasing) / float(len(seq) - 1)


def _local_peaks_count(mfe_so_far_r: np.ndarray) -> int:
    if len(mfe_so_far_r) <= 1:
        return 0
    return int(np.sum(np.diff(mfe_so_far_r) > 0))


def _pullback_magnitude_median(
    bar_offset: np.ndarray, close_r: np.ndarray, mfe_so_far_r: np.ndarray
) -> float:
    """Median between-peaks dip, per §6/Open-08 operational definition.

    Peaks are bar offsets where mfe_so_far_r strictly increases vs the prior
    bar. For each consecutive pair (P_i, P_{i+1}) of new-peak offsets, the
    dip is the earlier peak's mfe minus the min of `close_r` over the
    inclusive range [P_i, P_{i+1} - 1] (i.e., from the earlier peak bar up
    to but not including the next peak bar).
    """
    if len(mfe_so_far_r) <= 1:
        return 0.0
    peak_mask = np.r_[False, np.diff(mfe_so_far_r) > 0]
    peak_idx = np.flatnonzero(peak_mask)
    if peak_idx.size < 2:
        return 0.0
    dips: list[float] = []
    for i in range(peak_idx.size - 1):
        p_lo = int(peak_idx[i])
        p_hi = int(peak_idx[i + 1])
        earlier_peak_mfe = float(mfe_so_far_r[p_lo])
        # Range [p_lo, p_hi) — inclusive of earlier peak, exclusive of next.
        # bar_offset values are 0..bars_held; we index by position not by
        # bar_offset value, so this assumes the path is already sorted by
        # bar_offset (the caller guarantees this).
        between_min = float(np.min(close_r[p_lo:p_hi]))
        dips.append(earlier_peak_mfe - between_min)
    return float(np.median(dips))


def _time_to_peak_mfe_relative(
    bar_offset: np.ndarray, mfe_so_far_r: np.ndarray, bars_held: int
) -> float:
    if len(mfe_so_far_r) == 0:
        return 0.0
    max_mfe = float(np.max(mfe_so_far_r))
    if max_mfe <= 0.0:
        # Trade never in profit per §6 edge case.
        return 0.0
    # If multiple bars share the max, take the EARLIEST (first time the peak
    # was hit), so the value is deterministic.
    peak_pos = int(np.argmax(mfe_so_far_r))
    peak_offset = int(bar_offset[peak_pos])
    denom = max(int(bars_held), 1)
    return float(min(peak_offset / denom, 1.0))


def compute_features_for_trade(path_df: pd.DataFrame, bars_held: int) -> dict[str, float]:
    """Compute all four §6 features for a single trade's path."""
    # Sort defensively; caller should already pass sorted-by-bar_offset.
    p = path_df.sort_values("bar_offset")
    bar_offset = p["bar_offset"].to_numpy(dtype=np.int64)
    close_r = p["close_r"].to_numpy(dtype=np.float64)
    mfe = p["mfe_so_far_r"].to_numpy(dtype=np.float64)
    return {
        "monotonicity_ratio_in_profit": _monotonicity_ratio_in_profit(close_r),
        "local_peaks_count": float(_local_peaks_count(mfe)),
        "pullback_magnitude_median": _pullback_magnitude_median(bar_offset, close_r, mfe),
        "time_to_peak_mfe_relative": _time_to_peak_mfe_relative(bar_offset, mfe, int(bars_held)),
    }


def compute_features_for_all_trades(
    paths_df: pd.DataFrame, trades_df: pd.DataFrame
) -> pd.DataFrame:
    """Compute the §6 path-shape features for every trade in `trades_df`.

    Both `paths_df` (with columns trade_id, bar_offset, close_r, mfe_so_far_r)
    and `trades_df` (with columns trade_id, bars_held) are required.

    Returns a DataFrame with columns ['trade_id'] + FEATURE_COLUMNS, ordered
    deterministically by trade_id (lex sort).
    """
    bars_held_by_id = dict(zip(trades_df["trade_id"], trades_df["bars_held"].astype(int)))
    # Stable: sort the input paths once; groupby preserves group order.
    sorted_paths = paths_df.sort_values(["trade_id", "bar_offset"])
    rows: list[dict] = []
    for trade_id, group in sorted_paths.groupby("trade_id", sort=True):
        feats = compute_features_for_trade(group, bars_held_by_id[trade_id])
        rows.append({"trade_id": trade_id, **feats})
    out = pd.DataFrame(rows)
    return out[["trade_id", *FEATURE_COLUMNS]].sort_values("trade_id").reset_index(drop=True)
