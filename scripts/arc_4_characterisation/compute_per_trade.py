"""Arc 4 Characterisation — per-trade computations.

Pass 1 over trades_paths.csv:
  - For each trade_id:
    * Compute path-derived metrics (peak_mfe, max_mae, monotonicity, peaks, pullbacks, etc.)
    * Apply §11 row 2 policy on the full 240-bar window (R-frame = 2*ATR, matching paths)
    * Record outcome bin definition inputs

Output: results/arc_4_characterisation/per_trade_path_metrics.csv

R-frame note: paths use 2*ATR-normalized close_r/mfe/mae. §11 row 2 is applied in this frame
for slices A, B, D consistency. Slice C uses its own refit final_r (3*ATR cluster_R frame).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
P_PATHS = REPO / "results" / "l_arc_4" / "step1" / "trades_paths.csv"
P_TRADES = REPO / "results" / "l_arc_4" / "step1" / "trades_all.csv"
OUT = REPO / "results" / "arc_4_characterisation" / "per_trade_path_metrics.csv"

OUT.parent.mkdir(parents=True, exist_ok=True)

# §11 row 2: MFE-lock at 1R, trail 0.75R from new high; initial SL = -1R; time exit at bar H
SL_INITIAL = -1.0
MFE_LOCK_TRIGGER = 1.0
TRAIL_DISTANCE = 0.75
TIME_EXIT_BAR = 240  # cluster 1 horizon; consistent for all slices


def simulate_and_summarise(grp: pd.DataFrame) -> dict:
    """Apply §11 row 2 and compute path-derived metrics for one trade."""
    grp = grp.sort_values("bar_offset")
    bar_offset = grp["bar_offset"].to_numpy()
    close_r = grp["close_r"].to_numpy(dtype=np.float64)
    mfe = grp["mfe_so_far_r"].to_numpy(dtype=np.float64)
    mae = grp["mae_so_far_r"].to_numpy(dtype=np.float64)

    n = len(grp)
    max_bar = int(bar_offset.max())

    # Full-window peak / trough (for reference / "what we left on the table")
    cap = min(max_bar, TIME_EXIT_BAR)
    mask_full = bar_offset <= cap
    peak_mfe_r_full = float(mfe[mask_full].max())
    max_mae_r_full = float(mae[mask_full].min())

    # §11 row 2 simulation — also tracks alive-window peak / max during the held window
    stop = SL_INITIAL
    exited = False
    exit_bar = None
    exit_r = None
    exit_reason = None
    mfe_locked_bar = None  # bar at which peak first reached 1R while alive
    trail_triggered = False

    alive_peak_mfe = -np.inf
    alive_max_mae = np.inf
    bar_of_alive_peak_mfe = 0
    bar_of_alive_max_mae = 0

    for i in range(n):
        b = int(bar_offset[i])
        if b > TIME_EXIT_BAR:
            break
        cur_mfe = mfe[i]
        cur_mae = mae[i]
        cur_close = close_r[i]
        # Check stop FIRST (conservative within-bar ordering: stop hit takes priority)
        if cur_mae <= stop + 1e-12:
            exited = True
            exit_bar = b
            exit_r = stop
            if stop <= SL_INITIAL + 1e-9:
                exit_reason = "sl_initial"
            elif stop <= 0.0 + 1e-9:
                exit_reason = "trail_breakeven"
            else:
                exit_reason = "trail"
            # Alive window peak/MAE — update with this bar's prior-stop-aware values
            # The MFE/MAE so-far at this bar reflects what was reached during this bar; under
            # conservative ordering we treat the stop as firing before any MFE update.
            # We still record alive_peak_mfe with mfe[i-1] (last bar where stop did not fire).
            if i > 0:
                if mfe[i - 1] > alive_peak_mfe:
                    alive_peak_mfe = float(mfe[i - 1])
                    bar_of_alive_peak_mfe = int(bar_offset[i - 1])
                if mae[i - 1] < alive_max_mae:
                    alive_max_mae = float(mae[i - 1])
                    bar_of_alive_max_mae = int(bar_offset[i - 1])
            # The stop fires at value `stop`; treat that as the realised max_mae
            if stop < alive_max_mae:
                alive_max_mae = float(stop)
                bar_of_alive_max_mae = b
            break

        # No stop hit this bar — update alive window stats with current bar's running mfe/mae
        if cur_mfe > alive_peak_mfe:
            alive_peak_mfe = float(cur_mfe)
            bar_of_alive_peak_mfe = b
        if cur_mae < alive_max_mae:
            alive_max_mae = float(cur_mae)
            bar_of_alive_max_mae = b

        # Detect mfe-lock event (only counts if we got here without exiting first)
        if mfe_locked_bar is None and cur_mfe >= MFE_LOCK_TRIGGER:
            mfe_locked_bar = b
        # Update trailing stop based on current bar's mfe
        if cur_mfe >= MFE_LOCK_TRIGGER:
            new_stop = max(0.0, cur_mfe - TRAIL_DISTANCE)
            if new_stop > stop:
                stop = new_stop
                trail_triggered = True
        if b == TIME_EXIT_BAR:
            # Time exit at this bar's close
            exited = True
            exit_bar = b
            exit_r = float(cur_close)
            exit_reason = "time"
            break

    if not exited:
        # Path ended before TIME_EXIT_BAR — use last bar close
        last_idx = n - 1
        exit_bar = int(bar_offset[last_idx])
        exit_r = float(close_r[last_idx])
        exit_reason = "time"
        # alive_peak/max already captured during loop

    # Use alive-window canonical
    peak_mfe_r = float(alive_peak_mfe) if np.isfinite(alive_peak_mfe) else float(mfe[0])
    max_mae_r = float(alive_max_mae) if np.isfinite(alive_max_mae) else float(mae[0])
    bar_of_peak_mfe = int(bar_of_alive_peak_mfe)
    bar_of_max_mae = int(bar_of_alive_max_mae)

    # Path-shape metrics (computed up to exit_bar inclusive)
    mask_alive = bar_offset <= exit_bar
    close_alive = close_r[mask_alive]
    n_alive = len(close_alive)

    # monotonicity_ratio_in_profit: among bars where close_r>0, frac where close_r[i] >= close_r[i-1]
    # (i.e., upward steps when we're in profit)
    if n_alive >= 2:
        diffs = np.diff(close_alive)
        in_profit = close_alive[1:] > 0
        mono_in_profit = float(np.sum((diffs >= 0) & in_profit) / max(np.sum(in_profit), 1)) if np.sum(in_profit) > 0 else np.nan
    else:
        mono_in_profit = np.nan

    # monotonicity_ratio_pre_peak: bars up to bar_of_peak_mfe, frac where close_r[i] >= close_r[i-1]
    pre_peak_mask = bar_offset <= bar_of_peak_mfe
    close_pre = close_r[pre_peak_mask]
    if len(close_pre) >= 2:
        diffs_pre = np.diff(close_pre)
        mono_pre_peak = float(np.sum(diffs_pre >= 0) / len(diffs_pre))
    else:
        mono_pre_peak = np.nan

    # local_peaks_count: number of local maxima in close_r (alive)
    if n_alive >= 3:
        local_peaks = int(np.sum((close_alive[1:-1] > close_alive[:-2]) & (close_alive[1:-1] > close_alive[2:])))
    else:
        local_peaks = 0

    # pullback_magnitude_median: median drop from each local peak to next trough
    if n_alive >= 3:
        pullbacks = []
        peak_val = close_alive[0]
        in_pullback = False
        trough_val = peak_val
        for j in range(1, n_alive):
            if close_alive[j] >= peak_val:
                if in_pullback:
                    pullbacks.append(peak_val - trough_val)
                    in_pullback = False
                peak_val = close_alive[j]
                trough_val = close_alive[j]
            else:
                if close_alive[j] < trough_val:
                    trough_val = close_alive[j]
                in_pullback = True
        if in_pullback:
            pullbacks.append(peak_val - trough_val)
        pullback_magnitude_median = float(np.median(pullbacks)) if pullbacks else 0.0
    else:
        pullback_magnitude_median = 0.0

    # time_to_peak_mfe_relative
    if exit_bar and exit_bar > 0:
        time_to_peak_rel = bar_of_peak_mfe / exit_bar
    else:
        time_to_peak_rel = np.nan

    # velocity_first_t (close at bar 1, in R)
    bar1_mask = bar_offset == 1
    if bar1_mask.any():
        idx1 = int(np.argmax(bar1_mask))
        close_r_at_t1 = float(close_r[idx1])
        mfe_at_t1 = float(mfe[idx1])
        mae_at_t1 = float(mae[idx1])
        # velocity = close at bar 1 - close at bar 0 (using close_r values)
        bar0_mask = bar_offset == 0
        close_r_at_t0 = float(close_r[int(np.argmax(bar0_mask))]) if bar0_mask.any() else 0.0
        velocity_first_t = close_r_at_t1 - close_r_at_t0
    else:
        close_r_at_t1 = np.nan
        mfe_at_t1 = np.nan
        mae_at_t1 = np.nan
        velocity_first_t = np.nan

    # mae_before_peak_mfe_r
    mask_pre_peak = bar_offset <= bar_of_peak_mfe
    if mask_pre_peak.any():
        mae_before_peak_mfe_r = float(mae[mask_pre_peak].min())
    else:
        mae_before_peak_mfe_r = np.nan

    # mfe_after_max_mae_r
    mask_post_mae = bar_offset >= bar_of_max_mae
    if mask_post_mae.any():
        mfe_after_max_mae_r = float(mfe[mask_post_mae].max())
    else:
        mfe_after_max_mae_r = np.nan

    final_r = float(exit_r)
    giveback_r = peak_mfe_r - final_r
    ever_reached_1R_alive = peak_mfe_r >= 1.0
    hold_duration_bars = int(exit_bar)

    return {
        "final_r_s11r2": final_r,
        "exit_bar_s11r2": int(exit_bar),
        "exit_reason_s11r2": exit_reason,
        "peak_mfe_r": peak_mfe_r,
        "max_mae_r": max_mae_r,
        "peak_mfe_r_full_window": peak_mfe_r_full,
        "max_mae_r_full_window": max_mae_r_full,
        "bar_of_peak_mfe": bar_of_peak_mfe,
        "bar_of_max_mae": bar_of_max_mae,
        "giveback_r": giveback_r,
        "mfe_locked_bar": mfe_locked_bar if mfe_locked_bar is not None else -1,
        "trail_triggered": int(trail_triggered),
        "ever_reached_1R_alive": int(ever_reached_1R_alive),
        "monotonicity_ratio_in_profit": mono_in_profit,
        "monotonicity_ratio_pre_peak": mono_pre_peak,
        "local_peaks_count": local_peaks,
        "pullback_magnitude_median": pullback_magnitude_median,
        "time_to_peak_mfe_relative": time_to_peak_rel,
        "velocity_first_t": velocity_first_t,
        "close_r_at_t1": close_r_at_t1,
        "mae_so_far_r_at_t1": mae_at_t1,
        "mfe_so_far_r_at_t1": mfe_at_t1,
        "mae_before_peak_mfe_r": mae_before_peak_mfe_r,
        "mfe_after_max_mae_r": mfe_after_max_mae_r,
        "hold_duration_bars": hold_duration_bars,
    }


def main():
    print(f"[info] reading {P_PATHS}")
    # Read in chunks by trade_id; pandas can do all at once given size
    paths = pd.read_csv(P_PATHS)
    print(f"[info] paths rows: {len(paths):,}")
    print(f"[info] unique trade_ids: {paths['trade_id'].nunique():,}")

    print("[info] computing per-trade metrics...")
    out_rows = []
    total = paths["trade_id"].nunique()
    seen = 0
    for tid, grp in paths.groupby("trade_id", sort=True):
        rec = simulate_and_summarise(grp)
        rec["trade_id"] = int(tid)
        out_rows.append(rec)
        seen += 1
        if seen % 1000 == 0:
            print(f"  ...{seen}/{total}")

    df = pd.DataFrame(out_rows).set_index("trade_id").sort_index()
    df.to_csv(OUT)
    print(f"[done] wrote {OUT} ({len(df)} rows, {len(df.columns)} cols)")


if __name__ == "__main__":
    main()
