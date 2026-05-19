"""Shared vectorized §11 row 2-style policy simulator.

All simulation runs on dense numpy matrices indexed [trade_idx, bar_offset].
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

EXIT_CODE = {
    "sl_initial": 0,
    "trail_breakeven": 1,
    "trail": 2,
    "time": 3,
}
EXIT_CODE_TO_LABEL = {v: k for k, v in EXIT_CODE.items()}


def simulate_policy_vectorized(
    *,
    close_r: np.ndarray,
    mfe_so_far_r: np.ndarray,
    mae_so_far_r: np.ndarray,
    initial_sl: float = -1.0,
    trail_trigger: float = 1.0,
    trail_width: float = 0.75,
    time_exit_bar: int = 240,
    start_bar: int = 0,
    entry_close_r: np.ndarray | None = None,
):
    """Vectorised policy simulation over [N_trades, N_bars] matrices.

    If start_bar > 0, simulate from bar `start_bar` instead of bar 0. R-reference is
    `entry_close_r` (per-trade close_r at the new entry bar); the new close/mfe/mae
    are computed by subtracting `entry_close_r` from the original arrays from
    bar `start_bar` onwards. (Re-derived mfe/mae are running max/min of the new
    closes — NOTE: intrabar excursion vs original entry is NOT preserved here when
    start_bar > 0. For full intrabar high/low semantics, pass arrays already
    recomputed relative to the new entry.)

    Returns dict with arrays of length N_trades:
        final_r, exit_bar, exit_reason_code, peak_mfe_alive, max_mae_alive,
        trail_triggered (bool), valid (bool — False if path was too short to even start)
    """
    n_trades, n_bars_total = close_r.shape
    eff_close = close_r
    eff_mfe = mfe_so_far_r
    eff_mae = mae_so_far_r

    if start_bar > 0:
        if entry_close_r is None:
            raise ValueError("entry_close_r required when start_bar > 0")
        # Shift origin: new close_r' = close_r - entry_close_r
        # Re-derive MFE/MAE as running max/min of (close_r' from start_bar)
        # over the bar range [start_bar..end]
        new_close = close_r - entry_close_r[:, None]
        # Mask invalid bars (NaN) and pre-start bars
        n_after = n_bars_total - start_bar
        slice_close = new_close[:, start_bar:]
        eff_close = np.full_like(close_r, np.nan)
        eff_close[:, start_bar:] = slice_close
        # Running max/min from start_bar
        # Replace NaN with -inf / +inf for accumulate, then mask back
        running_max = np.full_like(slice_close, np.nan)
        running_min = np.full_like(slice_close, np.nan)
        running_max[:, 0] = slice_close[:, 0]
        running_min[:, 0] = slice_close[:, 0]
        for b in range(1, slice_close.shape[1]):
            # np.fmax handles NaN: NaN is treated as missing, propagates the other value
            running_max[:, b] = np.fmax(running_max[:, b - 1], slice_close[:, b])
            running_min[:, b] = np.fmin(running_min[:, b - 1], slice_close[:, b])
        eff_mfe = np.full_like(close_r, np.nan)
        eff_mae = np.full_like(close_r, np.nan)
        eff_mfe[:, start_bar:] = running_max
        eff_mae[:, start_bar:] = running_min

    stop = np.full(n_trades, initial_sl, dtype=np.float64)
    exited = np.zeros(n_trades, dtype=bool)
    valid_trade = ~np.isnan(eff_close[:, start_bar])  # need at least the start bar
    exit_bar = np.full(n_trades, -1, dtype=np.int32)
    exit_r = np.full(n_trades, np.nan, dtype=np.float64)
    exit_reason_code = np.full(n_trades, -1, dtype=np.int8)
    peak_mfe_alive = np.full(n_trades, -np.inf, dtype=np.float64)
    max_mae_alive = np.full(n_trades, np.inf, dtype=np.float64)
    trail_triggered = np.zeros(n_trades, dtype=bool)
    prev_mfe = np.full(n_trades, -np.inf, dtype=np.float64)
    prev_mae = np.full(n_trades, np.inf, dtype=np.float64)

    for b in range(start_bar, time_exit_bar + 1):
        cur_mfe = eff_mfe[:, b]
        cur_mae = eff_mae[:, b]
        cur_close = eff_close[:, b]

        bar_valid = ~np.isnan(cur_close) & valid_trade
        active = ~exited & bar_valid

        # Stop check first (conservative within-bar ordering)
        hit_mask = active & (cur_mae <= stop + 1e-12)

        # For trades that hit: record exit
        # alive peak/max should be from prior bar (before this stop fired)
        peak_for_hit = np.where(np.isfinite(prev_mfe), prev_mfe, cur_mfe)
        mae_for_hit = np.where(np.isfinite(prev_mae), prev_mae, cur_mae)
        # The stop fired at level `stop`; treat that as realised final_r and max_mae
        peak_mfe_alive[hit_mask] = peak_for_hit[hit_mask]
        max_mae_alive[hit_mask] = np.minimum(mae_for_hit[hit_mask], stop[hit_mask])
        exit_bar[hit_mask] = b
        exit_r[hit_mask] = stop[hit_mask]
        # Reason
        is_initial = hit_mask & (stop <= initial_sl + 1e-9)
        is_be = hit_mask & (stop > initial_sl + 1e-9) & (stop <= 0.0 + 1e-9)
        is_trail = hit_mask & (stop > 0.0 + 1e-9)
        exit_reason_code[is_initial] = EXIT_CODE["sl_initial"]
        exit_reason_code[is_be] = EXIT_CODE["trail_breakeven"]
        exit_reason_code[is_trail] = EXIT_CODE["trail"]
        exited |= hit_mask

        # For still-active trades: update alive peak/max, then update trail stop
        still = ~exited & bar_valid
        peak_mfe_alive[still] = np.maximum(peak_mfe_alive[still], cur_mfe[still])
        max_mae_alive[still] = np.minimum(max_mae_alive[still], cur_mae[still])
        prev_mfe[still] = cur_mfe[still]
        prev_mae[still] = cur_mae[still]
        trigger_mask = still & (cur_mfe >= trail_trigger)
        new_stop = np.maximum(0.0, cur_mfe - trail_width)
        update_mask = trigger_mask & (new_stop > stop)
        stop = np.where(update_mask, new_stop, stop)
        trail_triggered |= update_mask

        # Time exit at last bar
        if b == time_exit_bar:
            time_mask = ~exited & bar_valid
            exit_bar[time_mask] = b
            exit_r[time_mask] = cur_close[time_mask]
            exit_reason_code[time_mask] = EXIT_CODE["time"]
            exited |= time_mask

    # For any trade that never exited (e.g., short path), record at last valid bar
    never_exit = ~exited & valid_trade
    if never_exit.any():
        # Find last valid bar per trade
        last_valid = np.full(n_trades, -1, dtype=np.int32)
        for tidx in np.where(never_exit)[0]:
            mask = ~np.isnan(eff_close[tidx, :])
            valid_bars = np.where(mask)[0]
            if len(valid_bars):
                last_valid[tidx] = valid_bars[-1]
        for tidx in np.where(never_exit)[0]:
            lb = last_valid[tidx]
            if lb >= 0:
                exit_bar[tidx] = lb
                exit_r[tidx] = eff_close[tidx, lb]
                exit_reason_code[tidx] = EXIT_CODE["time"]
                exited[tidx] = True

    # Clean up unfinite peak/mae for invalid trades
    peak_mfe_alive[~exited] = np.nan
    max_mae_alive[~exited] = np.nan

    return {
        "final_r": exit_r,
        "exit_bar": exit_bar,
        "exit_reason_code": exit_reason_code,
        "peak_mfe_alive": peak_mfe_alive,
        "max_mae_alive": max_mae_alive,
        "trail_triggered": trail_triggered,
        "valid": valid_trade,
    }


def load_path_arrays(npz_path: Path):
    d = np.load(npz_path)
    return {k: d[k] for k in d.files}
