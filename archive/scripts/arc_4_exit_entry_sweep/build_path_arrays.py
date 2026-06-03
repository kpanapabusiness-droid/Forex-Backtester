"""Build dense numpy arrays from trades_paths.csv for fast policy re-simulation.

For each trade_id (sorted), build rows in matrices indexed [trade_idx, bar]:
  - close_r[bar]            — R-normalised close (2*ATR frame, original entry reference)
  - mfe_so_far_r[bar]       — running max R (intrabar high included)
  - mae_so_far_r[bar]       — running min R (intrabar low included)
  - high_r[bar]             — intrabar high in R (= close_r + delta to high)
  - low_r[bar]              — intrabar low in R

Padded with NaN beyond each trade's available bars.

Output: results/arc_4_exit_entry_sweep/path_arrays.npz
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
P_PATHS = REPO / "results" / "l_arc_4" / "step1" / "trades_paths.csv"
P_TRADES = REPO / "results" / "l_arc_4" / "step1" / "trades_all.csv"
OUT = REPO / "results" / "arc_4_exit_entry_sweep" / "path_arrays.npz"
OUT.parent.mkdir(parents=True, exist_ok=True)

MAX_BAR = 240

def main():
    print(f"[info] reading {P_TRADES}")
    trades = pd.read_csv(P_TRADES).sort_values("trade_id").reset_index(drop=True)
    trade_ids = trades["trade_id"].to_numpy()
    n_trades = len(trade_ids)
    print(f"[info] {n_trades} trades")

    print(f"[info] reading {P_PATHS}")
    paths = pd.read_csv(P_PATHS, usecols=["trade_id", "bar_offset", "open", "high", "low", "close", "close_r", "mfe_so_far_r", "mae_so_far_r"])
    paths = paths[paths["bar_offset"] <= MAX_BAR]
    print(f"[info] path rows: {len(paths):,}")

    # Merge in ATR + entry_price (for delayed entry recompute)
    paths = paths.merge(trades[["trade_id", "entry_price", "atr_14_at_signal"]], on="trade_id", how="left")
    # R unit (2*ATR) per row
    paths["R"] = 2.0 * paths["atr_14_at_signal"]
    paths["high_r"] = (paths["high"] - paths["entry_price"]) / paths["R"]
    paths["low_r"] = (paths["low"] - paths["entry_price"]) / paths["R"]
    paths["open_r"] = (paths["open"] - paths["entry_price"]) / paths["R"]

    # Pivot: index trade_id (rows in sorted order), columns bar_offset 0..240
    print("[info] pivoting to dense matrix...")
    tid_to_idx = {tid: i for i, tid in enumerate(trade_ids)}
    nbars = MAX_BAR + 1

    close_r_arr = np.full((n_trades, nbars), np.nan, dtype=np.float64)
    mfe_arr = np.full((n_trades, nbars), np.nan, dtype=np.float64)
    mae_arr = np.full((n_trades, nbars), np.nan, dtype=np.float64)
    high_r_arr = np.full((n_trades, nbars), np.nan, dtype=np.float64)
    low_r_arr = np.full((n_trades, nbars), np.nan, dtype=np.float64)
    open_r_arr = np.full((n_trades, nbars), np.nan, dtype=np.float64)

    idx = paths["trade_id"].map(tid_to_idx).to_numpy()
    bar = paths["bar_offset"].to_numpy()
    close_r_arr[idx, bar] = paths["close_r"].to_numpy()
    mfe_arr[idx, bar] = paths["mfe_so_far_r"].to_numpy()
    mae_arr[idx, bar] = paths["mae_so_far_r"].to_numpy()
    high_r_arr[idx, bar] = paths["high_r"].to_numpy()
    low_r_arr[idx, bar] = paths["low_r"].to_numpy()
    open_r_arr[idx, bar] = paths["open_r"].to_numpy()

    # Path length per trade: count non-NaN closes
    path_len = np.sum(~np.isnan(close_r_arr), axis=1)
    print(f"[info] median path length: {int(np.median(path_len))}, min {int(path_len.min())}, max {int(path_len.max())}")

    print(f"[info] saving {OUT}")
    np.savez_compressed(
        OUT,
        trade_ids=trade_ids,
        close_r=close_r_arr,
        mfe_so_far_r=mfe_arr,
        mae_so_far_r=mae_arr,
        high_r=high_r_arr,
        low_r=low_r_arr,
        open_r=open_r_arr,
        path_len=path_len,
    )
    print("[done]")


if __name__ == "__main__":
    main()
