"""Compute the rejected-trade bail-out PnL distribution for cluster 1 under
the per-fold refit classifier (Phase 4 deliverable).

Rejected trades at t=1: classifier verdict below threshold → close at bar 2 open
with whatever bar-1 → bar-2 PnL realised (Pipeline D1 bail-out per §8).

Inputs:
- results/l_arc_4_rerun/step1/trades_all.csv
- results/l_arc_4_rerun/step1/trades_paths.csv (bar 2 open for each trade)
- results/l_arc_4_rerun/step2/clusters_K4.csv
- results/l_arc_4_rerun/step5c/per_trade_simulated_refit_1.csv (admitted set)
- results/l_arc_4_rerun/step5c/fold_definitions.csv

Outputs:
- results/l_arc_4_rerun/step5c/rejected_trade_pnl_cluster_1.csv (per rejected trade)
- printed summary
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RR = REPO / "results" / "l_arc_4_rerun"

trades = pd.read_csv(RR / "step1" / "trades_all.csv", parse_dates=["entry_time", "signal_time"])
clusters = pd.read_csv(RR / "step2" / "clusters_K4.csv")
admitted = pd.read_csv(RR / "step5c" / "per_trade_simulated_refit_1.csv", parse_dates=["entry_ts"])
folds = pd.read_csv(RR / "step5c" / "fold_definitions.csv")

# Filter to cluster 1 trades
trades = trades.merge(clusters, on="trade_id", how="left")
c1 = trades[trades["cluster_id"] == 1].copy()
print(f"cluster 1 total trades: {len(c1)}")

# Attach fold based on entry_time
def assign_fold(ts):
    for _, r in folds.iterrows():
        if pd.Timestamp(r["oos_start"]) <= ts <= pd.Timestamp(r["oos_end"]):
            return int(r["fold"])
    return -1

c1["fold"] = c1["entry_time"].apply(assign_fold)
c1 = c1[c1["fold"].between(2, 7)].copy()
print(f"cluster 1 F2-F7: {len(c1)}")

admitted_ids = set(admitted["trade_id"].tolist())
c1["admitted"] = c1["trade_id"].isin(admitted_ids)
print(f"admitted: {c1['admitted'].sum()}; rejected: {(~c1['admitted']).sum()}")

# For rejected trades, compute bail-out PnL = (bar2_open - entry_price - exit_spread/2) / cluster_R
# cluster_R = 3 * ATR for cluster 1 (post-t SL would have been 3*ATR; pre-t was 2*ATR)
# Bail-out happens at bar 2 open with bar-2 spread for the exit.
# Load paths to get bar 2 open + we'd need bar-2 spread per trade; spread floor file
# applies a min floor — use spread_pips_exit from trades_all as the most recent
# spread proxy for now (note: this is the cap-binding exit spread, not bar-2 spread).
# For an accurate bar-2 spread we'd need raw spread data per bar. Approximation: use
# the same floor as entry (matches median behaviour under floor binding).

rejected = c1[~c1["admitted"]].copy()
print(f"computing bail-out PnL for {len(rejected)} rejected trades...")

# Load bar 2 opens
paths = pd.read_csv(
    RR / "step1" / "trades_paths.csv",
    usecols=["trade_id", "bar_offset", "open"],
)
paths_b2 = paths[paths["bar_offset"] == 2][["trade_id", "open"]].rename(columns={"open": "bar2_open"})
rejected = rejected.merge(paths_b2, on="trade_id", how="left")

# cluster_R in price units = 3 * atr_14_at_signal
rejected["cluster_R_price"] = 3.0 * rejected["atr_14_at_signal"]

# Bail-out exit spread approximation: use entry spread (spread_pips_used) divided by 10 for pip→price
# For accurate bar-2 spread, we'd need the bar-level spread CSVs. Use entry spread as floor-bound proxy.
# Exit fill (long) = bar2_open - exit_spread/2 (per SPREAD_SEMANTICS_LOCK).
rejected["bar2_exit_spread_pips_approx"] = rejected["spread_pips_used"]  # proxy
# Pip size: most pairs 0.0001; JPY pairs 0.01. Detect from pair name.
def pip_size(pair):
    return 0.01 if "JPY" in pair else 0.0001

rejected["pip_size"] = rejected["pair"].apply(pip_size)
rejected["bar2_exit_price"] = rejected["bar2_open"] - 0.5 * rejected["bar2_exit_spread_pips_approx"] * rejected["pip_size"]
rejected["bailout_pnl_r"] = (rejected["bar2_exit_price"] - rejected["entry_price"]) / rejected["cluster_R_price"]

# Summary
print()
print("=== Rejected-trade bail-out PnL distribution (cluster 1, F2-F7) ===")
s = rejected["bailout_pnl_r"]
print(f"n: {len(s)}")
print(f"mean: {s.mean():.6f}")
print(f"median: {s.median():.6f}")
print(f"std: {s.std():.6f}")
print(f"p5: {s.quantile(0.05):.6f}")
print(f"p25: {s.quantile(0.25):.6f}")
print(f"p50: {s.quantile(0.50):.6f}")
print(f"p75: {s.quantile(0.75):.6f}")
print(f"p95: {s.quantile(0.95):.6f}")
print(f"min: {s.min():.6f}")
print(f"max: {s.max():.6f}")
print(f"frac negative: {(s < 0).mean():.4%}")
print(f"frac > 0: {(s > 0).mean():.4%}")
print(f"frac in [-0.05, +0.05]: {((s >= -0.05) & (s <= 0.05)).mean():.4%}")
print()
print("=== Per-fold rejected mean R ===")
per_fold = rejected.groupby("fold")["bailout_pnl_r"].agg(["count", "mean", "median", "std"])
print(per_fold.to_string())

# Save
out = rejected[["trade_id", "pair", "fold", "entry_time", "entry_price", "bar2_open",
                "bar2_exit_price", "cluster_R_price", "bar2_exit_spread_pips_approx",
                "bailout_pnl_r"]].copy()
out.to_csv(RR / "step5c" / "rejected_trade_pnl_cluster_1.csv", index=False)
print(f"\nwrote {RR / 'step5c' / 'rejected_trade_pnl_cluster_1.csv'}")
