"""Arc 4 Characterisation — per-bar (trade, bar_offset) summary.

Produces a tall table keyed (trade_id, bar_offset) with close_r, mfe_so_far_r, mae_so_far_r
for bars 0..20 only. This is the input for bar_trajectory_mae/mfe.csv aggregations.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
P_PATHS = REPO / "results" / "l_arc_4" / "step1" / "trades_paths.csv"
OUT = REPO / "results" / "arc_4_characterisation" / "per_trade_bar_0_20.csv"

OUT.parent.mkdir(parents=True, exist_ok=True)


def main():
    print(f"[info] reading {P_PATHS}")
    # Only need bars 0..20 — read columns we need
    use_cols = ["trade_id", "bar_offset", "close_r", "mfe_so_far_r", "mae_so_far_r"]
    df = pd.read_csv(P_PATHS, usecols=use_cols)
    df = df[df["bar_offset"] <= 20].copy()
    df = df.sort_values(["trade_id", "bar_offset"])
    print(f"[info] rows: {len(df):,}")
    df.to_csv(OUT, index=False)
    print(f"[done] wrote {OUT}")


if __name__ == "__main__":
    main()
