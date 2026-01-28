import pandas as pd
import numpy as np

df = pd.read_csv("results/c1_batch_results.csv")

# Phase-1 foundation slice (entry-volume OFF, daily)
s = df[
    (df["timeframe"].astype(str) == "D") &
    (df["c2"].astype(str) == "none") &
    (df["baseline"].astype(str) == "none") &
    (df["volume"].astype(str) == "none")
].copy()

print("SLICE rows:", len(s))
print("SLICE pairs:", s["pair"].nunique())
print("SLICE exits:", sorted(s["exit"].astype(str).unique().tolist()))
print("SLICE c1s:", s["c1"].nunique())

# Pair-level profitability per (c1, exit, pair)
pair_flag = (
    s.assign(pos_pair = s["roi_pct"] > 0)
     .groupby(["c1","exit","pair"], as_index=False)["pos_pair"].max()
)

sys_pairs = pair_flag.groupby(["c1","exit"]).agg(
    pairs=("pair","nunique"),
    pos_roi_pairs=("pos_pair","sum"),
).reset_index()
sys_pairs["pos_roi_pair_rate"] = (sys_pairs["pos_roi_pairs"] / sys_pairs["pairs"]).round(3)

trades = s.groupby(["c1","exit"]).agg(
    total_trades=("total_trades","sum"),
    median_roi_pct=("roi_pct","median"),
    mean_roi_pct=("roi_pct","mean"),
).reset_index()

out = sys_pairs.merge(trades, on=["c1","exit"], how="inner")

# Filter dead systems
out = out[(out["pairs"] >= 20) & (out["total_trades"] >= 200)].copy()

out = out.sort_values(
    ["pos_roi_pair_rate","median_roi_pct","total_trades"],
    ascending=[False, False, False]
)

print("\nTOP 15 (Phase-1 foundation slice):")
print(out.head(15).to_string(index=False))

print("\nTOP 15 where exit == twiggs_money_flow:")
print(out[out["exit"].astype(str)=="twiggs_money_flow"].head(15).to_string(index=False))
