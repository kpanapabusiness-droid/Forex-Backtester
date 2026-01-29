import pandas as pd
import numpy as np

df = pd.read_csv("results/c1_batch_results.csv")

# pair-level win flag (roi_pct > 0) for each (c1, exit, pair)
pair_flag = (
    df.assign(pos_pair = df["roi_pct"] > 0)
      .groupby(["c1","exit","pair"], as_index=False)["pos_pair"].max()
)

# system-level stats
sys_pairs = pair_flag.groupby(["c1","exit"]).agg(
    pairs=("pair","nunique"),
    pos_roi_pairs=("pos_pair","sum"),
).reset_index()

sys_pairs["pos_roi_pair_rate"] = (sys_pairs["pos_roi_pairs"] / sys_pairs["pairs"]).round(3)

trades = df.groupby(["c1","exit"]).agg(
    total_trades=("total_trades","sum"),
    median_roi_pct=("roi_pct","median"),
    mean_roi_pct=("roi_pct","mean"),
).reset_index()

out = sys_pairs.merge(trades, on=["c1","exit"], how="inner")

# show the top systems
out = out.sort_values(
    ["pos_roi_pair_rate","median_roi_pct","total_trades"],
    ascending=[False, False, False]
)

print(out.head(20).to_string(index=False))
