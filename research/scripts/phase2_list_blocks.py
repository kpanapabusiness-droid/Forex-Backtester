import pandas as pd

df = pd.read_csv("results/c1_batch_results.csv")

df["from_date"] = df["from_date"].astype(str)
df["to_date"] = df["to_date"].astype(str)
df["baseline"] = df["baseline"].astype(str)
df["exit"] = df["exit"].astype(str)

blocks = (
    df.groupby(["baseline","exit","from_date","to_date"])
      .agg(rows=("run_id","size"), pairs=("pair","nunique"))
      .reset_index()
      .sort_values(["exit","baseline","from_date","to_date"])
)

print(blocks.to_string(index=False))
