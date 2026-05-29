"""Quick data-range probe for the diagnostic."""
from pathlib import Path

import pandas as pd

CACHE_ROOT = Path("C:/Users/panap/Documents/Forex-Backtester/data/cache")
for tf in ("H4", "D1", "W1"):
    p = CACHE_ROOT / tf / "EURUSD.parquet"
    if not p.exists():
        print(f"{tf}: NOT FOUND at {p}")
        continue
    df = pd.read_parquet(p)
    print(f"{tf}: {df.index.min()} .. {df.index.max()}  n={len(df)}")
