from pathlib import Path
import json

ROOT = Path("C:/Users/panap/Documents/Forex-Backtester/data/cache")
for tf in ("H4", "D1", "W1"):
    p = ROOT / tf / "EURUSD.parquet.meta.json"
    if p.exists():
        d = json.loads(p.read_text())
        print(f"=== {tf} EURUSD ===")
        for k, v in d.items():
            print(f"  {k}: {v}")
