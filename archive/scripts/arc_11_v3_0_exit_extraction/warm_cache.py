"""Warm the 5ers_eet aggregation cache for all 28 pairs x {H4,D1,W1}.

Cold-builds the EET parquet caches under data/cache/<TF>_5ers_eet/<PAIR>.parquet
so the Arc 11 v3.0 exit-extraction driver runs warm. Idempotent — aggregate()
hits the parquet cache after first build. Serial for deterministic progress logging.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.data.aggregator import aggregate  # noqa: E402

PAIRS = (
    "AUDCAD AUDCHF AUDJPY AUDNZD AUDUSD CADCHF CADJPY CHFJPY "
    "EURAUD EURCAD EURCHF EURGBP EURJPY EURNZD EURUSD GBPAUD "
    "GBPCAD GBPCHF GBPJPY GBPNZD GBPUSD NZDCAD NZDCHF NZDJPY "
    "NZDUSD USDCAD USDCHF USDJPY"
).split()
HISTDATA = r"C:\Users\panap\Documents\Forex-Backtester\data\histdata"
CACHE = r"C:\Users\panap\Documents\Forex-Backtester\data\cache"


def main() -> int:
    t0 = time.time()
    for tf in ("H4", "D1", "W1"):
        for i, p in enumerate(PAIRS, 1):
            ts = time.time()
            df = aggregate(p, tf, histdata_root=HISTDATA, cache_root=CACHE,
                           boundary_convention="5ers_eet")
            print(f"[warm] {tf} {p} ({i}/28) rows={len(df)} {time.time()-ts:.1f}s "
                  f"(elapsed {(time.time()-t0)/60:.1f}m)", flush=True)
    print(f"[warm] DONE all panels in {(time.time()-t0)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
