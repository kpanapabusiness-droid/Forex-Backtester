"""Generate the synthetic OHLCV CSV used by the WFO-v2 generic-regression harness.

Deterministic, no RNG. Produces a daily-cadence sinusoidal close so c1_coral flips
several times across the IS+OOS window, exercising the legacy NNFX entry/exit path
end-to-end without leaking randomness into the regression manifest.

Run once to (re)generate the fixture; no other code path imports it.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

OUT_PATH = Path(__file__).parent / "wfo_v2_synthetic_harness_data" / "EUR_USD.csv"
START = "2020-01-01"
N_DAYS = 130  # 2020-01-01 .. 2020-05-09 inclusive (covers 4-month IS+OOS window with margin)
BASE = 1.10
AMP = 0.020
PERIOD_DAYS = 30  # ~1 cycle/month → multiple c1 flips inside IS and OOS windows
HALF_RANGE = 0.0015  # daily H-L half-range


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range(START, periods=N_DAYS, freq="D")
    rows = []
    for i, d in enumerate(dates):
        close = BASE + AMP * math.sin(2.0 * math.pi * i / PERIOD_DAYS)
        prev_close = BASE + AMP * math.sin(2.0 * math.pi * (i - 1) / PERIOD_DAYS) if i > 0 else close
        open_ = prev_close
        high = max(open_, close) + HALF_RANGE
        low = min(open_, close) - HALF_RANGE
        rows.append(
            {
                "date": d.strftime("%Y-%m-%d"),
                "open": round(open_, 6),
                "high": round(high, 6),
                "low": round(low, 6),
                "close": round(close, 6),
                "volume": 1000,
            }
        )
    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
    df.to_csv(OUT_PATH, index=False, lineterminator="\n")
    print(f"Wrote {OUT_PATH} ({len(df)} rows)")


if __name__ == "__main__":
    main()
