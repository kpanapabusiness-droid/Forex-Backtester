"""Build the EET Phase-2 parity ledger from the locked v3.0.2 EET pool.

The UTC harness consumes a ``trade_ledger_utc.parquet`` with at least
``pair / signal_bar_time / entry_time``. The EET arc never emitted a
standalone ledger artefact, but the locked Step-1 pool
(``results/l_arc_10_v3.0.2/step_1/pool.parquet``) already carries the
per-trade ``entry_time`` alongside ``signal_bar_time`` — the exact two
columns the parity timing check needs. So the EET ledger is a faithful
projection of the pool, not a recomputation; the locked artefact stays
read-only.

Output: ``results/phase_2_parity_eet/trade_ledger_eet.parquet``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve()
REPO_ROOT = _HERE.parent.parent.parent

POOL_PATH = REPO_ROOT / "results" / "l_arc_10_v3.0.2" / "step_1" / "pool.parquet"
OUT_DIR = REPO_ROOT / "results" / "phase_2_parity_eet"
OUT_PATH = OUT_DIR / "trade_ledger_eet.parquet"


def build() -> pd.DataFrame:
    pool = pd.read_parquet(POOL_PATH)
    missing = {"pair", "signal_bar_time", "entry_time"} - set(pool.columns)
    if missing:
        raise SystemExit(f"pool is missing required columns: {sorted(missing)}")
    ledger = (
        pool[["pair", "signal_bar_time", "entry_time"]]
        .copy()
        .sort_values(["pair", "signal_bar_time"])
        .reset_index(drop=True)
    )
    ledger["signal_bar_time"] = pd.to_datetime(ledger["signal_bar_time"], utc=True)
    ledger["entry_time"] = pd.to_datetime(ledger["entry_time"], utc=True)
    return ledger


def main() -> int:
    ledger = build()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ledger.to_parquet(OUT_PATH, engine="pyarrow", compression="snappy", index=False)
    print(f"wrote {len(ledger)} rows x {ledger['pair'].nunique()} pairs -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
