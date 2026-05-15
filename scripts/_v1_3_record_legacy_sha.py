"""Helper: compute the slice-and-reserialise sha256 of the legacy column
subset of results/kh24/trades_all.csv. Used to populate BASELINE_SHA for
the v1.3 regression test (tests/test_kh24_trades_all_regression.py).

Run on the PRE-change file (before the v1.3 backtester extension lands)
to record the baseline. The regression test then runs the same slice-
and-hash on the POST-change file and asserts equality.

Why slice-and-hash instead of full-file hash: the post-extension file
gains 3 new columns; a full-file hash will obviously change. The legacy
subset is the part that must remain byte-identical, so we hash only
that subset (re-serialised through pandas the same way both pre and
post). Both runs go through pd.read_csv -> select -> to_csv(index=False)
-> sha256, so the comparison is fair.
"""
import hashlib
import sys
from pathlib import Path

import pandas as pd

LEGACY_COLUMNS = [
    "pair", "entry_date", "exit_date", "entry_price", "exit_price", "sl_price",
    "trail_active", "exit_reason", "classification", "bars_held", "net_pnl",
    "r_multiple", "spread_pips_used", "sl_distance_atr", "d1_dist_ratio",
    "d1_close_in_range", "h1_last_bar_close_in_range",
    "mae_final", "mfe_final",
    "mae_at_bar_3", "mfe_at_bar_3", "mae_at_bar_6", "mfe_at_bar_6",
    "first_bar_dir",
    "kh13_mae_at_check", "kh13_mfe_at_check", "kh13_triggered",
    "kh14_triggered", "kh14_state2",
    "atr_sized_down", "trade_type", "original_entry_price_ref",
    "signal_spread_pips", "spread_ratio", "atr_abs", "atr_ratio",
    "d1_kijun_slope", "session", "concurrent_signals",
]

TRADES_PATH = Path("results/kh24/trades_all.csv")


def main() -> int:
    df = pd.read_csv(TRADES_PATH)
    n_cols = df.shape[1]
    n_rows = len(df)
    cols = df.columns.tolist()
    missing = [c for c in LEGACY_COLUMNS if c not in cols]
    extra = [c for c in cols if c not in LEGACY_COLUMNS]
    print(f"File:    {TRADES_PATH}")
    print(f"Rows:    {n_rows}")
    print(f"Cols:    {n_cols}")
    print(f"Missing legacy columns: {missing}")
    print(f"Extra columns (not legacy): {extra}")
    slc = df[LEGACY_COLUMNS]
    sha = hashlib.sha256(slc.to_csv(index=False).encode()).hexdigest()
    print(f"\nLegacy-subset sha256 (slice-and-reserialise): {sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
