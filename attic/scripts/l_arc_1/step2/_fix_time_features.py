# ruff: noqa: E402  (sys.path.insert needed before project imports)
"""One-shot fix: patches signals_features.csv to repair time-derived columns.

The Phase A bug interpreted us-encoded int64 timestamps as ns when calling
`pd.Timestamp(int_value)`. This affected signal_bar_ts / entry_bar_ts /
exit_bar_ts ISO strings AND every time-derived column (hour_utc,
day_of_week, session, hour_in_4h_bar, bars_to_next_4h_close, hour_in_d1_bar,
bars_to_next_d1_close).

This script re-derives those columns from trades_verbatim.csv (which has the
correct timestamps) and rewrites signals_features.csv in place.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.l_arc_1.step2._io import STEP2_DIR, load_trades_verbatim
from scripts.l_arc_1.step2.phase_a_features import _session_of_hour


def main() -> None:
    feat_path = STEP2_DIR / "signals_features.csv"
    print(f"Loading {feat_path}...")
    f = pd.read_csv(feat_path)
    print(f"  features rows: {len(f):,}")

    trades = load_trades_verbatim()
    print(f"  trades rows:   {len(trades):,}")

    # Build trade_id -> correct timestamp maps
    sig_map = dict(zip(trades["trade_id"].astype(int), pd.to_datetime(trades["signal_bar_ts"])))
    ent_map = dict(zip(trades["trade_id"].astype(int), pd.to_datetime(trades["entry_bar_ts"])))
    ext_map = dict(zip(trades["trade_id"].astype(int), pd.to_datetime(trades["exit_bar_ts"])))

    f["signal_bar_ts"] = (
        f["trade_id"].astype(int).map(sig_map).apply(lambda x: x.isoformat() if pd.notna(x) else "")
    )
    f["entry_bar_ts"] = (
        f["trade_id"].astype(int).map(ent_map).apply(lambda x: x.isoformat() if pd.notna(x) else "")
    )
    f["exit_bar_ts"] = (
        f["trade_id"].astype(int).map(ext_map).apply(lambda x: x.isoformat() if pd.notna(x) else "")
    )

    sig_dt = f["trade_id"].astype(int).map(sig_map)
    f["hour_utc"] = sig_dt.dt.hour.astype(int)
    f["day_of_week"] = sig_dt.dt.dayofweek.astype(int)
    f["session"] = f["hour_utc"].apply(_session_of_hour)
    f["hour_in_4h_bar"] = (f["hour_utc"] % 4).astype(int)
    f["bars_to_next_4h_close"] = 3 - f["hour_in_4h_bar"]
    f["hour_in_d1_bar"] = f["hour_utc"]
    f["bars_to_next_d1_close"] = 23 - f["hour_in_d1_bar"]

    f.to_csv(feat_path, index=False, lineterminator="\n", float_format="%.10g")
    print("Patched. Sample fixed rows:")
    print(
        f[
            [
                "trade_id",
                "signal_bar_ts",
                "hour_utc",
                "day_of_week",
                "session",
                "hour_in_4h_bar",
                "bars_to_next_4h_close",
                "hour_in_d1_bar",
            ]
        ].head()
    )


if __name__ == "__main__":
    main()
