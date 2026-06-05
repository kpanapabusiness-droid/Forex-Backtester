"""arc 2013 robustness gauntlet for the UP-gap weekend SHORT on JPY crosses (the arc-2011 discipline).

The pooled obs LOOKED promising: honest i+1 SHORT capture 0.515-0.518 (>0.50), drift +0.08 at gap>=1.0.
But it is THIN (137 ev >=1.0, ~12/yr) and NON-MONOTONE (>=1.5 inverts to -0.223). arc 2011 was killed by
exactly this: a thin +drift that was a 2-pair thin-tail artifact (mean>>median, collapses leaving 2 pairs).
Decisive checks BEFORE any engine spend:
  R1  mean vs MEDIAN drift (thin-tail tell) at each gate.
  R2  leave-one-pair-out drift (does dropping the top contributor collapse it?).
  R3  fair same-event NULL: random-day SHORT on the SAME crosses (does the up-gap timing add over a
      random weekly-open short, given JPY-basket drift contaminates the cell?).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.sim.panel import Panel
from discovery.tools.observe_long_capture import observe_long_capture
from discovery.tools.trend_entry_signals import _atr_shift1_mid

CROSSES = ["EURJPY", "GBPJPY", "AUDJPY", "NZDJPY", "CADJPY", "CHFJPY"]
BACKUP = r"C:\Users\panap\histdata_backup"
GAP_HOURS = 20.0


def build_gap_cond(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index; n = len(df)
    open_mid = (df["open_bid"].to_numpy(float) + df["open_ask"].to_numpy(float)) / 2.0
    close_mid = (df["close_bid"].to_numpy(float) + df["close_ask"].to_numpy(float)) / 2.0
    atr = _atr_shift1_mid(df, 14)
    prev_close = np.empty(n); prev_close[0] = np.nan; prev_close[1:] = close_mid[:-1]
    dt_hours = idx.to_series().diff().dt.total_seconds().to_numpy() / 3600.0
    gap_open = dt_hours > GAP_HOURS
    with np.errstate(invalid="ignore", divide="ignore"):
        gap_atr = (open_mid - prev_close) / atr
    return pd.DataFrame({"pair": None, "signal_time": idx, "gap_open": gap_open, "gap_atr": gap_atr, "atr_c": atr})


def main() -> None:
    lo_ts, hi_ts = pd.Timestamp("2010-01-01", tz="UTC"), pd.Timestamp("2020-12-31", tz="UTC")
    panel = Panel.from_pairs(CROSSES, tf="H4", histdata_root=BACKUP, cache_root="data/cache", boundary_convention="5ers_eet")
    obs = observe_long_capture(panel, sl_mult=2.0, hold=120, drift_bars=24, direction="short").set_index(["pair", "signal_time"])
    cond = pd.concat([build_gap_cond(panel.pair_dfs[p]).assign(pair=p) for p in CROSSES]).set_index(["pair", "signal_time"])
    j = obs.join(cond, how="inner")
    st = j.index.get_level_values("signal_time")
    j = j[(st >= lo_ts) & (st <= hi_ts)]
    j = j[np.isfinite(j["fwd_drift_atr"]) & np.isfinite(j["atr_c"]) & np.isfinite(j["gap_atr"])]

    print("=== arc 2013 ROBUSTNESS — UP-gap SHORT, JPY crosses, IS 2010-2020 ===")
    for g in (0.5, 1.0):
        cell = j[j["gap_open"] & (j["gap_atr"] >= g)]
        d = cell["fwd_drift_atr"]
        print(f"\n--- gate up-gap>=+{g} (n={len(cell)}) ---")
        print(f"  R1 mean={d.mean():+.4f}  MEDIAN={d.median():+.4f}  cap={cell['capture'].mean():.4f}  "
              f"(mean>>median => thin-tail, not a robust edge)")
        # R2 leave-one-pair-out
        print("  R2 leave-one-pair-out mean drift:")
        for drop in CROSSES:
            sub = cell[cell.index.get_level_values("pair") != drop]
            print(f"     drop {drop}: n={len(sub):3d} mean={sub['fwd_drift_atr'].mean():+.4f}")

    # R3 fair null: ALL weekly-open SHORTs (random-day-equivalent at the same event class) vs the up-gap cell
    print("\n--- R3 fair NULL: ALL weekly-open bars, SHORT lens (the up-gap must beat this) ---")
    allopen = j[j["gap_open"]]
    print(f"  all weekly-open SHORT: n={len(allopen)} mean_drift={allopen['fwd_drift_atr'].mean():+.4f} cap={allopen['capture'].mean():.4f}")
    for g in (0.5, 1.0):
        cell = j[j["gap_open"] & (j["gap_atr"] >= g)]
        lift = cell["fwd_drift_atr"].mean() - allopen["fwd_drift_atr"].mean()
        print(f"  up-gap>=+{g}: drift {cell['fwd_drift_atr'].mean():+.4f}  (lift over weekly-open null {lift:+.4f})")


if __name__ == "__main__":
    main()
