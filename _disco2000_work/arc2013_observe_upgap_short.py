"""arc 2013 — Weekend UP-gap weekend SHORT (the gap-fill's STRONGER leg, finally tested). Cheap obs (b/d).

WHY (the *because*).
arc 2001 (2000s) found the weekly-open gap-fill is clean, monotone, SYMMETRIC on H4: DOWN >1ATR gaps
drift +0.45 (frac+ 0.59) and UP >1ATR gaps drift -0.57 (frac DOWN 0.64). The UP-gap leg is the STRONGER
one (bigger drift, higher hindsight acc) but was structurally untradeable under long-only → FLAG-1.
arc 1006 (1000s) found the DOWN-gap FILL LONG is mean-POSITIVE on JPY CROSSES (+0.685% IS, the corpus's
one PORTFOLIO edge) but mean-NEGATIVE on majors → JPY crosses are the better gap universe. Shorts are now
merged (PR #273). The natural, highest-acc, still-UNRUN test (1000s used arc 1015 for the portfolio
combination, not the up-gap short): the UP-gap weekend SHORT on JPY CROSSES — the mirror of 1006's
mean-positive long, on its STRONGER leg.

THE HONESTY CHECK (arc 2001's verdict, mirrored). 2001's down-gap LONG honest i+1 capture was only
0.45-0.47 despite 0.59 hindsight acc, because the i+1 entry (after the gap bar) lands INSIDE the adverse
continuation (the gap extends further before reverting; MAE -1.1R). By symmetry the up-gap SHORT entered
i+1 should face the SAME adverse continuation (price extends UP a bit more before filling down). So the
0.64 hindsight acc likely collapses to ~0.46 honest. BUT: (a) the universe differs (1006: crosses
mean-POS where 2001 majors mean-NEG), and (b) the up-gap leg is empirically stronger — so it must be
measured, not assumed. "The mechanism survives; does the TRADE?" — on crosses this time.

DECISIVE cheap tests (gap-detection mirrors the canonical WeekendGapFillLongSignal exactly):
  Q1  honest i+1 SHORT capture + drift of the UP-gap cell vs base, by gap-size gate, JPY CROSSES.
      (short drift > 0 => price fell => good for the short.)
  Q2  universe split — JPY CROSSES vs MAJORS (does 1006's cross>major asymmetry hold for the short?).
  Q3  symmetry check — the DOWN-gap LONG (1006, known +) vs the UP-gap SHORT (this), same crosses.
  Q4  per-pair robustness (watch the thin weekly-event count; ~1 event/pair/wk).
  Q5  per-year drift incl 2018 (a fade -> expected to bleed the trend year; confirms it is NOT the
      portfolio's 2018 leg but may be a 4th decorrelated PORTFOLIO component / standalone candidate).

CHARACTERIZATION ONLY (gross). Honest short capture/drift via direction-aware observe_long_capture
(BUILT). Gap conditioning mirrors gap_signals.py (ex-ante: open[i] & close[i-1] known at bar i, ATR
shift1, weekly-open via dt>gap_hours). Engine only if this clears the cheap-kill.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.sim.panel import Panel
from discovery.tools.observe_long_capture import observe_long_capture
from discovery.tools.trend_entry_signals import _atr_shift1_mid

CROSSES = ["EURJPY", "GBPJPY", "AUDJPY", "NZDJPY", "CADJPY", "CHFJPY"]
MAJORS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY"]
BACKUP = r"C:\Users\panap\histdata_backup"
GAP_HOURS = 20.0


def build_gap_cond(df: pd.DataFrame) -> pd.DataFrame:
    """Per-bar weekly-open gap conditioning, ex-ante (mirror of WeekendGapFillLongSignal)."""
    idx = df.index
    n = len(df)
    open_mid = (df["open_bid"].to_numpy(float) + df["open_ask"].to_numpy(float)) / 2.0
    close_mid = (df["close_bid"].to_numpy(float) + df["close_ask"].to_numpy(float)) / 2.0
    atr = _atr_shift1_mid(df, 14)
    prev_close = np.empty(n)
    prev_close[0] = np.nan
    prev_close[1:] = close_mid[:-1]
    dt_hours = idx.to_series().diff().dt.total_seconds().to_numpy() / 3600.0
    gap_open = dt_hours > GAP_HOURS
    with np.errstate(invalid="ignore", divide="ignore"):
        gap_atr = (open_mid - prev_close) / atr
    return pd.DataFrame({
        "pair": None, "signal_time": idx,
        "gap_open": gap_open, "gap_atr": gap_atr, "atr_c": atr,
    })


def observe_universe(panel: Panel, pairs: list[str], lo_ts, hi_ts) -> pd.DataFrame:
    obs = observe_long_capture(panel, sl_mult=2.0, hold=120, drift_bars=24, direction="short")
    obs = obs.set_index(["pair", "signal_time"])
    cond = pd.concat([build_gap_cond(panel.pair_dfs[p]).assign(pair=p) for p in pairs]).set_index(["pair", "signal_time"])
    j = obs.join(cond, how="inner")
    st = j.index.get_level_values("signal_time")
    j = j[(st >= lo_ts) & (st <= hi_ts)]
    return j[np.isfinite(j["fwd_drift_atr"]) & np.isfinite(j["atr_c"]) & np.isfinite(j["gap_atr"])]


def report_upgap(j: pd.DataFrame, label: str) -> None:
    base_cap, base_drift = j["capture"].mean(), j["fwd_drift_atr"].mean()
    print(f"\n=== {label}: BASE (all bars, SHORT lens) n={len(j)} cap={base_cap:.4f} drift={base_drift:+.4f} ===")
    print("  (short drift > 0 => price fell => good for the short)")
    print("  --- Q1: UP-gap SHORT cell (gap_open & gap_atr>=+g) ---")
    for g in (0.5, 1.0, 1.5):
        cell = j[j["gap_open"] & (j["gap_atr"] >= g)]
        if len(cell) < 30:
            print(f"    up-gap>=+{g}: n={len(cell)} (THIN)")
            continue
        print(f"    up-gap>=+{g}: n={len(cell):4d} cap={cell['capture'].mean():.4f} drift={cell['fwd_drift_atr'].mean():+.4f} "
              f"(lift cap {cell['capture'].mean()-base_cap:+.4f}, drift {cell['fwd_drift_atr'].mean()-base_drift:+.4f})")
    # symmetry: DOWN-gap LONG-equivalent = on the SHORT lens, a down-gap should have NEGATIVE short drift
    # (price rises = fill up = good for the 1006 LONG). Report the down-gap cell for the symmetry check (Q3).
    print("  --- Q3 symmetry: DOWN-gap cell on the SHORT lens (short drift<0 <=> price rose <=> 1006 long good) ---")
    for g in (0.5, 1.0, 1.5):
        cell = j[j["gap_open"] & (j["gap_atr"] <= -g)]
        if len(cell) < 30:
            print(f"    down-gap<=-{g}: n={len(cell)} (THIN)")
            continue
        print(f"    down-gap<=-{g}: n={len(cell):4d} cap={cell['capture'].mean():.4f} drift={cell['fwd_drift_atr'].mean():+.4f}")


def per_pair_year(j: pd.DataFrame, pairs: list[str], g: float, label: str) -> None:
    cell = j[j["gap_open"] & (j["gap_atr"] >= g)]
    print(f"\n--- Q4 per-pair UP-gap SHORT (gap>=+{g}) [{label}] ---")
    npos = 0; ntot = 0
    for pair in pairs:
        sub = cell[cell.index.get_level_values("pair") == pair]
        if len(sub) >= 10:
            dr = sub["fwd_drift_atr"].mean(); npos += int(dr > 0); ntot += 1
            print(f"    {pair}: n={len(sub):3d} cap={sub['capture'].mean():.4f} drift={dr:+.4f}")
        else:
            print(f"    {pair}: n={len(sub):3d} (thin)")
    print(f"  pairs with positive short drift: {npos}/{ntot}")
    print(f"\n--- Q5 per-year UP-gap SHORT drift (gap>=+{g}) [{label}] (the 2018 question) ---")
    yrs = cell.index.get_level_values("signal_time").year
    for y in range(2010, 2021):
        sub = cell[yrs == y]
        if len(sub) >= 8:
            print(f"    {y}: n={len(sub):3d} cap={sub['capture'].mean():.4f} drift={sub['fwd_drift_atr'].mean():+.4f}")
        else:
            print(f"    {y}: n={len(sub):3d} (thin)")


def main() -> None:
    lo_ts, hi_ts = pd.Timestamp("2010-01-01", tz="UTC"), pd.Timestamp("2020-12-31", tz="UTC")
    print("=== arc 2013 UP-gap weekend SHORT (H4, IS 2010-2020) — the gap-fill's stronger leg ===")

    pc = Panel.from_pairs(CROSSES, tf="H4", histdata_root=BACKUP, cache_root="data/cache", boundary_convention="5ers_eet")
    jc = observe_universe(pc, CROSSES, lo_ts, hi_ts)
    report_upgap(jc, "JPY CROSSES")
    per_pair_year(jc, CROSSES, 1.0, "JPY CROSSES")

    pm = Panel.from_pairs(MAJORS, tf="H4", histdata_root=BACKUP, cache_root="data/cache", boundary_convention="5ers_eet")
    jm = observe_universe(pm, MAJORS, lo_ts, hi_ts)
    report_upgap(jm, "MAJORS (Q2 universe split)")


if __name__ == "__main__":
    main()
