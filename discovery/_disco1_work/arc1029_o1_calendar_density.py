"""Arc 1029 — O1: inelasticity-STATE conditioning via the calendar-flow-DENSITY proxy.

The strategist MENU's O1: the surviving edges are too thin to resolve folds (arcs 2016/17/19);
the only surviving mechanism is FORCED FLOW and the only surviving calendar driver is MONTH-END
(arc 3015). O1's live proxy = forced-flow calendar DENSITY: concentrate the proven month-end
reversion onto its highest-inelasticity windows. Forced rebalancing flow is LARGER at
quarter-end (Jun/Sep) and largest at fiscal-year-end (Mar = Japan FYE; Dec = calendar/global YE)
than at an ordinary month-end -> the reversion edge-per-trade should rise MONOTONICALLY with
calendar density. If so, a density-concentrated book has a higher worst-fold (the council's
fold-resolution attack); if NOT monotone, calendar-density is "not a lever" (arc-3007 tell).

Pools BOTH proven legs as reversion trades: me_long (fade a big DOWN move into ME, arc 1011) +
me_short (fade a big UP move into ME, arc 1019) -> direction-aware honest capture IS the
reversion edge per trade. IS 2010-2020, D1, USD majors. Cheap OBSERVATION (no engine).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.sim.panel import Panel
from discovery.tools.month_end_signals import (
    MonthEndReversionLongSignal,
    MonthEndReversionShortSignal,
)
from discovery.tools.observe_long_capture import observe_long_capture

HIST = r"C:\Users\panap\histdata_backup"
CACHE = "data/cache"
PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCAD", "USDCHF"]
IS_START, IS_END = pd.Timestamp("2010-01-01", tz="UTC"), pd.Timestamp("2020-12-31", tz="UTC")


def density_tier(month: int) -> int:
    if month in (3, 12):      # fiscal-year-end: Japan FYE (Mar 31) / calendar+global YE (Dec 31)
        return 3
    if month in (6, 9):       # quarter-end
        return 2
    return 1                  # ordinary month-end


def _fire_mask(sig, panel):
    ev = sig.evaluate({"D1": panel})
    return {p: ev.per_pair[p].signal_mask.to_numpy(bool) for p in panel.pairs}


def main():
    panel = Panel.from_pairs(PAIRS, tf="D1", histdata_root=HIST, cache_root=CACHE,
                             boundary_convention="5ers_eet")
    # clip each pair to IS for the observation by zeroing fires outside IS via the mask
    long_sig = MonthEndReversionLongSignal(threshold_atr=1.0, into_bars=2)
    short_sig = MonthEndReversionShortSignal(threshold_atr=1.0, into_bars=2)
    long_fires = _fire_mask(long_sig, panel)
    short_fires = _fire_mask(short_sig, panel)

    frames = []
    for side, fires, direction in (("long", long_fires, "long"), ("short", short_fires, "short")):
        obs = observe_long_capture(panel, sl_mult=2.0, hold=120, drift_bars=2, warmup=30,
                                   restrict=fires, direction=direction)
        obs["side"] = side
        frames.append(obs)
    df = pd.concat(frames, ignore_index=True)
    # restrict to IS and tag density by the month that ENDED at the fire bar
    df = df[(df["signal_time"] >= IS_START) & (df["signal_time"] <= IS_END)].copy()
    df["month"] = df["signal_time"].dt.month
    df["year"] = df["signal_time"].dt.year
    df["tier"] = df["month"].map(density_tier)

    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(f"=== O1 calendar-density: pooled month-end reversion (me_long+me_short), IS, n={len(df)} ===")
    print("\n-- reversion edge per trade by DENSITY TIER (1=ordinary, 2=quarter-end, 3=fiscal-YE) --")
    g = df.groupby("tier").agg(n=("capture", "size"),
                               capture=("capture", "mean"),
                               drift_mean=("fwd_drift_atr", "mean"),
                               drift_med=("fwd_drift_atr", "median"))
    print(g.to_string())
    print("\n-- by MONTH (density driver detail) --")
    gm = df.groupby("month").agg(n=("capture", "size"), capture=("capture", "mean"),
                                 drift_mean=("fwd_drift_atr", "mean"),
                                 drift_med=("fwd_drift_atr", "median"))
    print(gm.to_string())

    # fold-resolution proxy: per-year reversion drift, high-density (tier>=2) vs ALL
    print("\n-- fold-resolution proxy: per-year mean reversion drift --")
    allyr = df.groupby("year")["fwd_drift_atr"].mean()
    hi = df[df["tier"] >= 2].groupby("year")["fwd_drift_atr"].mean()
    comp = pd.DataFrame({"all_drift": allyr, "hi_density_drift": hi,
                         "n_all": df.groupby("year").size(),
                         "n_hi": df[df["tier"] >= 2].groupby("year").size()})
    print(comp.to_string())
    print(f"\n  ALL: neg-years {int((allyr<0).sum())}/{allyr.notna().sum()}  mean drift {allyr.mean():+.4f}")
    print(f"  HI-DENSITY (tier>=2): neg-years {int((hi<0).sum())}/{hi.notna().sum()}  mean drift {hi.mean():+.4f}"
          f"  median trades/yr {comp['n_hi'].median():.0f}")


if __name__ == "__main__":
    main()
