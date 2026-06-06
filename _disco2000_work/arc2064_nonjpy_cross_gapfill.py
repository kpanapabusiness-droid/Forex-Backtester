"""arc 2064 — weekend gap-fill on the NON-JPY, NON-USD-major CROSS universe (cheap obs, IS-only).

The surviving gap edge (arc 1006) lives on 5 JPY crosses (mean-POS); arc 2001 found it mean-NEG on
USD majors. The 15 non-JPY / non-USD-major crosses were NEVER tested (arc 1006 thread #112 flagged it).
This isolates the gap-fill MECHANISM (weekend repricing overshoot -> reversion) from the JPY-basket
forward-drift confound arc 1009 found in the JPY null (the JPY null itself was +0.327%).

OBSERVATION ONLY (gross capture + ATR drift, no engine, no cost) -> §5d cheap-kill screen. IS 2010-2020
ONLY; OOS never touched here. Reuses BUILT WeekendGapFillLongSignal + observe_long_capture.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.sim.panel import Panel
from discovery.tools.gap_signals import WeekendGapFillLongSignal
from discovery.tools.observe_long_capture import observe_long_capture

HIST = r"C:\Users\panap\histdata_backup"
CACHE = "data/cache"

CROSSES = [
    "EURGBP", "EURAUD", "EURCAD", "EURCHF", "EURNZD",
    "GBPAUD", "GBPCAD", "GBPCHF", "GBPNZD",
    "AUDCAD", "AUDCHF", "AUDNZD", "CADCHF", "NZDCAD", "NZDCHF",
]
JPY = ["EURJPY", "GBPJPY", "AUDJPY", "CADJPY", "CHFJPY"]  # arc-1006 benchmark


def fire_mask(panel, thr):
    ev = WeekendGapFillLongSignal(threshold_atr=thr, gap_hours=20.0).evaluate({"H4": panel})
    return {p: ev.per_pair[p].signal_mask.to_numpy(bool) for p in panel.pairs}


def obs_block(panel, thr, label):
    masks = fire_mask(panel, thr)
    obs = observe_long_capture(panel, sl_mult=2.0, hold=120, drift_bars=24, restrict=masks)
    obs = obs[obs["signal_time"].dt.year <= 2020].copy()  # IS only
    obs["year"] = obs["signal_time"].dt.year
    n = len(obs)
    print(f"\n=== {label}  thr={thr}  (IS 2010-2020) ===")
    print(f"  n_fires={n}  (~{n/11:.1f}/yr)  pool-floor(>=50)={'OK' if n>=50 else 'FAIL'}")
    if n == 0:
        return obs
    print(f"  capture={obs['capture'].mean():.4f}   "
          f"fwd24_drift_atr mean={obs['fwd_drift_atr'].mean():+.4f} "
          f"median={obs['fwd_drift_atr'].median():+.4f}")
    pp = obs.groupby("pair").agg(n=("capture", "size"), cap=("capture", "mean"),
                                 drift=("fwd_drift_atr", "mean"))
    pp = pp.sort_values("drift", ascending=False)
    print("  per-pair (cap / mean-drift / n):")
    for p, r in pp.iterrows():
        print(f"    {p}: {r['cap']:.3f} / {r['drift']:+.3f} / {int(r['n'])}")
    print(f"  per-pair frac with drift>0: {(pp['drift']>0).mean():.2f} ({int((pp['drift']>0).sum())}/{len(pp)})")
    yr = obs.groupby("year").agg(n=("capture", "size"), cap=("capture", "mean"),
                                 drift=("fwd_drift_atr", "mean"))
    print("  per-year (cap / mean-drift / n)  [binding folds: 2015, 2018]:")
    for y, r in yr.iterrows():
        flag = "  <-- BIND" if y in (2015, 2018) else ""
        print(f"    {y}: {r['cap']:.3f} / {r['drift']:+.3f} / {int(r['n'])}{flag}")
    return obs


def main():
    print("Loading non-JPY cross panel (15 pairs)...")
    xpanel = Panel.from_pairs(CROSSES, tf="H4", histdata_root=HIST, cache_root=CACHE,
                             boundary_convention="5ers_eet")
    print("Loading JPY-cross benchmark panel (5 pairs)...")
    jpanel = Panel.from_pairs(JPY, tf="H4", histdata_root=HIST, cache_root=CACHE,
                             boundary_convention="5ers_eet")

    # Benchmark: reproduce arc-1006 JPY-cross gap-fill capture/drift (sanity anchor)
    obs_block(jpanel, 0.5, "JPY CROSSES (arc-1006 benchmark)")

    # The fresh test: non-JPY crosses
    obs_block(xpanel, 0.5, "NON-JPY CROSSES  thr0.5")
    obs_block(xpanel, 1.0, "NON-JPY CROSSES  thr1.0")

    # Structure control: is the gap LOAD-BEARING? capture/drift on ALL week-opens (no gap filter)
    print("\n=== STRUCTURE CONTROL: all week-opens (no gap threshold), non-JPY crosses, IS ===")
    allmask = {}
    for p in xpanel.pairs:
        df = xpanel.pair_dfs[p]
        dt_h = df.index.to_series().diff().dt.total_seconds().to_numpy() / 3600.0
        allmask[p] = dt_h > 20.0
    ctrl = observe_long_capture(xpanel, sl_mult=2.0, hold=120, drift_bars=24, restrict=allmask)
    ctrl = ctrl[ctrl["signal_time"].dt.year <= 2020]
    print(f"  all week-opens n={len(ctrl)}  capture={ctrl['capture'].mean():.4f}  "
          f"mean-drift={ctrl['fwd_drift_atr'].mean():+.4f}  median={ctrl['fwd_drift_atr'].median():+.4f}")
    print("  (if big-gap drift ~= all-week-open drift -> gap NOT load-bearing -> generic, dead)")


if __name__ == "__main__":
    main()
