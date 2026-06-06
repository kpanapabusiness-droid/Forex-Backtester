"""arc 1049 — failed-breakdown RECLAIM (arc 1013, the corpus's best edge) on CROSSES.

OBSERVATION ONLY (step b/c/d). No gate, no P&L. arc 1013 (USD majors H4) is the strongest
directional edge in the corpus (IS 9/10) but its ONLY failing fold is 2018 — an explicitly
USD-TREND artifact ("in risk-off the failed breakdown becomes a real breakdown"). 2018 was a
USD-SPECIFIC one-way trend; on USD-NEUTRAL crosses 2018 was not a clean trend → the reclaim
may HOLD there, supplying the regime-orthogonal (2018-positive) complement the book needs.

We reuse the BUILT `FailedBreakdownReclaimLongSignal` (canonical structural signal) on the
cached crosses, and read its honest +1R-before-SL CAPTURE + forward drift per YEAR, focused on
the 2018 sign vs the known USD-majors 2018-negative.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.sim.panel import Panel
from discovery.tools.failed_breakdown_signals import FailedBreakdownReclaimLongSignal
from discovery.tools.observe_long_capture import observe_long_capture

ROOT = r"C:\Users\panap\histdata_backup"
CROSSES = ["AUDJPY", "EURGBP", "EURJPY", "GBPJPY"]   # H4-cached, USD-neutral
MAJORS = ["EURUSD", "GBPUSD", "AUDUSD", "USDJPY", "USDCAD", "USDCHF", "NZDUSD"]  # 1013 universe (ref)


def run_universe(name, pairs):
    print(f"\n{'='*70}\n{name}: {pairs}\n{'='*70}")
    panel = Panel.from_pairs(pairs, tf="H4", histdata_root=ROOT,
                             cache_root="data/cache", boundary_convention="5ers_eet")
    sig = FailedBreakdownReclaimLongSignal(swing_lookback=40, min_shadow_atr=1.25)
    ev = sig.evaluate({"H4": panel})
    # fire masks per pair -> restrict observe to fbr fires
    restrict = {}
    for p in pairs:
        m = ev.per_pair[p].signal_mask.to_numpy(bool)
        restrict[p] = m
    obs = observe_long_capture(panel, sl_mult=2.0, hold=120, drift_bars=24,
                               direction="long", restrict=restrict)
    obs["year"] = pd.DatetimeIndex(obs["signal_time"]).year
    n = len(obs)
    cap = obs["capture"].mean()
    drift = obs["fwd_drift_atr"].mean()
    print(f"  total fbr fires: {n}   capture {cap:.4f}   mean fwd-drift {drift:+.4f} ATR")
    print(f"  per-year (capture / mean drift / n):")
    by = obs.groupby("year").agg(cap=("capture", "mean"), drift=("fwd_drift_atr", "mean"),
                                 n=("capture", "count"))
    npos_drift = int((by["drift"] > 0).sum())
    for y, r in by.iterrows():
        flag = "  <-2018" if y == 2018 else ("  <-2015" if y == 2015 else "")
        print(f"    {y}: cap {r['cap']:.3f}  drift {r['drift']:+.3f}  n{int(r['n'])}{flag}")
    print(f"  -> drift-positive years: {npos_drift}/{len(by)}")
    return obs


def main():
    obs_x = run_universe("CROSSES (USD-neutral, the 2018 test)", CROSSES)
    obs_m = run_universe("USD MAJORS (arc-1013 reference)", MAJORS)

    # Direct 2018 comparison
    print(f"\n{'='*70}\n2018 HEAD-TO-HEAD (the wall)\n{'='*70}")
    for name, o in [("CROSSES", obs_x), ("MAJORS", obs_m)]:
        s = o[o["year"] == 2018]
        if len(s):
            print(f"  {name} 2018: capture {s['capture'].mean():.3f}  drift {s['fwd_drift_atr'].mean():+.3f}  n{len(s)}")
        else:
            print(f"  {name} 2018: no fires")


if __name__ == "__main__":
    main()
