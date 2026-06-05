"""Arc 2025 observation: the GOTOBI (五十日) effect — Japanese settlement USD demand into the 09:55 JST
Tokyo fix on days divisible by 5 (5/10/15/20/25 + month-end). OBSERVATION ONLY (gross; NOT a gate).

because: a documented, calendar-anchored, regime-ORTHOGONAL forced flow distinct from the month-end WMR
fix (arc 3008): on "gotobi" days Japanese importers/corporates settle in USD, creating USD-buying demand
into the 09:55 JST Tokyo fix -> USDJPY (and USD-vs-JPY) tends to DRIFT UP through the Tokyo morning into
the fix, then the demand exhausts. If real and > FundedNext cost, a long-USDJPY-into-the-Tokyo-fix on
gotobi days is a thick (~6 events/month), calendar-driven candidate component.

Timing: 09:00 JST = 00:00 UTC; 09:55 JST = 00:55 UTC. The H1 bar timestamped 00:00 UTC (09:00-10:00 JST)
SPANS the 09:55 fix, so its open->close return is the cheap proxy for the pre-fix Tokyo-morning drift.
Also measure the broader Tokyo-morning window (open of 22:00-UTC prior-day Tokyo... -> close of 00:00 bar).

Question (§5d cheap-kill): is the gotobi-day Tokyo-morning USDJPY drift POSITIVE, larger than non-gotobi
days (gotobi EXCESS), directionally reliable (frac>0), and is it ABOVE the ~cost line? If coin-flip or
sub-cost -> KILL (complements arc 3008: intra-month fix flow also sub-cost).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.sim.panel import Panel

PAIRS = ["USDJPY", "EURJPY", "GBPJPY", "AUDJPY"]  # USD-quoted JPY + JPY crosses
HIST = r"C:\Users\panap\histdata_backup"
IS_LO, IS_HI = 2010, 2020

print("loading H1 panel (cached)...")
panel = Panel.from_pairs(PAIRS, tf="H1", histdata_root=HIST,
                         cache_root="data/cache", boundary_convention="5ers_eet")


def gotobi_flag(dates: pd.DatetimeIndex) -> np.ndarray:
    """True on gotobi days: day-of-month in {5,10,15,20,25} OR the last calendar day of the month.
    (Simple rule; weekend->prior-business-day shift is a refinement deferred unless the base shows signal.)"""
    dom = dates.day.to_numpy()
    # last day of month: next day is a different month
    is_month_end = (dates + pd.Timedelta(days=1)).month.to_numpy() != dates.month.to_numpy()
    return np.isin(dom, [5, 10, 15, 20, 25]) | is_month_end


for pair in PAIRS:
    df = panel.pair_dfs[pair]
    idx = df.index  # UTC
    open_mid = (df["open_bid"].to_numpy(float) + df["open_ask"].to_numpy(float)) / 2.0
    close_mid = (df["close_bid"].to_numpy(float) + df["close_ask"].to_numpy(float)) / 2.0
    spread = (df["close_ask"].to_numpy(float) - df["close_bid"].to_numpy(float))
    yr = idx.year.to_numpy()
    hour = idx.hour.to_numpy()
    # the 00:00 UTC bar spans the 09:55 JST fix; its UTC calendar date = the Tokyo trading day
    fix_bar = (hour == 0) & (yr >= IS_LO) & (yr <= IS_HI)
    ret = (close_mid - open_mid)  # open->close of the fix-spanning bar, in price
    # pip scale: JPY pairs => 1 pip = 0.01
    pip = 0.01
    ret_pips = ret / pip
    sp_pips = spread / pip

    dates = idx
    goto = gotobi_flag(dates)

    sel = fix_bar & np.isfinite(ret_pips)
    g = sel & goto
    ng = sel & ~goto

    def stats(m):
        r = ret_pips[m]
        return len(r), r.mean(), np.median(r), (r > 0).mean(), sp_pips[m].mean()

    ng_n, ng_mean, ng_med, ng_fp, ng_sp = stats(ng)
    g_n, g_mean, g_med, g_fp, g_sp = stats(g)
    excess = g_mean - ng_mean
    # FundedNext H1 RT cost proxy on this pair: 1.5x spread + ~0.5pip slippage + ~commission(~0.5pip-equiv)
    cost_pips = 1.5 * g_sp + 0.5 + 0.5
    print(f"\n=== {pair} (00:00-UTC fix-spanning H1 bar, IS 2010-2020) ===")
    print(f"  non-gotobi: n={ng_n:5d} mean={ng_mean:+.3f}p med={ng_med:+.3f}p frac+={ng_fp:.3f} spread={ng_sp:.2f}p")
    print(f"  GOTOBI    : n={g_n:5d} mean={g_mean:+.3f}p med={g_med:+.3f}p frac+={g_fp:.3f} spread={g_sp:.2f}p")
    print(f"  gotobi EXCESS mean = {excess:+.3f}p   |   ~RT cost proxy = {cost_pips:.2f}p   "
          f"-> {'ABOVE cost' if g_mean > cost_pips else 'SUB-COST'}")
    # per-year sign of gotobi mean (does it survive / which regime)
    yrs = dates.year.to_numpy()
    rows = []
    for y in range(IS_LO, IS_HI + 1):
        mm = g & (yrs == y)
        if mm.sum() > 0:
            rows.append((y, ret_pips[mm].mean(), int(mm.sum())))
    pos = sum(1 for _, mv, _ in rows if mv > 0)
    y2018 = next((mv for y, mv, _ in rows if y == 2018), float("nan"))
    y2015 = next((mv for y, mv, _ in rows if y == 2015), float("nan"))
    print(f"  per-year gotobi mean>0: {pos}/{len(rows)} folds   2015={y2015:+.3f}p  2018={y2018:+.3f}p")

print("\nDONE.")
