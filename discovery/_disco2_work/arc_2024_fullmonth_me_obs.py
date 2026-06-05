"""Arc 2024 observation: is the FULL-MONTH currency return a cleaner trigger for the month-end
rebalancing reversion than the corpus's 2-day `into_bars=2` move — and does a UNIFIED dual-direction,
thicker month-end fade RESOLVE folds (arc-2017 option B) while keeping the +2015/+2018 property?

OBSERVATION ONLY (gross, take-the-loss capture; NOT a gate). Reuses the canonical month-end geometry
(`_month_end_into_move`) + the direction-aware `observe_long_capture` harness. NO engine/cost here —
this is the §5d cheap-kill screen; the engine (`ArcFoldRunner`) only runs if obs clears.

because: the WMR 4pm-fix month-end rebalancing flow scales with the MONTH'S currency appreciation
(equity-hedge rebalancing on the full month's FX move), NOT the last-2-day move `me` (1011/1019) uses.
The economically-correct trigger is the full-month (~20 D1 bar) return. A UNIFIED dual-direction fade
(long if the currency fell over the window, short if it rose) is also ~2x thicker than each one-sided
`me` leg -> a shot at the arc-2017 option-B "thick enough that folds RESOLVE" standalone, via the ONE
mechanism that demonstrably carries the binding 2018 fold (me_long +0.90, me_short +0.86).

Questions:
  (Q1) Does capture / month-end-EXCESS drift RISE with into_bars (2->20)? (full-month cleaner than 2-day)
  (Q2) Does the unified dual-direction fade clear 0.50 capture and stay +2015 AND +2018 per-year?
  (Q3) Is it actually thicker, and does thickness come WITH or WITHOUT edge dilution (the arc-1025/2018
       unified-theory prediction: thickening a forced-flow reversion dilutes to coin-flip)?
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.sim.panel import Panel
from discovery.tools.month_end_signals import _month_end_into_move
from discovery.tools.observe_long_capture import observe_long_capture

PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY"]
HIST = r"C:\Users\panap\histdata_backup"
INTO_SET = [2, 5, 10, 20]
THR_SET = [1.0, 0.5]
IS_LO, IS_HI = 2010, 2020

print("loading D1 panel (cached)...")
panel = Panel.from_pairs(PAIRS, tf="D1", histdata_root=HIST,
                         cache_root="data/cache", boundary_convention="5ers_eet")


def _is_window(idx):
    yr = idx.year.to_numpy()
    return (yr >= IS_LO) & (yr <= IS_HI)


def fire_masks(into_bars, thr):
    """Per-pair boolean masks for ME-long / ME-short / control-long / control-short fires (IS window)."""
    me_long, me_short, ctl_long, ctl_short = {}, {}, {}, {}
    for pair in PAIRS:
        df = panel.pair_dfs[pair]
        idx, is_last, into, atr = _month_end_into_move(df, into_bars, 14)
        ok = np.isfinite(into) & np.isfinite(atr) & (atr > 0) & _is_window(idx)
        down = ok & (into <= -thr)
        up = ok & (into >= thr)
        me_long[pair] = down & is_last
        me_short[pair] = up & is_last
        ctl_long[pair] = down & ~is_last      # same big-down move, NOT month-end (random-day control)
        ctl_short[pair] = up & ~is_last
    return me_long, me_short, ctl_long, ctl_short


def unified_obs(long_mask, short_mask):
    """Run the direction-aware harness on the long-restricted and short-restricted fires, concat.
    capture & fwd_drift_atr are already direction-correct (positive = favorable for the position)."""
    parts = []
    obs_l = observe_long_capture(panel, sl_mult=2.0, hold=120, drift_bars=10,
                                 restrict=long_mask, direction="long")
    if len(obs_l):
        obs_l["dir"] = "long"
        parts.append(obs_l)
    obs_s = observe_long_capture(panel, sl_mult=2.0, hold=120, drift_bars=10,
                                 restrict=short_mask, direction="short")
    if len(obs_s):
        obs_s["dir"] = "short"
        parts.append(obs_s)
    if not parts:
        return pd.DataFrame(columns=["pair", "signal_time", "capture", "fwd_drift_atr", "atr", "dir"])
    return pd.concat(parts, ignore_index=True)


def summarize(obs, label):
    n = len(obs)
    if n == 0:
        print(f"  {label:28s}  n=0")
        return None
    cap = obs["capture"].mean()
    drift = obs["fwd_drift_atr"].mean()
    med = obs["fwd_drift_atr"].median()
    per_pair = (obs.groupby("pair")["fwd_drift_atr"].mean() > 0).sum()
    print(f"  {label:28s}  n={n:4d}  cap={cap:.4f}  drift={drift:+.4f}  med={med:+.4f}  "
          f"pairs+>0:{per_pair}/{obs['pair'].nunique()}")
    return {"n": n, "cap": cap, "drift": drift, "med": med}


def per_year(obs):
    obs = obs.copy()
    obs["yr"] = pd.to_datetime(obs["signal_time"]).dt.year
    g = obs.groupby("yr")["fwd_drift_atr"].agg(["mean", "count"])
    pos = int((g["mean"] > 0).sum())
    yr15 = g.loc[2015, "mean"] if 2015 in g.index else float("nan")
    yr18 = g.loc[2018, "mean"] if 2018 in g.index else float("nan")
    return g, pos, yr15, yr18


print("\n" + "=" * 96)
print("Q1/Q3 — capture & month-end-EXCESS vs into_bars (full-month trigger?) and thickness")
print("=" * 96)
results = {}
for thr in THR_SET:
    print(f"\n--- threshold = {thr} ATR ---")
    for into_bars in INTO_SET:
        ml, ms, cl, cs = fire_masks(into_bars, thr)
        me = unified_obs(ml, ms)
        ctl = unified_obs(cl, cs)
        print(f"\n into_bars={into_bars:2d}  (unified dual-direction fade)")
        s_me = summarize(me, "MONTH-END fade")
        s_ctl = summarize(ctl, "random-day control")
        if s_me and s_ctl:
            print(f"   month-end EXCESS: dcap={s_me['cap']-s_ctl['cap']:+.4f}  "
                  f"ddrift={s_me['drift']-s_ctl['drift']:+.4f}")
        if s_me:
            g, pos, y15, y18 = per_year(me)
            print(f"   per-year drift>0: {pos}/{g.shape[0]} folds   2015={y15:+.4f}  2018={y18:+.4f}")
            results[(thr, into_bars)] = {"me": s_me, "ctl": s_ctl, "yr_pos": pos,
                                         "y15": y15, "y18": y18, "per_year": g}

print("\n" + "=" * 96)
print("SUMMARY GRID (unified dual-direction fade)")
print("=" * 96)
print(f"{'thr':>4} {'into':>5} {'n':>5} {'cap':>7} {'excess_cap':>11} {'excess_drift':>13} "
      f"{'yr+':>5} {'y2015':>8} {'y2018':>8}")
for (thr, ib), r in results.items():
    ex_cap = r["me"]["cap"] - r["ctl"]["cap"] if r["ctl"] else float("nan")
    ex_dr = r["me"]["drift"] - r["ctl"]["drift"] if r["ctl"] else float("nan")
    print(f"{thr:>4} {ib:>5} {r['me']['n']:>5} {r['me']['cap']:>7.4f} {ex_cap:>11.4f} "
          f"{ex_dr:>13.4f} {r['yr_pos']:>5} {r['y15']:>8.4f} {r['y18']:>8.4f}")
print("\nDONE.")
