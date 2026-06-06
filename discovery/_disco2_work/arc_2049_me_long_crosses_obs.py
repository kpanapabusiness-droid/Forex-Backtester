"""arc 2049 OBSERVATION (cheap-kill) — Month-end reversion LONG on NON-USD crosses: a +2014/15/16 leg?

because: me_long (1011, USD majors) is the sole exit-robust survivor but DIES in the contiguous
2014/15/16 strong-USD block — a big DOWN move into month-end in a USD major IS the USD trend, so the
WMR reversion gets overrun. On a NON-USD cross (EURJPY / EURGBP / ...) a down-move into month-end is
NOT USD-driven -> the same forced month-end rebalancing reversion might HOLD in 2014/15/16 -> a
decorrelated +2014/15/16 leg (complementing me_short's +2018, the precise unfound spec of 1015/1020).
me_SHORT broad-cross was tested (1021: JPY carries 2016, XCROSS adds 2015, disqualified on robustness);
me_LONG on crosses specifically for +2014/15/16 is genuinely UNTESTED.

PRIOR (1018/2031/1022): corpus edges are USD-major-specific & don't port off the dollar factor — BUT
me is a FLOW edge and its sibling flow (gap-fill 1006) is NATIVELY JPY-cross-positive/major-negative,
so this is genuinely open.

FALSIFIERS (§5d): (1) cross me_long capture coin-flip (~baseline) / month-end EXCESS ~0 vs random-day
control (no real flow); (2) 2014/15/16 NOT positive (wrong-sign / regime-luck). Either -> KILL.
Only a real, 2014-16-positive, control-confirmed edge -> §5f engine.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.sim.panel import Panel
from discovery.tools.month_end_signals import _month_end_into_move
from discovery.tools.observe_long_capture import observe_long_capture

JPY_CROSSES = ["EURJPY", "GBPJPY", "AUDJPY"]
XCROSSES = ["EURGBP", "EURAUD", "AUDNZD"]
ALL = JPY_CROSSES + XCROSSES
BACKUP = r"C:\Users\panap\histdata_backup"
INTO_BARS, AFTER = 2, 2
THR = 1.0

print(f"loading D1 panel: {ALL}", flush=True)
panel = Panel.from_pairs(ALL, tf="D1", histdata_root=BACKUP, cache_root="data/cache",
                         boundary_convention="5ers_eet")

# month-end fires (down-move into ME) + a RANDOM-DAY control (same big-down on a non-ME day)
me_fire, ctrl_fire = {}, {}
for pair in ALL:
    df = panel.pair_dfs[pair]
    _, is_last, into, atr = _month_end_into_move(df, INTO_BARS, 14)
    big_down = np.isfinite(into) & np.isfinite(atr) & (atr > 0) & (into <= -THR)
    me_fire[pair] = is_last & big_down
    ctrl_fire[pair] = (~is_last) & big_down  # same magnitude move, NOT at month-end

def summarize(restrict, label):
    obs = observe_long_capture(panel, sl_mult=2.0, hold=120, drift_bars=AFTER,
                               restrict=restrict, direction="long")
    obs["year"] = obs["signal_time"].dt.year
    obs["grp"] = obs["pair"].map(lambda p: "JPY" if p in JPY_CROSSES else "XC")
    print(f"\n=== {label}: n={len(obs)}  capture {obs['capture'].mean():.4f}  "
          f"drift {obs['fwd_drift_atr'].mean():+.3f} ATR ===", flush=True)
    for g, sub in obs.groupby("grp"):
        print(f"  [{g}] n {len(sub):4d}  cap {sub['capture'].mean():.4f}  "
              f"drift {sub['fwd_drift_atr'].mean():+.3f}", flush=True)
    return obs

me_obs = summarize(me_fire, "MONTH-END long (down-move into ME)")
ctrl_obs = summarize(ctrl_fire, "RANDOM-DAY control (same big-down, NOT month-end)")

# month-end EXCESS (falsifier 1) — is the flow real (ME beats the matched random-day control)?
print(f"\nMONTH-END EXCESS drift = {me_obs['fwd_drift_atr'].mean() - ctrl_obs['fwd_drift_atr'].mean():+.3f} ATR "
      f"(ME {me_obs['fwd_drift_atr'].mean():+.3f} vs control {ctrl_obs['fwd_drift_atr'].mean():+.3f})", flush=True)

# per-year, with focus on the 2014/15/16 block (falsifier 2)
print("\nper-year MONTH-END long (cap / drift):", flush=True)
for y, g in me_obs.groupby("year"):
    if y > 2020:
        continue
    blk = "  <-- TARGET BLOCK" if y in (2014, 2015, 2016) else ""
    print(f"  {y}: n {len(g):3d}  cap {g['capture'].mean():.4f}  drift {g['fwd_drift_atr'].mean():+.3f}{blk}",
          flush=True)
blk = me_obs[me_obs["year"].isin([2014, 2015, 2016])]
print(f"\n  2014-16 block: n {len(blk)}  cap {blk['capture'].mean():.4f}  "
      f"drift {blk['fwd_drift_atr'].mean():+.3f}", flush=True)
# per-pair within block (noise/pair-mix check)
print("  per-pair 2014-16 drift:", flush=True)
for p, sub in blk.groupby("pair"):
    print(f"    {p}: n {len(sub):3d} cap {sub['capture'].mean():.4f} drift {sub['fwd_drift_atr'].mean():+.3f}",
          flush=True)
print("\nDONE.", flush=True)
