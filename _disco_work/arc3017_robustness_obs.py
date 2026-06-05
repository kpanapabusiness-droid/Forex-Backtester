"""Arc 3017 robustness obs — threshold sweep + leave-one-pair-out + outlier-pair drop.

Guards against the corpus's known false-positive tells (arc 2011/3011/3012): thin-tail
(mean>>median), single-pair carry, threshold-fragility.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.sim.panel import Panel
from core.features._helpers import mid_close, mid_high, mid_low, wilder_atr
from discovery.tools.observe_long_capture import observe_long_capture

PAIRS = ["AUDUSD", "EURUSD", "GBPUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY"]
INTO_BARS = 2
IS_START, IS_END = pd.Timestamp("2010-01-01", tz="UTC"), pd.Timestamp("2020-12-31", tz="UTC")

panel = Panel.from_pairs(PAIRS, tf="D1", histdata_root=r"C:\Users\panap\histdata_backup",
                         cache_root="data/cache", boundary_convention="5ers_eet")

cache = {}
for pair in PAIRS:
    df = panel.pair_dfs[pair]
    n = len(df); idx = df.index
    close_mid = (df["close_bid"].to_numpy(float) + df["close_ask"].to_numpy(float)) / 2.0
    atr = wilder_atr(mid_high(df), mid_low(df), mid_close(df), 14).shift(1).to_numpy(float)
    ym = pd.PeriodIndex(idx, freq="M")
    is_last = np.zeros(n, dtype=bool)
    if n >= 2:
        is_last[:-1] = ym[1:] != ym[:-1]
    into = np.full(n, np.nan)
    k = INTO_BARS
    with np.errstate(invalid="ignore", divide="ignore"):
        into[k:] = (close_mid[k:] - close_mid[:-k]) / atr[k:]
    in_is = np.asarray(idx >= IS_START) & np.asarray(idx <= IS_END)
    cache[pair] = (is_last, into, atr, in_is)


def me_restrict(thr, drop=None):
    r = {}
    for pair in PAIRS:
        if drop and pair in drop:
            r[pair] = np.zeros(len(panel.pair_dfs[pair]), dtype=bool); continue
        is_last, into, atr, in_is = cache[pair]
        big_up = np.isfinite(into) & np.isfinite(atr) & (atr > 0) & (into >= thr) & in_is
        r[pair] = big_up & is_last
    return r


def stat(restrict):
    obs = observe_long_capture(panel, sl_mult=2.0, hold=120, drift_bars=INTO_BARS,
                               restrict=restrict, direction="short")
    obs = obs[np.isfinite(obs["fwd_drift_atr"])]
    return len(obs), obs["capture"].mean(), obs["fwd_drift_atr"].mean(), obs["fwd_drift_atr"].median()


print("=== THRESHOLD SWEEP (month-end big-UP short) ===")
for thr in (0.75, 1.0, 1.25, 1.5):
    n, c, m, med = stat(me_restrict(thr))
    print(f"  thr {thr}: n={n:4d} cap={c:.4f} drift_mean={m:+.4f} median={med:+.4f}")

print("\n=== LEAVE-ONE-PAIR-OUT (thr 1.0) ===")
for drop in PAIRS:
    n, c, m, med = stat(me_restrict(1.0, drop={drop}))
    print(f"  drop {drop}: n={n:4d} cap={c:.4f} drift_mean={m:+.4f} median={med:+.4f}")

print("\n=== DROP TOP-2 outlier pairs GBPUSD+USDJPY (arc-2011 discipline; thr 1.0) ===")
n, c, m, med = stat(me_restrict(1.0, drop={"GBPUSD", "USDJPY"}))
print(f"  n={n} cap={c:.4f} drift_mean={m:+.4f} median={med:+.4f}")
