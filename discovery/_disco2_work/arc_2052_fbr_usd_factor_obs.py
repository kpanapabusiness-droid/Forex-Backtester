"""arc 2052 OBSERVATION (cheap-kill / diagnostic) — fbr × cross-sectional USD-FACTOR-trend conditioner.

because: fbr (1013) is a PAIR-SPECIFIC stop-run reclaim. arc 2014 found its 2018 failure is a near-uniform
-1R wipeout across ALL 7 USD majors SIMULTANEOUSLY, and entry-time-unconditionable on SINGLE-PAIR axes
(even the "best" per-pair balanced context wipes out in 2018). That simultaneity is the tell of an
UNTESTED axis: are the 2018 fires USD-FACTOR-WIDE (the whole dollar trending → an INFORMATIONAL breakdown,
reclaim fails) while good-year fires are PAIR-IDIOSYNCRATIC (a real stop-grab, USD calm → reclaim holds)?
arc 2014 tested per-pair trend; the cross-sectional USD-FACTOR trend (the common factor itself) was never
tested on fbr. Condition fbr on the move being IDIOSYNCRATIC (broad USD NOT trending in the aligned
direction) → keep good-year grabs, drop the 2018 USD-factor falling-knives.

FALSIFIERS (§5d): (1) 2018 fires are NOT more USD-factor-aligned than good-year fires (simultaneity
coincidental) → can't separate; (2) the idiosyncrasy bucket does NOT lift 2018 (kept idiosyncratic 2018
fires still wipe out) → mechanism-intrinsic confirmed; (3) no-free-lunch (the filter destroys the good
folds, like every 2014 gate). Any → KILL. A clean separation (low-USD-trend fbr all-folds-pos incl 2018)
→ engine + §5f.

H4, 7 USD majors, K=40, shadow>=1.25 (fbr canonical), IS 2010-2020 (OOS preserved).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.sim.panel import Panel
from discovery.tools.trend_entry_signals import _atr_shift1_mid
from discovery.tools.observe_long_capture import observe_long_capture

MAJORS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCAD", "USDCHF"]
SGN_USD = {"EURUSD": -1, "GBPUSD": -1, "AUDUSD": -1, "NZDUSD": -1,  # XXXUSD: pair-down = USD-up
           "USDJPY": +1, "USDCAD": +1, "USDCHF": +1}                # USDXXX: pair-down = USD-down
BACKUP = r"C:\Users\panap\histdata_backup"
K, SHADOW, L = 40, 1.25, 40

print(f"loading H4 panel: {MAJORS}", flush=True)
panel = Panel.from_pairs(MAJORS, tf="H4", histdata_root=BACKUP, cache_root="data/cache",
                         boundary_convention="5ers_eet")


def mid_close_s(df):
    return (df["close_bid"].astype(float) + df["close_ask"].astype(float)) / 2.0


# --- build the USD-FACTOR L-bar trend (signed, USD-up positive), on the common index ---
logret = {}
for p in MAJORS:
    mc = mid_close_s(panel.pair_dfs[p])
    logret[p] = np.log(mc).diff()
# USD index per-bar return = mean over pairs of sgn_usd * pair-logret (USD-up positive)
usd_ret = pd.DataFrame({p: SGN_USD[p] * logret[p] for p in MAJORS}).mean(axis=1)
usd_trend_L = usd_ret.rolling(L).sum()                       # L-bar USD move (known at bar t)
usd_vol_L = usd_ret.rolling(250).std() * np.sqrt(L)          # scale
usd_trend_z = (usd_trend_L / usd_vol_L)                       # signed z (USD-up positive)

# --- fbr fires per pair + the ALIGNED USD-factor trend at each fire ---
fire_mask, aligned_z = {}, {}
for p in MAJORS:
    df = panel.pair_dfs[p]
    idx = df.index
    low_bid = df["low_bid"].to_numpy(float)
    open_mid = (df["open_bid"].to_numpy(float) + df["open_ask"].to_numpy(float)) / 2.0
    close_mid = (df["close_bid"].to_numpy(float) + df["close_ask"].to_numpy(float)) / 2.0
    atr = _atr_shift1_mid(df, 14)
    prior_low = pd.Series(low_bid, index=idx).shift(1).rolling(K).min().to_numpy(float)
    with np.errstate(invalid="ignore"):
        pierced = low_bid < prior_low
        reclaim = close_mid > prior_low
        deep = (np.minimum(open_mid, close_mid) - low_bid) / atr >= SHADOW
        ok = np.isfinite(atr) & (atr > 0)
    fire_mask[p] = pierced & reclaim & deep & ok
    # aligned USD-trend: + means broad USD moved in the SAME direction P's down-move implies
    # (XXXUSD down -> USD up -> +usd_trend_z; USDXXX down -> USD down -> -usd_trend_z)
    z = usd_trend_z.reindex(idx).to_numpy(float)
    aligned_z[p] = (+z if SGN_USD[p] == -1 else -z)

# observe capture/drift on all fires, attach aligned_z + year
obs = observe_long_capture(panel, sl_mult=2.0, hold=120, drift_bars=24,
                           restrict=fire_mask, direction="long")
za = []
for _, r in obs.iterrows():
    p = r["pair"]
    pos = panel.pair_dfs[p].index.get_loc(r["signal_time"])
    za.append(aligned_z[p][pos])
obs["usd_aligned_z"] = za
obs["year"] = obs["signal_time"].dt.year
obs = obs[(obs["year"] >= 2010) & (obs["year"] <= 2020) & np.isfinite(obs["usd_aligned_z"])].copy()

print(f"\nfbr pooled: n={len(obs)} cap {obs['capture'].mean():.4f} drift {obs['fwd_drift_atr'].mean():+.4f}",
      flush=True)

# FALSIFIER 1 — are 2018 fires more USD-factor-aligned than good-year fires?
print("\n[FALSIFIER 1] mean aligned USD-trend-z by year (HIGH = USD-factor-wide breakdown):", flush=True)
for y, g in obs.groupby("year"):
    star = "  <-- strong-USD" if y in (2014, 2015, 2018) else ""
    print(f"  {y}: n {len(g):3d}  aligned_z {g['usd_aligned_z'].mean():+.3f}  "
          f"cap {g['capture'].mean():.4f}  drift {g['fwd_drift_atr'].mean():+.4f}{star}", flush=True)

# FALSIFIER 2 — does the edge separate by aligned-z bucket? (low = idiosyncratic = should be good)
print("\n[FALSIFIER 2] edge by aligned USD-trend-z tercile:", flush=True)
obs["zbkt"] = pd.qcut(obs["usd_aligned_z"], 3, labels=["LOW(idiosyncratic)", "MID", "HIGH(USD-wide)"])
for b, g in obs.groupby("zbkt", observed=True):
    print(f"  {b:20s}: n {len(g):4d}  cap {g['capture'].mean():.4f}  drift {g['fwd_drift_atr'].mean():+.4f}",
          flush=True)

# FALSIFIER 3 — no-free-lunch: does keeping LOW-z (aligned_z <= median) lift 2018 w/o killing good folds?
med = obs["usd_aligned_z"].median()
kept = obs[obs["usd_aligned_z"] <= med]
print(f"\n[FALSIFIER 3] keep aligned_z <= median ({med:+.3f})  -> n={len(kept)} "
      f"cap {kept['capture'].mean():.4f} drift {kept['fwd_drift_atr'].mean():+.4f}", flush=True)
print("per-year RAW fbr vs KEPT (drift):", flush=True)
for y in range(2010, 2021):
    raw = obs[obs["year"] == y]
    kp = kept[kept["year"] == y]
    star = "  <-- strong-USD" if y in (2014, 2015, 2018) else ""
    print(f"  {y}: raw n {len(raw):3d} drift {raw['fwd_drift_atr'].mean():+.4f}  |  "
          f"kept n {len(kp):3d} drift {(kp['fwd_drift_atr'].mean() if len(kp) else float('nan')):+.4f}{star}",
          flush=True)
print("\nDONE.", flush=True)
