"""arc 3020 — fbr × cross-pair BREADTH gate (target fbr's 2018 hole).

Observation-only cheap-kill. Reproduce arc-1013 fbr (K40/shadow1.25, 7 USD majors, H4),
then for each fbr fire compute BREADTH = how many OTHER USD majors fired an fbr within a
+/- window. Hypothesis: in strong-USD risk-off (2018) failed breakdowns cluster broadly
(many pairs fire together -> reclaims fail = real breakdowns); in normal regimes fbr fires
idiosyncratically (1-2 pairs) and reclaims hold. A LOW-breadth filter might drop 2018's
clustered false-reclaims while keeping the good-fold idiosyncratic winners -> a possible
solo-PASS. Distinct entry-time axis from 2014 (per-pair downtrend), 2020 (M1), 1025 (depth),
3013 (level). CHARACTERIZATION ONLY (gross, take-the-loss capture + drift); not a gate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.sim.panel import Panel
from discovery.tools.failed_breakdown_signals import FailedBreakdownReclaimLongSignal
from discovery.tools.observe_long_capture import observe_long_capture

PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCAD", "USDCHF"]

panel = Panel.from_pairs(
    PAIRS, tf="H4",
    histdata_root=r"C:\Users\panap\histdata_backup",
    cache_root="data/cache", boundary_convention="5ers_eet",
)

sig = FailedBreakdownReclaimLongSignal(swing_lookback=40, min_shadow_atr=1.25)
ev = sig.evaluate({"H4": panel})

# fire masks per pair (restrict obs to fires)
restrict = {p: ev.per_pair[p].signal_mask.to_numpy(bool) for p in PAIRS}

# IS window only (2010-2020) for development; keep 2021+ out of the screen
obs = observe_long_capture(panel, sl_mult=2.0, hold=120, drift_bars=24,
                           restrict=restrict, direction="long")
obs["year"] = obs["signal_time"].dt.year
obs = obs[(obs["year"] >= 2010) & (obs["year"] <= 2020)].copy()
print(f"total IS fbr fires: {len(obs)}  (arc-1013 ref ~237)")
print("unconditional capture:", round(obs["capture"].mean(), 4),
      "drift mean:", round(obs["fwd_drift_atr"].mean(), 4),
      "med:", round(obs["fwd_drift_atr"].median(), 4))

# ---- BREADTH: for each fire, count OTHER pairs firing within +/- window ----
# build a long table of all fire timestamps
fires = obs[["pair", "signal_time"]].copy().sort_values("signal_time").reset_index(drop=True)
all_ts = fires["signal_time"].to_numpy()
all_pair = fires["pair"].to_numpy()

def breadth_for(window_days: float) -> np.ndarray:
    w = pd.Timedelta(days=window_days)
    out = np.zeros(len(fires), dtype=int)
    ts = pd.to_datetime(all_ts)
    for i in range(len(fires)):
        lo, hi = ts[i] - w, ts[i] + w
        m = (ts >= lo) & (ts <= hi) & (all_pair != all_pair[i])
        out[i] = int(m.sum())
    return out

for wd in (0.17, 1.0, 3.0, 7.0):  # ~same-H4-bar(4h), 1d, 3d, 1wk
    fires[f"breadth_{wd}"] = breadth_for(wd)

# join breadth back onto obs (by pair+signal_time)
fires_idx = fires.set_index(["pair", "signal_time"])
obs = obs.set_index(["pair", "signal_time"])
for wd in (0.17, 1.0, 3.0, 7.0):
    obs[f"breadth_{wd}"] = fires_idx[f"breadth_{wd}"]
obs = obs.reset_index()

print("\n=== breadth distribution (other pairs firing in window) ===")
for wd in (0.17, 1.0, 3.0, 7.0):
    col = f"breadth_{wd}"
    print(f"window +/-{wd}d: mean={obs[col].mean():.2f}  "
          f"share>=1: {(obs[col]>=1).mean():.2f}  share>=2: {(obs[col]>=2).mean():.2f}")

# pick the 3-day window as the primary regime-clustering proxy
W = "breadth_3.0"
obs["breadth_bin"] = np.where(obs[W] == 0, "solo",
                       np.where(obs[W] <= 1, "low(1)", "high(>=2)"))

print(f"\n=== capture/drift by breadth bin ({W}) ===")
g = obs.groupby("breadth_bin").agg(
    n=("capture", "size"), cap=("capture", "mean"),
    drift=("fwd_drift_atr", "mean"), drift_med=("fwd_drift_atr", "median"))
print(g.round(4))

print("\n=== 2018 vs non-2018: breadth + outcome ===")
obs["is2018"] = obs["year"] == 2018
g2 = obs.groupby("is2018").agg(
    n=("capture", "size"), cap=("capture", "mean"),
    drift=("fwd_drift_atr", "mean"), breadth=(W, "mean"),
    share_high=(W, lambda s: (s >= 2).mean()))
print(g2.round(4))

print("\n=== 2018 outcome BY breadth bin (does low-breadth save 2018?) ===")
g3 = obs[obs.is2018].groupby("breadth_bin").agg(
    n=("capture", "size"), cap=("capture", "mean"), drift=("fwd_drift_atr", "mean"))
print(g3.round(4))

# ---- the decisive test: per-year drift if we KEEP only solo/low-breadth fires ----
print("\n=== per-year drift: ALL fires vs LOW-breadth-only (breadth<=1, i.e. drop high) ===")
keep = obs[W] <= 1
by_all = obs.groupby("year").agg(n=("capture","size"), drift=("fwd_drift_atr","mean"),
                                 cap=("capture","mean"))
by_keep = obs[keep].groupby("year").agg(n=("capture","size"), drift=("fwd_drift_atr","mean"),
                                        cap=("capture","mean"))
merged = by_all.join(by_keep, lsuffix="_all", rsuffix="_keep")
print(merged.round(4))
print("\nALL: years drift>0:", int((by_all["drift"] > 0).sum()), "/", len(by_all))
print("LOW-breadth-only: years drift>0:", int((by_keep["drift"] > 0).sum()), "/", len(by_keep),
      " | fires kept:", int(keep.sum()), "/", len(obs))

print("\n=== SOLO-ONLY (breadth_3.0 == 0) per-year (secondary finding check) ===")
solo = obs[obs[W] == 0]
by_solo = solo.groupby("year").agg(n=("capture","size"), drift=("fwd_drift_atr","mean"),
                                   drift_med=("fwd_drift_atr","median"), cap=("capture","mean"))
print(by_solo.round(4))
print("SOLO-only: years drift>0:", int((by_solo["drift"]>0).sum()), "/", len(by_solo),
      "| years cap>0.50:", int((by_solo["cap"]>0.50).sum()), "/", len(by_solo),
      "| fires:", len(solo), "/", len(obs))
print("SOLO 2018: n=%d cap=%.3f drift=%.3f" % (
    len(solo[solo.year==2018]), solo[solo.year==2018]["capture"].mean(),
    solo[solo.year==2018]["fwd_drift_atr"].mean()))

# robustness: solo per-pair (is the solo edge broad or single-pair?)
print("\n=== SOLO per-pair drift ===")
print(solo.groupby("pair").agg(n=("capture","size"), drift=("fwd_drift_atr","mean"),
                               cap=("capture","mean")).round(3))
