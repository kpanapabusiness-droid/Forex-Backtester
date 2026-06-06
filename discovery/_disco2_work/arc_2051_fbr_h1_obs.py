"""arc 2051 OBSERVATION (cheap-kill) — fbr (failed-breakdown reclaim) on H1: does the crown-jewel
structural edge survive a FINER timeframe, and does the ~4x thickness help (arc-2017 option-B)?

because: the only NON-operator-gated path to an AFP book is arc-2017's option-B — a component THICK
enough that its year-folds RESOLVE (the 4-way book's AFP failure is a NOISE FLOOR: thin components have
year-fold sampling-sd >= mean, arcs 1023/2016). fbr (1013) is the corpus's ONE robust structural edge
(H4 USD majors, capture 0.582 vs base 0.4877, control coin-flip) but was ONLY EVER run on H4 (~thin,
~210 IS trades). The swing-low stop-run mechanism is SCALE-FREE (sell-stops cluster below visible H1
swing lows too) -> fbr on H1 should fire ~4x as often -> thicker. If the capture edge SURVIVES on H1
(beats the H1 baseline 0.431 [arc 2050] AND a structure control), the extra thickness is the explicit
option-B lever -> engine + §5f. If it erodes to coin-flip/control -> KILL.

Cost-skeptic prior: H1 displacements are smaller -> the edge may erode toward cost on the finer TF
(the corpus's repeated intraday finding). Falsifiers (-> KILL): H1 fbr capture ~ baseline / ~ control
(structure not load-bearing on H1) / negative drift.

H1, 4 cached USD majors (EURUSD/GBPUSD/AUDUSD/USDJPY — fbr's universe; 3 of 7 not H1-cached, noted).
K in {40 (fast/thick ~1.7d), 160 (horizon-matched to H4 K40 ~6.7d)}, shadow>=1.25 (fbr canonical).
IS 2010-2020 only (OOS preserved).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.sim.panel import Panel
from core.features._helpers import mid_close
from discovery.tools.trend_entry_signals import _atr_shift1_mid
from discovery.tools.observe_long_capture import observe_long_capture

PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "USDJPY"]
BACKUP = r"C:\Users\panap\histdata_backup"
SHADOW = 1.25
HOLD = 120
DRIFT = 24

print(f"loading H1 panel: {PAIRS}", flush=True)
panel = Panel.from_pairs(PAIRS, tf="H1", histdata_root=BACKUP, cache_root="data/cache",
                         boundary_convention="5ers_eet")


def fbr_masks(df, K, shadow):
    """Replicate the BUILT FailedBreakdownReclaimLongSignal fire logic + a matched structure control."""
    idx = df.index
    low_bid = df["low_bid"].to_numpy(float)
    open_mid = (df["open_bid"].to_numpy(float) + df["open_ask"].to_numpy(float)) / 2.0
    close_mid = (df["close_bid"].to_numpy(float) + df["close_ask"].to_numpy(float)) / 2.0
    atr = _atr_shift1_mid(df, 14)
    prior_low = pd.Series(low_bid, index=idx).shift(1).rolling(K).min().to_numpy(float)
    with np.errstate(invalid="ignore"):
        pierced = low_bid < prior_low                       # swept stops below the swing low
        reclaim = close_mid > prior_low                     # failed breakdown (closed back above)
        shadow_atr = (np.minimum(open_mid, close_mid) - low_bid) / atr
        deep = shadow_atr >= shadow                         # deep rejection grab
        ok = np.isfinite(atr) & (atr > 0)
    fire = pierced & reclaim & deep & ok
    # structure control (fbr 1013 discipline): SAME deep lower-rejection wick but NOT at a swept swing
    # low (no pierce) -> isolates whether the swing-low STRUCTURE is load-bearing vs a generic wick.
    ctrl = (~pierced) & deep & ok
    return fire, ctrl


def summarize(restrict, label):
    obs = observe_long_capture(panel, sl_mult=2.0, hold=HOLD, drift_bars=DRIFT,
                               restrict=restrict, direction="long")
    obs["year"] = obs["signal_time"].dt.year
    obs = obs[(obs["year"] >= 2010) & (obs["year"] <= 2020)].copy()
    print(f"\n=== {label}: n={len(obs)}  capture {obs['capture'].mean():.4f}  "
          f"drift {obs['fwd_drift_atr'].mean():+.4f} ATR ===", flush=True)
    return obs


# H1 baseline reference (arc 2050 found ~0.431 over 7 pairs; recompute on these 4)
base = observe_long_capture(panel, sl_mult=2.0, hold=HOLD, drift_bars=DRIFT, direction="long")
base = base[(base["signal_time"].dt.year >= 2010) & (base["signal_time"].dt.year <= 2020)]
print(f"\nH1 baseline (4 USD majors): n={len(base)} capture {base['capture'].mean():.4f} "
      f"drift {base['fwd_drift_atr'].mean():+.4f}", flush=True)

for K in (40, 160):
    fire, ctrl = {}, {}
    for p in PAIRS:
        f, c = fbr_masks(panel.pair_dfs[p], K, SHADOW)
        fire[p], ctrl[p] = f, c
    print(f"\n################ K={K} (shadow>={SHADOW}) ################", flush=True)
    fobs = summarize(fire, f"fbr H1 K{K}")
    cobs = summarize(ctrl, f"structure control K{K} (deep wick, NO swing-low pierce)")
    print(f"\n[load-bearing?] fbr cap {fobs['capture'].mean():.4f} vs control {cobs['capture'].mean():.4f} "
          f"vs baseline {base['capture'].mean():.4f}  | fbr drift {fobs['fwd_drift_atr'].mean():+.4f} "
          f"vs control {cobs['fwd_drift_atr'].mean():+.4f}", flush=True)
    print("per-year fbr (cap / drift / n):", flush=True)
    for y, g in fobs.groupby("year"):
        star = "  <-- strong-USD" if y in (2014, 2015, 2018) else ""
        print(f"  {y}: n {len(g):5d}  cap {g['capture'].mean():.4f}  drift {g['fwd_drift_atr'].mean():+.4f}{star}",
              flush=True)
    print("per-pair fbr (cap / drift / n):", flush=True)
    for p, sub in fobs.groupby("pair"):
        print(f"  {p}: n {len(sub):5d}  cap {sub['capture'].mean():.4f}  drift {sub['fwd_drift_atr'].mean():+.4f}",
              flush=True)

print("\nDONE.", flush=True)
