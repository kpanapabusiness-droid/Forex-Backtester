"""Arc 1028 — Q1: central-bank PEG / boundary-defense persistence (the long-shot).

A price-insensitive actor DEFENDING a band is a forced, repeated counterparty that pins price
to one side -> fade touches of the defended boundary for a high-win-rate, THICK edge (the
thickness the survivor book lacks). The one clean instance in the corpus: SNB EURCHF 1.20
floor (2011-09 .. 2015-01-14), in-IS, H4 cached.

Cheap OBSERVATION (no pool/engine):
  1. Does fading the floor INSIDE the regime show the predicted asymmetry (high capture /
     positive drift) vs the SAME rule OUTSIDE the regime (coin-flip control)?
  2. Is the regime causally detectable from OHLC alone (vol-collapse + truncated distribution)?
  3. Characterize the Jan-2015 break tail (the un-hedgeable gap-through-stop risk).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.sim.panel import Panel
from core.features._helpers import mid_close, mid_high, mid_low, wilder_atr
from discovery.tools.observe_long_capture import observe_long_capture

HIST = r"C:\Users\panap\histdata_backup"
CACHE = "data/cache"
PAIR = "EURCHF"
# SNB floor regime (hindsight label, for the mechanism test): floor announced 2011-09-06,
# broken 2015-01-15. Use the rock-solid pinned window for the in-regime test.
REGIME_START = pd.Timestamp("2011-09-07", tz="UTC")
REGIME_END = pd.Timestamp("2015-01-14", tz="UTC")


def main():
    panel = Panel.from_pairs([PAIR], tf="H4", histdata_root=HIST, cache_root=CACHE,
                             boundary_convention="5ers_eet")
    df = panel.pair_dfs[PAIR]
    idx = df.index
    mc = mid_close(df)
    ml = mid_low(df)
    atr = wilder_atr(mid_high(df), ml, mc, 14).shift(1)

    # causal established floor = rolling min of mid-low over a long window, shifted 1
    floor = ml.rolling(250, min_periods=100).min().shift(1)
    dist_to_floor_atr = (ml - floor) / atr  # how close the bar dipped to the floor (in ATR)

    # near-floor touch = the bar dipped within touch_atr of the established floor
    TOUCH = 0.5
    near_floor = (dist_to_floor_atr <= TOUCH) & np.isfinite(dist_to_floor_atr.values)

    # causal vol-collapse regime detector: rolling realized vol below its trailing percentile
    ret = np.log(mc).diff()
    rvol = ret.rolling(120).std()  # ~20-day realized vol
    rvol_pct = rvol.rolling(750, min_periods=250).apply(lambda x: (x[-1] <= x).mean(), raw=True).shift(1)
    vol_collapsed = (rvol_pct <= 0.10)  # bottom-decile vol = candidate boundary regime

    in_regime = (idx >= REGIME_START) & (idx <= REGIME_END)
    in_regime = pd.Series(in_regime, index=idx)

    # honest capture (+1R-before-SL) + 24-bar fwd drift, restricted to near-floor bars
    obs = observe_long_capture(panel, sl_mult=2.0, hold=120, drift_bars=24, warmup=250,
                               restrict={PAIR: near_floor.values})
    obs = obs.set_index("signal_time")
    obs["in_regime"] = in_regime.reindex(obs.index).values
    obs["vol_collapsed"] = vol_collapsed.reindex(obs.index).fillna(False).values
    obs["dist"] = dist_to_floor_atr.reindex(obs.index).values

    print(f"=== Q1 EURCHF floor-fade (near-floor touches, dist<= {TOUCH} ATR) ===")
    print(f"total near-floor bars: {len(obs)}  base capture (all near-floor): {obs['capture'].mean():.4f}")
    print("\n-- HINDSIGHT regime label (mechanism test) --")
    g = obs.groupby("in_regime").agg(n=("capture", "size"), capture=("capture", "mean"),
                                     fwd_drift=("fwd_drift_atr", "mean"),
                                     fwd_drift_med=("fwd_drift_atr", "median"))
    print(g.to_string())

    print("\n-- CAUSAL vol-collapse detector (in vs out of bottom-decile vol) --")
    g2 = obs.groupby("vol_collapsed").agg(n=("capture", "size"), capture=("capture", "mean"),
                                          fwd_drift=("fwd_drift_atr", "mean"))
    print(g2.to_string())
    # how well does the causal detector overlap the hindsight regime?
    det = vol_collapsed.reindex(idx).fillna(False)
    print(f"\ncausal-detector firing years (vol_collapsed True count by year):")
    print(det.groupby(det.index.year).sum().to_string())

    # cost in R at this regime's ATR: spread/(2*ATR)
    spread_bp = (df["spread_close"] / mc * 1e4)
    reg_atr = atr[in_regime.values & np.isfinite(atr.values)]
    reg_spread = spread_bp[in_regime.values]
    print(f"\n-- cost reality (in-regime) --")
    print(f"  median ATR (price): {reg_atr.median():.5f}   median spread: {reg_spread.median():.2f} bp")
    # cost in R = spread_price / (sl_mult*ATR); spread_price = spread_bp/1e4 * price
    sp_price = (reg_spread / 1e4 * mc[in_regime.values]).median()
    print(f"  approx cost in R (spread/(2*ATR)): {sp_price/(2*reg_atr.median()):.4f} R")

    # the break tail
    print(f"\n-- Jan-2015 BREAK tail --")
    brk = mc[(idx >= pd.Timestamp('2015-01-14', tz='UTC')) & (idx <= pd.Timestamp('2015-01-16', tz='UTC'))]
    print(brk.to_string())
    pre = float(mc[idx <= pd.Timestamp('2015-01-14 21:00', tz='UTC')].iloc[-1])
    post = float(brk.min())
    print(f"  pre-break ~{pre:.4f} -> intraday low {post:.4f} = {(post/pre-1)*100:.1f}% gap "
          f"(~{(pre-post)/reg_atr.median():.0f} ATR through any stop)")


if __name__ == "__main__":
    main()
