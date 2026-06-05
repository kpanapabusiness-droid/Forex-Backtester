"""Arc 1022 observation — Failed-Breakdown RECLAIM long across UNIVERSES.

Idea + because. fbr (arc 1013) is the corpus's strongest edge but is -2018 on USD majors
(arc 2014: in strong-USD 2018 the failed breakdown becomes a REAL breakdown, reclaim fails —
18/19 trades -1R). That 2018-wipeout is plausibly USD-MAJOR-SPECIFIC: on JPY / non-USD crosses,
2018's risk-off spikes are mean-reverting (no persistent USD trend), so the stop-run-reclaim
reversal could survive 2018 there. The 4-way book (arc 1020) is blocked at 2015 & 2016; a fbr-class
component that is +2018 AND +2016 (fbr is already +2016 on majors) on a DECORRELATED universe would
be the regime-orthogonal 5th leg. The "universe lever" is proven productive (gap-fill works on JPY
crosses, NOT majors — arc 1006/2001).

OBSERVATION ONLY (gross, take-the-loss capture + fwd drift; NOT a gate). All on IS (2010-2020).
The binding portfolio folds 2016 & 2018 are both IS years, so OOS is never touched here.

Tests per universe:
  1. fbr capture (+1R-before-SL, long) + fwd-drift in ATR (the signal).
  2. STRUCTURE CONTROL (arc-1013 load-bearing check): same deep lower-wick (shadow>=1.25 ATR)
     but NOT at a swept swing low. AT-swept must beat ELSEWHERE or the structure isn't load-bearing.
  3. Per-year (esp. 2016 & 2018 acceptance) and per-pair sign.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.sim.panel import Panel
from discovery.tools.trend_entry_signals import _atr_shift1_mid
from discovery.tools.observe_long_capture import observe_long_capture

K = 40
SHADOW = 1.25
IS_START, IS_END = pd.Timestamp("2010-01-01", tz="UTC"), pd.Timestamp("2020-12-31", tz="UTC")

UNIVERSES = {
    "MAJORS (control)": ["AUDUSD", "EURUSD", "GBPUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY"],
    "JPY-crosses": ["AUDJPY", "CADJPY", "CHFJPY", "EURJPY", "GBPJPY", "NZDJPY"],
    "non-USD-crosses": ["EURGBP", "EURAUD", "EURCHF", "GBPCHF", "GBPAUD", "AUDNZD",
                        "AUDCAD", "EURCAD", "NZDCAD"],
}


def build_masks(df):
    """Return (fbr_fire, wick_elsewhere) ex-ante masks aligned to df, per fbr signal logic."""
    n = len(df)
    idx = df.index
    low_bid = df["low_bid"].to_numpy(float)
    open_mid = (df["open_bid"].to_numpy(float) + df["open_ask"].to_numpy(float)) / 2.0
    close_mid = (df["close_bid"].to_numpy(float) + df["close_ask"].to_numpy(float)) / 2.0
    atr = _atr_shift1_mid(df, 14)
    prior_low = pd.Series(low_bid, index=idx).shift(1).rolling(K).min().to_numpy(float)
    with np.errstate(invalid="ignore"):
        shadow = (np.minimum(open_mid, close_mid) - low_bid) / atr
    finite = np.isfinite(atr) & (atr > 0) & np.isfinite(prior_low)
    swept = low_bid < prior_low
    fbr = swept & (close_mid > prior_low) & (shadow >= SHADOW) & finite
    # structure control: deep wick (same shadow gate) but NOT a swept-low reclaim
    elsewhere = (shadow >= SHADOW) & (~swept) & finite
    return fbr, elsewhere


def restrict_in_is(df, mask):
    in_is = np.asarray(df.index >= IS_START) & np.asarray(df.index <= IS_END)
    return mask & in_is


def summarize(obs, label):
    obs = obs[np.isfinite(obs["fwd_drift_atr"])]
    n = len(obs)
    cap = obs["capture"].mean() if n else float("nan")
    dmean = obs["fwd_drift_atr"].mean() if n else float("nan")
    dmed = obs["fwd_drift_atr"].median() if n else float("nan")
    print(f"  {label:28s} n={n:5d}  cap={cap:.4f}  drift_mean={dmean:+.4f}  median={dmed:+.4f}")
    return obs


for uni_name, pairs in UNIVERSES.items():
    print(f"\n{'='*78}\nUNIVERSE: {uni_name}  ({len(pairs)} pairs)\n{'='*78}")
    panel = Panel.from_pairs(
        pairs, tf="H4", histdata_root=r"C:\Users\panap\histdata_backup",
        cache_root="data/cache", boundary_convention="5ers_eet",
    )
    fbr_restrict, elsewhere_restrict = {}, {}
    for pair in pairs:
        df = panel.pair_dfs[pair]
        fbr, elsewhere = build_masks(df)
        fbr_restrict[pair] = restrict_in_is(df, fbr)
        elsewhere_restrict[pair] = restrict_in_is(df, elsewhere)

    # drift_bars=12 (~2 days at H4) — the reclaim-bounce horizon
    sig = observe_long_capture(panel, sl_mult=2.0, hold=120, drift_bars=12,
                               restrict=fbr_restrict, direction="long")
    ctrl = observe_long_capture(panel, sl_mult=2.0, hold=120, drift_bars=12,
                                restrict=elsewhere_restrict, direction="long")
    sig = summarize(sig, "fbr AT-swept-low (signal)")
    ctrl = summarize(ctrl, "deep-wick ELSEWHERE (ctrl)")
    print(f"  >>> STRUCTURE EXCESS (signal cap - ctrl cap) = {sig['capture'].mean() - ctrl['capture'].mean():+.4f}"
          f" ; drift excess = {sig['fwd_drift_atr'].mean() - ctrl['fwd_drift_atr'].mean():+.4f} ATR")

    if len(sig):
        sig = sig.copy()
        sig["year"] = sig["signal_time"].dt.year
        by = sig.groupby("year").agg(n=("capture", "size"), cap=("capture", "mean"),
                                     drift=("fwd_drift_atr", "mean"), med=("fwd_drift_atr", "median"))
        print("  --- by YEAR (2016 & 2018 acceptance; cap>0.50 & drift>0 = good) ---")
        print(by.to_string(float_format=lambda x: f"{x:+.3f}").replace("\n", "\n  "))
        sig["pair"] = sig["pair"].astype(str)
        bp = sig.groupby("pair").agg(n=("capture", "size"), cap=("capture", "mean"),
                                     drift=("fwd_drift_atr", "mean"))
        n_pos = (bp["cap"] > 0.50).sum()
        print(f"  --- per-pair cap>0.50: {n_pos}/{len(bp)} ---")
        print(bp.to_string(float_format=lambda x: f"{x:+.3f}").replace("\n", "\n  "))
