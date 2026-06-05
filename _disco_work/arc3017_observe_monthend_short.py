"""Arc 3017 observation — Month-End Reversion SHORT (mirror of arc 1011 long).

Mechanism (because): WMR/index month-end rebalancing is INELASTIC, DIRECTION-SYMMETRIC
mechanical flow. Arc 1011 harvested only the LONG side (buy big DOWN move into month-end ->
revert UP), which is NEGATIVE in the strong-USD block 2014/15/16 (EURUSD-type down-moves
CONTINUE down). The unharvested SHORT side (sell big UP move into month-end -> revert DOWN)
should fire on USDXXX pairs in strong-USD years -> candidate 2015 & 2018-positive 4th leg.

Tests, all OBSERVATION ONLY (gross, not a gate):
  1. Capture + fwd-drift of the month-end big-UP-move SHORT (direction-aware harness).
  2. STRUCTURE CONTROL: same big-UP move on RANDOM non-month-end days (short). Generic big-UP
     should CONTINUE up (negative for a short) per arcs 3000/3001; month-end excess = mechanism.
  3. Per-year (esp. 2015 & 2018 acceptance test) and per-pair sign.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.sim.panel import Panel
from core.features._helpers import mid_close, mid_high, mid_low, wilder_atr
from discovery.tools.observe_long_capture import observe_long_capture

PAIRS = ["AUDUSD", "EURUSD", "GBPUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY"]
INTO_BARS = 2
THRESH = 1.0  # |up-move into month-end| in ATR
IS_START, IS_END = pd.Timestamp("2010-01-01", tz="UTC"), pd.Timestamp("2020-12-31", tz="UTC")

panel = Panel.from_pairs(
    PAIRS, tf="D1",
    histdata_root=r"C:\Users\panap\histdata_backup",
    cache_root="data/cache", boundary_convention="5ers_eet",
)


def build_masks(df):
    """Return (is_last_month_end, into_move_atr) aligned to df, ex-ante (arc 1011 convention)."""
    n = len(df)
    idx = df.index
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
    return is_last, into, atr


# Build restrict masks: month-end big-UP, and random-day big-UP (same magnitude).
me_restrict, rand_restrict = {}, {}
for pair in PAIRS:
    df = panel.pair_dfs[pair]
    is_last, into, atr = build_masks(df)
    in_is = np.asarray(df.index >= IS_START) & np.asarray(df.index <= IS_END)
    big_up = np.isfinite(into) & np.isfinite(atr) & (atr > 0) & (into >= THRESH) & in_is
    me_restrict[pair] = big_up & is_last
    rand_restrict[pair] = big_up & ~is_last


def summarize(obs, label):
    obs = obs[np.isfinite(obs["fwd_drift_atr"])]
    n = len(obs)
    cap = obs["capture"].mean() if n else float("nan")
    dmean = obs["fwd_drift_atr"].mean() if n else float("nan")
    dmed = obs["fwd_drift_atr"].median() if n else float("nan")
    fpos = (obs["fwd_drift_atr"] > 0).mean() if n else float("nan")
    print(f"\n=== {label} ===")
    print(f"  n={n}  capture(+1R-before-SL, short)={cap:.4f}  "
          f"fwd{INTO_BARS}drift_mean={dmean:+.4f}  median={dmed:+.4f}  frac_pos={fpos:.3f}")
    return obs


# direction-aware short observation; drift_bars=INTO_BARS (the 2-day reversion, short-signed)
me = observe_long_capture(panel, sl_mult=2.0, hold=120, drift_bars=INTO_BARS,
                          restrict=me_restrict, direction="short")
rd = observe_long_capture(panel, sl_mult=2.0, hold=120, drift_bars=INTO_BARS,
                          restrict=rand_restrict, direction="short")

me = summarize(me, "MONTH-END big-UP SHORT (the signal)")
rd = summarize(rd, "RANDOM-DAY big-UP SHORT (structure control)")
print(f"\n  >>> MONTH-END EXCESS drift = {me['fwd_drift_atr'].mean() - rd['fwd_drift_atr'].mean():+.4f} ATR")
print(f"      (positive => month-end short reverts more than generic big-up continues; mechanism load-bearing)")

# Per-year acceptance test (2015 & 2018 are the binding portfolio folds)
me["year"] = me["signal_time"].dt.year
print("\n=== MONTH-END SHORT by YEAR (drift short-signed; >0 good for short) ===")
by = me.groupby("year").agg(n=("capture", "size"), cap=("capture", "mean"),
                            drift=("fwd_drift_atr", "mean"), med=("fwd_drift_atr", "median"))
print(by.to_string(float_format=lambda x: f"{x:+.3f}"))

# Per-pair sign
me["pair"] = me["pair"].astype(str)
print("\n=== MONTH-END SHORT by PAIR ===")
bp = me.groupby("pair").agg(n=("capture", "size"), cap=("capture", "mean"),
                            drift=("fwd_drift_atr", "mean"), med=("fwd_drift_atr", "median"))
print(bp.to_string(float_format=lambda x: f"{x:+.3f}"))
n_pos = (bp["drift"] > 0).sum()
print(f"\n  per-pair positive: {n_pos}/{len(bp)}")
