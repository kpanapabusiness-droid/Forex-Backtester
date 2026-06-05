"""Arc 2020 observation: does M1 reclaim-QUALITY separate fbr's strong-USD-year (2018) failures?

OBSERVATION ONLY (gross, take-the-loss capture; NOT a gate). Reproduces the corpus crown-jewel
fbr signal (arc 1013: FailedBreakdownReclaimLongSignal K=40, shadow>=1.25) on the 7 USD majors,
then for every fire bar measures M1 reclaim-quality metrics WITHIN the H4 reclaim bar (no-lookahead:
all M1 data used is <= signal-bar close; entry is t+1 open). Asks:
  (Q1) Is any M1 reclaim-quality metric LOWER in 2018 than in good years? (a tell exists at entry)
  (Q2) Does M1 reclaim-quality predict the per-trade capture/drift within-sample?
If 2018 looks identical at M1 AND quality doesn't predict outcome -> no entry-time tell -> KILL
(closes the council's M1-reclaim-confirm thread; confirms arc-2014/2017 forward-failure via a 3rd lever).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.sim.panel import Panel
from discovery.tools.failed_breakdown_signals import FailedBreakdownReclaimLongSignal
from discovery.tools.observe_long_capture import observe_long_capture
from discovery.tools.trend_entry_signals import _atr_shift1_mid

PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY"]
HIST = r"C:\Users\panap\histdata_backup"
K = 40
SHADOW = 1.25

print("loading H4 panel (cached)...")
panel = Panel.from_pairs(PAIRS, tf="H4", histdata_root=HIST,
                         cache_root="data/cache", boundary_convention="5ers_eet")

sig = FailedBreakdownReclaimLongSignal(swing_lookback=K, min_shadow_atr=SHADOW)

# fire bars + swept level + atr per pair, restricted to IS window 2010-2020
fire_masks = {}
swept_by_pair = {}
atr_by_pair = {}
for pair in PAIRS:
    df = panel.pair_dfs[pair]
    idx = df.index
    low_bid = df["low_bid"].to_numpy(float)
    open_mid = (df["open_bid"].to_numpy(float) + df["open_ask"].to_numpy(float)) / 2.0
    close_mid = (df["close_bid"].to_numpy(float) + df["close_ask"].to_numpy(float)) / 2.0
    atr = _atr_shift1_mid(df, 14)
    prior_low = pd.Series(low_bid, index=idx).shift(1).rolling(K).min().to_numpy(float)
    with np.errstate(invalid="ignore"):
        shadow = (np.minimum(open_mid, close_mid) - low_bid) / atr
    fire = ((low_bid < prior_low) & (close_mid > prior_low) & (shadow >= SHADOW)
            & np.isfinite(atr) & (atr > 0) & np.isfinite(prior_low))
    yr = idx.year.to_numpy()
    fire = fire & (yr >= 2010) & (yr <= 2020)
    fire_masks[pair] = fire
    swept_by_pair[pair] = prior_low
    atr_by_pair[pair] = atr

# honest capture/drift per fire bar (restrict to fire mask)
restrict = {p: fire_masks[p] for p in PAIRS}
obs = observe_long_capture(panel, sl_mult=2.0, hold=120, drift_bars=24, restrict=restrict)
print(f"fbr fires (IS 2010-2020): {len(obs)}  (log records n_trades=172)")

# ---- M1 reclaim-quality per fire bar ----
records = []
for pair in PAIRS:
    df = panel.pair_dfs[pair]
    idx = df.index
    fire = fire_masks[pair]
    swept = swept_by_pair[pair]
    atr = atr_by_pair[pair]
    fire_pos = np.where(fire)[0]
    if len(fire_pos) == 0:
        continue
    m1 = pd.read_parquet(f"data/cache/m1/{pair}.parquet")
    m1_lowbid = m1["low_bid"]
    m1_mid = (m1["close_bid"] + m1["close_ask"]) / 2.0
    for t in fire_pos:
        bar_start = idx[t]
        bar_end = idx[t + 1] if t + 1 < len(idx) else bar_start + pd.Timedelta(hours=4)
        win = m1.loc[(m1.index >= bar_start) & (m1.index < bar_end)]
        if len(win) < 5:
            continue
        lvl = swept[t]
        a = atr[t]
        wlow = win["low_bid"].to_numpy(float)
        wmid = ((win["close_bid"] + win["close_ask"]) / 2.0).to_numpy(float)
        nrows = len(win)
        pierced = wlow < lvl                       # M1 bar poked the swept level
        n_pierce = int(pierced.sum())
        # last M1 pierce position (fraction through the 4h bar): late => fresh/weak reclaim
        last_pierce_i = int(np.where(pierced)[0].max()) if n_pierce > 0 else -1
        last_pierce_frac = (last_pierce_i / (nrows - 1)) if last_pierce_i >= 0 else 0.0
        # after the last pierce: how decisively did price hold above the swept level
        if last_pierce_i >= 0 and last_pierce_i < nrows - 1:
            after = wmid[last_pierce_i + 1:]
            hold_min_margin_atr = float((after.min() - lvl) / a)     # worst dip above level after pierce
            n_after = len(after)
        else:
            after = wmid[-1:]
            hold_min_margin_atr = float((wmid[-1] - lvl) / a)
            n_after = 0
        # decisiveness: minutes held above after last pierce / total
        time_above_after_frac = n_after / nrows
        # reclaim margin at H4-bar close (mid)
        close_margin_atr = float((wmid[-1] - lvl) / a)
        # overall above-ness across the whole bar
        frac_mid_above = float((wmid > lvl).mean())
        records.append({
            "pair": pair, "signal_time": bar_start,
            "m1_n": nrows, "m1_pierce_count": n_pierce,
            "m1_last_pierce_frac": last_pierce_frac,
            "m1_hold_min_margin_atr": hold_min_margin_atr,
            "m1_time_above_after_frac": time_above_after_frac,
            "m1_close_margin_atr": close_margin_atr,
            "m1_frac_mid_above": frac_mid_above,
        })

m1q = pd.DataFrame(records)
df = obs.merge(m1q, on=["pair", "signal_time"], how="inner")
df["year"] = pd.to_datetime(df["signal_time"]).dt.year
print(f"\nfires with M1 windows: {len(df)} of {len(obs)}")

METRICS = ["m1_pierce_count", "m1_last_pierce_frac", "m1_hold_min_margin_atr",
           "m1_time_above_after_frac", "m1_close_margin_atr", "m1_frac_mid_above"]

print("\n=== per-year capture + M1-quality (2018 is the failing fold) ===")
agg = df.groupby("year").agg(
    n=("capture", "size"), capture=("capture", "mean"), drift=("fwd_drift_atr", "mean"),
    **{m: (m, "mean") for m in METRICS})
pd.set_option("display.width", 200, "display.max_columns", 30)
print(agg.round(3).to_string())

print("\n=== 2018 vs other-years (mean M1-quality) ===")
is2018 = df["year"] == 2018
for m in METRICS + ["capture", "fwd_drift_atr"]:
    a18 = df.loc[is2018, m].mean()
    aoth = df.loc[~is2018, m].mean()
    print(f"  {m:28s} 2018={a18:+.3f}   other={aoth:+.3f}   delta={a18-aoth:+.3f}")

print("\n=== Q2: does M1-quality predict capture? (corr of metric with capture, all fires) ===")
for m in METRICS:
    c = df[[m, "capture"]].corr().iloc[0, 1]
    cd = df[[m, "fwd_drift_atr"]].corr().iloc[0, 1]
    print(f"  corr({m:28s}, capture)={c:+.3f}   corr(.,drift)={cd:+.3f}")

print("\n=== capture by M1-quality tercile (close_margin & hold_min_margin) ===")
for m in ["m1_close_margin_atr", "m1_hold_min_margin_atr", "m1_last_pierce_frac"]:
    try:
        df["_terc"] = pd.qcut(df[m], 3, labels=["lo", "mid", "hi"], duplicates="drop")
        g = df.groupby("_terc", observed=True).agg(n=("capture", "size"), cap=("capture", "mean"),
                                                    drift=("fwd_drift_atr", "mean"))
        print(f"\n  {m}:")
        print(g.round(3).to_string())
    except Exception as e:
        print(f"  {m}: qcut failed ({e})")

df.to_parquet("discovery/_disco2_work/arc_2020_fires_m1q.parquet")
print("\nsaved fires+M1q -> discovery/_disco2_work/arc_2020_fires_m1q.parquet")
