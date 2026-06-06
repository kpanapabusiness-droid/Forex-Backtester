"""Arc 1030 — O1 inelasticity-STATE via the entry-bar SPREAD-Z proxy (cheap observation).

The last untested O1 sub-thread (arc 1029 closed the calendar-density proxy; arc 1025 closed
the trigger-shallowing/depth proxy). The council named entry-bar spread-z the lag-free,
calendar-orthogonal proxy of book inelasticity. Hypothesis (O1): a wide bid-ask spread at the
dislocation bar = thin/inelastic book = larger price impact = larger subsequent reversion; so
per-trade realized edge should rise MONOTONICALLY with signal-bar spread-z within the surviving
forced-flow edges, and concentrating onto high-spread-z entries should raise the worst-fold ROI
(attack fold resolution). Cost trap: wide-spread bars cost the most — the engine charges that.

This is the CHEAP OBSERVATION (gross capture/drift lens, honest take-the-loss label) on the
corpus's ONLY fold-resolving edge (fbr, arc 2017) + the pooled month-end reversion (me) legs.
Decisive at obs if non-monotone/flat OR if it cannot separate fbr's binding 2018 fold.

CHARACTERIZATION ONLY — capture/drift are gross. The gate is MultiPairBacktester; this only
screens whether the spread-z axis separates before spending the engine.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from core.sim.panel import Panel
from core.features._helpers import wilder_atr, mid_high, mid_low, mid_close
from discovery.tools.observe_long_capture import observe_long_capture
from discovery.tools.failed_breakdown_signals import FailedBreakdownReclaimLongSignal
from discovery.tools.month_end_signals import (
    MonthEndReversionLongSignal,
    MonthEndReversionShortSignal,
)

HISTDATA = r"C:\Users\panap\histdata_backup"
USD_MAJORS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCHF", "USDCAD"]
WIN_START, WIN_END = date(2010, 1, 1), date(2020, 12, 31)
ZWIN = 250  # trailing window for the causal spread z-score


def spread_z_frame(panel, zwin=ZWIN):
    """Per-pair signal-bar spread metrics (causal). spread = close_ask - close_bid at bar t
    (known at t close; entry is t+1).
      spread_atr = spread/ATR (cost in price-vol units) — the clean inelasticity proxy.
      spread_z   = trailing rolling z-score of spread (shift1 so a bar never enters its own
                   stats). Guarded: where the trailing std is below a per-pair floor (flat-spread
                   HistData windows), z is set NaN so it does not blow up.
      spread_pct = trailing rolling rank/percentile of spread in [0,1] (robust, divide-free).
    Returns tidy DataFrame[pair, signal_time, spread, spread_atr, spread_z, spread_pct]."""
    rows = []
    for pair in sorted(panel.pairs):
        df = panel.pair_dfs[pair]
        idx = df.index
        atr = wilder_atr(mid_high(df), mid_low(df), mid_close(df), 14).shift(1)
        spread = (df["close_ask"] - df["close_bid"]).astype(float)
        roll = spread.shift(1).rolling(zwin)
        mu, sd = roll.mean(), roll.std(ddof=0)
        std_floor = sd[sd > 0].median() * 0.05 if (sd > 0).any() else 0.0
        z = (spread - mu) / sd
        z = z.where(sd > std_floor)  # kill divide-by-~0 flat-spread windows
        out = pd.DataFrame({
            "pair": pair, "signal_time": idx,
            "spread": spread.values, "spread_atr": (spread / atr).values,
            "spread_z": z.values,
        })
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def mask_dict(eval_):
    return {p: st.signal_mask.to_numpy(bool) for p, st in eval_.per_pair.items()}


def tiers(s, labels=("T1_low", "T2_mid", "T3_high")):
    return pd.qcut(s.rank(method="first"), q=len(labels), labels=labels)


def report(df, name, by="spread_z"):
    print(f"\n{'='*72}\n{name}  (n={len(df)})  conditioner={by}\n{'='*72}")
    d = df.dropna(subset=[by]).copy()
    if len(d) < 30:
        print(f"  too thin after dropna ({len(d)})")
        return
    d["tier"] = tiers(d[by])
    g = d.groupby("tier", observed=True).agg(
        n=("capture", "size"),
        capture=("capture", "mean"),
        drift=("fwd_drift_atr", "mean"),
        drift_med=("fwd_drift_atr", "median"),
        spread_atr=("spread_atr", "mean"),
    )
    print(g.to_string())
    # correlation of the conditioner with per-trade outcome
    cc = d[[by, "capture", "fwd_drift_atr"]].corr().loc[by]
    print(f"  corr({by}, capture)={cc['capture']:+.3f}   corr({by}, drift)={cc['fwd_drift_atr']:+.3f}")
    # 2018 separation: within the top-spread-z tier, what is the 2018 capture/drift?
    d["year"] = pd.to_datetime(d["signal_time"]).dt.year
    hi = d[d["tier"] == "T3_high"]
    yr_hi = hi.groupby("year").agg(n=("capture", "size"), cap=("capture", "mean"), drift=("fwd_drift_atr", "mean"))
    print("  --- top spread-z tier, per-year (2018 = fbr binding fold) ---")
    print(yr_hi.to_string())
    return g


def main():
    print("loading H4 panel for USD majors ...")
    panel = Panel.from_pairs(
        USD_MAJORS, tf="H4", histdata_root=HISTDATA,
        cache_root="data/cache", boundary_convention="5ers_eet",
    )
    spz = spread_z_frame(panel)
    print("spread metrics built. global spread_atr describe:")
    print(spz["spread_atr"].describe().to_string())
    print("global spread_z describe:")
    print(spz["spread_z"].describe().to_string())

    # ---- fbr (the fold-resolving edge) ----
    fbr = FailedBreakdownReclaimLongSignal(swing_lookback=40, min_shadow_atr=1.25).evaluate({"H4": panel})
    fbr_obs = observe_long_capture(panel, sl_mult=2.0, hold=120, drift_bars=24,
                                   restrict=mask_dict(fbr), direction="long")
    fbr_obs["signal_time"] = pd.to_datetime(fbr_obs["signal_time"])
    spz["signal_time"] = pd.to_datetime(spz["signal_time"])
    fbr_j = fbr_obs.merge(spz, on=["pair", "signal_time"], how="left")
    # restrict to IS window
    fbr_j = fbr_j[(fbr_j["signal_time"].dt.date >= WIN_START) & (fbr_j["signal_time"].dt.date <= WIN_END)]
    report(fbr_j, "fbr (failed-breakdown reclaim long, H4 USD majors, IS)", by="spread_z")
    report(fbr_j, "fbr (same)", by="spread_atr")

    # ---- pooled month-end reversion (me_long fades DOWN-into-ME, me_short fades UP-into-ME) on D1 ----
    print("\nloading D1 panel for month-end legs ...")
    panel_d1 = Panel.from_pairs(
        USD_MAJORS, tf="D1", histdata_root=HISTDATA,
        cache_root="data/cache", boundary_convention="5ers_eet",
    )
    spz_d1 = spread_z_frame(panel_d1)
    spz_d1["signal_time"] = pd.to_datetime(spz_d1["signal_time"])
    me_l = MonthEndReversionLongSignal(threshold_atr=1.0, into_bars=2).evaluate({"D1": panel_d1})
    me_s = MonthEndReversionShortSignal(threshold_atr=1.0, into_bars=2).evaluate({"D1": panel_d1})
    me_l_obs = observe_long_capture(panel_d1, sl_mult=2.0, hold=120, drift_bars=2,
                                    restrict=mask_dict(me_l), direction="long")
    me_s_obs = observe_long_capture(panel_d1, sl_mult=2.0, hold=120, drift_bars=2,
                                    restrict=mask_dict(me_s), direction="short")
    me = pd.concat([me_l_obs, me_s_obs], ignore_index=True)
    me["signal_time"] = pd.to_datetime(me["signal_time"])
    me_j = me.merge(spz_d1, on=["pair", "signal_time"], how="left")
    me_j = me_j[(me_j["signal_time"].dt.date >= WIN_START) & (me_j["signal_time"].dt.date <= WIN_END)]
    report(me_j, "month-end reversion pooled (me_long+me_short, D1 USD majors, IS)", by="spread_z")
    report(me_j, "month-end pooled (same)", by="spread_atr")


if __name__ == "__main__":
    main()
