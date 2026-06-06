"""Arc 1055 — O1 spread-z inelasticity-state conditioning of fbr (the last untested O1 proxy).

Strategist MENU O1 (DISCOVERY_DIRECTION): concentrate a proven edge onto its most-INELASTIC
entry bars (where a forced flow moves price more per unit depth → larger reversion). Arc 1029
closed the calendar-density proxy; arc 1025 the trigger-shallowing proxy. The bid-ask SPREAD-z
at the entry bar (a lag-free inelasticity proxy) is "the only untested O1 sub-thread" (arc 1029)
— and the one carrying the council's flagged COST TRAP (a wide-spread bar is more inelastic AND
more expensive). Honest test = does fbr's per-trade edge rise with entry-spread-z NET of the
realized wider spread, and does a top-decile book have a higher worst-fold?

OBSERVATION ONLY (gross drift/capture + realized-spread cost in R). No engine/null/council.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from core.sim.panel import Panel
from discovery.tools.failed_breakdown_signals import FailedBreakdownReclaimLongSignal
from discovery.tools.observe_long_capture import observe_long_capture

PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD"]
ROOT = r"C:\Users\panap\histdata_backup"
SL_MULT = 2.0
SPREAD_WIN = 250  # trailing H4 bars (~6 wks) for the spread-z reference


def main() -> None:
    panel = Panel.from_pairs(PAIRS, tf="H4", histdata_root=ROOT,
                             cache_root="data/cache", boundary_convention="5ers_eet")
    sig = FailedBreakdownReclaimLongSignal(swing_lookback=40, min_shadow_atr=1.25)
    ev = sig.evaluate({"H4": panel})

    obs = observe_long_capture(panel, sl_mult=SL_MULT, hold=120, drift_bars=24, direction="long")
    obs = obs.set_index(["pair", "signal_time"])

    rows = []
    for pair in PAIRS:
        df = panel.pair_dfs[pair]
        mask = ev.per_pair[pair].signal_mask
        fires = mask.index[mask.values]
        fires = fires[(fires >= pd.Timestamp(date(2010, 1, 1), tz="UTC")) &
                      (fires <= pd.Timestamp(date(2020, 12, 31), tz="UTC"))]
        if len(fires) == 0:
            continue
        mid = (df["close_bid"] + df["close_ask"]) / 2.0
        spread = df["spread_close"]
        # causal trailing spread-z: (spread - rolling mean)/rolling std, shift1 refs
        ref_mean = spread.rolling(SPREAD_WIN).mean().shift(1)
        ref_std = spread.rolling(SPREAD_WIN).std().shift(1)
        spread_z = (spread - ref_mean) / ref_std
        spread_bp = 1e4 * spread / mid
        for t in fires:
            if t not in obs.index.get_level_values("signal_time") or (pair, t) not in obs.index:
                continue
            o = obs.loc[(pair, t)]
            atr = float(o["atr"])
            if not np.isfinite(atr) or atr <= 0:
                continue
            sz = spread_z.get(t, np.nan)
            spr = float(spread.get(t, np.nan))
            if not np.isfinite(sz) or not np.isfinite(spr):
                continue
            # realized round-trip spread cost in R: 1.5x full spread / (sl_mult*ATR), 1R=sl_mult*ATR
            cost_R = (1.5 * spr) / (SL_MULT * atr)
            rows.append({
                "pair": pair, "year": t.year, "t": t,
                "capture": float(o["capture"]), "drift": float(o["fwd_drift_atr"]),
                "spread_z": float(sz), "spread_bp": float(spread_bp.get(t, np.nan)),
                "cost_R": cost_R, "net_drift": float(o["fwd_drift_atr"]) - cost_R,
            })
    d = pd.DataFrame(rows)
    print(f"fbr fires with spread-z: n={len(d)} (7 USD majors, IS 2010-2020)\n")

    # tercile by spread-z
    d["tier"] = pd.qcut(d["spread_z"], 3, labels=["LO_tight", "MID", "HI_wide"])
    print("--- by entry-spread-z tercile (gross drift / capture / realized spread cost / NET) ---")
    g = d.groupby("tier", observed=True).agg(
        n=("drift", "size"), spread_bp=("spread_bp", "median"),
        gross_drift=("drift", "mean"), capture=("capture", "mean"),
        cost_R=("cost_R", "mean"), net_drift=("net_drift", "mean"),
        med_drift=("drift", "median"))
    print(g.round(4).to_string())

    # top decile vs full: per-year worst fold (gross drift mean per year as a fold proxy)
    print("\n--- top-spread-z decile vs full pool: per-year gross-drift fold signs ---")
    thr = d["spread_z"].quantile(0.90)
    top = d[d["spread_z"] >= thr]
    for name, sub in (("FULL", d), ("TOP_DECILE", top)):
        yr = sub.groupby("year")["drift"].mean()
        worst = yr.min()
        npos = int((yr > 0).sum())
        print(f"{name:11} n={len(sub):4d}  years={len(yr)}  pos_years={npos}/{len(yr)}  "
              f"worst_year_drift={worst:+.3f}  2015={yr.get(2015, float('nan')):+.3f}  "
              f"2018={yr.get(2018, float('nan')):+.3f}")

    # monotonicity check
    gv = g["net_drift"].values
    print(f"\nNET-drift monotone-rising in spread-z? {bool(gv[0] < gv[1] < gv[2])}  "
          f"(LO={gv[0]:+.4f} MID={gv[1]:+.4f} HI={gv[2]:+.4f})")
    gg = g["gross_drift"].values
    print(f"GROSS-drift monotone-rising in spread-z? {bool(gg[0] < gg[1] < gg[2])}  "
          f"(LO={gg[0]:+.4f} MID={gg[1]:+.4f} HI={gg[2]:+.4f})")


if __name__ == "__main__":
    main()
