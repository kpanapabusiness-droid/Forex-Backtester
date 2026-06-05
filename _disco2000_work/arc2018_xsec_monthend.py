"""arc 2018 — Cross-sectional month-end rebalancing reversion, USD-NEUTRAL (cheap-kill obs).

Because: the proven me edge (1011/1019, WMR-fix forced rebalancing) is per-pair ABSOLUTE move
into month-end -> USD-beta-exposed (why 2015/2018 bind). A CROSS-SECTIONAL construction (rank
currencies by idiosyncratic month return vs USD; long laggard / short leader at month-end) is
USD-NEUTRAL by construction (strips the regime exposure) and THICKER (fires every month, both
sides) -> targets arc-2017 spec (B): a component thick enough to RESOLVE folds.

This is OBSERVATION ONLY (drift lens, gross). No engine, no cost realized. IS (2010-2020).
Control (arc-1011 style): same cross-sectional rank on a RANDOM day -> month-end timing must be
load-bearing (else it's the dead generic relative-value of 2003/2010).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.sim.panel import Panel

HISTDATA = r"C:\Users\panap\histdata_backup"
PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCHF", "USDCAD"]
# currency vs USD: XXXUSD -> +ret(pair); USDXXX -> -ret(pair)
CCY = {
    "EURUSD": ("EUR", +1), "GBPUSD": ("GBP", +1), "AUDUSD": ("AUD", +1),
    "NZDUSD": ("NZD", +1), "USDJPY": ("JPY", -1), "USDCHF": ("CHF", -1),
    "USDCAD": ("CAD", -1),
}
IS_START, IS_END = pd.Timestamp("2010-01-01", tz="UTC"), pd.Timestamp("2020-12-31", tz="UTC")


def load():
    p = Panel.from_pairs(PAIRS, tf="D1", histdata_root=HISTDATA,
                         cache_root="data/cache", boundary_convention="5ers_eet")
    return p


def logret_vs_usd(panel):
    """Per-currency daily log-return vs USD, aligned on a common date index."""
    series = {}
    for pair in PAIRS:
        df = panel.pair_dfs[pair]
        mid = (df["close_bid"] + df["close_ask"]) / 2.0
        lr = np.log(mid).diff()
        ccy, sign = CCY[pair]
        series[ccy] = sign * lr
    R = pd.DataFrame(series)  # index = dates, cols = currencies
    return R


def month_end_mask(idx):
    ym = pd.PeriodIndex(idx, freq="M")
    is_last = np.zeros(len(idx), dtype=bool)
    is_last[:-1] = ym[1:].asi8 != ym[:-1].asi8
    return is_last


def run():
    panel = load()
    R = logret_vs_usd(panel)  # daily vs-USD log returns per ccy
    idx = R.index
    is_me = month_end_mask(idx)

    # full prior-month vs-USD return: sum of daily log-rets within the calendar month, up to & incl bar t
    ym = pd.PeriodIndex(idx, freq="M")
    month_cum = R.groupby(ym).cumsum()  # cumulative within month, per ccy

    # forward FWD-day vs-USD return from t+1 (enter next day): sum of lr[t+1 .. t+FWD]
    def fwd_ret(FWD):
        # forward sum excluding current bar: shift(-1) cumulated
        fwd = R[::-1].rolling(FWD, min_periods=1).sum()[::-1].shift(-1)
        return fwd

    print("=== Cross-sectional month-end reversion, USD majors (7 ccy vs USD), IS 2010-2020 ===")
    print(f"rows total {len(idx)}, month-ends {is_me.sum()}")

    for FWD in (1, 2, 3, 5):
        fwd = fwd_ret(FWD)
        # restrict to IS
        in_is = (idx >= IS_START) & (idx <= IS_END)

        def spread_on(mask_days, label, top_k=2):
            rows = []
            day_pos = np.where(mask_days & in_is)[0]
            for t in day_pos:
                mret = month_cum.iloc[t]  # vs-USD month-to-date return per ccy
                fret = fwd.iloc[t]
                if mret.isna().any() or fret.isna().any():
                    continue
                order = mret.sort_values()
                bottom = order.index[:top_k]   # most depreciated -> expect revert UP (+)
                top = order.index[-top_k:]     # most appreciated -> expect revert DOWN (-)
                # market-neutral reversion spread: long bottom, short top
                spread = fret[bottom].mean() - fret[top].mean()
                rows.append({"t": idx[t], "year": idx[t].year, "spread": spread,
                             "fwd_top": fret[top].mean(), "fwd_bottom": fret[bottom].mean()})
            d = pd.DataFrame(rows)
            if d.empty:
                print(f"  [{label} FWD{FWD}] no obs"); return None
            mean_bp = d["spread"].mean() * 1e4
            med_bp = d["spread"].median() * 1e4
            fracpos = (d["spread"] > 0).mean()
            print(f"  [{label} FWD{FWD} top{top_k}] n={len(d)} spread mean={mean_bp:+.2f}bp "
                  f"median={med_bp:+.2f}bp frac+={fracpos:.3f} "
                  f"(top fwd {d['fwd_top'].mean()*1e4:+.2f}bp / bottom fwd {d['fwd_bottom'].mean()*1e4:+.2f}bp)")
            return d

        for tk in (1, 2):
            d_me = spread_on(is_me, "MONTH-END", top_k=tk)
            d_rand = spread_on(~is_me, "RANDOM-DAY", top_k=tk)
            if d_me is not None and d_rand is not None:
                excess = d_me["spread"].mean() - d_rand["spread"].mean()
                # cost hurdle: each leg ~ (1.5x spread + commission + slippage). USD-major D1
                # spread ~ 1-2bp; FundedNext 1.5x + $5/lot RT (~0.5bp) + 0.5pip slip. Conservative
                # ~ 3bp/leg round-turn; market-neutral = 2*top_k legs.
                cost_bp = 3.0 * 2 * tk
                print(f"     [top{tk}] ME-excess {excess*1e4:+.2f}bp | ~cost {cost_bp:.0f}bp "
                      f"({2*tk} legs) | net(median) {d_me['spread'].median()*1e4 - cost_bp:+.2f}bp")
                yr = d_me.groupby("year")["spread"].mean() * 1e4
                print(f"        per-year ME spread (bp):", {int(k): round(v, 1) for k, v in yr.items()})


if __name__ == "__main__":
    run()
