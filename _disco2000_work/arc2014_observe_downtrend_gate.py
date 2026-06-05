"""arc 2014 — OBSERVE/DIAGNOSE: does a downtrend-PERSISTENCE/STRENGTH gate explain arc-1013's 2018 loss?

arc 1013 (failed-breakdown reclaim long, USD majors H4) is the corpus's strongest edge: IS 9/10 folds
positive, ONLY 2018 negative (-4.20), OOS +0.94% (negs 2022/2025 share the strong-USD/risk-off signature).
arc 1013 thread #5 flagged the open refinement: condition OUT the strong-USD regime with a PRE-REGISTERED
causal measure (NOT fished to flip 2018, arc-1012 trap). obs #3 only tested a BINARY D1<=SMA50 split
(found mild-down is FINE for the reclaim); it never tested downtrend STRENGTH/PERSISTENCE as a continuous
gate.

MECHANISM / *because* (pre-registered before measuring the 2018 effect): the reclaim is a LIQUIDITY-grab
reversal. In an EXTREME / established downtrend (2018-style strong-USD), a swept swing low is an
INFORMATIONAL breakdown (real sellers) and the "reclaim" is a pause before continuation, not a grab. So the
edge should LIVE in balanced/mild-down context and DIE in extreme-persistent-downtrend context. This is a
TAIL gate on downtrend strength (consistent with obs #3: mild-down kept), NOT a generic uptrend filter.

This script only OBSERVES (gross, pool final_r) — no verdict. It tabulates the reclaim edge by several
candidate downtrend-strength features and decomposes 2018, to SEE whether the relationship is real before
committing to a single pre-registered gate + honest engine (§5f).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import date

from core.sim.panel import Panel
from core.arc.arc_pool_builder import ArcPoolConfig, build_arc_pool
from discovery.tools.failed_breakdown_signals import FailedBreakdownReclaimLongSignal

USD_MAJORS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCAD", "USDCHF"]
BACKUP = r"C:\Users\panap\histdata_backup"


def _atr_mid(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = (df["high_bid"] + df["high_ask"]) / 2.0
    low = (df["low_bid"] + df["low_ask"]) / 2.0
    close = (df["close_bid"] + df["close_ask"]) / 2.0
    pc = close.shift(1)
    tr = pd.concat([(high - low), (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean().shift(1)


def main() -> None:
    panel = Panel.from_pairs(USD_MAJORS, tf="H4", histdata_root=BACKUP, cache_root="data/cache",
                             boundary_convention="5ers_eet")
    sig = FailedBreakdownReclaimLongSignal(swing_lookback=40, min_shadow_atr=1.25)
    pool = build_arc_pool(sig, {"H4": panel}, ArcPoolConfig(
        arc_name="arc_2014_obs", sl_atr_mult=2.0, hold_bars=120, risk_pct=0.005,
        window_start=date(2010, 1, 1), window_end=date(2020, 12, 31)))
    tr = pool.trades.copy()
    print("POOL columns:", list(tr.columns))
    print("n trades:", len(tr), "mean final_r:", f"{tr['final_r'].mean():+.4f}", "win:", f"{(tr['final_r']>0).mean():.3f}")

    # locate the entry-time + pair columns robustly
    tcol = next((c for c in ["signal_time", "entry_time", "entry_ts", "signal_ts", "timestamp"] if c in tr.columns), None)
    pcol = next((c for c in ["pair", "symbol"] if c in tr.columns), None)
    print("time col:", tcol, "pair col:", pcol)
    if tcol is None or pcol is None:
        return
    tr[tcol] = pd.to_datetime(tr[tcol])
    tr["year"] = tr[tcol].dt.year

    # build per-pair downtrend-strength features at each H4 bar, then join onto each trade's entry bar
    feats = []
    for pair in USD_MAJORS:
        df = panel.pair_dfs[pair]
        close = (df["close_bid"] + df["close_ask"]) / 2.0
        atr = _atr_mid(df, 14)
        sma100 = close.rolling(100).mean()
        sma200 = close.rolling(200).mean()
        # candidate downtrend-STRENGTH features (all causal: use info up to bar i close)
        slope200 = (sma200 - sma200.shift(50)) / atr            # SMA200 slope over 50 bars in ATR (down<0)
        dist200 = (close - sma200) / atr                        # distance vs SMA200 in ATR (down<0)
        below200 = (close < sma200).astype(int)
        consec_below = below200.groupby((below200 != below200.shift()).cumsum()).cumcount() + 1
        consec_below = consec_below.where(below200 == 1, 0)     # consecutive bars below SMA200
        f = pd.DataFrame({
            "pair": pair, tcol: df.index,
            "slope200_atr": slope200.to_numpy(),
            "dist200_atr": dist200.to_numpy(),
            "consec_below200": consec_below.to_numpy(),
            "below_sma100": (close < sma100).astype(int).to_numpy(),
        })
        feats.append(f)
    feats = pd.concat(feats, ignore_index=True)
    feats[tcol] = pd.to_datetime(feats[tcol])

    m = tr.merge(feats, on=["pair", tcol], how="left")
    print("merged, NaN slope frac:", f"{m['slope200_atr'].isna().mean():.3f}")

    def tab(mask_name, mask):
        sub = m[mask]
        if len(sub) == 0:
            print(f"  {mask_name:42s} n=0")
            return
        print(f"  {mask_name:42s} n={len(sub):4d}  mean_r={sub['final_r'].mean():+.4f}  win={(sub['final_r']>0).mean():.3f}")

    print("\n=== overall ===")
    tab("ALL", m.index == m.index)

    print("\n=== by SMA200 slope (in ATR over 50 bars) — downtrend strength ===")
    for lo, hi, lbl in [(-99, -0.5, "slope<-0.5 (strong down)"), (-0.5, -0.1, "-0.5..-0.1 (mild down)"),
                        (-0.1, 0.1, "-0.1..0.1 (flat)"), (0.1, 99, "slope>0.1 (up)")]:
        tab(lbl, (m["slope200_atr"] > lo) & (m["slope200_atr"] <= hi))

    print("\n=== by distance below SMA200 (ATR) ===")
    for lo, hi, lbl in [(-99, -3, "dist<-3 (deep below)"), (-3, -1, "-3..-1 below"),
                        (-1, 1, "-1..1 near"), (1, 99, ">1 above")]:
        tab(lbl, (m["dist200_atr"] > lo) & (m["dist200_atr"] <= hi))

    print("\n=== by consecutive bars below SMA200 ===")
    for lo, hi, lbl in [(-1, 0, "0 (above)"), (0, 30, "1-30"), (30, 100, "31-100"), (100, 99999, ">100 persistent")]:
        tab(lbl, (m["consec_below200"] > lo) & (m["consec_below200"] <= hi))

    # the decisive cut: per-year mean_r overall vs after dropping the strong-down tail
    print("\n=== per-year: ALL vs gate (drop slope200<-0.5 OR consec_below200>100) ===")
    gate_out = (m["slope200_atr"] < -0.5) | (m["consec_below200"] > 100)
    for yr in range(2010, 2021):
        a = m[m["year"] == yr]
        g = m[(m["year"] == yr) & (~gate_out)]
        if len(a) == 0:
            continue
        print(f"  {yr}: ALL n={len(a):3d} mean_r={a['final_r'].mean():+.4f} | GATED n={len(g):3d} "
              f"mean_r={(g['final_r'].mean() if len(g) else float('nan')):+.4f}  (dropped {len(a)-len(g)})")

    # how concentrated are 2018 trades in the gated-out region?
    y18 = m[m["year"] == 2018]
    print(f"\n2018: n={len(y18)} mean_r={y18['final_r'].mean():+.4f}; "
          f"gate_out frac={gate_out[m['year']==2018].mean():.3f}; "
          f"kept2018 mean_r={(y18[~gate_out[m['year']==2018]]['final_r'].mean() if (~gate_out[m['year']==2018]).any() else float('nan')):+.4f}")


if __name__ == "__main__":
    main()


def followup() -> None:
    import pandas as pd
    from datetime import date
    panel = Panel.from_pairs(USD_MAJORS, tf="H4", histdata_root=BACKUP, cache_root="data/cache",
                             boundary_convention="5ers_eet")
    sig = FailedBreakdownReclaimLongSignal(swing_lookback=40, min_shadow_atr=1.25)
    pool = build_arc_pool(sig, {"H4": panel}, ArcPoolConfig(
        arc_name="arc_2014_obs2", sl_atr_mult=2.0, hold_bars=120, risk_pct=0.005,
        window_start=date(2010, 1, 1), window_end=date(2020, 12, 31)))
    tr = pool.trades.copy()
    tr["signal_time"] = pd.to_datetime(tr["signal_time"])
    tr["year"] = tr["signal_time"].dt.year
    feats = []
    for pair in USD_MAJORS:
        df = panel.pair_dfs[pair]
        close = (df["close_bid"] + df["close_ask"]) / 2.0
        atr = _atr_mid(df, 14)
        sma200 = close.rolling(200).mean()
        feats.append(pd.DataFrame({"pair": pair, "signal_time": df.index,
                                   "dist200_atr": ((close - sma200) / atr).to_numpy()}))
    feats = pd.concat(feats, ignore_index=True); feats["signal_time"] = pd.to_datetime(feats["signal_time"])
    m = tr.merge(feats, on=["pair", "signal_time"], how="left")
    y18 = m[m["year"] == 2018]
    print("\n2018 per-pair final_r:")
    print(y18.groupby("pair")["final_r"].agg(["count", "mean"]).round(3).to_string())
    near = y18[(y18["dist200_atr"] > -1) & (y18["dist200_atr"] <= 1)]
    print(f"\n2018 in BEST context (near SMA200, dist -1..1): n={len(near)} mean_r={near['final_r'].mean():+.4f}")
    print(f"2018 winners: {(y18['final_r']>0).sum()}/{len(y18)}; max final_r={y18['final_r'].max():+.3f}")

followup()
