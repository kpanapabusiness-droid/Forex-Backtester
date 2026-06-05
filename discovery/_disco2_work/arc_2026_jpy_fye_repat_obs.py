"""Arc 2026 observation: Japanese FISCAL-YEAR-END (March 31) repatriation — a multi-WEEK, calendar-anchored,
LARGE-displacement forced flow (distinct from the sub-cost intraday fix flows arc 2025/3008 closed, and from
the price-triggered month-end reversion `me`). OBSERVATION ONLY (gross; NOT a gate).

because: Japanese institutions (insurers, pensions, corporates) close books on 31 March and repatriate
foreign assets / settle FX hedges into the fiscal year-end -> persistent JPY-BUYING through late March ->
JPY-pairs (USDJPY, EURJPY, ...) tend to FALL into 31 March. Unlike gotobi/WMR (sub-pip intraday fix moves),
this is a MULTI-WEEK directional flow -> if real & directionally reliable, the displacement is multi-ATR and
clears cost easily. Tests the arc-2024/2025 unifying hypothesis (a LARGE-displacement forced flow CAN clear
cost) on a genuinely new instance; candidate decorrelated (risk-off-ish) PORTFOLIO component.

Distinct from `me` (1011/1019): `me` is PRICE-MOVE-triggered reversion at EVERY month-end; this is a
CALENDAR-directional bias SPECIFIC to March (the fiscal year-end), not conditioned on a price move.

Question (§5d): is the late-March JPY-strength drift (a) directionally reliable across years, (b) LARGER
than other month-ends (March EXCESS), and (c) > cost? If coin-flip / not-March-specific -> KILL.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.sim.panel import Panel

PAIRS = ["USDJPY", "EURJPY", "GBPJPY", "AUDJPY"]
HIST = r"C:\Users\panap\histdata_backup"
IS_LO, IS_HI = 2010, 2020
WINDOW = 10  # trailing trading days into month-end (the repatriation window)

print("loading D1 panel (cached)...")
panel = Panel.from_pairs(PAIRS, tf="D1", histdata_root=HIST,
                         cache_root="data/cache", boundary_convention="5ers_eet")


def month_end_window_returns(df):
    """For each (pair) return a tidy frame: one row per month-end, with the JPY-strength drift over the
    trailing WINDOW trading days (= -(pair return) so positive = JPY strengthened = repatriation direction),
    in ATR units, plus the calendar month + year. Ex-ante framing: this is a descriptive obs of realized
    month-end-window drift (no entry decision; characterization of whether March is special)."""
    idx = df.index
    close_mid = (df["close_bid"].to_numpy(float) + df["close_ask"].to_numpy(float)) / 2.0
    high = df["high_bid"].to_numpy(float); low = df["low_bid"].to_numpy(float)
    # Wilder ATR(14) on mid, simple
    tr = np.maximum(high - low, np.abs(np.diff(close_mid, prepend=close_mid[0])))
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().shift(1).to_numpy()
    ym = pd.PeriodIndex(idx, freq="M")
    is_last = np.zeros(len(df), dtype=bool)
    is_last[:-1] = ym[1:] != ym[:-1]
    rows = []
    pos = np.arange(len(df))
    for i in pos[is_last]:
        if i - WINDOW < 0:
            continue
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            continue
        pair_ret = close_mid[i] - close_mid[i - WINDOW]
        jpy_strength = -pair_ret / a   # positive = JPY strengthened into month-end
        rows.append({"ts": idx[i], "year": idx[i].year, "month": idx[i].month,
                     "jpy_strength_atr": jpy_strength})
    return pd.DataFrame(rows)


for pair in PAIRS:
    df = panel.pair_dfs[pair]
    d = month_end_window_returns(df)
    d = d[(d["year"] >= IS_LO) & (d["year"] <= IS_HI)]
    march = d[d["month"] == 3]["jpy_strength_atr"]
    other = d[d["month"] != 3]["jpy_strength_atr"]
    excess = march.mean() - other.mean()
    # per-year March sign (directional reliability across years)
    g = d[d["month"] == 3].groupby("year")["jpy_strength_atr"].mean()
    pos = int((g > 0).sum())
    y15 = g.get(2015, float("nan")); y18 = g.get(2018, float("nan"))
    print(f"\n=== {pair} (JPY-strength drift over last {WINDOW} D1 bars into month-end, ATR units, IS) ===")
    print(f"  MARCH  : n={len(march):3d}  mean={march.mean():+.4f}  med={march.median():+.4f}  "
          f"frac+={(march>0).mean():.3f}")
    print(f"  other  : n={len(other):3d}  mean={other.mean():+.4f}  med={other.median():+.4f}  "
          f"frac+={(other>0).mean():.3f}")
    print(f"  MARCH EXCESS (vs other months) = {excess:+.4f} ATR   "
          f"per-year March>0: {pos}/{len(g)}   2015={y15:+.3f}  2018={y18:+.3f}")

print("\nNote: a multi-week drift of >~0.3-0.5 ATR clears the D1 ~0.05-0.10R cost hurdle easily;")
print("the question is DIRECTIONAL RELIABILITY (per-year frac & March-specificity), not magnitude-vs-cost.")
print("DONE.")
