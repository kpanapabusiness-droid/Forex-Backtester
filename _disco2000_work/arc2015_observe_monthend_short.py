"""arc 2015 — OBSERVE: the SHORT side of month-end reversion (sell a big UP-move into month-end).

arc 1011 (month-end reversion LONG, USD majors D1) is a PORTFOLIO edge and the corpus's ONLY
demonstrably 2018-POSITIVE mechanical-flow reversion (arc 3015 diagnosis: it survives 2018 via the
hard inelastic WMR/index-rebalancing mandate). But 1011/1012/3008 only ever tested the LONG side
(buy a big DOWN move into month-end) — shorts were disabled then. PR #273 merged shorts today.

HYPOTHESIS / *because* (pre-registered): month-end WMR/index rebalancing reverts the month's move
REGARDLESS of sign (forced, inelastic). The SHORT leg = sell a big UP move into month-end, betting
reversion DOWN. In strong-USD years (2015/2018) the dominant month-moves are USD-UP → on USDXXX pairs
(USDJPY/USDCAD/USDCHF) that is a big UP move → the short side fires EXACTLY in the strong-USD months the
long-only 1011 under-covers. So a BIDIRECTIONAL month-end reversion could be +2015 AND +2018 (the precise
binding-fold spec the portfolio route needs) — not a directional coin-flip, but the short side of a proven
mechanism.

OBSERVE ONLY (gross capture + drift, direction-aware): does the up-move-into-month-end revert DOWN (short
capture >0.50, short drift >0), and what is its per-year sign (esp 2015 & 2018)? Compared head-to-head with
the long side. No verdict here; if real → build the short signal + honest engine §5f.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.sim.panel import Panel
from discovery.tools.observe_long_capture import observe_long_capture
from discovery.tools.trend_entry_signals import _atr_shift1_mid

USD_MAJORS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCAD", "USDCHF"]
BACKUP = r"C:\Users\panap\histdata_backup"
THRESH = 1.0
INTO = 2


def month_end_move_masks(panel, direction):
    """Per-pair bool mask: last-trading-day-of-month AND big move INTO it in `direction`.
    long  -> into_atr <= -THRESH (big DOWN move, buy it)
    short -> into_atr >= +THRESH (big UP move,   sell it)
    """
    masks = {}
    for pair in USD_MAJORS:
        df = panel.pair_dfs[pair]
        n = len(df)
        idx = df.index
        close_mid = (df["close_bid"].to_numpy(float) + df["close_ask"].to_numpy(float)) / 2.0
        atr = np.asarray(_atr_shift1_mid(df, 14), dtype=float)
        ym = pd.PeriodIndex(idx, freq="M")
        is_last = np.zeros(n, dtype=bool)
        if n >= 2:
            is_last[:-1] = ym[1:] != ym[:-1]
        into = np.full(n, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            into[INTO:] = (close_mid[INTO:] - close_mid[:-INTO]) / atr[INTO:]
        if direction == "long":
            cond = into <= -THRESH
        else:
            cond = into >= +THRESH
        masks[pair] = is_last & np.isfinite(into) & np.isfinite(atr) & (atr > 0) & cond
    return masks


def per_year(obs, label):
    obs = obs.copy()
    obs["year"] = pd.to_datetime(obs["signal_time"]).dt.year
    print(f"\n=== {label}: n={len(obs)}  cap={obs['capture'].mean():.3f}  "
          f"drift_mean={obs['fwd_drift_atr'].mean():+.3f}  drift_median={obs['fwd_drift_atr'].median():+.3f} ===")
    g = obs.groupby("year").agg(n=("capture", "size"), cap=("capture", "mean"),
                                drift=("fwd_drift_atr", "mean")).round(3)
    print(g.to_string())
    return obs


def per_pair(obs, label):
    print(f"  [{label}] per-pair cap/drift:")
    g = obs.groupby("pair").agg(n=("capture", "size"), cap=("capture", "mean"),
                                drift=("fwd_drift_atr", "mean")).round(3)
    print(g.to_string())


def main():
    panel = Panel.from_pairs(USD_MAJORS, tf="D1", histdata_root=BACKUP, cache_root="data/cache",
                             boundary_convention="5ers_eet")
    # restrict to IS window for the observation (match 1011 dev window)
    short_masks = month_end_move_masks(panel, "short")
    long_masks = month_end_move_masks(panel, "long")
    print("fires/pair SHORT:", {p: int(short_masks[p].sum()) for p in USD_MAJORS})
    print("fires/pair LONG :", {p: int(long_masks[p].sum()) for p in USD_MAJORS})

    obs_s = observe_long_capture(panel, sl_mult=2.0, hold=120, drift_bars=INTO, restrict=short_masks, direction="short")
    obs_l = observe_long_capture(panel, sl_mult=2.0, hold=120, drift_bars=INTO, restrict=long_masks, direction="long")
    # IS only (2010-2020) for the per-year table
    obs_s = obs_s[pd.to_datetime(obs_s["signal_time"]).dt.year <= 2020]
    obs_l = obs_l[pd.to_datetime(obs_l["signal_time"]).dt.year <= 2020]

    s = per_year(obs_s, "SHORT (sell big UP into month-end)")
    per_pair(obs_s, "SHORT")
    l = per_year(obs_l, "LONG (buy big DOWN into month-end) [1011 reproduction]")

    # the binding-fold acceptance test
    print("\n=== ACCEPTANCE TEST (short side per-year sign, focus 2015 & 2018) ===")
    for yr in (2015, 2018):
        sy = s[s["year"] == yr]
        ly = l[l["year"] == yr]
        print(f"  {yr}: SHORT n={len(sy):3d} cap={sy['capture'].mean():.3f} drift={sy['fwd_drift_atr'].mean():+.3f} | "
              f"LONG n={len(ly):3d} cap={ly['capture'].mean():.3f} drift={ly['fwd_drift_atr'].mean():+.3f}")


if __name__ == "__main__":
    main()
