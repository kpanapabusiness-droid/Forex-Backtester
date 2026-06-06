"""Arc 1059 — USD-TREND-BREADTH conditioning of me_long (the deploy object).

THE IDEA (a genuinely-untested conditioner of the SURVIVING edge, pointed at by arc 2018).
me_long (arc 1011 / the honest deploy object, 1046) buys a pair that fell >=1 ATR into month-end,
betting the post-fix WMR rebalancing reverts it UP. Its KNOWN weakness is the contiguous
2014/2015/2016 strong-USD-bull IS hole (-1.14/-0.51/-0.23) -- the *because* in the signal docstring
is "in a strong-USD trend a big down move into ME IS the trend -> continues, doesn't revert."

Arc 2018 found me_long carries a REAL positive 2018 via DIRECTIONAL WMR and that stripping USD beta
(cross-sectional/USD-neutral) REMOVES the +2018 help -> me_long's edge has a directional-USD
component. So the natural, untested refinement: condition me_long on the BREADTH / strength of the
COMMON USD move into month-end.

HYPOTHESIS (falsifiable, mechanistic). A me_long fire is the pair that fell into month-end. Split by
whether that fall was SYSTEMATIC (the whole USD complex trended that way -> a TREND that continues ->
reversion fails) vs IDIOSYNCRATIC (the pair fell on its own, against/orthogonal to the common USD move
-> a pair-specific flow over-extension -> reverts). Prediction:
  (A) me_long capture/drift DECREASES with |common-USD-move| (breadth = trend strength);
  (B) IDIOSYNCRATIC fires (pair's USD-signed move disagrees with the common USD move) REVERT
      (positive), SYSTEMATIC fires (agree) CONTINUE (negative);
  (C) the negative IS years (2014-16) concentrate in the high-breadth / systematic bucket, so an
      idiosyncratic-only filter LIFTS the worst fold.
If capture is flat/non-monotone (arc-3007 not-a-lever tell) OR the favorable bucket THINS folds worse
(the 1029/1055 concentration-death) -> cheap KILL. If a real, monotone, fold-lifting separation ->
PROCEED to the honest engine sec5f (me_long is non-coin-flip, so the engine, not obs, is the gate).

OBSERVATION cheap-kill (sec5d): honest take-the-loss capture + fwd-drift (BUILT observe_long_capture),
IS 2010-2020, D1, 7 USD majors. No engine / null / council here; OOS untouched.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.sim.panel import Panel
from discovery.tools.month_end_signals import (
    MonthEndReversionLongSignal,
    _month_end_into_move,
)
from discovery.tools.observe_long_capture import observe_long_capture

HIST = r"C:\Users\panap\histdata_backup"
CACHE = "data/cache"
PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCAD", "USDCHF"]
IS_START, IS_END = pd.Timestamp("2010-01-01", tz="UTC"), pd.Timestamp("2020-12-31", tz="UTC")

# USD-sign factor: pair UP-move -> USD direction. USDXXX up = USD up (+1); XXXUSD up = USD down (-1).
USD_FACTOR = {p: (+1.0 if p.startswith("USD") else -1.0) for p in PAIRS}


def build_common_usd_move(panel) -> pd.DataFrame:
    """Per month-end (keyed by year-month), the COMMON USD move = mean over the 7 majors of the
    USD-signed 2-day move into month-end (in ATR). Sign = USD direction, |.| = breadth/trend strength."""
    rows = []
    for pair in PAIRS:
        df = panel.pair_dfs[pair]
        idx, is_last, into, _atr = _month_end_into_move(df, into_bars=2, atr_period=14)
        m = is_last & np.isfinite(into)
        if not m.any():
            continue
        sub = pd.DataFrame({
            "pair": pair,
            "signal_time": idx[m],
            "into_atr": into[m],
            "usd_move": USD_FACTOR[pair] * into[m],
        })
        rows.append(sub)
    allme = pd.concat(rows, ignore_index=True)
    allme["ym"] = allme["signal_time"].dt.to_period("M")
    common = allme.groupby("ym").agg(
        common_usd_move=("usd_move", "mean"),
        n_pairs=("usd_move", "size"),
    )
    common["breadth"] = common["common_usd_move"].abs()
    return allme, common


def main():
    panel = Panel.from_pairs(PAIRS, tf="D1", histdata_root=HIST, cache_root=CACHE,
                             boundary_convention="5ers_eet")
    allme, common = build_common_usd_move(panel)

    # me_long fires + honest capture/drift per fire
    long_sig = MonthEndReversionLongSignal(threshold_atr=1.0, into_bars=2)
    ev = long_sig.evaluate({"D1": panel})
    fires = {p: ev.per_pair[p].signal_mask.to_numpy(bool) for p in panel.pairs}
    obs = observe_long_capture(panel, sl_mult=2.0, hold=120, drift_bars=2, warmup=30,
                               restrict=fires, direction="long")
    obs = obs[(obs["signal_time"] >= IS_START) & (obs["signal_time"] <= IS_END)].copy()
    obs["ym"] = obs["signal_time"].dt.to_period("M")
    obs["year"] = obs["signal_time"].dt.year

    # join each fire's own USD-signed move + the month's common USD move
    obs = obs.merge(allme[["pair", "signal_time", "into_atr", "usd_move"]],
                    on=["pair", "signal_time"], how="left")
    obs = obs.merge(common[["common_usd_move", "breadth"]], left_on="ym", right_index=True, how="left")
    obs = obs.dropna(subset=["common_usd_move"]).copy()

    # per-fire alignment: SYSTEMATIC if the pair's USD-signed move agrees with the common USD move
    obs["systematic"] = (np.sign(obs["usd_move"]) == np.sign(obs["common_usd_move"]))
    # breadth tercile (month-level trend strength)
    obs["breadth_t"] = pd.qcut(obs["breadth"], 3, labels=["LO", "MID", "HI"])

    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(f"=== arc 1059: USD-breadth conditioning of me_long | IS 2010-2020 D1 | n_fires={len(obs)} ===")
    base_cap = obs["capture"].mean()
    base_dr = obs["fwd_drift_atr"].mean()
    print(f"unconditional me_long: capture {base_cap:.4f}  drift_mean {base_dr:+.4f}  "
          f"drift_med {obs['fwd_drift_atr'].median():+.4f}")

    print("\n-- (A) by |common-USD-move| TERCILE (breadth = trend strength; predict capture DECREASES) --")
    gA = obs.groupby("breadth_t", observed=True).agg(
        n=("capture", "size"), capture=("capture", "mean"),
        drift_mean=("fwd_drift_atr", "mean"), drift_med=("fwd_drift_atr", "median"),
        breadth_lo=("breadth", "min"), breadth_hi=("breadth", "max"))
    print(gA.to_string())

    print("\n-- (B) by per-fire SYSTEMATIC vs IDIOSYNCRATIC (predict idiosyncratic REVERTS) --")
    gB = obs.groupby("systematic").agg(
        n=("capture", "size"), capture=("capture", "mean"),
        drift_mean=("fwd_drift_atr", "mean"), drift_med=("fwd_drift_atr", "median"))
    gB.index = ["IDIOSYNCRATIC", "SYSTEMATIC"] if list(gB.index) == [False, True] else gB.index
    print(gB.to_string())

    print("\n-- (C) fold-resolution: per-year mean drift  ALL vs IDIOSYNCRATIC-only vs LO-breadth-only --")
    allyr = obs.groupby("year")["fwd_drift_atr"].mean()
    idio = obs[~obs["systematic"]].groupby("year")["fwd_drift_atr"].mean()
    lob = obs[obs["breadth_t"] == "LO"].groupby("year")["fwd_drift_atr"].mean()
    comp = pd.DataFrame({
        "all_drift": allyr, "n_all": obs.groupby("year").size(),
        "idio_drift": idio, "n_idio": obs[~obs["systematic"]].groupby("year").size(),
        "lobreadth_drift": lob, "n_lob": obs[obs["breadth_t"] == "LO"].groupby("year").size(),
    })
    print(comp.to_string())
    for name, ser, n in (("ALL", allyr, comp["n_all"]),
                         ("IDIOSYNCRATIC", idio, comp["n_idio"]),
                         ("LO-breadth", lob, comp["n_lob"])):
        print(f"  {name:14s}: neg-years {int((ser < 0).sum())}/{ser.notna().sum()}  "
              f"mean-of-year-drift {ser.mean():+.4f}  median trades/yr {n.median():.0f}  total n {int(n.sum())}")

    # known strong-USD hole years vs the rest, by bucket
    print("\n-- strong-USD-bull block 2014-16 (the IS hole): drift by bucket --")
    obs["block"] = np.where(obs["year"].isin([2014, 2015, 2016]), "2014-16", "other")
    gblk = obs.groupby(["block", "systematic"]).agg(
        n=("capture", "size"), capture=("capture", "mean"), drift_mean=("fwd_drift_atr", "mean"))
    print(gblk.to_string())

    # sec5f BEST VERSION: don't CONCENTRATE (thins) -- EXCLUDE only the worst HI-breadth tercile.
    # keeps ~2/3 of trades (no thinning death); does dropping HI-breadth fires LIFT the worst fold?
    print("\n-- (D) BEST filter: EXCLUDE HI-breadth tercile (keep LO+MID) vs ALL -- per-year drift --")
    keep = obs[obs["breadth_t"] != "HI"].copy()
    keepyr = keep.groupby("year")["fwd_drift_atr"].mean()
    keepcap = keep.groupby("year")["capture"].mean()
    compD = pd.DataFrame({
        "all_drift": allyr, "n_all": obs.groupby("year").size(),
        "keep_drift": keepyr, "n_keep": keep.groupby("year").size(),
        "keep_capture": keepcap,
    })
    print(compD.to_string())
    print(f"  ALL          : neg-years {int((allyr < 0).sum())}/{allyr.notna().sum()}  "
          f"mean-drift {allyr.mean():+.4f}  worst-yr {allyr.min():+.4f}  total n {len(obs)}")
    print(f"  EXCL-HI (D)  : neg-years {int((keepyr < 0).sum())}/{keepyr.notna().sum()}  "
          f"mean-drift {keepyr.mean():+.4f}  worst-yr {keepyr.min():+.4f}  total n {len(keep)}  "
          f"capture {keep['capture'].mean():.4f}")
    # pooled capture/drift for the kept set vs full
    print(f"  pooled: ALL capture {obs['capture'].mean():.4f} drift {obs['fwd_drift_atr'].mean():+.4f} | "
          f"EXCL-HI capture {keep['capture'].mean():.4f} drift {keep['fwd_drift_atr'].mean():+.4f}")


if __name__ == "__main__":
    main()
