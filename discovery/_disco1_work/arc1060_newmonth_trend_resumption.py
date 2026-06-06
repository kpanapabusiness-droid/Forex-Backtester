"""Arc 1060 — POST-month-end TREND RESUMPTION (the regime-orthogonal 5th-leg candidate).

THE IDEA (a genuinely-untested NON-reversion complement, pointed at by arc 1059).
The 4-way reversion book (gap/me_long/me_short/fbr) is never all-folds-positive because of the
strong-USD wall (2014-16, 2018). Arc 1059 data-CONFIRMED the *because*: in a strong-USD trend the
big down-move into month-end IS the trend and CONTINUES (capture falls monotonically with USD
breadth; HI-breadth = 0.405, sub-coin-flip), so the reversion fails exactly in those years. The book
has always needed (arc 1020/2019 spec) a 5th component that is POSITIVE in 2015 AND 2016 WITHOUT
dragging 2018, and NON-reversion (the gap/month-end reversion family is regime-saturated; arc 2019
proved a 5th *reversion* leg can't make the book AFP).

So the untested complement of the SAME mechanism: TREND RESUMPTION. The month-end rebalancing flow is
a TEMPORARY counter-trend dislocation (the reversion edges harvest the snap-back). Once it clears
(the first few days of the new month), the PREVAILING trend re-asserts. Entering at new-month start in
the prevailing-trend direction should be POSITIVE precisely in the strong-trend years (2014-16, 2018)
where the reversion book fails -- the regime-orthogonal NON-reversion leg.

WHY this is NOT closed shallow momentum (§5a): it is calendar-GATED (fires only at start-of-month,
after the flow clears, not every bar), keyed to a specific market-mechanism (flow-clearing trend
resumption), and the KILL/PROCEED criterion is NOT "does it beat cost overall" (a trend leg is
coin-flip across all years by construction -- it pays in trend years, loses in chop) but "is it
RELIABLY POSITIVE in the strong-trend folds 2014-16 AND 2018 the book needs covered." A trend leg can
be overall-coin-flip and STILL be the regime-orthogonal 5th leg if its positive years are exactly the
book's negative years.

HYPOTHESIS (falsifiable, mechanistic).
  (A) new-month resumption (enter in prevailing-trend direction at start of new month) has POSITIVE
      forward drift/capture in the strong-USD-trend years 2014/2015/2016/2018;
  (B) it is the COMPLEMENT of me_long: positive where me_long is negative (anti-correlated fold sign);
  (C) it is more positive when the prevailing trend is STRONG (|trailing return| large) -- the
      genuine-trend setup -- than when flat.
KILL if 2015 AND 2016 (or 2018) are NOT positive, or if "resumption" is just coin-flip every year
(no regime concentration) -> it cannot be the regime-orthogonal leg. PROCEED to honest-engine triage
only if it is reliably positive in the strong-trend folds.

OBSERVATION cheap-kill (§5d): honest take-the-loss capture + fwd-drift (BUILT observe_long_capture,
direction-aware), IS 2010-2020, D1, 7 USD majors. No engine / null / council here; OOS untouched.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.features._helpers import mid_close, wilder_atr, mid_high, mid_low
from core.sim.panel import Panel
from discovery.tools.observe_long_capture import observe_long_capture

HIST = r"C:\Users\panap\histdata_backup"
CACHE = "data/cache"
PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCAD", "USDCHF"]
IS_START, IS_END = pd.Timestamp("2010-01-01", tz="UTC"), pd.Timestamp("2020-12-31", tz="UTC")

STRONG_USD_FOLDS = [2014, 2015, 2016, 2018]  # the reversion book's wall (arc 1020/1059)


def kth_trading_bar_of_month(idx: pd.DatetimeIndex, k: int) -> np.ndarray:
    """Causal mask: True at the bar that is the (1-based) k-th trading bar of its calendar month.
    Pure calendar position -- no lookahead (the bar's own month-rank is known at the bar)."""
    ym = idx.to_period("M")
    rank = np.zeros(len(idx), dtype=int)
    # rank within each month (0-based) in chronological order
    s = pd.Series(np.arange(len(idx)), index=ym)
    for _, grp in s.groupby(level=0, sort=False):
        for j, pos in enumerate(grp.values):
            rank[pos] = j
    return rank == (k - 1)


def build_obs(panel, *, k_entry: int, trend_lb: int, strong_atr: float, hold: int, drift_bars: int):
    """For each pair, flag the k-th trading bar of each month; prevailing-trend direction = sign of
    the trailing `trend_lb`-bar return at that bar (causal); strength = |return| / ATR. Build
    direction-split restrict masks and observe honest resumption capture/drift in the trend direction."""
    long_restrict, short_restrict = {}, {}
    meta_rows = []
    for pair in PAIRS:
        df = panel.pair_dfs[pair]
        idx = df.index
        n = len(df)
        mc = mid_close(df).values
        atr = wilder_atr(mid_high(df), mid_low(df), mid_close(df), 14).shift(1).values
        kth = kth_trading_bar_of_month(idx, k_entry)
        trail_ret = np.full(n, np.nan)
        trail_ret[trend_lb:] = mc[trend_lb:] - mc[:-trend_lb]
        trend_atr = trail_ret / atr  # signed trend strength in ATR
        up = kth & (trend_atr >= strong_atr)
        dn = kth & (trend_atr <= -strong_atr)
        long_restrict[pair] = up & np.isfinite(trend_atr)
        short_restrict[pair] = dn & np.isfinite(trend_atr)
        for t in np.where(kth & np.isfinite(trend_atr))[0]:
            meta_rows.append({"pair": pair, "signal_time": idx[t], "trend_atr": float(trend_atr[t])})
    meta = pd.DataFrame(meta_rows)

    obs_l = observe_long_capture(panel, sl_mult=2.0, hold=hold, drift_bars=drift_bars, warmup=trend_lb + 5,
                                 restrict=long_restrict, direction="long")
    obs_s = observe_long_capture(panel, sl_mult=2.0, hold=hold, drift_bars=drift_bars, warmup=trend_lb + 5,
                                 restrict=short_restrict, direction="short")
    obs = pd.concat([obs_l, obs_s], ignore_index=True)
    obs = obs[(obs["signal_time"] >= IS_START) & (obs["signal_time"] <= IS_END)].copy()
    obs["year"] = obs["signal_time"].dt.year
    obs = obs.merge(meta, on=["pair", "signal_time"], how="left")
    return obs


def report(obs, *, label: str):
    print(f"\n{'='*78}\n{label}  | n={len(obs)} | "
          f"capture {obs['capture'].mean():.4f} drift_mean {obs['fwd_drift_atr'].mean():+.4f} "
          f"drift_med {obs['fwd_drift_atr'].median():+.4f}")
    yr = obs.groupby("year").agg(n=("capture", "size"), capture=("capture", "mean"),
                                 drift=("fwd_drift_atr", "mean"))
    print(yr.to_string())
    neg = int((yr["drift"] < 0).sum())
    print(f"  per-year drift: neg-years {neg}/{len(yr)}  worst {yr['drift'].min():+.4f}  "
          f"mean-of-yr {yr['drift'].mean():+.4f}")
    strong = yr.reindex(STRONG_USD_FOLDS)
    sp = int((strong["drift"] > 0).sum())
    print(f"  STRONG-USD folds {STRONG_USD_FOLDS}: drift "
          f"{[f'{v:+.3f}' if pd.notna(v) else 'NA' for v in strong['drift']]}  "
          f"-> {sp}/4 positive  (the regime-orthogonal test: need 2015 & 2016 & 2018 positive)")
    return yr


def main():
    panel = Panel.from_pairs(PAIRS, tf="D1", histdata_root=HIST, cache_root=CACHE,
                             boundary_convention="5ers_eet")
    pd.set_option("display.width", 200, "display.max_columns", 30)

    print("### arc 1060: POST-month-end TREND RESUMPTION as the regime-orthogonal 5th leg ###")
    print("IS 2010-2020, D1, 7 USD majors. Enter at k-th trading day of new month in prevailing-trend")
    print("direction; honest take-the-loss capture; OOS untouched. KILL unless reliably +ve in 2015&2016&2018.")

    # default: enter day 3 of new month (flow cleared), 60-bar trend, strong = |ret|>=1 ATR
    base = build_obs(panel, k_entry=3, trend_lb=60, strong_atr=1.0, hold=20, drift_bars=10)
    yr_base = report(base, label="(default) k=3 entry | 60d trend | strong>=1ATR | hold20 drift10")

    # sweep the levers (cheap-obs robustness, NOT optimization -- just see if ANY cell is regime-orthogonal)
    for (k, lb, s, h, d) in [
        (1, 60, 1.0, 20, 10),   # enter day 1 (no flow-clear wait)
        (5, 60, 1.0, 20, 10),   # enter day 5 (more clearance)
        (3, 20, 1.0, 20, 10),   # shorter trend lookback
        (3, 60, 0.0, 20, 10),   # ALL trends (no strength filter)
        (3, 60, 1.5, 20, 10),   # only very strong trends
        (3, 60, 1.0, 10, 5),    # shorter hold
    ]:
        o = build_obs(panel, k_entry=k, trend_lb=lb, strong_atr=s, hold=h, drift_bars=d)
        report(o, label=f"k={k} | {lb}d trend | strong>={s}ATR | hold{h} drift{d}")

    # (B) complement test: correlate resumption per-year drift vs me_long per-year drift sign
    # (me_long negative years are 2014/2015/2016 per the docstring; a complement is +ve there)
    print(f"\n{'='*78}\n(B) COMPLEMENT vs me_long: me_long neg IS years = 2014/2015/2016 (docstring).")
    print("  resumption (default) drift in those years:",
          [f"{yr_base['drift'].get(y, float('nan')):+.3f}" for y in [2014, 2015, 2016]])


if __name__ == "__main__":
    main()
