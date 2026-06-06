"""Arc 1062 — DIRECTION-AGNOSTIC COMPRESSION-EXPANSION convexity leg (the long-vol 5th-leg candidate).

THE IDEA (a genuinely-untested construction aimed squarely at the proven binding constraint).
The 4-way reversion book (gap/me_long/me_short/fbr) is never all-folds-positive because of the
strong-USD WALL: its two binding negative folds are 2015 and 2018. Arc 1060 sharpened *why* they are
hard and *different*: 2015 = CHF-de-peg + choppy-then-trending whipsaw; 2018 = persistent strong-USD
trend. Arc 2022 proved no single +2015&+2018 leg survives the weighting dilemma when that leg is a
DIRECTIONAL one (shock-continuation 3019, a post-confirmation directional vol-expansion entry, gets
throttled to an overfit-tiny weight). The ONE thing 2015 and 2018 SHARE is ELEVATED REALIZED
VOLATILITY (both are high-vol years relative to 2014/2016/2017): a de-peg shock and a trend are both
big-range regimes.

So the untested construction is a LONG-VOLATILITY / CONVEXITY leg that does NOT pick direction in
advance: a multi-bar COMPRESSION (low-ATR coil) that BREAKS, taken in WHICHEVER direction it breaks
(long the up-break, short the down-break — a single-leg straddle analog now that shorts are enabled).
The *because*: a compression-expansion convexity exposure is positive precisely when realized vol
SPIKES — i.e. in 2015 AND 2018 — regardless of direction, so it is the regime-orthogonal complement of
the reversion book (positive in the book's negative folds).

WHY this is NOT closed ground (§5a):
  - It is NOT long-only arc-1001 (vol-contraction breakout LONG, killed at triage): that could only
    catch up-breaks; this is the DIRECTION-AGNOSTIC version shorts now unlock — the symmetric straddle
    the long-only test structurally could not be.
  - It is NOT arc-2022's shock-continuation: that enters AFTER a confirmed big bar in that bar's
    direction (directional momentum-of-a-shock). This enters AT the break of a tight coil, BEFORE the
    expansion fully forms — capturing the convexity of the expansion itself, not continuation of a
    completed move.
  - The KILL/PROCEED criterion is NOT "does it beat cost overall" (a breakout leg is coin-flip across
    all years by construction). It is the regime-orthogonal test: is it RELIABLY POSITIVE in the two
    binding folds 2015 AND 2018? A coin-flip-overall leg is STILL the 5th leg IF its positive years are
    exactly the book's negative years.

HYPOTHESIS (falsifiable, mechanistic).
  (A) compression-break (either direction) has POSITIVE honest take-the-loss capture / fwd-drift in the
      binding folds 2015 AND 2018 (the high-realized-vol years);
  (B) it is more positive when the coil is TIGHTER (lower ATR-ratio) — the genuine compression setup —
      than at a loose "break" (which is just a shallow breakout = closed momentum);
  (C) its positive years are concentrated in high-vol folds, not uniform (regime concentration, not a
      flat coin-flip).
KILL if 2015 OR 2018 is NOT positive, or if tighter-coil does not separate from loose-break (no
compression signal, just closed breakout), or if it is a flat coin-flip every year. PROCEED to honest-
engine triage only if it is reliably positive in BOTH binding folds with a genuine compression gradient.

OBSERVATION cheap-kill (§5d): honest take-the-loss capture + fwd-drift (BUILT observe_long_capture,
direction-aware), IS 2010-2020, H4, 7 USD majors. No engine / null / council here; OOS untouched.
Driver is single-use BUILT-tools-only; no canonical change; OOS-preserving.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.features._helpers import mid_close, mid_high, mid_low, wilder_atr
from core.sim.panel import Panel
from discovery.tools.observe_long_capture import observe_long_capture

HIST = r"C:\Users\panap\histdata_backup"
CACHE = "data/cache"
PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCAD", "USDCHF"]
IS_START, IS_END = pd.Timestamp("2010-01-01", tz="UTC"), pd.Timestamp("2020-12-31", tz="UTC")

BINDING_FOLDS = [2015, 2018]            # the book's two binding negative folds (arc 1020/1060)
STRONG_USD_FOLDS = [2014, 2015, 2016, 2018]


def build_obs(panel, *, coil_lb: int, atr_long: int, comp_thresh: float, break_lb: int,
              hold: int, drift_bars: int):
    """For each pair flag a CAUSAL compression-break at bar t:
       - compression: (coil_lb-bar high-low range)/(atr_long-bar ATR) <= comp_thresh  (tight coil)
       - up-break:   close_t > max(high over the prior break_lb bars, EXCLUDING t)  -> long next bar
       - down-break: close_t < min(low  over the prior break_lb bars, EXCLUDING t)  -> short next bar
    All quantities use info available AT bar t (range/ATR through t; breakout level from bars < t).
    Returns direction-split restrict masks + a meta frame carrying coil-tightness for the gradient test.
    """
    long_restrict, short_restrict = {}, {}
    meta_rows = []
    warm = max(coil_lb, atr_long, break_lb) + 20
    for pair in PAIRS:
        df = panel.pair_dfs[pair]
        idx = df.index
        n = len(df)
        hi, lo, mc = mid_high(df), mid_low(df), mid_close(df)
        hv, lv, cv = hi.values, lo.values, mc.values
        atr_l = wilder_atr(hi, lo, mc, atr_long).shift(1).values    # causal long ATR (vol scale)
        # coil range over the trailing coil_lb bars INCLUDING t (known at t's close), in ATR units
        rng = (hi.rolling(coil_lb).max() - lo.rolling(coil_lb).min()).values
        coil_ratio = rng / atr_l   # coil width in ATR units (~2-8); tighter coil = smaller ratio
        # breakout reference from the prior break_lb bars EXCLUDING t (shift(1))
        prior_hi = hi.rolling(break_lb).max().shift(1).values
        prior_lo = lo.rolling(break_lb).min().shift(1).values
        compressed = np.isfinite(coil_ratio) & (coil_ratio <= comp_thresh)
        up = compressed & (cv > prior_hi)
        dn = compressed & (cv < prior_lo)
        long_restrict[pair] = up
        short_restrict[pair] = dn
        for t in np.where(up | dn)[0]:
            if t < warm:
                continue
            meta_rows.append({"pair": pair, "signal_time": idx[t],
                              "coil_ratio": float(coil_ratio[t]),
                              "side": "long" if up[t] else "short"})
    meta = pd.DataFrame(meta_rows)

    obs_l = observe_long_capture(panel, sl_mult=2.0, hold=hold, drift_bars=drift_bars, warmup=warm,
                                 restrict=long_restrict, direction="long")
    obs_s = observe_long_capture(panel, sl_mult=2.0, hold=hold, drift_bars=drift_bars, warmup=warm,
                                 restrict=short_restrict, direction="short")
    obs = pd.concat([obs_l, obs_s], ignore_index=True)
    obs = obs[(obs["signal_time"] >= IS_START) & (obs["signal_time"] <= IS_END)].copy()
    obs["year"] = obs["signal_time"].dt.year
    obs = obs.merge(meta, on=["pair", "signal_time"], how="left")
    return obs


def report(obs, *, label: str):
    print(f"\n{'='*80}\n{label}  | n={len(obs)} | "
          f"capture {obs['capture'].mean():.4f}  drift_mean {obs['fwd_drift_atr'].mean():+.4f}  "
          f"drift_med {obs['fwd_drift_atr'].median():+.4f}")
    if len(obs) == 0:
        print("  (no fires)")
        return None
    yr = obs.groupby("year").agg(n=("capture", "size"), capture=("capture", "mean"),
                                 drift=("fwd_drift_atr", "mean"))
    print(yr.to_string())
    neg = int((yr["drift"] < 0).sum())
    print(f"  per-year drift: neg-years {neg}/{len(yr)}  worst {yr['drift'].min():+.4f}  "
          f"mean-of-yr {yr['drift'].mean():+.4f}")
    binding = yr.reindex(BINDING_FOLDS)
    bp = int((binding["drift"] > 0).sum())
    print(f"  BINDING folds {BINDING_FOLDS}: drift "
          f"{[f'{v:+.3f}' if pd.notna(v) else 'NA' for v in binding['drift']]} cap "
          f"{[f'{c:.3f}' if pd.notna(c) else 'NA' for c in binding['capture']]}"
          f"  -> {bp}/2 positive  (regime-orthogonal test: need BOTH 2015 & 2018 positive)")
    return yr


def main():
    panel = Panel.from_pairs(PAIRS, tf="H4", histdata_root=HIST, cache_root=CACHE,
                             boundary_convention="5ers_eet")
    pd.set_option("display.width", 220, "display.max_columns", 30)

    print("### arc 1062: DIRECTION-AGNOSTIC COMPRESSION-EXPANSION convexity leg ###")
    print("IS 2010-2020, H4, 7 USD majors. Tight coil that BREAKS -> trade the break direction;")
    print("honest take-the-loss capture; OOS untouched. KILL unless reliably +ve in BOTH 2015 & 2018.")

    # default: 12-bar coil (~2 days H4), width <= 3.5 ATR (tight), 12-bar breakout ref
    base = build_obs(panel, coil_lb=12, atr_long=60, comp_thresh=3.5, break_lb=12, hold=24, drift_bars=12)
    yr_base = report(base, label="(default) coil12 thr3.5ATR break12 hold24 drift12")

    # sweep the levers (cheap-obs robustness, NOT optimization -- is ANY cell regime-orthogonal?)
    for (cl, thr, bl, h, d) in [
        (12, 2.5, 12, 24, 12),    # TIGHTER coil (compression gradient test, (B))
        (12, 6.0, 12, 24, 12),    # LOOSER coil (toward plain breakout -- should be WORSE if (B) holds)
        (8,  3.5,  8, 24, 12),    # shorter coil + break
        (20, 3.5, 20, 24, 12),    # longer coil + break
        (12, 3.5, 12, 12,  6),    # shorter hold/drift (faster convexity capture)
        (12, 3.5, 12, 48, 24),    # longer hold (let the expansion run)
    ]:
        report(build_obs(panel, coil_lb=cl, atr_long=60, comp_thresh=thr, break_lb=bl, hold=h, drift_bars=d),
               label=f"coil{cl} thr{thr} break{bl} hold{h} drift{d}")

    # (B) compression gradient: split the DEFAULT fires into tightest vs loosest coil tercile
    if yr_base is not None and len(base):
        q = base["coil_ratio"].quantile([1/3, 2/3]).values
        tight = base[base["coil_ratio"] <= q[0]]
        loose = base[base["coil_ratio"] >= q[1]]
        print(f"\n{'='*80}\n(B) COMPRESSION GRADIENT (default fires): does a TIGHTER coil separate?")
        print(f"  tight-tercile (coil<= {q[0]:.3f}) n={len(tight)} cap {tight['capture'].mean():.4f} "
              f"drift {tight['fwd_drift_atr'].mean():+.4f}")
        print(f"  loose-tercile (coil>= {q[1]:.3f}) n={len(loose)} cap {loose['capture'].mean():.4f} "
              f"drift {loose['fwd_drift_atr'].mean():+.4f}")

    # (C) strong-USD-fold concentration of the DEFAULT cell
    if yr_base is not None:
        s = yr_base.reindex(STRONG_USD_FOLDS)
        print(f"\n(C) STRONG-USD folds {STRONG_USD_FOLDS} drift "
              f"{[f'{v:+.3f}' if pd.notna(v) else 'NA' for v in s['drift']]}  "
              f"(regime concentration check)")


if __name__ == "__main__":
    main()
