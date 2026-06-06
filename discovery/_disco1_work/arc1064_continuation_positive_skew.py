"""Arc 1064 — CONTINUATION / POSITIVE-SKEW shape family (the one untested LENS).

COUNCIL-SURFACED FORWARD THREAD (arc 1063 thread (b), logged-not-executed). The whole honest-era
corpus has judged every entry by ONE of two lenses: (i) +1R-before-SL CAPTURE (a >0.50 win-rate
test) or (ii) fixed-horizon mean forward-drift. BOTH lenses structurally PENALIZE a positive-skew
CONTINUATION edge: a trend-follower's real profile is capture < 0.50 (loses small often — take-the-
loss caps each miss at -1R) but POSITIVE EXPECTANCY because the few winners RUN FAR under a let-it-
run trailing exit. Such an entry is cheap-killed at the capture stage ("capture < 0.50 -> coin-flip")
*before* its final-R expectancy under a runner exit is ever measured. That lens-gap is the genuinely
untested ground: the corpus tested ONLY reversion shapes (where >0.50 capture IS the edge).

WHY this is NOT a re-run of closed ground (§5a):
  - arc 2000 (Donchian + full-size trailing) killed UNCONDITIONAL Donchian at engine triage by
    WORST-FOLD ROI ("fat tail generic not trend-selected"); it did NOT apply the lens-corrected
    median-per-fold + tail-removed-expectancy guard, and it did NOT trend-FILTER the breakout.
  - arc 2012 (deep continuation-long) killed by CAPTURE / structure-control inversion.
  - arc 3019 (shock-continuation) is a DIFFERENT shape (post-extreme-shock); it reached the engine
    and IS-passed under tp_3r but OOS-epoch-failed.
  - arc 1062 (compression-expansion straddle) killed at CAPTURE (0.45 < null) -> never reached the
    exit sweep. THAT is exactly the lens-gap: a capture cheap-kill is NOT a valid kill for a skew edge.
  This arc takes the TEXTBOOK trend-CONTINUATION entry (a fresh Donchian break inside an established
  trend — the canonical CTA positive-skew entry, BOTH directions now shorts are enabled) and judges it
  by the CORRECT lens for skew: the full forward-return DISTRIBUTION, not capture.

OBSERVATION SCREEN (this file; §5d, BUILT tools only, no engine/null/council, OOS untouched):
  fwd_drift_atr is GROSS (ignores the stop), so it is a GENEROUS upper-ish proxy for a runner edge —
  if even the gross distribution fails the guard, the honest take-the-loss engine is strictly worse
  (clean cheap-kill). Use a LONG drift horizon (60 H4 bars ~= 10 trading days) so a genuine runner
  is not truncated by the measurement window.

PRE-REGISTERED GUARD (thread (b)'s discipline against relabeling thin-tail luck as "positive skew"):
  (G1) mean fwd_drift > 0  (gross-positive — the §5f trigger to spend the engine on the exit sweep);
  (G2) NOT a 1-2-jackpot mirage: tail-removed mean (drop top 5% of fwd_drift) still > 0
       AND per-year drift positive in a MAJORITY of folds;
  (G3) DISTRIBUTED: positive across a majority of the 7 pairs (not single-pair carry).
  KILL at obs if G1 fails (no gross drift + capture<0.50 = genuine coin-flip-or-worse, §5d).
  KILL at obs if G1 holds but G2/G3 fail (the positive mean is a thin-tail artifact — the capture
  lens was right after all; confirms 2000/2011/3014/1062's thin-tail finding on the skew axis).
  PROCEED to honest-engine §5f trailing-exit sweep ONLY if G1+G2+G3 all hold (a skew edge the capture
  lens would have missed).

IS 2010-2020, H4, 7 USD majors (the structural TF/universe where the corpus's continuation/structural
mechanisms live). Driver single-use, BUILT-tools-only; no canonical change; OOS-preserving.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from core.features._helpers import mid_close, mid_high, mid_low, wilder_atr
from core.sim.panel import Panel
from discovery.tools.observe_long_capture import observe_long_capture

HIST = r"C:\Users\panap\histdata_backup"
CACHE = "data/cache"
PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCAD", "USDCHF"]
IS_START, IS_END = pd.Timestamp("2010-01-01", tz="UTC"), pd.Timestamp("2020-12-31", tz="UTC")
BINDING_FOLDS = [2015, 2018]


def _sma(s: pd.Series, n: int) -> np.ndarray:
    return s.rolling(n).mean().values


def build_obs(panel, *, donch: int, sma_fast: int, sma_slow: int, hold: int, drift_bars: int):
    """CAUSAL trend-continuation entries at bar t (entry next bar via observe_long_capture):
       LONG : established uptrend (mc>SMA_slow & SMA_fast>SMA_slow) AND fresh Donchian-`donch` HIGH break
              (mc[t] > max high over prior `donch` bars excl. t, and t-1 was NOT above -> crossing bar)
       SHORT: established downtrend (mc<SMA_slow & SMA_fast<SMA_slow) AND fresh Donchian-`donch` LOW break
    All quantities known at t's close. Returns direction-split restrict masks + meta(side)."""
    long_restrict, short_restrict, meta_rows = {}, {}, []
    warm = max(donch, sma_slow) + 20
    for pair in PAIRS:
        df = panel.pair_dfs[pair]
        idx = df.index
        hi, lo, mc = mid_high(df), mid_low(df), mid_close(df)
        cv = mc.values
        sma_f, sma_s = _sma(mc, sma_fast), _sma(mc, sma_slow)
        prior_hi = hi.rolling(donch).max().shift(1).values     # highest high of prior `donch` bars
        prior_lo = lo.rolling(donch).min().shift(1).values
        up_trend = (cv > sma_s) & (sma_f > sma_s)
        dn_trend = (cv < sma_s) & (sma_f < sma_s)
        above = cv > prior_hi
        below = cv < prior_lo
        prev_above = np.r_[False, above[:-1]]
        prev_below = np.r_[False, below[:-1]]
        up = up_trend & above & (~prev_above)                  # fresh up-break in an uptrend
        dn = dn_trend & below & (~prev_below)                  # fresh down-break in a downtrend
        long_restrict[pair] = up
        short_restrict[pair] = dn
        for t in np.where(up | dn)[0]:
            if t < warm:
                continue
            meta_rows.append({"pair": pair, "signal_time": idx[t],
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
    d = obs["fwd_drift_atr"].dropna()
    n = len(obs)
    print(f"\n{'='*88}\n{label}  | n={n}")
    if n == 0 or len(d) == 0:
        print("  (no fires)")
        return
    cap = obs["capture"].mean()
    mean_d, med_d = d.mean(), d.median()
    skew = sps.skew(d.values) if len(d) > 2 else float("nan")
    # tail-removed mean: drop the top 5% of fwd_drift (the jackpot winners)
    cut = d.quantile(0.95)
    tr_mean = d[d <= cut].mean()
    frac_big = float((d > 2.0).mean())          # winners running > 2 ATR
    frac_pos = float((d > 0).mean())
    print(f"  capture {cap:.4f}  (capture-lens would {'KILL' if cap < 0.50 else 'pass'}: base ~0.488)")
    print(f"  drift: mean {mean_d:+.4f}  median {med_d:+.4f}  skew {skew:+.3f}  "
          f"frac>0 {frac_pos:.3f}  frac>2ATR {frac_big:.3f}")
    print(f"  GUARD: G1 mean>0 {'PASS' if mean_d > 0 else 'FAIL'} | "
          f"tail-removed(drop top5%) mean {tr_mean:+.4f} ({'PASS' if tr_mean > 0 else 'FAIL'})")
    yr = obs.groupby("year").agg(n=("fwd_drift_atr", "size"),
                                 drift=("fwd_drift_atr", "mean"),
                                 drift_med=("fwd_drift_atr", "median"))
    print(yr.to_string())
    neg = int((yr["drift"] < 0).sum())
    medpos = int((yr["drift_med"] > 0).sum())
    print(f"  per-year: neg-mean-years {neg}/{len(yr)}  median>0 years {medpos}/{len(yr)}  "
          f"(G2 majority-fold needs >{len(yr)//2})")
    binding = yr.reindex(BINDING_FOLDS)
    print(f"  BINDING {BINDING_FOLDS}: drift "
          f"{[f'{v:+.3f}' if pd.notna(v) else 'NA' for v in binding['drift']]}")
    pp = obs.groupby("pair")["fwd_drift_atr"].mean()
    pppos = int((pp > 0).sum())
    print(f"  per-pair drift>0: {pppos}/{len(pp)} (G3 distributed)   "
          f"{ {k: round(float(v), 3) for k, v in pp.items()} }")


def main():
    panel = Panel.from_pairs(PAIRS, tf="H4", histdata_root=HIST, cache_root=CACHE,
                             boundary_convention="5ers_eet")
    pd.set_option("display.width", 240, "display.max_columns", 30)

    print("### arc 1064: CONTINUATION / POSITIVE-SKEW shape family (council thread 1063b) ###")
    print("Lens-corrected: judge a trend-CONTINUATION entry by the fwd-return DISTRIBUTION (skew,")
    print("median, tail-removed), NOT by +1R capture. fwd_drift is GROSS (generous proxy); engine §5f")
    print("only if the pre-registered G1+G2+G3 guard holds. IS 2010-2020, H4, 7 USD majors. OOS untouched.")

    # DEFAULT: Donchian-20 break inside an established trend (SMA50/SMA200), 60-bar drift (let it run)
    base = build_obs(panel, donch=20, sma_fast=50, sma_slow=200, hold=120, drift_bars=60)
    report(base, label="(default) Donchian20 trend(50/200) hold120 drift60")

    # robustness sweep (NOT optimization — is ANY cell a distributed positive-skew edge?)
    for (dn, sf, ss, h, dr) in [
        (40, 50, 200, 120, 60),     # slower breakout
        (20, 50, 200, 120, 120),    # even longer runner horizon
        (20, 50, 200, 120, 30),     # shorter horizon (faster continuation)
        (20, 20, 100, 120, 60),     # faster trend filter (more fires)
        (55, 50, 200, 120, 120),    # classic 55-bar Donchian (Turtle), long runner
    ]:
        report(build_obs(panel, donch=dn, sma_fast=sf, sma_slow=ss, hold=h, drift_bars=dr),
               label=f"Donchian{dn} trend({sf}/{ss}) hold{h} drift{dr}")


if __name__ == "__main__":
    main()
