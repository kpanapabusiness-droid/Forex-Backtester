"""arc 2065 — market-neutral cross-sectional momentum SPREAD (long top-k / short bottom-k).

WHY (because): the 4-way reversion book's binding wall is 2015 & 2018 (strong-USD TREND years) — the
reversion legs die there because the move CONTINUES (arc 1059: me_long capture monotone-down with
USD-breadth; arc 1060: trend-resumption also dies, 2015 is a whipsaw). Every prior 5th-leg attempt was a
single-instrument directional bet (long OR short), all coin-flip-or-worse, and NONE was positive in BOTH
2015 & 2018. The ONE construction NEVER run as a two-leg book is the LESSONS #1 frontier item:
cross-sectional momentum done DOLLAR-NEUTRAL (long the strongest pairs, short the weakest). It is the only
lever that does not require beating 0.50 per trade — a RELATIVE bet. arc 1000 killed cross-sectional
momentum LONG-ONLY (top-quintile); arc 2003 found relative perf PERSISTS (laggards keep lagging) which
SUPPORTS a long-winner/short-loser spread. The market-neutral spread nets the common (USD) factor, so what
remains is the persistent relative dispersion — and in a strongly-trending-USD year (2015/2018) the spread
should naturally express that trend (long USDxxx-risers / short xxxUSD-fallers) => candidate REGIME-ORTHOGONAL
5th leg, positive precisely where the reversion book is negative.

OBSERVATION ONLY (gross spread of forward log-returns, mid prices, causal ranking; NO engine, NO P&L claim,
NO cost netting) -> §5d cheap-kill screen. IS 2010-2020 ONLY; OOS (2021+) NEVER touched here. If the gross
spread is mean-positive AND positive in 2015/2018, escalate to the honest engine + FundedNext costs (§5f/g)
before ANY disposition above KILL. If coin-flip / 2015 or 2018 negative -> cheap KILL.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.sim.panel import Panel
from discovery.tools.observe_long_capture import observe_long_capture

HIST = r"C:\Users\panap\histdata_backup"
CACHE = "data/cache"

# D1-cached liquid cross-section (13 pairs spanning USD majors + crosses)
UNIVERSE = [
    "AUDJPY", "AUDNZD", "AUDUSD", "EURAUD", "EURGBP", "EURJPY", "EURUSD",
    "GBPJPY", "GBPUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY",
]


def load_mid(panel) -> pd.DataFrame:
    """Aligned mid-close matrix (rows=D1 bars, cols=pairs)."""
    cols = {}
    for p in panel.pairs:
        df = panel.pair_dfs[p]
        cols[p] = (df["close_bid"] + df["close_ask"]) / 2.0
    mid = pd.DataFrame(cols).sort_index()
    return mid


def spread_obs(mid: pd.DataFrame, L: int, h: int, k: int, label: str):
    """Long top-k / short bottom-k by trailing L-bar log-return; forward h-bar spread. Causal."""
    logp = np.log(mid)
    n = len(mid)
    rows = []
    # rebalance every h bars; require L history behind and h future ahead
    for t in range(L, n - h, h):
        trail = logp.iloc[t] - logp.iloc[t - L]          # causal: ranking uses info up to bar t
        fwd = logp.iloc[t + h] - logp.iloc[t]            # forward return realized AFTER ranking
        valid = trail.dropna().index.intersection(fwd.dropna().index)
        if len(valid) < 2 * k:
            continue
        tr = trail[valid].sort_values()
        longs = tr.index[-k:]                            # top-k trailing return
        shorts = tr.index[:k]                            # bottom-k trailing return
        long_fwd = fwd[longs].mean()
        short_fwd = fwd[shorts].mean()
        spread = long_fwd - short_fwd                    # book gross return (long winners, short losers)
        rows.append({
            "date": mid.index[t], "year": mid.index[t].year,
            "long_fwd": long_fwd, "short_fwd": short_fwd, "spread": spread,
        })
    res = pd.DataFrame(rows)
    res = res[res["year"] <= 2020]                       # IS ONLY
    n_reb = len(res)
    print(f"\n=== {label}  L={L} h={h} k={k}  (IS 2010-2020, rebalance every {h}d) ===")
    if n_reb == 0:
        print("  no rebalances"); return res
    mean_sp = res["spread"].mean()
    med_sp = res["spread"].median()
    frac_pos = (res["spread"] > 0).mean()
    # rough per-leg gross magnitude vs a ~weekly cost hurdle (info only; engine nets honestly)
    print(f"  n_rebal={n_reb}  spread mean={mean_sp*100:+.4f}%  median={med_sp*100:+.4f}%  frac_pos={frac_pos:.3f}")
    print(f"  long-leg fwd mean={res['long_fwd'].mean()*100:+.4f}%   short-leg fwd mean={res['short_fwd'].mean()*100:+.4f}%"
          f"   (short PROFITS when this is negative)")
    yr = res.groupby("year").agg(n=("spread", "size"), sp=("spread", "mean"),
                                 longf=("long_fwd", "mean"), shortf=("short_fwd", "mean"))
    print("  per-year spread mean  [binding folds: 2015, 2018]:")
    pos_years = 0
    for y, r in yr.iterrows():
        flag = "  <-- BIND" if y in (2015, 2018) else ""
        if r["sp"] > 0:
            pos_years += 1
        print(f"    {y}: spread={r['sp']*100:+.4f}%  (long {r['longf']*100:+.3f}% / short {r['shortf']*100:+.3f}%)  n={int(r['n'])}{flag}")
    print(f"  years spread>0: {pos_years}/{len(yr)}   "
          f"2015={'POS' if yr.loc[2015,'sp']>0 else 'NEG'}  2018={'POS' if yr.loc[2018,'sp']>0 else 'NEG'}"
          if (2015 in yr.index and 2018 in yr.index) else "  (binding folds missing)")
    return res


def reversal_check(mid: pd.DataFrame, L: int, h: int, k: int, label: str):
    """Reversal mirror = -momentum spread (long bottom-k / short top-k). mean>0 by construction;
    the question is mean-vs-median (fat-tail mirage) + binding-fold robustness + leave-one-year-out."""
    logp = np.log(mid)
    n = len(mid)
    rows = []
    for t in range(L, n - h, h):
        trail = logp.iloc[t] - logp.iloc[t - L]
        fwd = logp.iloc[t + h] - logp.iloc[t]
        valid = trail.dropna().index.intersection(fwd.dropna().index)
        if len(valid) < 2 * k:
            continue
        tr = trail[valid].sort_values()
        losers, winners = tr.index[:k], tr.index[-k:]      # reversal: long losers, short winners
        rev = fwd[losers].mean() - fwd[winners].mean()     # = -(momentum spread)
        rows.append({"year": mid.index[t].year, "rev": rev})
    res = pd.DataFrame(rows)
    res = res[res["year"] <= 2020]
    mean_r, med_r = res["rev"].mean(), res["rev"].median()
    frac = (res["rev"] > 0).mean()
    print(f"\n=== {label}  L={L} h={h} k={k}  (IS 2010-2020) ===")
    print(f"  n_rebal={len(res)}  mean={mean_r*100:+.4f}%  median={med_r*100:+.4f}%  frac_pos={frac:.3f}"
          f"   -> {'FAT-TAIL MIRAGE (mean>0,med<0)' if mean_r>0 and med_r<0 else 'broad' if med_r>0 else 'mean-neg'}")
    yr = res.groupby("year")["rev"].mean()
    pos_bind = sum(1 for y in (2015, 2018) if y in yr.index and yr[y] > 0)
    print(f"  binding folds: 2015={yr.get(2015, float('nan'))*100:+.3f}%  2018={yr.get(2018, float('nan'))*100:+.3f}%  ({pos_bind}/2 pos)")
    # leave-one-year-out on the mean: is the positive mean carried by ONE year?
    loo = {y: res[res["year"] != y]["rev"].mean() for y in yr.index}
    worst_y = min(loo, key=lambda y: loo[y])
    print(f"  leave-one-year-out mean range: [{min(loo.values())*100:+.4f}% (drop {worst_y}), "
          f"{max(loo.values())*100:+.4f}%]  -> {'SIGN-FRAGILE' if min(loo.values())<=0 else 'sign-robust'}")
    # rough cost hurdle: 2k legs/rebalance, ~1 pip (~0.01%) round-turn each on a major
    print(f"  rough cost hurdle ~{2*k}legs x ~0.01% = ~{2*k*0.01:.2f}% per rebalance vs gross mean {mean_r*100:+.4f}% "
          f"-> {'SUB-COST' if mean_r < 2*k*0.01/100 else 'clears rough hurdle'}")
    return res


def reversal_masks(panel, mid: pd.DataFrame, L: int, h: int, k: int):
    """Per-pair LONG (bottom-k trailing) and SHORT (top-k trailing) rebalance masks for the reversal
    book, aligned to each pair's own bar index. Causal: ranking uses trailing-L return at bar t."""
    logp = np.log(mid)
    n = len(mid)
    long_ts = {p: [] for p in panel.pairs}   # timestamps where pair is a LONG (loser) fire
    short_ts = {p: [] for p in panel.pairs}
    for t in range(L, n - 1, h):
        trail = (logp.iloc[t] - logp.iloc[t - L]).dropna()
        if len(trail) < 2 * k:
            continue
        tr = trail.sort_values()
        ts = mid.index[t]
        for p in tr.index[:k]:
            long_ts[p].append(ts)
        for p in tr.index[-k:]:
            short_ts[p].append(ts)
    long_masks, short_masks = {}, {}
    for p in panel.pairs:
        idx = panel.pair_dfs[p].index
        lm = np.zeros(len(idx), bool); sm = np.zeros(len(idx), bool)
        pos = {ts: i for i, ts in enumerate(idx)}
        for ts in long_ts[p]:
            if ts in pos:
                lm[pos[ts]] = True
        for ts in short_ts[p]:
            if ts in pos:
                sm[pos[ts]] = True
        long_masks[p] = lm; short_masks[p] = sm
    return long_masks, short_masks


def honest_reversal_screen(panel, mid, L, h, k, label):
    """Honest TAKE-THE-LOSS capture (+1R-before-2ATR-SL) of the reversal book legs — the lens the
    gross-drift obs is blind to. If the gross +0.22% is a fat-tail mirage it COLLAPSES here."""
    lm, sm = reversal_masks(panel, mid, L, h, k)
    lo = observe_long_capture(panel, sl_mult=2.0, hold=h, drift_bars=h, restrict=lm, direction="long")
    sh = observe_long_capture(panel, sl_mult=2.0, hold=h, drift_bars=h, restrict=sm, direction="short")
    lo = lo[lo["signal_time"].dt.year <= 2020].copy(); lo["year"] = lo["signal_time"].dt.year
    sh = sh[sh["signal_time"].dt.year <= 2020].copy(); sh["year"] = sh["signal_time"].dt.year
    both = pd.concat([lo, sh], ignore_index=True)
    print(f"\n=== HONEST TAKE-THE-LOSS reversal screen  {label}  L={L} h={h} k={k} (IS) ===")
    print(f"  LONG legs (losers):  n={len(lo)}  capture={lo['capture'].mean():.4f}  drift={lo['fwd_drift_atr'].mean():+.4f} (median {lo['fwd_drift_atr'].median():+.4f})")
    print(f"  SHORT legs (winners): n={len(sh)}  capture={sh['capture'].mean():.4f}  drift={sh['fwd_drift_atr'].mean():+.4f} (median {sh['fwd_drift_atr'].median():+.4f})")
    print(f"  POOLED book: n={len(both)}  capture={both['capture'].mean():.4f}  "
          f"drift mean={both['fwd_drift_atr'].mean():+.4f}  median={both['fwd_drift_atr'].median():+.4f}  "
          f"-> {'>0.50 SURVIVES take-the-loss' if both['capture'].mean()>0.50 else 'COLLAPSES sub-0.50 (gross was mirage)'}")
    yr = both.groupby("year").agg(cap=("capture", "mean"), drift=("fwd_drift_atr", "mean"), n=("capture", "size"))
    print("  per-year pooled (cap / drift / n)  [binding 2015, 2018]:")
    for y, r in yr.iterrows():
        flag = "  <-- BIND" if y in (2015, 2018) else ""
        print(f"    {y}: {r['cap']:.3f} / {r['drift']:+.3f} / {int(r['n'])}{flag}")
    return both


def main():
    print(f"Loading D1 universe ({len(UNIVERSE)} pairs)...")
    panel = Panel.from_pairs(UNIVERSE, tf="D1", histdata_root=HIST, cache_root=CACHE,
                             boundary_convention="5ers_eet")
    mid = load_mid(panel)
    print(f"  mid matrix: {mid.shape[0]} bars x {mid.shape[1]} pairs, "
          f"{mid.index.min().date()} .. {mid.index.max().date()}")

    # Primary cell + robustness sweep (confirm SIGN-robustness, not optimize)
    spread_obs(mid, L=60, h=5, k=4, label="XSEC-MOM SPREAD (primary)")
    spread_obs(mid, L=20, h=5, k=4, label="XSEC-MOM SPREAD (short lookback)")
    spread_obs(mid, L=120, h=10, k=4, label="XSEC-MOM SPREAD (long lookback, biweekly)")
    spread_obs(mid, L=60, h=20, k=3, label="XSEC-MOM SPREAD (monthly hold, tighter k)")
    spread_obs(mid, L=60, h=5, k=3, label="XSEC-MOM SPREAD (k=3)")

    # §5f best-version check: the MIRROR. momentum is gross-NEG (cross-section reverts), so the
    # REVERSAL spread (long losers / short winners) is gross-positive by construction. Is it a real
    # edge or a fat-tail mirage? mean>0 but median<0 (the arc-2011/2058 tell) + leave-one-year-out.
    reversal_check(mid, L=60, h=5, k=4, label="XSEC-REVERSAL MIRROR (primary)")
    reversal_check(mid, L=60, h=20, k=3, label="XSEC-REVERSAL MIRROR (monthly, lower turnover)")

    # the decisive honest screen on the best reversal cell (monthly, broad gross median)
    honest_reversal_screen(panel, mid, L=60, h=20, k=3, label="monthly reversal (best gross cell)")

    # §5f SL-dimension sweep (cheap, observation-level): does ANY SL multiple clear 0.50 on the
    # pooled book + each leg? If not, the gross drift is unharvestable -> §5f-honest KILL.
    print("\n=== §5f SL-multiple sweep on the monthly reversal book (honest capture) ===")
    lm, sm = reversal_masks(panel, mid, L=60, h=20, k=3)
    for slm in (1.0, 1.5, 2.0, 2.5):
        lo = observe_long_capture(panel, sl_mult=slm, hold=20, restrict=lm, direction="long")
        sh = observe_long_capture(panel, sl_mult=slm, hold=20, restrict=sm, direction="short")
        lo = lo[lo["signal_time"].dt.year <= 2020]; sh = sh[sh["signal_time"].dt.year <= 2020]
        pooled = pd.concat([lo, sh])["capture"].mean()
        print(f"  SL={slm}: long(losers) {lo['capture'].mean():.4f} / short(winners) {sh['capture'].mean():.4f} "
              f"/ POOLED {pooled:.4f}  {'<-- clears 0.50' if pooled>0.50 else ''}")


if __name__ == "__main__":
    main()
