"""arc 2010 — market-neutral relative-MOMENTUM observation (cheap screen, step (b)/(d)).

The arc-3004 escalation's #1 named unlock is relative-value / market-neutral (a 2nd simultaneous leg —
"the only lever that does NOT require beating 0.50 per trade"), which arc 2003 could only CONCEDE under
the long-only constraint now lifted (PR #273). arc 2003 found relative performance of correlated majors
is NOT mean-reverting — "the laggard keeps lagging" (relative MOMENTUM). So the tradeable side is
momentum, and it needs shorts to harvest market-neutrally (long the outperformer, short the
underperformer of a cointegrated pair). The because: persistent relative-strength ordering among
correlated currencies (cross-sectional FX momentum, Menkhoff et al.); positive when orderings PERSIST =
trending years (2018) = the regime-orthogonal 4th-component the portfolio route needs.

THE COST REALITY (the decisive question this observation answers): a market-neutral pair trade pays
FundedNext cost on BOTH legs (~2x the single-leg ~0.05-0.10R hurdle that the WHOLE corpus could not
clear directionally). So the relative-momentum drift must be LARGE (clear ~2x cost) to be viable. This
screen measures, honestly and ex-ante, whether relative-strength predicts forward relative-strength
(momentum) and how big the per-trade spread move is vs the 2-leg cost.

CHARACTERIZATION ONLY (gross). If the relative momentum is real AND plausibly clears 2x cost, the next
step builds the two-leg book (each leg a canonical single-pair signal scored by MultiPairBacktester,
combined via BUILT combine_fold_roi — additive, sidesteps multi-leg apparatus). Else cheap-kill.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.sim.panel import Panel

BACKUP = r"C:\Users\panap\histdata_backup"
# Cointegrated / common-bloc correlated pairs (both quoted XXX/USD so a common USD factor cancels in the
# spread). EURUSD-GBPUSD (Europe), AUDUSD-NZDUSD (commodity-Pacific) are the classic FX stat-arb pairs.
PAIR_GROUPS = [("EURUSD", "GBPUSD"), ("AUDUSD", "NZDUSD"), ("EURUSD", "AUDUSD")]
LOOKBACK = 20    # bars for the relative-strength ranking signal
FWD = 10         # forward bars for the relative move
ATR_P = 14


def logret_atr(panel: Panel, pair: str):
    df = panel.pair_dfs[pair]
    c = (df["close_bid"].to_numpy(float) + df["close_ask"].to_numpy(float)) / 2.0
    h = (df["high_bid"].to_numpy(float) + df["high_ask"].to_numpy(float)) / 2.0
    l = (df["low_bid"].to_numpy(float) + df["low_ask"].to_numpy(float)) / 2.0
    spread = (df["close_ask"].to_numpy(float) - df["close_bid"].to_numpy(float))
    logc = np.log(c)
    # ATR in log-return units (relative), Wilder
    tr = np.full(len(df), np.nan)
    pc = np.concatenate([[np.nan], c[:-1]])
    tr[1:] = np.maximum.reduce([(h - l)[1:], np.abs(h[1:] - pc[1:]), np.abs(l[1:] - pc[1:])])
    atr = np.full(len(df), np.nan)
    if len(df) > ATR_P:
        atr[ATR_P] = np.nanmean(tr[1:ATR_P + 1])
        for i in range(ATR_P + 1, len(df)):
            atr[i] = (atr[i - 1] * (ATR_P - 1) + tr[i]) / ATR_P
    return pd.Series(logc, index=df.index), pd.Series(c, index=df.index), \
        pd.Series(atr / c, index=df.index), pd.Series(spread / c, index=df.index)


def run_tf(tf: str, lookback: int, fwd: int) -> None:
    pairs = sorted({p for g in PAIR_GROUPS for p in g})
    panel = Panel.from_pairs(pairs, tf=tf, histdata_root=BACKUP, cache_root="data/cache",
                             boundary_convention="5ers_eet")
    lo, hi = pd.Timestamp("2010-01-01", tz="UTC"), pd.Timestamp("2020-12-31", tz="UTC")

    print(f"\n=== arc 2010 relative-MOMENTUM observation ({tf}, IS 2010-2020) ===")
    print(f"lookback={lookback} fwd={fwd}; relative-strength = logret_A(LB) - logret_B(LB)\n")
    global LOOKBACK, FWD
    LOOKBACK, FWD = lookback, fwd
    zwin = 250 if tf == "H4" else 120

    data = {p: logret_atr(panel, p) for p in pairs}
    for A, B in PAIR_GROUPS:
        logA, cA, relatrA, relspreadA = data[A]
        logB, cB, relatrB, relspreadB = data[B]
        df = pd.DataFrame({"logA": logA, "logB": logB,
                           "ratrA": relatrA, "ratrB": relatrB,
                           "rspA": relspreadA, "rspB": relspreadB}).dropna()
        df = df[(df.index >= lo) & (df.index <= hi)]
        # rolling relative strength (past LOOKBACK log-return diff), ex-ante (no shift needed: uses past)
        relstr = (df["logA"] - df["logA"].shift(LOOKBACK)) - (df["logB"] - df["logB"].shift(LOOKBACK))
        # forward relative move over next FWD bars (the spread momentum to harvest)
        fwd_rel = (df["logA"].shift(-FWD) - df["logA"]) - (df["logB"].shift(-FWD) - df["logB"])
        z = (relstr - relstr.rolling(zwin).mean()) / relstr.rolling(zwin).std()
        good = relstr.notna() & fwd_rel.notna() & z.notna()
        rs, fr, zz = relstr[good], fwd_rel[good], z[good]
        # momentum: does positive relstr predict positive fwd_rel?
        corr = np.corrcoef(rs, fr)[0, 1]
        # per-leg cost in the same (log/relative) units: ~1.5*spread/price + slippage+commission proxy.
        # Round-trip 2-leg cost ~ 2 * (1.5*rel_spread) (ignoring fixed commission/slippage here, gross screen).
        cost2leg = 2.0 * 1.5 * (df["rspA"].mean() + df["rspB"].mean()) / 2.0
        print(f"--- {A} vs {B} ---  n={len(rs)}  corr(relstr, fwd_rel)={corr:+.4f}")
        print(f"    ~2-leg gross cost (1.5x spread x2) ~= {cost2leg*1e4:.2f} bp")
        # condition on extreme |z|: trade only when relative-strength is strong (momentum regime)
        for thr in (1.0, 1.5, 2.0):
            ext = zz.abs() >= thr
            if ext.sum() < 50:
                print(f"    |z|>={thr}: n={int(ext.sum())} (thin)")
                continue
            # signed forward move in the DIRECTION of the relative strength (the momentum bet)
            signed = np.sign(zz[ext]) * fr[ext]
            print(f"    |z|>={thr}: n={int(ext.sum()):5d}  "
                  f"mean signed fwd-rel move = {signed.mean()*1e4:+.2f} bp  "
                  f"(vs ~{cost2leg*1e4:.2f} bp cost; net {(signed.mean()-cost2leg)*1e4:+.2f} bp)  "
                  f"frac+ {(signed>0).mean():.3f}")
    print("\nINTERPRET: net (signed move - 2-leg cost) must be clearly POSITIVE with frac+ > 0.5 to")
    print("justify building the two-leg book. Net<=0 or frac+~0.5 => relative momentum is sub-2x-cost.")
    print("(NOTE: cost shown is SPREAD-ONLY 1.5x; real FundedNext adds ~0.5pip slippage + $5/lot RT")
    print(" per leg => true 2-leg cost is HIGHER, so a marginal positive net here is still sub-cost.)")


if __name__ == "__main__":
    run_tf("H4", lookback=20, fwd=10)   # ~80h hold
    run_tf("D1", lookback=10, fwd=5)    # ~5-day hold (arc-2003 horizon; bigger move per spread-cost)
