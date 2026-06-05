"""Arc 1025 observation — does the corpus's ONLY fold-resolving edge (fbr) THICKEN?

Idea + because (arc-2017 option B + the book's one real obstacle).
The 4 portfolio components are all THIN forced-flow reversions (gap ~28/yr, me ~11/yr,
fbr ~17/yr); arc 2016/2017/1023 proved the per-year all-folds gate sits BELOW the noise
floor for thin components -> the leg-hunt route is noise-blocked. Arc 2017 named the ONLY
surviving productive spec: option (B) a component THICK enough that its folds RESOLVE.
Arc 2018 attacked (B) via cross-sectional `me` (killed by multi-leg cost). The obvious
un-tried (B) route is to THICKEN the corpus's single fold-RESOLVING edge itself: fbr
(arc 1013), the strongest + cleanest edge.

fbr fires only ~17/yr BECAUSE it needs a DEEP (shadow>=1.25 ATR) reclaim of a LONG (K=40)
swing low -- rare, deep forced-flow stop-runs. The SAME stop-run-reversal mechanism fires
far more often at SHORTER lookbacks (K=10/20, swept more often) and SHALLOWER reclaims
(shadow>=0.5/0.75). Two genuinely-untested questions, ONE cheap obs sweep:

  HYP-A (thicken, option B): does a denser (K,shadow) cell keep enough edge -- capture>0.50
  + load-bearing structure control + null-beatable drift -- to be a fold-RESOLVING thicker
  component? (arc 1013 found "deeper grab = stronger", which PREDICTS shallow dilutes to
  coin-flip -- but the full count/edge tradeoff surface was never published.)

  HYP-B (the book's one real obstacle): arc 2014 found fbr-2018 is a near-total wipeout
  (18/19 -1R) at the DEEP cell. Is that wipeout (K,shadow)-INVARIANT (every fbr-class trade
  loses in strong-USD 2018 -> mechanism-intrinsic, closes the obstacle harder) or do
  shallower/shorter triggers catch tradeable 2018 trades (-> a real lead on the one obstacle
  that blocks the book)?

OBSERVATION ONLY (gross, take-the-loss capture + fwd drift; NOT a gate). All on IS
(2010-2020). USD majors only (arc 1022: fbr is USD-major-specific, stop-liquidity follows
the most-participated instrument; crosses carry no fbr edge). 2018 is an IS year -> OOS
never touched. Escalate to the honest engine §5f ONLY if a denser cell is non-coin-flip.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.sim.panel import Panel
from discovery.tools.trend_entry_signals import _atr_shift1_mid
from discovery.tools.observe_long_capture import observe_long_capture

IS_START, IS_END = pd.Timestamp("2010-01-01", tz="UTC"), pd.Timestamp("2020-12-31", tz="UTC")
MAJORS = ["AUDUSD", "EURUSD", "GBPUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY"]

K_SET = [10, 20, 40, 60]          # swing-low lookback: shorter = swept more often = thicker
SHADOW_SET = [0.5, 0.75, 1.0, 1.25]  # reclaim depth: shallower = fires more = thicker
# arc-1013 reference (load-bearing) cell = K=40, shadow=1.25


def build_masks(df, K, SHADOW):
    """(fbr_fire, wick_elsewhere) ex-ante masks, exactly per the BUILT fbr signal logic."""
    idx = df.index
    low_bid = df["low_bid"].to_numpy(float)
    open_mid = (df["open_bid"].to_numpy(float) + df["open_ask"].to_numpy(float)) / 2.0
    close_mid = (df["close_bid"].to_numpy(float) + df["close_ask"].to_numpy(float)) / 2.0
    atr = _atr_shift1_mid(df, 14)
    prior_low = pd.Series(low_bid, index=idx).shift(1).rolling(K).min().to_numpy(float)
    with np.errstate(invalid="ignore"):
        shadow = (np.minimum(open_mid, close_mid) - low_bid) / atr
    finite = np.isfinite(atr) & (atr > 0) & np.isfinite(prior_low)
    swept = low_bid < prior_low
    fbr = swept & (close_mid > prior_low) & (shadow >= SHADOW) & finite
    elsewhere = (shadow >= SHADOW) & (~swept) & finite      # deep wick, NOT a swept-low reclaim
    return fbr, elsewhere


def restrict_in_is(df, mask):
    in_is = np.asarray(df.index >= IS_START) & np.asarray(df.index <= IS_END)
    return mask & in_is


print("Loading H4 USD majors panel ...")
panel = Panel.from_pairs(
    MAJORS, tf="H4", histdata_root=r"C:\Users\panap\histdata_backup",
    cache_root="data/cache", boundary_convention="5ers_eet",
)

rows = []
year2018 = {}
for K in K_SET:
    for SHADOW in SHADOW_SET:
        fbr_restrict, elsewhere_restrict = {}, {}
        for pair in MAJORS:
            df = panel.pair_dfs[pair]
            fbr, elsewhere = build_masks(df, K, SHADOW)
            fbr_restrict[pair] = restrict_in_is(df, fbr)
            elsewhere_restrict[pair] = restrict_in_is(df, elsewhere)

        sig = observe_long_capture(panel, sl_mult=2.0, hold=120, drift_bars=12,
                                   restrict=fbr_restrict, direction="long")
        ctrl = observe_long_capture(panel, sl_mult=2.0, hold=120, drift_bars=12,
                                    restrict=elsewhere_restrict, direction="long")
        sig = sig[np.isfinite(sig["fwd_drift_atr"])].copy()
        ctrl = ctrl[np.isfinite(ctrl["fwd_drift_atr"])].copy()
        n = len(sig)
        per_yr = n / 11.0
        cap = sig["capture"].mean() if n else float("nan")
        drift = sig["fwd_drift_atr"].mean() if n else float("nan")
        dmed = sig["fwd_drift_atr"].median() if n else float("nan")
        struct_cap = (sig["capture"].mean() - ctrl["capture"].mean()) if len(ctrl) else float("nan")
        struct_drift = (sig["fwd_drift_atr"].mean() - ctrl["fwd_drift_atr"].mean()) if len(ctrl) else float("nan")
        # 2018 acceptance
        sig["year"] = sig["signal_time"].dt.year
        y18 = sig[sig["year"] == 2018]
        n18 = len(y18)
        cap18 = y18["capture"].mean() if n18 else float("nan")
        drift18 = y18["fwd_drift_atr"].mean() if n18 else float("nan")
        # per-pair cap>0.50 consistency
        sig["pair"] = sig["pair"].astype(str)
        bp = sig.groupby("pair")["capture"].mean()
        npos = int((bp > 0.50).sum())
        rows.append(dict(K=K, shadow=SHADOW, n=n, per_yr=round(per_yr, 1), cap=cap,
                         drift=drift, med=dmed, struct_cap=struct_cap, struct_drift=struct_drift,
                         pp=f"{npos}/{len(bp)}", n18=n18, cap18=cap18, drift18=drift18))

res = pd.DataFrame(rows)
pd.set_option("display.width", 200, "display.max_columns", 30)
print("\n" + "=" * 110)
print("fbr DENSITY SWEEP  (K x shadow), H4 USD majors, IS 2010-2020")
print("  cap>0.50 + struct_cap>0 + struct_drift>0 = load-bearing edge survives at this density")
print("  ref cell K=40 shadow=1.25 = the committed arc-1013 fbr")
print("=" * 110)
fmt = {c: (lambda x: f"{x:+.4f}") for c in ["cap", "drift", "med", "struct_cap", "struct_drift", "cap18", "drift18"]}
print(res.to_string(index=False, formatters=fmt))

print("\n--- READ ---")
ref = res[(res.K == 40) & (res.shadow == 1.25)].iloc[0]
print(f"ref (K40,s1.25): n={ref.n} ({ref.per_yr}/yr) cap={ref.cap:+.4f} struct_cap={ref.struct_cap:+.4f} "
      f"2018 cap={ref.cap18:+.4f} drift={ref.drift18:+.4f} (n18={ref.n18})")
# best DENSER cell: more trades than ref AND still load-bearing (cap>0.50 & struct excess>0)
denser = res[(res.n > ref.n) & (res.cap > 0.50) & (res.struct_cap > 0) & (res.struct_drift > 0)]
if len(denser):
    denser = denser.sort_values("n", ascending=False)
    print(f"\nDENSER cells that KEEP a load-bearing edge (n>{ref.n}, cap>0.50, struct excess>0):")
    print(denser[["K", "shadow", "n", "per_yr", "cap", "struct_cap", "struct_drift", "cap18", "drift18"]]
          .to_string(index=False, formatters=fmt))
    print(">>> HYP-A candidate exists -> escalate the densest such cell to honest engine §5f.")
else:
    print("\nNO denser cell keeps a load-bearing edge -> HYP-A FALSE: fbr does NOT thicken "
          "(shallower/shorter triggers dilute to coin-flip / lose structure control).")
# 2018 invariance
print(f"\n2018 capture across all cells: min={res.cap18.min():+.4f} max={res.cap18.max():+.4f} "
      f"(0.50 = coin-flip; all <0.50 -> HYP-B: 2018 wipeout is (K,shadow)-INVARIANT)")
