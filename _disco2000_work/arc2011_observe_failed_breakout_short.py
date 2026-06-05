"""arc 2011 — failed-BREAKOUT reject SHORT at a swing HIGH (the TRUE forward-confirming mirror of
arc 1013's reclaim-long). Cheap observation (step b/d).

arc 1013's failed-breakdown RECLAIM long is the strongest edge in the corpus *because* the reclaim is
FORWARD-confirming: a wick pierces a swing LOW (sweeps sell-stops) and CLOSES BACK ABOVE it, so the
honest i+1 long enters AFTER the adverse low — the up-move it bets on has not started. arc 1014 tested
"the short mirror" but used the WRONG mirror (swing-LOW pierce + close BELOW = continuation,
BACKWARD-confirming → enters the local low → reverts → KILL). arc 2009 tested the climax continuation
short (also backward-confirming → KILL). NOBODY has tested the TRUE forward-confirming short mirror:

  a failed-BREAKOUT REJECT at a swing HIGH — price spikes ABOVE a prior K-bar swing high (sweeps resting
  BUY-stops / triggers breakout-longs), then REJECTS back BELOW it within the bar with a large UPPER
  rejection wick (>= min_shadow ATR) = a bull-trap / failed breakout → reversal DOWN. The reject is
  FORWARD-confirming (the up-move is over AT the signal bar; the i+1 short enters AFTER the adverse
  high, not into it) — the exact property that made arc 1013 work, mirrored to the high.

WHY it could be the 2018-positive 4th PORTFOLIO leg: on a USD major (e.g. EURUSD), a failed rally that
rejects at a swing high → short → CONTINUES DOWN in a strong-USD trend year (2018) = positive exactly
where the three fade longs bleed. A forward-confirming structural reversal short, not a coin-flip
continuation (1014/2009) nor a fade.

DECISIVE cheap tests (the arc-1013/1014/2009 discipline, mirrored to the high):
  Q1  Short capture + honest i+1 short drift of the reject cell vs base. (short drift>0 ⇔ price fell)
  Q2  Is the REJECT/rejection-wick load-bearing? deepen the upper shadow gate; does the edge GROW?
  Q3  STRUCTURE CONTROL — big upper-wick AT a swept swing-HIGH vs the SAME wick ELSEWHERE. If
      AT-swept ~= elsewhere, the swing-high is inert (just "big upper wick fades" = shallow). For a
      REAL mirror of 1013 we need AT-swept >> elsewhere (1013: AT-low 0.55-0.61 vs elsewhere ~0.49).
  Q4  Per-pair robustness (3/7 = noise; watch the USD quote-convention split, arc 2009).

CHARACTERIZATION ONLY (gross). Honest short capture/drift via direction-aware observe_long_capture
(BUILT). Conditioning inline (one-off scratch observer). Engine only if this clears the cheap-kill.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.sim.panel import Panel
from discovery.tools.observe_long_capture import observe_long_capture

PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY"]
BACKUP = r"C:\Users\panap\histdata_backup"
K = 40          # swing-high lookback (match arc 1013's deep construction)
ATR_P = 14


def atr_shift1(df: pd.DataFrame, p: int = ATR_P) -> np.ndarray:
    h = (df["high_bid"].to_numpy(float) + df["high_ask"].to_numpy(float)) / 2.0
    l = (df["low_bid"].to_numpy(float) + df["low_ask"].to_numpy(float)) / 2.0
    c = (df["close_bid"].to_numpy(float) + df["close_ask"].to_numpy(float)) / 2.0
    tr = np.full(len(df), np.nan)
    pc = np.concatenate([[np.nan], c[:-1]])
    tr[1:] = np.maximum.reduce([(h - l)[1:], np.abs(h[1:] - pc[1:]), np.abs(l[1:] - pc[1:])])
    a = np.full(len(df), np.nan)
    if len(df) > p:
        a[p] = np.nanmean(tr[1:p + 1])
        for i in range(p + 1, len(df)):
            a[i] = (a[i - 1] * (p - 1) + tr[i]) / p
    return pd.Series(a, index=df.index).shift(1).to_numpy()


def build_cond(df: pd.DataFrame) -> pd.DataFrame:
    """Per-bar conditioning for the failed-breakout reject short, ex-ante (shift1). Mirror of
    FailedBreakdownReclaimLongSignal to the swing HIGH."""
    idx = df.index
    n = len(df)
    open_mid = (df["open_bid"].to_numpy(float) + df["open_ask"].to_numpy(float)) / 2.0
    close_mid = (df["close_bid"].to_numpy(float) + df["close_ask"].to_numpy(float)) / 2.0
    high_mid = (df["high_bid"].to_numpy(float) + df["high_ask"].to_numpy(float)) / 2.0
    atr = atr_shift1(df)

    hm = pd.Series(high_mid, index=idx)
    swing_high = hm.rolling(K).max().shift(1).to_numpy()  # prior K-bar swing high

    with np.errstate(invalid="ignore", divide="ignore"):
        # upper rejection shadow above body top (mirror of 1013's lower shadow below body bottom)
        upper_shadow = (high_mid - np.maximum(open_mid, close_mid)) / atr
        overshoot = (high_mid - swing_high) / atr  # how far above the swing high it pierced

    swept_high = high_mid > swing_high                    # pierced the swing high (ran buy-stops)
    reject = close_mid < swing_high                       # closed back below = failed breakout (reject)
    return pd.DataFrame({
        "pair": None, "signal_time": idx,
        "swept_high": swept_high, "reject": reject,
        "upper_shadow": upper_shadow, "overshoot": overshoot, "atr_c": atr,
    })


def main() -> None:
    panel = Panel.from_pairs(PAIRS, tf="H4", histdata_root=BACKUP, cache_root="data/cache",
                             boundary_convention="5ers_eet")
    print("=== arc 2011 failed-breakout REJECT SHORT @ swing HIGH (H4 USD majors, IS 2010-2020) ===")
    obs = observe_long_capture(panel, sl_mult=2.0, hold=120, drift_bars=24, direction="short")
    obs = obs.set_index(["pair", "signal_time"])

    cond = pd.concat([build_cond(panel.pair_dfs[p]).assign(pair=p) for p in PAIRS]).set_index(["pair", "signal_time"])
    j = obs.join(cond, how="inner")
    st = j.index.get_level_values("signal_time")
    lo_ts, hi_ts = pd.Timestamp("2010-01-01", tz="UTC"), pd.Timestamp("2020-12-31", tz="UTC")
    j = j[(st >= lo_ts) & (st <= hi_ts)]
    j = j[np.isfinite(j["fwd_drift_atr"]) & np.isfinite(j["atr_c"])]

    base_cap, base_drift = j["capture"].mean(), j["fwd_drift_atr"].mean()
    print(f"\nBASE (all bars, short lens): n={len(j)} cap={base_cap:.4f} drift={base_drift:+.4f} ATR")
    print("  (short drift > 0 => price fell => good for short)")

    # ---- Q1: the reject cell vs base, by upper-shadow gate (mirror of 1013's shadow>=1.25) ----
    print("\n--- Q1: failed-breakout REJECT SHORT cell (swept_high & reject & upper_shadow>=s) ---")
    for s in (0.5, 1.0, 1.25, 1.5):
        cell = j[j["swept_high"] & j["reject"] & (j["upper_shadow"] >= s)]
        if len(cell) < 30:
            print(f"  shadow>={s}: n={len(cell)} (THIN)")
            continue
        print(f"  shadow>={s}: n={len(cell):4d} cap={cell['capture'].mean():.4f} "
              f"drift={cell['fwd_drift_atr'].mean():+.4f} "
              f"(lift cap {cell['capture'].mean()-base_cap:+.4f}, drift {cell['fwd_drift_atr'].mean()-base_drift:+.4f})")

    # ---- Q2: is the reject/shadow load-bearing? does deeper shadow GROW the edge? (1013 signature) ----
    print("\n--- Q2: shadow monotonicity (within swept_high & reject) ---")
    pop = j[j["swept_high"] & j["reject"]]
    if len(pop) >= 50:
        for lo, hi, lab in [(0, 0.5, "small<0.5"), (0.5, 1.0, "0.5-1.0"), (1.0, 1.5, "1.0-1.5"), (1.5, 1e9, "deep>1.5")]:
            b = pop[(pop["upper_shadow"] >= lo) & (pop["upper_shadow"] < hi)]
            if len(b) >= 20:
                print(f"    shadow {lab:9s} n={len(b):4d} cap={b['capture'].mean():.4f} drift={b['fwd_drift_atr'].mean():+.4f}")

    # ---- Q3: STRUCTURE CONTROL — big upper-wick AT swept swing-high vs ELSEWHERE ----
    print("\n--- Q3: STRUCTURE CONTROL (the load-bearing test, mirror of 1013) ---")
    big_wick = j[j["upper_shadow"] >= 1.25]
    at_swept = big_wick[big_wick["swept_high"] & big_wick["reject"]]
    elsewhere = big_wick[~big_wick["swept_high"]]
    print(f"  big upper-wick AT swept-high (reject): n={len(at_swept):4d} "
          f"cap={at_swept['capture'].mean():.4f} drift={at_swept['fwd_drift_atr'].mean():+.4f}")
    print(f"  big upper-wick ELSEWHERE (not swept):  n={len(elsewhere):4d} "
          f"cap={elsewhere['capture'].mean():.4f} drift={elsewhere['fwd_drift_atr'].mean():+.4f}")
    print("  => REAL 1013-mirror needs AT-swept >> elsewhere (cap & drift); ~= means swing-high INERT.")

    # ---- Q4: per-pair robustness of the reject cell ----
    print("\n--- Q4: per-pair (swept_high & reject & upper_shadow>=1.25) ---")
    cell = j[j["swept_high"] & j["reject"] & (j["upper_shadow"] >= 1.25)]
    npos = 0
    for pair in PAIRS:
        sub = cell[cell.index.get_level_values("pair") == pair]
        if len(sub) >= 10:
            dr = sub["fwd_drift_atr"].mean()
            npos += int(dr > 0)
            print(f"    {pair}: n={len(sub):3d} cap={sub['capture'].mean():.4f} drift={dr:+.4f}")
        else:
            print(f"    {pair}: n={len(sub):3d} (thin)")
    print(f"  pairs with positive short drift: {npos}/7  (3/7 = noise signature)")


if __name__ == "__main__":
    main()
