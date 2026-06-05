"""arc 2009 — climax-sweep SHORT observation (cheap-kill at observation, step (b)/(d)).

The named 4th-PORTFOLIO-component target (arc 2008): a DOWN-TREND-CONTINUATION short, positive in
the binding 2018 fold where all three current fade/reversion longs lose. arc 2007 measured (as a
side-observation of its LONG arc) that the CLIMAX (big-range, violent) sweep below a swing low is a
FALLING KNIFE — forward drift -0.124 -> -0.280 -> -0.333 ATR, worse the bigger the climax = the SHORT
leg. arc 1014 (clean confirmed-breakdown short) found the CLEAN close-below reverts and the swing-low
is NOT load-bearing, but explicitly left the CLIMAX / fast-violent-drop variant engine-unvalidated:
"the -0.33 falling knife was inside a fast-3-bar-drop construction; the clean confirmed-breakdown
doesn't carry it."

This script tests whether the climax-sweep SHORT is a real, capturable, STRUCTURE-LOAD-BEARING short
edge, via the decisive cheap discriminators (mirroring arc 1014's structure control):

  Q1  Short capture + honest i+1 short drift of the climax-sweep cell vs the unconditional base.
  Q2  CLIMAX MONOTONICITY — does the short drift get MORE positive (price more negative) as the
      bar's range/violence grows? (reproduce arc 2007's -0.124 -> -0.333 in a clean SHORT construction)
  Q3  STRUCTURE CONTROL (the load-bearing test) — does a big-range fast-drop (climax) bar AT a swept
      swing-low continue down MORE than the SAME big-range fast-drop bar ELSEWHERE (no swept low)?
      If AT-swept ~= elsewhere, the structure is inert -> it's just "big red bar continues" = shallow
      momentum/breakout short = CLOSED GROUND (dead by symmetry, LESSONS).
  Q4  Per-pair robustness (3/7 positive = the arc-1010/1014 noise signature).

CHARACTERIZATION ONLY (gross, not cost-aware). Honest short capture/drift via the direction-aware
`observe_long_capture(direction="short")` (BUILT). Conditioning masks computed inline (one-off
scratch observer, arc-1014 convention). No engine compute unless this clears the cheap-kill.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.sim.panel import Panel
from discovery.tools.observe_long_capture import observe_long_capture

PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY"]
BACKUP = r"C:\Users\panap\histdata_backup"
K = 40          # swing-low lookback (match arc 1013/1014 deep construction)
ATR_P = 14


def atr_shift1(df: pd.DataFrame, p: int = ATR_P) -> np.ndarray:
    h = (df["high_bid"].to_numpy(float) + df["high_ask"].to_numpy(float)) / 2.0
    l = (df["low_bid"].to_numpy(float) + df["low_ask"].to_numpy(float)) / 2.0
    c = (df["close_bid"].to_numpy(float) + df["close_ask"].to_numpy(float)) / 2.0
    tr = np.full(len(df), np.nan)
    pc = np.concatenate([[np.nan], c[:-1]])
    tr[1:] = np.maximum.reduce([
        (h - l)[1:],
        np.abs(h[1:] - pc[1:]),
        np.abs(l[1:] - pc[1:]),
    ])
    # Wilder
    a = np.full(len(df), np.nan)
    if len(df) > p:
        a[p] = np.nanmean(tr[1:p + 1])
        for i in range(p + 1, len(df)):
            a[i] = (a[i - 1] * (p - 1) + tr[i]) / p
    return pd.Series(a, index=df.index).shift(1).to_numpy()


def build_cond(df: pd.DataFrame) -> pd.DataFrame:
    """Per-bar conditioning columns for the climax-sweep short, ex-ante (shift1)."""
    idx = df.index
    n = len(df)
    close_mid = (df["close_bid"].to_numpy(float) + df["close_ask"].to_numpy(float)) / 2.0
    high_mid = (df["high_bid"].to_numpy(float) + df["high_ask"].to_numpy(float)) / 2.0
    low_mid = (df["low_bid"].to_numpy(float) + df["low_ask"].to_numpy(float)) / 2.0
    atr = atr_shift1(df)

    lm = pd.Series(low_mid, index=idx)
    swing_low = lm.rolling(K).min().shift(1).to_numpy()  # prior K-bar swing low

    drop3 = np.full(n, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        drop3[3:] = (close_mid[3:] - close_mid[:-3]) / atr[3:]
        bar_range = (high_mid - low_mid) / atr           # climax size (bar i range in ATR)
        depth = (swing_low - low_mid) / atr              # how far below swing low it pierced
        close_below = close_mid < swing_low              # confirmed breakdown (no reclaim)

    swept = low_mid < swing_low                          # pierced the swing low (ran stops)
    return pd.DataFrame({
        "pair": None, "signal_time": idx,
        "swept": swept, "close_below": close_below,
        "drop3": drop3, "bar_range": bar_range, "depth": depth, "atr_c": atr,
    })


def main() -> None:
    panel = Panel.from_pairs(
        PAIRS, tf="H4", histdata_root=BACKUP, cache_root="data/cache",
        boundary_convention="5ers_eet",
    )
    # restrict to IS 2010-2020 by slicing each pair df is done via warmup + date filter below

    print("=== arc 2009 climax-sweep SHORT observation (H4 USD majors, IS 2010-2020) ===")
    obs = observe_long_capture(panel, sl_mult=2.0, hold=120, drift_bars=24, direction="short")
    obs = obs.set_index(["pair", "signal_time"])

    cond_frames = []
    for pair in PAIRS:
        df = panel.pair_dfs[pair]
        c = build_cond(df)
        c["pair"] = pair
        cond_frames.append(c)
    cond = pd.concat(cond_frames).set_index(["pair", "signal_time"])

    j = obs.join(cond, how="inner")
    # IS window only
    st = j.index.get_level_values("signal_time")
    lo_ts = pd.Timestamp("2010-01-01", tz="UTC")
    hi_ts = pd.Timestamp("2020-12-31", tz="UTC")
    j = j[(st >= lo_ts) & (st <= hi_ts)]
    j = j[np.isfinite(j["fwd_drift_atr"])]

    base_cap = j["capture"].mean()
    base_drift = j["fwd_drift_atr"].mean()
    print(f"\nBASE (all bars, short lens): n={len(j)} cap={base_cap:.4f} drift={base_drift:+.4f} ATR")
    print("  (short drift > 0 => price fell => good for short; cap base ~0.485 expected)")

    # ---- Q1: climax-sweep cell (swept & close_below & fast drop & big range) ----
    print("\n--- Q1: climax-sweep SHORT cell vs base ---")
    for rng in (1.0, 1.5, 2.0):
        for d3 in (1.0, 1.5):
            cell = j[j["swept"] & j["close_below"] & (j["drop3"] <= -d3) & (j["bar_range"] >= rng)]
            if len(cell) < 30:
                print(f"  range>={rng} drop3<=-{d3}: n={len(cell)} (THIN, skip)")
                continue
            print(f"  range>={rng} drop3<=-{d3}: n={len(cell):4d} "
                  f"cap={cell['capture'].mean():.4f} drift={cell['fwd_drift_atr'].mean():+.4f} "
                  f"(lift cap {cell['capture'].mean()-base_cap:+.4f}, drift {cell['fwd_drift_atr'].mean()-base_drift:+.4f})")

    # ---- Q2: climax monotonicity — drift by bar-range quantile within swept&close_below&fast ----
    print("\n--- Q2: CLIMAX MONOTONICITY (within swept & close_below & drop3<=-1.0) ---")
    pop = j[j["swept"] & j["close_below"] & (j["drop3"] <= -1.0)]
    print(f"  population n={len(pop)}")
    if len(pop) >= 50:
        qs = pop["bar_range"].quantile([0, .33, .66, 1.0]).to_numpy()
        for lo, hi, lab in [(qs[0], qs[1], "small"), (qs[1], qs[2], "mid"), (qs[2], qs[3] + 1e9, "CLIMAX")]:
            b = pop[(pop["bar_range"] >= lo) & (pop["bar_range"] < hi)]
            print(f"    {lab:7s} range[{lo:.2f},{hi if hi<1e8 else qs[3]:.2f}] n={len(b):4d} "
                  f"cap={b['capture'].mean():.4f} drift={b['fwd_drift_atr'].mean():+.4f}")
        # also by depth (how deep below swing low)
        print("  by PIERCE DEPTH (swing_low-low)/atr:")
        for lo, hi, lab in [(0, 0.5, "shallow<0.5"), (0.5, 1.0, "0.5-1.0"), (1.0, 1e9, "deep>1.0")]:
            b = pop[(pop["depth"] >= lo) & (pop["depth"] < hi)]
            if len(b) >= 20:
                print(f"    {lab:12s} n={len(b):4d} cap={b['capture'].mean():.4f} drift={b['fwd_drift_atr'].mean():+.4f}")

    # ---- Q3: STRUCTURE CONTROL — climax AT swept-low vs climax ELSEWHERE ----
    print("\n--- Q3: STRUCTURE CONTROL (the load-bearing test) ---")
    print("  Same big-range fast-drop CLIMAX bar, AT a swept swing-low vs NOT at one.")
    climax = j[(j["drop3"] <= -1.0) & (j["bar_range"] >= 1.5)]
    at_swept = climax[climax["swept"] & climax["close_below"]]
    elsewhere = climax[~climax["swept"]]
    print(f"  climax AT swept-low (close_below): n={len(at_swept):4d} "
          f"cap={at_swept['capture'].mean():.4f} drift={at_swept['fwd_drift_atr'].mean():+.4f}")
    print(f"  climax ELSEWHERE (not swept):      n={len(elsewhere):4d} "
          f"cap={elsewhere['capture'].mean():.4f} drift={elsewhere['fwd_drift_atr'].mean():+.4f}")
    print("  => if AT-swept ~= elsewhere, the swing-low structure is INERT (shallow momentum short).")

    # ---- Q4: per-pair robustness of the climax-sweep cell ----
    print("\n--- Q4: per-pair (climax-sweep short cell range>=1.5 drop3<=-1.0 swept close_below) ---")
    cell = j[j["swept"] & j["close_below"] & (j["drop3"] <= -1.0) & (j["bar_range"] >= 1.5)]
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
