"""arc 2012 — DEEP multi-factor trend-CONTINUATION long (forward-confirmed resume after a shallow,
controlled pullback in a strong uptrend). Cheap observation (step b/d).

WHY (the *because*).
Every SHALLOW trend cut in the corpus is a coin-flip net of cost (Donchian/SMA single-condition:
arcs 0,1000-1004,2000,3000-3003), and 3003 sharpened it: strong-trend regimes REVERT (regime
detection is anti-predictive). BUT arc 1013 proved a DEEP conjunction (structure x sequence x
magnitude) extracts a clean edge where the shallow version is dead — and its load-bearing property
was being FORWARD-confirming: the reclaim enters AFTER the adverse move, so the i+1 entry isn't
buying into adverse continuation. arc 0's pullback-long FAILED partly because it bought INTO the dip
(not forward-confirmed). Nobody has applied 1013's forward-confirm property to trend CONTINUATION.

The portfolio route (2006/2008/3009) is BLOCKED on 2018 (strong-USD trend year): all 3 components are
fade/reversion -> tail-correlated, all bleed trend years. The missing 4th leg must be
trend/2018-POSITIVE. A continuation long is intrinsically trend-positive (the one flavor that profits
when fades bleed), is a LONG (no short-mirror death: 1014/2009/2011), and is in-scope FX, in my range.

THE DEEP CONJUNCTION (ex-ante, shift1; entry i+1):
  structure : close>SMA200 & SMA50>SMA200          (established uptrend)
  sequence  : a short pullback (P-bar local dip) that HOLDS above the prior swing low (higher-low intact)
  magnitude : the impulse leg before the pullback is STRONG (sma50 slope / impulse range >= thr ATR)
  resume    : close[i] > high[i-1]  (bullish reclaim of prior-bar high = resumption RESTARTED at i) -> FORWARD-confirming

DECISIVE cheap tests (1013 discipline):
  Q1  continuation-cell capture + honest i+1 LONG drift vs base, by impulse-strength gate.
  Q2  is the forward-confirm RESUME load-bearing? resume-confirmed vs into-the-dip (no resume).
  Q3  STRUCTURE CONTROL — the full deep conjunction vs the SAME resume bar ELSEWHERE (generic bullish
      bar in an uptrend, no pullback/higher-low/impulse). REAL 1013-mirror needs FULL >> generic.
  Q4  per-pair robustness (3/7=noise; watch USD quote-convention split).
  Q5  2018 / per-year drift — does the continuation cell survive the trend year where fades bleed?

CHARACTERIZATION ONLY (gross). Honest long capture/drift via direction-aware observe_long_capture
(BUILT). Conditioning inline (one-off scratch observer). Engine only if this clears the cheap-kill.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.sim.panel import Panel
from discovery.tools.observe_long_capture import observe_long_capture

PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY"]
BACKUP = r"C:\Users\panap\histdata_backup"
ATR_P = 14
SWING_K = 40     # higher-low reference (match 1013's swing lookback)
PULLBACK_P = 5   # local pullback window


def _mid(df, col):
    return (df[f"{col}_bid"].to_numpy(float) + df[f"{col}_ask"].to_numpy(float)) / 2.0


def atr_shift1(df: pd.DataFrame, p: int = ATR_P) -> np.ndarray:
    h, l, c = _mid(df, "high"), _mid(df, "low"), _mid(df, "close")
    tr = np.full(len(df), np.nan)
    pc = np.concatenate([[np.nan], c[:-1]])
    tr[1:] = np.maximum.reduce([(h - l)[1:], np.abs(h[1:] - pc[1:]), np.abs(l[1:] - pc[1:])])
    a = np.full(len(df), np.nan)
    if len(df) > p:
        a[p] = np.nanmean(tr[1:p + 1])
        for i in range(p + 1, len(df)):
            a[i] = (a[i - 1] * (p - 1) + tr[i]) / p
    return pd.Series(a, index=df.index).shift(1).to_numpy()


def sma_shift1(x: np.ndarray, idx, n: int) -> np.ndarray:
    return pd.Series(x, index=idx).rolling(n).mean().shift(1).to_numpy()


def build_cond(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    open_mid, close_mid = _mid(df, "open"), _mid(df, "close")
    high_mid, low_mid = _mid(df, "high"), _mid(df, "low")
    atr = atr_shift1(df)

    sma50 = sma_shift1(close_mid, idx, 50)
    sma200 = sma_shift1(close_mid, idx, 200)
    sma50_10ago = pd.Series(sma50, index=idx).shift(10).to_numpy()

    hm = pd.Series(high_mid, index=idx)
    lm = pd.Series(low_mid, index=idx)
    prior_high = hm.shift(1).to_numpy()                       # high[i-1]
    swing_low = lm.rolling(SWING_K).min().shift(1).to_numpy() # prior K-bar swing low (higher-low ref)
    pullback_low = lm.rolling(PULLBACK_P).min().shift(1).to_numpy()  # local dip low (prior P bars)

    with np.errstate(invalid="ignore", divide="ignore"):
        uptrend = (close_mid > sma200) & (sma50 > sma200)
        impulse = (sma50 - sma50_10ago) / atr                # impulse strength (10-bar sma50 slope in ATR)
        # a short pullback happened recently and held the higher-low structure:
        pulled_back = (pullback_low - swing_low) / atr        # how far the local dip sits ABOVE the swing low
        higher_low_hold = pullback_low > swing_low            # dip did NOT break the prior swing low
        # the local dip is meaningfully below the current bar (a real pullback, now resuming):
        dip_depth = (close_mid - pullback_low) / atr          # current close above the local dip (resumption extent)
        resume = close_mid > prior_high                       # bullish reclaim of prior-bar high (forward-confirm)

    return pd.DataFrame({
        "pair": None, "signal_time": idx,
        "uptrend": uptrend, "impulse": impulse, "higher_low_hold": higher_low_hold,
        "dip_depth": dip_depth, "resume": resume, "atr_c": atr,
    })


def _stat(cell, base_cap, base_drift, label):
    if len(cell) < 30:
        print(f"  {label}: n={len(cell)} (THIN)")
        return
    print(f"  {label}: n={len(cell):5d} cap={cell['capture'].mean():.4f} "
          f"drift={cell['fwd_drift_atr'].mean():+.4f} "
          f"(lift cap {cell['capture'].mean()-base_cap:+.4f}, drift {cell['fwd_drift_atr'].mean()-base_drift:+.4f})")


def main() -> None:
    panel = Panel.from_pairs(PAIRS, tf="H4", histdata_root=BACKUP, cache_root="data/cache",
                             boundary_convention="5ers_eet")
    print("=== arc 2012 DEEP trend-CONTINUATION long (H4 USD majors, IS 2010-2020) ===")
    obs = observe_long_capture(panel, sl_mult=2.0, hold=120, drift_bars=24, direction="long")
    obs = obs.set_index(["pair", "signal_time"])

    cond = pd.concat([build_cond(panel.pair_dfs[p]).assign(pair=p) for p in PAIRS]).set_index(["pair", "signal_time"])
    j = obs.join(cond, how="inner")
    st = j.index.get_level_values("signal_time")
    lo_ts, hi_ts = pd.Timestamp("2010-01-01", tz="UTC"), pd.Timestamp("2020-12-31", tz="UTC")
    j = j[(st >= lo_ts) & (st <= hi_ts)]
    j = j[np.isfinite(j["fwd_drift_atr"]) & np.isfinite(j["atr_c"]) & np.isfinite(j["impulse"])]

    base_cap, base_drift = j["capture"].mean(), j["fwd_drift_atr"].mean()
    print(f"\nBASE (all bars, long lens): n={len(j)} cap={base_cap:.4f} drift={base_drift:+.4f} ATR")

    # full deep conjunction: uptrend & higher-low-hold & a real prior dip now resuming
    core = j["uptrend"] & j["higher_low_hold"] & j["resume"] & (j["dip_depth"] >= 0.5)

    # ---- Q1: continuation cell vs base, by impulse-strength gate ----
    print("\n--- Q1: deep continuation cell (uptrend & HL-hold & resume & dip>=0.5ATR), by impulse gate ---")
    for imp in (0.0, 0.25, 0.5, 1.0):
        _stat(j[core & (j["impulse"] >= imp)], base_cap, base_drift, f"impulse>={imp}")

    # ---- Q2: is the forward-confirm RESUME load-bearing? ----
    print("\n--- Q2: forward-confirm RESUME load-bearing? (uptrend & HL-hold & dip>=0.5, impulse>=0.5) ---")
    g = j["uptrend"] & j["higher_low_hold"] & (j["dip_depth"] >= 0.5) & (j["impulse"] >= 0.5)
    _stat(j[g & j["resume"]], base_cap, base_drift, "RESUME-confirmed (forward)")
    _stat(j[g & ~j["resume"]], base_cap, base_drift, "into-the-dip (no resume) ")

    # ---- Q3: STRUCTURE CONTROL — full conjunction vs SAME resume bar ELSEWHERE ----
    print("\n--- Q3: STRUCTURE CONTROL (1013-mirror) ---")
    full = j[core & (j["impulse"] >= 0.5)]
    # generic: a resume bar in an uptrend but WITHOUT the deep pullback/HL/impulse structure
    generic = j[j["uptrend"] & j["resume"] & ~(j["higher_low_hold"] & (j["dip_depth"] >= 0.5) & (j["impulse"] >= 0.5))]
    _stat(full, base_cap, base_drift, "FULL deep conjunction        ")
    _stat(generic, base_cap, base_drift, "generic resume-in-uptrend    ")
    print("  => REAL 1013-mirror needs FULL >> generic (cap & drift); ~= means the conjunction is INERT.")

    # ---- Q4: per-pair robustness ----
    print("\n--- Q4: per-pair (full deep conjunction, impulse>=0.5) ---")
    npos = 0
    for pair in PAIRS:
        sub = full[full.index.get_level_values("pair") == pair]
        if len(sub) >= 10:
            dr = sub["fwd_drift_atr"].mean()
            npos += int(dr > 0)
            print(f"    {pair}: n={len(sub):4d} cap={sub['capture'].mean():.4f} drift={dr:+.4f}")
        else:
            print(f"    {pair}: n={len(sub):4d} (thin)")
    print(f"  pairs with positive long drift: {npos}/7  (3-4/7 = noise signature)")

    # ---- Q5: per-year drift (the 2018 question) ----
    print("\n--- Q5: per-year drift of the full deep conjunction (the 2018/trend-year leg) ---")
    yrs = full.index.get_level_values("signal_time").year
    for y in range(2010, 2021):
        sub = full[yrs == y]
        if len(sub) >= 10:
            print(f"    {y}: n={len(sub):4d} cap={sub['capture'].mean():.4f} drift={sub['fwd_drift_atr'].mean():+.4f}")
        else:
            print(f"    {y}: n={len(sub):4d} (thin)")


if __name__ == "__main__":
    main()
