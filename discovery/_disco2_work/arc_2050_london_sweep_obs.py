"""arc 2050 OBSERVATION (cheap-kill) — London-open Asian-range liquidity SWEEP-AND-REVERSAL (long).

because: fbr (1013) is the corpus's ONE robust structural edge — a stop-run sweep of a VISIBLE swing
low that fails and reclaims REVERTS, because the grab is an information-free forced flow (stop cascade).
The most DOCUMENTED, densest, universally-watched intraday stop pool in FX is the ASIAN-SESSION RANGE,
raided at the LONDON OPEN ("judas swing" / opening liquidity grab): algos sweep the Asian low to fill
resting sell-stops, then reverse into the real session. This is fbr's EXACT mechanism (pierce a
structural level -> fail -> reclaim -> revert), anchored to a documented level (Asian range) + a
documented forced-flow time (London open), instead of a rolling algorithmic swing. It is
decorrelated-by-timing from the weekend (1006) / month-end (1011/1019) flows -> candidate NEW component;
and it fires ~daily (THICK -> arc-2017 option-B). Genuinely UNTESTED: 1047 = session DRIFT (continuous,
sub-cost); 1050 = vol-vacuum spike with NO structural level (became momentum); 2029 = DAILY
prior-day/week lows. None is the intraday session-open Asian-range sweep.

EXPLICIT cost-skeptic prior: the intraday lane is repeatedly sub-cost (1047/1008/1010/3008). So the
screen must show the post-sweep reversal (a) CLEARS the FundedNext cost line AND (b) BEATS a matched
failed-breakdown-reclaim STRUCTURE control (same shape, NOT at the session/Asian level) -- the fbr
control discipline. Falsifiers (-> KILL): coin-flip capture / sub-cost reversal / session ~ control /
not 2014-18-relevant.

Asian range = bars hour in [0..6] UTC (Tokyo). London sweep window = hour in [7..10] UTC.
LONG fire = first sweep-window bar whose low_bid pierces the day's Asian low AND close_mid reclaims
above it (failed breakdown at the session level). Entry next H1 bar (canonical observe_long_capture).
UTC fixed windows (not Europe/London DST) -- acceptable for a cheap obs-screen; noted as a caveat.
IS 2010-2020 only (OOS preserved).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.sim.panel import Panel
from core.features._helpers import mid_close
from discovery.tools.observe_long_capture import observe_long_capture

PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "USDJPY", "EURJPY", "GBPJPY", "AUDJPY"]
BACKUP = r"C:\Users\panap\histdata_backup"
ASIAN_HRS = set(range(0, 7))      # 00:00..06:00 UTC (Tokyo range)
SWEEP_HRS = set(range(7, 11))     # 07:00..10:00 UTC (London-open raid)
HOLD = 24                         # ~1 trading day of H1 bars
DRIFT = 6                         # ~6h reversal horizon
ROLL = 7                          # rolling-low length for the matched control (~Asian window len)

print(f"loading H1 panel: {PAIRS}", flush=True)
panel = Panel.from_pairs(PAIRS, tf="H1", histdata_root=BACKUP, cache_root="data/cache",
                         boundary_convention="5ers_eet")


def build_masks(df):
    """Return (session_fire, ctrl_fire, sweep_depth_atr) aligned to df bars."""
    from core.features._helpers import mid_high, mid_low, wilder_atr
    idx = df.index
    n = len(df)
    hour = idx.hour.values
    date = idx.normalize()  # UTC calendar day key
    low_bid = df["low_bid"].values
    mc = mid_close(df).values
    atr = wilder_atr(mid_high(df), mid_low(df), mid_close(df), 14).shift(1).values

    # per-day Asian-range low (min low_bid over Asian hours), broadcast to all bars of that day
    asian_mask = np.isin(hour, list(ASIAN_HRS))
    tmp = pd.DataFrame({"date": date, "lb": np.where(asian_mask, low_bid, np.nan)})
    asian_low_by_day = tmp.groupby("date")["lb"].min()
    asian_low = date.map(asian_low_by_day).to_numpy(dtype=float)

    in_sweep = np.isin(hour, list(SWEEP_HRS))
    pierced = low_bid < asian_low                    # swept sell-stops below the Asian low
    reclaim = mc > asian_low                          # closed back above (failed breakdown)
    raw_fire = in_sweep & pierced & reclaim & np.isfinite(atr) & (atr > 0)

    # keep only the FIRST sweep fire per day (the raid), else multi-count noise
    session_fire = np.zeros(n, dtype=bool)
    seen = set()
    for t in np.flatnonzero(raw_fire):
        d = date[t]
        if d in seen:
            continue
        seen.add(d)
        session_fire[t] = True

    sweep_depth = np.where(session_fire, (asian_low - low_bid) / atr, np.nan)

    # MATCHED structure control: same failed-breakdown-reclaim shape on a ROLL-bar rolling low,
    # OUTSIDE the sweep window (NOT the session/Asian level). Isolates the session+level specificity.
    roll_low = pd.Series(low_bid).shift(1).rolling(ROLL).min().to_numpy(dtype=float)
    ctrl = (~in_sweep) & (low_bid < roll_low) & (mc > roll_low) & np.isfinite(atr) & (atr > 0)
    return session_fire, ctrl, sweep_depth


sess_fire, ctrl_fire, depths = {}, {}, {}
for p in PAIRS:
    sf, cf, dep = build_masks(panel.pair_dfs[p])
    sess_fire[p], ctrl_fire[p], depths[p] = sf, cf, dep


def summarize(restrict, label, is_only=True):
    obs = observe_long_capture(panel, sl_mult=2.0, hold=HOLD, drift_bars=DRIFT,
                               restrict=restrict, direction="long")
    obs["year"] = obs["signal_time"].dt.year
    if is_only:
        obs = obs[(obs["year"] >= 2010) & (obs["year"] <= 2020)].copy()
    print(f"\n=== {label}: n={len(obs)}  capture {obs['capture'].mean():.4f}  "
          f"drift {obs['fwd_drift_atr'].mean():+.4f} ATR ===", flush=True)
    return obs


# unconditional H1 baseline (every bar) for context
base = observe_long_capture(panel, sl_mult=2.0, hold=HOLD, drift_bars=DRIFT, direction="long")
base = base[(base["signal_time"].dt.year >= 2010) & (base["signal_time"].dt.year <= 2020)]
print(f"\nUNCONDITIONAL H1 baseline: n={len(base)} capture {base['capture'].mean():.4f} "
      f"drift {base['fwd_drift_atr'].mean():+.4f}", flush=True)

sess = summarize(sess_fire, "LONDON SWEEP long (Asian-low sweep+reclaim @ 07-10 UTC)")
ctrl = summarize(ctrl_fire, "STRUCTURE control (rolling-low reclaim, OUTSIDE sweep window)")

print(f"\n[FALSIFIER 1 — session structure load-bearing?] "
      f"session cap {sess['capture'].mean():.4f} vs control {ctrl['capture'].mean():.4f}  | "
      f"session drift {sess['fwd_drift_atr'].mean():+.4f} vs control {ctrl['fwd_drift_atr'].mean():+.4f}",
      flush=True)

# sweep-depth & reversal magnitude (cost line ~0.05-0.10 R; +1R label target = 2*ATR)
dep_all = np.concatenate([depths[p][np.isfinite(depths[p])] for p in PAIRS])
print(f"\n[FALSIFIER 2 — clears cost?] sweep depth median {np.median(dep_all):.3f} ATR "
      f"mean {np.mean(dep_all):.3f}; session fwd drift mean {sess['fwd_drift_atr'].mean():+.4f} ATR "
      f"median {sess['fwd_drift_atr'].median():+.4f}. (FundedNext H1 cost ~0.05-0.10 R hurdle.)", flush=True)

print("\nper-pair session (cap / drift / n):", flush=True)
for p, sub in sess.groupby("pair"):
    print(f"  {p}: n {len(sub):4d}  cap {sub['capture'].mean():.4f}  drift {sub['fwd_drift_atr'].mean():+.4f}",
          flush=True)

print("\n[FALSIFIER 3 — 2014/15/18 relevance] per-year session (cap / drift):", flush=True)
for y, g in sess.groupby("year"):
    star = "  <-- strong-USD" if y in (2014, 2015, 2018) else ""
    print(f"  {y}: n {len(g):4d}  cap {g['capture'].mean():.4f}  drift {g['fwd_drift_atr'].mean():+.4f}{star}",
          flush=True)
print("\nDONE.", flush=True)
