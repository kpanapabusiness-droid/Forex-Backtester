"""arc 2070 OBSERVATION (cheap-kill) — E1: prop-firm SYNCHRONIZED-LIQUIDATION footprint at the EET
daily-DD boundary (the Run-2 strategist MENU item, DISCOVERY_DIRECTION.md §3-E1; never run — the
terminal-confirmation arcs 2066-2069 declared "MENU exhausted (M1/O1/L1/Q1/G1/S1)" = the RUN-1 items,
overlooking the run-2 additions E1/E2).

MECHANISM (because). Tens of thousands of funded-prop accounts share near-identical daily-DD rules
(~5%, EOD-EET) and concentrate in the same instruments (EURUSD/GBPUSD/USDJPY). On a large adverse
intraday move, broker risk engines auto-flatten en masse at the SAME EET boundary -> a synchronized
forced-liquidation cascade in the FINAL EET hour, then a snap-back when forced flow exhausts -> a
NEXT-SESSION reversion. Direction-agnostic: a big DOWN day flattens losing longs (overshoot down ->
revert UP / a LONG); a big UP day flattens losing shorts (overshoot up -> revert DOWN / a SHORT).

FALSIFIABLE PREDICTION. On big-|day-move| days the FINAL EET hour shows abnormal range (the cascade
footprint) and the next session reverts it (capture > base, drift > 0 in the reversion direction),
STRONGER than the same big-move reversion entered at a NON-final hour (the decisive discriminator:
does EET-boundary clustering add edge OVER the generic big-move/stop-sweep reversion the corpus already
killed?). Placebo: small-move final-hour days show nothing.

KILL falsifiers: final-hour reversion capture ~ base (~0.49) / ~ non-final-hour generic / drift <= 0;
no abnormal final-hour range on big days (premise false, cf. 2061/2062). §5d obs cheap-kill; §5f only
bites if a non-coin-flip above-null cell appears. H1, IS 2010-2020 only (OOS untouched).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.sim.panel import Panel
from core.features._helpers import mid_close, mid_high, mid_low, wilder_atr
from core.time_utils.session_boundary import utc_to_eet_trading_day
from discovery.tools.observe_long_capture import observe_long_capture

# prop-CONCENTRATED majors (most funded-prop volume), H1-cached
PAIRS = ["EURUSD", "GBPUSD", "USDJPY"]
BACKUP = r"C:\Users\panap\histdata_backup"
HOLD = 24            # next-session window (~1 EET day of H1 bars)
DRIFT = 24
SL_MULT = 2.0
STRONG_USD = (2014, 2015, 2018)

print(f"loading H1 panel: {PAIRS}", flush=True)
panel = Panel.from_pairs(PAIRS, tf="H1", histdata_root=BACKUP, cache_root="data/cache",
                         boundary_convention="5ers_eet")


def day_context(df):
    """Per-bar EET-day context: is-last-hour flag, day-move-so-far in H1-ATR, final-hour range/ATR."""
    idx = df.index
    n = len(df)
    atr = wilder_atr(mid_high(df), mid_low(df), mid_close(df), 14).shift(1).to_numpy(float)
    cm = mid_close(df).to_numpy(float)
    om = ((df["open_bid"] + df["open_ask"]) / 2.0).to_numpy(float)
    rng_atr = (mid_high(df).to_numpy(float) - mid_low(df).to_numpy(float)) / atr  # this-bar range in ATR
    eet = utc_to_eet_trading_day(idx)
    eet_key = pd.Series(eet, index=idx)
    # last hour of its EET day: next bar is a new EET day (timestamp knowledge; entry is at t+1 anyway)
    nxt = eet_key.shift(-1)
    is_last = (eet_key.to_numpy() != nxt.to_numpy())
    is_last[-1] = False
    # day open (first open_mid of each EET day), broadcast to every bar
    day_open = pd.Series(om, index=idx).groupby(eet_key).transform("first").to_numpy(float)
    day_move_atr = (cm - day_open) / atr   # move from day-open to THIS bar's close, in H1-ATR (causal)
    return {
        "atr": atr, "is_last": is_last, "day_move_atr": day_move_atr,
        "rng_atr": rng_atr, "eet_key": eet_key.to_numpy(),
    }


CTX = {p: day_context(panel.pair_dfs[p]) for p in PAIRS}

# ---- distribution of |day-move at final hour| to pick a "big day" threshold -------------------------
allmv = np.concatenate([
    np.abs(CTX[p]["day_move_atr"][CTX[p]["is_last"] & np.isfinite(CTX[p]["day_move_atr"])])
    for p in PAIRS
])
print(f"\n|day-move at final hour| (H1-ATR) quantiles over {len(allmv)} pair-days:", flush=True)
for q in (0.5, 0.7, 0.8, 0.9, 0.95):
    print(f"  q{int(q*100)}: {np.quantile(allmv, q):.2f}", flush=True)

THR = float(np.quantile(allmv, 0.8))   # "big day" = top ~20% of |day move|
print(f"\nbig-day threshold THR = {THR:.2f} H1-ATR (80th pct)", flush=True)


# ---- (1) CASCADE FOOTPRINT: is the final EET hour abnormally large on big days? --------------------
print("\n################ (1) CASCADE FOOTPRINT — final-hour range/ATR ################", flush=True)
for p in PAIRS:
    c = CTX[p]
    fin = c["is_last"] & np.isfinite(c["rng_atr"]) & np.isfinite(c["day_move_atr"])
    nonfin = (~c["is_last"]) & np.isfinite(c["rng_atr"])
    big = fin & (np.abs(c["day_move_atr"]) >= THR)
    small = fin & (np.abs(c["day_move_atr"]) <= 0.5)
    print(f"  {p}: final-hr range/ATR  all {c['rng_atr'][fin].mean():.3f} | "
          f"BIG-day {c['rng_atr'][big].mean():.3f} (n{big.sum()}) | "
          f"small-day {c['rng_atr'][small].mean():.3f} | non-final-hr {c['rng_atr'][nonfin].mean():.3f}",
          flush=True)


# ---- helpers for the reversion observation --------------------------------------------------------
def restrict_from(masks):
    return {p: masks[p] for p in PAIRS}


def run(masks, direction, label):
    obs = observe_long_capture(panel, sl_mult=SL_MULT, hold=HOLD, drift_bars=DRIFT,
                               restrict=restrict_from(masks), direction=direction)
    obs["year"] = obs["signal_time"].dt.year
    obs = obs[(obs["year"] >= 2010) & (obs["year"] <= 2020)].copy()
    print(f"  {label}: n={len(obs):5d}  capture {obs['capture'].mean():.4f}  "
          f"drift {obs['fwd_drift_atr'].mean():+.4f} ATR", flush=True)
    return obs


# baseline H1 capture per direction (reference)
print("\n################ baseline H1 capture (reference) ################", flush=True)
base_l = observe_long_capture(panel, sl_mult=SL_MULT, hold=HOLD, drift_bars=DRIFT, direction="long")
base_s = observe_long_capture(panel, sl_mult=SL_MULT, hold=HOLD, drift_bars=DRIFT, direction="short")
for nm, b in (("long", base_l), ("short", base_s)):
    b = b[(b["signal_time"].dt.year >= 2010) & (b["signal_time"].dt.year <= 2020)]
    print(f"  base {nm}: n={len(b)} capture {b['capture'].mean():.4f} drift {b['fwd_drift_atr'].mean():+.4f}",
          flush=True)


# ---- (2) FINAL-HOUR reversion vs (3) NON-FINAL-HOUR generic (the decisive discriminator) -----------
def big_down_final(c):   # long reversion at EOD cascade
    return c["is_last"] & (c["day_move_atr"] <= -THR) & np.isfinite(c["day_move_atr"])

def big_up_final(c):     # short reversion at EOD cascade
    return c["is_last"] & (c["day_move_atr"] >= THR) & np.isfinite(c["day_move_atr"])

def big_down_nonfinal(c):
    return (~c["is_last"]) & (c["day_move_atr"] <= -THR) & np.isfinite(c["day_move_atr"])

def big_up_nonfinal(c):
    return (~c["is_last"]) & (c["day_move_atr"] >= THR) & np.isfinite(c["day_move_atr"])

def small_final(c):
    return c["is_last"] & (np.abs(c["day_move_atr"]) <= 0.5) & np.isfinite(c["day_move_atr"])


print("\n################ (2)/(3) REVERSION — final-hour cascade vs non-final generic ################", flush=True)
print("[LONG reversion — big DOWN days]", flush=True)
ld_fin = run({p: big_down_final(CTX[p]) for p in PAIRS}, "long", "final-hour  (EOD cascade)")
ld_non = run({p: big_down_nonfinal(CTX[p]) for p in PAIRS}, "long", "non-final   (generic big-move)")
print("[SHORT reversion — big UP days]", flush=True)
su_fin = run({p: big_up_final(CTX[p]) for p in PAIRS}, "short", "final-hour  (EOD cascade)")
su_non = run({p: big_up_nonfinal(CTX[p]) for p in PAIRS}, "short", "non-final   (generic big-move)")

print("\n[PLACEBO — small-move final-hour days, expect ~base]", flush=True)
pl_l = run({p: small_final(CTX[p]) for p in PAIRS}, "long", "placebo long")
pl_s = run({p: small_final(CTX[p]) for p in PAIRS}, "short", "placebo short")


# ---- (4) per-year for the binding folds 2015/2018 (pooled long+short reversion) --------------------
print("\n################ (4) per-year reversion (final-hour, long+short pooled) ################", flush=True)
pooled = pd.concat([ld_fin.assign(side="L"), su_fin.assign(side="S")], ignore_index=True)
for y, g in pooled.groupby("year"):
    star = "  <-- strong-USD" if y in STRONG_USD else ""
    print(f"  {y}: n {len(g):4d}  cap {g['capture'].mean():.4f}  drift {g['fwd_drift_atr'].mean():+.4f}{star}",
          flush=True)

print("\n[VERDICT INPUTS] final vs non-final lift (the discriminator):", flush=True)
print(f"  LONG  final {ld_fin['capture'].mean():.4f} vs non-final {ld_non['capture'].mean():.4f}  "
      f"(drift {ld_fin['fwd_drift_atr'].mean():+.4f} vs {ld_non['fwd_drift_atr'].mean():+.4f})", flush=True)
print(f"  SHORT final {su_fin['capture'].mean():.4f} vs non-final {su_non['capture'].mean():.4f}  "
      f"(drift {su_fin['fwd_drift_atr'].mean():+.4f} vs {su_non['fwd_drift_atr'].mean():+.4f})", flush=True)
print("\nDONE.", flush=True)
