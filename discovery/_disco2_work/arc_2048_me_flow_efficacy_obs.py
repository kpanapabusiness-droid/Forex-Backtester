"""arc 2048 OBSERVATION (cheap-kill, no engine yet) — Month-end flow-EFFICACY regime persistence.

HYPOTHESIS (because): me_long (1011) is the corpus's lone exit-robust survivor (1042/2042/1046).
Its binding negative folds are the contiguous 2014/2015/2016 strong-USD block (1012) — where a big
DOWN move into month-end in a USD major IS the USD trend, so the WMR reversion gets overrun. EVERY
PRICE-trend regime gate to separate that block failed (2014 SMA-slope, 1012 trend-filter, 1029 cal-
density, 2024 full-month window). The ONE regime detector never tried: the FLOW'S OWN RECENT EFFICACY.
The WMR month-end rebalancing flow's tradeable strength tracks the standing stock of cross-border
hedges (slow-moving) and whether the prevailing trend is currently overrunning it (slow-moving) ->
the month-end reversion's realized strength should be POSITIVELY AUTOCORRELATED month-to-month, and a
month where last month's month-end reversion FAILED signals the flow is being overrun (strong-trend
regime) -> SKIP. A STATE conditioner (the edge's own recent outcome as the regime tell), distinct from
price-trend (2014), aimed at turning the sole survivor into a solo all-folds-positive PASS.

FALSIFIERS (cheap-kill, §5d):
  1. month-to-month autocorr of the cross-sectional reversion coefficient ~ 0 (within ~2 SE) ->
     flow strength NOT persistent -> no regime to detect -> KILL.
  2. prior-month efficacy does NOT separate the 2014/15/16 dead block from the good years -> KILL.
Only if BOTH pass -> §5f engine run (non-coin-flip entry conditioned on a real regime tell).

CHARACTERIZATION ONLY: observe_long_capture (gross, take-the-loss) + raw D1 month-end geometry.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from core.sim.panel import Panel
from discovery.tools.month_end_signals import _month_end_into_move
from discovery.tools.observe_long_capture import observe_long_capture

PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCHF", "USDCAD"]
BACKUP = r"C:\Users\panap\histdata_backup"
INTO_BARS = 2
AFTER_BARS = 2  # reversion horizon (matches me_long into_bars / 2-bar hold)

print("loading D1 panel (USD majors)...", flush=True)
panel = Panel.from_pairs(
    PAIRS, tf="D1", histdata_root=BACKUP, cache_root="data/cache",
    boundary_convention="5ers_eet",
)

# -------- 1) dense per-(pair,month-end) into/after moves (ALL month-ends, unconditional) --------
recs = []
for pair in PAIRS:
    df = panel.pair_dfs[pair]
    idx, is_last, into, atr = _month_end_into_move(df, INTO_BARS, 14)
    close_mid = (df["close_bid"].to_numpy(float) + df["close_ask"].to_numpy(float)) / 2.0
    n = len(df)
    for t in np.where(is_last)[0]:
        if not (np.isfinite(into[t]) and np.isfinite(atr[t]) and atr[t] > 0):
            continue
        if t + AFTER_BARS >= n:
            continue
        after = (close_mid[t + AFTER_BARS] - close_mid[t]) / atr[t]
        recs.append({
            "pair": pair, "me_time": idx[t], "month": pd.Period(idx[t], freq="M"),
            "year": idx[t].year, "into": float(into[t]), "after": float(after),
        })
me = pd.DataFrame(recs)
print(f"\nunconditional month-end obs: {len(me)} (pairs x month-ends), "
      f"months={me['month'].nunique()}", flush=True)

# Cross-sectional reversion coefficient per month: a fader goes -sign(into); reversion 'works' when
# after has the OPPOSITE sign to into across the cross-section. coef = -corr(into, after) across the
# >=4 pairs that month (positive coef = reversion active; negative/zero = continuation/overrun).
mc = []
for m, g in me.groupby("month"):
    if len(g) < 4:
        continue
    if g["into"].std() == 0 or g["after"].std() == 0:
        continue
    coef = -float(np.corrcoef(g["into"], g["after"])[0, 1])
    # also a magnitude-aware fade return: long-the-loser/short-the-winner, sized by |into|
    fade_ret = float((-np.sign(g["into"]) * g["after"]).mean())
    mc.append({"month": m, "year": m.year, "rev_coef": coef, "fade_ret": fade_ret, "n": len(g)})
mc = pd.DataFrame(mc).sort_values("month").reset_index(drop=True)
print(f"monthly reversion-coef series: {len(mc)} months", flush=True)

# -------- 2) month-to-month AUTOCORRELATION of efficacy (FALSIFIER 1) --------
for col in ["rev_coef", "fade_ret"]:
    s = mc[col].to_numpy(float)
    if len(s) > 3:
        ac1 = float(np.corrcoef(s[:-1], s[1:])[0, 1])
        se = 1.0 / np.sqrt(len(s))  # ~SE of an autocorr under white-noise null
        print(f"  autocorr({col}, lag1) = {ac1:+.3f}   (~2SE band +-{2*se:.3f}, n={len(s)})",
          flush=True)
print(f"  mean rev_coef = {mc['rev_coef'].mean():+.3f}   mean fade_ret = {mc['fade_ret'].mean():+.3f} ATR",
      flush=True)

# -------- 3) does the dead 2014/15/16 block show LOW efficacy? (mechanism check) --------
print("\nper-year mean efficacy (rev_coef / fade_ret ATR):", flush=True)
for y, g in mc.groupby("year"):
    block = "  <-- DEAD BLOCK" if y in (2014, 2015, 2016) else ""
    print(f"  {y}: rev_coef {g['rev_coef'].mean():+.3f}  fade_ret {g['fade_ret'].mean():+.3f}"
          f"  (n_mo {len(g)}){block}", flush=True)
dead = mc[mc["year"].isin([2014, 2015, 2016])]
good = mc[~mc["year"].isin([2014, 2015, 2016]) & (mc["year"] <= 2020)]
print(f"\n  DEAD 2014-16 mean fade_ret {dead['fade_ret'].mean():+.3f} (n {len(dead)}) vs "
      f"GOOD other-IS {good['fade_ret'].mean():+.3f} (n {len(good)})", flush=True)

# -------- 4) does PRIOR-month efficacy predict the TRADEABLE me_long fires? (FALSIFIER 2) --------
# tradeable me_long: into <= -1.0 ATR at month-end; honest +1R capture + fwd drift on those fires.
restrict = {}
fire_times = {}
for pair in PAIRS:
    df = panel.pair_dfs[pair]
    _, is_last, into, atr = _month_end_into_move(df, INTO_BARS, 14)
    fire = is_last & np.isfinite(into) & np.isfinite(atr) & (atr > 0) & (into <= -1.0)
    restrict[pair] = fire
    fire_times[pair] = set(df.index[np.where(fire)[0]])

obs = observe_long_capture(panel, sl_mult=2.0, hold=120, drift_bars=AFTER_BARS,
                           restrict=restrict, direction="long")
obs["month"] = obs["signal_time"].dt.to_period("M")
obs["year"] = obs["signal_time"].dt.year
# prior-month efficacy joined by month (shift the monthly series by 1)
mc_idx = mc.set_index("month")
prior_coef = mc_idx["rev_coef"].reindex(
    [m - 1 for m in obs["month"]]).to_numpy()
prior_fade = mc_idx["fade_ret"].reindex(
    [m - 1 for m in obs["month"]]).to_numpy()
obs["prior_rev_coef"] = prior_coef
obs["prior_fade_ret"] = prior_fade

print(f"\ntradeable me_long fires (into<=-1.0 ATR): {len(obs)}  "
      f"base capture {obs['capture'].mean():.4f}  base drift {obs['fwd_drift_atr'].mean():+.3f}",
      flush=True)

f = obs.dropna(subset=["prior_fade_ret"]).copy()
print(f"fires with a prior-month efficacy value: {len(f)}", flush=True)
# split fires by prior-month efficacy sign
for lbl, sub in [("prior fade_ret > 0 (flow ON)", f[f["prior_fade_ret"] > 0]),
                 ("prior fade_ret <=0 (flow OFF)", f[f["prior_fade_ret"] <= 0])]:
    if len(sub):
        print(f"  {lbl}: n {len(sub):4d}  capture {sub['capture'].mean():.4f}  "
              f"drift {sub['fwd_drift_atr'].mean():+.3f}", flush=True)
# the decisive cut: among fires, does prior-on vs prior-off separate the dead block?
print("\n  conditional capture within the DEAD 2014-16 block vs prior efficacy:", flush=True)
fd = f[f["year"].isin([2014, 2015, 2016])]
for lbl, sub in [("  dead & prior-ON ", fd[fd["prior_fade_ret"] > 0]),
                 ("  dead & prior-OFF", fd[fd["prior_fade_ret"] <= 0])]:
    if len(sub):
        print(f"  {lbl}: n {len(sub):3d}  capture {sub['capture'].mean():.4f}  "
              f"drift {sub['fwd_drift_atr'].mean():+.3f}", flush=True)
print("\nDONE.", flush=True)
