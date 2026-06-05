# Arc 1029 — O1: inelasticity-STATE conditioning via the calendar-flow-DENSITY proxy

> **Arc id:** 1029 · **Chat:** 1000–1999 (continuous) · **Date:** 2026-06-05
> **Final verdict:** **KILL (cheap-kill at observation)** — concentrating the proven month-end reversion
> onto its highest forced-flow-density windows (quarter-end / fiscal-year-end) does NOT raise edge-per-trade
> monotonically and **WORSENS fold resolution** (high-density subset 7/11 negative years vs the full pool's
> 2/11). The apparent monotone capture (T1 0.521 → T2 0.543 → T3 0.600) is an artifact of **March (n=10,
> capture 1.000 = small-sample luck)**; **December — the highest global-rebalancing month — is the WORST
> (0.440, below coin-flip)**, and by-month captures are noise (ordinary January highest at 0.769). Calendar
> density is "not a lever" (the arc-3007 non-monotone tell); it just thins the edge → the council's exact
> predicted death (#1). The fold-resolution attack via density is dead.
> **Idea source:** `DISCOVERY_DIRECTION.md` MENU item **O1** (the strategist council's #2-EV `explore-now`),
> the one item that attacks the book's actual blocker (per-year fold resolution) — tested via its single
> council-rated-LIVE proxy, forced-flow calendar density.

Scored by inspection of the canonical D1 USD-major panel (real bid/ask, 5ers_eet) via the direction-aware
honest capture harness. No pool/engine — the observation is decisive (§5d). OOS never touched.

## (a) Log read + synthesis

(Carried from arcs 1027/1028, same chat/session — corpus mature; path-B closed, 3021; lever = operator
gate-governance.) O1 is the MENU's #2 and the only item targeting the actual deficit: the 4 survivors are
mean-positive but **too thin to resolve per-year folds** (arcs 2016/17/19); the only surviving mechanism is
FORCED FLOW and the only surviving calendar driver is MONTH-END (arc 3015). O1's hypothesis: an
inelasticity-STATE variable (the council named 3 collinear proxies — forced-flow calendar-density, entry-bar
spread-z, trigger-shallowing) concentrates risk into high-inelasticity bars, raising edge-per-trade without
adding coin-flip trades → a direct attack on fold resolution. The council rated **calendar-density** the one
LIVE proxy (spread-z is a cost trap; shallowing just confirms arc 1025 — fbr doesn't thicken), so I tested it.

## (b) Idea + because

Forced rebalancing flow is NOT uniform across month-ends: it is larger at **quarter-end** (Jun/Sep) and
largest at **fiscal-year-end** (Mar = Japan FYE 3/31, the biggest index/pension rebalancing; Dec = calendar
& global YE). **Because** the month-end reversion (me_long arc 1011 / me_short arc 1019) is the post-fix
reversion of inelastic WMR/index rebalancing, a higher-density window should over-extend more and revert
harder → reversion edge-per-trade rising **monotonically** with calendar density. **Falsifiable prediction
(O1):** per-trade reversion capture/drift rises T1 < T2 < T3, AND a density-concentrated book has a higher
worst-fold than the full edge. **Falsifier:** non-monotone in density (arc-3007 "not-a-lever" tell) OR
concentration worsens fold resolution (the council's death #1: smaller sample → worse thinness).

## (c)/(d) Observation → verdict (cheap-kill)

Pooled BOTH proven legs as reversion trades (me_long fades a big DOWN move into ME; me_short fades a big UP
move into ME — direction-aware honest +1R-before-SL capture IS the reversion edge per trade), D1, 7 USD
majors, IS 2010-2020, threshold 1.0 ATR / into_bars 2 (the arc-1011/1019 config). n=248. Density tier from
the month that ended at the fire bar: **T3** = {Mar, Dec} (fiscal-YE), **T2** = {Jun, Sep} (quarter-end),
**T1** = the other 8 (ordinary).

**By density tier:**

| tier | n | capture | drift mean | drift median |
|---|---|---|---|---|
| 1 (ordinary) | 167 | 0.521 | 0.080 | 0.135 |
| 2 (quarter-end) | 46 | 0.543 | 0.187 | 0.178 |
| 3 (fiscal-YE) | 35 | 0.600 | 0.076 | 0.128 |

The capture looks monotone — **but it is a small-sample artifact.** The by-month detail decomposes T3:
**March n=10 capture 1.000** (10/10 winners = noise at this scale) vs **December n=25 capture 0.440**
(below coin-flip, below ordinary T1). The two fiscal-YE months DISAGREE wildly; December — the single
highest global-rebalancing month — is the *worst*. Drift is non-monotone (T2 0.187 > T3 0.076). By-month
captures are noise: ordinary **January is highest (0.769, n=26)**; quarter-ends Jun/Sep are 0.563/0.500 ≈
coin-flip. There is no calendar-density structure.

**Fold-resolution proxy (decisive — the council's death #1):** restricting to high-density (tier≥2) makes
per-year resolution WORSE, not better:

| | neg-years | mean drift | trades/yr |
|---|---|---|---|
| ALL month-ends | **2 / 11** | +0.099 | ~23 |
| HI-DENSITY (tier≥2) | **7 / 11** | +0.085 | ~7 (median) |

Concentration thins ~23→~7 trades/yr and the per-year drift swings wildly (2010 +0.36, 2011 −0.06, 2013
−0.10, 2016 +0.66, 2019 +0.58) → 2/11 negative years becomes 7/11. The full, denser pool is MORE
fold-stable. Concentrating onto "more inelastic" windows trades one thinness problem for a worse one.

## Final verdict — KILL (cheap-kill at observation)

Calendar-flow density does NOT concentrate the month-end reversion edge — it is non-monotone in density
(the highest-density month, December, is the worst; the monotone-looking tier capture is March's n=10 fluke)
and concentration WORSENS fold resolution (2/11 → 7/11 neg-years). O1's single council-rated-LIVE proxy is
dead; the inelasticity-state fold-resolution attack fails on its best proxy. §5f does not bite (falsified at
observation, no new entry to run an exit menu on). No engine / null / council spent; OOS never touched.
Components UNCHANGED (all 4 PORTFOLIO); the corpus's lever remains the operator gate-governance call.

## Lessons (candidate for LESSONS.md compression)

1. **Forced-flow calendar DENSITY does not concentrate the month-end reversion edge.** Pooled me_long+me_short
   reversion capture is non-monotone in density (quarter-end ≈ ordinary; the two fiscal-YE months disagree —
   March n=10 a 1.000 fluke, December 0.440 worst-of-all; ordinary January highest at 0.769). The
   inelasticity-state proxy the council rated LIVE is the arc-3007 "not-a-lever" pattern. The month-end edge
   is not denser/cleaner at higher-rebalancing windows — its thinness is uniform, not concentrable.
2. **Concentrating a thin edge onto "higher-inelasticity" windows WORSENS fold resolution, not improves it.**
   High-density-only goes 2/11 → 7/11 negative years (the full pool is more fold-stable). This is the
   council's predicted death #1 made concrete and independently re-confirms arcs 2016/17 (components too thin
   to resolve folds) and 1025 (fbr doesn't thicken): you cannot buy fold-resolution by sub-sampling a thin
   edge — fewer trades = higher per-year variance = MORE trip-chances on the per-year gate.
3. **Decompose a "monotone tier" by its members before believing it.** A clean-looking T1<T2<T3 capture trend
   collapsed once T3 was split into March (n=10, 1.000) vs December (0.440). Tier aggregates hide small-sample
   flukes; check the per-member (here per-month) breakdown — the same median-vs-mean / per-pair discipline,
   on the calendar axis.

## Threads / what didn't help

- **Closed:** O1's calendar-density proxy (the council's LIVE one). The other two O1 proxies are
  pre-discounted: **spread-z** is a flagged COST TRAP (wide-spread bars are exactly where you pay most — any
  test must charge the realized wide spread at entry, else Arc-10-class optimism); **trigger-shallowing**
  confirms arc 1025 (fbr depth and edge are coupled). A spread-z test charging the honest wide spread remains
  a (low-EV, cost-trap-flagged) untested sub-thread, but the inelasticity-state thesis is substantially closed
  by its live proxy failing.
- **Operative state unchanged:** lever = operator gate-governance (path-A; arcs 2019/3021). The `explore-now`
  frontier is now largely exhausted from this chat: M1 (1027), Q1 (1028), O1 (1029) all KILL; only L1
  (triangulation second moment, cost-walled) and the cost-trap spread-z sub-proxy remain.

## Flags (code NOT merged)

None (engine/canonical untouched; reused `month_end_signals` + the canonical `observe_long_capture`). No new
BUILT tool (the density-tier tag is a trivial inline calendar map). Driver:
`discovery/_disco1_work/arc1029_o1_calendar_density.py`.

## Reproduction

`Panel.from_pairs([EURUSD GBPUSD AUDUSD NZDUSD USDJPY USDCAD USDCHF], "D1", histdata_root=
C:\Users\panap\histdata_backup, cache_root=data/cache, boundary_convention="5ers_eet")`; fires from
`MonthEndReversionLongSignal/ShortSignal(threshold_atr=1.0, into_bars=2)`; direction-aware
`observe_long_capture(restrict=fires, drift_bars=2)`; IS 2010-2020; density tier {Mar,Dec}=3 / {Jun,Sep}=2 /
else 1. Driver: `discovery/_disco1_work/arc1029_o1_calendar_density.py`.
