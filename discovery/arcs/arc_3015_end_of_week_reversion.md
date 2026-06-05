# arc 3015 — End-of-week (Friday) position-squaring reversion: a weekly analog of month-end?

**Chat:** 3000s · **Date:** 2026-06-05 · **Verdict:** FAIL (cheap-kill at observation) → **KILL**
**Disposition:** KILL · **passed:** N

> The one demonstrably 2018-positive PORTFOLIO leg (`me`, arc 1011) is a MECHANICAL calendar-flow
> reversion, direction-agnostic — exactly the "non-price-direction construction" arc 3012 said the
> 4th leg needs. Question: is there a WEEKLY analog? Does mechanical reversion concentrate at the
> end-of-week (Friday) squaring boundary, positive in 2015 & 2018? **Falsified at observation: the
> reversion edge is month-end-SPECIFIC, it does not generalize to the weekly boundary.**

---

## Log reading (step a — FRESH EYES, honest-era only)

Resumed 3000s range at arc 3015 (highest in-range = 3014). Pulled main; no `discovery/STOP`.
State of the programme after 40 arcs:

- **Directional space comprehensively closed** (long OR short; momentum/breakout/reversion/trend;
  H1/H4/D1/**W1** added by arc 3014; majors+crosses; capture AND drift lens; stop-removed 3004):
  coin-flip ~0.49 base, forward drift ≈ cost everywhere. Regime conditioning failed across
  dispersion / vol-LEVEL / vol-EXPANSION / Kaufman-ER; strong-trend regimes INVERT. Carry OFF.
- **Three PORTFOLIO components** (net-positive long-only, NOT all-folds-positive): gap-fill JPY-cross
  H4 (1006), **month-end USD-major D1 (1011 = `me`)**, failed-breakdown-reclaim USD-major H4 (1013,
  the strongest+cleanest corpus edge, 9/10 IS).
- **Portfolio combination** (2008/3009/1015, triple-independent): the 3-way book is the strongest
  corpus result (risk-parity 2/10 neg, worst −0.77%, mean +0.55%) but **provably blocked,
  combination-invariant, by 2015 AND 2018** (0/5151 convex weightings pass; 2015 positive only in
  fbr, 2018 positive only in `me` — mutually exclusive). Needs a **4th component positive in BOTH
  2015 & 2018** — the route is exactly ONE regime-orthogonal component from deployable.
- **2018-leg hunt exhausted across 9 directional/flow/structure constructions** — structural short
  (1014/2009/2011/3011), trend short (3010), flow short up-gap weekend (1016), deep continuation
  long (2012), relative-value (2010), risk-off vol-expansion short (3012), weekly convexity (3014),
  session-liquidity levels (3013). Refrain (arc-3012 thread #4): "the 4th leg needs a genuinely
  **non-price-direction construction** not yet conceived."

## Idea + why (the un-tested mechanical-flow boundary, with a *because*)

The corpus has hunted the 2018 leg almost entirely via **price-direction** bets (structure / trend /
vol-state / relative-value, long and short) and every one fails by the SAME reason: 2018's USD
strength is **mean-reverting bursts / capitulations**, so any *fade* bleeds and any *vol/structure
short* shorts into the bounce (arc-3012 diagnosis). The ONE construction that is demonstrably
**+2018** is `me` (month-end reversion, +0.90 in the 2018 fold) — and it works **because it is a
MECHANICAL calendar flow** (WMR/index rebalancing reverts a trend-extension regardless of trend
direction), not a price-direction prediction. That is precisely the "non-price-direction
construction" arc 3012 flagged as the surviving frontier.

So: **is `me` a singular calendar anchor, or one instance of a generic "calendar-boundary squaring"
effect?** The natural untested sibling is the **end-of-week (Friday) position-squaring** boundary —
traders flatten / de-risk into reduced Friday-afternoon and weekend liquidity, so an over-extended
intra-week move should partially REVERT, mechanically and direction-agnostically, like month-end.
It is ~4× more frequent than month-end (clears the pool floor easily) and is distinct from both
existing flow edges: from `me` (monthly timing) and from the weekend gap-fill 1006 (which trades the
Monday-OPEN gap discontinuity, not the Friday-CLOSE intra-week over-extension).

**Acceptance test (pre-registered, per arc-3010/3012 regime-luck screen):** a Friday-boundary
reversion that (1) clears cap > 0.50 with a robust (median, per-pair, pool-floor) positive drift,
AND (2) is robustly positive in BOTH 2015 & 2018 — not regime-luck within a coin-flip.
Observation-first cheap-kill (§5d) before any engine spend.

## Method (CALLED canonical; observation only)

`Panel.from_pairs` (D1, cached, 5ers_eet) on the 7 USD majors (`me`'s universe). BUILT direction-aware
`observe_long_capture` (honest +1R-before-SL long CAPTURE + forward drift in ATR; CHARACTERIZATION
ONLY — gross, not a gate). Conditioner: the `me`-style big DOWN move into bar i
(`into = (close_mid[i]−close_mid[i−2])/ATR <= −1.0`, the long-reversion setup = buy the sold side),
grouped by **day-of-week of bar i** (0=Mon..4=Fri; Friday entry i+1 = Monday), then dow × year for the
2015/2018 test, and the month-end (`me`) anchor for contrast. Best-version sweep: threshold
{1.0,1.5,2.0} × drift horizon {2,5}. Drivers: `_disco3_work/arc3015_observe_dow_reversion.py` +
`arc3015_friday_bestversion.py` (reproducible).

## What happened — FALSIFIED at observation

**Generic "big down move → buy reversion" (pooled, n=4225):** cap **0.4892** (<0.50), drift_mean
−0.031 / median −0.019 — generic reversion is dead (re-confirms the arc-3000/3001 finding).

**No day-of-week concentrates the reversion** — the whole hypothesis falls here:

| dow | n | cap | drift_mean | drift_med |
|---|---|---|---|---|
| 0 Mon | 790 | 0.504 | −0.022 | −0.033 |
| 1 Tue | 876 | 0.459 | −0.077 | −0.060 |
| 2 Wed | 944 | 0.482 | −0.009 | +0.025 |
| 3 Thu | 888 | 0.487 | −0.014 | −0.006 |
| **4 Fri** | **261** | **0.502** | **−0.045** | **−0.083** |

Friday is cap **0.50 / negative drift** — no end-of-week squaring edge. (Anchor: the **month-end**
subset of the same setup is cap **0.511 / drift +0.193 / median +0.247** — POSITIVE, reproducing `me`.
Month-end concentrates reversion; the weekly boundary does not.)

**The 2015/2018 acceptance test FAILS.** Friday by year: **2018** cap **0.421** / drift **−0.346**
(robustly negative — the binding fold), **2015** cap **0.417** / drift_med **−0.51** (coin-flip /
typ-negative). Excluding month-end Fridays (orthogonal to `me`): n=242, cap 0.496, drift −0.064 — same
coin-flip. **Per-pair Friday is a USD-quote-beta split** (EURUSD/AUDUSD/NZDUSD/USDJPY mildly +,
USDCAD/USDCHF −) — the arc-2009/3012 directional-USD-beta tell, 4/7 = noise, not a mechanism.

**Best-version sweep does not rescue it.** Deepening the move thins below the pool floor before it
turns positive: thr1.0 (n=261) cap 0.50 / 2018 −0.38; thr1.5 (n=73) the only "positive" drift cells
rest on **2 samples in 2015 (cap 0.0) and 5 in 2018** = thin-tail noise (arc-2011/3012 mean≫median
tell); thr2.0 (n=21) has 2015 n=1, 2018 n=0. No robust, pool-floor-clearing Friday cell is
2015-AND-2018-positive with cap > 0.50.

## Diagnosis — month-end reversion is a SINGULAR anchor, not a generic boundary effect

`me` survives 2018 because month-end rebalancing is a **hard, quantified, inelastic** flow (index /
WMR-fix rebalancing forced by mandate) that mechanically reverts the month's over-extension on a
date-certain. The **weekly boundary has no equivalent forced-rebalancing driver** — Friday squaring
is a soft, discretionary de-risking that is (a) far smaller relative to the week's noise and (b)
swamped by the same directional dynamics as any other day (capture ≈ 0.50, USD-beta split). So the
calendar-flow reversion edge does NOT generalize down the calendar hierarchy: month-end ≠ week-end.
This is the §5f-doesn't-bite case (coin-flip capture, negative/median-≈0 drift, target-fold-negative —
it collapses SL-honest, arcs 3003/3012); no engine spend warranted.

## Verdict: KILL (cheap-kill at observation)

Friday end-of-week squaring confers no reversion edge (cap 0.50, drift negative at the only
pool-floor threshold), is a USD-beta split per-pair, and FAILS the 2015/2018 acceptance test (2018
robustly negative). No pool/engine/null/council spent (arc-3010/3012/3014 efficiency discipline).

## Threads / lessons

1. **The mechanical-flow reversion edge is MONTH-END-SPECIFIC, not a generic calendar-boundary
   effect.** A weekly (end-of-week / day-of-week) analog of `me` does NOT exist — Friday reversion is
   coin-flip (cap 0.50) with negative drift, while the same setup at month-end reproduces `me`
   (+0.193 drift). The thing that makes `me` work is the **hard inelastic rebalancing mandate** unique
   to month-end, not the calendar boundary per se. This narrows the "non-price-direction construction"
   frontier (arc-3012 thread #4): mechanical-flow reversion is a singular anchor, and the weekly scale
   of that well is now **dry**.
2. **9th dead route to the 2018 leg** (structure 1014/2009/2011/3011, trend 3010, flow-short 1016,
   continuation-long 2012, vol-state 3012, weekly-convexity 3014, session-levels 3013, now
   end-of-week reversion 3015). Every route except `me` fails 2018; `me` does not have a weekly twin.
3. **Generic reversion re-confirmed dead at D1** (big-down-move → buy: cap 0.489, drift −0.03),
   independent of the day-of-week conditioner — consistent with arcs 3000/3001/1011's "generic
   reversion is negative; only the month-end TIMING makes it positive."
4. **Surviving frontier (unchanged / narrowed):** the in-apparatus calendar-flow reversion leads are
   now exhausted at both the monthly (taken: `me`) and weekly (dead: 3015) scales. What remains for
   the 2018 leg is operator-gated (a tighter-cost execution regime, #3 of the arc-3004 escalation) or
   a genuinely new non-price, non-calendar-boundary construction not yet conceived. The portfolio
   route stands exactly one regime-orthogonal component from deployable, with that component
   increasingly looking like it must come from outside the H4/D1-FX-major in-apparatus search space.

## Tooling

No new BUILT tool — reused BUILT direction-aware `observe_long_capture`; the day-of-week / into-move /
month-end conditioners are one-off scratch columns (like arcs 3005/3012/2010 observers). Drivers
`_disco3_work/arc3015_observe_dow_reversion.py` + `arc3015_friday_bestversion.py` (reproducible).

## FLAGS (code not merged)

None. No canonical-core change. Carries the standing FLAG-1 (the named structure/flow/trend/vol/
calendar-boundary leads for the 2018 leg are now all dead) + the standing `A1Config.time_exit_bars`-
unwired flag.
