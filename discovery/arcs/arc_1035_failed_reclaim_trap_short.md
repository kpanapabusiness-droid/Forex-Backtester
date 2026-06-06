# arc 1035 — Failed-reclaim trap SHORT (the complement of fbr; candidate +2018 leg)

**Chat:** 1000s · **Date:** 2026-06-06 · **Verdict:** FAIL (cheap-kill at observation) → **KILL**
**Disposition:** KILL · **passed:** N

> arc 2014 documented that fbr's reclaim FAILS in strong-USD years (2018) — the failed breakdown becomes
> a REAL breakdown. **Hypothesis:** a swing low swept + briefly reclaimed (an fbr fire) that then closes
> BACK BELOW the level within M=6 bars = a bull-trap; the fbr longs are trapped → forced liquidation →
> continuation DOWN. SHORT it. It would fire exactly when fbr loses (2018) → a candidate +2018 portfolio
> leg via a genuinely NEW forced-flow mechanism (trapped-long liquidation), using the now-open short path.
> **FALSIFIED at obs: short capture 0.4086 (<0.50, adverse), per-pair 6/7 sub-0.50, drift median ≈0.**
> 2018 *looks* positive (cap 0.636 / drift +1.32) but is n=11 regime-luck within a coin-flip base (the
> arc-1016/3010/2031 signature; 2014 −1.72, 2015 −1.56, 2019 −2.93 deeply negative). The trap is just
> arc-1014's confirmed-breakdown short with an extra reclaim step — the same backward-confirming death:
> by the time the failed reclaim is CONFIRMED, the i+1 short enters the local area and reverts.

---

## Log reading (step a — FRESH EYES, honest-era; pulled main, no STOP)

Continued 1000s after landing arc 1034 (gap × structure, KILL). Loop state: range 1000-1999, next id 1035.
Corpus state unchanged from the arc-1034 frame: 4 PORTFOLIO components, mean-positive-not-AFP 4-way book,
edge-hunt structurally closed (path-B 3021 / leg-hunt 2022 / MENU exhausted), lever = operator path-A
(1032). The unfound piece across ~15 routes is a **2018-positive (strong-USD) portfolio leg**; the only
one ever found is `me_short` (1019). Every structural/trend/flow SHORT aimed at 2018 has died as
**regime-luck within a coin-flip** (structure 1014/2009/2011/3011, trend 3010, up-gap flow 1016/2013,
vol 3012, rel-value 2010, fbr-off-USD 2031).

## Idea + why (log-seeded *because*)

This is NOT a re-run of the dead structural shorts — it is a genuinely distinct construction seeded by
arc 2014's finding. fbr (1013) longs a swept-and-reclaimed swing low. arc 2014 proved that in strong-USD
2018 the reclaim does NOT hold — the failed breakdown becomes a real breakdown (this is WHY fbr is −2018,
mechanism-intrinsic). That loss mode is information: a swing low swept, briefly reclaimed (fbr fire), then
closing BACK BELOW within M bars = a **bull-trap**. The traders who bought the reclaim (fbr longs) are now
underwater below the level → stop-loss liquidation adds **forced selling** → continuation DOWN. Shorting
the failure confirmation is a forced-flow short (the corpus's winning template) that, by construction,
fires when fbr fails → exactly in the 2018 strong-USD regime → a candidate +2018 leg. Distinct from arc
1014 (one-stage confirmed breakdown, no prior reclaim) by the two-stage trap mechanism.

Falsifier (§5d): honest i+1 short capture ≤ 0.50 / pair-mixed / 2018 inside a coin-flip base.

## Observation (cheap-kill — H4, IS 2010-2020, `observe_long_capture(direction="short")`, no engine)

fbr fires (K=40 swept swing low, close reclaims, lower shadow ≥1.25 ATR) whose reclaim then FAILS (first
close back below the swept level within M=6 bars); SHORT entered at the bar after the failure confirmation.
USD majors, n=93 events. All ex-ante (swing low shift1; failure close known at its bar; entry next bar).

- **Overall short capture 0.4086 (<0.50, adverse for a short); drift +0.024 / median +0.035 ≈ 0** — the
  trap does NOT produce a tradeable continuation down.
- **Per-pair 6/7 sub-0.50** (NZDUSD 0.143, USDCHF 0.350, USDCAD 0.412, GBPUSD 0.444, USDJPY 0.458,
  EURUSD 0.400; only AUDUSD 0.545 clears) — pair-mixed (the noise tell).
- **Per-year:** 2018 cap 0.636 / drift +1.32 LOOKS like the target leg — but on n=11, inside an overall
  sub-0.50 base, with 2014 (−1.72), 2015 (−1.56), 2016 (−0.03), 2019 (−2.93) negative → **regime-luck
  within a coin-flip** (the exact arc-1016/3010/3012/2031 disqualifier).

## Diagnosis (the *why*)

This is arc-1014's confirmed-breakdown-short death with one extra step. The "trapped-fbr-longs liquidate"
mechanism is **real but un-capturable at the honest i+1 entry**: the failed reclaim is a BACKWARD
confirmation — by the time the close prints back below the swept level, the liquidation down-move is
already underway, so the i+1 short enters at/after the local low → reverts (drift ≈0). The corpus's only
+2018 short, `me_short` (1019), works precisely because the WMR month-end fix flow is **FORWARD** (the
rebalancing happens AFTER the entry bar), whereas every structural confirmation — sweep, reclaim, failed
reclaim — is backward (the move that confirms the setup is the move you needed to capture). Adding a
reclaim stage before the breakdown does not change this; it just delays the confirmation further.

## Verdict / disposition

**KILL (obs cheap-kill).** Sub-0.50 short capture, pair-mixed, 2018-positive = regime-luck within a
coin-flip → §5d, no engine/null/council (same handling as 1014/2009/1016/3010/2031). §5f does not bite
(capture <0.50, median drift ≈0 — not a non-coin-flip / gross-drift entry). OOS never touched. Components
UNCHANGED (all 4 PORTFOLIO). The 2018 leg remains unfound; lever unchanged = operator path-A.

## Threads

- Closes the "trapped-fbr-long liquidation" short. Combined with arc 1034 (gap can't compose with fbr
  structure), this session re-establishes from two new angles that **the only forward-confirming
  structural reversal in the corpus is fbr's live wick, and it has no tradeable short complement.**
- The structural-short route to 2018 is now closed across SIX constructions (1014 confirmed-breakdown,
  2009 climax-sweep, 2011/3011 reject, 3010 trend, **1035 failed-reclaim trap**) — all the same
  backward-confirming i+1 death.

## NEW lesson

A structural reversal's FAILURE is itself a backward-confirming event: by the time price confirms that a
reclaim/breakout failed, the resulting move is already underway, so the honest i+1 entry catches the local
extreme and reverts (drift ≈0) — the "trapped traders liquidate" forced-flow story is real but spent
before a no-lookahead entry can act. Only a FORWARD-scheduled forced flow (the WMR fix, `me_short` 1019)
gives a capturable +2018 short; no price-structural confirmation does. A 2018-positive per-year cell on
n≈10 inside a sub-0.50 capture base is regime-luck, never an edge (re-confirms 1016/3010/3012/2031).
