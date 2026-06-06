# arc_1061 — JAPANESE FISCAL-YEAR-END (March 31) JPY REPATRIATION seasonal

- **Chat:** 1000s
- **Date:** 2026-06-06
- **Type:** observation cheap-kill (§5d), fresh-eyes novel mechanism (§5a); BUILT tools only;
  no engine / null / council; no canonical change; no FLAG; OOS untouched.
- **Disposition:** **KILL**
- **Components UNCHANGED** (all 4 PORTFOLIO; me_long-solo the honest deploy object).

## Fresh-eyes (step a)

Pulled main (up to date; highest in-range arc 1060 → resume 1061); read protocol + DISCOVERY_LOG
(both tiers) + LESSONS + TOOL_REGISTRY (canonical + all BUILT); no `discovery/STOP`. The corpus is at
a **rigorously-proven** terminus, not a soft one:
- 4 PORTFOLIO components (gap 1006, me_long 1011, fbr 1013, me_short 1019); the 4-way book is never
  all-folds-positive (the 2015/2018 strong-USD wall); co-sim (item E) confirmed that failure is
  **fundamental, not a combiner artifact**; the honest deploy object collapsed to me_long-solo (1046),
  vehicle-infeasible (2033/2059) and non-certifiable (1057/1058).
- The **5th-leg route is structurally closed** (arc **2022**): even the strongest possible 5th leg
  (shock-continuation 3019 — positive in BOTH 2015 and 2018, the most-decorrelated component in the
  corpus) fails the honest gate via a **weighting dilemma** — a high-variance tail leg that can lift a
  deep blocker fold is throttled to a tiny weight by honest risk-parity, and the weight that suffices
  is overfit (1383/10626 IS-optimized weightings give AFP = weight-painting).
- The **densification route (path-B) is closed by portfolio-math PROOF** (arc **3021**): at the corpus's
  own residual correlation ρ≈+0.12 a shared dollar/risk factor floors book variance, so P(AFP) plateaus
  ~0.30 and is unreachable at ANY leg count N.
- The EDGE frontier is closed across ~60 arcs on every documented axis (shallow directional long+short,
  reversion, trend/convexity/shock, calendar forced-flow, RV/triangulation both moments, regime
  conditioning, structural conjunctions, state-conditioning of the deploy object).

Per §2 / §5a + the arc-3004 *"apparatus is incapable is seductive"* warning, I first interrogated the
strongest remaining candidate (a long-volatility / convexity leg as the regime-orthogonal 5th leg) and
found it **already closed by 2022** (shock-continuation is exactly that, and the weighting dilemma kills
it). I then tested one **documented mechanism the corpus has genuinely never touched**, with a *because*
independent of the binding folds (no outcome-aware fold-selection — §1/§4 clean).

## Idea + because

Japan's fiscal year ends **March 31**. Japanese institutions (life insurers, corporates, pensions)
repatriate foreign earnings before the books close → sell foreign currency, **buy JPY** → JPY-quoted
crosses (XXXJPY) should **fall** into late March. This is a cited, recurring FX seasonal — the "fiscal
year-end repatriation" / Japanese-exporter flow — **distinct** from the corpus's already-mapped GENERIC
monthly month-end (me_long/me_short fire EVERY month on USD majors) and turn-of-month USD drift (1005).
Falsifiable claim: **SHORT XXXJPY** in the final week of March has a real, **March-specific** edge.

Why this is **not** closed ground / not a re-grind / not fold-fished: it is calendar-GATED,
currency-SPECIFIC (JPY-quote pairs only), keyed to ONE month with a documented institutional mechanism;
the decisive control is **March vs other months** in the same late-month window (a generic late-month
effect would be month-end, already mapped); and the *because* predates and is independent of 2015/2018.

## Method

BUILT tools only. `observe_long_capture` (direction-aware honest take-the-loss capture + N-bar forward
drift), IS 2010-2020, **D1, 7 JPY-quote crosses** (USDJPY, EURJPY, GBPJPY, AUDJPY, NZDJPY, CADJPY,
CHFJPY). Repatriation window = calendar **day-of-month ≥ 24** in March (the last ~week; day-of-month is
known AT the bar → causal, unlike "last N trading days" which needs future bars). Controls: (B) same
late-window in **non-March** months (March-specific test), (D) the **LONG** direction (is the spot
effect reversed?), (E) day≥20 + longer hold robustness. SHORT-drift is signed so a *falling* XXXJPY
reads positive. Driver: `discovery/_disco1_work/arc1061_jpy_fiscal_yearend_repatriation.py`.

## Result — the textbook direction is BACKWARD; the reverse tilt is real-but-sub-capturable

**(A) The repatriation SHORT is falsified.** Late-March (day≥24) SHORT XXXJPY: **capture 0.240**
(far below coin-flip), **drift_mean −0.288** (short-signed → XXXJPY actually **ROSE**, i.e. JPY
**WEAKENED**, the OPPOSITE of the repatriation narrative), **0/7 pairs** positive. The textbook
"repatriation buys JPY into FYE" does not hold in spot at the honest gate.

**(B) But it IS March-specific — in the reverse direction.** March-excess (March minus non-March
late-window, short): **capture −0.075, drift −0.213** (day≥20: −0.098 / −0.285). March is *more* adverse
to the short than other late-months → there is a genuine **March-specific JPY-WEAKNESS tilt** (≈+0.17
ATR of XXXJPY drift vs other months). This is consistent with the empirical literature that
repatriation flows are **forward-hedged and pre-positioned**, so spot JPY tends to *weaken* into the
fiscal year-end rather than strengthen (plus new-fiscal-year April outward-investment positioning).

**(D) The reverse (LONG XXXJPY) is real but not honestly capturable.** Late-March LONG: drift_mean
**+0.251**, 7/11 positive years — but **capture only 0.327** (sub-coin-flip under take-the-loss: the
drift comes with enough adverse excursion that a long is stopped at 2·ATR before +1R two-thirds of the
time), and the mean is **convex-tail-carried** (2011 +2.35, 2013 +1.14 dominate; 2016 −1.50, 2017 −0.94,
2020 −0.41 negative) — the arc-2011/2063 thin-tail KILL tell. **It does not cover the binding folds:**
2015 +0.002 (≈0), 2016 **−1.497**, 2018 +0.362 → not the 2015-and-2016-and-2018-positive leg the book
needs. (E) day≥20 / hold-12 robustness reproduces the same picture.

## Diagnosis + meaning

A documented institutional calendar-flow seasonal, tested honestly, is (1) **directionally backward to
its own textbook narrative in spot** — the repatriation is pre-hedged, so spot JPY *weakens* into March
31, it doesn't strengthen — and (2) the real reverse tilt (March-specific JPY weakness, ~+0.17 ATR) is
**sub-capturable under take-the-loss** (capture 0.327), **thin** (one event per year → un-scalable, the
arc-1017/3019 lesson), **convex-tail-carried** (two years dominate), and **does not cover the binding
folds**. §5f exit-sweep is moot: a sub-0.50-capture / convex-tail entry has no honest exit that lifts it,
and the failure is at the fold-sign / capturability level, not the exit.

## NEW lesson

**Documented discrete calendar-FLOW seasonals on liquid FX are arbitraged to sub-cost / sub-capturable
at the honest take-the-loss gate — and the narrative DIRECTION is not even reliable.** The Japanese
fiscal-year-end "repatriation buys JPY" story is *backward* in spot (pre-hedged flow → spot JPY weakens
into FYE); honestly tested, the only real signal is a thin, convex-tail-carried, sub-0.50-capture
JPY-weakness tilt that cannot be deployed and does not cover the binding folds. This adds JPY-FYE
repatriation to the mapped-dead calendar-flow family (gotobi 1008, round-number 1010, month-end-fix
3008, turn-of-month 1005, IMM-roll 2057, quarter-end 2058): **every documented discrete calendar flow
the corpus has tested is either sub-cost, sub-capturable, or directionally backward to its narrative** —
the closed-ground "forward drift ≈ cost everywhere" extends to currency-specific institutional-flow
seasonals, with the added twist that a cited flow's *spot* direction can be the reverse of its textbook
story once hedging/pre-positioning is accounted for. A novel *because* still earns its test (this one
did); the test still lands on the closed-ground floor.

## Threads / handoff

Components UNCHANGED (all 4 PORTFOLIO; me_long-solo the honest deploy object, vehicle-infeasible). Lever
= **operator path-A** (gate-governance decision on a sound mean-positive ~3-bet book — arcs 1023/2021/
3021) **OR a charter unlock** (operator-gated macro/options/COT/calendar-flow data,
`NEEDS_ENABLEMENT.md`). This arc is a fresh-chat (1000s) confirmation that the within-charter EDGE
frontier is genuinely exhausted: the strongest remaining EDGE candidate (long-vol/convexity) is closed
by 2022's weighting-dilemma, path-B is closed by 3021's portfolio-math proof, and a genuinely-novel,
documented, clean calendar-flow mechanism (JPY FYE repatriation) lands on the closed-ground floor.
**Marginal EV of further autonomous within-charter arcs is near-zero** absent an operator unlock — the
remaining edge lane (a novel OHLC mechanism positive in BOTH 2015 and 2018 at deployable variance) is,
per arc 2022, of a different-mechanism profile in two different-regime years and cannot be found by clean
search without §1/§4-violating fold-selection. **Datum:** JPY-FYE late-March (day≥24) SHORT capture
0.240 / 0/7 pairs (repatriation narrative backward); LONG drift +0.251 but capture 0.327 sub-coin-flip,
convex-tail-carried (7/11; 2011/2013 dominate), binding folds 2015 +0.00 / 2016 −1.50 / 2018 +0.36.
Driver `discovery/_disco1_work/arc1061_jpy_fiscal_yearend_repatriation.py` (BUILT tools only, single-use
diagnostic; no new BUILT tool registered — pure `observe_long_capture` + a causal day-of-month mask).
