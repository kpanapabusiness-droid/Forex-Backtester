# arc_1060 — POST-month-end TREND RESUMPTION (regime-orthogonal 5th-leg candidate)

- **Chat:** 1000s
- **Date:** 2026-06-06
- **Type:** observation cheap-kill (§5d), fresh-eyes novel mechanism (§5a); BUILT tools only;
  no engine / null / council; no canonical change; no FLAG; OOS untouched.
- **Disposition:** **KILL**
- **Components UNCHANGED** (all 4 PORTFOLIO; me_long-solo the honest deploy object).

## Fresh-eyes (step a)

Read protocol + DISCOVERY_LOG (both tiers) + LESSONS + TOOL_REGISTRY + DISCOVERY_DIRECTION +
NEEDS_ENABLEMENT; pulled main (up to date); no `discovery/STOP`. The corpus is at a deeply-documented
terminus: 4 PORTFOLIO components (gap 1006, me_long 1011, fbr 1013, me_short 1019); the 4-way book is
never all-folds-positive (the 2015/2018 strong-USD wall) and co-sim (item E) confirmed that failure is
**fundamental, not a linear-combiner artifact**; the honest deploy object collapsed to me_long-solo
(1046), which is vehicle-infeasible; the strategist `explore-now` MENU (M1/O1/L1/Q1/G1; S1 a modifier)
is exhaustively closed; closed ground covers all shallow directional (long+short), calendar forced-flow,
microstructure, RV/triangulation (both moments), regime conditioning, and structural conjunctions. The
sibling 2000s chat is saturating the vehicle-geometry + residual calendar-forced-flow lane (2053-2063).
The binding gate the book has always needed (arc 1020/2019 spec): a 5th component **positive in 2015 AND
2016 without dragging 2018, and NON-reversion** — arc 2019 proved a 5th *reversion* leg cannot make the
book AFP. Per §2 / §5a / the arc-3004 "apparatus is incapable is seductive" warning, I tested one
genuinely-untested, mechanistically-grounded NON-reversion construction aimed squarely at that gate.

## Idea + because

Arc **1059** just data-confirmed *why* the reversion book dies in strong-USD years: the big down-move
into month-end **IS the trend and continues** (me_long capture falls monotonically with USD breadth; the
HI-breadth tercile = 0.405, sub-coin-flip). The untested complement of the SAME mechanism is **trend
RESUMPTION**: month-end rebalancing is a *temporary* counter-trend dislocation (the reversion edges
harvest the snap-back); once it clears (the first days of the new month) the **prevailing trend
re-asserts**. Entering at new-month start in the prevailing-trend direction should be positive precisely
in the strong-trend years (2014-16, 2018) where the reversion book fails — the regime-orthogonal
NON-reversion leg.

Why this is **not** closed shallow momentum (§5a): it is calendar-GATED (fires only at start-of-month,
after the flow clears), keyed to a specific market-mechanism (flow-clearing trend resumption), and the
KILL/PROCEED criterion is not "does it beat cost overall" (a trend leg is overall-coin-flip by
construction) but **"is it reliably positive in the strong-trend folds 2014-16 AND 2018 the book needs."**
A trend leg can be overall-coin-flip and still be the regime-orthogonal 5th leg *if* its positive years
are exactly the book's negative years.

## Method

BUILT tools only. `observe_long_capture` (direction-aware honest take-the-loss capture + N-bar forward
drift) over IS 2010-2020, D1, 7 USD majors. For each pair, flag the k-th trading bar of each month
(causal calendar position); prevailing-trend direction = sign of the trailing `trend_lb`-bar return at
that bar (causal); strength = |return| / ATR. Build long/short restrict masks (long if trend up & strong,
short if trend down & strong) and observe honest capture/drift in the trend direction. Swept k ∈ {1,3,5}
(days into the new month), trend_lb ∈ {20,60}, strength ∈ {0,1,1.5} ATR, hold/drift — as cheap-obs
robustness, not optimization. Driver: `discovery/_disco1_work/arc1060_newmonth_trend_resumption.py`.

## Result — coin-flip-or-worse, and the binding fold (2015) is robustly NEGATIVE

Default cell (k=3 entry, 60d trend, strong≥1ATR, hold20/drift10): **n=755, capture 0.4252 (< coin-flip),
drift_mean −0.0274, drift_median −0.1408**; per-year 5/11 negative, worst −0.85, mean-of-year −0.046.

- **(A) Sub-coin-flip everywhere; the typical trade loses.** Capture 0.42–0.44 in every cell;
  **drift_median negative in every cell** (−0.13 to −0.19). The occasional positive *year-mean* is
  carried by a handful of trend-tail winners while the median trade loses — the convex-tail-carried
  pattern arc 2063 flagged as a KILL tell. Mechanistically: under take-the-loss, trend entries touch the
  2·ATR stop on the pullbacks before reaching +1R, so honest capture stays sub-coin-flip even when raw
  drift is occasionally positive. This is closed-ground momentum, confirmed honestly.
- **(B) The decisive fold fails: 2015 is robustly NEGATIVE.** Strong-USD folds [2014, 2015, 2016, 2018]
  by cell: default +0.224 / **−0.304** / +0.138 / +0.265; k=1 +0.626/**−0.275**/+0.181/+0.082; k=5
  +0.372/**−0.249**/+0.147/+0.467; 20d-trend −0.674/**−0.145**/−0.109/+0.569; all-trends
  +0.205/**−0.271**/+0.163/+0.284; strong≥1.5 +0.181/**−0.213**/+0.181/+0.240. **2015 is negative in 6 of
  7 cells** (the only positive is the degenerate hold-10/drift-5 cell: +0.093 with capture cratered to
  0.32 — noise). The gate needs 2015 **and** 2016 **and** 2018 positive; 2015 kills it.
- **(C) Not the me_long complement.** me_long's negative IS years are 2014/2015/2016; resumption drift
  there is +0.224 / **−0.304** / +0.138 — in **2015 both directions are negative**, so the resumption
  cannot cover the book's single binding fold even as a decorrelated leg.

## Diagnosis + meaning

The post-month-end trend resumption is (1) coin-flip-or-worse overall (sub-0.50 capture, negative median
drift — gross-negative on the typical trade, before costs), confirming closed-ground momentum under an
honest take-the-loss lens, and (2) **not reliably positive in the strong-USD folds** — the single most
binding fold, 2015, is robustly negative across the lever sweep. Crucially, **2015 defeats BOTH the
reversion and the trend-resumption** at the monthly horizon: 2015 (CHF de-peg in January + whipsaw,
choppy-then-trending USD) was not a clean persistent trend you could resume — strong moves that violently
reversed — so the trend side gets whipsawed exactly where the reversion side gets trend-run. The
strong-USD wall is therefore **not a "reversion fails here, trade the trend instead" directional-coverage
gap** that a NON-reversion leg could fill; it is a genuinely-hard price-structure regime that no
single-direction OHLC mechanism (reversion OR continuation) covers. §5f exit-sweep is moot — the failure
is at the fold-sign level (which years are positive), which no exit/SL choice moves, and the construction
is gross-negative on the median trade regardless.

## NEW lesson

The reversion book's strong-USD wall (2015/2018) is **not a directional-coverage gap.** Arc 1059 showed
the reversion fails in strong-USD years because the move continues; the natural inference is "add the
trend-continuation side as the regime-orthogonal leg." This arc falsifies that inference: the
trend-resumption is *also* negative in the binding 2015 fold, because 2015 is a whipsaw regime (CHF
de-peg + violent reversals) that runs the reversion AND whipsaws the trend. A regime where reversion
fails does NOT imply continuation succeeds — both can fail when the regime is choppy-trending rather than
cleanly trending. Adds **trend-resumption (post-flow continuation)** to the mapped-dead non-reversion
attempts at the 2018/2015 leg (~22 short/continuation routes): the binding folds are hard price-structure
regimes, not coverable by flipping direction. Under take-the-loss, a trend leg is additionally
self-penalized (stop hit on pullbacks → sub-coin-flip honest capture), so the trend side is *structurally*
worse than the reversion side on this engine, not just a coin-flip.

## Threads / handoff

Components UNCHANGED (all 4 PORTFOLIO; me_long-solo the honest deploy object, vehicle-infeasible). Lever =
operator path-A. This arc closes the "trend-side complement of the reversion book" as the regime-orthogonal
5th leg, and sharpens the terminus: the binding 2015/2018 folds are hard price-structure regimes, not
directional gaps. Within the OHLC-only charter the EDGE frontier stays mined out; the next genuinely-new
direction needs a charter unlock (operator-gated macro/options/COT/calendar data, `NEEDS_ENABLEMENT.md`)
or the operator's path-A decision. **Datum:** new-month trend-resumption (k=3, 60d, strong≥1ATR) capture
0.425 (sub-coin-flip), drift-median negative all cells; strong-USD folds 2014 +0.22 / 2015 −0.30 / 2016
+0.14 / 2018 +0.27 (2015 negative in 6/7 sweep cells). Driver
`discovery/_disco1_work/arc1060_newmonth_trend_resumption.py` (BUILT tools only, single-use diagnostic).
