# arc 3014 — WEEKLY (W1) trend-following CONVEXITY (the last untested timeframe lever)

**Chat:** 3000s · **Date:** 2026-06-05 · **Verdict:** FAIL (cheap-kill at observation) → **KILL**
**Disposition:** KILL · **passed:** N

> The classic CTA edge on the one horizon the corpus never tested. Does weekly trend persistence
> create a real, trend-SELECTED convex tail (arc 2000's lens), and is it +2018 (a "clean weekly USD
> trend")? **No, both fail.** Timeframe-invariance extends to W1; convexity is median-negative &
> single-pair-carried; −2018 in BOTH directions.

---

## Log reading (step a — FRESH EYES, honest-era only)

Resumed 3000s at arc 3014 (prior in-range 3013). No `discovery/STOP`. State after 39 arcs: directional
space closed H1/H4/D1, majors+crosses, both lenses; three PORTFOLIO components; the 3-way book
(1015/2008/3009) is the strongest corpus result but provably blocked by **2015 & 2018**; the 2018-leg
hunt is dead across structure (1014/2009/2011), trend-short (3010), flow (1016), vol-state short
(3012, mine), continuation-long (2012), relative-value (2010), and re-leveling the best edge (3013,
mine). Refrain: escalation-bound (operator tighter-cost regime, or a non-price construction).

## Idea + why (the untested timeframe × the convexity lens × the 2018 intuition)

Two corpus facts leave exactly one un-probed cell: (1) **closed ground covers H1/H4/D1 only** — the
**WEEKLY (W1)** horizon, the classic CTA trend-following timeframe where trend persistence is strongest,
was never tested (arc 1002's "timeframe-invariant" was H4-vs-D1). (2) arc 2000's insight: trend edge
lives in **CONVEXITY (the fat right tail)**, not win-rate — but it tested only H4 and found the tail
*generic* (not trend-selected). The weekly/monthly horizon is exactly where trend-persistence convexity
*should* live. And **2018 was a clean weekly USD trend** → weekly trend-following is the natural
2018-positive candidate, now expressible **both directions** (shorts open, PR #273). Hypothesis: a
weekly Donchian breakout has a real trend-SELECTED convex tail and is +2018. Convexity-lens observation
first (forward weekly MFE distribution + drift), not the win-rate lens.

## Method (CALLED canonical loader; observation only)

`Panel.from_pairs` D1 (cached) on 7 USD majors → resampled to **W-FRI** weekly mid OHLC; weekly Wilder
ATR(14) shift1. Signal: weekly Donchian breakout — long if `close > prior-K-week high`, short if
`close < prior-K-week low` (K=8w ≈ 2 months); forward HOLD=8w MFE (in weekly-ATR units), final drift,
vs the unconditional (every-week) base (arc 2000's periodic-base method). CHARACTERIZATION ONLY (gross,
not a gate). Driver: `_disco3_work/arc3014_observe_weekly_trend.py`.

## What happened — FALSIFIED at observation, both lenses

**WEEKLY LONG** (Donchian-8w break, 8w hold):

| | n | drift mean | drift med | mfe | P(mfe≥3) | P(mfe≥5) |
|---|---|---|---|---|---|---|
| unconditional base | 5873 | −0.005 | 0.00 | 1.41 | 0.088 | 0.015 |
| trend-breakout | 619 | **+0.07** | **−0.027** | 1.59 | 0.126 | 0.044 |

- The breakout **does** fatten the tail (P(mfe≥5) 0.015→0.044, ~3×) — convexity is *mildly*
  trend-selected at W1 (unlike H4's "generic", arc 2000). **BUT the median drift is −0.027** → the
  positive *mean* is a thin fat-tail artifact (the median breakout trade loses), the same signature
  that is sub-cost everywhere.
- **Per-pair only 2/7 positive** — carried by USDJPY +1.09 & USDCAD +0.38 (the trend/carry pairs);
  GBPUSD −0.27, AUDUSD −0.32, NZDUSD −0.41, USDCHF −0.38, EURUSD 0.00. A single-pair (USDJPY) artifact,
  not a structural weekly-trend edge.
- **By-year: 2018 drift −0.26 (NEGATIVE)**, 2019 −1.04; 2015 +0.32.

**WEEKLY SHORT:** trend-breakout mean +0.005 / **median −0.126** (worse); per-pair split (EURUSD +0.50
but USDCAD −0.45, USDCHF −0.32); **2018 drift −0.07 (NEGATIVE)**.

## Diagnosis — the weekly USD trend is not a tradeable breakout edge

The "2018 = clean weekly USD trend → trend-following wins" intuition is **false**: 2018's USD strength
came in **mean-reverting bursts** (re-confirming 3012's capitulation finding at the weekly scale), so
weekly Donchian breakouts **whipsawed** (long breakout −0.26, short breakout −0.07 — both lose in 2018).
The mild convexity selection at W1 (fatter tail than H4) is real but does not survive: the **median is
negative and the edge is one carry-pair (USDJPY)**, so after FundedNext cost + SL-first take-the-loss
it is sub-cost — the exact arc-2000 conclusion, now confirmed one timeframe higher. The directional /
trend base is **timeframe-invariant through the weekly horizon** (1002's H4≈D1 extends to W1).

## Verdict: KILL (cheap-kill at observation)

Median-negative drift (both directions), single-pair-carried mean, −2018 both directions → §5f doesn't
bite (a median-negative / single-pair / target-fold-negative cell collapses SL-honest, arc 3003/2000).
No pool/engine/null/council spent.

## Threads / lessons

1. **The directional/trend base is timeframe-invariant THROUGH the weekly horizon** — W1, the last
   untested timeframe (the CTA classic), behaves like H4/D1: mean ≈0, median negative, the convex tail
   is *mildly* trend-selected (fatter than H4) but median-negative and single-pair-carried → sub-cost.
   Closes the timeframe lever entirely (H1/H4/D1/W1 all mapped).
2. **Weekly trend-following does NOT supply the 2018 leg** (−2018 both long and short) — the "clean
   weekly USD trend" is a whipsaw at the breakout level; 2018's USD strength is mean-reverting bursts
   (capitulations), not a tradeable persistent breakout. This is the **EIGHTH** dead route to the 2018
   leg (structure 1014/2009/2011, trend-short 3010, flow 1016, continuation-long 2012, vol-state short
   3012, re-leveling 3013, now weekly-trend 3014, both directions) — and the most decisive, since
   trend-following is the one construction the 2018 intuition most strongly predicted would work.
3. **Convexity IS mildly horizon-dependent** (W1 tail fatter than H4) — a refinement of arc 2000's
   "generic tail": the tail gets *somewhat* trend-selected as the horizon lengthens, but never enough
   to flip the median positive or beat cost. Trend-following's convexity is real but sub-cost at every
   FX-major timeframe.
4. **Meta-signal (three consecutive cheap-kills, 3012/3013/3014):** the in-apparatus directional /
   structural / trend / vol / timeframe frontier is now comprehensively exhausted for the 2018 leg AND
   for a standalone edge. Future chats should weight away from re-grinding directional/trend/vol cuts
   (proven-dead, all timeframes/directions/levels now mapped) and toward the operator-gated tighter-cost
   execution regime or a genuinely non-price-direction construction — the only frontier the apparatus
   has not closed.

## Tooling

No new BUILT tool — weekly resample + Donchian convexity observation is a one-off scratch observer
(like arc 2000's MFE-distribution probe at H4; not reused enough to register). The BUILT
`DonchianBreakoutLongSignal` (arc 2000) is H4/D1-oriented; the weekly resample is the novel piece and is
a throwaway characterization, not a gate component.

## FLAGS (code not merged)

None. No canonical-core change. Carries the standing `A1Config.time_exit_bars`-unwired flag + FLAG-1
(the 2018 leg). Driver `_disco3_work/arc3014_observe_weekly_trend.py` (reproducible from this doc).
