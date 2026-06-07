# arc 2085 — Positive-skew trend CONTINUATION on the WEEKLY timeframe (the final timeframe-axis closer)

> **Arc id:** 2085 · **Chat:** 2000–2999 (continuous) · **Date:** 2026-06-07
> **Type:** EDGE-HUNT / definitive closer on the operator-redirected frontier (positive-skew /
> continuation) — closes the **timeframe axis** completely (H4 done 1074/2081/2082/2083; D1 done 2084;
> W1 = this arc, the last finer-grained cell). **Disposition:** KILL.
> **Scored solely by** `MultiPairBacktester` (FundedNext costs ON, SL-first / take-the-loss). Canonical
> apparatus CALLED; entry = `TrendContinuationBreakoutSignal` (arc 1074) on W1; same guard + null as 2084.

## Why this arc (close the axis, don't leave a residual)

arc 2084 took the positive-skew continuation thread from H4 to D1 and found a clean two-part result:
**timeframe is a real geometry lever** (D1 bleeds ~8× less than H4; the Turtle-long even beats the null +
is +2015) **but NOT a sign lever** (median trade still −1R; mean never crosses zero → KILL). It named W1 as
the only finer-grained cell left and predicted it closes the same way with worse fold resolution. Rather
than leave "W1 untested" as a forever-residual a future chat could re-raise, this arc runs it — a clean
closure of "is there ANY timeframe where continuation works on liquid-FX OHLC" is high corpus value (cf.
arc 3004 "stop removed" closing that axis definitively).

## Hypothesis / falsifier

Same as 2084, on W1: a weekly-trend continuation book is mean-positive and survives the tail guard.
*Falsifier (and the prior):* if the median trade is still ≈ −1R on W1 (stop-wall timeframe-invariant) and
the mean stays ≤ 0, the timeframe axis is fully closed — continuation has no exploitable edge at any
sampling frequency on liquid FX.

## Method

Driver `_disco2_work/arc2085_w1_trend_skew.py` — arc 2084's harness, **TF=W1 + weekly-appropriate params**
(`sma_fast=10, sma_slow=40` ≈ 50/200-day; Donchian {10, 26} weeks ≈ 50/130-day). 28 pairs, both
directions, trailing exits × SL{1.5,2,2.5}, §5f nested selection, the pre-registered tail-removed guard
(G1 mean / G2 +2R-cap+top-5%+drop-top-K / G3 median-per-fold), same-side random null (warmup 45 weeks).
IS 2011–2020; OOS spent only if a cell passes G1+G2+G3 + beats null.

## Results — all 4 cells KILL at IS; OOS preserved

| cell | fires | n_trades | mean ROI/yr | folds+ | mean R | median R | +2R-cap | top5%-rm | vs null | 2015 / 2018 |
|---|---|---|---|---|---|---|---|---|---|---|
| long Donchian10 | 525 | 112 | −3.13% | 2/8 | −0.4478 | −0.938 | −3.52% | −3.59% | +0.55pp (beats) | −2.67 / −6.60 |
| short Donchian10 | 529 | 91 | **−0.52%** | 4/8 | −0.0917 | −0.969 | −1.44% | −1.95% | **+1.54pp (beats)** | **+7.40** / −3.88 |
| long Donchian26 | 455 | 87 | −2.20% | 3/8 | −0.4045 | −0.983 | −2.38% | −2.56% | −2.38pp (LOSES) | +0.47 / −6.57 |
| short Donchian26 | 431 | 76 | **−0.40%** | 2/8 | −0.0838 | −0.977 | −1.65% | −2.13% | **+1.17pp (beats)** | **+9.71** / −5.89 |

Pool counts (76–112) clear the ≥50 floor — this is a genuine **mean-negative** KILL, not a pool-floor
artifact. Every cell fails G1 (mean ROI and per-trade R both negative), G2 (tail-removal worsens the mean
everywhere), and G3 (≤ majority median-positive). 3 of 4 cells beat the null and the shorts are strongly
+2015 (+7.4% / +9.7%) — the same long-vol-in-the-reversion-death-year signature 2084 found — but all stay
net-negative ⇒ KILL (§11, beats-null-but-net-negative).

## Diagnosis — the −1R stop-wall is TIMEFRAME-INVARIANT; the bleed shrinks but the sign is fixed at every TF

Stacking the three timeframes gives the unifying proof of the whole positive-skew continuation closure:

| timeframe | best-cell mean ROI/yr | per-trade mean R range | **median trade R** | tail-removed | crosses 0? |
|---|---|---|---|---|---|
| **H4** (1074/2081) | −6% to −22% | ≈ −0.13R | ≈ **−0.81 to −0.9** | worsens | no |
| **D1** (2084) | −0.88% (Turtle-long) | −0.04 to −0.17R | ≈ **−0.84 to −0.94** | worsens | no |
| **W1** (this arc) | −0.40% (slow-short) | −0.08 to −0.45R | ≈ **−0.94 to −0.98** | worsens | no |

1. **The median trade R is ≈ −0.9 at EVERY timeframe** — the typical post-breakout path takes the −1R stop
   before +1R whether you sample at 4 hours, 1 day, or 1 week. This is the decisive datum: the stop-wall is
   a property of the price's near-martingale structure, NOT of the sampling frequency, so no timeframe can
   break it. The breakout is adverse-first at every clock speed.
2. **The bleed shrinks as the clock slows** (geometry lever, 2084's finding confirmed and extended): the
   best-cell mean improves monotonically H4 → D1 → W1 (−6% → −0.88% → −0.40%/yr) because a wider stop
   relative to the trend cuts the per-loss size. But it **asymptotes from below — it never crosses zero.**
   Slowing the clock is a geometry lever, not a sign lever, at every step.
3. **The "almost" cells repeat across D1 and W1** (Turtle-long-D55 +2015/beats-null; W1 shorts
   +2015/beats-null) — a genuine but sub-zero long-vol-in-2015 signature. It confirms the *structural*
   intuition (trend books are long-vol, positive in the reversion book's death year) while proving the
   *magnitude* is insufficient: net-negative after take-the-loss + FundedNext costs ⇒ not a portfolio
   component (§11).

Conservation law, fully resolved on the timeframe axis: lowering sampling frequency moves the loss geometry
without flipping the sign, because the underlying liquid-FX price is a near-martingale at every interval.

## Council — NOT convened

§5d cheap-kill: every cell mean-negative, fails the objective pre-registered guard verbatim; the beats-null
cells are net-negative = KILL (§11). A confirmatory closer of an axis already established by 2084 needs no
adversarial review ("ritual not rigor", arc-1002). OOS NEVER touched.

## Verdict: KILL

No new component. **The timeframe axis of positive-skew continuation is now closed completely** — H4
(1074/2081/2082/2083), D1 (2084), W1 (this arc), all mean-negative with a timeframe-invariant −1R median
stop-wall and tail-removal that worsens the mean. Combined with the entry-geometry closures (breakout /
vol-expansion / pullback-resume / shock), positive-skew continuation is comprehensively dead on liquid-FX
OHLC across **entry geometry × timeframe × direction × universe × exit menu.** Deployable count = 0;
components UNCHANGED (4 PORTFOLIO: gap 1006 / me_long 1011 / fbr 1013 / me_short 1019).

## Lessons (candidate for LESSONS.md)

1. **The −1R median stop-wall is TIMEFRAME-INVARIANT (the unifying proof of the continuation closure).**
   Across H4 / D1 / W1 the *median* breakout-continuation trade is ≈ −0.9R — it takes the stop before +1R
   at every sampling frequency. The wall is a property of the price's near-martingale structure, not the
   clock, so no timeframe breaks it. This is *why* continuation is dead, stated mechanistically.
2. **Slowing the clock is a geometry lever, not a sign lever — at every step.** Best-cell mean ROI improves
   monotonically H4 −6% → D1 −0.88% → W1 −0.40%/yr (wider stop ⇒ smaller per-loss) but asymptotes from
   below and never crosses zero. Timeframe cannot manufacture positive expectancy where the median trade is
   a −1R stop.
3. **The +2015/beats-null "almost" cell recurs at D1 and W1 but stays sub-zero.** Trend books are genuinely
   long-vol (positive in the reversion book's 2015 death year, beat the being-in-vol null) — confirming the
   structural complement intuition — but net-negative after take-the-loss + costs ⇒ KILL, not PORTFOLIO.
   The documented positive-skew trend edge lives in a less-efficient instrument universe (NEEDS_ENABLEMENT
   #1), not FX-OHLC at any timeframe.

## Threads / handoff

- **Closed (definitively):** the timeframe axis of positive-skew continuation — H4/D1/W1 all mean-negative,
  median trade −1R timeframe-invariant, tail-removal worsens it. With the entry-geometry closures, the
  operator's ONE open in-charter thread (positive-skew continuation, LESSONS 2026-06-06) is comprehensively
  closed across every axis the apparatus can vary.
- **Out-of-band (the only live levers):** operator path-A gate-governance call on the 4-component book
  (mean-positive, fails strict AFP only at within-noise 2015/2018), and/or a charter unlock —
  `NEEDS_ENABLEMENT.md` #1 cross-asset trend is the SAME positive-skew shape on a less-efficient universe
  where it is documented-fundable; the in-charter FX version is what the H4/D1/W1 arcs just closed.

## Flags / Tooling

No canonical-core change (no FLAG). Carries the standing arc-3017 `risk_pct` FLAG (LINEAR regime, risk
0.005). **No new BUILT tool** — reused `TrendContinuationBreakoutSignal` (arc 1074, on W1 via `primary_tf`),
`nested_exit_selection` (2040), `build_null_signal_evaluation` (1000/2013), arc-1074/2081 tail-removal
arithmetic inline. Driver `_disco2_work/arc2085_w1_trend_skew.py` (reproducible:
`PYTHONPATH=. py discovery/_disco2_work/arc2085_w1_trend_skew.py`). Data: 28 pairs, W1, `5ers_eet`,
`histdata_root=C:\Users\panap\histdata_backup`, `cache_root=C:/Users/panap/Documents/Forex-Backtester/data/cache`.
