# Arc 2000 — Trend-Following Long via Full-Size Convexity Harvest

> **Arc id:** 2000 · **Chat:** 2000–2999 (continuous, first arc) · **Date:** 2026-06-04
> **Final verdict:** **FAIL (cheap-kill at triage).** The fat right tail in FX-major longs is NOT
> created by the trend entry (a random/periodic long has the same tail), and no full-size trailing
> harvest exit — the one lever no prior arc tested — produces all-folds-positive expectancy after
> costs. Same real-but-sub-cost signature as arcs 0/1000.
> **Lever tested:** PAYOFF STRUCTURE / convexity — is the prior arcs' win-rate screen blind to a
> fat-tailed trend edge a full-size trailing exit could harvest?

Scored solely by `MultiPairBacktester` (FundedNext costs ON, SL-first). Canonical apparatus called
(`build_arc_pool`, `ArcFoldRunner`→`A1Architecture`→`MultiPairBacktester`, `build_v3_folds`); the trend
entry is an experiment tool (`discovery/tools/trend_entry_signals.py`, registered).

## (a) Log read — FRESH EYES

Pulled main (`aad9e8b`). Honest-era corpus = arc 0 (pullback long H4, FAIL full-WFO), arc 1000 (XS
momentum H4, FAIL WFO+council KILL), arc 1001 (contraction-breakout H4, FAIL triage), arc 1002
(trend-following D1, FAIL triage). LESSONS.md empty. No STOP. **Accumulated finding:** four
independent directional longs fail identically — the honest +1R-before-SL *capture* sits at
~0.487–0.494 everywhere (H4 and D1, every trigger), so the binding constraint is the cost/SL-first
hurdle against a ~coin-flip **win-rate** base. Logged steer: change a fundamental lever (crosses,
portfolio/selection, or non-directional).

**The fresh insight (why this arc is not "arc 1002 again"):** every prior arc *screened* on
+1R-before-SL capture — a **win-rate** statistic. A win-rate lens is structurally blind to a
low-win-rate, **fat-tailed / convex** payoff (the canonical time-series-momentum profile, where the
edge is the right tail, not the hit rate). And every engine triage used `sl_partial_close_1r_runner_trail`,
which **banks 50% at +1R and caps the right tail at half size**. So the prior failures might be a
measurement artifact, not a closed family. The untested question: does a trend entry have a fat,
harvestable right tail, and does a **full-size, tail-preserving** trailing exit (`sl_plus_trailing_atr`:
−1R floor, uncapped upside, give back 1R from peak) bank enough of it to clear costs?

## (b) Idea + observation (convexity lens, not win-rate lens)

**Idea:** Donchian breakout long (fresh N-bar-high break = canonical trend / time-series-momentum
entry) + full-size trailing harvest, measured on the **MFE distribution / expectancy**, not win-rate.
Observe the canonical pool's full MFE distribution (H4 majors, IS 2010–2020, hold 360 bars ≈3mo,
SL=2·ATR) for Donchian breakouts vs a periodic (time-random) long base — does the breakout fatten the
harvestable right tail?

| signal | n | stop | cap+1R | meanFinalR | meanMFE | P(mfe≥2) | P(mfe≥3) | P(mfe≥5) | P(mfe≥8) |
|---|---|---|---|---|---|---|---|---|---|
| **periodic base (p30)** | 4622 | .884 | .494 | −0.068 | 2.46 | .318 | .241 | **.147** | **.077** |
| donchian_20 | 4089 | .888 | .494 | −0.034 | 2.62 | .324 | .240 | .153 | .085 |
| donchian_55 | 2449 | .889 | .486 | −0.021 | 2.61 | .310 | .236 | .154 | .086 |
| donchian_120 | 1617 | .884 | .489 | +0.016 | 2.69 | .315 | .246 | **.165** | **.088** |
| donchian_120+sma200 | 1610 | .884 | .489 | +0.021 | 2.69 | .315 | .245 | .165 | .088 |

**The fat tail is real but the trend entry does NOT create it.** A periodic/random long already has
P(mfe≥5R)=.147, P(mfe≥8R)=.077, meanMFE 2.46R. Donchian-120 lifts these only to .165/.088/2.69 — a
trivial gain. The right tail is a **generic property of being long a vol-clustering FX major at any
time**, not something the breakout selects for. The convexity-selection hypothesis is **falsified at
observation** (the same way arc 1002's "D1 trends better" was). cap+1R ≈ .49 everywhere too — the
win-rate lens and the convexity lens *agree*: the trend entry adds ~no separable edge on either axis.

## (c)+(d) Characterize + cheap-kill triage — the untested full-size trailing harvest

Pool floor PASS (donchian_120 = 1617 IS trades). One lever remained genuinely untested: a **full-size**
trailing exit (prior arcs used the 50%-partial). Triaged donchian_120 under both full-size trails on 3
representative IS folds (OOS 2013/2016/2019; `build_v3_folds` ids 4/7/10), honest engine + FundedNext
costs, `trail_enabled=False` so the exit policy is the sole exit manager. Ran the periodic base under
the same harvest exit as an inline null.

| version | 2013 | 2016 | 2019 | worst | mean | all-folds-pos |
|---|---|---|---|---|---|---|
| donchian_120 + `sl_plus_trailing_atr` (thesis) | −3.13% / dd10.2 / n97 | −11.39% / dd12.1 / n91 | −13.97% / dd15.8 / n76 | −13.97% | −9.50% | **N** (3/3 neg) |
| donchian_120 + `sl_plus_trailing_swing` (wide) | +20.81% / dd9.7 / n23 | −1.06% / dd11.4 / n39 | −19.86% / dd20.0 / n49 | −19.86% | −0.04% | **N** (2/3 neg) |
| donchian_120 + `sl_partial_close_1r_runner_trail` (prior-arc exit) | −1.18% / dd7.9 / n151 | −9.30% / dd10.1 / n131 | −10.94% / dd12.8 / n110 | −10.94% | −7.14% | **N** (3/3 neg) |
| **periodic null** + `sl_plus_trailing_atr` | −7.60% / dd12.7 / n152 | −9.13% / dd10.9 / n143 | −16.26% / dd19.1 / n147 | −16.26% | −11.00% | **N** (3/3 neg) |

**KILL at the cheap stage (protocol §5d).** Reads:
1. **The full-size trail does not rescue it.** `sl_plus_trailing_atr` is uniformly negative (mean
   −9.50%) and actually *worse* than the 50%-partial (−7.14%) — the fixed 1R give-back from peak on
   every winner, plus keeping full size through the losers' whipsaws, costs more than the partial
   banks. The thesis ("full-size harvest beats the tail-capping partial") is false here.
2. **The wide swing trail is regime-luck, not edge.** `sl_plus_trailing_swing` posts one big fold
   (+20.81%, but a thin n=23 — it caught the 2013 trends) and a −19.86% / DD-20% blowup (2019); mean
   −0.04%, 2/3 folds negative. This is the textbook FX-majors-only trend-following signature: a few
   sporadic, USD-factor-correlated trends carry rare big folds, the rest bleeds in chop, and it nets
   ~zero before the variance/blowup even fails the risk profile. NOT all-folds-positive.
3. **Trend entry ≈ random under the harvest exit.** donchian_120 (−9.50%) barely beats the periodic
   null (−11.00%) under the identical `sl_plus_trailing_atr` — a ~1.5pp lift, the SAME real-but-sub-cost
   signature as arcs 0/1000. The edge that exists is far below the cost/SL-first hurdle.

## Council — NOT convened

No worthwhile-ceiling / diagnosis fork. The convexity-selection hypothesis is falsified at observation
(trend entry doesn't fatten the tail vs random), and all three apparatus exits — including the two
genuinely-untested full-size trails — are cleanly sub-cost / not all-folds-positive. Same family and
lesson as the already-councled arc-1000 pattern; re-convening would re-derive it ("ritual, not rigor",
per the arc-1002 precedent). Cheap-killed on the apparatus.

## Final verdict — FAIL (cheap-kill)

The **trend-following-long-via-convexity-harvest family is not deployable on H4 majors.** The fat right
tail is generic (not trend-selected), and no full-size trailing harvest — the one lever the prior arcs
never tested — clears FundedNext costs on an all-folds-positive basis. The payoff-structure lever is now
exhausted alongside entry-construction and timeframe.

## Lessons (candidate for LESSONS.md)

1. **The fat right tail in FX-major longs is NOT created by a trend entry.** A periodic/random long
   already has P(mfe≥5R)≈.15, P(mfe≥8R)≈.08, meanMFE ≈2.5R; a Donchian-120 breakout lifts these only
   trivially (.165/.088/2.69). The convexity (fat tail) is a generic property of being long a
   vol-clustering FX major at any time — not a selectable edge. The win-rate lens (+1R capture ≈.49)
   and the convexity lens (MFE tail) **agree**: the trend entry adds ~nothing on either axis.
2. **No full-size trailing harvest rescues a ~coin-flip long.** `sl_plus_trailing_atr` (full size, +1R
   activate, 1R-from-peak trail) is uniformly negative and *worse* than the 50%-partial (the per-winner
   1R give-back + full-size whipsaw losses dominate). `sl_plus_trailing_swing` is high-variance
   regime-luck (+20.81% thin fold vs −19.86%/DD-20% blowup, mean −0.04%, 2/3 neg) — the FX-majors-only
   trend-following signature, NOT all-folds edge.
3. **FIFTH independent long family, same real-but-sub-cost signature** (arc 0 pullback, 1000 XS-mom,
   1001 contraction-brk, 1002 D1 trend-following, **2000 convexity-harvest**). The binding constraint —
   cost/SL-first hurdle against a ~coin-flip base — is now shown invariant to **PAYOFF STRUCTURE** (both
   win-rate exits and fat-tail/convex exits fail), on top of entry construction and timeframe. The
   convexity lever joins the closed list; the prior arcs' win-rate screen was NOT hiding a convexity
   edge.
4. **Strong steer reinforced:** the majors-directional-long search space is comprehensively dry across
   entry, timeframe, AND payoff structure. Change the UNIVERSE (less-efficient crosses — the corpus has
   ~20 crosses cached) or go non-directional/portfolio. A swing-trail's regime-luck spread hints that if
   trend-following is ever viable it needs MANY decorrelated instruments (the diversification FX-majors-
   only lacks) — which ties the trend thread to the crosses/portfolio thread, not to majors.

## Threads

- **Closed:** trend-following / time-series-momentum LONGS on H4 majors via full-size trailing harvest
  (convexity lever exhausted); Donchian breakout entry (lookback 20/55/120, ±SMA200) adds no
  tail-selection edge; the win-rate-lens-was-blind hypothesis is itself falsified (both lenses agree).
- **Open (carried + sharpened):** (1) **instrument universe** — do less-efficient CROSSES (JPY /
  commodity / EM-ish; ~20 cached) lift the directional base or the harvestable tail above majors? The
  single most-supported untested lever now. (2) portfolio/selection of decorrelated sub-cost signals
  (XS-as-universe). (3) non-directional construction. (4) trend-following may only work *diversified*
  across many instruments — a portfolio claim, not a single-pair-on-majors claim.

## Flags / Reproduction

No canonical-core change (no flags). Experiment tool committed + registered:
`discovery/tools/trend_entry_signals.py` (`DonchianBreakoutLongSignal`, `PeriodicLongSignal`). Drivers
in scratch (`_disco2000_work/arc2000_observe.py`, `arc2000_triage.py`); run `PYTHONPATH=. py
_disco2000_work/<script>.py`. Data: `histdata_root=C:\Users\panap\histdata_backup`, `cache_root=...\data\cache`,
**tf H4** 5ers_eet, 8 majors (EURUSD GBPUSD USDJPY USDCHF AUDUSD USDCAD NZDUSD EURGBP). Pool: SL=2·ATR,
hold 360 bars, IS 2010-2020. Engine: `A1Architecture`+`ArcFoldRunner`, FundedNext costs at
`build_fold_stats_from_run`, `trail_enabled=False`, triage folds `build_v3_folds` ids 4/7/10.
