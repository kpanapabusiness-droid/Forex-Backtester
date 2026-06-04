# Arc 1003 — Cross Trend-Momentum Long (Instrument-Universe Lever)

> **Arc id:** 1003 · **Chat:** 1000–1999 (continuous) · **Date:** 2026-06-04
> **Final verdict:** **FAIL (cheap-kill at triage).** Crosses trend (positive gross drift) but the wider
> cross spreads sink it — triage worst −9.75%, mean −5.49%. The directional base is still coin-flip.
> **Lever tested:** INSTRUMENT UNIVERSE — are less-efficient crosses a better directional-long base than majors?

Scored solely by `MultiPairBacktester` (FundedNext costs ON, real cross bid/ask, SL-first). Canonical
apparatus called; the cross signal is an experiment tool (scratch).

## (a) Log read — FRESH EYES

Pulled main. Corpus = arc 0 + 1000 + 1001 (H4-major directional longs, FAIL) + arc 1002 (D1 directional long,
FAIL). FOUR directional-long attempts, all coin-flip/sub-cost across entry construction AND timeframe. Logged
steer: change a more fundamental lever — instrument universe, portfolio/selection, or non-directional. No STOP.
(Noted: a portfolio of *negative-expectancy* sub-cost signals can't yield positive expectancy — decorrelation
cuts variance, not the sign of the mean — so seek a positive base first → the universe lever.)

## (b) Idea + observation

**Lever = INSTRUMENT UNIVERSE.** Majors are the most-arbitraged/efficient FX instruments. Crosses (JPY +
commodity-currency crosses) carry rate differentials and are reputed to trend more persistently. **Test:**
does the directional-long base clear coin-flip on crosses? Observe H4 honest +1R-before-SL LONG capture on
8 trending crosses (EURJPY GBPJPY AUDJPY CADJPY CHFJPY EURAUD GBPAUD AUDNZD), IS 2010–2020, SL=2·ATR, hold 120.

| trigger | n | capture | lift vs cross base | pairs>0.50 |
|---|---|---|---|---|
| **unconditional (cross base)** | 138,750 | **0.4712** | — | 0/8 |
| uptrend (close>SMA50) | 68,683 | 0.4813 | +0.0101 | 2/8 |
| **momentum breakout (20-bar high)** | 8,207 | **0.4982** | **+0.0270** | 4/8 |
| pullback in uptrend | 9,470 | 0.4738 | +0.0026 | 2/8 |

per-pair base: EURJPY 0.470, GBPJPY 0.475, AUDJPY 0.493, CADJPY 0.478, CHFJPY 0.465, EURAUD 0.469, GBPAUD
0.475, AUDNZD 0.445.

**Crosses are a WORSE long base than majors** (0.4712 vs 0.4877) — every cross < 0.50 unconditionally. BUT
the momentum-breakout lift on crosses (**+2.7pp**, to 0.4982) is the **largest conditioning effect in the
whole run** and mechanistically sensible (crosses trend after breaks). Still coin-flip in absolute terms, and
gross of the crosses' WIDER spreads (harder cost hurdle). Because it was the strongest lift seen, it earned
the apparatus check.

## (c)+(d) Characterize + cheap kill — cross trend-momentum long

Pooled the **cross Donchian-20 breakout-in-uptrend long** (`close > prior-20-bar-high AND close > SMA50`,
refractory 6) — the reasoned best version. `pool_sha256 ba2f42ac7b0117ab…`.
- **4,078 IS trades** (475–545/pair). Honest +1R-before-SL **0.4809**. **Mean final_r +0.1021 (positive,
  gross)** — crosses DO trend; the runner harvests it. Pool floor PASS.
- **3-fold honest triage** (A1, `sl_partial_close_1r_runner_trail`, costs ON, real cross spreads):

  | fold | OOS 2013 | OOS 2016 | OOS 2019 |
  |---|---|---|---|
  | ROI | +1.43% | −8.15% | −9.75% |
  | DD | 8.00% | 12.39% | 11.73% |
  | n | 233 | 217 | 229 |

  **TRIAGE worst −9.75%, mean −5.49% — 2/3 deeply negative.** → KILL at the cheap stage (protocol §5d).

The contrast is the lesson: gross drift is POSITIVE (+0.10R/trade; crosses trend) yet the net is deeply
negative — the **wider cross spreads (1.5× at the gate) eat the thin directional edge.** On crosses the
problem isn't "no edge"; it's "edge < cost."

## Council — NOT convened

No worthwhile-ceiling/diagnosis fork that differs from the already-councled pattern: the cross directional
base is coin-flip and the trend-momentum triage is mostly deeply-negative (cost > edge). Cheap-killed on the
apparatus; re-convening would re-derive the logged lesson.

## Final verdict — FAIL (cheap-kill)

The **cross trend-momentum long is not deployable**: crosses have a real but THIN positive gross drift on
momentum-breakouts that the wider cross spreads more than consume (triage worst −9.75%, mean −5.49%). The
instrument-universe lever does NOT escape the binding constraint.

## Lessons (candidate for LESSONS.md)

1. **Crosses are NOT a better directional-long universe than majors.** Unconditional cross long capture
   (0.4712) is WORSE than majors (0.4877); the momentum-breakout lift (+2.7pp, the run's largest) only reaches
   coin-flip (0.4982) gross.
2. **The binding constraint is now precisely characterized: EDGE < COST.** On crosses the gross directional
   drift is POSITIVE (mean final_r +0.10R on trend-breakouts — crosses trend) but the wider cross spreads
   (1.5× at the gate) consume it → net deeply-negative. On majors the gross edge is ~zero (coin-flip). Both
   net sub-cost. The hurdle isn't finding *a* directional edge; it's finding one LARGE enough to clear the
   spread/commission/slippage + SL-first take-the-loss.
3. **FIFTH directional-long FAIL** across entry construction (pullback/momentum/breakout/contraction),
   timeframe (H4/D1), AND universe (majors/crosses). The price-direction-long approach on liquid FX is
   comprehensively dry vs FundedNext costs. **Future arcs must change the MECHANISM, not just the parameters
   of a directional bet:** a non-price-direction edge (calendar/flow, e.g. turn-of-month rebalancing; not yet
   tested — arc 1000 covered hour/day-of-week but NOT day-of-month), or a construction whose gross edge is
   large enough to clear cost.

## Threads

- **Closed:** cross directional longs (trend/momentum/pullback); the instrument-universe lever for direction.
- **Open / next:** non-price-direction mechanisms — turn-of-month / month-end rebalancing FLOW (documented
  institutional flow, structurally different from price-structure); the broader open question of whether ANY
  long-only edge on liquid FX clears FundedNext cost (5 directional families say price-direction does not).

## Flags / Reproduction

No canonical-core change (no flags). Signal + drivers scratch (`_disco_work/arc1003_observe.py`,
`arc1003_kill.py`); `PYTHONPATH=. py _disco_work/<script>.py`. Data: `histdata_root=C:\Users\panap\histdata_backup`,
H4 5ers_eet, 8 crosses (EURJPY GBPJPY AUDJPY CADJPY CHFJPY EURAUD GBPAUD AUDNZD). `pool_sha256 ba2f42ac7b0117ab…`.
Engine: `A1Architecture`+`ArcFoldRunner`, FundedNext costs at `build_fold_stats_from_run`, triage folds
`build_v3_folds` ids 4/7/10, hold 120.
