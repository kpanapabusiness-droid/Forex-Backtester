# Arc 1001 — Volatility-Contraction Breakout Long

> **Arc id:** 1001 · **Chat:** 1000–1999 (continuous) · **Date:** 2026-06-04
> **Final verdict:** **FAIL (cheap-kill at triage)** — idea falsified at observation (coil adds zero
> directional lift); 3-fold honest triage all-negative (worst −9.41%, mean −5.33%). Killed before full WFO
> per protocol §5d (deeply-negative triage → KILL).
> **Idea:** after a low-vol coil, an upside break should follow through better than a generic breakout —
> exploiting volatility-clustering (the one regularity the prior direction-prediction arcs ignored).

Scored solely by `MultiPairBacktester` (FundedNext costs ON, SL-first). Engine/measurement called, never
re-rolled; the signal is an experiment tool (scratch).

## (a) Log read — FRESH EYES

Pulled main. Honest-era corpus = arc 0 (pullback long, FAIL) + arc 1000 (cross-sectional momentum long,
FAIL). Both share the SAME real-but-sub-cost signature: a ~coin-flip directional base on H4 majors at 1:1
(returns autocorr ≈ 0); the binding constraint is the cost/SL-first hurdle, not the specific entry. Closed:
single-pair entry-time price structure, cross-sectional momentum (trade-level), temporal/regime conditioning
of naive longs. LESSONS.md empty. No STOP. Fresh eyes (no pre-reset list).

## (b) Idea + observation

Both prior arcs bet on PRICE DIRECTION (continuation/reversion) and failed because direction is ~coin-flip.
The ignored regularity: VOLATILITY clusters and is autocorrelated. **Idea:** volatility-contraction breakout
long — after a low-vol coil (ATR percentile-in-trailing-100 < 0.33), enter long when the close breaks above
the prior 10-bar high. **Because:** vol-clustering makes "a move is imminent" predictable (the part direction-
prediction can't get); an upside squeeze-break triggers stop-cascades + breakout flow, giving follow-through
that *generic* breakouts (arc 0: fade) lack — the coil is the differentiator. Long-only.

**Observation** (honest +1R-before-SL capture, in-tree take-the-loss label, SL=2·ATR, hold 120, 8 majors,
IS 2010–2020, 138,816 hypothetical longs; unconditional capture **0.4877**):

| trigger | n | capture | lift | pair_min..max | n>base |
|---|---|---|---|---|---|
| generic upside breakout (10-bar high) | 11,128 | 0.4863 | −0.0013 | 0.465–0.501 | 4/8 |
| coil only (low ATR pct) | 54,878 | 0.4895 | +0.0018 | 0.443–0.524 | 5/8 |
| **contraction breakout (coil & break)** | 5,271 | **0.4880** | **+0.0003** | 0.444–0.516 | 6/8 |
| oversold reversion (z<−2) | 12,618 | 0.4844 | −0.0033 | 0.466–0.505 | 4/8 |

**The hypothesis is FALSIFIED at observation:** contraction-breakout capture (0.4880) ≈ generic-breakout
(0.4863) ≈ unconditional (0.4877). The coil adds **zero** directional lift. The generic-breakout fade
reconfirms arc 0. Oversold reversion is also dry. (Mechanism note: the coil predicts vol EXPANSION, but the
upside break is still ~coin-flip on DIRECTION — exactly the binding constraint the corpus already identified.)

## (c)+(d) Characterize + cheap kill

`build_arc_pool` (signal `ContractionBreakoutLong(brk_lb=10, coil_pct=0.33, refractory=6)`), 8 majors,
H4 5ers_eet, IS 2010–2020, SL=2·ATR, hold 120. `pool_sha256 7d88dc5b2b9ef9ea…`.
- **3,186 IS trades** (379–414/pair). Honest +1R-before-SL **0.4862** (matches observation — sanity). Mean
  final_r −0.0542. **Pool floor PASS.**
- **3-fold honest triage** (A1, `sl_partial_close_1r_runner_trail`, costs ON):

  | fold | OOS 2013 | OOS 2016 | OOS 2019 |
  |---|---|---|---|
  | ROI | −9.41% | −1.15% | −5.43% |
  | DD | 12.30% | 8.33% | 7.59% |

  **TRIAGE worst −9.41%, mean −5.33% — all three folds negative.** Deeply/consistently negative, with NO
  lucky trending fold (contrast arc 1000's +11.98% 2013). → **KILL at the cheap stage** (protocol §5d).

Oracle ceiling / clustering not run: with zero observational lift + a falsified mechanism + an all-negative
triage, there is no worthwhile ceiling to chase (the arc-0/1000 lesson: a high oracle ceiling on a ~0.49
capture is hindsight, not a reachable edge).

## Council — NOT convened

The §5e heavy council is for ideas with a worthwhile ceiling that PROCEED to diagnosis (a genuine
entry/exit/selection fork). Here the mechanism is falsified at observation (the coil adds nothing) and the
triage is cleanly all-negative — there is no fork to evaluate. Re-convening would re-derive the already-
councled arc-1000 finding (a ~coin-flip directional long is sub-cost) — the council's own "ritual, not rigor"
principle. Not convened; cheap-killed on the apparatus instead.

## Final verdict — FAIL (cheap-kill)

The **volatility-contraction breakout long family is not deployable**: the coil adds no directional lift
(capture 0.4880 = unconditional), and the honest 3-fold triage is all-negative (worst −9.41%, mean −5.33%).
Volatility-clustering makes the *timing* of expansion somewhat predictable, but the *direction* of the
upside break remains ~coin-flip — so the long-only contraction-breakout inherits the same sub-cost base as
the prior two arcs.

## Lessons (candidate for LESSONS.md)

1. **A volatility coil does NOT confer a directional long edge on H4-major breakouts.** Contraction-breakout
   capture equals generic-breakout equals unconditional (~0.487). The coil predicts EXPANSION, not DIRECTION;
   a long-only break of a coil is still a directional coin-flip. (Generic breakouts fade — arc 0 reconfirmed.)
2. **THIRD independent long family with the same ~coin-flip-base / sub-cost outcome** (arc 0 pullback, arc
   1000 XS momentum, arc 1001 contraction-breakout). Strong accumulating evidence: on H4 majors, the limiting
   factor for a LONG is the cost/SL-first hurdle against a ~0.49 directional base — not the entry construction.
   Direction-prediction entries (continuation, reversion, momentum, breakout, squeeze) are looking
   systematically dry at this TF/RR. Future arcs should consider a structurally different LEVER — payoff/RR
   asymmetry, portfolio/selection (the open XS-as-universe thread), a different timeframe, or a non-directional
   construction — rather than another H4 directional-long entry trigger.

## Threads

- **Closed:** volatility-contraction breakout long; generic breakout (fade); oversold-reversion long (dry).
- **Open (carried):** XS-rank-as-universe/portfolio-selector (arc 1000); the broader question of whether ANY
  H4-major directional long can clear cost (three families now say no) — argues for a different lever next.

## Flags / Reproduction

No canonical-core change (no flags). Signal + drivers scratch (`_disco_work/arc1001_signal.py`,
`arc1001_observe.py`, `arc1001_kill.py`), reproducible: `PYTHONPATH=. py _disco_work/<script>.py`.
Data: `histdata_root=C:\Users\panap\histdata_backup`, H4 5ers_eet, pairs EURUSD GBPUSD USDJPY AUDUSD NZDUSD
USDCAD USDCHF EURGBP. `pool_sha256 7d88dc5b2b9ef9ea…`. Engine: `A1Architecture` + `ArcFoldRunner`, FundedNext
costs at `build_fold_stats_from_run`, triage folds `build_v3_folds` ids 4/7/10.
