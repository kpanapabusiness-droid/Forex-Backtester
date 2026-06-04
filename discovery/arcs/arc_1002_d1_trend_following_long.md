# Arc 1002 — D1 Timeframe Directional Long (Daily Trend-Following)

> **Arc id:** 1002 · **Chat:** 1000–1999 (continuous) · **Date:** 2026-06-04
> **Final verdict:** **FAIL (cheap-kill at triage).** The D1 directional-long base is ~coin-flip
> (TF-invariant with H4); textbook daily trend-following (Donchian-20 breakout in uptrend) is sub-cost
> (3-fold triage all-negative, worst −5.72%, mean −4.13%).
> **Lever tested:** TIMEFRAME — is the H4-major coin-flip directional base specific to H4, or general?

Scored solely by `MultiPairBacktester` (FundedNext costs ON, SL-first). Canonical apparatus called; the
D1 signal is an experiment tool (scratch).

## (a) Log read — FRESH EYES

Pulled main. Honest-era corpus = arc 0 (pullback long, FAIL) + arc 1000 (XS momentum long, FAIL) + arc 1001
(contraction-breakout long, FAIL). THREE independent H4-major directional longs, all the SAME real-but-sub-
cost signature → the binding constraint is the cost/SL-first hurdle against a ~0.49 directional base, NOT the
entry construction. Logged implication: try a structurally different LEVER. LESSONS.md empty. No STOP.

## (b) Idea + observation

**Lever = TIMEFRAME.** All prior arcs were H4. Daily FX trends are classically more persistent (lower
noise-to-signal); daily trend-following is the textbook systematic strategy. **Test:** is the coin-flip
directional base H4-specific, or does it hold at D1? Observe D1 honest +1R-before-SL LONG capture
(SL=2·ATR, hold 60 D1 bars), unconditional and per canonical directional trigger, IS 2010–2020, 8 majors.

| trigger | n | capture | lift vs base | n>base |
|---|---|---|---|---|
| **unconditional** | 25,171 | **0.4908** | — | — |
| uptrend (close>SMA50) | 12,308 | 0.4908 | +0.0000 | 6/8 |
| momentum breakout (20-day high) | 1,585 | 0.4959 | +0.0051 | 5/8 |
| pullback in uptrend | 1,626 | 0.4895 | −0.0013 | 5/8 |
| oversold reversion (z<−2) | 2,230 | 0.4996 | +0.0087 | 5/8 |

**D1 is ALSO coin-flip.** The unconditional D1 long capture (0.4908) ≈ H4 (0.4877); NO directional trigger
lifts it meaningfully (best = oversold 0.4996, exactly coin-flip, not cross-pair consistent). The coin-flip
directional base is **timeframe-invariant**. Hypothesis (D1 trends better) FALSIFIED at observation.

## (c)+(d) Characterize + cheap kill — textbook daily trend-following

To render an apparatus-scored verdict on the strongest D1 directional case, pooled the **D1 Donchian-20
breakout-in-uptrend long** (`close > prior-20-day-high AND close > SMA50`, refractory 5) — the canonical
daily trend-following entry. `pool_sha256 70a4c6acf54bdeb0…`.
- **831 IS trades** (98–108/pair). Honest +1R-before-SL **0.4838**. Mean final_r +0.0085 (~flat). Pool floor PASS.
- **3-fold honest triage** (A1, `sl_partial_close_1r_runner_trail`, costs ON):

  | fold | OOS 2013 | OOS 2016 | OOS 2019 |
  |---|---|---|---|
  | ROI | −1.95% | −5.72% | −4.71% |
  | DD | 5.07% | 6.61% | 6.05% |
  | n | 55 | 53 | 39 |

  **TRIAGE worst −5.72%, mean −4.13% — all three folds negative.** → KILL at the cheap stage (protocol §5d).

## Council — NOT convened

No worthwhile-ceiling/diagnosis fork: the D1 directional base is coin-flip (observation) and the textbook
trend-following triage is cleanly all-negative. Same family/lesson as the already-councled arc-1000 pattern;
re-convening would re-derive it ("ritual, not rigor"). Cheap-killed on the apparatus.

## Final verdict — FAIL (cheap-kill)

The **D1 directional-long family is not deployable**: the daily directional base is ~coin-flip (0.4908,
TF-invariant), and textbook daily trend-following is sub-cost (triage all-negative). Changing the timeframe
does NOT escape the binding constraint.

## Lessons (candidate for LESSONS.md)

1. **The FX-major directional-long coin-flip base is TIMEFRAME-INVARIANT.** D1 unconditional long capture
   0.4908 ≈ H4 0.4877; no D1 directional trigger (trend, momentum-breakout, pullback, oversold) lifts it.
   Textbook daily trend-following (Donchian-20 + uptrend) is sub-cost (triage worst −5.72%, mean −4.13%).
2. **FOUR independent directional-long attempts now FAIL the same way** (arc 0 pullback H4, arc 1000 XS-mom
   H4, arc 1001 contraction-breakout H4, arc 1002 trend-following D1). The directional-long-on-FX-majors
   approach is comprehensively dry across entry construction AND timeframe. **Strong steer for future arcs:
   STOP testing directional-long entries on majors; change a more fundamental lever** — instrument universe
   (less-efficient crosses), portfolio/selection of decorrelated sub-cost signals (the open XS-as-universe
   thread), or a non-directional construction. Another major-directional-long entry is very likely wasted.

## Threads

- **Closed:** D1 directional longs (trend/momentum/pullback/oversold); daily trend-following on majors.
- **Open (carried + sharpened):** (1) instrument universe — do less-efficient CROSSES (JPY/commodity/EM-ish)
  have a directional base above coin-flip? (the HistData backup has ~28 pairs incl. many crosses); (2)
  portfolio/selection of decorrelated sub-cost signals (XS-as-universe); (3) whether ANY long on this market
  can clear cost (4 families say majors+direction cannot).

## Flags / Reproduction

No canonical-core change (no flags). Signal + drivers scratch (`_disco_work/arc1002_observe.py`,
`arc1002_d1_trend.py`); `PYTHONPATH=. py _disco_work/<script>.py`. Data: `histdata_root=C:\Users\panap\histdata_backup`,
**tf D1** 5ers_eet, 8 majors. `pool_sha256 70a4c6acf54bdeb0…`. Engine: `A1Architecture`+`ArcFoldRunner`,
FundedNext costs at `build_fold_stats_from_run`, triage folds `build_v3_folds` ids 4/7/10, hold 60 D1 bars.
