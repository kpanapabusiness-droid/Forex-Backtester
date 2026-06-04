# Arc 1004 — Cross Trend: Exit/Cost Engineering vs EDGE<COST

> **Arc id:** 1004 · **Chat:** 1000–1999 (continuous) · **Date:** 2026-06-04
> **Final verdict:** **FAIL (cheap-kill at triage).** No exit structure (let-it-run wide trail, standard
> trail, 3R target) flips the cross-trend triage net-positive; worst fold −9.75% to −11.7%, mean −4 to −5.5%.
> **Lever tested:** EXIT / HOLDING STRUCTURE — can let-it-run / asymmetric exits amortize cost over larger
> cross-trend moves and beat the EDGE<COST hurdle that sank arc 1003?

Scored solely by `MultiPairBacktester` (FundedNext costs ON, real cross spreads, SL-first). Canonical exit
policies (no new code); the cross signal is the arc-1003 experiment tool reused.

## (a) Log read — FRESH EYES

Pulled main. Corpus = 5 directional-long FAILs (arc 0/1000/1001 H4 majors, 1002 D1, 1003 crosses). Sharpened
diagnosis (arc 1003): **EDGE < COST** — crosses have POSITIVE gross drift (+0.10R/trade; they trend) but the
+1R-partial-runner caps runners and the wider cross spread eats the thin edge → net sub-cost. No STOP.

## (b)+(f) Idea + address

**Lever = EXIT/HOLDING STRUCTURE.** Hypothesis: arc 1003's exit (`sl_partial_close_1r_runner_trail`) banks a
+1R partial and trails the rest — it CAPS the runner near the partial, leaving the cross trend's positive
drift on the table, while still paying spread on every coin-flip entry. A **let-it-run exit** (pure wide ATR
trail, no partial) should capture more of the +drift per winning trade, raising R-per-trade and amortizing
the fixed per-trade cost — the direct lever for EDGE<COST. Test reasoned exit variants (not a blind sweep) on
the SAME arc-1003 cross trend signal (Donchian-20 breakout in uptrend, 8 crosses, 4,078 IS trades, hold 240).

## (g) Triage — exit variants (3 folds, FundedNext costs ON)

| exit variant | 2013 | 2016 | 2019 | worst | mean |
|---|---|---|---|---|---|
| baseline partial+runner (arc 1003) | +1.43% | −8.15% | −9.75% | −9.75% | −5.49% |
| **wide trail, no partial (let-it-run)** | +4.75% | −9.75% | −7.22% | −9.75% | −4.07% |
| standard trail, no partial | +4.85% | −8.89% | −11.45% | −11.45% | −5.16% |
| 3R target (asymmetric) | +6.43% | −8.90% | −11.71% | −11.71% | −4.73% |

**The let-it-run / asymmetric exits IMPROVE the trending year** (2013: +1.43% → +4.75–6.43% — riding the full
cross trend captures more) **but do NOTHING for the choppy folds** (2016 −8 to −10%, 2019 −7 to −12%). Every
variant's worst fold and mean stay clearly negative. → KILL.

## Council — NOT convened

No worthwhile-ceiling/diagnosis fork: every exit structure leaves the worst fold deeply negative; this is the
arc-0 exit lesson (exits can't rescue a ~coin-flip entry) re-confirmed on the +drift cross trend. Cheap-killed.

## Final verdict — FAIL (cheap-kill)

**Exit/holding engineering does NOT beat the EDGE<COST hurdle on the cross trend.** Letting winners run
amplifies the trending-year upside but cannot lift the choppy-year folds positive — because the entry is still
~coin-flip and every entry pays the (wide cross) spread. The cross-trend family is closed across all exit
structures.

## Lessons (candidate for LESSONS.md)

1. **Exits cannot rescue the cross trend** — across let-it-run wide trail, standard trail, and 3R target, the
   worst fold stays −9.75% to −11.7% and the mean negative. Exit engineering reshapes the trending-year upside
   (2013 +1.4%→+6.4%) but leaves choppy years deeply negative. Arc 0's "exits don't rescue a coin-flip entry"
   lesson now extends to a POSITIVE-drift instrument: a thin +drift + a coin-flip win/lose split + per-trade
   spread = net-negative regardless of how winners are harvested.
2. **EDGE<COST is an ENTRY/COST problem, not an exit problem.** The fix (if one exists) must raise the per-trade
   gross edge or cut per-trade cost (frequency), not redistribute the exit. Confirms the steer to change the
   MECHANISM (non-price-direction), not keep engineering directional trend systems.

## Threads

- **Closed:** the entire cross-trend family (entry + all exit structures); exit engineering as a lever to beat
  EDGE<COST on directional longs.
- **Open / next:** non-price-direction mechanisms — calendar-flow (turn-of-month rebalancing), to be tested
  with the CORRECT metric (mean forward DRIFT, not +1R-before-SL which is blind to small drifts) and correct
  long-only USD-pair-structure handling. The broad open question: does ANY long-only edge clear FundedNext cost?

## Flags / Reproduction

No canonical-core change. Driver scratch `_disco_work/arc1004_exits.py`; `PYTHONPATH=. py _disco_work/arc1004_exits.py`.
Signal = arc-1003 `CrossTrendBreakoutLong` (reused). Data: 8 crosses, H4 5ers_eet, `histdata_root=C:\Users\panap\histdata_backup`.
Engine: `A1Architecture`+`ArcFoldRunner`, canonical exit policies (`sl_plus_trailing_atr`, `sl_plus_tp_3r`,
`sl_partial_close_1r_runner_trail`), FundedNext costs, triage folds `build_v3_folds` ids 4/7/10.
