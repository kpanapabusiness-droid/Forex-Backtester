# Arc 1007 — Gap-Fill Best-Version Test: Exit AT the Fill Target

> **Arc id:** 1007 · **Chat:** 1000–1999 (continuous) · **Date:** 2026-06-04
> **Final verdict:** **FAIL** — the mechanism-aligned "exit at the gap origin" exit is WORSE than arc 1006's
> 24-bar time exit (IS mean +0.69% → −1.21%, and it LOSES to the random null). The gap-fill edge is in the
> OVERSHOOT past the origin, not the fill-to-origin. Arc 1006's 24-bar version remains the best version;
> the gap-fill is not all-folds-positive in any tested exit.
> **Lever tested:** EXIT GEOMETRY (structural target vs time) on the run's one net-positive lead.

Scored solely by `MultiPairBacktester` (FundedNext costs ON, SL-first). Built + used a reusable price-target
exit predicate (BUILT tool); canonical engine realizes P&L.

## (a) Log read — FRESH EYES

Pulled main. Corpus = 13 arcs (0; 1000–1006; 2000; 3000–3003), all FAIL. The ONLY net-positive lead is arc
1006 (JPY-cross weekend gap-down-fill: IS mean +0.69%, beats random, low-DD, but fold-fragile → FAIL
all-folds-positive). Arc 1006 used an ARBITRARY 24-bar time exit. No STOP.

## (b)+(f) Idea — develop the best version (mechanism-aligned exit)

Hypothesis: the gap-fill edge is the reversion TO the gap origin, so the 24-bar time exit holds past the fill
and gives it back. Exit AT the gap origin (prior-week close) should capture the reversion cleanly + free
capital + cut the give-back → higher per-trade edge + better fold-consistency. Built
`make_price_target_exit_predicate` (BUILT tool): long exits at a per-trade target price (gap origin) when the
bar reaches it, else a 24-bar time fallback; SL=2·ATR (engine, SL-first). Same trigger as arc 1006 (5 JPY
crosses, week-open gap-down < −0.5 ATR). One reasoned version, MEASURED on IS then OOS (no OOS tuning).

## (g) Full WFO + null

**IS (10 folds):** −2.64 / −1.65 / −1.09 / +0.13 / +0.07 / +4.61 / −2.36 / −2.49 / +0.13 / −6.76 %.
**all-folds-positive NO** — worst −6.76%, **6/10 negative, mean −1.21%** (WORSE than arc 1006's +0.69%).

**OOS:** −2.21 / +4.32 / −3.93 / −0.27 / −0.78 / +0.60 % — all-folds-positive NO, 4/6 negative.

**Null:** REAL IS mean **−1.21%** does NOT beat random (−0.60%) — the exit-at-target version is no better than
random entry. (Arc 1006's 24-bar version DID beat random at +0.69%.)

## Diagnosis — why the "improvement" hurt

The exit-at-origin **caps the upside at the fill level**, but the gap-fill winners in the 24-bar version
OVERSHOOT the origin (the JPY-cross snapback runs past the gap and continues up). Capping at the origin cut
those big winners while keeping the full −1R losers on trades that hit SL → the win/loss asymmetry flipped
negative. **The edge is the OVERSHOOT/continuation after a big weekend gap-down, not the mean-reversion to
the origin.** The arbitrary "let it run 24 bars" exit was, by accident, capturing the right thing.

## Final verdict — FAIL (best-version test: the proposed exit is worse)

Arc 1006's 24-bar time exit remains the best version of the gap-fill. The mechanism-aligned exit-at-target is
worse (caps the overshoot). The gap-fill is **not all-folds-positive in any tested exit** — its fold-fragility
is driven by tail-event TIMING (which weekends produce big JPY-cross gaps), not by the exit, so exit
engineering cannot fix it (consistent with arc 1004's general finding).

## Lessons (candidate for LESSONS.md)

1. **The JPY-cross weekend gap-down edge is OVERSHOOT/continuation, not fill-to-origin.** Exiting at the gap
   origin (capping at the "fill") turns the mean-positive 24-bar version (+0.69%, beats random) into a
   net-NEGATIVE one (−1.21%, loses to random) — it cuts the winners' overshoot while keeping full losers.
   Re-frames the mechanism: a big weekend gap-down on a JPY cross is followed by a SNAPBACK that runs past the
   prior close.
2. **Confirms arc 1004 on the one positive lead:** exit engineering does not fix fold-fragility — the
   fragility is tail-event-timing-driven, not exit-driven. The arc-1006 24-bar version stays the best;
   the gap-fill remains a real-but-fold-fragile, NOT-all-folds-positive edge.
3. **The BEST version of a lead can be the naive one** — the "mechanism-aligned refinement" (exit at fill)
   was a worse hypothesis than the arbitrary time exit, because the mechanism was mis-stated (overshoot, not
   reversion-to-level). Test the refinement, don't assume it.

## Threads

- **Open (carried):** gap-fill as a portfolio COMPONENT (arc 1006's 24-bar version is the net-positive,
  fold-fragile candidate; needs a 2nd net-positive edge to combine). A let-the-overshoot-run TRAIL might
  capture more than the 24-bar exit — but exit tweaks won't fix the tail-event fold-fragility (low priority;
  beware over-tuning a thin signal). 
- **Closed:** exit-at-fill-target for the gap-fill (worse — caps the overshoot); exit engineering as a way to
  make the gap-fill all-folds-positive.
- **Queued (council, untested):** spread-tier gating, vol/cost-ratio conditioning, Asia→London timing, and the
  Devil's random-entry/passive-buy-and-hold null-confirmation (the decisive "is the apparatus capable of any
  all-folds-positive long" closure test — increasingly the right move as leads narrow).

## Flags / Reproduction

No canonical-core change. Built + registered `make_price_target_exit_predicate` (works correctly — it just
isn't the right exit for THIS signal). Driver scratch `_disco_work/arc1007_wfo.py`; signal = `GapFillToTargetLong`
(5 JPY crosses, week-open gap-down<−0.5 ATR, TP at prior-week close, 24-bar fallback, SL=2·ATR). Engine:
`A1Architecture`+`ArcFoldRunner` (`exit_policy=None, trail_enabled=False`), FundedNext costs, IS `build_v3_folds`,
OOS `build_oos_year_folds(2021)`, null `discovery/tools/null_entry_baseline`.
