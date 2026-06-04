# Arc 2002 — H1 Entry-Resolution Test of the Weekend Gap-Fill

> **Arc id:** 2002 · **Chat:** 2000–2999 (continuous) · **Date:** 2026-06-05
> **Final verdict:** **FAIL (confirmatory cheap-kill).** Entering ~6h closer to the gap open (H1 i+1 vs H4
> i+1) does NOT rescue the gap-fill — capture (0.450 ≡ H4 0.45–0.47) and the adverse continuation (MAE
> −1.2R ≡ H4) are IDENTICAL. The down-gap's adverse continuation is TIMEFRAME-INVARIANT; entry resolution
> was never the binding constraint. **Empirically vindicates the arc-2001 HEAVY council: the gap-fill long
> is uncapturable, period — not an H4 artifact.**
> **Lever tested:** entry RESOLUTION — the one UNBLOCKED, non-curve-fit lever the arc-2001 council left open
> ("uncapturable at H4 resolution"). Is "at H4 resolution" load-bearing, or is it uncapturable at any?

Scored solely by `MultiPairBacktester` (FundedNext costs ON, SL-first). Reused the BUILT
`WeekendGapFillLongSignal` (arc 2001) on H1 panels; no new tool, no canonical-core change.

## (a) Log read — FRESH EYES

Pulled main (`d534905`). 14 arcs, all FAIL. Directional price-structure long is metric-robustly closed
(arc 3001: capture AND drift lenses agree; down-spikes CONTINUE, don't revert — so intraday discontinuities
≠ weekend gaps). The realest edge is the weekend gap-fill: arc 1006 (1000s, JPY crosses) net-POSITIVE but
fold-fragile FAIL; arc 2001 (mine, majors) net-NEGATIVE FAIL — its HEAVY council ruled it "uncapturable at
H4 resolution," thin, and with the stronger side (UP-gap short) blocked by the long-only apparatus. No STOP.

**My non-colliding lever (distinct from 1000s' cross-gap-portfolio):** the council said *at H4 resolution*.
The honest i+1 entry on H4 fills 8h after the gap open — deep in the continuation. On **H1** the i+1 entry
is ~2h after the open (~6h earlier), so it should capture the fill the H4 entry misses. This is a
MECHANISTIC change (finer bars), NOT a curve-fit parameter tuned on IS — so it's the one refinement the
council's contamination/thinness objection does not block. If H1 lifts capture/edge → entry resolution was
the problem; if not → the edge is fundamentally uncapturable on the long side (council vindicated).

## (b)+(c) Characterize on H1 (vs the arc-2001 H4 baseline)

H1 majors, IS 2010–2020, hold 120 (~5 trading days). Threshold ~2× the H4 value selects comparable big gaps
(ATR_H1 < ATR_H4). Canonical pool:

| config | n | cap+1R | meanFinalR | stop | meanMAE | medMAE |
|---|---|---|---|---|---|---|
| H1 thr1.0 SL2.0 (≈H4 thr0.5) | 555 | **0.452** | −0.032 | 0.796 | −1.21 | −1.13 |
| H1 thr1.5 SL2.0 | 291 | 0.454 | +0.197 | 0.763 | −1.24 | −1.14 |
| H1 thr2.0 SL2.0 (≈H4 thr1.0) | 151 | **0.450** | +0.189 | 0.788 | −1.33 | −1.17 |
| **H4 baseline (arc 2001)** | 176–589 | **0.45–0.47** | +0.07..+0.18 | 0.61–0.80 | **−1.0 to −1.3** | −1.05 to −1.30 |

**H1 capture (0.450–0.454) is IDENTICAL to H4 (0.45–0.47); MAE (−1.2 to −1.3R) is identical-to-slightly-
worse.** Entering 6h closer changes nothing. Mechanism: the down-gap continues adversely for *hours-to-days*
(MAE plays out over the hold), so shaving 6h off a multi-hour continuation does not escape it.

## (d) Triage on H1 (3 folds 2013/16/19) — vs H4 baseline ~−1 to −2.5%

| config | 2013 | 2016 | 2019 | mean | neg |
|---|---|---|---|---|---|
| H1 thr1.0 partial | −0.26% | −5.99% | −2.45% | −2.90% | 3/3 |
| H1 thr1.0 tp_2r | −1.84% | −5.34% | −5.31% | −4.16% | 3/3 |
| H1 thr2.0 partial (best, ≈H4-thr1.0) | +2.43% | −2.02% | −1.04% | −0.21% | 2/3 |

Same near-break-even-to-sub-cost picture as H4 (the thr2.0 best mirrors H4's lucky-fold near-break-even).
No improvement from entry resolution. → **confirmatory cheap-kill.**

## Final verdict — FAIL (confirmatory)

**Entry resolution is NOT the binding constraint of the gap-fill.** H1 (6h-earlier entry) reproduces the H4
capture (0.45), MAE (−1.2R), and sub-cost triage exactly. The down-gap's adverse continuation is
TIMEFRAME-INVARIANT; the honest fill entry is structurally late on *any* bar resolution. This empirically
vindicates the arc-2001 HEAVY council ("the mechanism survives; the trade does not") — "uncapturable at H4
resolution" was really "uncapturable on the long side, period." 10th EDGE/cost-stack FAIL.

## Lessons (candidate for LESSONS.md)

1. **The weekend gap-fill's adverse continuation is TIMEFRAME-INVARIANT** (H4 capture 0.45 / MAE −1.2R ≡ H1
   capture 0.45 / MAE −1.2R). Entering 6h closer to the gap open does not escape it — the down-gap continues
   adversely for hours-to-days. Entry resolution was never the lever; the long-only fill entry is
   structurally late at any bar size.
2. **Confirms the arc-2001 council empirically** (not just by reasoning): the gap-fill long is uncapturable,
   not an H4 artifact. Combined with arc 3001 (intraday down-spikes CONTINUE, don't revert), the long-side
   reversion-capture space is now closed at multiple resolutions.
3. **The only live gap-fill paths remain the two the council/2001 named** — neither is a long-only-H4/H1
   price-signal: (a) the cross-universe portfolio of net-positive gap edges (1000s' thread, JPY-cross gap-
   fill is the lone net-positive component), (b) the UP-gap SHORT side (stronger, 0.64 accuracy) which needs
   human-gated short support (FLAG-1, arc 2001). Adding finer timeframe is exhausted.

## Threads / FLAGS

- **Closed:** weekend gap-fill LONG via entry-resolution (H1 ≡ H4 — resolution is not the lever). The
  long-side gap-fill is now closed at H4 AND H1.
- **Carried (unchanged):** FLAG-1 (long-only blocks the stronger UP-gap short — operator/human-gated code);
  FLAG-2 (is H4+FundedNext generically hostile to fill/reversion? — arc 3001 + this arc strengthen "yes" for
  the LONG side). The gap-portfolio is 1000s' active thread.
- **Open for the 2000s lane:** the price-only long-only space is now comprehensively mapped across 14 arcs;
  the genuinely-novel, non-colliding, unblocked directions are nearly exhausted. Next arc should weigh a
  LIGHT generative council (§5b, flagged by arc 3001) to surface a non-price-structure construction, vs
  conceding the price-only long-only space and documenting the structural blockers (short side, frequency)
  as the programme's standing result.

## Reproduction

Reused BUILT `discovery/tools/gap_signals.py` (`WeekendGapFillLongSignal(primary_tf="H1")`). Driver scratch
`_disco2000_work/arc2002_h1_gap.py`; `PYTHONPATH=. py _disco2000_work/arc2002_h1_gap.py`. Data:
`histdata_root=C:\Users\panap\histdata_backup`, `cache_root=...\data\cache`, **tf H1** 5ers_eet (loads ~1s/pair
from m1 cache), 8 majors. Pool hold 120, IS 2010–2020. Engine `A1Architecture`+`ArcFoldRunner`, FundedNext
costs, `trail_enabled=False`, triage folds `build_v3_folds` ids 4/7/10. OOS NOT run (confirmatory cheap-kill).
