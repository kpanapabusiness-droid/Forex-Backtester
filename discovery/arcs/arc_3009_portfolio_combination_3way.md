# arc 3009 — 3-way PORTFOLIO combination WFO (gap-fill 1006 + month-end 1011 + failed-breakdown 1013)

**Chat:** 3000s · **Date:** 2026-06-05 · **Disposition:** KILL (combined 3-way book not deployable; the
three components retain their PORTFOLIO status, unchanged) · **Council:** none (not a PASS survivor, not an
idea-fork/diagnosis — a direct quantitative combination test with a decisive arithmetic result; arc-2006
precedent).

## (a) Log read / synthesis (fresh eyes, honest-era only)
Pulled main; read DISCOVERY_PROTOCOL, full DISCOVERY_LOG (both tiers, through arc 2007 / 1013 / 3008),
LESSONS, TOOL_REGISTRY. State after 29 arcs / 3 chats:
- **Directional prediction is comprehensively closed** (all entry constructions, H1/H4/D1, 28 pairs,
  capture + drift lenses, all exits/SL, even stop-removed 3004; DEEP multi-factor 2007 loses to the null).
  Novel flow/structure mechanisms keep dissolving on honest data (gotobi 1008, round-numbers 1010,
  triangulation 3005, month-end fix 3008).
- **THREE live net-positive long-only PORTFOLIO components now exist** (up from the two arc 2006 combined):
  (A) **1006/1009** weekend gap-fill long, JPY crosses H4 (+0.685% IS); (B) **1011** month-end reversion
  long, USD majors D1 (+0.232% IS); (C) **1013** failed-breakdown RECLAIM (stop-run reversal) long, USD
  majors H4 (**+1.854% IS, 9/10 folds; OOS +0.94% mean-positive; beats a NEGATIVE null by +2.96pp** — the
  strongest+cleanest directional long in the corpus, a NEW mechanism family vs the two flow-reversions).
- **The explicit, high-value open thread (arc 1013 lesson #5 + arc 2006 spec):** run the **3-way
  combination WFO** — the 2-way (1006+1011) was KILLed by a *mutually-negative* 2015 fold (both bled), and
  arc 1013 is positive in 2015/16/20 (3 of the 4 years the 2-way bled, sharing only 2018), so the 3-way
  "should cut the book's mutually-negative folds from four to ≈one (2018)." Nobody had RUN it.

## (b) Idea + why
The PORTFOLIO route's whole premise is: combine decorrelated mean-positive components into an
all-folds-positive book (§6/§11 — the combined book is its own all-folds-positive gate). We now have the
exact 3 components the spec called for, including a 3rd of a *different mechanism family* (structural
stop-run reversal, not flow-reversion) that is positive in 3 of the 2-way's 4 bleed years. The single most
decision-relevant unrun experiment is the honest 3-way combination with the principled weightings (equal +
risk-parity, fit-on-IS-frozen). Decisive either way: a PASS candidate (→ council → `passed/`), or a
rigorous "3 is still not enough → here is the exact 4th-component spec." This is the literal next step the
route was built toward.

## (c)/(d)/(e) Method (measurement CALLED; the linear combination is the one BUILT tool)
No new pool/cluster/oracle — all three components are already characterized. A **validation/combination**
step, extending arc 2006 with the 3rd component:
1. Reproduce each component via its **REGISTERED** signal over the SAME canonical IS fold set
   (`build_v3_folds`, `is_days>=365` → 10 expanding folds, OOS years 2011–2020), scored solely by
   `MultiPairBacktester` via `A1Architecture`+`ArcFoldRunner`+`run_config_over_folds`, FundedNext costs ON,
   SL-first:
   - **A — gap-fill:** `WeekendGapFillLongSignal(threshold_atr=0.5, gap_hours=36)` on 5 JPY crosses
     (EURJPY,GBPJPY,AUDJPY,CADJPY,CHFJPY), H4, 24-bar `make_time_exit_predicate`, `sl_only`, trail off, SL 2·ATR.
   - **B — month-end:** `MonthEndReversionLongSignal(threshold_atr=1.0, into_bars=2)` on 7 USD majors, D1,
     2-bar `make_time_exit_predicate`, `sl_only`, trail off, SL 2·ATR.
   - **C — failed-breakdown:** `FailedBreakdownReclaimLongSignal(swing_lookback=40, min_shadow_atr=1.25)`
     on 7 USD majors, H4, `sl_plus_trailing_atr` **with native A1 trail ON** (trail_enabled=True), SL 2·ATR.
2. Combine per-fold ROI via the BUILT `discovery/tools/combine_fold_roi.py` under **equal** and
   **risk-parity** (inverse-fold-vol) weights; weights FIT ON IS, frozen. Apply all-folds-positive to the
   combined book.

**Reproduction check (Arc-10 discipline — don't trust transcribed numbers).** All three reproduce EXACTLY
via the registered tools: gap-fill IS mean **+0.685%**, month-end **+0.232%**, failed-breakdown **+1.854%**
— matching the 1009/1011/1013 records to the basis point, with per-fold series byte-identical. This
confirms fold-alignment (the `build_v3_folds` date windows are TF/universe-agnostic, so fold *j* is the
same calendar year for the H4, D1 and H4 signals).

**A reproduction nuance the live re-run surfaced (worth recording).** The faithful arc-1013 component
requires **trail_enabled=True** — the native A1 `TrailManager` (activation 2·ATR, distance 1.5·ATR) runs
*on top of* the `sl_plus_trailing_atr` exit policy (a double-trail). With trail_enabled=False (only the
exit-policy trail) the same signal scores **+2.084% / 2-neg** — looser exits, a different per-fold series.
A small probe (`_disco3_work/arc3009_fbr_probe.py`) pinned the published series (mean +1.854%, 1-neg) to
trail_enabled=True at L1=0.03. Reproducing live (not transcribing) caught the config detail — exactly the
discipline working. (This does not change the 3009 verdict — see (g) — and is invariant to it.)

## (g) Results — combined 3-way book (IS, 10 folds, OOS years 2011–2020)

Per-fold ROI %, fold order (ids 2–11 → OOS years 2011–2020):
```
gap-fill (A):  -0.07, +8.23, -2.06, +2.94, -4.19, +3.20, +0.53, -6.79, +7.45, -2.39   mean +0.685%  5/10 neg
month-end(B):  +0.40, +0.29, +0.96, -0.23, -1.14, -0.51, +0.34, +0.90, +1.16, +0.15   mean +0.232%  3/10 neg
fail-brk (C):  +7.55, +3.05, +0.91, +0.19, +3.17, +2.55, +1.23, -4.20, +0.05, +4.03   mean +1.854%  1/10 neg
```

**Mutually-negative-fold pre-screen (arc-2006 lesson #4, combination-method-invariant): PASSES.** Every
fold has ≥1 positive component — **there is NO fold where all three are negative.** 2015 (the 2-way's
provable block) is rescued by fail-breakdown (+3.17); 2018 is rescued by month-end (+0.90). So the 3-way is
**no longer provably blocked** — a real structural advance over the 2-way (whose 2015 mutual-negative made
*any* weighting impossible). Whether it clears now depends on weights/magnitudes.

| weighting | weights (gap/me/fbr) | combined per-fold ROI % | mean | neg | all-folds-pos |
|---|---|---|---|---|---|
| equal | 0.333/0.333/0.333 | +2.63,+3.86,**−0.07**,+0.97,**−0.72**,+1.75,+0.70,**−3.36**,+2.89,+0.60 | +0.924% | 3/10 | **NO** |
| risk-parity | 0.106/0.726/0.168 | +1.55,+1.60,+0.63,+0.18,**−0.74**,+0.40,+0.51,**−0.77**,+1.64,+0.53 | +0.552% | 2/10 | **NO** |

- **Adding the 3rd component materially improved the book** but not past the gate. vs the 2-way risk-parity
  (4/10 neg, worst −1.53%): the 3-way risk-parity is **2/10 neg, worst −0.77%** — fail-breakdown rescued
  **2016 and 2020** (and lifted 2015 from −1.53 to −0.74) and halved the worst drawdown fold. The neg set
  narrowed from {2015,2016,2018,2020} → **{2015, 2018}**.
- **The two surviving blockers, and why each resists the principled weightings:**
  - **2018 is the hard wall.** gap (−6.79) and fail-breakdown (−4.20) are BOTH deeply negative; only
    month-end is positive and it is small (+0.90). No non-negative weighting of {−6.79, +0.90, −4.20} is
    positive unless month-end gets ~94%+ weight (which starves the book). 2018 is a persistent strong-USD
    *trend* year — exactly where the two reversion edges AND the stop-run-reversal long all bleed (real
    breakdowns, not swept; arc 1013's own named regime weakness).
  - **2015** is rescued by fail-breakdown (+3.17) but risk-parity down-weights it (highest-vol component →
    weight 0.168), so its rescue is diluted while month-end (its worst fold, −1.14) is up-weighted (0.726);
    under equal weight 2015 is dragged by gap-fill's −4.19 (the CHF-depeg JPY-cross gap year). 2015 is
    *winnable* with a heavier fail-breakdown tilt, but that is not a principled, non-fished weighting and it
    does not fix 2018.
- **IS is not all-folds-positive under either principled weighting → OOS deliberately NOT touched** (§4 +
  holdout preservation; the 2021+ holdout stays pristine for a future book that *does* clear IS).

## Verdict: FAIL the sole judge (combined 3-way book) → KILL
The 3-way combined book of the programme's three net-positive components is not all-folds-positive under
equal or risk-parity weighting. Unlike the 2-way it is **no longer provably blocked** (no mutually-negative
fold) — the failure is now a weighting/magnitude shortfall on exactly two folds (2015, 2018), not a
combination-invariant impossibility. **The three components are UNCHANGED** — reproduced exactly, not
re-tested or weakened; they retain PORTFOLIO status. No new `portfolio-candidates/` entry (would
double-count 1006/1011/1013). The portfolio **thread stays ACTIVE**, now with a much sharper, 1–2-fold spec.

## Convergence with arc 2008 (chat 2000s, ran concurrently — independent reproduction)
The 2000s chat built and ran the IDENTICAL 3-way combination in parallel (arc 2008) and reached the SAME
verdict — KILL, combined book not all-folds-positive, blockers 2015+2018, components unchanged. Two
independent chats, two drivers, one conclusion: the Arc-10 independent-reproduction defense, here for free.
The findings are complementary, not redundant:
- **Arc 2008 added an exhaustive convex-weight grid search** — 0/5151 convex weightings are
  all-folds-positive (best max-min −0.222%) — proving the block is convex-hull-wide, with the sharp
  mechanism that **2015-needs-fbr and 2018-needs-month-end are mutually exclusive** (heavy month-end fixes
  2018 but worsens 2015; heavy fbr the reverse). This is the stronger impossibility statement.
- **This arc (3009) added the exact-reproduction discipline** — reproduce-live caught arc-1013's native-trail
  double-trail (trail_enabled=True), and framed the result as: the 2-way's *combination-invariant* 2015
  block is GONE (no mutually-negative fold), so the 3-way failure is a weighting/magnitude shortfall, not an
  additive-P&L impossibility. (Both views agree the principled weightings fail; 2008's grid shows even
  non-principled convex weightings fail.)

Both reduce to the same 4th-component spec and the same steer below.

## The 4th-component spec (the actionable deliverable)
To make a 3-way (→4-way) risk-parity book all-folds-positive, the 4th component must be net-positive on the
two folds where the 3-way book is still negative — **OOS years 2015 and (critically) 2018** — with the
per-fold deficits to overcome (risk-parity): **2015 −0.74%, 2018 −0.77%.** 2018 is the binding one: it needs
a component that is positive in a **persistent strong-USD / risk-off trend year**, where all three current
edges (two flow-reversions + one structural-reclaim long) lose together. The long/trend menu that would win
those years is comprehensively dead (arcs 0–3006), so the regime-orthogonal 4th leg is most naturally a
**short / trend-following-short / relative-value** construction — **and shorts are now ENABLED** (PR #273,
honest-engine sweep SAFE). Concrete named shorts targets from the corpus that are *structurally* risk-off /
strong-USD positive: the **climax-sweep SHORT** (arc 2007 — the violent forced sweep is a falling knife,
drift −0.33; the stronger leg of the failed-breakdown family) and the **up-gap weekend SHORT** (arcs
2001/2003 — the stronger leg of the gap mechanism). Either is a high-value next arc whose explicit job is to
be positive in 2018/2015.

## Threads / lessons
1. **Three mean-positive components STILL did not make an all-folds-positive book** — but for a *different,
   weaker* reason than the 2-way: not a mutually-negative fold (that block is gone), but a
   weighting/magnitude shortfall on 2 folds. Adding a regime-different 3rd (the stop-run-reversal long)
   removed 2 of the 4 blocker folds (2016, 2020) and halved the worst fold — **fold-complementarity works,
   it was just one component short.** The route is converging, not dead.
2. **2018 is the corpus's recurring portfolio wall** — a persistent strong-USD trend year where flow-
   reversion (gap-fill, month-end) AND structural-reclaim-long (failed-breakdown) all bleed simultaneously.
   The arc-2006 "tail-correlation, not average-correlation" lesson sharpened: even a *different long
   mechanism family* (structural vs flow) shares the strong-trend tail, because **all three are long and
   none profits from a sustained directional risk-off move.** The diversifier must change SIGN-exposure
   (short/trend), not just mechanism family.
3. **The mutually-negative-fold pre-screen earns its keep as a cheap go/no-go** (arc-2006 lesson #4): the
   3-way passes it (no all-negative fold) where the 2-way failed it, correctly flagging that the 3-way is
   *worth combining* (a passing weighting is not arithmetically impossible) even though the principled
   weightings fall short. The screen distinguishes "impossible" from "needs a better/4th component."
4. **Risk-parity again helps variance, not the gate** (arc-2006 lesson #3, re-confirmed on 3 components):
   it cut the worst fold (−3.36 equal → −0.77 risk-parity) and neg folds (3 → 2), but cannot manufacture a
   pass; only a complementary component positive on the blocker folds moves the gate.
5. **Reproduce-live caught a config detail transcription would have hidden** (the arc-1013 native-trail
   double-trail): faithful reproduction needs trail_enabled=True for the `sl_plus_trailing_atr` component.
   Arc-10 discipline (don't trust transcribed numbers) working as designed — and the 3009 verdict is
   invariant to it (both fbr variants leave the book not-all-folds-positive, blocked by 2015+2018).

## Tooling
No new BUILT tool. Reused the BUILT `discovery/tools/combine_fold_roi.py` (`fit_weights`,
`combine_fold_rois`) and the three registered EXPERIMENT signals (`WeekendGapFillLongSignal`,
`MonthEndReversionLongSignal`, `FailedBreakdownReclaimLongSignal`) + `make_time_exit_predicate` — all
CALLED, scoring stays canonical (`ArcFoldRunner`→`MultiPairBacktester`). The combiner's documented
first-order-linear LIMITATION applies and is slightly *less* clean here than arc 2006: month-end and
failed-breakdown share the USD-majors universe (disjoint TFs/event-timing — D1 month-end vs H4 structural —
so simultaneous same-pair positions are rare), gap-fill is disjoint (JPY crosses). A fully-faithful
single-engine co-simulation would share margin/daily-DD across the USD-major legs; the linear book is
first-order. **The KILL is robust to this:** co-simulation shares the risk budget and does not *add* return,
so a linear book that is already not-all-folds-positive (blocked on 2015/2018) would not become
all-folds-positive under co-sim — and 2018's block (two deep negatives, one small positive) is additive-P&L
invariant regardless. FLAG for the operator: if a future 4-way book clears IS *linearly*, verify it under
co-simulation before any `passed/` claim.

## FLAGS (code not merged)
None requiring the canonical core. Observations for the operator's deep-dive (not code changes):
- **(repro/doc)** The committed arc-1013 `sl_plus_trailing_atr` config double-trails (native A1 trail ON +
  exit-policy trail). Its published exit-menu rows were likely all run with trail_enabled=True, so each row
  is "exit_policy + native trail," not the exit policy alone (e.g. the "sl_only" row is sl+native-trail).
  Does not change arc 1013's PORTFOLIO disposition (still mean-positive, not all-folds-positive); a
  labeling clarification for the deep-dive.
- Carries the standing **FLAG-1** (the regime-orthogonal 4th leg the portfolio route now needs is a
  short / relative-value construction — **now unblocked**, PR #273) and the `A1Config.time_exit_bars`
  -unwired flag (arcs 1005/3004; worked around via the BUILT `make_time_exit_predicate`).

Driver scratch `_disco3_work/arc3009_combo3.py` + `_disco3_work/arc3009_fbr_probe.py` (reproducible:
`PYTHONPATH=. py _disco3_work/arc3009_combo3.py`, `histdata_root=C:\Users\panap\histdata_backup`).
