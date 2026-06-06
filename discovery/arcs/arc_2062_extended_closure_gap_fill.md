# arc_2062 — Extended-closure (long-weekend) gap-fill: does more closure strengthen the edge?

**Chat:** 2000s | **Date:** 2026-06-06 | **Verdict:** KILL (obs cheap-kill) | **Disposition:** KILL

## (a) Log read + synthesis (carried from arc 2061, same session)
Corpus state unchanged from 2061's read: 4 PORTFOLIO components (gap 1006 / me_long 1011 / fbr 1013 /
me_short 1019), never-AFP 4-way book, honest-§5f deploy object = me_long-solo (mean) / {me_long,fbr}
(OOS vehicle), all vehicle-infeasible; explore-now MENU exhausted; edge frontier closed on every mapped
axis; lever = operator path-A. The gap modality was declared "closed on every face" at arc 2032
(down-gap fill long 1006 = the surviving JPY-cross edge; up-gap short 1016/2013 dead; USD-neutral cross
gaps 1018 efficient; up-gap continuation 2032 dead).

## (b) Idea (seeded by arc-2061's NEW lesson)
Arc 2061 found single-center (US-bank-holiday) thinness is too mild to be tradeable in 24h-global FX,
and diagnosed that the weekend gap-fill (1006) works because of a genuine market-CLOSURE window (~48h of
no price discovery → an accumulated repricing gap that reverts). The natural follow-up: if a *closure
window* is the mechanism, does a **LONGER** closure strengthen it? A holiday-extended long weekend
(Fri close → Tue open, ~72h, spanning a US/JP holiday Monday) gives ~24h more closure than a normal
~48h weekend → *because:* more closure → more accumulated repricing → a BIGGER down-gap that fills MORE
reliably → a potentially cleaner gap sub-edge. Genuinely untested (2032 closed the gap *faces*, never
the closure-*duration* axis); tests the 2061 mechanism on the edge that survives; checked 2015/2018.

## (c)/(d) Characterize + obs cheap-kill
Canonical H4 cache, 6 JPY crosses (gap-fill universe: EUR/GBP/AUD/CAD/CHF/USD-JPY), 2010-2026,
Wilder(14) MID ATR shift1. Weekly-open bars flagged by a >20h index time-gap from the prior bar;
`closure_h` = hours since prior bar; gap = `(open_mid − prior_close_mid)/ATR`; down-gap fill = honest
+1R-before-SL long capture (1006 convention, entry t+1 `open_ask`, SL `close_ask[t]−2·ATR`) + forward
6-bar (~1 day) reversion drift. Buckets by closure_h.

**Key data fact:** closure_h is overwhelmingly **48h** (median 48, p90 52, max 76) — H4/EET bucketing
collapses weekend hours, and genuine ≥72h holiday long weekends are RARE: **n=43** weekly-open bars over
16yr × 6 pairs (~0.4/pair/yr), of which only **n=3** are down-gaps ≥0.5 ATR → already untradeably thin.

**FALSIFIED — closure duration does NOT scale or strengthen the gap:**
- **No bigger gap (premise false):** |gap| mean by closure bucket = **NORM ≤48h 0.418 / 52-71h 0.368 /
  LONG ≥72h 0.308** ATR — a longer closure produces a SMALLER gap, not bigger. (A holiday Monday is a
  quiet, low-catalyst day; the extra ~24h of closure over it adds duration without adding repricing
  news → no larger gap.)
- **No better fill:** down-gap (≤−0.5 ATR) fill capture/drift = NORM ≤48h **0.514 / +0.119**, 52-71h
  0.514 / +0.300, LONG ≥72h **0.333 / −0.747 (n=3 = noise)**. Capture is flat ~coin-flip across closure
  length; the genuine long-weekend subset is too thin to read. Deeper-gap (≤−1.0) subset: long weekends
  fill if anything slightly WORSE than normal weekends (cap 0.564 vs 0.580).
- **Not a 2015/2018 leg:** per-year noise; 2015 ~+0.11/cap 0.55, 2018 −0.27/cap 0.42 (neg, as usual);
  per-pair drift consistently + (the base 1006 edge), but closure-conditioning adds nothing over base.

§5d cheap-kill (premise false + coin-flip cap + untradeably-thin long-weekend subset). §5f does not bite
(no improvement over the base gap; and base gap is the exit-fragile / honest-§5f-mean-neg component
2042 — strengthening it would be marginal anyway, but it cannot be strengthened this way). No
engine/null/council. OOS untouched.

## (e) Diagnosis — the *because* it fails (NEW lesson; bounds arc 2061)
The gap-fill's magnitude is set by the **weekend NEWS/repricing flow** accumulated over the closure, NOT
by the closure **clock**. A normal weekend already captures the full "weekend's worth" of pent-up
repricing; tacking on a quiet holiday Monday adds hours without a catalyst → the gap is no bigger and
fills no better. Together with 2061, this **fully bounds the closure/thinness dimension of the gap
mechanism:**
- (2061) reduced-participation thinness (single-center holiday) is too mild — FX stays globally liquid,
  the move is muted (the "thinness-too-mild" failure mode);
- (2062) extending the closure window beyond a normal weekend does NOT scale the gap — it is
  news-driven, not duration-driven, and saturates at the standard weekend.

So the weekend gap-fill needs a closure window to **EXIST** (vs continuous trading — that is what
distinguishes it from a mid-week move), but the edge does not scale with how *long* the closure is once
a standard weekend is reached. The gap modality is now closed on the closure-duration axis too (after
2032's face-by-face closure).

## Outcome
- Closes the extended-closure / long-weekend gap-fill sub-lane; complements arc 2061 to bound the
  gap mechanism's thinness/closure dimension.
- Components UNCHANGED (all 4 PORTFOLIO). Deploy object UNCHANGED (me_long-solo / {me_long,fbr}). Lever
  = operator path-A. No new BUILT tool (canonical loader + inline obs; scratch removed). No canonical
  change, no FLAG, no council, OOS untouched.

**Repro:** H4 5ers_eet, 6 JPY crosses; weekly-open bars (>20h index gap); gap=`(open−prior_close)/ATR`,
down-gap (≤−0.5) honest long fill-capture + 6-bar drift, bucketed by `closure_h` (NORM ≤48h / 52-71h /
LONG ≥72h). (Inline obs, deleted post-run; numbers above.)
