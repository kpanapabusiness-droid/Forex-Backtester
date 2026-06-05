# Arc 1019 — Month-End Reversion SHORT (the untested mirror of the `me` long, for the 2018 leg)

> **Arc id:** 1019 · **Chat:** 1000–1999 (continuous) · **Date:** 2026-06-05
> **Final verdict:** **FAIL the sole judge (not all-folds-positive) → disposition PORTFOLIO.** A big UP
> move into month-end REVERTS DOWN (the short side of the same WMR-fix rebalancing flow arc 1011 proved on
> the long side). The short capture is **mean-positive net of FundedNext costs under every exit tested**
> (+0.04% sl_only/2-bar → **+0.68% partial-runner**, 7/10 folds), **beats the fair same-exit null by
> +0.80pp**, is **mechanism-controlled** (month-end +0.089 ATR excess vs the random-day control), and —
> the headline — **is robustly POSITIVE in 2018** (the portfolio's binding wall: positive under every exit
> except tp_2r, every threshold 0.75–1.5, and every leave-one-pair-out drop) with a **fragile-but-present
> 2015 tilt** (lives at threshold ≤1.0, leans on GBPUSD). **The 4th net-positive, regime-complementary
> component — and the FIRST one robustly positive in 2018**, the fold that blocked the 3-way book
> (arcs 1015/2008/3009). NOT all-folds-positive (7/10) → not a survivor → PORTFOLIO.
> **Lever tested:** MECHANISM × SHORT — the proven month-end mechanical-flow reversion (arc 1011),
> applied to the now-open short side, targeting the strong-USD binding folds 2015 & 2018.

Scored solely by `MultiPairBacktester` (FundedNext costs ON, SL-first take-the-loss) via the canonical
`ArcFoldRunner` → `build_v3_folds` IS folds + the discovery judge. No council (council is mandatory only for
a PASS survivor → `passed/`; this is PORTFOLIO, mirroring arc 1011). OOS deliberately **NOT touched** (not
all-folds-positive on IS → preserve the holdout, §4 + the arc-2001 discipline). Reused BUILT
`make_time_exit_predicate`, `build_null_signal_evaluation`, `observe_long_capture` (direction-aware short);
built + registered `MonthEndReversionShortSignal` (the direction-mirror of `MonthEndReversionLongSignal`).

## (a) Log read — FRESH EYES (honest-era only)

Pulled main. Honest-era corpus = arc 0 + arcs 1000–1018 / 2000–2013 / 3000–3015 (3 chats). My synthesis:

- **The deployment route is a portfolio of decorrelated components.** Three net-positive PORTFOLIO
  components exist: **gap-fill 1006** (JPY-cross H4 weekend down-gap long, +0.69%), **month-end `me` 1011**
  (USD-major D1 big-down-into-month-end long, +0.23% — *the one demonstrably 2018-positive mechanical-flow
  edge*), **failed-breakdown reclaim `fbr` 1013** (USD-major H4 stop-run reversal long, +1.85%, strongest).
- **The 3-way combination is PROVABLY BLOCKED** (arcs 1015/2008/3009, triple-independent; 0/5151 convex
  weightings all-folds-positive) by **2015 & 2018**: 2015 positive ONLY in fbr (+3.17), 2018 positive ONLY
  in me (+0.90, weakly) → mutually exclusive (2018 wants heavy me, but me drags 2015; 2015 wants heavy fbr).
  **The route needs a 4th component that fills 2018 (and/or 2015) without dragging the other.**
- **The 2018-leg hunt is a graveyard — ~12 dead routes:** structural shorts (1014/2009/2011/3011), trend
  short (3010), up-gap flow short (1016/2013), vol-expansion short (3012), relative-value (2010),
  deep-continuation long (2012), carry-unwind (1017), USD-neutral gap-fill (1018), weekly trend (3014),
  end-of-week squaring (3015), session-liquidity reclaim (3013).
- **The one survivor of the 2018 hunt is `me`** — mechanical/inelastic-flow reversion, 2018-positive via
  the hard WMR/index rebalancing mandate (arc 3015 showed it does NOT generalize down the calendar
  hierarchy to the weekly boundary — no forced-rebalancing driver there). LESSONS "closed ground":
  shallow single-condition directional (long OR short) is dead; carry is OFF.

**The gap I picked:** `me` (arc 1011) tested ONLY the long/down side and is **2015-NEGATIVE** (arc 1012:
negatives = contiguous 2014/15/16 USD-bull block). The **short side of the same flow was never tested.**
STOP absent. Highest arc-id in range = 1018 → resume at 1019.

## (b) Idea + observation — the SHORT mirror of `me` (documented *because*)

**Because:** month-end inelastic rebalancing (the WMR 4pm London fix on the last business day) reverts
price **both directions** — arc 1011 proved this on the long side (a big DOWN move into month-end reverts
UP, +0.249 ATR excess over a random-day control). The **short side** fades a big UP move into month-end.
Mechanistically this should be POSITIVE precisely in strong-USD years (2015/2018), the exact regime where
the long `me` is weak/negative: in a strong-USD trend a big UP move into month-end on a USD major (e.g.
EURUSD up = a USD-weakness bounce) is COUNTER-trend, so the mechanical reversion + trend-resumption pushes
it back down (short wins). This is **not** a structural coin-flip short (which died by symmetry — 1014/2009/
2011): the month-end flow is real, directional-agnostic, and `me` already proved its timing is load-bearing.

**Observation (D1, 7 USD majors, IS 2010–2020;** `_disco_work/arc1019_observe_monthend_short.py`**).** For each
month-end, a big UP move INTO it (last 2 days, ATR) → forward 2-day SHORT reversion (−fwd2):

| UP move into month-end | n | mean | median | frac short-pos |
|---|---|---|---|---|
| **≥ +1.0 ATR** | 122 | **+0.075** | **+0.096** | 0.566 |
| +0.5..+1.0 | 147 | +0.070 | +0.050 | 0.531 |

Median ≈ mean → **NOT a thin-tail artifact** (the tell that killed 2011/3012/3014, where median ≪ mean).
Honest +1R-before-SL **short capture pooled 0.5508 > 0.50** (the FIRST short in the corpus to clear 0.50
with a controlled mechanism — every prior short was sub-0.50: 1014 .489, 2011/3011 .473, 1016 .448).

## (c) CRITICAL CONTROL — is month-end load-bearing, or generic reversion (dead)?

Same big UP move (≥ +1 ATR), month-end vs RANDOM non-month-end days:

| ≥ +1 ATR up move | n | mean (short drift) | median |
|---|---|---|---|
| **MONTH-END** | 122 | **+0.075** | +0.096 |
| **RANDOM DAY** | 2731 | **−0.015** | −0.009 |

**Month-end EXCESS = +0.089 ATR** (smaller than the long `me`'s +0.249 but the SAME sign and direction —
the month-end *timing* is load-bearing for the short too, not generic reversion).

## (d) Binding-fold acceptance (the whole point) — 2015 & 2018

Gross per-year short drift (month-end up-move ≥+1 ATR) and honest short capture:

| year | n | mean | median | frac+ | capture |
|---|---|---|---|---|---|
| **2015** | 11 | **+0.464** | +0.273 | 0.727 | 0.545 |
| **2018** | 11 | **+0.365** | +0.122 | 0.727 | **0.818** |

Both binding folds robustly positive (median-positive, not a single trade). Per-pair (≥+1 ATR month-end):
EURUSD +0.137, GBPUSD +0.286, USDJPY +0.146, USDCAD +0.096, USDCHF +0.008 positive; **AUDUSD −0.100,
NZDUSD −0.274 negative** — the OPPOSITE pairs from the long `me` (AUD/NZD-driven) → a decorrelation bonus.

## (e)–(g) Honest engine WFO (§5f exit sweep) + fair null

Non-coin-flip (capture 0.55, beats control) → §5f mandates the honest engine with the exit menu swept.
`_disco_work/arc1019_wfo.py`, IS folds 2011–2020 (fold 6=2015, fold 9=2018), FundedNext costs ON:

| exit | folds_pos | mean | worst | 2015 | 2018 | AFP |
|---|---|---|---|---|---|---|
| sl_only + time-exit 2-bar | 5/10 | +0.043% | −0.93% | +0.61 | +0.14 | N |
| sl_only + time-exit 3-bar | 6/10 | +0.102% | −1.52% | +1.84 | +0.14 | N |
| sl_only + time-exit 5-bar | 6/10 | +0.218% | −1.32% | +1.85 | +0.54 | N |
| sl_plus_tp_2r | 5/10 | +0.362% | −1.49% | +1.50 | **−0.43** | N |
| **sl_partial_close_1r_runner_trail** | **7/10** | **+0.683%** | **−0.91%** | **+0.40** | **+0.86** | N |
| **FAIR NULL** (random entry, sl_only 2-bar) | 2/10 | **−0.761%** | −2.35% | −1.13 | +0.51 | N |

**Real beats the fair same-exit null by +0.80pp** (sl_only te2 real +0.043% vs null −0.761%; 5/10 vs 2/10)
— the entry timing carries information, mirroring the long `me`'s +0.56pp and corroborating the +0.089 ATR
gross control. **Mean-positive net of costs under EVERY exit** (+0.04% → +0.68%). Best = partial-runner
(7/10, both binding folds positive). Pool n=116 (floor PASS), gross mean_final_r +0.1713.

## Robustness (`_disco_work/arc1019_robust.py`, partial-runner exit)

- **Threshold sweep:** thr 0.75/1.0/1.25/1.5 → mean +0.84/+0.68/+0.75/+0.31%, 6–8/10 positive. **2018
  positive at EVERY threshold (+0.68 to +1.72).** **2015 positive at thr ≤1.0 (+0.54/+0.40) but flips
  negative at thr 1.25/1.5 (−0.76)** → 2015 is threshold-fragile; 2018 is not.
- **Leave-one-pair-out:** **2018 positive in EVERY drop (+0.59 to +1.99)** — robustly broad. But
  **drop-GBPUSD collapses the mean to −0.016% and flips 2015 to −0.16** → 2015's positivity leans on
  GBPUSD; 2018 does not.
- **Per-fold (partial-runner, full 7 pairs):** `2011 +3.39 · 2012 +1.69 · 2013 −0.90 · 2014 +0.98 ·
  2015 +0.40 · 2016 −0.91 · 2017 −0.68 · 2018 +0.86 · 2019 +1.29 · 2020 +0.71` (negatives 2013/16/17).

## (h)/(i) Verdict, decorrelation, threads

**Verdict: FAIL the sole judge (not all-folds-positive, best 7/10) → PORTFOLIO** (§11): mean-positive net
of costs under every exit, beats the fair null by +0.80pp, mechanism-controlled (+0.089 ATR month-end
excess), and **robustly 2018-positive** — the binding wall that 12+ prior routes failed. A CANDIDATE
component, NOT deployable solo, never a survivor.

**Decorrelation vs the sibling `me`-long (arc 1011):** fold-ROI corr **+0.157** (near-zero) despite the
shared month-end event — and they **cannot co-fire** (a move into month-end is either up or down) and split
by pair (short = EUR/GBP/JPY/CAD-driven; long = AUD/NZD-driven). Crucially **regime-complementary on the
binding fold**: the short is **2015 +0.40** where me-long is **2015 −1.14**, and both are 2018-positive
(short +0.86, me +0.90). This is the regime-orthogonality the 3-way book lacks.

**The honest caveat (documented, not hidden):** the 2015 leg is **fragile** — threshold-sensitive (dies at
thr ≥1.25) and GBPUSD-leaning. The 2018 leg is the robust one. For the portfolio this is acceptable because
2015 is already strongly handled by fbr (+3.17); the component's unique value is the **robust 2018** leg
that only `me` weakly provided. Whether this unlocks an all-folds-positive 4-way book is the gated next arc.

**Threads / next step (FLAG — the gated combination arc):** there are now **FOUR** net-positive components,
and for the FIRST time one is robustly 2018-positive AND (fragile-)2015-positive. The immediate gated step
is the **4-way combination WFO** (gap 1006 + me 1011 + fbr 1013 + month-end-short 1019), all-folds-positive
on the combined book, via `combine_fold_roi` — the 2018 wall (arc 2008's binding fold) now has a robust
contributor it lacked. That is arc 1020, not claimed here.

**FLAGS (code not merged):** none requiring the canonical core. Built + registered
`discovery/tools/month_end_signals.py :: MonthEndReversionShortSignal` (EXPERIMENT tool — mask + ATR
geometry + `Direction.SHORT` only; scoring canonical). Carries the standing `A1Config.time_exit_bars`-
unwired flag (arcs 1005/1011; worked around via the BUILT `make_time_exit_predicate`). Drivers scratch
`_disco_work/arc1019_*.py` (reproducible from this doc). OOS untouched.
