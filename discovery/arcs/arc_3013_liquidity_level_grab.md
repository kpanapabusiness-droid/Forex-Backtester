# arc 3013 — Failed-breakdown RECLAIM long at session-liquidity levels (prior-DAY / prior-WEEK low)

**Chat:** 3000s · **Date:** 2026-06-05 · **Verdict:** FAIL (cheap-kill at observation) → **KILL**
**Disposition:** KILL · **passed:** N · **Component touched:** arc-1013 (UNCHANGED, still PORTFOLIO)

> Can a higher-significance liquidity pool — the prior-DAY / prior-WEEK extreme (classic ICT
> session-liquidity levels where stops cluster most densely) — give a cleaner/stronger grab than
> arc 1013's rolling-40-bar swing low? **No.** The structural pivot (1013) is the best level; the
> time-based pools dilute it, and all remain −2018. 1013's level choice is vindicated as load-bearing.

---

## Log reading (step a — FRESH EYES, honest-era only)

Resumed 3000s at arc 3013 (prior in-range = 3012). No `discovery/STOP`. State after 38 arcs unchanged
from arc 3012's reading: directional space closed; three PORTFOLIO components (gap-fill 1006, month-end
1011, failed-breakdown-reclaim 1013 — the strongest+cleanest corpus edge, 9/10 IS); the 3-way book
(1015/2008/3009) is the strongest corpus result but provably blocked by **2015 & 2018**; every
directional/flow/vol short for the 2018 leg is dead (1014/2009/2011/3010/1016/**3012**, the last my own).

## Idea + why (improve the corpus's best edge — a PASS would be the highest-value win)

Rather than force a 7th strained 2018-leg construction, attack the higher-value target: **arc 1013 is
9/10 IS — one fold from a standalone PASS.** 1013 defines its liquidity pool as a **rolling 40-bar
swing low** (`low_bid.shift(1).rolling(40).min()`). A well-documented *because* never tested in the
corpus: stops cluster most densely at **session-defined extremes — the prior-DAY low and prior-WEEK
low** (the ICT/Wyckoff "liquidity pool" levels), arguably cleaner grab targets than a rolling-window
pivot. Hypothesis: sweeping + reclaiming a prior-day/prior-week low is a stronger, cleaner grab → a
higher-capture, possibly all-folds-positive (PASS) version of 1013, or at least a +2018-robust one.

## Method (CALLED canonical; observation only)

`Panel.from_pairs` (H4, cached) on the 1013 universe — 7 USD majors (EURUSD, GBPUSD, AUDUSD, NZDUSD,
USDJPY, USDCAD, USDCHF). For each pair built the failed-breakdown-reclaim mask (low pierces the level,
close_mid reclaims above it, lower rejection shadow ≥ **1.25 ATR** — 1013's deep-grab spec) at **three
levels**: `swing40` (1013 baseline, reproduction control), `prior_day` low (prior completed EET day),
`prior_week` low (prior ISO week) — all causal (shift1). Honest long capture + 24-bar drift via BUILT
`observe_long_capture(direction="long", restrict=<level mask>)`. CHARACTERIZATION ONLY (gross, not a
gate). Driver: `_disco3_work/arc3013_observe_liquidity_levels.py`.

## What happened — improvement FALSIFIED; the rolling swing is best

| level | n | cap | drift mean | drift med | per-pair >0.50 | 2018 cap/drift |
|---|---|---|---|---|---|---|
| **swing40 (1013)** | 356 | **0.5815** | **+0.191** | +0.141 | **7/7** | 0.47 / −1.28 |
| prior_day | 587 | 0.5520 | **−0.062** | +0.016 | 5/7 | 0.63 / −0.72 |
| prior_week | 242 | 0.5620 | +0.103 | +0.084 | 5/7 | 0.36 / −2.30 |

- **swing40 reproduces 1013's quality** (cap 0.58 ≈ 1013's 0.55–0.61; positive median drift; **all 7
  pairs >0.50**) — the apparatus and the edge reproduce (Arc-10 discipline).
- **prior_day is WORSE:** more fires (587) but **negative mean drift (−0.062)**, only 5/7 pairs positive
  (USDCHF 0.46, USDJPY 0.49) — diluted.
- **prior_week is WORSE:** lower cap, lower drift, only 5/7 pairs (AUDUSD 0.41), thin (n=242).
- **All three levels remain −2018** (swing40 −1.28, prior_day −0.72, prior_week −2.30). The time-based
  levels do NOT rescue 2018; prior_week is the WORST there.

(OOS years 2021+ appeared in the by-year print but were only glanced, never optimized against — the
level ranking is an aggregate structural-quality comparison dominated by the IS period; §4 intact.)

## Diagnosis — structural pivot > time-based liquidity pool

The 40-bar swing low is a **structurally significant pivot**: a low that *held for 40 bars* is a real
support / reversal point — so a sweep-and-reclaim there is a genuine failed-breakdown (stops swept AND
a high prior probability of reversal). The prior-day / prior-week low is a **time-based** level: it is
swept routinely in normal trending/ranging WITHOUT being a reversal point, so the reclaim filter
catches many non-reversals → dilution (lower/negative drift, fewer pairs positive). **1013's
swing-pivot definition is load-bearing**, not an arbitrary parameter; the ICT session-liquidity-pool
hypothesis does not improve it. And the **2018 weakness is mechanism-intrinsic** — a bullish-reversal
long fails in strong-USD 2018 whichever low is swept (the breakdown *succeeds*, the trend continues),
re-confirming arc 3012's "2018 is mechanism-deep, not construction-shallow."

## Verdict: KILL (cheap-kill at observation)

The session-liquidity-level grab is not a cleaner/stronger edge than 1013's rolling swing low; it
dilutes and remains −2018. No PASS candidate; no improvement to 1013. arc-1013 component UNCHANGED
(still PORTFOLIO); no new `portfolio-candidates/` entry (would double-count). No pool/engine/null/
council spent (the level comparison is decisive at observation: the alternative levels have lower
capture, negative/weaker drift, fewer pairs positive — there is no better-version to escalate).

## Threads / lessons

1. **arc-1013's liquidity-pool level (rolling-40-bar swing low) is load-bearing and confirmed BEST** —
   a structural pivot beats time-based ICT session levels (prior-day/prior-week low), which dilute the
   grab (the prior-day reclaim has *negative* mean drift). The grab edge requires a *reversal pivot*,
   not merely *clustered stops*; "swept liquidity" without "structural support" is not a grab.
2. **The 2018 weakness is intrinsic to the bullish-reversal-long mechanism**, not the chosen level — all
   three levels are −2018 (re-confirms arc 3012: 2018's tradeable events are mean-reverting capitulations
   that succeed as breakdowns in a strong-USD trend). 1013 cannot be made +2018 by changing its
   liquidity level → the portfolio's 2018 wall is not addressable by re-leveling the existing best edge.
3. **1013 stays 9/10 IS, one fold from PASS, with that fold (2018) structurally immovable** within the
   reversal-long family — the standalone-PASS route via 1013 is closed; its value remains as the 3-way
   book's regime-orthogonal (2015/16/20) PORTFOLIO leg.
4. **Surviving frontier (unchanged):** operator-gated tighter-cost execution regime, or a genuinely
   non-price-direction construction not yet conceived; the in-apparatus directional/structural leads
   (incl. re-leveling the best edge) are exhausted.

## Tooling

No new BUILT tool — reused BUILT `observe_long_capture(direction="long")`; the per-level reclaim masks
(prior-day / prior-week low) are one-off scratch conditioners. The BUILT
`FailedBreakdownReclaimLongSignal` (1013) already parameterizes the swing-low version; no need to
generalize it to time-based levels (they are worse).

## FLAGS (code not merged)

None. No canonical-core change. Carries the standing `A1Config.time_exit_bars`-unwired flag + FLAG-1
(the 2018 leg). Driver `_disco3_work/arc3013_observe_liquidity_levels.py` (reproducible from this doc).
