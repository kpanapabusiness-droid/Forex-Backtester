# arc 1022 — Failed-breakdown RECLAIM long across UNIVERSES (JPY / non-USD crosses)

**Chat:** 1000s · **Date:** 2026-06-05 · **Verdict:** FAIL (cheap-kill at observation) → **KILL**
**Disposition:** KILL · **passed:** N · **Component touched:** arc-1013 (UNCHANGED, still PORTFOLIO)

> Does the corpus's strongest edge — the failed-breakdown stop-run-reclaim reversal (arc 1013) — extend
> to a JPY-cross or non-USD-major universe, where 2018's strong-USD wipeout (arc 2014) need not apply,
> yielding a regime-complementary +2018 (and ideally +2016) 5th PORTFOLIO leg? **No.** The reclaim's
> structure is load-bearing **only on USD majors**; off them it dilutes to a coin-flip. The universe
> lever does not transfer the edge — fbr's edge is tied to USD-major stop-liquidity, not a universal
> stop-run property.

---

## Log reading (step a — FRESH EYES, honest-era only)

Resumed 1000s at arc 1022 (prior in-range = 1021; pulled main). No `discovery/STOP`. State: four
net-positive PORTFOLIO components — gap-fill (1006, JPY-cross H4), month-end-long (1011, USD-major D1),
failed-breakdown-reclaim (1013, USD-major H4, the strongest 9/10), month-end-SHORT (1019, USD-major D1,
the first robustly **+2018** leg). The 4-way book (arc 1020, indep. 2015) **breached the 2018 wall**
(worst fold −0.115%, strongest in corpus) but is still **0/all convex weightings all-folds-positive**,
now blocked by **2015 & 2016**. Arc 1021 broadened me-short → Arc-10 near-miss (partial-runner fat-tail
× IS-weight overfit), KILL; **2016 is robustly negative in every reversion leg under every exit**. The
sharp open spec: a 5th component **+2016 (ideally +2015), non-reversion, without dragging 2018**.

## Idea + why (the universe lever on the corpus's best edge)

Chasing a 2016-event-specific patch (Brexit/election) is overfit — the OOS holdout has no Brexit. The
honest framing: a 5th component with a **generalizable mechanism** whose fold profile relieves 2016/2018
for *structural* reasons that recur OOS. Key structural fact (arc 1020): **2016 is strongly positive in
fbr (+2.55) & gap (+3.20)**; the block is the *convex tension* (weighting fbr/gap up to carry 2016
breaks 2018, where fbr is wiped). So the ideal 5th leg is **+2018 AND +2016** so 2016 needn't lean on the
2018-toxic legs.

fbr is +2016 but −2018 on USD majors because in strong-USD 2018 the failed breakdown becomes a *real*
breakdown — the reclaim doesn't hold (arc 2014: 18/19 trades −1R; arc 3013: 2018 immovable across all
levels). **Hypothesis:** that 2018-wipeout is *USD-major-specific*. On JPY / non-USD crosses there is no
persistent USD trend in 2018; risk-off spikes are mean-reverting, so the stop-run-reclaim could survive
2018 there → a decorrelated fbr-class +2018 leg. The "universe lever" is proven productive (gap-fill is
mean-POSITIVE on JPY crosses, NEGATIVE on majors — arc 1006/2001).

## Method (CALLED canonical; observation only, IS only)

`Panel.from_pairs` (H4, cached) on three universes, fbr fire mask per the BUILT
`FailedBreakdownReclaimLongSignal` logic (K=40 swing low, deep lower shadow ≥1.25 ATR, reclaim above the
swept level — arc-1013 spec), honest long capture + 12-bar drift via BUILT `observe_long_capture`. The
**arc-1013 load-bearing STRUCTURE CONTROL**: same deep wick (shadow ≥1.25) but NOT at a swept swing low
("elsewhere") — AT-swept must beat elsewhere or the swing-low pivot is not load-bearing. All restricted
to IS (2010–2020); the binding portfolio folds 2016 & 2018 are both IS years → OOS untouched.
CHARACTERIZATION ONLY (gross, not a gate). Driver: `_disco_work/arc1022_obs_universe.py`.

## What happened — the edge is USD-major-specific; the universe lever DILUTES it

| universe | n | cap | structure excess (cap) | drift excess (ATR) | 2016 cap/drift | 2018 cap/drift | per-pair >0.50 |
|---|---|---|---|---|---|---|---|
| **MAJORS (control)** | 237 | **0.5865** | **+0.108** | +0.321 | 0.571 / −0.519 | 0.474 / −0.729 | **7/7** |
| JPY-crosses | 212 | 0.4811 | **−0.001** | +0.117 | 0.360 / −0.116 | 0.389 / −0.211 | 3/6 |
| non-USD-crosses | 309 | 0.4951 | +0.035 | +0.060 | 0.500 / +0.251 | 0.515 / −0.525 | 4/9 |

- **MAJORS reproduce arc 1013 exactly** — cap 0.587 (≈1013's 0.55–0.61), structure excess +0.108
  (load-bearing), 7/7 pairs >0.50, 2018 cap 0.474 (the known wipeout). Apparatus + edge reproduce.
- **JPY-crosses: the structure is NOT load-bearing** — capture 0.481 (<coin-flip), structure excess
  **−0.001** (AT-swept ≈ deep-wick-elsewhere), 2018 cap 0.389 (WORSE than majors). The swing-low
  location carries no information here.
- **non-USD-crosses: coin-flip, weak structure** — cap 0.495, structure excess only +0.035 (vs majors
  +0.108). 2018 cap 0.515 looks marginally positive but sits inside an **overall sub-0.50, weak-structure
  universe** = regime-luck within a coin-flip (the arc-3010/3012/2013 tell: a single +year inside an
  overall coin-flip with negative drift will not survive SL-honest take-the-loss). 4/9 pairs >0.50 is
  consistent with noise around 0.495 — carving the >0.50 pairs (GBPCHF/AUDCAD/EURAUD) would be
  outcome-aware filtering (forbidden, §5; the arc-2009/3012 per-pair-noise tell).

## Diagnosis — fbr is a USD-major stop-liquidity phenomenon, not a universal stop-run

The stop-run-reclaim reversal requires **deep, clustered resting stops below a structurally significant
swing low**. USD majors carry the deepest, most-coordinated stop liquidity (highest participation,
clearest technical levels) → the sweep-and-reclaim is a genuine, load-bearing reversal (excess +0.108).
On crosses the stop liquidity below swing lows is thinner / less coordinated, so the sweep is not a
reliable reversal trigger — the swing-low location stops being load-bearing (JPY excess −0.001) and the
mechanism collapses to a coin-flip. This is the *same lesson as arc 3013* (the LEVEL must be a structural
pivot, not merely clustered stops) extended to the UNIVERSE axis: the edge needs both clustered stops AND
the deep, coordinated stop book that only USD majors provide.

The hypothesis (crosses avoid the 2018 wipeout) is moot: crosses don't carry the edge in ANY year, so
there is no +2018 fbr-class leg to harvest. The 2018-wipeout on majors is not "compensated" by a cross
universe — the cross universe simply has no fbr edge.

## Verdict: KILL (cheap-kill at observation)

fbr does not extend to a non-USD-major universe; the universe lever dilutes the stop-run-reclaim to a
coin-flip (cap 0.481 / 0.495, structure excess −0.001 / +0.035 vs majors +0.108). No +2018 (or +2016)
regime-complementary 5th leg is available this way. arc-1013 component UNCHANGED (still PORTFOLIO); no
new `portfolio-candidates/` entry. **No pool / engine / null / council spent** — the decision is
decisive at observation (coin-flip capture + non-load-bearing structure on both cross universes; §5f
does not bite on a coin-flip entry, like arcs 1014/2009/2011/3010/3013).

## Threads / lessons

1. **fbr's edge is USD-major-SPECIFIC — tied to deep, coordinated stop liquidity, not a universal
   stop-run property.** The swing-low structure is load-bearing only on USD majors (excess +0.108) and
   decays to a coin-flip on JPY crosses (−0.001) and non-USD crosses (+0.035). Re-confirms arc 3013's
   "clustered stops ≠ grab without structural support" on the universe axis: the grab needs BOTH a
   structural pivot AND the dense USD-major stop book.
2. **The universe lever does not transfer a STRUCTURAL edge the way it transfers a FLOW edge.** Gap-fill
   (a forced-flow reversion) moved cleanly from majors→JPY crosses (1006); fbr (a stop-liquidity
   structural reversal) does NOT — flow follows the instrument, structural stop-liquidity follows the
   most-participated instrument (majors). A heuristic for future universe-lever arcs: flow edges port,
   stop-structure edges may not.
3. **The fbr-on-crosses route to the +2018 leg is CLOSED** — adding to the long list of dead 2018 routes
   (structure 1014/2009/2011/3011, trend 3010, flow 1016/2013, vol 3012, rel-value 2010,
   continuation-long 2012, carry 1017, re-leveling 3013, downtrend-gate 2014). The 4-way book's residual
   2015/2016 block is NOT addressable by re-universing the existing best edge; 1013 stays 9/10 IS with
   2018 structurally immovable in the reversal-long family.
4. **Surviving frontier (unchanged):** a genuinely novel non-price-direction mechanism, or the
   operator-gated tighter-cost execution regime. The in-apparatus directional/structural/universe leads
   on the existing edges are exhausted.

## Tooling

No new BUILT tool — reused BUILT `observe_long_capture(direction="long")` + the BUILT
`FailedBreakdownReclaimLongSignal` (1013) fire logic (replicated inline as a per-pair restrict mask, same
K=40/shadow≥1.25 spec) and a structure-control "deep-wick-elsewhere" conditioner (one-off scratch).
Driver `_disco_work/arc1022_obs_universe.py` (reproducible from this doc).

## FLAGS (code not merged)

None. No canonical-core change. Carries the standing `A1Config.time_exit_bars`-unwired flag + FLAG-1
(the 2018→2015/2016 leg). OOS never touched.
