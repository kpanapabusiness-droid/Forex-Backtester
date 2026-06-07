# arc_1064 — CONTINUATION / POSITIVE-SKEW shape family (the one untested LENS)

- **Chat:** 1000s
- **Date:** 2026-06-06
- **Type:** observation cheap-kill (§5d), council-surfaced forward thread (arc 1063 thread (b),
  logged-not-executed); BUILT tools only; no engine / null / council; no canonical change; no FLAG;
  OOS untouched.
- **Disposition:** **KILL**
- **Components UNCHANGED** (all 4 PORTFOLIO; me_long-solo the honest deploy object).

## Fresh-eyes (step a)

Pulled main (highest in-range arc 1063 → resume 1064); read `DISCOVERY_PROTOCOL`, `DISCOVERY_LOG`
(both tiers, all ranges, incl. disposition), `LESSONS`, `DISCOVERY_DIRECTION`, `NEEDS_ENABLEMENT`,
`TOOL_REGISTRY` (canonical + all BUILT); no `discovery/STOP`. Synthesis of the honest-era corpus:

- **4 PORTFOLIO components**, all forced-flow REVERSION: gap-fill (1006), me_long (1011), fbr (1013),
  me_short (1019). The 4-way book is mean-positive (~+0.59% RP, t=2.66, P(mean<0)≈0.004) but NEVER
  all-folds-positive; co-sim item E confirmed the failure is FUNDAMENTAL, blocked combination-invariantly
  by 2015 & 2018 (the strong-USD wall). Honest §5f exits weakened 3/4 legs (2042) → me_long-solo the
  honest deploy object, itself marginal/tail-fragile (1056/1057/1058).
- **The route's lever is the operator's path-A gate-governance call** (1023/2019/3021/1032): edge-hunt
  closed (path-B densification proven impossible at empirical correlation, 3021), MENU exhausted
  (M1/O1/L1/S1/Q1/G1 all dead), 5th-leg unfound across ~22 routes, shorts (1014/2009/2011/3011/1016/
  2013/3010/3012/1017) and relative-value (2003/2010/2018) all KILL.
- **The one fresh thread:** arc 1063 (POWER AUDIT of the all-folds judge) REFRAMED "frontier exhausted"
  — the 0/90 PASS rate is partly a property of an extremely strict ≈Sharpe-2/year consistency screen,
  not pure edge-absence — and logged forward TWO un-executed council threads: (a) the USD-factor
  sign-conflict crux; **(b) the continuation/positive-skew SHAPE family — the corpus tested ONLY
  reversion shapes, judged by +1R CAPTURE; a continuation edge survives take-the-loss differently
  (needs positive SKEW under a runner exit, not >0.50 capture), with a pre-registered guard against
  relabeling tail-luck as skew.** This arc executes thread (b).

## The lens-gap (the *because* — why this is genuinely untested, not a re-run)

Every honest-era entry was judged by (i) +1R-before-SL CAPTURE (a >0.50 win-rate test) or (ii)
fixed-horizon mean drift. BOTH structurally penalize a positive-skew continuation: a trend-follower's
real profile is **capture < 0.50** (loses small often — take-the-loss caps each miss at −1R) but
**positive expectancy via a few winners that RUN FAR** under a let-it-run trailing exit. Such an entry
is cheap-killed at the capture stage *before* its runner-exit expectancy is ever measured (arc 1062 is
the exact case: capture 0.45 < null → "never reached the exit sweep"). NOT closed ground:
- **≠ arc 2000** (unconditional Donchian + full-size trailing, killed at engine triage by WORST-FOLD
  ROI "fat tail generic not trend-selected") — this trend-FILTERS the breakout and applies the
  lens-corrected median + tail-removed guard 2000 never used.
- **≠ arc 2012** (deep continuation-long, killed by capture / structure-control inversion).
- **≠ arc 3019** (shock-continuation, a different post-extreme-shock shape; reached engine, OOS-epoch-died).
- **≠ arc 1062** (compression-expansion straddle, killed at capture — exactly the lens-gap).

## Method

CAUSAL trend-CONTINUATION entry (the textbook CTA positive-skew entry, BOTH directions now shorts are
enabled): at bar t, LONG = established uptrend (`mc>SMA200 & SMA50>SMA200`) AND a FRESH Donchian-N HIGH
break (`mc[t] > prior-N-bar high`, crossing bar); SHORT = the downtrend mirror. Entry next bar via the
BUILT `observe_long_capture` (direction-aware, honest take-the-loss). IS 2010-2020, H4, 7 USD majors.

**The lens correction:** `fwd_drift_atr` is GROSS (ignores the stop) → a GENEROUS upper-ish proxy for a
runner edge; measured at a LONG horizon (60–120 H4 bars ≈ 10–20 days) so a genuine runner is not
truncated. **Pre-registered guard** (thread (b)'s discipline): **G1** mean drift > 0 (the §5f trigger);
**G2** NOT a jackpot mirage — tail-removed mean (drop top 5%) > 0 AND per-year median>0 in a MAJORITY of
folds; **G3** distributed across a majority of the 7 pairs. KILL at obs if G1 fails (genuine coin-flip),
OR if G1 holds but G2/G3 fail (thin-tail artifact — capture lens was right). Engine §5f ONLY if all hold.
Levers swept (Donchian 20/40/55, trend 50/200 & 20/100, drift 30/60/120) for robustness, NOT optimization.
Driver `discovery/_disco1_work/arc1064_continuation_positive_skew.py` (single-use, BUILT-tools-only).

## Result — positive skew is REAL but a THIN-TAIL MIRAGE; the capture-lens kill was CORRECT (KILL)

| cell | n | capture | drift mean | drift median | skew | tail-removed mean | per-yr median>0 | binding 2015/2018 | per-pair >0 |
|---|---|---|---|---|---|---|---|---|---|
| **D20 trend50/200 drift60 (default)** | 5086 | 0.488 | −0.244 | −0.231 | −0.80 | **−0.926** | 3/11 | −0.54 / −0.52 | 0/7 |
| D40 drift60 | 3927 | 0.485 | −0.371 | −0.328 | −1.02 | −1.054 | — | — | — |
| D20 drift120 | 5086 | 0.488 | **+0.114** | −0.335 | +0.46 | **−1.012** | 4/11 | −0.84 / −0.48 | 4/7 |
| D20 drift30 | 5086 | 0.488 | −0.121 | −0.112 | −0.28 | −0.588 | 4/11 | −0.11 / −0.24 | 1/7 |
| D20 trend20/100 drift60 | 5797 | 0.477 | −0.301 | −0.195 | −0.78 | −0.974 | 5/11 | −0.71 / −0.55 | 0/7 |
| D55 trend50/200 drift120 | 3627 | 0.485 | **−0.018** | −0.444 | +0.42 | −1.141 | 3/11 | −1.08 / −0.48 | 5/7 |

1. **Capture ≈ 0.477–0.488 (≈ unconditional base 0.488) in EVERY cell** — the trend filter adds NOTHING
   to +1R capture; continuation is coin-flip on that lens (closed-ground confirmation). The capture-lens
   would kill all six, as the premise predicts.
2. **Positive skew IS real — but ONLY at long runner horizons** (drift120: skew +0.46 / +0.42;
   frac>2ATR ≈ 0.38 — winners genuinely run). At shorter horizons skew is NEGATIVE (the moves fail fast).
   So thread (b)'s premise (continuation HAS positive skew the capture lens can't see) is *confirmed at
   the distribution level*.
3. **But the GUARD decisively FAILS in EVERY cell:**
   - **median drift NEGATIVE everywhere** (−0.11 … −0.44) → the typical trade loses;
   - **tail-removed mean STRONGLY NEGATIVE everywhere** (−0.59 … −1.14) → drop the top 5% and the edge
     vanishes; the ONLY positive mean (D20 drift120 **+0.114**) is **entirely** the top-5% jackpot tail;
   - per-year median>0 only 3–5/11 (never a majority, G2 fails); 6–8/11 neg-mean-years;
   - **G3** positive-per-pair is the fat-tail cells only (0/7 in the median-honest cells).
4. **Binding folds 2015 & 2018 NEGATIVE in every cell** — the regime-orthogonal hope (continuation pays
   in strong-trend years) also fails: under take-the-loss even strong-trend years' pullbacks stop-run the
   entries to −1R before the runner develops (the arc-1060/1062 lesson, now on the continuation axis).

`fwd_drift` is GROSS (no take-the-loss); the honest engine (SL-first, which stops many would-be runners
at −1R before they run) is **strictly worse** → no engine run warranted. §5f does not bite: the lone
G1-passing cell passes G1 ONLY via the top-5% tail that G2 explicitly disqualifies (median −0.33,
tail-removed −1.01). §5d obs cheap-kill.

## Diagnosis + meaning

The textbook trend-continuation entry, judged by the CORRECT lens for a skew edge, is a **thin-tail
mirage**: positive skew exists (winners run > 2 ATR ~38% of the time) but the expectancy is carried
ENTIRELY by the top-5% of trades — the median trade loses, tail-removed expectancy is strongly negative,
the majority of folds are negative, and the binding folds are negative. This is precisely arc-2000's
"fat tail generic, not trend-selected," now confirmed on the EXPLICIT skew axis with the pre-registered
guard. **The capture-lens cheap-kill of continuation entries was CORRECT** — the two lenses (capture and
expectancy+guard) AGREE. Continuation on liquid FX majors does not become a deployable edge by switching
to a runner exit / skew lens; the apparent positive mean is jackpot variance, not a distributed edge.

## NEW lesson

**A positive-skew CONTINUATION edge on liquid FX majors is a thin-tail mirage — the capture lens was
right.** The corpus's continuation kills (2000/2012/3019/1062) are NOT an artifact of judging by +1R
capture: when you judge the textbook trend-continuation entry by the *skew-appropriate* lens (long-horizon
forward-return distribution: mean, median, tail-removed expectancy, per-pair/per-fold), the positive skew
that genuinely exists (winners do run) is **undistributed jackpot variance** — top-5% removal flips the
mean strongly negative and the median trade loses. This **CLOSES the last open SHAPE-family lens-gap arc
1063 flagged** and REMOVES the "maybe the capture lens was wrong for skew edges" caveat from the
frontier-exhausted reframe: the capture and expectancy+guard lenses converge on KILL. Reusable guard for
ANY future skew/continuation candidate: require per-fold median > 0 AND tail-removed (drop top 5%)
expectancy > 0 BEFORE believing a positive mean — a positive mean alone, on a skewed distribution, is the
jackpot tell, not an edge.

## Threads / handoff

Components UNCHANGED (all 4 PORTFOLIO; me_long-solo the honest deploy object). Lever = **operator path-A**
(gate-governance on the sound mean-positive ~3-bet reversion book — 1023/2021/3021/1032) OR a charter
unlock (operator-gated macro/options/COT data, `NEEDS_ENABLEMENT.md`). This arc executes arc-1063 forward
thread (b) and closes it: the within-charter terminus now holds on the reversion shapes (capture lens) AND
the continuation/positive-skew shapes (expectancy+guard lens) — the two-lens convergence means the
"frontier exhausted" verdict is no longer caveated by the lens-gap. Remaining un-executed 1063 thread:
(a) the USD-factor correlation-vs-sign crux (cheap panel query — are me_long's 2018 losing bars
USD-factor-wide, or is 2015/2018 an irreducible sign conflict). No new BUILT tool registered (pure
`observe_long_capture` + a causal trend-continuation mask + `scipy.stats.skew` and the inline
median/tail-removed guard — single-use, mirroring arc 1062). No engine / null / council; OOS untouched.
