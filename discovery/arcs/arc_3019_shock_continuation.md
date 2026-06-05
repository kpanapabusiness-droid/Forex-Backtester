# arc 3019 — Forward-confirmed extreme-shock CONTINUATION (the candidate 5th portfolio leg)

**Chat:** 3000s · **Date:** 2026-06-05 · **Verdict:** FAIL (OOS) → **KILL**
**Disposition:** KILL · **passed:** N · **Council:** HEAVY (convened at the disposition fork; verdict
"don't record on IS — run the cheap event-removal test, then OOS"; both run, OOS killed it)

> The corpus's STRONGEST continuation result and the most promising 5th-leg candidate ever found —
> honest-engine IS 9/10, beats the fair null, positive in BOTH 2015 AND 2016 (the exact 4-way book
> blockers), mechanistically clean (magnitude-monotone capture, forward-confirmation load-bearing),
> survives mega-event-window removal. **The one-shot frozen-exit OOS (2021-2026) KILLED it: 2/6 folds,
> mean −0.0022, LOSES to the random-entry null.** The shock→continuation directionality is
> EPOCH-DEPENDENT — 2010-2020 shocks continued; 2021+ shocks mean-revert. A textbook Arc-10 save:
> a strikingly strong IS edge that OOS reveals as in-sample-specific. Recording it PORTFOLIO on the IS
> evidence (my initial lean) would have been a false positive; the council's "measure OOS first" was
> vindicated.

---

## Log reading (step a — FRESH EYES, honest-era only; pulled main, no STOP)

Resumed 3000s; just closed my own arc 3018 (fbr-on-crosses KILL). Pulled main — arc 1021 landed (1000s):
broad-universe me-short looked like a deployable book but was an **Arc-10 near-miss** (partial-runner
fat-tail × IS-weight-overfit), KILLED; new rule = require all-folds-positive under a NON-fat-tail exit AND
non-optimized weights. 1021 also **sharpened the residual frontier to 2016**: the 4-way book (arc 1020,
~0.11% from deployable) is blocked by marginal 2015 & 2016; 2016 is "negative in every reversion leg under
every exit; positive only in fbr & gap, which 2018 caps; Brexit/election = a genuine non-reversion regime."
me-short (1019) supplied the 2018 leg. So the precise live need: a **+2015/+2016, non-(−2018)** component.

State: directional/flow/calendar/structure/vol/relative-value space comprehensively mapped (~56 arcs);
the in-apparatus well is "nearly dry" per arc 3016 — BUT me-short was found after that pessimism, so it
isn't fully dry. My arc 3018 closed cross-`fbr`; the only mechanistic complement left for 2015/2016 (both
extreme-EVENT years where reversions bled because moves CONTINUED) is a CONTINUATION leg.

## Idea + why (a documented *because*; the genuinely-novel continuation mechanism)

2015 (SNB, China-deval) and 2016 (oil-panic, Brexit, Trump) were extreme-EVENT years where the book's
reversion legs bled because the big moves CONTINUED, not reverted. The missing leg must be a CONTINUATION
firing in exactly those years. But generic momentum/breakout/trend is closed ground (coin-flip) and the
i+1-entry-after-a-big-move death (backward-confirming → catches the bounce) killed every prior shock trade
(1016/2009/3012).

**Mechanism (*because*):** an EXTREME single-bar shock (≥3 ATR close-to-close body, a ~tail event =
FORCED flow: margin liquidations / stop cascades / CB action) creates persistent multi-day one-way
pressure as leveraged players unwind — UNLIKE an ordinary 1-1.5 ATR vol spike (which reverts, arc 3012).
Harvest it only when **FORWARD-CONFIRMED** (the bar after the shock breaks the shock bar's extreme in the
shock direction) — entering on the RESUMPTION, the fix to the i+1-bounce death, exactly as arc 1013's
reclaim fixed gap-fill's capturability wall. Either direction (shorts open).

## Method (CALLED canonical; built one EXPERIMENT signal tool)

Built `discovery/tools/shock_continuation_signals.py::ShockContinuationSignal` (mask + ATR + per-pair
Direction only; ex-ante: shock body & ATR known at i-1, confirmation at i, enter i+1; conforms to the
LOCKED `SignalModule`). Direction is per-pair-fixed → ran SHORT (down-shocks) + LONG (up-shocks)
separately and SUMMED per-fold ROIs (disjoint event timing). Observation via canonical
`observe_long_capture`; §5f via `build_arc_pool`/`ArcFoldRunner`/`build_v3_folds`/`run_config_over_folds`;
fair direction-aware null via `build_null_signal_evaluation`; OOS via `build_oos_year_folds`. 7 USD majors,
H4. Risk convention per arc-3017 FLAG (judge signs/null in the LINEAR low-risk regime; report
risk-sensitivity). Drivers `_disco3_work/arc3019_*.py` (reproducible).

## What happened — IS is the strongest continuation result in the corpus

**Observation (gross capture, IS).** MAGNITUDE-MONOTONE — the edge lives ONLY in the extreme tail:
capture 0.498 @1.0 ATR (coin-flip = generic momentum dead, closed ground) → 0.528 @2.0 → 0.549 @2.5 →
**0.589 @3.0** → 0.568 @3.5. Forward-confirmation LOAD-BEARING (confirmed 0.589 > unconfirmed 0.543 > base
0.488). Per-pair robust (all 7 majors >0.50). 2016 positive across all 7 pairs.

**Honest engine §5f (3.0 ATR, both directions combined, IS folds, LINEAR regime, cap ~0 breaches).**
EVERY exit mean-positive and beats the fair null; 2015 & 2016 positive under EVERY exit; 2018 the only
consistently weak fold. The non-fat-tail TP exits are the BEST (passes the arc-1021 rule):

| exit | mean | npos | 2015 | 2016 | 2018 | real−null (real vs null npos) |
|---|---|---|---|---|---|---|
| sl_only | +0.015 | 4/10 | +0.040 | +0.003 | −0.035 | +0.046 (4 vs 3) |
| trailing_atr | +0.025 | 7/10 | +0.081 | +0.121 | −0.006 | +0.046 (7 vs 4) |
| trailing_swing | +0.017 | 6/10 | +0.043 | +0.022 | −0.032 | +0.036 (6 vs 2) |
| **tp_2r** | +0.030 | 9/10 | +0.072 | +0.088 | −0.011 | **+0.051 (9 vs 1)** |
| **tp_3r** | +0.034 | 9/10 | +0.037 | +0.109 | +0.012 | **+0.048 (9 vs 3)** |
| partial_runner | +0.020 | 7/10 | +0.060 | +0.092 | −0.007 | +0.040 (7 vs 2) |

It is the exact MIRROR of me-short (1019): me-short +2018/weak-2015-2016 ; shock-continuation
+2015/+2016/weak-2018 — and 2015 & 2016 are precisely the book's residual blockers. This looked like the
route-unblocking 5th leg.

## Why I did NOT record it on the IS evidence — the council + two decisive tests

**HEAVY council (`/llm-council-discovery`).** All 5 reviewers independently ranked the Soundness lens
strongest. The chairman verdict: **do NOT record on IS — the IS edge is selection-suspect (15 routes died
against the same 2015/2016/2018 target; "+ exactly where the book needed it" is the selection criterion
made visible), the sample is thin (n=287 ≈ 26/yr ≈ 14/fold), and the 3.0-ATR peak is a post-hoc threshold
pick. Run the cheapest decisive test first (strip the SNB-2015 / Brexit-Trump-2016 event-days — does
anything remain?), then OOS, before any recording.** Strongest dissent: epoch-dependence — post-2020
shocks (COVID, Fed-hiking, LDI) may MEAN-REVERT not continue, inverting the core causal claim. I committed
to the council (heavy weight; the reasoning was stronger than my initial PORTFOLIO lean).

**Test 1 — event-removal (refutes the naive coincidence, confirms thinness).** Dropping the labeled
mega-event windows (SNB/CNY/Brexit/election) leaves BOTH years positive (2015 drift +0.188, **2016
+1.204**) — so it is NOT "just long SNB+Brexit+Trump"; the big days are a VARIETY of macro shocks (FOMC
Dec-2016 +20.9, oil-panic Feb-2016 +18.0, NFP). BUT 2015 is THIN/fragile (drops to −0.049 removing its
top-2 of 22 days); 2016 is robust (survives top-2-day removal at +0.528). A real-but-thin-and-clustered
tail edge: 2016 robust, 2015 fragile.

**Test 2 — one-shot frozen-exit OOS (the decisive gate; §5f-faithful: exit tp_3r + thr 3.0 SELECTED on IS,
FROZEN, scored on 2021+, never re-selected). VERDICT: FAIL.**

| OOS year (linear) | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | mean | npos |
|---|---|---|---|---|---|---|---|---|
| REAL combined | +0.075 | +0.029 | −0.063 | −0.008 | −0.028 | −0.018 | **−0.0022** | **2/6** |
| null combined | +0.011 | −0.048 | −0.034 | +0.033 | +0.084 | −0.028 | +0.0031 | 3/6 |

**OOS real−null margin = −0.0052 — the real signal LOSES to the random-entry null out-of-sample.**
In-sample it was 9/10 and beat null by +0.048; out-of-sample it is 2/6 and below null. The shock→
continuation directionality did NOT persist post-2020 (2022 Fed-hiking/LDI shocks continued → +0.029;
2023-2025 post-shock mean-reverted → negative). The council's epoch-dependence dissent is confirmed.

## Diagnosis — a real but EPOCH-SPECIFIC (and cap-clustered) tail edge

The IS edge was real (mechanistically clean, monotone, event-removal-surviving) but **epoch-specific**: it
captured the 2010-2020 shock regime (where forced-flow shocks trended for days, esp. 2016) and does not
generalize to the 2021+ regime (where shocks mean-revert faster — faster liquidity provision / different
positioning). It is ALSO un-scalable: extreme shocks CLUSTER across pairs on macro-event days, so the
FundedNext 5%-daily-DD cap binds catastrophically at deployment risk (combined breaches 0 → 7-84/yr from
risk 0.005 → 0.5; signs flip, fold dispersion +10.9/−5.5 from cap-truncated monster days) — the daily-DD-cap
analog of arc-1017's currency-cap un-scalability. Either failure alone is disqualifying; OOS is decisive.

## Verdict: KILL

Strong IS edge (the corpus's best continuation result) but **fails the one-shot OOS and loses to the null
OOS** → §11 net-negative-OOS → KILL (not PORTFOLIO). OOS spent honestly (one-shot, frozen config, no
re-selection); nothing to preserve (the component is dead). The route's 2015/2016 blockers stand. No
component recorded.

## Threads / lessons

1. **Forward-confirmed extreme-shock continuation is a REAL but EPOCH-DEPENDENT edge.** It is the FIRST
   continuation mechanism in the corpus to clear capture 0.50 and beat the null on IS (magnitude-monotone
   0.498→0.589, forward-confirm load-bearing — generic momentum at 1.0 ATR is the coin-flip that the
   closed-ground prior describes; the edge is genuinely in the ≥3-ATR tail). **But it does not survive
   OOS** — 2010-2020 shocks trended, 2021+ shocks mean-revert. **New lesson: a continuation edge can be
   regime/epoch-specific even with a clean mechanism and a clean IS null-beat; the directionality of
   "shocks" is not a constant of the market.** This is distinct from the prior continuation kills (those
   were coin-flip on IS; this one PASSED IS and died OOS — a higher-value, harder-won negative).
2. **Arc-10 defense vindicated — twice over.** (a) The HEAVY council correctly refused to let me record on
   the IS evidence and named the exact decisive test (OOS / epoch-dependence). (b) The one-shot OOS caught
   an IS result that would otherwise have been the corpus's headline "route-unblocking 5th leg." Had I
   recorded PORTFOLIO on the 9/10-beats-null IS numbers (my initial lean), it would have been a fabricated
   component. **Re-usable rule confirmed: a continuation/tail component must clear OOS before PORTFOLIO
   recording — IS-mean-positive + beats-null is necessary, NOT sufficient, for an epoch-vulnerable edge.**
3. **The event-removal test is a cheap, high-value soundness instrument** (the council's idea): strip the
   named mega-event windows and re-run — it cleanly separated "fold-targeting coincidence" (refuted here)
   from "thin/concentrated" (confirmed for 2015). Worth reusing on any event-year-concentrated edge.
4. **Cap-clustering is a second, independent disqualifier** (would have mattered even if OOS held): an edge
   whose fires CLUSTER across pairs on macro-event days breaches the daily-DD cap at deployment risk — the
   daily-DD analog of arc-1017's currency-cap un-scalability. Tail/shock strategies are structurally
   cap-hostile on FundedNext.
5. **5th-leg status (unchanged):** the +2015/+2016/non-(−2018) component remains unfound; the strongest
   candidate yet (this one) died OOS. The route's honest state is arc-1020's: narrow 4-way blocked at
   2015 & 2016, worst −0.11%. The honest read (arcs 3016/3018, reaffirmed): the in-apparatus FX-major
   OHLCV well is nearly dry for this bi-fold spec; the residual likely needs an operator lever
   (tighter-cost execution regime / non-OHLCV data) — OR a +2015/+2016 mechanism that is NOT epoch-vulnerable
   (calendar/flow-mechanical like me, not regime/event-driven like this).

## Tooling

NEW BUILT tool: `ShockContinuationSignal` (`discovery/tools/shock_continuation_signals.py`) — registered in
`TOOL_REGISTRY.md`. Kept despite KILL (valid reusable forward-confirmed shock-continuation signal,
direction-aware; like `WeekendUpGapShortSignal` which was also KILL-but-kept). Scratch drivers
`_disco3_work/arc3019_*.py` (observe / wfo / event-removal / oos — reproducible).

## FLAGS (code not merged)

None requiring the canonical core. Carries the standing arc-3017 `risk_pct` PERCENT-vs-FRACTION FLAG
(used the linear regime + reported risk-sensitivity, as 3017 recommends) and the arc-1005
`time_exit_bars`-unwired flag. The cap-clustering finding REINFORCES the arc-3017 daily-DD-cap-nonlinearity
flag (here the nonlinearity is severe — risk-convention changes the disposition's magnitude, though OOS
kills it at every risk).
