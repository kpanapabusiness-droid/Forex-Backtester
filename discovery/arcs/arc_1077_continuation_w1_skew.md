# arc 1077 — positive-skew trend-CONTINUATION on the HONEST ENGINE at W1 (the last untested TIMEFRAME cell)

**Chat:** 1000s · **Range:** 1000–1999 · **Disposition: KILL** (all 8 cells KILL at IS; the only
positive folds are 2015-tail-luck shorts that die the pre-registered guard; OOS never touched).

Driver `discovery/_disco1_work/arc1077_continuation_w1_skew.py` (reuses
`discovery/tools/trend_continuation_signal.py::TrendContinuationBreakoutSignal` at `primary_tf="W1"` —
no new tool). Scored solely by `ArcFoldRunner → A1Architecture → MultiPairBacktester`, FundedNext costs
ON, take-the-loss; experiment side = entry signal + tail-removal/null arithmetic only.

## Log reading (step a — what's been tried, what's open)

Pulled main (up to date). Read `DISCOVERY_PROTOCOL.md`, the `LESSONS.md` 2026-06-06 operator
compression, `DISCOVERY_DIRECTION.md` (runs 1 & 2), `TOOL_REGISTRY.md`, recent Tier-2. No `discovery/STOP`.

The operator's 2026-06-06 redirection opened ONE in-charter thread: **positive-skew CONTINUATION** —
the corpus called trend-following dead using only +1R-capture and mean-forward-drift (win-rate lenses
blind to a low-capture / fat-right-tail payoff). It must be judged on **MEAN + median-per-fold +
TAIL-REMOVED** with **take-the-loss + a trailing/runner exit** (the positive-skew engine). Mandatory
pre-registered guard: **tail-luck ≠ skew** — mean-positive ONLY via the top-K winners ⇒ KILL.

That thread is now nearly closed by direct engine measurement under the mandated lens:
- **Entry GEOMETRY axis (all H4):** arc 1074 (Donchian breakout-in-trend) KILL; arc 1075
  (pullback-resume, favorable-first) KILL — favorable-first REFUTED; arc 2081 (continuation book +
  shock under trailing) KILL; arc 2082 (vol-expansion breakout) KILL; arc 2083 (favorable-first
  pullback, 28-pair) KILL.
- **TIMEFRAME axis:** arc 1076 closed **D1** (+ both universes, both dirs, §5f) — 7/8 cells
  mean-negative, the lone positive cell tail-luck, median trade takes the −1R stop. arc 1076 titled
  itself "the untested TIMEFRAME axis" but reached only D1.

**The one remaining cell of that axis is W1 (weekly).** It is genuinely untested and has the strongest
prior *for* a positive result: the documented CTA / managed-futures / TSMOM trend premium is strongest
at the weekly+ horizon, and arc 1076's own thesis — "a trailing runner can ride a multi-week move into
a fat right tail" — is **maximally** expressed at W1, where one winner can ride a multi-MONTH move (the
fattest possible right tail). If positive-skew continuation lives anywhere in-charter, W1 is its best
chance. Closed-ground's "directional coin-flip is timeframe-invariant" was measured under the CAPTURE
lens (arc 3004); the operator's whole point is capture is blind to skew — so TF-invariance under
capture does NOT pre-settle W1 under mean+tail-removed.

## Why this arc (the specific gap) + the *because*

Test arc-1074's EXACT continuation entry (`TrendContinuationBreakoutSignal`, both directions) at **W1**,
under the positive-skew runner exits and the mandated lens — the literal completion of arc 1076's
timeframe axis (H4 done → D1 done → **W1**).

**W1-appropriate params (documented because).** A 200-week SMA needs ~3.8 yr warmup and is INFEASIBLE on
the 11-yr IS window, so 1076's daily-Turtle SMA50/200 cannot be ported verbatim. Used the canonical
WEEKLY long-term trend filter **SMA10/40** (the 40-week MA ≈ 200-day MA, Weinstein stage analysis;
10-week ≈ 50-day) with Donchian **{10,20}-week** breakouts (the Turtle's 55-DAY ≈ 11-week intermediate
breakout range). Only TF + trend params change vs 1076; engine machinery / exits / guard / null identical.

**Falsifiable prediction.** If continuation failure on liquid majors is H4/D1-timeframe-specific (noise
dominates intraday), W1 should lift the mean toward 0/positive and grow a BROAD right tail that survives
+2R-cap and top-K removal. **Falsifier:** if at W1 the median trade still takes the −1R stop and the only
positive folds are single-outlier-carried, continuation is timeframe-invariant under the skew lens too →
airtight closure of the timeframe axis.

## Method

- Signal: `TrendContinuationBreakoutSignal(direction, lookback∈{10,20}, sma_fast=10, sma_slow=40,
  spacing_bars=4, primary_tf="W1")`. Ex-ante shift1 SMAs / Donchian / Wilder(14) MID ATR; entry i+1.
- Exits: runner only — `sl_plus_trailing_atr` / `sl_plus_trailing_swing` /
  `sl_partial_close_1r_runner_trail` × SL{1.5,2.0,2.5}, **§5f nested walk-forward selection**
  (`afp_then_mean`, select on strictly-earlier folds, score that fold). No tp_2r/3r (they cap the tail).
- Universes: USD7 (control vs 1074/1002/1076) AND ALL28 (trendy-cross steelman; maximizes W1 trade count).
- Cells: {long,short} × Donchian{20,10} × SMA10/40 = 4 per universe, 8 total. IS 2010–2020, 8 evaluable
  folds (2013–2020 after the §5f warmup). **OOS never touched.**
- Guard (pre-registered BEFORE the run, applied verbatim): **G1** mean per-fold ROI > 0 AND mean
  per-trade R > 0; **G2 (tail-removed)** +2R-cap mean > 0 AND drop-top-5%/fold mean > 0 AND
  drop-top-K{1,3} global mean > 0; **G3** per-fold median ROI > 0 in a majority of folds. + fair same-side
  random-entry null (`build_null_signal_evaluation`, warmup 45). A cell touches OOS ONLY if G1∧G2∧G3.

## Result — KILL in all 8 cells (decisive, not marginal)

| universe | cell | n_trades | mean ROI %/yr | mean R | median R | folds+ | +2R-cap | drop-5% | null |
|---|---|---|---|---|---|---|---|---|---|
| USD7 | long D20 | 38 | **−0.082** | −0.034 | −0.90 | 3/8 | FAIL | FAIL | loses |
| USD7 | short D20 | 28 | −0.316 | −0.181 | −0.97 | 3/8 | FAIL | FAIL | — |
| USD7 | long D10 | 26 | −0.397 | −0.244 | −0.48 | 3/8 | FAIL | FAIL | loses |
| USD7 | short D10 | 27 | −0.339 | −0.201 | −0.95 | 2/8 | FAIL | FAIL | loses |
| ALL28 | long D20 | 66 | −3.373 | −0.818 | −1.00 | 0/8 | FAIL | FAIL | loses |
| ALL28 | short D20 | 79 | −0.451 | −0.091 | −0.97 | 2/8 | FAIL | FAIL | beats* |
| ALL28 | long D10 | 105 | −2.703 | −0.412 | −0.96 | 2/8 | FAIL | FAIL | loses |
| ALL28 | short D10 | 94 | −0.628 | −0.107 | −0.98 | 4/8 | FAIL | FAIL | beats* |

\* the two "beats null" cells are net-NEGATIVE → **beats-null-but-net-negative = KILL (§11)**.

- **Median R ≈ −0.90 to −1.00 in 7/8 cells** — the median trade takes the −1R stop, the SAME
  adverse-first / near-martingale signature as H4 (1074/1075) and D1 (1076). Even at the weekly horizon,
  a Donchian-break-in-trend entry on liquid FX is coin-flip-to-adverse net of cost.
- **No broad skew anywhere.** The "least bad" cells (ALL28 short D20/D10) are carried ENTIRELY by the
  2015 fold (+8.05% / +6.10% — the SNB-day / risk-off CHF-cross shorts) and **die the guard the instant
  the tail is removed**: +2R-cap flips them to −1.70% / −1.54%, drop-top-5%/fold to −2.18% / −2.06%.
  Largest single winners +18.59R and +12.87R — textbook thin-tail luck (cf. arc 2082's +132R SNB-day
  CHF short, arcs 2011/2063). A handful of outlier trades carrying a flat-to-negative book = exactly the
  pre-registered "mean-positive ONLY via top-K winners → KILL."
- **2018 (the standing binding fold) negative in ALL 8 cells.** 0/8 cells AFP; not one survives G1.

## Verdict + meaning for the frontier

**KILL — and the timeframe axis is now airtight.** The "W1 amortizes noise / rides a multi-week move into
a fat tail" thesis did NOT survive the engine. The falsifier was hit: at W1, on the trendiest 28-pair
universe, with §5f-selected runner exits, the median trade still takes the −1R stop and the only positive
folds are single-outlier-carried 2015 shorts that the tail-removed guard erases. So the failure is NOT
H4/D1-timeframe-specificity — it is the root cause arc 1075 named (**liquid FX price has no exploitable
post-trend continuation**), now shown **timeframe-invariant under the positive-skew lens across the ENTIRE
in-charter timeframe range (H4 + D1 + W1)**, just as arc 3004/1002 showed it under the capture lens. A
trend-structural entry samples the near-martingale at any sampling interval; take-the-loss + a runner
cannot manufacture broad skew from a martingale, and the discrete tail that forms (here the 2015 CHF
shorts) is too thin to survive the guard. This is the conservation law (`frequency × edge ≈ const`,
LESSONS run-2) on the continuation axis, now closed on the last timeframe.

**Combined with arcs 1074/1075/1076/2081/2082/2083, positive-skew continuation is now comprehensively
closed across every in-charter axis: entry GEOMETRY (breakout / pullback-resume / favorable-first /
vol-expansion / shock / continuation-book) × TIMEFRAME (H4 / D1 / W1) × UNIVERSE (USD7 majors / 28-pair
crosses) × EXIT (all runner exits × SL × §5f) × DIRECTION (long / short)** — all under the
operator-mandated mean + median-per-fold + tail-removed lens on the honest engine. This was the operator's
ONE open in-charter thread (LESSONS 2026-06-06); it is now closed by direct engine measurement.

The in-charter frontier returns to the convergent terminus: 0 PASS, 4 PORTFOLIO, deployable = 0.
Components UNCHANGED. The standing levers remain out-of-band: the operator's path-A gate-governance call
and/or a charter unlock (`NEEDS_ENABLEMENT.md` #1 cross-asset trend — note this is the SAME positive-skew
shape on a less-efficient universe, where it is documented-fundable; the in-charter FX version is exactly
what these arcs just closed across all timeframes).

## Process notes

- No canonical change; no FLAG (apparatus behaved as designed). No council (decisive negative, not a
  survivor / idea-fork). No new BUILT tool (reused `TrendContinuationBreakoutSignal` at W1, confirmed
  TF-agnostic; the aggregator supports W1 = `W-MON` natively).
- Honest caveat: W1 folds are thinner than H4/D1 (38–105 pooled trades/cell over 8 folds; early folds
  sparse after the §5f warmup) — the verdict rests on the **pooled per-trade R** (median ≈ −0.9 to −1.0,
  robust to fold sparsity) and the tail-removal arithmetic, exactly as 1074/1076 handled sparse folds.
