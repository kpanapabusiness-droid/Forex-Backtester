# arc 1075 — positive-skew PULLBACK-RESUME continuation on the HONEST ENGINE (the last entry geometry)

**Chat:** 1000s · **Date:** 2026-06-07 · **TF/universe:** H4, 7 USD majors · **Window:** IS 2010–2020
(OOS 2021+ FROZEN, never touched). **Disposition: KILL** (all 4 cells; decisive engine negative).

## Step (a)/(b) — idea + the *because* (built on arc 1074's diagnosis)

Arc 1074 spent the honest engine on the Donchian-breakout-in-trend continuation entry and found it
decisively negative (mean per-trade R ≈ −0.5, median ≈ −0.9), diagnosing the cause as **adverse-first
geometry**: a breakout buys the exact channel high, price retraces through the −1R stop on the majority,
take-the-loss fires before any runner develops. The mechanistic complement
([`DISCOVERY_DIRECTION.md`](../DISCOVERY_DIRECTION.md) S1 take-the-loss-tax geometry; `fbr` is its reversion
instance): enter the continuation **after** the pullback, on the **resumption** bar, so the path is
**favorable-first** and the −1R tax falls on noise trades, not trend trades. This is the last continuation
entry GEOMETRY not yet run on the engine + positive-skew (mean + tail-removed) lens. *Falsifiable
prediction:* if "adverse-first" is the cause, a favorable-first resumption entry should lift mean R toward
0/positive and produce a right tail. *Falsifier:* if it is still −R with no tail, the problem is not
geometry — there is simply no continuation edge.

New BUILT tool `TrendPullbackResumeSignal` (dual-SMA trend + a recent dip into the fast MA + a resumption
bar = close back above prior-bar high & fast MA; long AND short; ex-ante shift1). Same engine machinery as
arc 1074 (runner exits only — `sl_plus_trailing_atr`/`_swing`/partial-runner × SL{1.5,2,2.5}, §5f nested),
same **pre-registered tail-luck≠skew guard** (G1 mean>0 & mean-R>0; G2 +2R-cap & drop-top-5%/fold &
drop-top-K all >0; G3 per-fold median>0 majority; beat fair same-side null).

## Results — KILL in all 4 cells (8 evaluable folds 2013–2020); OOS NEVER touched

| cell | n_trades | MEAN per-fold ROI | folds+ | mean R | median R | +2R-cap | drop-top5% | max winner R | vs null |
|---|---|---|---|---|---|---|---|---|---|
| long  sma20/100 pb10 | 237 | **−9.65%/yr** | 0/8 | −0.651 | −0.882 | −9.65% | −9.58% | **−0.04R** | LOSES −5.05pp |
| short sma20/100 pb10 | 228 | **−5.42%/yr** | 0/8 | −0.380 | −0.778 | −5.47% | −5.97% | +2.81R | LOSES −0.12pp |
| long  sma50/200 pb15 | 257 | **−10.93%/yr** | 0/8 | −0.681 | −0.897 | −10.93% | −10.83% | **−0.03R** | beats +1.37pp (still −11%) |
| short sma50/200 pb15 | 310 | **−6.79%/yr** | 0/8 | −0.350 | −0.895 | −6.87% | −7.95% | +2.55R | LOSES −0.64pp |

- **G1/G2/G3 FAIL everywhere.** Mean per-fold ROI −5.4 to −10.9%/yr; mean per-trade R −0.35 to −0.68;
  median R ≈ −0.78 to −0.90 (a majority of trades take the −1R stop). 0/8 folds positive; binding
  2015/2018 negative.
- **The hypothesis is REFUTED.** Favorable-first geometry did NOT help — the long cells are *worse* than
  arc-1074's breakout (R −0.65/−0.68 vs −0.52) and have **no right tail at all** (largest single winning
  position ≈ 0R), so tail-removal is moot. Entering after the pullback does not make the path
  favorable-first on liquid majors: the "resumption" immediately fails just as often as the breakout.
- Loses to the fair same-side null in 3/4 cells; the one "beats null" cell is −11%/yr (beats-null-but-
  net-negative = KILL §11).

## Why it fails — the corrected mechanism

The root cause is NOT *where* you enter the trend (edge vs after-pullback); it is that **liquid-major price
has no exploitable post-trend continuation**. Whether you buy the breakout (1074) or the resumption (1075),
the next move is coin-flip-to-adverse, the −1R take-the-loss fires on the median trade, and no right tail
develops to pay for the stop wall. This is the conservation law (`frequency × edge ≈ const`, LESSONS run-2)
restated on the continuation axis: a trend-structural entry samples the near-martingale, so it is a
coin-flip net of cost — and take-the-loss + a runner exit cannot manufacture skew from a martingale.

## Verdict — KILL (no new component; the positive-skew continuation frontier is now airtight-closed)

Combined with arc 1074 (Donchian breakout edge), arc 2081 (continuation book + shock re-measured), and arc
2082 (vol-expansion breakout) — all KILL under the operator-mandated mean + median-per-fold + tail-removed
lens on the honest engine — **positive-skew continuation is closed across every entry geometry**
(breakout-edge, vol-expansion, shock, continuation-book, and now favorable-first pullback-resume). This was
the operator's ONE open in-charter thread (LESSONS 2026-06-06); it is now comprehensively closed by direct
engine measurement. Components UNCHANGED (4 PORTFOLIO; me_long-solo the honest deploy object; deployable=0).
No council (decisive negative, not a survivor/fork). No canonical change; no FLAG. New BUILT tool registered:
`TrendPullbackResumeSignal`. Driver `discovery/_disco1_work/arc1075_pullback_resume_skew.py`.
