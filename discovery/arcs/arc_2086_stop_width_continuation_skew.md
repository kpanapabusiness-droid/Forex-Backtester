# arc 2086 — STOP-WIDTH (SL-multiple) sweep: the last un-varied axis of positive-skew continuation

> **Arc id:** 2086 · **Chat:** 2000–2999 (continuous) · **Date:** 2026-06-07
> **Type:** EDGE-HUNT on the operator-redirected frontier (positive-skew / continuation; LESSONS
> compression 2026-06-06) — interrogating the ONE dimension every prior positive-skew arc held fixed.
> **Disposition:** _(filled at end)_
> **Scored solely by** `MultiPairBacktester` (FundedNext costs ON, SL-first / take-the-loss). Canonical
> apparatus CALLED (`Panel.from_pairs`, `ArcFoldRunner`→`A1Architecture`, `build_v3_folds`,
> `apply_cost_model`); the trend entry is the registered EXPERIMENT tool `DonchianBreakoutLongSignal`
> (arc 2000); the tail-removed metric reuses arc 2063 winsorization arithmetic / arc 2081 `tail_metrics`.

## Why this arc (the one un-varied axis — NOT a re-derivation of closed ground)

The positive-skew / continuation thread (the operator's ONE open in-charter thread, LESSONS 2026-06-06)
has been closed across **geometry** (Donchian breakout 2081, vol-expansion 2082, forward-confirmed shock
2081/3019, favorable-first pullback 2083/1075), **timeframe** (H4 2081–2083, D1 2084/1076, W1 2085/3014),
**direction** (long + short), **universe** (USD7 majors + full 28-pair), and the **exit family** (`sl_only`
+ the three tail-preserving trailing exits). Every one of those arcs converged on the SAME unified finding
(the cross-range gradient stated in the 2084/2085 correction): **an invariant median trade R ≈ −0.9**, with
the mean monotone-from-below-zero, the right tail generic-and-insufficient, losing to a being-long null.

**That invariance is the clue this arc chases.** Every prior arc varied the entry, the timeframe, the
direction, the universe, and the exit — and got the SAME −0.9 median. The ONE thing they ALL held constant
was **SL = 2·ATR.** A median trade R of −0.9 is mechanically a *stop-width* artifact: with a 2·ATR stop,
the majority of continuation entries are whipsawed out by normal retracement noise BEFORE the trend
develops, registering ≈ −1R (take-the-loss). That is the take-the-loss tax landing on the *entry*, and it
is governed by stop width, not by the entry rule that prior arcs swept. §2 (question everything) says
interrogate the one constant behind an invariant result.

A positive-skew continuation strategy's *defining* design choice is the stop width relative to the noise:
the loss should be taken only when the trend thesis is genuinely wrong, not on routine retracement. So the
genuinely-untested cell = **does widening the stop (3·/4·/6·ATR) lift the median off the −0.9 wall and flip
the mean positive on a continuation entry, under the tail-preserving trailing exit and the mandated
mean + median-per-fold + tail-removed lens?**

**Why this is not closed by arc 3004.** arc 3004 tested the *limit* case (stop REMOVED) on a *momentum*
entry with a *time exit* under the *capture/drift* lens — "lifts mean ~4–5pp but still not all-folds-pos."
That is (a) a different lens (the operator's whole redirect is that capture/drift is blind to skew), (b) a
different exit (time-exit caps the tail, defeating the premise; this arc uses trailing), and (c) only the
two endpoints {2·ATR, ∞}, never the intermediate widths where the survival-to-tail gain might beat the
R-compression. The take-the-loss invariant also forbids a *removed* stop as a deployable; the live question
is the intermediate, finite, still-take-the-loss widths. So 3004 is a strong KILL prior (it makes this
likely confirmatory), but the intermediate-width × trailing × mean-lens cell is genuinely un-run.

**Risk-normalization correctness (verified before running).** `A1Architecture` risk-sizes each position
from the entry→SL distance at `risk_pct=0.005` (a1_system_level_filter.py:253), and the exit policies define
`R_atr = sl_atr_mult × atr_at_entry` as the size of 1R in price (_base.py:46/95). So a stop-out loses
exactly $500 = −1R **regardless of sl_atr_mult**, and per-trade `net_pnl/$500` is correctly normalized
across SL widths. Widening the SL therefore means: each stop-out is still −1R but the position is smaller,
the stop is further (fewer trades stop out → median should rise), and 1R is a bigger price distance (so the
right tail in R-units COMPRESSES — winners bank fewer R for the same ATR move). The mean's sign under this
tension is genuinely ambiguous → worth one bounded measurement.

## Hypothesis

Widening the stop on the canonical Donchian-breakout continuation entry (full 28-pair, take-the-loss +
tail-preserving trailing) lifts the median trade R off the −0.9 wall, and — *if* the survival-to-tail gain
beats the R-compression — could flip the per-trade mean positive AND survive the tail-removed guard.
*Because:* the invariant −0.9 median across all prior arcs is a 2·ATR-stop whipsaw artifact, not an
entry-quality artifact; a continuation edge's profitability lives in how rarely routine noise takes the
loss. Prior from arc 3004 (the ∞-stop limit stayed negative on momentum) predicts this likely returns KILL
— in which case it is the confirmatory closer of the LAST un-varied axis (stop width), completing the
thread's closure.

## ⚠️ PRE-REGISTERED KILL-RULE (written BEFORE results — applied verbatim afterwards)

Identical tail-luck-≠-skew guard as arcs 2081–2085. With per-trade realized R = `net_pnl / ($500)`:

1. **Mean expectancy** — per-trade mean R and per-fold mean ROI must be **> 0 net of costs**. If the raw
   net mean is ≤ 0 at every SL width → **KILL** (you cannot diversify a negative-mean book positive; §11).
2. **TAIL-REMOVED guard (decisive).** Re-compute the mean with the right tail capped/dropped: (a) **+2R
   cap** (`min(net, 2·$500)`, losses untouched) and (b) **top-5% winners removed.** If the mean goes ≤ 0
   under EITHER tail-removal at the best SL width — i.e. mean-positive ONLY via the top-K winners — it is
   **KILL** (tail-luck; no relabeling 2–3 monster trades as skew; cf. arcs 2011/2063, and the SNB-day
   +132.7R trap arc 2082 caught).
3. A real edge shows **broad-based** right-tail behaviour across folds (median-per-fold supportive), AND a
   widened-stop cell must **beat the being-long-anytime null** under the same SL+exit to count as a real
   entry edge (not generic long-vol exposure).

Disposition (§11): **PASS** = all-folds-positive IS+OOS + council + survives the guard. **PORTFOLIO** =
mean-positive net of costs AND survives the guard, not all-folds-positive. **KILL** = everything else
(incl. mean-positive-only-via-tail, beats-null-but-net-negative, coin-flip). **OOS stays FROZEN** unless a
cell clears the IS cheap-kill + tail guard + beats the null.

## Method (CALLS canonical; experiment-side = entry mask + SL-multiple parameter + winsorization only)

1. Load all **28 pairs**, H4, `5ers_eet`, real bid/ask.
2. Entry = `DonchianBreakoutLongSignal(lookback=40, spacing_bars=6)` (registered, arc 2000) — the canonical
   continuation entry arc 2081 used at 2·ATR.
3. **Sweep `sl_atr_mult ∈ {2.0, 3.0, 4.0, 6.0}`** (the one un-varied axis; 2.0 = the prior-arc baseline,
   reproduced as a control) × exit ∈ {`sl_plus_trailing_atr`, `sl_partial_close_1r_runner_trail`} (the
   tail-preserving trailing exits — NOT tp_2r/3r, which cap the tail).
4. Score over the full IS folds (`build_v3_folds`, 2011–2020); per-trade net R from
   `apply_cost_model(...).breakdown.net_pnl / $500` filtered to each fold's OOS slice (the arc-2063/2081
   unit). Compute, per cell: per-trade mean R, **median R** (the load-bearing number — does it climb off
   −0.9?), win rate, +2R-cap mean, top-5%-removed mean; per-fold ROI, folds-negative, 2015/2018 signs.
5. Null = `PeriodicLongSignal` (being-long-anytime) under the same exit at SL ∈ {2.0, 4.0} — does a wider
   stop on the breakout beat a wider stop on generic long exposure?
6. Apply the pre-registered kill-rule verbatim. OOS only if a cell clears IS cheap-kill + tail guard + null.

## Results

Driver: `_disco2000_work/arc2086_stop_width_sweep.py` (Donchian-40 long, full 28-pair, SL ∈ {2,3,4,6}·ATR
× {trailing_atr, partial_runner} + a being-long null at SL ∈ {2,4}). IS folds = `build_v3_folds` (2011–2020,
10 folds); per-trade net R = `apply_cost_model(...).breakdown.net_pnl / $500`.

### The median trade R DOES climb off the −0.9 wall as SL widens — the invariance was a stop-width artifact

`sl_plus_trailing_atr` (the partial_runner column is near-identical):

| SL (·ATR) | median R | win | per-trade mean R | fold mean ROI | folds neg | worst fold | 2015 / 2018 | +2R-cap | top5%-rm | max R |
|---|---|---|---|---|---|---|---|---|---|---|
| 2.0 (baseline) | −0.784 | 0.387 | −0.1301 | −22.60% | 10/10 | −37.4% | −33.0 / −13.1 | −0.1948 | −0.3091 | +7.5 |
| 3.0 | −0.013 | 0.495 | −0.0739 | −9.51% | 7/10 | −24.3% | −20.4 / −6.8 | −0.1052 | −0.2148 | +8.4 |
| 4.0 | +0.106 | 0.565 | −0.0519 | −5.88% | 7/10 | −17.9% | −14.2 / −4.4 | −0.0636 | −0.1625 | +6.2 |
| 6.0 | +0.131 | 0.649 | −0.0369 | −3.32% | 5/10 | −14.6% | −6.3 / +0.4 | −0.0388 | −0.1149 | +4.1 |

The hypothesis's mechanism is **directly confirmed:** widening the stop monotonically lifts the median trade
R from −0.78 (the corpus's invariant −0.9 wall) to +0.13, and the win rate from 0.39 to 0.65. The invariant
−0.9 median across every prior arc (geometry/timeframe/direction/universe/exit) WAS the 2·ATR-stop whipsaw —
fewer trades are stopped before the trend develops once the stop is wider.

### …but the per-trade MEAN asymptotes to ≈0 FROM BELOW and never crosses — and it LOSES TO THE NULL

- **Mean stays negative at every SL width** (−0.130 → −0.074 → −0.052 → −0.037), monotone-approaching-zero
  from below. The best cell anywhere is `partial_runner SL=6`: mean **−0.0314R**, +2R-cap **−0.0317**,
  top-5%-removed **−0.0968** — **all negative → KILL by the pre-registered guard.** Fails guard #1 (raw mean
  ≤ 0) outright; the tail guard is moot but confirms it.
- **The negativity is NOT tail-luck — it is broad-based ≈ the cost hurdle.** As SL widens the right tail
  COMPRESSES (max R 7.5 → 4.1; +2R-cap survival 150% → 101%), so by SL=6 there is essentially no right tail
  left to remove — the tail-removed mean ≈ the raw mean. This is the *opposite* of the thin-tail trap: the
  book is uniformly, mildly negative (≈ the FundedNext cost), not positive-via-a-few-monsters.
- **Loses to the being-long-anytime null at matched SL.** Null trajectory is identical: SL=2 mean −0.106 →
  SL=4 −0.0425, the SAME monotone climb. At SL=4 the Donchian breakout (−0.052) is WORSE than the null
  (−0.0425); the entire improvement from widening is a generic long-FX-vol effect, NOT entry-selectable. This
  re-confirms arc 2000/2081's "the right tail is generic, not entry-selectable" — now also along the SL axis.

OOS NEVER touched (no cell cleared the IS cheap-kill).

## Diagnosis — the invariant −0.9 median was the stop; removing it just converges the book to ≈cost from below

The redirect-mandated lens, applied to the one un-varied axis, gives a clean and *informative* answer. Two
facts, both new to the corpus:

1. **The corpus's invariant median trade R ≈ −0.9 was a 2·ATR-stop artifact, not an entry-quality artifact.**
   Widening the stop lifts the median monotonically to +0.13 and the win rate to 0.65 — directly falsifying
   any reading of the −0.9 wall as "the entry is bad." The entry was fine; the *stop* manufactured the −1R
   median by whipsawing continuation entries out before the move.
2. **But fixing the median does not fix the mean.** The mean climbs monotonically toward zero and *asymptotes
   from below without crossing* — because widening the stop trades the win-rate gain against R-compression
   (a wider stop ⇒ smaller risk-sized position ⇒ the same ATR move banks fewer R; max R falls 7.5 → 4.1)
   one-for-one, net of cost. The conservation law made concrete on the stop axis: you can move WHERE the
   take-the-loss tax falls (frequent small-R losses → rare wider losses) but not its TOTAL, and FundedNext
   cost keeps the converged mean just below zero. The breakout entry adds *negative* value over a being-long
   null at every width, so even the asymptote is generic long-vol, not an edge.

This also closes the residual "wide-stop continuation as the long-vol 2015/2018 sleeve" hope with a real
nuance: a 6·ATR stop genuinely de-bleeds the reversion book's death years (2015 −33%→−6.3%, 2018 −13%→+0.4%)
— the long-vol complement is *less* false at wide stops — but the book is still net-negative-mean and loses
to the null, so it cannot be the Calmar-lifting 5th leg (§11: cannot diversify a negative-mean component
positive). Arc 3004's ∞-stop limit (lifts mean but stays non-positive) is confirmed as the endpoint of a
monotone-from-below curve: no *finite* stop width flips it either.

## Council — NOT convened (clean cheap-kill / objective pre-registered guard is the rigor)

§5d cheap-kill: the mean is negative at every cell (no worthwhile ceiling) and the best cell is adjudicated
KILL by the objective pre-registered guard I committed to applying verbatim, and it loses to the null at
matched SL. Convening would re-derive the arc-2081/2082/2083 continuation verdict ("ritual not rigor",
arc-1002 precedent). OOS NEVER touched.

## Verdict: KILL

No new component. The stop-width (SL-multiple) axis — the ONE dimension every prior positive-skew arc held
fixed at 2·ATR — is now closed: widening the stop lifts the median off the −0.9 wall (proving that wall was
a stop artifact) but the per-trade mean only asymptotes to ≈0 from below (best cell −0.031R, tail-removed
also negative), loses to a being-long null at every matched width, and converges to ≈the cost hurdle, not to
a positive edge. With geometry, timeframe, direction, universe, exit-family, AND now stop-width all varied,
the positive-skew / continuation thread is closed on every axis under its own mandated mean + median-per-fold
+ tail-removed lens. Deployable count = 0; components UNCHANGED (4 PORTFOLIO: gap 1006 / me_long 1011 /
fbr 1013 / me_short 1019).

## Lessons (candidate for LESSONS.md)

1. **The corpus's invariant median trade R ≈ −0.9 across all positive-skew arcs was a 2·ATR-STOP artifact,
   not an entry-quality artifact.** Sweeping SL ∈ {2,3,4,6}·ATR on the Donchian continuation entry lifts the
   median monotonically −0.78 → +0.13 and the win rate 0.39 → 0.65. The entry was never the problem; the
   tight stop manufactured the −1R median by whipsawing continuation entries out before the move.
2. **But fixing the median does not fix the mean — it asymptotes to ≈0 from below without crossing.** Wider
   stop = higher win rate but R-compressed tail (max R 7.5 → 4.1), trading off ~one-for-one net of cost; the
   mean climbs −0.130 → −0.037 and stops just under zero. The breakout LOSES to a being-long-anytime null at
   every matched SL, so even the asymptote is generic long-FX-vol, not entry-selectable. Stop width moves
   WHERE the take-the-loss tax falls, not its total. This is the last un-varied axis of the positive-skew
   thread, now closed; arc 3004's ∞-stop limit is the endpoint of this monotone-from-below curve.
3. **A wide stop de-bleeds 2015/2018 but is still not the long-vol sleeve.** 6·ATR turns 2015 −33%→−6% and
   2018 −13%→+0.4% (the long-vol-complement hope is *less* false at wide stops), but the book stays
   net-negative-mean and loses to the null → cannot be the Calmar-lifting 5th leg (§11). The genuine long-vol
   diversifier remains the operator-gated cross-asset trend universe (NEEDS_ENABLEMENT #1).

## Flags / Tooling

No canonical-core change (no FLAG). Carries the standing arc-3017 `risk_pct` percent-vs-fraction FLAG (ran
the LINEAR regime, risk 0.005). Reuses EXPERIMENT tools `DonchianBreakoutLongSignal` / `PeriodicLongSignal`
(arc 2000) and the arc-2081 `tail_metrics` arithmetic — no new BUILT tool (a pure SL-parameter sweep over
existing tools). Driver: `_disco2000_work/arc2086_stop_width_sweep.py`
(`PYTHONPATH=. py _disco2000_work/arc2086_stop_width_sweep.py`). Data: 28 pairs, H4, `5ers_eet`,
`histdata_root=C:\Users\panap\histdata_backup`, `cache_root=C:/Users/panap/Documents/Forex-Backtester/data/cache`.
