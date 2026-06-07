# arc 1074 — positive-skew CONTINUATION on the HONEST ENGINE (close arc-1064's deferred engine test)

**Chat:** 1000s · **Range:** 1000–1999 · **Date:** 2026-06-07 · **TF/universe:** H4, 7 USD majors
(EURUSD GBPUSD AUDUSD NZDUSD USDJPY USDCAD USDCHF) · **Window:** IS 2010–2020 (OOS 2021+ FROZEN, one-shot)

## Step (a) — log + LESSONS reading (shown)

- **Frontier (operator redirection 2026-06-06, LESSONS.md):** the in-charter reversion/directional search is
  a 16×-convergent terminus and genuinely closed. The ONE open in-charter thread is **positive-skew
  CONTINUATION**: the corpus measured "trend-following = dead" ONLY by +1R-before-SL **capture** and
  **mean-forward-drift** — win-rate lenses **structurally blind** to a positive-skew payoff (low capture +
  near-zero/negative *median* drift, but strongly +ve *MEAN* from a fat right tail). Judge on **MEAN +
  median-per-fold + TAIL-REMOVED expectancy**, exit = **take-the-loss −1R + let winners RUN via trailing**.
- **The decisive prior — arc 1064** ("Continuation / positive-skew shape family"): ran the SKEW lens on the
  textbook trend-continuation entry (fresh Donchian-N break inside an SMA50/200 trend, both directions, H4
  USD majors) and KILLED it — capture ≈base, median drift NEGATIVE everywhere, tail-removed (drop top-5%)
  mean STRONGLY NEGATIVE (−0.59..−1.14), binding 2015/2018 negative. **BUT arc 1064 cheap-killed on GROSS
  `fwd_drift_atr` and explicitly DID NOT spend the engine**, on the argument: *"fwd_drift is GROSS (ignores
  the stop) → the honest SL-first engine is strictly worse → clean cheap-kill."*
- **Why that bound is wrong for THIS exit (the gap this arc attacks).** Take-the-loss at −1R **caps every
  gross loser** (−3,−4,−5 ATR) at −1R. arc-1064's strongly-negative tail-removed *gross* mean is dominated
  by exactly those uncapped left-tail losers — which the honest engine TRUNCATES. So the engine is **not
  "strictly worse"**; for a favorable-first breakout entry its MEAN can differ in sign from the gross mean
  (it caps the left tail; it only loses the dip-then-recover winners). The honest MEAN under a trailing
  runner is a genuinely-open empirical question gross drift cannot answer. The operator's 2026-06-06
  redirection ("Scored solely by MultiPairBacktester … trailing/run exits are the point") targets exactly
  this deferred run.
- **Adjacent priors (not re-run here):** arc 2000 (unconditional Donchian + full-size trailing) cheap-killed
  by worst-fold ROI, no skew guard, no trend filter; arc 3019 (shock-continuation) reached the engine but
  under **tp_3r** (a fixed target that CAPS the tail — the wrong exit for skew) and OOS-epoch-failed; arc
  1062 (compression-expansion) cheap-killed at capture. None spent the engine on a trend-continuation entry
  with a TRAILING-RUNNER exit under the MEAN+tail-removed lens. That is this arc.

## Step (b) — idea / hypothesis (the *because*)

Take arc-1064's EXACT entry family (trend-continuation Donchian break inside an established dual-SMA trend,
long AND short) and **actually spend the honest `MultiPairBacktester`** under the positive-skew RUNNER exits
(`sl_plus_trailing_atr`, `sl_plus_trailing_swing`, `sl_partial_close_1r_runner_trail`) with §5f nested
exit/SL selection. Judge by per-fold MEAN ROI + median-per-fold + TAIL-REMOVED expectancy, vs a fair
same-side, same-exit random-entry null. *Because:* the −1R take-the-loss truncates the left tail that drove
arc-1064's negative tail-removed gross mean, so the engine MEAN is not predictable from gross drift — the
only honest way to close this thread is to run the engine. NO tp_2r/tp_3r (they cap the right tail and
defeat the skew premise).

## ⚠️ PRE-REGISTERED KILL-RULE — tail-luck ≠ skew (written BEFORE results)

Per-trade R := position `net_pnl / (risk_pct × fold-initial-equity)` (constant-notional). Per fold: ROI =
Σ net_pnl / equity. Honest §5f series = the nested-walk-forward-selected exit/SL per fold (select on strictly
earlier IS folds, score that fold; freeze the all-fold best onto OOS).

- **G1 (mean):** honest per-fold MEAN ROI > 0, net of FundedNext costs, AND mean per-trade R > 0.
- **G2 (TAIL-REMOVED — decisive):** re-compute the mean under (a) a **+2R cap** on every position and (b)
  **dropping the top-5% (and top-K=1,3,5)** winners. **If the mean goes flat/negative under EITHER → KILL.**
  No relabeling a handful of outlier trades as "skew." The edge must be broad-based across folds, not 2–3
  monsters carrying everything (cf. thin-tail traps arcs 2011/2063; and arc 1064's gross result).
- **G3 (distributed):** per-fold median ROI > 0 in a MAJORITY of folds AND per-pair mean > 0 in a majority.
- **Null:** must beat the fair same-side / same-exit random-entry null (arc-1000 `build_null_signal_evaluation`).

**Dispositions (§11):** PASS = all-folds-positive IS+OOS + survives G2 + council → `passed/`. PORTFOLIO =
mean-positive net of costs AND survives G2, but NOT all-folds-positive → `portfolio-candidates/`. KILL =
everything else (incl. mean-positive-ONLY-via-the-tail, or beats-null-but-net-negative, or coin-flip).

OOS is touched ONLY if the IS result survives G1+G2+G3 + beats null (else KILL at IS, OOS preserved).

## Step (c)/(d)/(f)/(g) — results (honest engine, IS 2010–2020, 8 evaluable folds 2013–2020)

Entry = `TrendContinuationBreakoutSignal` (new BUILT tool; arc-1064's exact dual-SMA trend + Donchian
crossing-bar entry, as a `SignalModule`). Scored by `ArcFoldRunner`→`A1Architecture`→`MultiPairBacktester`,
FundedNext costs, risk 0.005. §5f nested exit/SL over {`sl_plus_trailing_atr`,`sl_plus_trailing_swing`,
`sl_partial_close_1r_runner_trail`} × SL{1.5,2.0,2.5} (runner exits ONLY — no tp_2r/3r). Driver:
`discovery/_disco1_work/arc1074_continuation_engine_skew.py`.

| cell | n_trades | MEAN per-fold ROI | folds+ | mean per-trade R | median R | +2R-cap mean | drop-top5% mean | max winner R | vs null |
|---|---|---|---|---|---|---|---|---|---|
| long  Donchian20 50/200 | 281 | **−9.06%/yr** | 0/8 | −0.516 | −0.878 | −9.52% | −9.92% | +5.54R | LOSES −4.25pp |
| short Donchian20 50/200 | 324 | **−8.59%/yr** | 0/8 | −0.424 | −0.946 | −8.81% | −9.85% | +3.28R | LOSES −0.90pp |
| long  Donchian55 50/200 (Turtle) | 209 | **−9.09%/yr** | 0/8 | −0.696 | −0.911 | −9.09% | −9.03% | +0.26R | beats +3.63pp (still −9%) |
| short Donchian55 50/200 (Turtle) | 206 | **−6.38%/yr** | 0/8 | −0.495 | −0.907 | −6.40% | −6.73% | +2.43R | beats +0.25pp (still −6%) |

- **G1 FAIL in every cell** — mean per-fold ROI −6.4% to −9.1%/yr; mean per-trade R −0.42 to −0.70. The
  result is **decisively negative, not marginal**.
- **G2 is moot but confirms NO positive skew** — tail-removal (+2R cap, drop-top-5%/fold, drop-top-K global)
  **barely moves the already-negative mean** (≤0.9pp), because there is **no fat right tail**: the largest
  single winner across the whole grid is +5.54R, and drop-top-K{1,3,5} leaves the mean essentially flat at
  −9%. This is the *opposite* of a skew edge whose mean lives in the tail — here the mean is negative and
  the tail is thin.
- **G3 FAIL** — per-fold median ROI > 0 in 0/8 folds; binding 2015/2018 negative.
- **§5f overwhelmingly selected `sl_plus_trailing_swing` at the TIGHTEST SL (1.5)** — even the best honest
  runner exit cannot make the entry positive; the runner does its job (max winner +5.54R) but the entry
  feeds it a wall of −1R stops (median R ≈ −0.9 ⇒ a majority of trades hit the stop first).
- **Null:** the two negative-est cells LOSE to or barely beat the fair same-side random-entry null; the two
  "beats-null" cells are still −6% to −9%/yr → beats-null-but-net-negative = KILL (§11).
- *Caveat (honest):* two OOS windows (2014, 2015) had ≈0 qualifying trend-breakouts and read 0.0% — the
  verdict rests on the **window-independent pooled per-trade R** (−0.42..−0.70 over 209–324 trades/cell),
  which is robust to that sparsity.

## Why the engine is WORSE than gross drift (the corrected lesson)

Arc 1064 deferred this run arguing "gross fwd_drift is generous → the SL-first engine is strictly worse,
so a gross cheap-kill suffices." The engine confirms KILL but the *bound was wrong in direction*: I
hypothesized take-the-loss would LIFT the negative gross mean by capping the −3..−5 ATR left tail. It does
not. A Donchian-break-in-trend entry on liquid majors is **adverse-first** (the classic false-breakout
whipsaw): price pierces the channel, then immediately retraces through the stop. So the −1R take-the-loss
fires on the *majority* (median R ≈ −0.9), and the dip-then-recover winners the gross lens credited are
exactly the ones take-the-loss kills. The engine is decisively negative *because of*, not despite,
take-the-loss — and there is no right tail fat enough to pay for the stop wall.

## Verdict — KILL (all 4 cells; OOS NEVER touched)

Positive-skew continuation is now **closed on BOTH lenses**: gross fwd-drift (arc 1064) AND the honest
`MultiPairBacktester` under the correct positive-skew runner exits (this arc). This was the operator's
**one open in-charter thread** (LESSONS 2026-06-06); it is now closed by direct engine measurement, not a
proxy. Components UNCHANGED (4 PORTFOLIO; me_long-solo the honest deploy object). Disposition: **KILL**.
No council (decisive negative, not a survivor/fork). New BUILT tool registered: `TrendContinuationBreakoutSignal`.
Reusable lesson: a +mean on a skewed distribution is the jackpot tell; but here even the +mean is absent —
the entry is adverse-first so the runner exit feeds on a wall of −1R stops. No canonical change; no FLAG.
