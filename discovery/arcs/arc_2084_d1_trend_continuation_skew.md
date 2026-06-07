# arc 2084 — Positive-skew trend CONTINUATION on the DAILY timeframe (the Turtle/managed-futures home)

> **Arc id:** 2084 · **Chat:** 2000–2999 (continuous) · **Date:** 2026-06-07
> **Type:** EDGE-HUNT on the operator-redirected frontier (positive-skew / continuation; LESSONS
> compression 2026-06-06) — the **timeframe axis** none of the H4 positive-skew arcs varied.
> **Disposition:** KILL.
> **Scored solely by** `MultiPairBacktester` (FundedNext costs ON, SL-first / take-the-loss). Canonical
> apparatus CALLED (`Panel.from_pairs`, `ArcFoldRunner`→`A1Architecture`, `build_v3_folds`,
> `nested_exit_selection` §5f); the trend entry is the registered EXPERIMENT tool
> `TrendContinuationBreakoutSignal` (arc 1074) exercised on D1; null = `build_null_signal_evaluation`.

## Why this arc (the one un-varied axis of the redirected frontier — NOT a re-derivation)

The operator's 2026-06-06 redirect opened ONE in-charter thread: positive-skew CONTINUATION, judged on
**MEAN + median-per-fold + TAIL-REMOVED** (lenses the corpus's "trend = dead" verdict was structurally
blind to). The 2000s + 1000s ranges have since closed it across **four entry geometries**:
- breakout (adverse-first Donchian) — arc 1074 (H4 majors), arc 2081 (H4 full 28-pair);
- vol-expansion breakout — arc 2082 (H4);
- pullback-resume favorable-first — arc 1075, arc 2083 (H4);
- forward-confirmed shock continuation — arc 2081 addendum (H4).

**Every one of those arcs ran on H4.** The single axis none varied is the **sampling timeframe** — and
that is not a cosmetic variation for THIS payoff shape:

**The mechanistic *because* for D1 ≠ H4.** A take-the-loss trailing exit's whipsaw rate is governed by
the ratio (trailing-stop width) / (intra-trend pullback size). On H4 a 2·ATR(H4) stop is *small* relative
to a trend's ordinary daily pullbacks, so it is hit by intra-trend noise (arc 2081: **median trade
−0.81R** = stopped out mid-move; the winner can never run to its fat-tail length). Classic positive-skew
trend-following — the Turtle 20/55-day Donchian with a 2N stop — is a **DAILY** phenomenon precisely
because at lower sampling frequency the stop sits ~√6× wider relative to the trend and survives ordinary
pullbacks, letting winners run for months. So the trailing-exit × take-the-loss interaction the
positive-skew premise *depends on* is structurally different on D1, and it had **never** been run on the
honest engine under the mandated lens. The Run-2 council itself flagged trend-following as the
"documented, capacity-large, positive-skew, fundable" shape (deferred to other instruments) — the
in-charter D1-FX version is the cheapest first test of exactly that claim.

## Hypothesis

The SAME continuation construction as arc 1074, moved from H4 to **D1** (full 28-pair universe), has a
**mean-positive, broad-based** positive-skew edge net of FundedNext costs — because on D1 the wider stop
survives intra-trend noise so the winner runs to its fat right tail, and a broad-diversified daily-trend
book is the long-vol sleeve that is positive in the reversion book's death years (2015/2018). *Falsifier:*
if the median trade is still ≈ −1R on D1 (stop-wall unchanged) and the mean stays ≤ 0 / dies tail-removal,
the timeframe does not rescue continuation — liquid-FX OHLC has no exploitable post-trend continuation.

## ⚠️ PRE-REGISTERED KILL-RULE (written BEFORE results — applied verbatim)

Tail-luck ≠ skew. Per-trade realized R = `net_pnl / ($500 = risk_pct·SB)` from the canonical cost
chokepoint (`apply_cost_model(...).breakdown`):
- **G1 (mean):** mean per-fold ROI **and** mean per-trade R must be **> 0** net of costs. A negative-mean
  book cannot be diversified positive (§11, arcs 3000/3001) → KILL.
- **G2 (tail-removed, decisive):** re-compute the mean with the right tail capped/dropped — (a) **+2R cap**
  (`min(net, 2·$500)`, losses untouched), (b) **top-5%/fold removed**, (c) **global drop-top-K {1,3,5}**.
  If the mean goes ≤ 0 under ANY tail-removal — mean-positive ONLY via the top-K winners — it is
  tail-luck → KILL. No relabeling 2–3 monster trades as "skew" (cf. arcs 2011/2063/2081).
- **G3 (broad-based):** per-fold median ROI > 0 in a majority of folds (not one monster fold).

Disposition (§11): **PASS** = all-folds-positive IS+OOS + council + survives the guard. **PORTFOLIO** =
mean-positive net of costs AND survives the guard (ideally +2015/+2018 / Calmar-lifting). **KILL** =
everything else.

## Method (CALLS canonical; experiment-side = entry mask + tail-removal/null arithmetic only)

Driver `_disco2_work/arc2084_d1_trend_skew.py` — arc 1074's EXACT harness, **sole change TF H4 → D1 +
full 28-pair universe** (so timeframe is the only varied axis vs the H4 closures):
1. Load all **28 pairs**, **D1**, `5ers_eet`, real bid/ask (cache warm at the main-repo `data/cache`).
2. Entry = `TrendContinuationBreakoutSignal(direction, lookback, sma_fast=50, sma_slow=200, primary_tf="D1")`
   — dual-SMA trend filter + a fresh Donchian-N break (crossing bar + 6-bar spacing). Cells: long/short ×
   Donchian {20 (Turtle fast), 55 (Turtle slow)}.
3. Exit = the **trailing family** (`sl_plus_trailing_atr`, `sl_plus_trailing_swing`,
   `sl_partial_close_1r_runner_trail`) × SL {1.5, 2.0, 2.5}·ATR — **NOT** tp_2r/tp_3r (tail-capping
   defeats the premise). §5f nested-WFO exit/SL selection (`metric_afp_then_mean`, select on strictly
   earlier folds, score that fold) — no exit-fishing. risk 0.005 (LINEAR; cap ~never binds).
4. Score over IS folds (`build_v3_folds`, 2011–2020; nested selection evaluates 2013–2020 after 2-fold
   warmup). Per-trade `net_pnl` filtered to each fold's OOS slice → mean R, median R, +2R-cap, top-5%/fold,
   global drop-top-K; per-fold mean ROI, folds-positive, 2015/2018 signs.
5. Null = `PeriodicLongSignal`-style random same-side entry (`build_null_signal_evaluation`, seed 42)
   under the cell's frozen all-fold-best exit — does the breakout beat generic being-in-vol exposure?
6. Apply the pre-registered guard verbatim. OOS spent ONLY if a cell passes G1+G2+G3 + beats null.

## Results — all 4 cells KILL at IS; OOS NOT touched

| cell | fires | n_trades | mean ROI/yr | folds+ | per-trade mean R | median R | +2R-cap | top5%-rm | vs null | 2015 / 2018 |
|---|---|---|---|---|---|---|---|---|---|---|
| long Donchian20 | 1964 | 349 | **−3.66%** | 2/8 | −0.1677 | −0.934 | −4.69% | −6.90% | **−2.11pp (LOSES)** | −6.41 / −4.58 |
| short Donchian20 | 1804 | 412 | **−4.29%** | 1/8 | −0.1667 | −0.923 | −6.31% | −8.27% | **−5.47pp (LOSES)** | −0.59 / −9.32 |
| long Donchian55 (Turtle) | 1351 | 327 | **−0.88%** | 2/8 | −0.0431 | −0.844 | −3.56% | −5.03% | **+3.16pp (beats)** | **+10.09** / −2.60 |
| short Donchian55 (Turtle) | 1191 | 286 | **−2.21%** | 3/8 | −0.1238 | −0.940 | −2.52% | −4.64% | −1.59pp (LOSES) | −1.37 / −6.38 |

**Every cell fails G1 (mean ROI and per-trade R both negative), G2 (tail-removal makes the mean MORE
negative everywhere — the inverse of positive skew), and G3 (≤ majority folds median-positive).** OOS
preserved frozen.

## Diagnosis — slowing the clock REDUCES the bleed but does NOT create the skew; the −1R stop-wall survives

Two findings, one decisive:

1. **The timeframe DOES matter — directionally, not categorically.** D1 is materially less-dead than H4:
   the Turtle-long-D55 cell bleeds **−0.88%/yr** vs the H4 continuation book's **−6 to −9%/yr** (arc 1074)
   / Donchian-40's **−22%/yr** (arc 2081); per-trade mean R improves from ≈ −0.13R (H4, 2081) to −0.043R
   (Turtle-long-D55). The mechanistic prediction — slower sampling ⇒ wider stop ⇒ less whipsaw — is
   **partially borne out** (the bleed shrinks ~8×). And the Turtle-long does the long-vol thing the thread
   was looking for: **+10.09% in 2015** (the reversion book's death year) and it **beats the
   being-in-vol null by +3.16pp** — the daily trend ENTRY adds positive value, unlike on H4 (arc
   2081/1074: entry ≈ random or worse).

2. **…but the −1R stop-wall is INTACT and the mean never crosses zero (decisive).** The median trade R is
   still **−0.84 to −0.94** on D1 — i.e. the *typical* trade still takes the −1R stop, exactly as on H4.
   The breakout is still **adverse-first** even at daily resolution: the trend resumes often enough to
   fire but pulls back through the stop before the fat tail develops, on D1 as on H4. So the wider stop
   shrinks the per-loss bleed but does not flip the sign, and **tail-removal worsens every cell** (the
   precise inverse of positive skew — the right tail, max +4.5R to +7.9R, is REAL but INSUFFICIENT to pay
   for the frequent −1R losses + FundedNext costs). The one beats-null cell (Turtle-long-D55) is still
   **net-negative**, which is **KILL not PORTFOLIO** (§11: you cannot diversify a negative-mean component
   positive — beats-null-but-net-negative is the canonical KILL case).

This is the conservation law (`frequency × edge ≈ const`) made concrete on the timeframe axis: lowering
the sampling frequency moves the loss *geometry* (smaller per-trade bleed) but cannot manufacture
positive expectancy where the underlying liquid-FX price is a near-martingale. The documented
positive-skew trend edge that the Run-2 council named lives in **less-efficient instrument universes**
(cross-asset trend, NEEDS_ENABLEMENT #1), not in FX-OHLC at any timeframe.

## Council — NOT convened (clean cheap-kill / objective pre-registered guard is the rigor)

§5d cheap-kill: every cell is mean-negative with no worthwhile ceiling, fails the objective pre-registered
tail-removed guard verbatim, and the single least-dead cell (Turtle-long-D55) is beats-null-but-
net-negative = KILL by §11. Convening would re-derive the 1074/2081/2083 continuation verdict ("ritual
not rigor", arc-1002 precedent). OOS NEVER touched.

## Verdict: KILL

No new component. The positive-skew / continuation thread is now closed on the **timeframe axis** too:
moving the Turtle/managed-futures classic from H4 to **D1** (full 28-pair universe, both directions,
Donchian {20,55}, all trailing exits, §5f nested) **reduces** the bleed ~8× and the Turtle-long even
beats the null + is +2015, but the median trade is still a −1R stop, the mean never crosses zero, and
tail-removal worsens every cell. Deployable count = 0; components UNCHANGED (4 PORTFOLIO: gap 1006 /
me_long 1011 / fbr 1013 / me_short 1019).

## Lessons (candidate for LESSONS.md)

1. **Positive-skew continuation is dead on liquid-FX OHLC on the DAILY timeframe too — closing the last
   un-varied axis of the operator-redirected thread.** The Turtle/managed-futures classic (Donchian
   {20,55} + dual-SMA 50/200, full 28-pair, both directions, trailing exits, §5f) is per-trade
   **mean-negative** at D1, with the **median trade still ≈ −1R** (the breakout is adverse-first even at
   daily resolution), and tail-removal worsens the mean — the right tail (max +4.5–7.9R) is generic and
   insufficient, the inverse of positive skew.
2. **Slower sampling REDUCES the bleed ~8× but does not create the edge (the timeframe is a geometry
   lever, not a sign lever).** Turtle-long-D55 bleeds −0.88%/yr vs H4's −6 to −22%/yr; per-trade mean R
   improves −0.13R→−0.043R; it even beats the being-in-vol null (+3.16pp) and is +10.09% in 2015 (the
   long-vol-in-the-reversion-death-year signature the thread sought). But it stays net-negative ⇒ KILL
   (beats-null-but-net-negative, §11). The wider D1 stop shrinks per-loss size without flipping the sign —
   conservation law on the timeframe axis.
3. **The documented positive-skew trend edge is NOT in FX-OHLC at any timeframe.** Across H4 (1074/2081/
   2082/2083/1075) AND D1 (this arc), every continuation geometry is mean-negative or tail-luck. The
   genuine long-vol / positive-skew diversifier the vehicle wall needs lives in a **less-efficient
   instrument universe** (cross-asset trend, NEEDS_ENABLEMENT #1; same shape, blocker = historical data),
   confirming the Run-2 council's structural placement.

## Threads / handoff

- **Closed (under the mandated metric):** positive-skew trend continuation on D1 (full 28-pair, both
  directions, Donchian {20,55}, all trailing exits, §5f nested) — mean-negative, median trade −1R,
  tail-removal worsens it, the one beats-null cell is net-negative. With the H4 closures (1074/2081/2082/
  2083/1075), the positive-skew continuation thread is now closed across **entry geometry AND timeframe**.
- **Residual (low expected value):** W1 (weekly) is the only finer-grained timeframe cell left, but the
  D1→H4 gradient (bleed shrinks but sign fixed; median stays −1R) plus the W1 thinness (a 20/55-week
  Donchian over 2011–2020 fires too rarely to populate per-year folds — pool-floor risk) predict it
  closes the same way with worse fold resolution. A fresh chat may run it as a final closer if desired.
- **Beyond charter (operator-gated):** the genuine long-vol / positive-skew diversifier is the cross-asset
  trend universe (`NEEDS_ENABLEMENT.md` #1) — the in-charter FX-OHLC version is now closed on both axes.

## Flags / Tooling

No canonical-core change (no FLAG). Carries the standing arc-3017 `risk_pct` percent-vs-fraction FLAG
(ran the LINEAR regime, risk 0.005, cap ~never binds — consistent with arcs 2063/3019/2081). **No new
BUILT tool** — reused `TrendContinuationBreakoutSignal` (arc 1074, exercised on D1 via its `primary_tf`
field — no signal change), `nested_exit_selection` (arc 2040), `build_null_signal_evaluation` (arc
1000/2013), and arc-1074/2081's tail-removal arithmetic inline. Driver `_disco2_work/arc2084_d1_trend_skew.py`
(reproducible: `PYTHONPATH=. py discovery/_disco2_work/arc2084_d1_trend_skew.py`). Data: 28 pairs, D1,
`5ers_eet`, `histdata_root=C:\Users\panap\histdata_backup`,
`cache_root=C:/Users/panap/Documents/Forex-Backtester/data/cache`.
