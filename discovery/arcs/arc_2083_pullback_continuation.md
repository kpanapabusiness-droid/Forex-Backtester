# arc 2083 — Trend-PULLBACK continuation (the FAVORABLE-FIRST residual) on the honest engine + TAIL-REMOVED

> **Arc id:** 2083 · **Chat:** 2000–2999 (continuous) · **Date:** 2026-06-07
> **Type:** EDGE-HUNT, positive-skew / continuation frontier (the favorable-first residual; closes the
> 4th dispatch-named construction under the mandated metric).
> **Disposition:** **KILL** (no new component; 4 PORTFOLIO unchanged; deployable=0; OOS NEVER touched).
> Scored solely by `MultiPairBacktester` (FundedNext costs ON, SL-first / take-the-loss). CALLS canonical;
> reuses the arc-2081 measurement harness + the BUILT `tail_removed_expectancy` arithmetic.

## Why this arc (the genuinely-distinct geometry the breakout closers did NOT cover)

The positive-skew / continuation thread — the operator's ONE open in-charter thread (LESSONS 2026-06-06)
— was closed for the **breakout** family: arc 1074 (Donchian-break-in-SMA-trend, honest engine + trailing
+ tail-removed), arc 2081 (Donchian-40 + forward-confirmed shock, full universe), arc 2082 (vol-expansion
breakout). All KILL.

But arc 1074's KILL came with a **precise mechanistic diagnosis** that itself names the one untested
geometry: a Donchian-break-in-trend entry on liquid majors is **ADVERSE-FIRST** (the classic false-breakout
whipsaw) — price pierces the channel, then immediately retraces THROUGH the 2·ATR stop, so take-the-loss
fires on the MAJORITY (median R ≈ −0.9) *before* any right tail can develop. The structurally-opposite
construction — **FAVORABLE-FIRST** — is the trend-**PULLBACK** entry: established trend → RETRACEMENT
against it → enter on the **resume** bar with the stop beyond the pullback extreme. By construction the
adverse move (the pullback) has *already happened* and a confirming resume bar has printed before entry, so
the −1R tax should fall on noise rather than on the entry itself (the S1 "take-the-loss-tax as entry
geometry" idea, `DISCOVERY_DIRECTION.md`). **This is the one continuation geometry whose median R could
exceed −0.9 and whose MEAN could flip positive under a trailing runner.**

The pullback entry was tested ONCE (arc 2012) but only on the **capture/drift cheap-kill lens** — the exact
win-rate lens the operator's 2026-06-06 redirect declares structurally blind to a positive-skew payoff. The
honest engine + tail-preserving trailing exits + per-trade MEAN + median-per-fold + TAIL-REMOVED lens has
**never** been spent on the pullback geometry. That is this arc — the favorable-first complement to arc 1074.

## Hypothesis

A trend-pullback-resume continuation entry across the full 28-pair universe, exited take-the-loss −1R +
tail-preserving trailing, has a **mean-positive, broad-based** positive-skew edge net of FundedNext costs —
*because* the favorable-first geometry (enter after the retracement + a confirming resume bar, stop beyond
the pullback extreme) should lift the median R above the breakout's −0.9 wall and let the right tail pay.

## ⚠️ PRE-REGISTERED KILL-RULE (written BEFORE results — applied verbatim afterwards)

Per-trade R := position `net_pnl / ($500 = risk_pct·SB)`. Same guard as arcs 2081/1074:
1. **G1 (mean):** per-fold MEAN ROI **> 0** net of costs AND mean per-trade R **> 0**. Deeply-negative → KILL.
2. **G2 (TAIL-REMOVED, decisive):** re-compute the mean under (a) +2R cap and (b) top-5% / top-K removal
   (losses untouched). **If the mean goes flat/negative under EITHER → KILL** (mean-positive-only-via-tail
   = tail-luck, not skew; cf. arcs 2011/2063).
3. **G3 (distributed):** per-fold median ROI > 0 in a MAJORITY of folds; binding 2015/2018 supportive.
4. **Null:** must beat the fair being-long random/periodic null AND be net-positive. **Beats-null-but-net-
   negative = KILL** (§11). OOS touched only if IS survives G1+G2+G3+null.

## Method (CALLS canonical; experiment-side = entry mask + winsorization arithmetic only)

NEW BUILT SignalModule `PullbackContinuationSignal` (both directions): HH/HL swing-structure (≥2 ascending
swing-highs AND ≥2 ascending swing-lows in [t−30, t−4], right-edge lag 4 → no lookahead) → pullback
`close_mid[t−1] ≤ last_swing_high − 0.5·ATR[t−1]` → resume bar at t (bullish close, breaks prior-bar high,
close in upper half); short = exact mirror (LH/LL, pullback up, resume down). All MID OHLC, ATR(14) Wilder
shift(1), entry fills next-bar open; lineage matches `core.signals.pullback_resume_hhhl` (arc-8 long spec)
generalized to MID + a short mirror. Full **28-pair** universe, H4, SL=2·ATR, risk 0.005 (linear regime).
Swept direction {long, short} × exits {sl_only, trailing_atr, trailing_swing, partial_runner} (runner/
tail-preserving only — NO tp_2r/3r) vs a being-long periodic null. IS folds only (2011–2020); OOS untouched.
Driver `_disco2000_work/arc2083_pullback_continuation.py`.

## Results — KILL on every cell; the favorable-first hypothesis is FALSIFIED

| cell (best per dir/exit) | fold mean ROI | folds neg | per-trade mean R | median R | win | +2R-cap | top5%-rm | max R | guard |
|---|---|---|---|---|---|---|---|---|---|
| **long trailing_atr (BEST of 8)** | ≈ −13% | 9–10/10 | **−0.0938** | −0.86 | 0.38 | −0.169 | −0.275 | +7.7 | KILL |
| long partial_runner | −13.83% | 8/10 | −0.0964 | −0.84 | 0.476 | −0.109 | −0.214 | +4.2 | KILL |
| long sl_only / swing | ≈ −0.10R | 9–10/10 | −0.1021 | −0.85 | 0.38 | −0.175 | −0.287 | +7.7 | KILL |
| short trailing_atr | −14.97% | 10/10 | −0.1028 | −0.86 | 0.38 | −0.193 | −0.302 | +18.9 | KILL |
| short partial_runner | −15.01% | 9/10 | −0.1039 | −0.86 | 0.474 | −0.121 | −0.230 | +10.0 | KILL |
| short sl_only / swing | −20 to −23% | 9/10 | −0.59 to −0.85 | −0.73/−0.85 | ≈0 | (=) | worse | <+0.2 | KILL |
| _null: periodic-long trailing_atr_ | −21.61% | 9/10 | −0.1057 | −0.85 | 0.39 | −0.183 | −0.296 | +10.6 | — |

- **G1 FAIL everywhere.** The BEST of all 8 cells is **long trailing_atr at −0.0938R per trade** (per-fold
  ≈ −13%); every other cell is more negative (−0.10 to −0.85R; per-fold −13% to −23%; 8–10/10 folds neg).
  Decisively negative, not marginal.
- **The favorable-first hypothesis is directly FALSIFIED.** The median R is **−0.84 to −0.86** — essentially
  the SAME −1R wall as arc 1074's breakout (−0.878). Entering *after* a retracement + a confirming resume
  bar does **not** keep the path off the stop: the 2·ATR stop is still hit before +1R on the majority
  (win 0.38–0.48). Win-rate ticks up modestly under the partial-runner exit (≈0.48), but mean stays negative.
- **G2 confirms NO positive skew.** Tail-removal makes the already-negative mean **MORE** negative in every
  cell (+2R-cap and top-5%-removed both worsen it; "% survives" > 100% = the tail was *helping* an otherwise
  worse mean, not carrying a positive one). The largest single winner (+18.9R, a short) is INSUFFICIENT vs
  the −0.85R median loser + costs — the inverse of positive skew.
- **G3 FAIL.** Per-fold median ROI > 0 in ~1–2/10 folds; **2015 and 2018 negative in every cell** — the
  pullback book also bleeds in the reversion book's death years (NOT the long-vol complement).
- **Null / decisive §11 verdict.** The best pullback cell (−0.0938R) *barely* beats the being-long periodic
  null (−0.1057R) by +0.012R — the pullback ENTRY adds at most a sliver over generic long-vol exposure — but
  **both are net-NEGATIVE.** Beats-null-but-net-negative = **KILL** (§11).

## Verdict: KILL

The favorable-first geometry — the one mechanistic hope arc 1074's own diagnosis left open — does NOT
rescue continuation. A trend-pullback-resume book across the full 28-pair universe, both directions, all
tail-preserving trailing exits, is per-trade **mean-negative** (best −0.0938R), its median R is the same
−0.85 stop-wall as the breakout, tail-removal only worsens it, it bleeds 2015/2018, and it merely ties the
being-long null while staying net-negative. With arcs 1074 / 2081 / 2082 this closes the **fourth and last
distinct positive-skew continuation construction** under the operator's own mandated metric (mean +
median-per-fold + tail-removed). Deployable count = 0; components UNCHANGED (4 PORTFOLIO: gap 1006 /
me_long 1011 / fbr 1013 / me_short 1019). OOS NEVER touched.

## Lessons (candidate for LESSONS.md)

1. **Favorable-first entry geometry does NOT rescue continuation on liquid-FX OHLC.** The trend-pullback-
   resume entry (enter after the retracement, stop beyond the pullback extreme) was the one continuation
   construction structurally able to dodge arc 1074's adverse-first −1R wall. It does not: median R stays
   −0.85 (same wall), mean stays negative (best −0.094R), tail-removal worsens it, and it only ties a
   being-long null. The take-the-loss tax falls on the *entry*, not just noise — even a confirmed resume
   bar is followed by a stop-out more often than a +1R run on majors AND crosses.
2. **The pullback closes the S1 "take-the-loss-tax as entry geometry" thread empirically.** S1 hypothesized
   the favorable-first geometry would shrink the booked−theoretical gap; here it does not flip the sign —
   confirming fbr (arc 1013) is favorable-first *and* a reversion-of-overextension (forced-flow), not a
   generic geometry trick portable to continuation. The geometry is not the edge; the forced flow is.
3. **All four named positive-skew continuation constructions now KILL under the mandated lens** (breakout —
   1074/2081; vol-expansion — 2082; forward-confirmed shock — 2081; trend-pullback — 2083), both directions,
   full universe, trailing exits, vs a being-long null. The FX right tail is generic to being-long/short
   vol, not entry-selectable, and insufficient after costs — the conservation law made concrete again.

## Threads / handoff

- **Closed (under the mandated mean + median-per-fold + tail-removed metric):** trend-PULLBACK continuation
  (favorable-first), both directions, full 28-pair universe, all trailing exits, vs being-long null.
  Together with 1074/2081/2082 the positive-skew continuation thread is closed across all four canonical
  constructions; the favorable-first geometry was the last mechanistic hope.
- **Residual (adjacent-to-covered, low expected value):** raw **momentum-persistence** (K-consecutive-bar /
  strong-N-bar-return continuation) is a breakout-class TS-momentum entry → adverse-first, mechanically the
  same family arc 1074/2081 (Donchian = TS-momentum) already KILLed; a fresh chat may run a dedicated closer
  but the mechanism is mapped.
- **Beyond charter (operator-gated):** the genuine long-vol / positive-skew diversifier remains **cross-asset
  trend** (`NEEDS_ENABLEMENT.md` #1 — same shape, less-efficient universe where the tail may actually pay;
  blocker = historical data). The in-charter FX-OHLC continuation thread is now comprehensively closed.

## Flags / Tooling / Council

No canonical-core change (no FLAG). Carries the standing arc-3017 `risk_pct` percent-vs-fraction FLAG (ran
the LINEAR regime, risk 0.005, cap ~never binds). NEW BUILT tool `discovery/tools/pullback_continuation_signal.py`
(`PullbackContinuationSignal`, registered; valid reusable favorable-first continuation signal, kept despite
KILL). Reused: arc-2081 measurement harness (`collect`/`report`/`split_metrics` + `tail_removed_expectancy`
arithmetic), `PeriodicLongSignal` null (arc 2000), `_atr_shift1_mid`/`_apply_spacing`. **No council** —
decisive cheap-kill at IS + objective pre-registered tail guard (the §5d/§11 rigor for a deeply-negative
non-survivor; convening would re-derive the 1074/2081/2082 continuation verdict — "ritual not rigor",
arc-1002 precedent). OOS NEVER touched. Driver `_disco2000_work/arc2083_pullback_continuation.py`
(reproducible: `PYTHONPATH=. py _disco2000_work/arc2083_pullback_continuation.py`).
