# arc 2082 — Volatility-EXPANSION breakout continuation (the positive-skew thread's final closer)

> **Arc id:** 2082 · **Chat:** 2000–2999 (continuous) · **Date:** 2026-06-07
> **Type:** EDGE-HUNT, positive-skew / continuation frontier (confirmatory closer for arc 2081).
> **Disposition:** **KILL** (no new component; 4 PORTFOLIO unchanged; deployable=0; OOS never touched).
> Scored solely by `MultiPairBacktester` (FundedNext costs ON, SL-first). CALLS canonical; reuses the
> arc-2081 harness + the BUILT `tail_removed_expectancy` guard.

## Why this arc

arc 2081 closed positive-skew continuation for **Donchian breakout** + **forward-confirmed shock** under
the operator-mandated mean + median-per-fold + tail-removed metric. The dispatch's third explicit example
— **volatility-regime EXPANSION breakouts** — was the one named residual I owed: structurally the most
likely to carry a real positive-skew tail (a breakout that fires DURING a vol-ignition bar, where
forced-flow trend-INITIATION is most plausible). This runs it to make the closure comprehensive across the
canonical positive-skew constructions.

## Method

NEW EXPERIMENT signal `VolExpansionBreakoutSignal` (registered): a Donchian-N break that fires **only on a
vol-ignition bar** (`TR/ATR ≥ vol_mult`), direction-aware (long = up-break, short = down-break). Ex-ante
(ATR shift1, Donchian shift1, current-bar TR known at close, crossing-bar + spacing, entry i+1). Full
**28-pair** universe, H4, SL=2·ATR, risk 0.005 (linear regime). Swept vol_mult ∈ {1.5, 2.0} × direction
{long, short} × the tail-preserving exits {trailing_atr, partial_runner}. Per-trade R = `net_pnl/$500`;
the **pre-registered tail-luck kill-rule from arc 2081 applies verbatim** (mean>0 AND +2R-cap>0 AND
top-5%-removed>0, else KILL). IS only; OOS untouched.

## Results — KILL on every cell (the guard fires on all 8)

| cell (vol_mult, dir, exit) | fold mean ROI | folds neg | per-trade mean R | +2R-cap | top5%-rm | guard |
|---|---|---|---|---|---|---|
| 1.5 long trailing_atr | −18.85% | 10/10 | −0.137 | −0.149 | −0.254 | KILL |
| 1.5 short trailing_atr | −15.53% | 9/10 | −0.107 | −0.239 | −0.353 | KILL |
| 1.5 short partial_runner | −15.39% | 9/10 | −0.109 | −0.155 | −0.263 | KILL |
| 2.0 long trailing_atr | −12.91% | 8/10 | −0.133 | −0.200 | −0.314 | KILL |
| 2.0 long partial_runner | −12.36% | 9/10 | −0.128 | −0.140 | −0.246 | KILL |
| 2.0 short trailing_atr | −13.76% | 8/10 | −0.130 | −0.219 | −0.334 | KILL |
| 2.0 short partial_runner | −14.13% | 9/10 | −0.136 | −0.151 | −0.258 | KILL |

(plus 1.5 long partial_runner, identical signature.) Every cell is **mean-negative** (per-trade −0.05 to
−0.16R; per-fold −12 to −19%; 8–10/10 folds negative); majors and crosses both negative; the guard returns
TAIL-LUCK/KILL on all eight.

**The vivid thin-tail demonstration (`volexp1.5 short`):** its 2015 fold reads **+43.1%** — but per-trade
mean is −0.107R and the **single largest trade is +132.7R** (an SNB-day CHF-cross short, 2015-01-15). One
catastrophic-outlier trade makes a whole year look stellar; the guard removes it (+2R-cap −0.239,
top-5%-removed −0.353) and the book is decisively negative. This is the EXACT thin-tail trap (arcs
2011/2063) the operator mandated the guard for — a result that, on the headline +2015 number, would have
been mistaken for the long-vol "5th leg," but is one lucky SNB short.

## Verdict: KILL

Vol-expansion breakout continuation is mean-negative on the full universe under every direction / vol
threshold / trailing exit, and its only positive-looking folds are single-outlier tail-luck. With arc
2081 (Donchian breakout + forward-confirmed shock), the positive-skew / continuation thread — the operator's
ONE open in-charter thread — is now **comprehensively closed across all three canonical constructions**
(unconditional breakout, vol-conditioned breakout, forward-confirmed shock), both directions, the full
28-pair universe, the tail-preserving trailing exits, under the mandated mean + median-per-fold +
tail-removed metric. Deployable count = 0; components UNCHANGED (4 PORTFOLIO).

## Lessons (candidate for LESSONS.md)

1. **Vol-expansion breakouts do NOT supply the positive-skew edge either.** Conditioning the breakout on a
   vol-ignition bar (the structurally-best continuation candidate) leaves it mean-negative everywhere; the
   right tail (even a +132R SNB outlier) does not pay for the −0.85R median loser + costs. Re-confirms arc
   2000/2081: the FX right tail is generic to being-long/short vol, not entry-selectable.
2. **The tail-removed guard is now battle-tested:** across arcs 2081–2082 it correctly KILLed every
   apparently-mean-positive +2015/+2018 continuation cell as single-outlier tail-luck — exactly the
   thin-tail trap it was built for. A +43% fold carried by one trade is the canonical false positive.

## Threads / handoff

Positive-skew continuation is closed (3 constructions × 2 directions × full universe × trailing exits ×
tail-removed metric). The genuine long-vol diversifier is now firmly an operator-gated charter expansion:
**cross-asset trend** (`NEEDS_ENABLEMENT.md` #1 — the same positive-skew shape on a less-efficient,
less-arbitraged universe where the tail may actually pay). No in-charter price-functional continuation
remains untested. NEW BUILT tool `discovery/tools/vol_expansion_breakout_signals.py` (registered; valid
reusable signal, kept despite KILL). No engine/null/canonical change; no FLAG (carries the standing
arc-3017 risk-convention FLAG; linear regime). No council (clean cheap-kill + objective guard). OOS NEVER
touched. Driver `_disco2000_work/arc2082_vol_expansion.py`.
