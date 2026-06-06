# arc 2050 — London-open Asian-range liquidity SWEEP-AND-REVERSAL (long)

**Chat 2000s · VERDICT: KILL (obs cheap-kill, §5d).** No engine/null/council spent; OOS untouched;
components UNCHANGED (all 4 PORTFOLIO; honest deploy core = me_long-solo, arc 1046).

## Because (idea + mechanism)
`fbr` (1013) is the corpus's ONE robust structural edge: a stop-run sweep of a **visible swing low** that
fails and reclaims **reverts**, because the grab is an information-free forced flow (stop cascade). The
most DOCUMENTED, densest, universally-watched intraday stop pool in FX is the **Asian-session range**,
raided at the **London open** (the "judas swing" / opening liquidity grab): algos sweep the Asian low to
fill resting sell-stops, then — per the folklore — reverse into the real session.

This arc transplants fbr's exact mechanism (pierce a structural level → fail → reclaim → revert) to a
**documented level** (Asian range) + a **documented forced-flow time** (London open), instead of a
rolling algorithmic swing. Attractions: (a) decorrelated-by-timing from the weekend (1006) / month-end
(1011/1019) flows → a candidate NEW portfolio component; (b) it fires ~daily → THICK (arc-2017 option-B's
fold-resolving need). Genuinely UNTESTED vs the corpus: 1047 = session **drift** (continuous, sub-cost);
1050 = vol-vacuum spike with **no structural level** (collapsed to momentum); 2029 = **daily**
prior-day/week lows. None is the intraday session-open Asian-range sweep.

**Explicit cost-skeptic prior:** the intraday lane is repeatedly sub-cost (1047/1008/1010/3008). So the
screen demanded the post-sweep reversal (a) CLEAR the FundedNext cost line AND (b) BEAT a matched
failed-breakdown-reclaim STRUCTURE control (same shape, NOT at the session/Asian level) — the fbr
control discipline.

## Method (observation only; no engine/P&L)
H1, 7 cached pairs (EURUSD/GBPUSD/AUDUSD/USDJPY/EURJPY/GBPJPY/AUDJPY), UTC bars, IS 2010-2020 only (OOS
preserved). Asian range = bars hour∈[0..6] UTC (Tokyo); London sweep window = hour∈[7..10] UTC. LONG fire
= the FIRST sweep-window bar whose `low_bid` pierces the day's Asian-range low AND `close_mid` reclaims
above it (failed breakdown at the session level); entry next H1 bar via canonical `observe_long_capture`
(honest +1R-before-SL capture, take-the-loss; 6-bar forward drift in ATR; gross, characterization only).
Matched STRUCTURE control = the same `low<rolling-7-bar-low & close>level` reclaim shape OUTSIDE the
sweep window (a non-session level). Caveat: fixed UTC windows (not Europe/London DST) — acceptable for a
cheap obs-screen. Driver `discovery/_disco2_work/arc_2050_london_sweep_obs.py` (scratch).

## FALSIFIED three ways
- **(1) NO reversion — the sweep CONTINUES.** Session forward drift **−0.0744 ATR** (median −0.0472),
  NEGATIVE. Price keeps falling after the Asian-low sweep-reclaim; the move does not revert.
- **(2) Structure NOT load-bearing + sub-baseline.** Session capture 0.4444 vs control 0.4292 = a trivial
  +0.015 lift, but session *drift* (−0.074) is WORSE than control (−0.049) — the capture/drift
  disagreement = the take-the-loss-label-vs-forward-move artifact, not a real edge. Unconditional H1
  baseline = capture 0.4310 / drift −0.0608, so both fire-sets sit at/below baseline. **Sub-cost trivially:
  the reversal is the wrong sign** (sweep depth is a real median 0.327 ATR displacement, but forward
  drift is negative → no captured reversion to clear the ~0.05–0.10 R cost hurdle).
- **(3) Wrong-sign in the target regime.** Per-year session drift: 2014 −0.182 / 2015 −0.334 / 2018
  −0.213 — the sweep continues HARDEST in the strong-USD years a 2018 leg needs (worst possible sign).
  Positive only in choppy/range years (2012 +0.081, 2016 +0.070, 2017 +0.061, 2019 +0.144). Per-pair
  drift negative on 6/7 (GBPJPY ≈ 0); capture > 0.47 only on GBPUSD (0.4789) = pair-mix, not robust.

## Why it fails (mechanism) — the survivor DNA refined
`fbr` works because it is a **forward-confirming** reversal: the down-move is OVER and the reclaim catches
the reversion before it starts (i+1 enters before the up-move). The London-open sweep is the OPPOSITE — a
**backward-confirming** event: the raid IS the OPENING of the real London session move, so it CONTINUES
(the "judas swing reverses" folklore is survivorship-biased hindsight; on average the London-open Asian
break sets, not fades, the session direction). This re-confirms arc 2047 (continuation-side / backward-
confirming structure is coin-flip) and arc 1050 (a price-shape/time proxy for "forced flow" collapses to
momentum). **Key refinement of the survivor DNA:** a stop-run at a structural level reverts (fbr) ONLY
when it is NOT coincident with a directional-flow WINDOW — the London open brings genuine order flow that
overrides the stop-cascade reversion. "Surprise displacement + identifiable forced flow" (1050) must add:
the forced flow must be **information-free** (settlement gap / WMR rebalancing / a stand-alone stop-run),
NOT a session-open that carries real directional order flow.

## What this closes
The **session-open structural sweep-reversal** lane (the documented "judas swing" / London-open Asian-
range raid) on liquid FX majors at H1: it does not revert, it continues — and continues hardest in the
strong-USD regime. Together with 1047 (session drift, sub-cost) the **intraday session axis is now fully
mapped** — neither continuous local-hours drift nor the discrete session-open liquidity grab yields a
capturable, regime-orthogonal edge. The 2018/2014 regime-orthogonal component remains unfound; me_long
stays the lone honest OOS-robust PORTFOLIO leg (1046). Deployable-system count = 0.

## Tooling
No new BUILT tool — single-use exploratory scan (matching 2047/2048/2049/1050), canonical
`Panel.from_pairs` + `observe_long_capture` + `_helpers` only; no TOOL_REGISTRY append. No canonical
change, no FLAG, no council, OOS untouched. Driver `discovery/_disco2_work/arc_2050_london_sweep_obs.py`.

## NEW lesson
A stop-run sweep of a structural level **reverts only when it is information-free and NOT coincident with
a directional-flow window.** fbr's swing-low sweep reverts because it is a stand-alone stop-cascade at a
random time; the London-open sweep of the Asian range does NOT revert because the London open injects
genuine directional order flow — the raid is the START of the session trend, not the end of a cascade. So
the same structural shape (pierce-a-level-and-reclaim) flips from reversion (fbr) to continuation purely
by WHEN it fires. Extends arc 1050 ("surprise displacement is necessary but not sufficient"): the
accompanying forced flow must be information-free; a session-open carries real flow and the structure
becomes momentum (continues hardest in trend years). Also a useful datum: the H1 unconditional long
capture baseline on the 7 majors/crosses is ~0.431 (vs H4 ~0.4877).

## Threads / handoff
The intraday session axis (drift 1047 + session-open sweep 2050) is closed. The surviving edges are
information-free MECHANICAL forced flows keyed to a LARGE surprise displacement at a NON-directional-flow
time, living ON the USD factor; their strong-USD failure folds are regime-intrinsic. The 2018/2014
regime-orthogonal leg remains unfound (~22 dead routes); deployability lever = operator path-A (gate
governance, arcs 2019/3021/1046) + the arc-2033/2045 vehicle wall.
