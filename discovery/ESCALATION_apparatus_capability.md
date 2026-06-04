# ESCALATION — Apparatus Capability (operator decision recommended)

> **Raised by:** chat 3000–3999, arc 3004, 2026-06-05, via the generative discovery council.
> **Status:** a RECOMMENDATION for the operator's review — NOT a halt. The discovery loop continues (no
> `STOP`); this note records a structural finding that the operator should weigh on return. It does not
> override the protocol or merge any code.
> **One line:** 14 arcs across 3 chats + a council-driven test that lifts the last confound are strong
> evidence the FIXED apparatus cannot express a deployable FX edge; the binding constraint is structural, and
> the productive next step likely requires a structural unlock only the operator can authorize.

## The finding

Every honest-era discovery arc (0; 1000–1005; 2000; 3000–3004) has FAILED the all-folds-positive judge. The
failures are not scattered — they converge on one structural fact:

- **The only signal the apparatus can express is DIRECTION** (long-only, single-instrument, per-trade SL/TP).
- **Direction on liquid FX H4/D1 is a ~0.49 coin-flip after costs**, and this is now established as
  **instrument-invariant** (28 pairs; crosses no better than majors), **timeframe-invariant** (D1≈H4),
  **metric-robust** (the +1R-before-SL capture lens AND the mean-forward-drift lens agree; best conditioned
  drift cell +0.023R vs a +0.05–0.10R cost hurdle), and robust across every entry mechanism (continuation,
  reversion, momentum, breakout, trend-following, cross-sectional), every regime filter (dispersion,
  volatility, Efficiency-Ratio — all fail; strong-trend regimes *invert*), the volume axis, the calendar axis,
  and exit/convexity engineering.
- **Arc 3004 lifted the one remaining confound:** every prior arc used a 2·ATR SL-first stop. A clean test
  (strongest +gross-drift entry, pure time-exit, FundedNext costs, scored only by `MultiPairBacktester`)
  shows removing the stop *helps* (mean fold ROI −4.52% → ~0%, the take-the-loss tax is a real ~4–5pp drag)
  but is **still not all-folds-positive** (best horizon: 5/10 folds negative, worst −20%, mean +0.62%). The
  residual edge ≈ cost and is regime-dependent. **So the stop was a contributing drag, not the binding wall.**

## Why this is mechanistic, not a search-coverage gap

The apparatus forbids the inputs that actually drive FX *direction* (interest-rate differentials, central-bank
flow, sovereign positioning). What remains in price+tick-volume is a faint momentum/rebalancing drift (~+0.10R
gross on crosses) that the dealer spread is precisely priced to consume. FX's *durable, paid* edges are
**relative-value and carry** — carry is a market-neutral risk premium; value/cointegration is a *spread*
between two instruments. **None of these can be expressed long-only, single-instrument.** The 14-arc result is
the apparatus correctly reporting that the residual after removing relative-value is a coin-flip.

(Honest caveat, per the council's Soundness lens: "the apparatus is incapable" is a *seductive, search-ending*
conclusion, and the 14 arcs cluster into ~2 mechanism families × several levers. This note is therefore framed
as a recommendation to *weigh*, and the discovery loop is **not** halted — a few thin in-apparatus threads
remain, e.g. a cross-rate triangulation-lag microstructure probe. But the directional price-structure space,
including the stop confound, is now cleanly closed.)

## Recommended unlocks (ranked; all require operator authorization)

1. **A second simultaneous leg** (highest value). Enables market-neutral cross-spreads and **cost-mutualising**
   constructions — net the flow internally so the portfolio pays the spread once per capital round-trip instead
   of per-trade. This is the single lever that does NOT require beating the 0.50 directional wall trade-by-trade
   (it changes the *cost* term, not the *edge* term), and it is what makes the otherwise-blocked
   "portfolio of decorrelated sub-cost edges" coherent.
2. **Shorting.** Doubles the expressible signal and lets mean-reversion/relative-value trade both sides;
   currently the apparatus + pool builder are long-only by construction.
3. **A different execution/cost regime** (e.g. raw-spread ECN). FundedNext's per-round-turn cost (1.5× spread,
   $5/lot, slippage) is the explicit hurdle the gross edges fail against; it is fixed by protocol §1 (inviolable
   for CC), so testing a tighter cost model is an operator decision. NB: a tighter cost regime that the live
   account cannot actually obtain would re-introduce the Arc-10 class of error (a gate that doesn't match
   reality) — so this unlock must match a *real* obtainable broker, not an optimistic assumption.

## What does NOT need changing

The engine, the honest accounting, the all-folds-positive judge, and the take-the-loss invariant are sound and
are not implicated — they correctly reported the negative result. The constraint is the *expressible strategy
space*, not the measurement.

## Pointers

Clean proof + council: [`arcs/arc_3004_stop_confound_and_escalation.md`](arcs/arc_3004_stop_confound_and_escalation.md),
[`results/arc_3004_stop_confound/council_transcript.md`](results/arc_3004_stop_confound/council_transcript.md).
Full evidence: `DISCOVERY_LOG.md` (14-arc ledger). Code FLAG (not merged): `A1Config.time_exit_bars` is defined
but unwired (arcs 1005, 3004).
