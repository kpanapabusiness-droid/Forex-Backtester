# Arc 3005 — Cross-Rate Triangulation Residual (closing the last in-apparatus thread)

> **Arc id:** 3005 · **Chat:** 3000–3999 (continuous) · **Date:** 2026-06-05
> **Final verdict:** **FAIL (cheap-kill at observation)** — triangular arbitrage pins the quoted cross to its
> synthetic triangulation at H4; the residual is within spread-noise (~0.02 bps median) and has **zero forward
> predictive power** (convergence corr ≈ 0.00–0.02). No tradeable dislocation at H4.
> **Idea source:** the one remaining concrete in-apparatus thread the arc-3004 council named (Mechanism lens),
> the only *non-directional* idea expressible long-only single-instrument. Closing it completes the 3000s
> chat's in-apparatus coverage and makes the arc-3004 escalation complete.

Scored by inspection of the canonical panel (real bid/ask, H4 5ers_eet). No pool needed — the observation is
decisive.

## (a) Log read + synthesis

Pulled `origin/main`. 14 arcs + arc 3004's escalation (apparatus structurally incapable; directional space
cleanly closed including the stop confound). Arc 3004's council named ONE remaining concrete in-apparatus idea
that is genuinely *non-directional*: **cross-rate triangulation lag** — the synthetic cross built from its two
USD legs vs the quoted cross. The Mechanism lens flagged it low-conviction ("likely M1, dislocation may be <
the cross spread"). Testing it at H4 either finds a residual or closes the thread; either way it completes the
escalation's "we checked the last idea" honesty.

## (b) Observation → verdict

For each triple (cross = legA op legB, all vs USD): synthetic mid = `EURUSD/GBPUSD` (EURGBP), `EURUSD×USDJPY`
(EURJPY), etc.; residual = (quoted_mid − synthetic)/synthetic, in bps. IS+OOS full sample, contemporaneous H4
closes. Mechanism to test: if the quoted cross *lags* its triangulation, a long when quoted < synthetic should
converge UP (a non-directional, market-structural edge).

| cross | resid bps (med) | resid bps (std) | cross spread bps (med) | \|resid\|>spread % | fwd-6-bar convergence corr |
|---|---|---|---|---|---|
| EURGBP | +0.018 | 1.23 | 1.17 | 2.2% | 0.009 |
| EURJPY | −0.004 | 1.34 | 0.79 | 6.1% | 0.010 |
| GBPJPY | −0.012 | 1.51 | 1.29 | 2.7% | 0.016 |
| AUDJPY | −0.022 | 1.58 | 1.09 | 4.1% | 0.002 |
| EURAUD | +0.021 | 1.31 | 1.52 | 1.6% | −0.003 |
| AUDCAD | −0.050 | 1.32 | 2.33 | 1.3% | −0.000 |

**Three facts kill it at observation:** (1) the residual median is ≈ 0 (sub-0.05 bps) — quoted = synthetic on
average; (2) |residual| exceeds the cross spread only 1.3–6.1% of the time — i.e. it is *within spread-noise*,
not a real dislocation; (3) the residual has **no forward predictive power** — the correlation between
−residual and the forward 6-bar move (would-be convergence) is ≈ 0.00–0.02 across every triple. There is
nothing to trade.

**Mechanistically expected:** triangular arbitrage is the most basic, most-policed FX arb; any H4-close
dislocation is instantly arbed. Whatever lag exists is sub-second (tick), far below the H4 apparatus and below
the cost floor. (M1 is out of this apparatus's scope — H4 is primary — and would be a different cost regime, an
escalation item, not an in-apparatus lever.)

## Final verdict — FAIL (cheap-kill at observation)

No tradeable triangulation dislocation at H4. The last concrete *non-directional* in-apparatus thread is
closed. The arc-3004 escalation now stands with the in-apparatus space comprehensively exhausted from the
3000s side: directional (both metrics, all instruments/timeframes/regimes/exits/stop-geometry), volume,
calendar, convexity, and the triangulation microstructure idea — all closed.

## Lessons (candidate for LESSONS.md compression)

1. **No H4 triangular-arb residual.** Quoted crosses equal their USD-leg synthetic to within spread-noise on
   contemporaneous H4 closes (residual median ≈ 0; exceeds spread 1–6% of bars; forward-convergence corr ≈ 0).
   Triangulation lag, if any, is sub-second — out of apparatus scope. The one named *non-directional*
   in-apparatus idea is dead.
2. **In-apparatus space exhausted (3000s side).** With this, every concrete grounded in-apparatus lever the
   3000s chat (and the corpus) could name is closed. The operative finding is the arc-3004 escalation: a
   deployable FX edge needs a structural unlock (second leg / shorts / real tighter-cost regime), which is the
   operator's decision.

## Threads / what didn't help

- **Closed:** triangulation residual reversion (sub-spread, non-predictive at H4).
- **Operative state:** the arc-3004 escalation. No grounded in-apparatus arc remains; continuing to generate
  long-only single-instrument directional variants would be grinding a cleanly-closed space. The productive
  next step is the operator's structural-unlock decision (see `ESCALATION_apparatus_capability.md`).

## Flags (code NOT merged)

None. Observation driver scratch `_disco3_work/observe6_triangulation.py`.

## Reproduction

`Panel.from_pairs([EURUSD GBPUSD USDJPY AUDUSD USDCAD EURGBP EURJPY GBPJPY AUDJPY EURAUD AUDCAD], "H4",
histdata_root=C:\Users\panap\histdata_backup, cache_root=data/cache, boundary_convention="5ers_eet")`;
synthetic mid = leg ratio/product per triple; residual bps + forward-6-bar convergence corr.
Driver: `_disco3_work/observe6_triangulation.py`.
