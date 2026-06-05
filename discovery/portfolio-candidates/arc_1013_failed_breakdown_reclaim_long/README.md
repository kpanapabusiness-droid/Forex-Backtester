# PORTFOLIO CANDIDATE — arc 1013: Failed-Breakdown Reclaim Long (stop-run reversal, USD majors)

> **Disposition: PORTFOLIO** (protocol §11) — mean-positive net of costs (IS **and** OOS) but **NOT**
> all-folds-positive. A **CANDIDATE component, NOT deployable solo**, never a survivor. A decorrelated
> input to a future portfolio-combination arc — itself gated by all-folds-positive WFO on the *combined*
> book. Never auto-deploys.
>
> **Provenance:** produced live by arc 1013 (NOT transcribed) — full record
> [`../../arcs/arc_1013_failed_breakdown_reclaim_long.md`](../../arcs/arc_1013_failed_breakdown_reclaim_long.md)
> and Tier-1 ledger row 1013 in [`../../DISCOVERY_LOG.md`](../../DISCOVERY_LOG.md). Scored solely by
> `MultiPairBacktester` (FundedNext costs ON, SL-first). The signal is the registered
> [`FailedBreakdownReclaimLongSignal`](../../tools/failed_breakdown_signals.py) → turnkey-reproducible (see `config.yaml`).

## The strongest, cleanest directional component in the corpus
After 25 arcs found liquid-FX direction ≈ a coin-flip (shallow single-trigger cuts), this is the first
**deep, multi-factor directional** edge to clear the honest engine — the dispatch's thesis (depth, not
breadth) confirmed.

- **Mean-positive net of costs, IS AND OOS:** IS mean fold ROI **+1.854%** (9/10 folds positive, only 2018
  negative); **frozen one-shot OOS (2021+) mean +0.936%** (3/6 positive). The edge **persists forward** —
  where arc 2005 died (OOS −2.26%). The highest IS mean of any component (gap-fill +0.69%, month-end +0.23%).
- **Beats a NEGATIVE null by +2.96pp** (real +1.854% vs random-entry, matched fire-rate, SAME exit/pairs:
  −1.107%, 3 seeds). The cleanest null in the corpus — the structure is the entire edge, not exit geometry
  or regime drift (cf. arc 1009's gap-fill null was *positive* +0.327%, halving its headline).
- **Structure is mechanism-proven:** a same-magnitude rejection wick AT a swept swing low captures 0.52→0.61,
  the SAME wick elsewhere stays a coin-flip (~0.49–0.51) with negative drift; the excess grows with rejection
  size. Generic deep-down reversion is dead (3000/3001) — this is a structural stop-run reclaim, not that.
- **Robust:** mean-positive across K∈{40,60}×shadow∈{1.0,1.25,1.5} (sweet spot shadow≥1.25), every exit
  mean-positive (+0.83% to +1.85%), and leave-one-pair-out positive dropping any pair (broad, not one-pair luck).
- **But NOT all-folds-positive:** IS 9/10 (2018), OOS 3/6 — the negatives cluster in persistent strong-USD /
  risk-off years (2018, 2022, 2025) where breakdowns are *real*, not swept. Fails the sole judge → not a
  survivor (no `passed/`, council not reached). A regime drag (arc-1012 class), not exit-fixable, not fished.
- **Not KILL:** genuinely net-positive IS+OOS (you cannot diversify a net-negative component positive, arcs
  3000/3001) → it lands here.

## Contents
| File | What |
|---|---|
| [`config.yaml`](./config.yaml) | Exact signal + engine config — turnkey-reproducible (registered signal). |
| [`is_oos_results.md`](./is_oos_results.md) | Per-fold IS+OOS series, exit sweep, null, structure control, K/shadow + LOO robustness. |
| [`correlation_profile.md`](./correlation_profile.md) | Per-fold year vectors + complementarity vs the existing book (the arc-2006 3rd-component test). |

## This is the high-value 3rd component the portfolio route needed (arc-2006 spec)
Arc 2006 KILLed the 2-way book (gap-fill 1006 + month-end 1011) on a mutually-negative fold (2015) and
tail-correlation in 2015/16/18/20, specifying a 3rd component **selected by its ROI on the book's NEGATIVE
folds** — a *different mechanism family* (positive in risk-off). This component is exactly that: a
**structural stop-run reversal** (vs the two flow-event reversions → genuinely decorrelated), **positive in
2015 (+3.17%), 2016 (+2.55%), 2020 (+4.03%)** — three of the four years that book bled — sharing only **2018**.
The 3-way combination (the 2000s' arc-2006 machinery, all-folds-positive WFO on the co-simulated combined
book) should cut the book's mutually-negative folds from four (2015/16/18/20) to ≈one (2018). That combination
is its OWN gated arc — **NOT claimed by this record.**
