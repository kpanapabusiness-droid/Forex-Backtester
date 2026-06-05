# PORTFOLIO CANDIDATE — arc 1011: Month-End Reversion Long (USD majors)

> **Disposition: PORTFOLIO** (protocol §11) — mean-positive net of costs but **NOT** all-folds-positive.
> A **CANDIDATE component, NOT deployable solo**, never a survivor. A decorrelated input to a future
> portfolio-combination arc — itself gated by all-folds-positive WFO on the *combined* book. Never
> auto-deploys.
>
> **Provenance:** produced live by arc 1011 (NOT transcribed) — full record
> [`../../arcs/arc_1011_month_end_reversion_long.md`](../../arcs/arc_1011_month_end_reversion_long.md) and
> Tier-1 ledger row 1011 in [`../../DISCOVERY_LOG.md`](../../DISCOVERY_LOG.md). Scored solely by
> `MultiPairBacktester` (FundedNext costs ON, SL-first). The signal is the registered
> [`MonthEndReversionLongSignal`](../../tools/month_end_signals.py) → turnkey-reproducible (see `config.yaml`).

## Why PORTFOLIO (not PASS, not KILL)
- **Mean-positive net of costs:** IS mean fold ROI **+0.23%** (sl_only, 2-bar time exit; +0.50% at 3-bar),
  positive across thresholds 0.75–1.5 (+0.18 to +0.27%).
- **Beats the fair same-exit null by +0.56pp** (real +0.23% vs random-entry −0.33%, matched fire-rate, same
  exit/SL/universe) — the apples-to-apples null (arc-1009 discipline). Independently corroborated by the gross
  **month-end-vs-random-day control: +0.249 ATR excess** (generic big-down reversion is NEGATIVE −0.063 →
  the month-end *timing* is the mechanism, not generic reversion which is dead, arcs 3000/3001).
- **Mechanism-robust:** threshold-robust (0.75–1.5), broad-based (7/10 folds), NOT single-pair (leave-one-out
  all positive). *More* threshold-robust than the gap-fill (arc 1006, which lived only at 0.5 ATR).
- **But NOT all-folds-positive:** IS 7/10 folds positive (worst −1.14%) → fails the sole discovery judge → not
  a survivor (no `passed/` record; council not reached). OOS deliberately NOT touched (holdout preserved).
- **Not KILL:** genuinely net-positive (you cannot diversify a net-negative component positive, arcs
  3000/3001) → it lands here.

## Contents
| File | What |
|---|---|
| [`config.yaml`](./config.yaml) | Exact signal + engine config — turnkey-reproducible (registered signal). |
| [`is_oos_results.md`](./is_oos_results.md) | Per-fold IS series, exit sweep, null, control, robustness. |
| [`correlation_profile.md`](./correlation_profile.md) | Correlation vs the candidate set (the FIRST real pair: +0.117 vs arc 1006) + per-fold vectors. |

## The portfolio thread is now ACTIVE
Arc 1006 (gap-fill) was the FIRST net-positive component; the thread "activates the moment a SECOND
net-positive component is found." **Arc 1011 is that second component** — decorrelated (fold-ROI corr **+0.117**
vs arc 1006; disjoint events + universes). The combination is its OWN gated arc (all-folds-positive WFO on the
co-simulated combined book); a naive equal-add of the two transcribed fold vectors is still 6/10 positive (the
gap-fill's ±8% fold swings dominate), so the combination likely needs risk-weighting and/or a 3rd component —
that is the next portfolio step, NOT claimed by this record.
