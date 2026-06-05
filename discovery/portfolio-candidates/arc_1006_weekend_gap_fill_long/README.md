# PORTFOLIO CANDIDATE — arc 1006: Weekend Gap-Down-Fill Long (JPY crosses)

> **Disposition: PORTFOLIO** (protocol §11) — mean-positive net of costs but **NOT**
> all-folds-positive. This is a **CANDIDATE component, NOT deployable solo**, and never a
> survivor. It is a decorrelated input to a future portfolio-combination arc — itself gated by
> all-folds-positive WFO on the *combined* book. It never auto-deploys.
>
> **Provenance (no re-run):** transcribed from the arc 1006 record
> [`../../arcs/arc_1006_weekend_gap_fill_long.md`](../../arcs/arc_1006_weekend_gap_fill_long.md)
> and Tier-1 ledger row 1006 in [`../../DISCOVERY_LOG.md`](../../DISCOVERY_LOG.md). This deep record
> was **created 2026-06-05** to back the A2 disposition backfill: the ledger was reclassified to
> `disposition=PORTFOLIO` as a column edit, but the canonical `portfolio-candidates/<name>/` record
> did not exist until now. No backtest was re-run; every number here is copied from the existing
> arc record. Scored solely by `MultiPairBacktester` (FundedNext costs ON, SL-first).

## Why PORTFOLIO (not PASS, not KILL)
- **Mean-positive net of costs:** IS mean fold ROI **+0.69%** — the FIRST and only mean-positive IS
  edge of the run.
- **Beats the random-entry null:** real IS mean +0.69% vs random −0.60% → a genuine edge, not luck.
- **But NOT all-folds-positive:** IS 5/10 folds negative (worst −6.79%); OOS 4/6 negative (worst
  −4.13%) → fails the sole discovery judge → not a survivor (no `passed/` record; council not reached).
- **Not KILL:** it is genuinely net-positive. You cannot diversify a net-negative component positive
  (arcs 3000/3001), so net-negative = KILL — but arc 1006 clears that bar, so it lands here.

## Contents
| File | What |
|---|---|
| [`config.yaml`](./config.yaml) | Exact signal + engine config (transcribed). |
| [`is_oos_results.md`](./is_oos_results.md) | Per-fold IS + OOS series, triage, null comparison, DDs. |
| [`correlation_profile.md`](./correlation_profile.md) | Correlation vs the candidate set (this is the FIRST component) + the per-fold return vectors a future 2nd component is correlated against. |

## The portfolio thread (why this record matters)
Arc 3001 established you cannot combine net-negative components into a positive system. Arc 1006 is
the FIRST net-positive (if fold-fragile) component, which **re-opens the portfolio/selection thread**:
a decorrelated combination of several net-positive-but-fold-fragile edges could plausibly reach
all-folds-positive (diversification cuts fold-variance while preserving positive mean). That thread
**activates the moment a SECOND net-positive component is found** — at which point its per-fold series
(in `correlation_profile.md`) becomes the correlation target.
