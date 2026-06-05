# arc 1006 — correlation profile (PORTFOLIO component)

> Purpose: the PORTFOLIO route combines *decorrelated* net-positive components into an
> all-folds-positive *combined* book (protocol §6 / §11). This file records (1) this component's
> correlation vs the existing candidate set and (2) its per-fold return vectors, so a future 2nd
> component can be correlated against it. Transcribed from the arc 1006 record; nothing re-run.

## Correlation vs the existing candidate set
**N/A — this is the FIRST and (as of 2026-06-05) ONLY component in `portfolio-candidates/`.**
There is no peer to correlate against yet, so pairwise correlation is **undefined**. The
portfolio-combination arc cannot run until a SECOND net-positive component exists (arc 1006 lesson 2:
"the thread activates the moment a 2nd net-positive edge is found").

## Per-fold return vectors (the correlation inputs for a future 2nd component)
When a 2nd net-positive component lands, correlate its per-fold ROI vector against these (Pearson on
the aligned fold-ROI series; IS and OOS separately). **Low / negative correlation + both-net-positive**
is the property that could combine to all-folds-positive (diversification cuts fold-variance while
preserving positive mean).

```
# IS — 10 folds (2010–2020), ROI %        mean = +0.69%
is_roi_pct  = [-0.07, +8.23, -2.06, +2.94, -4.19, +3.20, +0.53, -6.79, +7.45, -2.39]

# OOS — 6 folds (2021–2026, per-year), ROI %   mean ≈ -0.00% (worst -4.13%)
oos_roi_pct = [-4.13, +6.84, -3.26, -2.25, -3.17, +5.98]
```

(Per-trade return series, if needed for a finer correlation than fold-level, must be regenerated from
the engine — it was not persisted in the arc-1006 scratch run. **Status: pending** — fold-level
vectors above are what the arc record preserves; do not fabricate a per-trade series.)

## Decorrelation rationale (qualitative, from the arc record)
- **Distinct mechanism:** weekend positioning / liquidity overhang reverting Monday — a
  **tail-event-timing** effect (a few big risk-event weekends drive the fill), structurally unlike the
  trend / momentum / mean-reversion / calendar mechanisms behind the run's other (net-negative) arcs.
- **Universe-specific:** concentrated in JPY crosses (EURJPY strongest, mean fwd +0.48 ATR); majors
  continue down (arc 2001). A majors-based or non-gap component is unlikely to co-fire with it.
- **Expectation (UNVERIFIED until a peer exists):** low co-firing and low fold-correlation with a
  trend/calendar component. **Status: pending** — to be computed when a 2nd component is recorded here.
