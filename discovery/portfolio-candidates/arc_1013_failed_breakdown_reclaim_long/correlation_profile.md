# arc 1013 — correlation / complementarity profile

The portfolio gate is the *combined* book's all-folds-positive WFO (§6/§11); the decision-relevant
question for a 3rd component is NOT average correlation but **fold-complementarity** — does it cover the
existing book's NEGATIVE folds? (arc 2006: "rank a candidate 3rd by its ROI on the existing book's negative
folds, not by standalone mean or average correlation; average-correlation decorrelation ≠ tail-complementarity.")

## The existing book (arc 2006)

The 2-way book (gap-fill 1006 JPY-cross H4 + month-end 1011 USD-major D1, corr +0.117) was **KILLed**:
risk-parity cut worst-fold −6.79%→−1.53% but stayed 4/10 negative, **blocked by a mutually-negative fold
(2015) and tail-correlation** — both components bleed in the same risk-off years **2015 / 2016 / 2018 / 2020**.
The route needs a 3rd component positive in those years.

## This component's per-fold IS year vector (committed config)

| year | 2011 | 2012 | 2013 | 2014 | **2015** | **2016** | 2017 | **2018** | 2019 | **2020** |
|---|---|---|---|---|---|---|---|---|---|---|
| ROI | +7.55 | +3.05 | +0.91 | +0.19 | **+3.17** | **+2.55** | +1.23 | **−4.20** | +0.05 | **+4.03** |

**Complementarity (the arc-2006 test):** of the four years the existing book bled (2015/16/18/20), this
component is **POSITIVE in three — 2015 (+3.17), 2016 (+2.55), 2020 (+4.03)** — and shares only **2018
(−4.20)**. It is the **regime-orthogonal leg** the route needs: a *different mechanism family* (structural
stop-run reversal vs the two flow-event reversions), positive precisely when reversion bleeds.

## Mechanism decorrelation (why it should be tail-orthogonal, not just numerically)
- **Different family:** gap-fill + month-end are both *discrete-flow-event reversions* (weekend illiquidity;
  month-end rebalancing) → they share the risk-off tail. This is a *price-structure stop-run reversal* —
  fires intraweek on any large swing-low sweep, not tied to a calendar/flow event.
- **Different universe/TF:** gap-fill = JPY crosses H4; month-end = USD majors D1; this = USD majors H4.
- **Its OWN weakness is a DIFFERENT regime:** it bleeds in *persistent strong-USD trend* years (2018/2022/2025
  — dips that don't fail), not the illiquidity/flow tail. The overlap with the existing book is only 2018.

## What this is NOT (scope discipline)
This record establishes the **per-fold vectors + the complementarity case**. It does NOT run the 3-way
co-simulated all-folds-positive combination — that is the arc-2006-machinery combination arc (2000s range),
gated on the combined book. Quantitative cross-correlation vs the two components (reproduced over the same
folds) is best computed there, with all three books co-simulated. **Predicted from the vectors:** the 3-way
book's mutually-negative folds drop from four (2015/16/18/20) to ≈one (2018) → worth running, with 2018 the
remaining shared-negative risk to watch.
