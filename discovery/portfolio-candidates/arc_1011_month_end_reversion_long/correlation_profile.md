# arc 1011 — correlation profile (PORTFOLIO component)

> Purpose: the PORTFOLIO route combines *decorrelated* net-positive components into an all-folds-positive
> *combined* book (protocol §6 / §11). This file records this component's correlation vs the existing
> candidate set and its per-fold return vector, so a future combination arc can use it.

## Correlation vs the existing candidate set
**This is the FIRST real correlation pair in `portfolio-candidates/`** (arc 1006 was the only prior component).

| pair | corr (Pearson, 10 IS fold-ROI) | co-fire? | universes | events |
|---|---|---|---|---|
| arc 1011 (month-end rev) vs **arc 1006 (weekend gap-fill)** | **+0.117** | NO (cannot — month-end last-trading-day vs weekly-open) | USD majors vs JPY crosses (disjoint) | month-end fix vs weekend gap (disjoint) |

**+0.117 = near-zero / genuinely decorrelated**, and structurally the two cannot co-fire (different calendar
triggers) on disjoint universes — the decorrelation is structural, not just sample-estimated.

## Per-fold return vectors (combination inputs)
Both on `build_v3_folds` IS folds (same fold windows; 1006 H4, 1011 D1 — comparable by fold index/date):

```
# arc 1011 month-end reversion (sl_only, 2-bar time exit, thr 1.0)   mean = +0.23%
is_roi_1011 = [+0.40, +0.29, +0.96, -0.23, -1.14, -0.51, +0.34, +0.90, +1.16, +0.15]

# arc 1006 weekend gap-fill (for reference, from its record)         mean = +0.685%
is_roi_1006 = [-0.07, +8.23, -2.06, +2.94, -4.19, +3.20, +0.53, -6.79, +7.45, -2.39]
```

## Naive 2-way combination (indicative ONLY — NOT the gate)
A naive equal-add of the two fold vectors: **6/10 positive, worst −5.89%** (the gap-fill's ±8% fold swings
dominate the month-end ±1%). Vol-normalized (risk-parity-ish): also 6/10. So **two components do not yet reach
all-folds-positive** — the gap-fill's fold-variance is too large relative to its mean for one decorrelated
partner to tame.

**This is indicative, NOT a gate.** The real combination is its own arc: the components co-simulated through
`MultiPairBacktester` (one book, shared daily-DD/cost accounting), risk-weighted, and judged by
all-folds-positive WFO on the COMBINED book (§11). The honest read: a 3rd decorrelated component and/or
risk-weighting that down-weights the gap-fill's variance is likely required. The portfolio thread is ACTIVE
(2 decorrelated net-positive components now exist); the combination gate is the operator's / a future arc's.
