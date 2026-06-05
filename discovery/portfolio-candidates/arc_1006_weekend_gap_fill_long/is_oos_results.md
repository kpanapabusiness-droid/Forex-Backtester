# arc 1006 — per-fold IS + OOS results (transcribed, not re-run)

> Source: [`../../arcs/arc_1006_weekend_gap_fill_long.md`](../../arcs/arc_1006_weekend_gap_fill_long.md)
> §(c)/(d) + §(g). One config (IS-defined; no OOS tuning). Honest engine (`MultiPairBacktester`),
> FundedNext costs ON, SL-first. Numbers copied verbatim from the arc record — nothing re-computed.

## Verdict
**FAIL the sole judge** (all-folds-positive on neither IS nor OOS) → **disposition PORTFOLIO**
(mean-positive, beats null, lowest DDs of the run, but fold-fragile).

## IS — 10 folds (2010–2020)
Per-fold ROI %, in fold order:

```
-0.07, +8.23, -2.06, +2.94, -4.19, +3.20, +0.53, -6.79, +7.45, -2.39
```

- all-folds-positive: **NO** — 5/10 folds negative
- worst fold ROI: **-6.79%**
- mean fold ROI: **+0.69%**  ← the only mean-positive IS of any arc in the run
- worst per-day DD: **8.57%** (ledger); per-fold DDs ~2–9%

## OOS — 6 folds (2021–2026, per-year)
Per-fold ROI %, in year order:

```
-4.13, +6.84, -3.26, -2.25, -3.17, +5.98
```

- all-folds-positive: **NO** — 4/6 folds negative
- worst fold ROI: **-4.13%**

## Cheap-kill triage (3 representative folds — informational, pre-WFO)
`2013 -2.06%  ·  2016 +3.20%  ·  2019 +7.45%` — worst -2.06%, mean +2.86%, 2/3 positive, tiny DDs
(2.5–5%). The positives are in the CHOPPY years (2016/2019), not a single trending fluke →
did NOT cheap-kill → proceeded to full WFO.

## Null comparison (soundness control)
| | mean fold ROI | negative folds | worst |
|---|---|---|---|
| REAL signal (IS) | **+0.69%** | 5/10 | -6.79% |
| RANDOM entry (matched fire-rate, same exit) | -0.60% | 5–7/10 | ~-14% |

→ real **BEATS** random → a genuine (if fold-fragile) edge, not luck.

## Best-version refinement (tested, failed)
Hypothesis: gap-downs fill when they are dips in an uptrend but continue in a downtrend → add
`close > SMA50`. Result (IS): negatives 5/10 → 4/10, folds mostly small-positive, but **still not
all-folds-positive** (worst -4.40%, mean +0.34%) AND it over-thinned folds (min 5 trades/fold —
statistically meaningless). Does not rescue fold-consistency; no further tuning (would be overfitting).

## Pool
396 IS trades · capture 0.5000 · mean final_r +0.1127 (gross, positive — the fill works before
costs and fold-fragility bite).

## Why fold-fragile (mechanism)
JPY-cross weekend gaps are tail-event-timing-dependent: a few big risk-event weekends drive the fill;
quiet years are flat-to-negative. That is exactly why the mean is positive while individual folds
swing negative — and why it is a PORTFOLIO component (diversify the fold-variance), not a solo system.
