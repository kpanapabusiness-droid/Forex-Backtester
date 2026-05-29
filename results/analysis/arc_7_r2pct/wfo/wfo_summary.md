# Step 5 — WFO Summary (Arc 7 r=2%)

**Config:** `A6::A6::cl1::sl2.0::thr0.5-0.7::exp2`
**r_base:** 2.0000%  (canonical: 0.5000%)
**Boundary convention:** 5ers_eet
**Window end:** 2025-12-31

## Per-fold OOS metrics

| Fold | OOS start | n_trades | ROI % | DD % | Daily 5% breaches | Ratio |
|---:|---|---:|---:|---:|---:|---:|
| 2 | 2011-01-01 | 14 | 12.28% | 2.93% | 0 | 4.19 |
| 3 | 2012-01-01 | 11 | 9.29% | 2.28% | 0 | 4.07 |
| 4 | 2013-01-01 | 17 | 29.89% | 2.33% | 0 | 12.82 |
| 5 | 2014-01-01 | 11 | 3.52% | 4.57% | 0 | 0.77 |
| 6 | 2015-01-01 | 14 | 10.91% | 2.71% | 0 | 4.02 |
| 7 | 2016-01-01 | 14 | 14.96% | 3.49% | 0 | 4.28 |
| 8 | 2017-01-01 | 17 | 10.88% | 3.90% | 0 | 2.79 |
| 9 | 2018-01-01 | 14 | 12.81% | 4.54% | 0 | 2.82 |
| 10 | 2019-01-01 | 19 | 10.48% | 3.14% | 0 | 3.34 |
| 11 | 2020-01-01 | 16 | 15.07% | 2.25% | 0 | 6.69 |
| H (12) | 2021-01-01 | 87 | 61.95% | 5.84% | 1 | 10.61 |

## Aggregate

- Worst-fold ratio: 0.77
- Mean-fold ratio: 4.58
- Worst-fold ROI %: 3.52% (fold 5)
- Worst-fold DD %: 4.57% (fold 5)
- All folds positive ROI: True
- Min trades / fold: 11
- Chained max DD (IS + holdout): 5.84%
