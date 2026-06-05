# arc 2013 — Weekend UP-gap SHORT (JPY crosses): IS results

Honest engine (`MultiPairBacktester`, FundedNext costs ON, SL-first), JPY crosses
[EURJPY, GBPJPY, AUDJPY, NZDJPY, CADJPY, CHFJPY], H4, IS 2010–2020, `build_v3_folds` folds 2–11.
**OOS (2021+) PRESERVED — not measured.**

## Gross pool (characterization, build_arc_pool, thr up-gap≥1.0)
- IS pool n = **137** trades; mean `final_r` **+0.137** (short-signed: positive = short profited); win 0.212.
- thr ≥0.5: n=427, mean final_r −0.050 (loses) → all exits net-negative; thr ≥0.5 is dead.

## Exit-menu IS WFO (§5f nested hyperparameter), thr up-gap≥1.0 — mean fold ROI
| exit_policy | AFP | mean | worst | neg folds | min trades/fold |
|---|---|---|---|---|---|
| `sl_plus_trailing_atr` | No | **+0.745%** | −3.18% | 5/10 | 1 |
| `sl_plus_tp_3r` | No | +0.510% | −4.05% | 6/10 | 1 |
| `sl_partial_close_1r_runner_trail` | No | +0.324% | −3.14% | 5/10 | 1 |
| `sl_plus_tp_2r` | No | +0.015% | −4.05% | 6/10 | 1 |
| `sl_plus_trailing_swing` | No | −1.427% | −8.66% | 8/10 | 0 |
| `sl_only` | No | −1.974% | −8.66% | 8/10 | 0 |

The four OVERSHOOT-capturing exits are ALL positive (the 1006/1007 mechanism: let the fill-overshoot DOWN
run); the two non-overshoot exits are negative. The positive sign is mechanism-robust, not a single-exit
cherry-pick. NOT all-folds-positive under any exit.

## Per-fold IS series — component exit `sl_plus_trailing_atr`, thr up-gap≥1.0
| fold | OOS year | ROI % | n trades |
|---|---|---|---|
| 2 | 2011 | +1.91 | 5 |
| 3 | 2012 | +2.21 | 14 |
| 4 | 2013 | −2.24 | 12 |
| 5 | 2014 | −3.14 | 6 |
| 6 | 2015 | −0.57 | 2 |
| 7 | 2016 | −3.18 | 8 |
| 8 | 2017 | +1.97 | 8 |
| **9** | **2018** | **+5.08** | 8 |
| **10** | **2019** | **+5.92** | 10 |
| 11 | 2020 | −0.51 | 1 |

**This per-fold ROI vector is the correlation/combination target for the future 4-way arc.**
Positive folds: 2011, 2012, 2017, **2018, 2019**. Negative: 2013, 2014, 2015(n=2 noise), 2016, 2020.

## Regime-complementarity (the portfolio point)
- **2018 = +5.08%** — the binding portfolio wall. The three long components there: gap-fill −6.79, fbr
  −4.20, month-end +0.90. The up-gap short adds a SECOND strong 2018-positive leg.
- Bleed set {2013, 2014, 2016} differs from the long fades' {2015, 2016, 2018, 2020} — complementary on
  2018, partial overlap only on 2016.
- Does NOT cover 2015 (−0.57 on n=2 = noise) → the 2015 half of 1015's spec remains open.

## Fair same-side NULL (random JPY-cross weekly-open SHORT, matched fire-count, 5 seeds)
| exit | REAL | NULL (5-seed mean) | LIFT |
|---|---|---|---|
| `sl_plus_trailing_atr` | +0.745% | −0.994% (−0.81..−1.16) | **+1.739%** |
| `sl_plus_tp_3r` | +0.510% | −1.172% (−0.58..−1.93) | **+1.682%** |

The random short LOSES (JPY-basket drifts up against shorts); the up-gap timing adds **+1.7%** — a large,
seed-tight margin, cleaner than 1006's +0.36pp. The edge is in the TIMING, not the JPY-cross-short basket.

## Caveats (honesty)
- **Thin:** ~14 trades/yr; folds 6/2015 (n=2), 11/2020 (n=1) are too thin for a per-fold judgment — the
  all-folds-positive verdict is not the right lens for this component (it is a PORTFOLIO input, not a PASS).
- The component exit (`sl_plus_trailing_atr`) was chosen as the mechanism-aligned overshoot exit, with
  tp_3r/partial/tp_2r corroborating the positive sign. The combination arc should freeze ONE exit per the
  §5f IS-select / OOS-score / freeze-on-holdout discipline before any OOS touch.
