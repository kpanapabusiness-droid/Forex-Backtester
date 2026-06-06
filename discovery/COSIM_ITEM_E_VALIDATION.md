# Item E — co-simulated portfolio equity curve: 4-way book validation

> Reproduces and re-judges the portfolio-route thread (arcs 1006/1011/1013/1019,
> combined in arc 1020) through the NEW single co-simulated equity-curve gate
> (`core/wfo/cosim_book.py`), side-by-side with the per-fold LINEAR combiner
> (`discovery/tools/combine_fold_roi.py`). Answers the open question arc 2019 left
> for the operator: **is the 4-way book's all-folds-positive failure a
> linear-combiner artifact, or fundamental?**
>
> Reproduce: `PYTHONPATH=. py scripts/cosim_validation/validate_4way_book.py`
> Date: 2026-06-06 · IS folds: per-year 2011-2020 · costs: FundedNext ON · engine: `MultiPairBacktester` (sole gate engine)

## Components (canonical A1 + MultiPairBacktester, SL=2.0 ATR)

| name | signal | universe | TF | exit |
|------|--------|----------|----|------|
| gap | `WeekendGapFillLongSignal(0.5, 36)` | 5 JPY crosses | H4 | 24-bar time-exit |
| me_long | `MonthEndReversionLongSignal(1.0, 2)` | 7 USD majors | D1 | 2-bar time-exit + `sl_only` |
| fbr | `FailedBreakdownReclaimLongSignal(40, 1.25)` | 7 USD majors | H4 | `sl_plus_trailing_atr` |
| me_short | `MonthEndReversionShortSignal(1.0, 2)` | 7 USD majors | D1 | `sl_partial_close_1r_runner_trail` |

### Reproduction fidelity vs arc-1020 recorded per-fold ROI (%)

gap / me_long / me_short reproduce the recorded numbers **exactly**; fbr is within
~1.4% (the `sl_plus_trailing_atr` trail params were not pinned in the arc record —
immaterial to the verdict at fbr's 0.107 weight).

```
year |   gap (mine/rec) | me_long | fbr (mine/rec)  | me_short
2011 | -0.07 / -0.07 |  +0.40 |  +8.97 / +7.55 |  +3.39
2012 | +8.23 / +8.23 |  +0.29 |  +3.12 / +3.05 |  +1.69
2013 | -2.06 / -2.06 |  +0.96 |  +0.86 / +0.91 |  -0.90
2014 | +2.94 / +2.94 |  -0.23 |  +0.24 / +0.19 |  +0.98
2015 | -4.19 / -4.19 |  -1.14 |  +3.36 / +3.17 |  +0.40
2016 | +3.20 / +3.20 |  -0.51 |  +4.04 / +2.55 |  -0.91
2017 | +0.53 / +0.53 |  +0.34 |  +0.80 / +1.23 |  -0.68
2018 | -6.79 / -6.79 |  +0.90 |  -4.39 / -4.20 |  +0.86
2019 | +7.45 / +7.45 |  +1.16 |  -0.17 / +0.05 |  +1.29
2020 | -2.39 / -2.39 |  +0.15 |  +3.99 / +4.03 |  +0.71
```

IS-frozen risk-parity weights: `gap=0.078, me_long=0.531, fbr=0.107, me_short=0.284`.

## Co-sim vs linear — per-fold ROI (%), book max-DD (%), exposure drops

Two co-sim measurements: **cap-OFF** (the strictly-monotone bound — real DD
interaction only) and **cap-ON** (the faithful book under the 2-per-currency
limit). `dOFF`/`dON` = co-sim − linear.

```
year |  linear  cosOFF   dOFF bookDD |   cosON    dON dropON
2011 |   +2.13   +2.05  -0.08   0.35 |   +1.05  -1.08     13
2012 |   +1.61   +1.59  -0.02   0.83 |   +0.97  -0.64     11
2013 |   +0.18   +0.18  -0.01   0.59 |   +0.34  +0.16      6
2014 |   +0.41   +0.41  -0.00   0.80 |   +0.57  +0.16     10
2015 |   -0.46   -0.46  +0.00   1.51 |   -0.50  -0.04     12
2016 |   +0.15   +0.16  +0.01   0.87 |   +0.61  +0.45      8
2017 |   +0.11   +0.02  -0.10   0.82 |   -0.07  -0.18     10
2018 |   -0.27   -0.27  +0.00   0.65 |   -0.58  -0.30      7
2019 |   +1.54   +1.53  -0.01   0.64 |   +1.18  -0.37     14
2020 |   +0.53   +0.53  +0.00   0.78 |   +0.31  -0.22     12
```

### Anti-optimism anchor (cap OFF)

`max |cosim_off − linear| = 0.096%` (risk-parity), `0.310%` (equal) — boundary-only
(the union clock's first OOS bar is an H4 bar, each D1 component's own is a D1 bar).
**The co-sim equity/cost composition adds NO return source.** The only NEW
information cap-OFF surfaces is the real BOOK max-DD (0.35–1.51%), which the linear
combiner structurally cannot see — and it never approaches the 5% daily cap (so the
daily cap never binds for this book; it is correctly inert here, not silently off).

### Exposure cap (cap ON)

The book is **USD-concentrated** (3 of 4 components trade USD majors), so the
2-per-currency cap **binds hard**: 103 positions dropped across 10 folds (~10/fold).
Because dropping is a faithful real constraint (not lookahead-free monotone),
cap-ON ROI is mixed-sign vs linear (dropping a net-loser raises ROI on 2013/14/16).
This is surfaced transparently, not hidden; the cap-OFF column is the
strictly-monotone bound.

## VERDICTS — all-folds-positive (IS 2011-2020)

| measurement | all-folds-positive | worst fold | n_nonpositive |
|-------------|:--:|:--:|:--:|
| linear, equal | **False** | −2.35% | 3 |
| co-sim cap-OFF, equal | **False** | −2.36% | 4 |
| co-sim cap-ON, equal | **False** | −2.75% | 3 |
| linear, risk-parity | **False** | −0.46% | 2 (2015, 2018) |
| co-sim cap-OFF, risk-parity | **False** | −0.46% | 2 (2015, 2018) |
| co-sim cap-ON, risk-parity | **False** | −0.58% | 3 (2015, 2017, 2018) |

## FOLD-FLIP VERDICT

- linear all-folds-positive: **False**
- co-sim cap-OFF: **False** — sign flips vs linear: **none** (tracks linear, deepens DD only)
- co-sim cap-ON: **False** — sign flips vs linear: **[2017]** (cap drops push 2017 marginally negative — strictly harder)

**Co-simulation CONFIRMS the linear verdict. The 4-way book's all-folds-positive
failure is FUNDAMENTAL, not a linear-combiner artifact.** Every measurement —
linear, co-sim with the shared book caps off, and co-sim with them on — fails
all-folds-positive on IS (2015 and 2018 are negative under every weighting; the
cap-on book adds 2017). The portfolio route is closed **cleanly**: the thinness is
real risk geometry, not an artifact of summing independently-simulated legs.

Per §4 holdout preservation: **IS did not clear, so the frozen 2021+ OOS was NOT
touched.** No PASS candidate; no `passed/` record; deployable-system count stays 0.
