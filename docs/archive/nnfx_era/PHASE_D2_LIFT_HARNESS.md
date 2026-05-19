# Phase D-2: Predictive Lift Harness

## What D-2 Does

Phase D-2 measures **whether a signal fired at time t increases the probability of a Zone A/B/C opportunity** in the same direction. It is strategy-agnostic and does not run backtests.

- **Inputs**: Opportunity labels (Phase D-1) + signal events
- **Outputs**: Lift, coverage, base rates, and hit rates per signal and zone

## What D-2 Does NOT Do

- ❌ ROI, DD, expectancy, PnL, or backtests
- ❌ SL/BE/trailing simulation
- ❌ Walk-forward optimization
- ❌ Plots or visuals
- ❌ Parameter optimization
- ❌ Indicator evaluation (real indicators may be wired as optional adapters later)

## Metrics

For a signal with fire rate F/N:

| Metric | Definition |
|--------|------------|
| Base rate P(A) | Proportion of rows where zone_a_1r_10 == 1 |
| Hit rate P(A\|fire) | Mean of A over rows where signal == 1 |
| Lift A | P(A\|fire) / P(A) |
| Coverage A | count(fire AND A) / count(A) |
| Fire rate | F / N |

Similar for zones B and C.

## Control Signals

Three control signals validate harness correctness:

- **always_fire**: signal=1 for all rows → lift ≈ 1.0, coverage = 1.0
- **random_fire_p**: fires at probability p (deterministic via hash) → lift ≈ 1.0
- **oracle_zone_X**: fires when zone is true (test only) → P(X\|fire)=1, coverage_X=1

## Running

```bash
# Prerequisite: Phase D-1 labels at results/phaseD/labels/opportunity_labels.parquet
python scripts/phaseD2_run_lift_harness.py -c configs/phaseD/phaseD2_lift_harness.yaml
```

## Outputs

All under `results/phaseD2/`:

- `joined/signals_x_labels.parquet` — joined dataset for auditing
- `metrics/metrics_by_signal_global.csv`
- `metrics/metrics_by_signal_split.csv`
- `metrics/metrics_by_signal_pair.csv`
- `metrics/coverage_by_signal.csv`
- `run_manifest.json` — hashes and determinism fingerprints

## Tests

```bash
python -m pytest -q tests/test_phaseD2_lift_harness.py
```
