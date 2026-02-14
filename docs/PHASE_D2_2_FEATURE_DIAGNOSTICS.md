# Phase D-2.2: Opportunity Precursor Feature Diagnostics

## Overview

Phase D-2.2 is a **feature diagnostics pipeline** that discovers which causal market properties at time `t` correlate with future Zone B/C opportunity (truth labels from Phase D-1). This is **not** trading, indicator evaluation in the ROI sense, or WFO.

**Goal:** Generate a ranked, reproducible report of feature families that show stable discriminative lift from Discovery → Validation.

## What These Diagnostics Mean

- **Lift:** A feature increases the probability of Zone B or Zone C (same direction) compared to baseline when the feature falls in a high-value bin.
- **Stability:** The lift that appears in discovery also appears in validation when using the same (frozen) bin edges.
- **Effect size (Cohen's d):** Standardized difference between feature distributions for label-positive vs label-negative rows.
- **AUC proxy:** Rank-based AUC (Mann–Whitney U / (n_pos × n_neg)) — how well the feature separates positive from negative labels.

## What These Diagnostics Do NOT Mean

- **No ROI, PnL, expectancy:** We do not measure profitability.
- **No backtesting:** No SL/BE/trailing simulation.
- **No indicator selection:** Features are not scored by profits.
- **No WFO:** No walk-forward optimization.
- **No ML:** No trained models; purely deterministic, interpretable metrics.
- **No lookahead:** All features use only bars ≤ t.

## Interpretation

### Stability Across Split

- **Good:** `lift_disc` and `lift_val` similar; `stability_gap` small.
- **Unstable:** Large `stability_gap` (e.g. > 0.2) — discovery lift does not hold in validation.
- **Insufficient sample:** `n_pos_validation < 200` (Zone B) or `< 100` (Zone C) — results may be noisy.

### Scoring

- **Score** = `min(lift_disc, lift_val) × (1 - min(stability_gap, 1.0))`
- Higher score = more stable, modest lift. We prefer modest lift that persists over high lift that vanishes.

## Outputs

| Path | Description |
|------|-------------|
| `results/phaseD2_2/features/features.parquet` | Feature dataset (pair, date, all features) |
| `results/phaseD2_2/reports/feature_summary_global.csv` | Summary stats (global, split, target) |
| `results/phaseD2_2/reports/feature_summary_split.csv` | Split-specific summaries |
| `results/phaseD2_2/reports/feature_summary_by_pair.csv` | Per-pair summaries (optional) |
| `results/phaseD2_2/reports/feature_rankings.csv` | Top 20 features per Zone B/C |
| `results/phaseD2_2/reports/feature_rankings.json` | Same as CSV, JSON format |
| `results/phaseD2_2/run_manifest.json` | Run metadata (hashes, config, timestamp) |

## Run

```bash
python scripts/phaseD2_2_run_feature_diagnostics.py -c configs/phaseD/phaseD2_2_feature_diagnostics.yaml
```

**Prerequisites:**

- `results/phaseD/labels/opportunity_labels.parquet` (Phase D-1 labels)
- OHLC data in `data/daily/<PAIR>.csv` for each configured pair

## Config

- `configs/phaseD/phaseD2_2_feature_diagnostics.yaml`
- Locked: date range 2019-01-01 → 2026-01-01, discovery split end 2022-12-31
