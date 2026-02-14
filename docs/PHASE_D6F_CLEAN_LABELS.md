# Phase D-6F: Clean Opportunity Labels (Drawdown-Constrained)

## Overview

Phase D-6F generates a **clean** label family that measures favorable excursion **before** adverse excursion reaches a threshold. Unlike legacy labels, clean labels exclude paths where price hits an unacceptable drawdown before achieving the favorable move.

## What Legacy Labels Are

Legacy labels (`results/phaseD/labels/opportunity_labels.csv`) from Phase D-1 measure:

- **MFE (Max Favorable Excursion)** in R units over fixed horizons (10, 20, 40 bars)
- Zones: A (1R in 10), B (3R in 20), C (6R in 40)
- Reference: `ref_open = open[t+1]`, R = 1.5 × ATR(14) at bar t
- Unconditional: MFE is the max favorable move in the window regardless of drawdown

## What Clean Labels Measure

**"How much favorable excursion occurs BEFORE adverse excursion reaches X R."**

- Same ATR(14) at bar t; R = 1.5 × ATR_t
- `ref_px = open[t+1]` (next day open)
- For each X ∈ {1, 2, 3} and horizon H ∈ {10, 20, 40}:
  - Find first bar k where `adverse_r(k) ≥ X`
  - **If found:** `clean_mfe = max_{j < k} favorable_r(j)` (excludes breach bar)
  - **If not found:** `clean_mfe = max_{j ≤ H} favorable_r(j)`

## Why (Avoid Choppy Paths / Unacceptable Heat)

Legacy MFE can overstate opportunity on paths that:

1. Spike to 6R, then collapse to -3R before recovery
2. Require holding through large adverse swings

Clean labels identify **"clean"** moves: favorable excursion before the market shows X R of adverse pressure. This supports:

- Filtering choppy or unacceptable-heat paths
- Targeting entries with favorable path structure
- Diagnostic analysis only (not for live signals)

## Exact Math + Strict Cutoff Rule

### Excursions in R (per bar k in forward window)

**LONG:**

- `favorable_r(k) = (high[t+k] - ref_px) / R_value`
- `adverse_r(k) = (ref_px - low[t+k]) / R_value`

**SHORT:**

- `favorable_r(k) = (ref_px - low[t+k]) / R_value`
- `adverse_r(k) = (high[t+k] - ref_px) / R_value`

### Clean Metric

1. Find first k where `adverse_r(k) ≥ X`
2. If found at k: `clean_mfe = max_{j < k} favorable_r(j)` — **do not include breach bar**
3. If not found: `clean_mfe = max_{j ≤ H} favorable_r(j)`

### Zones

- `clean_zoneA_X`: `clean_mfe_before_mae_Xr_10 ≥ 1` (within 10 bars)
- `clean_zoneB_X`: `clean_mfe_before_mae_Xr_20 ≥ 3` (within 20 bars)
- `clean_zoneC_X`: `clean_mfe_before_mae_Xr_40 ≥ 6` (within 40 bars)

## Output

- **File:** `results/phaseD/labels/opportunity_labels_clean.csv`
- One row per (pair, date)
- Columns: `pair`, `date`, `ref_px`, `atr14_t`, `r_value`, validity flags, clean MFE metrics, zone flags
- Legacy labels remain untouched

## Commands

```bash
python scripts/phaseD6F_generate_clean_labels.py -c configs/phaseD/phaseD6F_clean_labels.yaml

python -m analytics.phaseD6F_clean_label_summaries --clean results/phaseD/labels/opportunity_labels_clean.csv --legacy results/phaseD/labels/opportunity_labels.csv --outdir results/phaseD/labels/clean_summaries
```

## Sanity Summaries

- `counts_clean_zoneC_by_year_pair.csv` — counts per year per pair for zone C
- `legacy_vs_clean_ordering.csv` — rate comparison (clean expected ≤ legacy)
- `clean_stability_discovery_vs_validation.csv` — discovery vs validation rates
- `clean_mfe_stats.csv` — distribution stats (count, mean, std, p50, p75, p90, p95, max)
