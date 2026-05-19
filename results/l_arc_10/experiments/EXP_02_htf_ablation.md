# EXP-02 — HTF feature ablation

**Status:** experimental (not a Step 5 gate).

## Question
Which of {L_1, L_0, age, slope} drives the AUC lift over the entry-time-only baseline?

## Method
- c1 cohort: n=228 trades, base success 0.4825.
- Walk-forward 5-fold TimeSeriesSplit, RF seed=42 (matches Step 4).
- HTF buckets:
  - **L_1**: `L1_to_atr_proximity, reject_buffer_atr`
  - **L_0**: `L1_minus_L0_atr`
  - **age**: `L1_age_d1_bars, L0_age_d1_bars, L1_minus_L0_d1_bars`
  - **slope**: `upper_fraction`
- Variants: full E, full E minus each bucket (LOO), full E minus all HTF (base).
- Pipeline D1 reported for context — D1 feature set does not include the
  signal-bar HTF features so bucket ablation does not apply.

## Variant AUCs

| variant | pipeline | n_feat | mean AUC | std | gate | margin | pass |
|---|---|---:|---:|---:|---:|---:|:---:|
| full_E | E | 25 | 0.6296 | 0.0468 | 0.65 | -0.0204 | no |
| no_L_1 | E | 23 | 0.6107 | 0.0490 | 0.65 | -0.0393 | no |
| no_L_0 | E | 24 | 0.6019 | 0.0228 | 0.65 | -0.0481 | no |
| no_age | E | 22 | 0.6254 | 0.0287 | 0.65 | -0.0246 | no |
| no_slope | E | 24 | 0.6112 | 0.0365 | 0.65 | -0.0388 | no |
| base_no_HTF | E | 18 | 0.6057 | 0.0562 | 0.65 | -0.0443 | no |
| full_D1_reference | D1 | 11 | 0.5897 | 0.0959 | 0.60 | -0.0103 | no |

## Attribution — Pipeline E HTF buckets

- Full E mean AUC: **0.6296**
- Base (no HTF) E mean AUC: **0.6057**
- Total HTF lift: **+0.0239**

| bucket | features | AUC without bucket | LOO drop from full | % of total HTF lift |
|---|---|---:|---:|---:|
| L_0 | `L1_minus_L0_atr` | 0.6019 | +0.0277 | 115.7% |
| L_1 | `L1_to_atr_proximity,reject_buffer_atr` | 0.6107 | +0.0189 | 79.0% |
| slope | `upper_fraction` | 0.6112 | +0.0184 | 77.0% |
| age | `L1_age_d1_bars,L0_age_d1_bars,L1_minus_L0_d1_bars` | 0.6254 | +0.0042 | 17.6% |

## Interpretation
- HTF features collectively contribute +0.0239 to mean E AUC.
- The `L_0` bucket carries **115.7%** of that lift — candidate for cross-arc feature promotion if Arc 8/9/11 show a similar pattern.

## Artefacts
- `raw/exp_02_ablation_summary.csv` (sha256 `8c457253d1c33672…`)
- `raw/exp_02_per_fold_auc.csv` (sha256 `0dcd111e7b5bedec…`)
- `raw/exp_02_attribution.csv` (sha256 `a3713613e7db2fec…`)

