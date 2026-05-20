# EXP-01 — AUC bootstrap confidence intervals

**Status:** experimental (not a Step 5 gate).

## Question
Is c1 E AUC 0.6296 statistically distinguishable from the 0.65 threshold?
Is D1 AUC 0.5897 statistically distinguishable from 0.60?

## Method
- c1 cohort: n=228 trades, base success rate 0.4825.
- Walk-forward 5-fold TimeSeriesSplit (matches Step 4); RF seed=42.
- Pool out-of-fold predictions across folds: n_oof_pairs=190.
- Bootstrap-resample (p, y) pairs WITH replacement n=2000 times,
  paired resamples (same indices for E and D1) so joint P is exact.
- Note: per-fold-AUC bootstrap is intentionally not done — re-fitting 2000
  RFs would add model-variance noise without changing the qualitative answer.
  AUC sampling-distribution bootstrap on OOF (p, y) is the standard estimator.

## Results

| Pipeline | Point estimate (mean per-fold) | OOF pooled AUC | Bootstrap mean | 95% CI | Threshold | P(AUC >= threshold) | Step 4 margin |
|---|---:|---:|---:|---|---:|---:|---:|
| E  | 0.6296 | 0.6026 | 0.6014 | [0.5227, 0.6839] | 0.65 | 0.1250 | -0.0204 |
| D1 | 0.5897 | 0.5831 | 0.5823 | [0.5029, 0.6628] | 0.60 | 0.3380 | -0.0103 |

### Joint distribution (paired resamples)

- P(E AUC >= 0.65 OR D1 AUC >= 0.6) = **0.4050**
- P(E AUC >= 0.65 AND D1 AUC >= 0.6) = 0.0580
- n paired resamples: 2000 of 2000

## Interpretation
- Pipeline E is **inside the 95% CI band around the threshold** — P(AUC>=0.65) = 0.1250; not statistically distinguishable from threshold.
- Pipeline D1 is **inside the 95% CI band around the threshold** — P(AUC>=0.60) = 0.3380; not statistically distinguishable from threshold.

## Artefacts
- `raw/exp_01_bootstrap_auc_E.csv` (sha256 `2a4f80ede60c2f05…`)
- `raw/exp_01_bootstrap_auc_D1.csv` (sha256 `cab908b5f60ce811…`)
- `raw/exp_01_summary.json` (sha256 `431394070b4098cf…`)

