# EXP-06 — Open-04 plausibility probe (INFORMATIONAL)

**Status:** experimental, INFORMATIONAL ONLY. Does NOT propose commission. 
Open-04 commission is a separate v2.4 governance decision.

## Question
Do one or two external candidate features add measurable E AUC, holding the
HTF + base features constant?

## Candidates probed
1. **d1_kijun_dist_atr** — `(D1_close − D1_Kijun(26)) / D1_ATR(14)`, one-day-lagged.
   Entry-time observable; matches KH-24 anchor's D1 Kijun regime convention.
2. **session dummies** — 4 one-hot dummies (Asia / London / NY / late NY) from entry hour.
   Entry-time observable; matches the standard FX volatility-regime taxonomy.

## Results

| variant | n_feat | mean AUC | std | margin vs 0.65 | per-fold AUC |
|---|---:|---:|---:|---:|---|
| baseline (25 features) | 25 | 0.6296 | 0.0468 | -0.0204 | 0.6247, 0.6695, 0.5512, 0.6464, 0.6562 |
| baseline + d1_kijun_dist_atr | 26 | 0.6150 | 0.0575 | -0.0350 | 0.6524, 0.6331, 0.5429, 0.5679, 0.6790 |
| baseline + session dummies | 29 | 0.6081 | 0.0312 | -0.0419 | 0.6122, 0.6275, 0.5983, 0.5607, 0.6420 |
| baseline + both candidates | 30 | 0.6193 | 0.0507 | -0.0307 | 0.6233, 0.6303, 0.5789, 0.5679, 0.6960 |

## Interpretation (informational only)
- **baseline + d1_kijun_dist_atr**: delta vs baseline -0.0146; still below 0.65.
- **baseline + session dummies**: delta vs baseline -0.0215; still below 0.65.
- **baseline + both candidates**: delta vs baseline -0.0103; still below 0.65.

**Reminder:** these numbers are evidence for v2.4 calibration discussion, not a
commission proposal. Any positive lift here would need:
- Reproduction on Arc 8 / 9 / 11 (when their step1+step4 outputs land);
- A formal Open-04 commission proposal with cross-arc evidence and feature-class definition;
- v2.4 governance review.

## Caveats
- n=228 with 5 folds → per-fold AUC variance dominates; single-experiment AUC deltas of ≤0.02 are noise-class.
- Session dummies overlap with the existing `hour_sin / hour_cos` cyclic features; RF should down-weight redundancy but cannot eliminate it.
- `d1_kijun_dist_atr` uses one-day-lagged D1 data per the canonical convention; lag verified by the D1 lag audit in Step 4.

## Artefacts
- `raw/exp_06_probe_results.csv` (sha256 `b04ec9bed94f4e76…`)

