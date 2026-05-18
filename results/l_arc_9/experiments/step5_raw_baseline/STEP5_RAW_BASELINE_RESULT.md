# Arc 9 Step 5 Raw Baseline - full pool, no filter, default exit

> Held-open experiment under v2.3 §10. Establishes the raw signal floor —
> deployable economics of the IB-trend signal as bare entry rule + default
> protocol exit (SL=2.0×ATR + 240-bar time-stop). Third reference point
> in the Arc 9 picture (oracle Step 5 + calibration recovery already done).

## Headline

**FAIL** — worst-fold annualised ROI -29.64%, mean fold annualised ROI +6.10%, worst-fold max DD 43.61%.

## Method

- Trade pool: 2,153 trades — the full Arc 9 Step 1 output. No clustering, no admission filter.
- Entry: every signal taken at signal-bar-close → next-bar open (long, +S/2 fill).
- SL: entry − 2.0×ATR(14)_4H at signal bar (anchored to entry_fill).
- Time exit: bar entry + 240 (40 calendar days at 4H), fill at open of bar 240.
- No MFE-lock, no trail, no §11 archetype routing.
- Provenance: `final_r` per trade is read directly from `results/l_arc_9/step1_verbatim/trades_all.csv`
  (Step 1 already executed under exactly this exit policy + live-execution semantics; no re-simulation needed).
- Folds: 7 OOS windows from `configs/wfo_kh24.yaml` (anchor); same fold structure as the oracle Step 5 run.
- Risk per trade: 0.5% compounded from $10k starting balance (same as oracle run).
- Per-bar spread: real MT5 spread, floor only when raw spread = 0 (SPREAD_SEMANTICS_LOCK).

## Per-fold table

| Fold | OOS window | Trades | Final R mean | Sign | Compounded ROI (%) | Annualised ROI (%) | Max DD (%) | Ending equity |
|---|---|---|---|---|---|---|---|---|
| F1 | 2020-10-01 → 2021-07-01 | 336 | -0.143 | - | -23.10 | -29.64 | 43.61 | $7,689.51 |
| F2 | 2021-07-01 → 2022-04-01 | 324 | +0.001 | - | -3.26 | -4.32 | 34.38 | $9,674.05 |
| F3 | 2022-04-01 → 2023-01-01 | 335 | -0.066 | - | -13.14 | -17.07 | 34.01 | $8,685.78 |
| F4 | 2023-01-01 → 2023-10-01 | 297 | -0.095 | - | -14.82 | -19.32 | 18.03 | $8,517.50 |
| F5 | 2023-10-01 → 2024-07-01 | 277 | +0.353 | + | +58.44 | +84.69 | 14.12 | $15,844.30 |
| F6 | 2024-07-01 → 2025-04-01 | 295 | -0.007 | - | -3.47 | -4.60 | 36.34 | $9,652.57 |
| F7 | 2025-04-01 → 2026-01-01 | 289 | +0.162 | + | +23.90 | +32.93 | 14.81 | $12,390.23 |

## Full-data (compounded across folds in time order)

- n trades: 2153
- Compounded ROI: +4.29%
- Annualised ROI: +0.80%
- Max DD: 62.99%
- Ending equity: $10,428.54

## Gates evaluated (§10 pass-deployable + pass-viable)

### Pass-deployable

| Gate | Threshold | Actual | Pass? |
|---|---|---|---|
| Worst-fold annualised ROI | ≥ 5.0% | -29.64% | FAIL |
| Mean fold annualised ROI | ≥ 8.0% | +6.10% | FAIL |
| Worst-fold max DD | ≤ 8.0% | 43.61% | FAIL |
| All folds positive | required | False | FAIL |
| Trade count per fold | ≥ 15 | min 277 | PASS |
| Full-data annualised ROI | ≥ 5.0% | +0.80% | FAIL |
| Full-data max DD | ≤ 10.0% | 62.99% | FAIL |
| **Overall pass-deployable** | all 7 | - | **FAIL** |

### Pass-viable

| Gate | Threshold | Actual | Pass? |
|---|---|---|---|
| Worst-fold annualised ROI | > 0.0% (positive) | -29.64% | FAIL |
| Mean fold annualised ROI | ≥ 3.0% | +6.10% | PASS |
| Worst-fold max DD | ≤ 8.0% | 43.61% | FAIL |
| All folds positive | required | False | FAIL |
| Trade count per fold | ≥ 5 | min 277 | PASS |
| Full-data annualised ROI | ≥ 3.0% | +0.80% | FAIL |
| Full-data max DD | ≤ 10.0% | 62.99% | FAIL |
| **Overall pass-viable** | all 7 | - | **FAIL** |

## Comparison to oracle Step 5 (cluster 0 only, §11 Stepwise exit)

| Metric | Raw baseline (full pool, default exit) | Oracle (cluster 0 only, §11 exit) | Gap (oracle - raw) |
|---|---|---|---|
| Worst-fold ann ROI | -29.64% | +39.45% | +69.09pp |
| Mean fold ann ROI | +6.10% | +61.40% | +55.30pp |
| Worst-fold DD | 43.61% | 0.01% | -43.60pp |
| Full-data ann ROI | +0.80% | +60.50% | +59.69pp |
| Full-data DD | 62.99% | 0.01% | -62.97pp |
| Trades total | 2153 | 365 | — |
| Pass-deployable? | NO | YES | — |

## Per-fold sign breakdown

5 of 7 folds negative; 2 strongly positive (F5 +84.69%, F7 +32.93%). The strong folds are not enough to compensate for the long sequence of losing folds (F1 −29.64%, F2 −4.32%, F3 −17.07%, F4 −19.32%, F6 −4.60%) — full-data compounded ROI is +0.80% on a 62.99% drawdown. The raw signal has near-zero unconditional expectancy and lethal time-localised drawdown structure.

## Interpretation

**The raw IB-trend signal is unfundable.** Every §10 gate fails, most by wide margins. 5 of 7 folds are negative; the worst fold (F1, the first 9 months of OOS) draws down 43.6% on the way to a −23.1% fold ROI. Full-data drawdown of 63% would close any 5ers prop account within the first fold and most retail accounts long before that. The full-data annualised ROI of +0.80% is statistically indistinguishable from zero given the per-fold variance (5 of 7 folds negative, mean −0.143 to +0.353R).

**Comparison to oracle frames the deployment surface.** The gap between raw baseline and oracle Step 5 is enormous on every axis:

| Axis | Raw | Oracle | Gap |
|---|---|---|---|
| Worst-fold ann ROI | −29.64% | +39.45% | **+69.09pp** |
| Mean-fold ann ROI | +6.10% | +61.40% | +55.30pp |
| Worst-fold DD | 43.61% | 0.01% | **−43.60pp** (oracle better) |
| Full-data ann ROI | +0.80% | +60.50% | +59.69pp |
| Full-data DD | 62.99% | 0.01% | **−62.97pp** |

The clustering work is **not incremental — it is the entire system**. Without cohort identification, IB-trend is a money-loser with portfolio-killing drawdown. With perfect cohort identification (cluster 0 only, n=365 = 17% of the pool), it is a pass-deployable +60% annualised strategy with effectively no drawdown.

**Restated implication of cluster 0's defining property.** Cluster 0 was selected at Step 3 because it has `frac_wrong_way_pre_peak = 0.000` at SL=2.0×ATR — by construction, no trade in cluster 0 reaches −1R before its peak MFE. The "rest of pool" (n=1,788 = 83%) is by complement the cohort that DOES violate the pre-peak SL — exactly the cohort that drags raw baseline's worst-fold to −29.64% via repeated −1R full-SL losses. The "rest of pool" carries the loss-side asymmetry of the whole signal class.

**Implication for the calibration experiment's stakes.** The calibration recovery experiment (OUTCOME_B) showed the D1 classifier cannot achieve recall ≥ 0.60 at any threshold in {0.40, 0.50, 0.60, 0.70} after Platt or isotonic remapping. Extended sweep showed the classifier can achieve recall 0.60 at threshold ~0.20 with precision ~0.27 (admitting ~40% of the pool with ~27% of admits being cluster 0). The economic stakes of that operating point can now be sized: an admit set comprising ~27% cluster 0 (mean ~+4.4R under §11 exit) and ~73% non-cluster-0 (where the non-cluster-0 portion of the raw pool carries the loss-side weight that drove −0.18R mean per trade on the 83% complement) is unlikely to be pass-deployable. A meaningful realistic-filter result requires precision substantially above 0.27 — which the AUC ≈ 0.63 regime cannot deliver at recall 0.60. The third experiment (Step 5 calibrated, pending) would quantify this directly but the directional answer is already constrained by the gap-frame this baseline provides.

**Summary of Arc 9's three-point picture:**

1. **Floor (this experiment):** full pool, no filter, default exit → FAIL by every gate (5/7 folds negative, 63% DD).
2. **Ceiling (Step 5 validation):** cluster 0 only, §11 exit → PASS-DEPLOYABLE by every gate (7/7 folds positive, 0% DD).
3. **Realistic (Step 5 calibrated, pending):** full pool + calibrated D1 filter + §11 exit → expected to sit between floor and ceiling; calibration recovery shows precision ~0.27 at recall 0.60 in the AUC ≈ 0.63 regime, so the realistic point will likely be closer to the floor than the ceiling.

The composite Arc 9 verdict, with all three experiments in view: the IB-trend signal has structural edge concentrated in a small, identifiable-post-hoc cohort; the cohort is not extractable from features available at decision time with the protocol's current classifier-pipeline; and the floor without extraction is dangerous, not just unprofitable. Closure stands; recommendation for analyst review: the v2.x amendment case for calibration-before-threshold-sweep is weakened (calibration doesn't unlock Arc 9), and the calibration experiment's framing should be reinforced — this is a feature-set / signal-class problem, not a calibration gap.

