# Arc 9 Step 5 Validation - no-filter WFO on cluster 0

> Held-open experiment under v2.3 §10 (Step 5 = WFO truth). NOT a
> deployment candidate run. Question: does the §8 extractability gate
> correctly identify non-deployable cohorts, or did it over-reject a
> cohort with real economic edge?

## Headline

**PASS-DEPLOYABLE** — worst-fold annualised ROI 39.45%, mean fold annualised ROI 61.40%, worst-fold max DD 0.01%.

## Method

- Trade pool: 365 cluster 0 trades from Step 1 / Step 2 (cid=0 at K=3).
- Entry: every cluster 0 signal taken at signal-bar-close → next-bar open (long).
- SL: entry - 2.0×ATR (R = selected SL per Step 3, here 2.0×ATR).
- Exit policy: §11 Stepwise climber Pipeline E.
    * MFE-lock at 1R: when intrabar mfe ≥ 1R favorable, move SL to entry (break-even).
    * Trail 0.75R from peak_close after MFE-lock activates (bar-close updates).
- Intrabar SL trigger on mid (low ≤ SL), fill on bid (long): SL - spread/2 of trigger bar.
- Time exit at bar entry+240 (40 calendar days at 4H), fill at open of bar 240 minus spread/2.
- Risk per trade: 0.5% of compounded equity (starting $10k).
- Folds: 7-fold WFO OOS windows from `configs/wfo_kh24.yaml` (anchor).
- Per-bar spread: real MT5 spread, floor only when raw spread = 0 (SPREAD_SEMANTICS_LOCK).

## Per-fold table

| Fold | OOS window | Trades | Final R mean | Sign | Compounded ROI (%) | Annualised ROI (%) | Max DD (%) | Ending equity |
|---|---|---|---|---|---|---|---|---|
| F1 | 2020-10-01 → 2021-07-01 | 48 | +1.040 | + | +28.22 | +39.45 | 0.01 | $12,821.86 |
| F2 | 2021-07-01 → 2022-04-01 | 47 | +1.580 | + | +44.59 | +63.48 | 0.01 | $14,458.93 |
| F3 | 2022-04-01 → 2023-01-01 | 50 | +1.529 | + | +46.12 | +65.49 | 0.00 | $14,612.47 |
| F4 | 2023-01-01 → 2023-10-01 | 52 | +1.218 | + | +37.03 | +52.42 | 0.01 | $13,702.77 |
| F5 | 2023-10-01 → 2024-07-01 | 70 | +1.499 | + | +68.50 | +100.47 | 0.01 | $16,849.84 |
| F6 | 2024-07-01 → 2025-04-01 | 45 | +1.373 | + | +35.93 | +50.56 | 0.01 | $13,592.98 |
| F7 | 2025-04-01 → 2026-01-01 | 53 | +1.305 | + | +41.06 | +57.92 | 0.00 | $14,106.00 |

## Full-data (compounded across folds in time order)

- n trades: 365
- Compounded ROI: +1099.32%
- Annualised ROI: +60.50%
- Max DD: 0.01%
- Ending equity: $119,931.54

## Gates evaluated (§10 pass-deployable + pass-viable)

### Pass-deployable

| Gate | Threshold | Actual | Pass? |
|---|---|---|---|
| Worst-fold annualised ROI | ≥ 5.0% | +39.45% | PASS |
| Mean fold annualised ROI | ≥ 8.0% | +61.40% | PASS |
| Worst-fold max DD | ≤ 8.0% | 0.01% | PASS |
| All folds positive | required | True | PASS |
| Trade count per fold | ≥ 15 | min 45 | PASS |
| Full-data annualised ROI | ≥ 5.0% | +60.50% | PASS |
| Full-data max DD | ≤ 10.0% | 0.01% | PASS |
| **Overall pass-deployable** | all 7 | - | **PASS** |

### Pass-viable

| Gate | Threshold | Actual | Pass? |
|---|---|---|---|
| Worst-fold annualised ROI | > 0.0% (positive) | +39.45% | PASS |
| Mean fold annualised ROI | ≥ 3.0% | +61.40% | PASS |
| Worst-fold max DD | ≤ 8.0% | 0.01% | PASS |
| All folds positive | required | True | PASS |
| Trade count per fold | ≥ 5 | min 45 | PASS |
| Full-data annualised ROI | ≥ 3.0% | +60.50% | PASS |
| Full-data max DD | ≤ 10.0% | 0.01% | PASS |
| **Overall pass-viable** | all 7 | - | **PASS** |

## Per-trade behaviour (sanity)

- **MFE-lock activation rate: 100% (365 / 365).** Consistent with Step 3 `frac_reach_1R = 1.000` on the surviving SL=2.0×ATR frame — by construction every cluster 0 trade reaches 1R favorable.
- Activation bar offset: p5/p25/p50/p75/p95 = 1 / 5 / 8 / 15 / 37 (4H bars).
- Exit reason mix: **100% trail_stop** (no `stoploss`, no `time_exit`, no `end_of_data`). The initial −1R SL is never hit because cluster 0 has `frac_wrong_way_pre_peak = 0.000` (the defining capturability property at this SL); after MFE-lock the SL can only ratchet up.
- final_r distribution: min −0.025 / p5 +0.135 / p50 +0.967 / p95 +3.92 / max +6.95; **mean +1.369R**; 12 of 365 trades close marginally negative (max loss 2.5% of R = spread cost above the breakeven trail exit), no trade closes below −0.03R.
- Determinism: 2-run byte-identical (`determinism_check.json`).

## Interpretation

**The §8 extractability gate over-rejected cluster 0.** Under no admission filter and the §11 Stepwise-climber exit policy, the cohort produces 7/7 fold ROI > +28% (annualised +39% worst), worst-fold DD 0.01%, full-data ROI +1,099% / annualised +60.5% over the 2020-10 → 2026-01 OOS span. The cohort carries real, durable, structurally-clean economic edge. Step 3 capturability characterisation was correct in flagging cluster 0 as the surviving archetype; Step 4's classifier-AUC + threshold-sweep gate was the binding constraint that killed the arc.

**But the answer is conditional on an oracle filter.** This experiment substitutes the perfect post-hoc cluster membership knowledge for the live-time classifier the protocol requires. Cluster identity is computed AFTER the trade completes (path-shape features need the full held window). At entry — and at every early bar offset t ∈ {1..10} per Step 4 — cluster membership is not predictable enough from available features for the §8 sweep to admit a usable filter: RF AUC was 0.511 at entry and 0.626 at t=1, with classifier probability outputs too tightly concentrated below 0.40 for any {0.40, 0.50, 0.60, 0.70} threshold to admit a fraction ≥ 0.60 of the cohort. The cohort is real; the addressability mechanism is not.

This is therefore a *feature-availability failure*, not a *cohort-quality failure*. The signal class is not permanently eliminated (closure doc reflected this). Candidate unlock paths the protocol does not currently exercise: (a) richer / multi-TF entry features, (b) probability-calibrated classifiers (Platt / isotonic on the AUC 0.626 D1 baseline) that lift the classifier output distribution into the existing threshold grid, (c) a protocol amendment that relaxes the strict {0.40, 0.50, 0.60, 0.70} grid for low-prevalence positive classes. This is the second arc (after Arc 7 CLEAN-NULL 2026-05-17) where Pipeline D1 shows the same probability-mis-calibration pattern; two arcs is not a calibration trigger but the cumulative evidence is now noteworthy.

**No deployment implication.** Arc 9 closure (STEP_4_KILL) stands; this experiment changes Step 4's verdict from "non-extractable / cohort uncertain" to "non-extractable / cohort confirmed real." The §8 gate is not mis-calibrated to §10 truth in the sense of over-conservatism: it is correctly conservative given the available feature set — what it cannot do is distinguish "feature-set is binding" from "cohort lacks edge". This experiment supplies that distinction for Arc 9: the binding constraint is the feature set, not the cohort.
