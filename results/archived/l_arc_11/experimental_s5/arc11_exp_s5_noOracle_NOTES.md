# Arc 11 — Experimental Step 5 WFO, NO-ORACLE (off-protocol, documentation only)

> **Status:** documentation only. Arc 11 remains **Closed-HALT** per §16a Path A regardless of this script's outcome. No queue / registry / protocol mutation.
> Companion to `arc11_exp_s5_NOTES.md` (oracle version A/B); this run removes the cluster-ID-at-entry oracle.

## What changed vs Runs A / B

| Aspect | A / B (oracle) | C / D (no oracle) |
|---|---|---|
| Universe | c1 only (A) or agg_c1_c3 only (B), assumed known at entry | Full 2,299-trade SHB pool, every signal a candidate |
| Admission filter | None (A) or D1 classifier only (B) | Live E classifier predicting cluster membership (C, D Stage 1), plus D1 t=5 cascade (D Stage 2) |
| Cluster ID at entry | Oracle (cluster_id known from post-hoc Step 2 clustering) | NOT known — has to be predicted by E classifier from entry-bar features |

## Comparison table — A, B, C (baseline + best), D

| Run | Description | Oracle? | Sign-consist | Worst-fold ROI ann % | Mean-fold ROI ann % | Worst-fold DD % | Min trades | Full-data ROI % | Full-data DD % | Pass-deployable |
|---|---|:---:|:---:|---:|---:|---:|---:|---:|---:|:---:|
| **A** | c1 raw at SL=3 | YES | YES | 101.39 | 152.12 | 2.48 | 42 | 13,006.57 | 2.48 | **YES** |
| **B** | agg_c1_c3 + D1 t=5, SL=3 | YES | YES | 26.30 | 53.33 | 4.95 | 114 | 704.89 | 4.95 | **YES** |
| **C** | Live E → c1 (t=0.50 baseline) | NO | no | −20.22 | −2.36 | 33.46 | 1 | −11.91 | 33.46 | **no** |
| **C-best** | Live E → c1 (t=0.30 best) | NO | no | −20.22 | +4.78 | 33.46 | 28 | +17.30 | 39.75 | **no** |
| **D** | Live E → D1 t=5 cascade | NO | no | −22.83 | −2.76 | 33.46 | 4 | −15.55 | 35.08 | **no** |

**Oracle premium (mean-fold ROI ann, no-oracle minus oracle):**
- c1 leg: −154.48 pp (A 152.12% → C 0.50 baseline −2.36%; with best-threshold C 0.30, −147.34 pp)
- agg leg: −56.09 pp (B 53.33% → D −2.76%)

Both no-oracle runs FAIL pass-deployable on multiple gates simultaneously: sign-consistency (fold 1 always negative), worst-fold DD (33%+ everywhere), trade count (1–28 min per fold), full-data DD (33–40%). No combination of admission threshold rescues the result.

## Run C — threshold sweep (target c1, live E classifier)

| Threshold | Sign-consist | Worst-fold ROI ann % | Mean-fold ROI ann % | Worst-fold DD % | Min trades | Pass-deployable |
|---:|:---:|---:|---:|---:|---:|:---:|
| 0.30 | no | −20.22 | **+4.78** | 33.46 | 28 | no |
| 0.40 | no | −20.22 | −2.30 | 33.46 | 3 | no |
| 0.50 | no | −20.22 | −2.36 | 33.46 | 1 | no |
| 0.60 | no | −20.22 | −20.22 | 33.46 | 368 | no |
| 0.70 | no | −20.22 | −20.22 | 33.46 | 368 | no |

Notes:
- Worst-fold ROI is **identical (−20.22%) across all thresholds.** That's because fold 1 (Oct 2020 – Jul 2021) has zero IS data; the E classifier can't train; admission falls back to admit-all (368 trades = raw SHB pool at SL=3, which loses heavily because c0/c2 cluster trades drag the mix). No threshold setting can repair the cold-start fold.
- Mean-fold ROI is best at t=0.30 (+4.78%), but still below the pass-deployable 8% mean threshold AND fails sign-consistency (worst-fold −20%).
- At t=0.60 / 0.70 the classifier admits 0 trades in folds 2–7; only fold 1's 368 admit-all trades count → min = 368 in the aggregate but only one fold contributes any trades; mean = worst = −20.22%.

## Run D — stage attrition (cascade)

| Fold | n_universe | n_admit_E | E admit % | E precision on agg % | n_pre_t | n_admit_D1 | D1 admit % of E |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 368 | 368 | 100.0 | 36.14 | 31 | 337 | 91.6 |
| 2 | 333 | 34 | 10.2 | 41.18 | 3 | 11 | 32.4 |
| 3 | 370 | 16 | 4.3 | 37.50 | 1 | 6 | 37.5 |
| 4 | 345 | 19 | 5.5 | 36.84 | 1 | 7 | 36.8 |
| 5 | 281 | 7 | 2.5 | 42.86 | 0 | 1 | 14.3 |
| 6 | 303 | 4 | 1.3 | 100.00 | 0 | 1 | 25.0 |
| 7 | 299 | 11 | 3.7 | 54.55 | 1 | 0 | 0.0 |

Cumulative across folds: universe (2299) → E-admit (459, 20.0%) → pre-t losses (37) + D1-admit (363, 79.1% of E-admit).

Observations:
- **E precision on agg membership is 36–43% across folds 2–7** (fold 6 hits 100% but only on 4 trades). The agg base rate in the universe is 38.6% — meaning the live E classifier is **at or barely above random** at identifying agg membership. This matches the S4 5-fold TSCV AUC of 0.514.
- **Heavy E filtering** (1–10% admit rate folds 2–7) starves the D1 stage of data; folds 5/6/7 admit just 7/4/11 OOS trades for D1 evaluation.
- **Fold 7 D1 admits 0 trades** of 11 E-admits — the classifier rejects everything. Combined with pre-t losses, fold 7 contributes near-zero or net-negative.
- **Fold 1 is the universal disaster** (no IS data, admit-all). It dominates worst-fold DD across all configurations.

## Commentary — what no-oracle reality looks like

1. **The S4 AUC gate was right.** Pipeline E AUC 0.42 (c1) and 0.51 (agg) translates directly into a deployed system that can't pass-deployable on a single gate. The classifier-membership prediction is so close to random that any filtering it does adds little; relaxing the threshold (t=0.30) buys positive mean ROI but fails sign-consistency and DD. Tightening (t=0.60/0.70) starves the system. There is no operating point that gets to deployable.
2. **The capturable-not-extractable framing holds when both legs are live.** The cohort edge from S3 (c1 mfe_p50 4.48R, reach_1R 100%) is genuinely there in the data — Run A's 152% mean ROI proves it. But that edge is inaccessible at decision time because the entry-time feature regime can't predict cluster membership above random. The 154pp oracle premium is the exact measure of "what extractability is protecting against": deploying a system that looks great when graded on the post-hoc-known cohort and falls apart when graded on the live-decision-time cohort.
3. **Cold-start fold is unfair but informative.** Fold 1 (Oct 2020 – Jul 2021) has no IS data for this signal — Arc 11's data window starts right at fold 1's OOS start. Real deployment would just not turn the system on until IS data accumulates. But even removing fold 1, folds 2–7 of Run C-best have mean R 0.03–1.5 with worst-fold DD 8–15% — still doesn't clear pass-deployable. Cold-start makes the headline DD worse, but the body of the WFO is also weak.
4. **The Arc 6 pairing now sharpens.** Both arcs (Arc 6 failed-breakout, Arc 11 SHB) close S4 capturable-not-extractable HALT. The no-oracle S5 evidence here confirms: this isn't a near-miss that should clear; it's a structural ceiling on the §8 feature regime for trend-continuation breakout cohorts. Calibration candidate (from Arc 6 + Arc 11): feature-set extension (multi-TF, order-flow proxies, regime conditioners, ensembles) is the bottleneck, not threshold-relaxation. Both arcs would still HALT — and the no-oracle WFO would still fail — at AUC thresholds below ~0.60.
5. **Prior NOTES.md claim revised.** The earlier note suggested "AUC gate may be too conservative for high-magnitude cohorts; pair with `mfe_p50 ≥ 3R` to relax." That claim is **not supported** once the oracle is removed: even with cohort `mfe_p50 = 4.48R`, the no-oracle live system FAILS on every gate. The AUC gate is doing exactly the job it should. Strike that protocol amendment candidate from any synthesis doc.

## Artefacts

- `arc11_exp_s5_noOracle_C_threshold_sweep.csv` — Run C aggregate metrics at thresholds 0.30/0.40/0.50/0.60/0.70
- `arc11_exp_s5_noOracle_C_per_fold.csv` — Run C per-fold detail at t=0.50 baseline
- `arc11_exp_s5_noOracle_C_per_fold_best_t0.3.csv` — Run C per-fold detail at t=0.30 (best operating point)
- `arc11_exp_s5_noOracle_D_per_fold.csv` — Run D per-fold detail (cascade)
- `arc11_exp_s5_noOracle_D_stage_attrition.csv` — Run D stage breakdown (n_universe → n_admit_E → n_admit_D1)
- `comparison_table.csv` — A, B, C (baseline + best), D side-by-side on identical metric columns
- `arc11_exp_s5_noOracle_NOTES.md` — this doc
- `scripts/l_arc_11/experimental_s5_no_oracle_wfo.py` — runner

## Disposition

**Arc 11 remains Closed-HALT.** Strengthens (not weakens) the §16a Path A HALT decision: AUC margin 0.027 was the right call. The no-oracle WFO is the deployment truth — Arc 11 does not deploy.

Cross-arc finding for the capturable-not-extractable write-up (with Arc 6): when extractability gates fail with AUC < 0.55, the no-oracle live deployment also fails by a wide margin. The S4 AUC gate is a correct early-stop, not a false-alarm. Suggested calibration *direction* for next protocol cycle is feature-regime extension, not threshold relaxation.
