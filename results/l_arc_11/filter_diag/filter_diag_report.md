# Arc 11 — Filter-diagnosis report (off-protocol, documentation only)

> Arc 11 remains **Closed-HALT** per §16a Path A. No queue / registry / protocol mutation. Diagnostic only.

Wall-clock: 68.6s

## Pre-test sanity

- `clusters_K4.csv` sha256: `a380b57badf84dd79ba774a4855498d1cae67052eb4e0d6192d8ef0666dd2bb6`
- expected (S4 commit): `a380b57badf84dd79ba774a4855498d1cae67052eb4e0d6192d8ef0666dd2bb6`
- cluster labels stable: **YES**

WFO folds:
- fold 1: OOS 2020-10-01 → 2021-07-01
- fold 2: OOS 2021-07-01 → 2022-04-01
- fold 3: OOS 2022-04-01 → 2023-01-01
- fold 4: OOS 2023-01-01 → 2023-10-01
- fold 5: OOS 2023-10-01 → 2024-07-01
- fold 6: OOS 2024-07-01 → 2025-04-01
- fold 7: OOS 2025-04-01 → 2026-01-31

Multi-TF alignment audit: **2299/2299 trades OK** (PASS — no D1 future-leak, no 1H future-leak).

## Headline AUC table (4 regimes × per-fold AUC)

| Regime | F1 | F2 | F3 | F4 | F5 | F6 | F7 | mean | clears 0.65 | clears 0.70 | features |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|---:|
| A_baseline_c1_PE | — | 0.4811 | 0.5535 | 0.5268 | 0.5311 | 0.4942 | 0.5563 | **0.5238** | 0/6 | 0/6 | 23 |
| B_delayed_t1_c1 | — | 0.6680 | 0.5486 | 0.6218 | 0.6197 | 0.5898 | 0.6687 | **0.6194** | 2/6 | 0/6 | 15 |
| B_delayed_t3_c1 | — | 0.6810 | 0.5513 | 0.6692 | 0.6503 | 0.5776 | 0.7132 | **0.6404** | 4/6 | 1/6 | 15 |
| B_delayed_t5_c1 | — | 0.6634 | 0.5923 | 0.6464 | 0.6417 | 0.6171 | 0.6327 | **0.6323** | 1/6 | 0/6 | 15 |
| B_delayed_t8_c1 | — | 0.6732 | 0.6144 | 0.6318 | 0.6497 | 0.6280 | 0.6483 | **0.6409** | 1/6 | 0/6 | 15 |
| C_multiTF_c1_PE+D1+1H | — | 0.5103 | 0.5410 | 0.5572 | 0.5579 | 0.5467 | 0.5754 | **0.5481** | 0/6 | 0/6 | 30 |
| D_target_reach1R_atSL3 | — | 0.5546 | 0.4798 | 0.4505 | 0.5087 | 0.4265 | 0.4872 | **0.4846** | 0/6 | 0/6 | 23 |
| D_target_mfe2R_atSL3 | — | 0.5520 | 0.5556 | 0.4908 | 0.5304 | 0.4019 | 0.5304 | **0.5102** | 0/6 | 0/6 | 23 |

No regime cleared AUC ≥ 0.70 in ≥5/7 folds.

## Regime B — delayed entry mfe-cost curve

Question: does waiting N bars lift AUC above 0.65, and what R is given up per delay step?

| t | n_surviving | n_excluded_pre_t_sl | mean mfe_so_far R at t | median mfe at t | AUC at t | clears 0.65 |
|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 2264 | 35 | 0.3503 | 0.2626 | 0.6194 | no |
| 3 | 2111 | 188 | 0.5011 | 0.3969 | 0.6404 | no |
| 5 | 1950 | 349 | 0.6649 | 0.5551 | 0.6323 | no |
| 8 | 1659 | 640 | 0.8870 | 0.7674 | 0.6409 | no |

R units in original SL=2 frame (mfe_so_far_r as recorded in `trades_paths.csv`).

## Regime C — multi-TF feature importance (last fold's classifier)

Top 15:

| rank | feature | gini importance | TF |
|---:|---|---:|:---:|
| 0 | `h1_atr_14` | 0.0872 | 1H |
| 1 | `ret_5bar_atr` | 0.0503 | 4H |
| 2 | `prior_leg_length_atr` | 0.0422 | 4H |
| 3 | `atr14_at_signal_pct_close` | 0.0412 | 4H |
| 4 | `d1_atr_ratio_to_close` | 0.0403 | D1 |
| 5 | `ema50_4h_slope_5bar_atr` | 0.0399 | 4H |
| 6 | `ret_20bar_atr` | 0.0396 | 4H |
| 7 | `h1_pullback_depth_atr` | 0.0395 | 1H |
| 8 | `d1_trend_slope_5bar` | 0.0394 | D1 |
| 9 | `trend_filter_dist_atr` | 0.0386 | 4H |
| 10 | `d1_pos_in_20bar_range` | 0.0379 | D1 |
| 11 | `ema20_4h_slope_5bar_atr` | 0.0367 | 4H |
| 12 | `d1_rsi_14` | 0.0355 | D1 |
| 13 | `break_magnitude_atr` | 0.0352 | 4H |
| 14 | `range_to_atr_14` | 0.0345 | 4H |

Importance mass by timeframe (across all features):
- 4H: 70.6%
- D1: 15.3%
- 1H: 14.1%

## Regime D — target comparison (cluster vs reach_1R vs mfe≥2R, same features)

| Target | mean AUC | clears 0.65 |
|---|---:|:---:|
| predict cluster=c1 | 0.5238 | 0/6 |
| predict reach_1R at SL=3 | 0.4846 | 0/6 |
| predict mfe≥2R at SL=3 | 0.5102 | 0/6 |

Direct deployment-outcome targets do not outperform cluster membership.

## Live-deployable simulation (best regime only)

Best regime: **B_delayed_t8_c1** (mean AUC 0.6409). Threshold 0.50.

| metric | value |
|---|---:|
| sign consistency | YES |
| worst-fold ROI ann % | 4.1638 |
| mean-fold ROI ann % | 20.2504 |
| worst-fold DD % | 18.9522 |
| min trade count | 12 |
| full-data ROI % | 162.1948 |
| full-data DD % | 19.9107 |
| pass-deployable | no |
| pass-viable | no |

## Live-deployable comparison — Regime B t=3 vs t=5 vs t=8

Mean AUC ties at t=3 (0.640) and t=8 (0.641); per-fold breadth differs (t=3 clears 0.65 in 4/6 folds, t=8 in 1/6). Live-deployable WFO clarifies the trade-off — t=8 wins on economics:

| t | mean AUC | folds ≥ 0.65 | sign-consist | worst-fold ROI ann % | mean-fold ROI ann % | worst-fold DD % | min trades | full-data ROI / DD % | pass-deployable |
|---:|---:|:---:|:---:|---:|---:|---:|---:|---:|:---:|
| 3 | 0.6404 | 4/6 | no | −13.52 | +8.03 | 31.09 | 13 | 46.47 / 32.24 | no |
| 5 | 0.6323 | 1/6 | no | −5.82 | +11.95 | 26.82 | 10 | 77.48 / 27.68 | no |
| 8 | 0.6409 | 1/6 | YES | **+4.16** | +20.25 | 18.95 | 12 | 162.19 / 19.91 | no |

Why t=8 wins on economics despite tied AUC: the pre-t SL natural filter at t=8 excludes **640 of 2,299 trades (28%)** — biases the surviving pool toward winners (mean MFE-so-far 0.89R), so even a mediocre classifier produces sign-consistent positive returns. At t=3 only 188 trades are excluded; the residual loser pool drags any classifier into negative folds.

None of the three clears pass-deployable: DD too high (19–31%, gate 8%), trade count too low (10–13/fold, gate 15). All within ~3 pp on pass-viable's relaxed bar.

## Commentary — direction(s) that move the needle

1. **Delayed entry is the only direction that lifts AUC meaningfully.** Regime B at t=3 / t=8 lands mean AUC ≈ 0.640 (+0.12 over baseline 0.524). Both touch or cross the §8 gate. Multi-TF (Regime C +0.024) and reframed target (Regime D negative deltas) are dead — D-reach1R at 0.485 is *worse* than cluster membership at 0.524, so cluster IDs are NOT the wrong supervision target for this signal.
2. **The §8 gate (0.65 mean AUC) is binding-but-close** for the delayed-entry direction. t=3 mean AUC 0.6404 misses the gate by 0.0096; t=8 misses by 0.0091. A protocol amendment that pairs the AUC gate with a delayed-entry-allowed clause could plausibly re-evaluate Arc 11 as PASS at Step 4 — but the live-deployable WFO says even t=8 with sign-consistency cannot pass deployable on DD (18.9%, gate 8%) or trade count (12/fold, gate 15).
3. **Multi-TF features get used but don't crack the ceiling.** RF imports h1_atr_14 as the top feature (gini 0.087) and assigns D1+1H features ~29% of importance mass collectively. But mean AUC only moves +0.024. Suggests the LIMIT isn't feature availability — it's that *cluster membership at this signal is intrinsically weakly predictable at entry-bar regardless of TF context*. The information needed for prediction is in the post-signal price action (Regime B confirms this), not in pre-signal context (Regime C disconfirms this).
4. **Reframed target is dead direction.** Predicting reach_1R (AUC 0.485) or mfe≥2R (0.510) directly is *harder* than predicting cluster membership (0.524). Both deployment-outcome targets are noisier than cluster ID at entry-bar features. Strike "supervision target redesign" from the amendment candidate list.
5. **Best live-deployable variant (Regime B t=8) is sign-consistent +20% mean ROI but fails DD + trade count gates.** This is the rabbit hole: deferred entry recovers magnitude (admit-pool mean R is high) but the natural pre-t SL filter excludes 28% of pool → low trade counts per fold → high DD per fold. The path forward would either combine delayed entry with relaxed trade-count gate (pass-viable lite) or shore up trade count via co-fire batching with sibling arcs.
6. **Protocol amendment candidate (revised from prior NOTES):** the candidate is **"add Pipeline DE (Deferred-Entry)"** as a third pipeline option — same architecture as Pipeline E but classifier features include path-so-far at bar t (default t=8), with AUC gate at 0.60 (matching Pipeline D1) instead of 0.65. The early MFE cost (~0.89R given up) is accepted in exchange for the cleaner extractable cohort. Different from Pipeline D1 because there is no post-entry classifier decision — the trade enters after t bars or not at all, and runs to its archetype's §11 exit policy. Pairs with the Arc 6 capturable-not-extractable write-up.
7. **Strike from prior amendment candidates.** Multi-TF feature extension was floated in the prior `arc11_exp_s5_NOTES.md` as a direction — Regime C disconfirms (+0.02 only). The earlier suggestion to "relax AUC gate when mfe_p50 ≥ 3R" was already retracted in `arc11_exp_s5_noOracle_NOTES.md`; this report does not change that retraction (Regime C reinforces it — even AUC 0.64 with no-oracle Stage 1 wouldn't reach pass-deployable per the t=8 simulation).
8. **Stop-condition reading.** No regime exceeded 0.70 in ≥5/7 folds → no strong candidate flag. All four regimes did NOT finish < 0.55 mean AUC (t=3/t=5/t=8 cleared 0.55) → no structural-ceiling flag. The result sits in between: signal redesign is *not* required, but the §8 gate + the §10 DD/trade-count gates jointly form a tight binding constraint for this signal class. Amendment direction: **add Pipeline DE**, accept the magnitude tax, evaluate at relaxed §10 gates (or push for co-fire batching across arcs 8/9/10/11 to fix the trade-count problem).

## Artefacts

- `filter_diag_regime_summary.csv` — per-regime mean AUC + clears-gate counts
- `filter_diag_regime_aucs.csv` — per-regime per-fold AUC + n_train/n_test/base_test
- `filter_diag_regimeB_mfe_cost.csv` — Regime B mfe-cost curve
- `filter_diag_regimeC_feature_importance.csv` — Regime C top features by gini
- `filter_diag_multiTF_alignment_audit_sample.csv` — D1/1H lookahead audit (first 20)
- `filter_diag_best_regime_live_deployable.csv` — best regime live WFO
- `scripts/l_arc_11/filter_diag.py` — runner

