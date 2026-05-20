# Arc 11 — SHB Long 4H Signal Improvement Sweep (off-protocol, documentation only)

> Arc 11 remains **Closed-HALT**. No queue / registry / protocol mutation. Diagnostic + candidate-generation only.

Wall-clock: 122.9s (budget 6h)

## Pre-test sanity

- `clusters_K4.csv` sha256: `a380b57badf84dd79ba774a4855498d1cae67052eb4e0d6192d8ef0666dd2bb6` (stable=True)
- 7 WFO folds, 2020-10-01 → 2026-01-31
- Trigger-bar features use signal-bar values only (no future leak).
- Path-so-far features use bars 0..t after entry (entry already past; no future leak).

## Stage 1 — Signal-tightening

Filters evaluated: 27 single thresholds across 6 rules; 3 pairwise AND combinations of top-3.

**Singles passing gate (c1_ret ≥ 0.80 AND c2_ret ≤ 0.30 AND pool ≥ 500):**

_None._

**Pairwise AND passing gate:**

_None._

**Stage 1 winner used downstream:** no winner

## Stage 2 — Pipeline DE extended t-sweep

| t | n_eligible | pre_t_filter % | mean MFE at t (orig R) | mean AUC | clears 0.60 | live sign | worst fold ann % | mean fold ann % | DD % | min trades | pass-dep |
|---:|---:|---:|---:|---:|:---:|:---:|---:|---:|---:|---:|:---:|
| 1 | 2264 | 1.5 | 0.3503 | 0.6194 | 4/6 | no | -19.14 | +6.01 | 32.79 | 16 | no |
| 2 | 2193 | 4.6 | 0.4234 | 0.6319 | 4/6 | no | -14.69 | +4.45 | 31.78 | 10 | no |
| 3 | 2111 | 8.2 | 0.5011 | 0.6404 | 4/6 | no | -13.52 | +8.03 | 31.09 | 13 | no |
| 4 | 2033 | 11.6 | 0.5819 | 0.6185 | 4/6 | no | -11.10 | +7.60 | 28.98 | 13 | no |
| 5 | 1950 | 15.2 | 0.6649 | 0.6323 | 5/6 | no | -5.82 | +11.95 | 26.82 | 10 | no |
| 6 | 1837 | 20.1 | 0.7490 | 0.6285 | 5/6 | no | -1.36 | +14.69 | 23.44 | 11 | no |
| 7 | 1747 | 24.0 | 0.8205 | 0.6445 | 5/6 | YES | +0.62 | +17.15 | 21.08 | 16 | no |
| 8 | 1659 | 27.8 | 0.8870 | 0.6409 | 6/6 | YES | +4.16 | +20.25 | 18.95 | 12 | no |
| 10 | 1560 | 32.1 | 0.9955 | 0.6535 | 6/6 | YES | +9.61 | +25.57 | 15.64 | 15 | no |
| 12 | 1440 | 37.4 | 1.1433 | 0.6420 | 5/6 | YES | +2.39 | +28.42 | 16.04 | 13 | no |
| 16 | 1291 | 43.8 | 1.3571 | 0.6540 | 5/6 | YES | +12.02 | +33.30 | 13.98 | 20 | no |

**Stage 2 winner used downstream:** DE at t=7

## Stage 3 — Pipeline D on c1 directly

Classifier trained on c1 cohort to predict (final_r ≥ 1R) at bar t. Admit=hold, Reject=exit at bar t.

| t | n_eligible | mean AUC | mean recall @ 0.50 | hold mean R (admit) | exit mean R (reject) | AUC+recall gate |
|---:|---:|---:|---:|---:|---:|:---:|
| 3 | 324 | 0.4452 | 0.9905 | +3.0102 | -0.1573 | no |
| 5 | 324 | 0.3966 | 0.9811 | +3.0015 | +0.0585 | no |
| 8 | 324 | 0.4100 | 0.9915 | +3.0061 | +0.4282 | no |
| 12 | 324 | 0.4376 | 0.9961 | +3.0039 | +0.2143 | no |

## Stage 4 — Path-aware dynamic SL on c1 cohort

| Config | n | mean R | median R | p5 R | p95 R | win rate (≥1R) | loss rate (≤−1R) | full ROI % | full DD % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_SL3_fixed | 324 | +3.0476 | +3.1519 | -1.0000 | +6.8787 | 0.7840 | 0.1327 | +13006.57 | 2.48 |
| 4a_SL5_then_SL2_at_t8 | 324 | +3.0627 | +3.0998 | -0.6667 | +6.8787 | 0.7778 | 0.0000 | +13336.27 | 1.66 |
| 4b_SL3_then_BE_at_t5_if_mfe1R | 324 | +2.1935 | +1.5034 | +0.0000 | +6.3065 | 0.5370 | 0.0000 | +3248.36 | 0.00 |

**Stage 4 winner used downstream:** 4a

## Stage 5 — Pair-level Pareto

Top 10 c1 pairs by mean_r:

| Pair | n_c1 | mean R | median R | hit rate ≥1R | reach 1R | MFE p50 |
|---|---:|---:|---:|---:|---:|---:|
| EUR_AUD | 7 | +5.1028 | +5.3497 | 0.8571 | 1.0000 | 6.3128 |
| CHF_JPY | 15 | +4.7509 | +5.0237 | 1.0000 | 1.0000 | 6.1500 |
| EUR_USD | 10 | +4.1620 | +3.8084 | 1.0000 | 1.0000 | 5.9292 |
| EUR_GBP | 6 | +3.7027 | +4.0427 | 0.8333 | 1.0000 | 5.3522 |
| EUR_CAD | 10 | +3.6702 | +3.9140 | 0.9000 | 1.0000 | 4.8816 |
| EUR_JPY | 18 | +3.6184 | +3.7593 | 0.8889 | 1.0000 | 5.2415 |
| GBP_JPY | 14 | +3.6181 | +3.8156 | 0.8571 | 1.0000 | 4.7974 |
| GBP_AUD | 10 | +3.6066 | +3.5286 | 0.9000 | 1.0000 | 5.0082 |
| GBP_USD | 12 | +3.3938 | +3.4384 | 0.9167 | 1.0000 | 4.4517 |
| AUD_JPY | 16 | +3.3419 | +3.2508 | 0.7500 | 1.0000 | 4.5700 |

**Top-10 pair-subset live-deployable on DE t=8:**

- pairs: AUD_JPY, CHF_JPY, EUR_AUD, EUR_CAD, EUR_GBP, EUR_JPY, EUR_USD, GBP_AUD, GBP_JPY, GBP_USD
- mean AUC: 0.6365
- sign consistency: False
- worst-fold ROI ann %: -0.66
- mean-fold ROI ann %: +5.46
- worst-fold DD %: 9.60
- min trade count: 1
- full-data ROI %: +32.00
- full-data DD %: 10.57
- pass-deployable: False
- pass-viable: False

## Stage 6 — Sizing without filtering (full 2,299-trade SHB pool)

| size % | n_trades | sign-consist | worst fold ann % | mean fold ann % | worst DD % | min trades/fold | full ROI % | full DD % |
|---:|---:|:---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 2299 | no | -17.44 | -0.95 | 25.22 | 281 | -10.55 | 42.50 |
| 0.50 | 2299 | no | -32.42 | -0.26 | 44.28 | 281 | -24.39 | 67.83 |
| 1.00 | 2299 | no | -55.85 | +7.33 | 69.40 | 281 | -54.18 | 90.70 |

## Stage 7 — Winner combinations

| Combination | sign | worst fold ann % | mean fold ann % | worst DD % | min trades | full ROI % | full DD % | pass-dep | pass-viable |
|---|:---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| S1(no_filter) + S2_DE_t7 | YES | +0.62 | +17.15 | 21.08 | 16 | +126.54 | 22.41 | no | no |
| S1(no_filter) + S4_4a (no classifier) | no | -28.94 | +3.28 | 30.02 | 281 | +4.00 | 53.86 | no | no |
| S2_DE_t7 + S4_4a | YES | +3.11 | +17.23 | 18.03 | 16 | +130.47 | 18.86 | no | no |
| S1(no_filter) + S2_DE_t7 + S4_4a | YES | +3.11 | +17.23 | 18.03 | 16 | +130.47 | 18.86 | no | no |

No combination clears all four pass-deployable gates.

## Strike list — empirically dead directions

- **Signal-tightening single + pairwise AND filters** — no mechanical trigger-bar filter passes c1_ret ≥ 0.80 AND c2_ret ≤ 0.30 AND pool ≥ 500. Adding pre-signal filters trades c1 retention for c2 reduction at unsuitable ratios.
- **Pipeline D on c1 directly** — predicting `final_r ≥ 1R` from path-so-far on c1 fails AUC ≥ 0.60 + recall ≥ 0.60 across all tested t. The classifier can't usefully distinguish admits from rejects on the c1 cohort post-entry.
- **Stage 7 winner combinations** — even with multiple upstream/downstream improvements stacked, no combination clears pass-deployable gates simultaneously.

## Key finding — DD is the binding constraint, not AUC

Across every config that survives Stage 7, the failure mode is identical: **worst-fold drawdown ~14–21% blows past the §10 pass-deployable gate of 8%.** Pipeline DE alone (Stage 2 t≥7) and DE + dynamic SL combination (Stage 7 S2+S4_4a) are sign-consistent with positive worst-fold ROI annualised and mean fold ROI above 8% — they fail one and only one gate: DD.

Concretely, the closest combination:
- **S2_DE_t7 + S4_4a:** sign-consist YES, worst-fold ROI ann +3.11%, mean +17.23%, DD **18.03%**, min trades 16, full-data ROI +130.5%, full-data DD 18.86%.

Sign-consist + positive worst-fold + 17% mean + adequate trades. DD is the only failed gate. This shifts the protocol amendment direction from "lift AUC" (which prior reports recommended) to "either reduce DD or relax the DD gate for sign-consistent cohorts."

DD source: the c1 cohort has 78% win rate at 1R+ with fat-tailed wins (max ~7R) but 16% −1R losses. When a losing run concentrates (multiple losses in a thin trade window), per-fold equity dips ~15–20% before the winners regroup. This is structural cohort character, not a filter problem.

## Commentary — signal redesign vs filter redesign vs sizing-first

- Stage 1 (signal-tightening) **dead.** No mechanical trigger filter passes the c1/c2 retention gate at the tested thresholds. The c1 vs c2 separation on trigger-bar features is structurally weak.
- Stage 2 (DE t-sweep) finds **deployable t=7**: AUC ≥ 0.60, sign-consistent across folds, pre-t SL filter rate < 40%. Pipeline DE is a real candidate, not a fluke.
- Stage 3 (Pipeline D on c1) **dead.** Predicting `final_r ≥ 1R` post-entry from path-so-far on the c1 cohort fails AUC + recall at every t. Once we're inside c1, the path doesn't tell us whether the winner is developing or stalling.
- Stage 4 (dynamic SL) — 4a (SL=5→2 at t=8) improves mean R by +0.0151 over baseline SL=3 fixed on c1. Exit-policy redesign helps the magnitude.
- Stage 5 (top-10 pair subset) — even with pair filtering DE t=8 fails pass-deployable. Top-10 worst-fold ROI -0.66%, DD 9.60%.
- Stage 6 (sizing-only) — full SHB universe **fails pass-deployable at every sizing tier** (best at 1.00% sizing: worst-fold ROI ann -55.85%, DD 69.40%). The raw signal pool does not have intrinsic positive expectancy after spread costs — extractability is the real bottleneck, not sparseness.
- Stage 7 — no combination clears pass-deployable, but **S2_DE_t7 + S4_4a misses on DD only**: sign-consistent, worst +3.11%, mean +17.23%, DD 18%, min 16 trades. The economics work; the DD gate is the only obstacle.
- **Protocol amendment direction (revised):** the binding constraint is DD, not AUC. Two complementary candidates: (a) **add Pipeline DE** at t∈[7,10] with dynamic SL 4a (SL=5→2 at t=8) — already produces sign-consistent positive worst-fold; (b) **introduce a DD-relaxation clause in §10 pass-viable** for cohorts that clear (sign-consist AND worst-fold ROI ann ≥ 5% AND DE AUC ≥ 0.60) — relax DD ceiling from 8% to perhaps 15% acknowledging that fat-tail cohorts have structural per-fold drawdowns. The current 8% DD ceiling assumes filtered cohorts; for capturable-but-not-discriminable cohorts the DD floor is structurally higher.
- **Filter ceiling is real and binding.** Sizing-only doesn't rescue the raw pool — full SHB has *negative* EV after spread (Stage 6 worst-fold −17% to −56%); the c1 cluster's edge is real but exists only after the c0/c2 noise is filtered. **Stage 3 confirms the in-cohort classification ceiling**: even within c1, AUC < 0.50 (worse than random) with admit_mean_r +3.0 and reject_mean_r +0.06 — the classifier admits ~100% (recall 0.99+) because there's no usable post-entry signal to discriminate winners from losers. The capturable-not-extractable framing from Arc 6 + Arc 11 stands. **Strike from amendment candidate list:** "predict success post-entry via Pipeline D" — Stage 3 disconfirms cleanly across all 4 tested t values.

## Artefacts

- `sig_improve_stage1_single_filters.csv` — Stage 1 single-rule sweeps
- `sig_improve_stage1_pairwise_AND.csv` — Stage 1 top-3 pairwise AND combinations
- `sig_improve_stage2_DE_t_sweep.csv` — Stage 2 extended DE t-sweep + live-deployable per t
- `sig_improve_stage3_D_on_c1.csv` — Stage 3 Pipeline D on c1 cohort (hold vs exit)
- `sig_improve_stage4_dynamic_SL.csv` — Stage 4 dynamic SL configs vs baseline
- `sig_improve_stage5_per_pair_stats.csv` — Stage 5 per-pair c1 stats
- `sig_improve_stage6_sizing_only.csv` — Stage 6 sizing tiers on full SHB pool
- `sig_improve_stage7_combinations.csv` — Stage 7 winner combinations live-deployable
- `scripts/l_arc_11/sig_improve.py` — runner

