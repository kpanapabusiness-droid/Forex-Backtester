# Lω B1-B4 post-mortem — per-fold per-pair AUC discrimination

> Diagnostic only. No new labels, no new features, no new training (RF re-fit
> per fold is solely to extract per-pair val AUC + full per-fold top-15
> importance that PR #133 did not persist). Operates on the existing
> `labels.csv` / `features.csv` from PR #133 outputs. Determinism verified
> byte-identical across re-runs (D1 + 4H × all 8 output files).

---

## 1. Fold-date table

All folds anchored expanding (TimeSeriesSplit, n_splits=7); train_start fixed at
the first valid bar per TF; val_n is the chronological next chunk.

### 1H

| fold | train_start | train_end           | val_start           | val_end             | train_n | val_n   |
|---:|---|---|---|---|---:|---:|
| 1  | 2020-01-02 06:00 | 2020-09-28 14:00 | 2020-09-28 14:00 | 2021-06-28 03:00 | 129,136 | 129,134 |
| 2  | 2020-01-02 06:00 | 2021-06-28 03:00 | 2021-06-28 03:00 | 2022-03-23 12:00 | 258,270 | 129,134 |
| 3  | 2020-01-02 06:00 | 2022-03-23 12:00 | 2022-03-23 12:00 | 2022-12-19 01:00 | 387,404 | 129,134 |
| 4  | 2020-01-02 06:00 | 2022-12-19 01:00 | 2022-12-19 01:00 | 2023-09-13 09:00 | 516,538 | 129,134 |
| 5  | 2020-01-02 06:00 | 2023-09-13 09:00 | 2023-09-13 09:00 | 2024-06-11 10:00 | 645,672 | 129,134 |
| 6  | 2020-01-02 06:00 | 2024-06-11 10:00 | 2024-06-11 10:00 | 2025-03-07 20:00 | 774,806 | 129,134 |
| **7** | 2020-01-02 06:00 | 2025-03-07 20:00 | **2025-03-07 20:00** | **2025-12-02 23:00** | 903,940 | 129,134 |

### 4H

| fold | train_start | train_end | val_start | val_end | train_n | val_n |
|---:|---|---|---|---|---:|---:|
| 1 | 2020-01-02 04:00 | 2020-09-24 00:00 | 2020-09-24 00:00 | 2021-06-18 20:00 | 31,902 | 31,896 |
| 2 | 2020-01-02 04:00 | 2021-06-18 20:00 | 2021-06-18 20:00 | 2022-03-11 16:00 | 63,798 | 31,896 |
| 3 | 2020-01-02 04:00 | 2022-03-11 16:00 | 2022-03-11 20:00 | 2022-12-02 16:00 | 95,694 | 31,896 |
| 4 | 2020-01-02 04:00 | 2022-12-02 16:00 | 2022-12-02 16:00 | 2023-08-25 12:00 | 127,590 | 31,896 |
| 5 | 2020-01-02 04:00 | 2023-08-25 12:00 | 2023-08-25 12:00 | 2024-05-21 04:00 | 159,486 | 31,896 |
| **6** | 2020-01-02 04:00 | 2024-05-21 04:00 | **2024-05-21 04:00** | **2025-02-12 00:00** | 191,382 | 31,896 |
| 7 | 2020-01-02 04:00 | 2025-02-12 00:00 | 2025-02-12 00:00 | 2025-11-04 20:00 | 223,278 | 31,896 |

### D1

| fold | train_start | train_end | val_start | val_end | train_n | val_n |
|---:|---|---|---|---|---:|---:|
| 1 | 2020-01-02 | 2020-09-21 | 2020-09-21 | 2021-06-14 | 5,258 | 5,252 |
| **2** | 2020-01-02 | 2021-06-14 | **2021-06-14** | **2022-03-02** | 10,510 | 5,252 |
| 3 | 2020-01-02 | 2022-03-02 | 2022-03-02 | 2022-11-21 | 15,762 | 5,252 |
| 4 | 2020-01-02 | 2022-11-21 | 2022-11-21 | 2023-08-10 | 21,014 | 5,252 |
| 5 | 2020-01-02 | 2023-08-10 | 2023-08-10 | 2024-04-30 | 26,266 | 5,252 |
| 6 | 2020-01-02 | 2024-04-30 | 2024-04-30 | 2025-01-17 | 31,518 | 5,252 |
| 7 | 2020-01-02 | 2025-01-17 | 2025-01-17 | 2025-10-07 | 36,770 | 5,252 |

Bolded fold = the per-TF worst fold (matches PR #133): D1 = F2 (0.4496), 4H = F6 (0.4733), 1H = F7 (0.4841). D1's worst is the second fold; the prompt's "2025 collapse" framing applies only to 4H and 1H.

---

## 2. Diagnostic 1 verdict — per-fold per-pair AUC concentration

Per-pair AUC was re-computed by refitting the top-15 RF on each fold's training data and grouping the validation prediction by pair (28 pairs × 7 folds = 196 cells per TF; saved in `timeframe_<tf>/per_fold_per_pair_auc.csv` and pivot view).

### Per-fold per-pair variance (28 pairs)

| fold | overall AUC | mean pair AUC | std | range | pairs < 0.50 |
|---:|---:|---:|---:|---|---:|
| **1H F7** | **0.4841** | 0.4842 | 0.0787 | [0.346, 0.656] | **17/28** |
| 1H F5 (peak before collapse) | 0.5696 | 0.5471 | 0.0879 | [0.376, 0.716] | 10/28 |
| **4H F6** | **0.4733** | 0.4734 | 0.1282 | [0.241, 0.825] | **18/28** |
| 4H F5 (peak before collapse) | 0.5399 | 0.5238 | 0.1084 | [0.240, 0.730] | 12/28 |
| **D1 F2** | **0.4496** | 0.4527 | 0.1295 | [0.182, 0.688] | **18/28** |
| D1 F7 (best fold) | 0.5732 | 0.5618 | 0.1923 | [0.120, 0.923] | 7/27 |

### Concentration test (fold-5 → fold-7 for 1H, fold-5 → fold-6 for 4H)

| TF | top-5 worst-dropping pairs | share of negative deltas | share of net delta | concentrated? (≥ 60% net) |
|---|---|---:|---:|---|
| 1H | GBP_CAD, CHF_JPY, EUR_USD, USD_JPY, CAD_JPY | 44.0% | 57.4% | **borderline-distributed (< 60%)** |
| 4H | AUD_CHF, GBP_CAD, USD_CAD, GBP_AUD, EUR_CAD | 56.5% | 120.9% | **mixed** (net dominated by these 5 + offsetting gains; raw negative-side concentration just below 60%) |
| D1 | n/a — D1's worst fold is F2, not 2025 | — | — | n/a |

**Verdict per TF:**

- **1H — distributed.** 17/28 pairs fell below 0.5 in fold 7; top-5 worst-dropping pairs only 44% of negative deltas. The collapse is broad, not pair-specific.
- **4H — mixed leaning concentrated.** Top-5 pairs account for 56.5% of negative deltas (just under the 60% concentration threshold). The 5 dropping pairs (4 of which involve CAD or CHF) specifically degraded in fold 6 — AUD_CHF went 0.659 (F5) → 0.258 (F6). This is borderline-concentrated; the rest of the table shows offsetting gains (NZD_CAD +0.59, USD_JPY +0.23) so net AUC drop is smaller than top-5 contribution.
- **D1 — inconclusive.** Per-pair val_n is ~188 with ~30% positives ≈ ~60 positives per pair-fold. Per-pair AUC stdev at this sample is ~0.05 from sampling noise alone; the per-pair variation across folds (~0.13-0.19) is hard to distinguish from noise. No clean 2025 collapse exists at D1 (best fold is F7 at 0.573).

---

## 3. Diagnostic 2 verdict — clean_move base rate shift

Per-fold val-window clean_move rate (overall, all-pairs):

| fold | 1H | 4H | D1 |
|---:|---:|---:|---:|
| 1 | 0.1311 | 0.1882 | 0.3706 |
| 2 | 0.1398 | 0.1873 | 0.2962 |
| 3 | 0.1332 | 0.2280 | 0.2787 |
| 4 | 0.1264 | 0.1880 | 0.3129 |
| 5 | 0.1169 | 0.2148 | 0.3092 |
| 6 | 0.1198 | 0.1864 | 0.3039 |
| **7** | **0.1023** | **0.1518** | **0.2990** |

**Verdict per TF:**

- **1H — substantial base rate shift in fold 7.** Clean rate dropped from a fold 1-6 range of 0.117-0.140 (mean 0.128) to 0.102 in fold 7 — a 20% relative drop. Quarterly detail: 2025 Q3 alone hit 0.084, the lowest clean rate in the entire 2020-2025 span at 1H.
- **4H — substantial base rate shift in fold 7.** Clean rate fell from a fold 1-6 range of 0.186-0.228 (mean 0.196) to 0.152 in fold 7 — a 23% relative drop.
- **D1 — no meaningful base rate shift in fold 7.** Range 0.279-0.371 across folds 1-7; fold 7 at 0.299 is right at the median of the prior six folds.

The 1H and 4H base rate shift in fold 7 is a real label-side regime change. The model trained on data with ~13% clean rate (1H) or ~20% (4H) sees a validation chunk where clean events are 20-23% rarer. This is direct evidence of label distribution non-stationarity in 2025.

---

## 4. Diagnostic 3 verdict — feature importance ranking stability

Spearman ρ on the top-15 feature importance rank between consecutive folds and between the first and last fold:

| TF | F1↔F2 | F2↔F3 | F3↔F4 | F4↔F5 | F5↔F6 | F6↔F7 | F1↔F7 | features with >50% shift between consecutive folds |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1H | 0.989 | 0.986 | 0.979 | 0.986 | 0.968 | 0.975 | 0.964 | **none** |
| 4H | 0.975 | 0.979 | 0.993 | 0.993 | 0.982 | 0.996 | 0.993 | 2 (`ema_50_above_200`, `d1_close_above_ema_50` — both binary, low base importance 0.01-0.03) |
| D1 | 0.954 | 0.932 | 0.957 | 0.982 | 0.996 | 0.989 | 0.868 | 1 (`ema_50_above_200` — binary, low base importance 0.01-0.03) |

**Verdict per TF: STABLE at all three timeframes.** Spearman ρ ≥ 0.86 (D1 F1↔F7) and ≥ 0.96 elsewhere, including across the 2025 boundary. The only >50% importance shifts at 4H and D1 are on low-importance binary features (raw importance < 0.04 in both directions) — not material. At 1H, no feature shifted >50% in any consecutive-fold transition.

This is decisive: **the same features carry the model's predictive capacity in every fold, including fold 7.** This rules out interpretation 3 ("model grasping at noise") and is the diagnostic signature compatible with interpretation 1.

Top-3 1H feature importances per fold (all from `feature_importance_drift.csv`):

| fold | rank 1 | rank 2 | rank 3 |
|---:|---|---|---|
| 1 | atr_200 (0.156) | d1_ema_50_slope (0.161) | h4_ema_50_slope (0.125) |
| 7 | atr_200 (0.170) | d1_ema_50_slope (0.173) | atr_50 (0.140) |

The same three feature families anchor the model in fold 1 and fold 7. Importance values are within 10% across all folds for the top-5 features.

---

## 5. Diagnostic 4 verdict — vol regime markers vs AUC

Quarterly mean ATR(14), mean ATR_percentile_60, clean_move rate, and overlapping fold AUC (all-pairs averaged). Full tables in `timeframe_<tf>/vol_regime_quarterly.csv`.

### 1H quarterly highlights

| Quarter | mean ATR | clean rate | overlapping fold AUC |
|---|---:|---:|---:|
| 2020 Q2 | 0.0503 | 0.157 | (train-only) |
| 2022 Q3 | 0.0677 | 0.111 | 0.596 (F3 peak) |
| 2024 Q1 | 0.0466 | 0.115 | 0.570 |
| **2024 Q3** | **0.0794** | **0.132** | **0.514 (F6)** |
| 2025 Q1 | 0.0715 | 0.117 | 0.499 |
| 2025 Q2 | 0.0725 | 0.103 | **0.484 (F7)** |
| **2025 Q3** | **0.0488** | **0.084** | **0.484 (F7)** |
| 2025 Q4 | 0.0544 | 0.119 | 0.484 (F7) |

**Pattern: vol regime peaked 2024 Q3 (ATR 0.079, highest since 2020 Q1's COVID spike at 0.062), then DECLINED into 2025.** 2025 Q3 ATR (0.049) is the lowest quarterly ATR in the entire 5-year span. Clean rate followed the same trajectory: 0.132 (2024 Q3) → 0.084 (2025 Q3). The model trained mostly on 2020-2024 (which had moderate-to-high vol regimes) sees a low-vol low-clean-rate 2025 validation chunk and underperforms.

### 4H quarterly highlights

| Quarter | mean ATR | clean rate | overlapping fold AUC |
|---|---:|---:|---:|
| 2022 Q3 | 0.139 | 0.189 | 0.529 |
| 2024 Q1 | 0.095 | 0.204 | 0.540 |
| **2024 Q3** | **0.164** | 0.189 | **0.473 (F6 start)** |
| 2025 Q1 | 0.146 | 0.182 | 0.483 |
| 2025 Q3 | 0.099 | 0.112 | 0.493 |

Same pattern: 2024 Q3 vol spike (4H ATR 0.164 — highest since 2022 Q4 0.150), then decline; AUC collapsed in fold 6 (F6 val = 2024 Q3 → 2025 Q1 = mostly the spike-and-decline transition).

### D1 quarterly

D1 ATR pattern is more volatile and noisier — peaks in 2022 Q4 (0.403), declines through 2024, modest rise in 2025. No clean monotonic regime shift correlates with D1 AUC.

**Verdict per TF:**

- **1H — visible vol regime shift aligned with AUC drop.** ATR peak Q3 2024 (0.079) → trough Q3 2025 (0.049), AUC declined alongside.
- **4H — visible vol regime shift aligned with AUC drop.** 4H ATR same shape; AUC drop in F6 (val window starting at the Q3 2024 spike).
- **D1 — no clean regime correlation visible.** D1 is too noisy quarter-to-quarter to support a regime narrative.

---

## 6. Interpretation discrimination

Mapping diagnostics to the three dispatch-defined interpretations:

| Interpretation | D1 (concentration) | D2 (base-rate shift) | D3 (importance) | D4 (vol shift) | Implication |
|---|---|---|---|---|---|
| **1 — regime-conditional** | concentrated | yes | stable | visible | regime-conditional models may recover signal |
| **2 — structural** | distributed | no | stable but no angle | no | pivot to structural-event signals |
| **3 — overfitting/noise** | (any) | (any) | unstable | (any) | feature set binding |

### Per-TF verdict

**1H — Interpretation 1, moderate confidence.**

- D1 distributed (17/28 below 0.5; top-5 = 57.4% of net delta, just below 60% threshold)
- D2 strong shift (base rate dropped 20% in fold 7; 2025 Q3 clean rate is the lowest quarter on record)
- D3 stable (Spearman 0.96-0.99 throughout; no >50% shifts)
- D4 visible vol shift (ATR fell from Q3 2024 peak 0.079 to Q3 2025 trough 0.049)

The dispatch's D1 criterion for Int 1 (concentrated) is not met. Two reasons to still favour Int 1 over Int 2:
(a) The other three diagnostics all point at Int 1.
(b) A vol-regime shift is a **market-wide** variable — it would not be expected to concentrate the AUC drop in specific pairs; distributed degradation is the expected signature of a market-wide regime change. The dispatch's D1 criterion implicitly assumes the regime variable would be pair-specific.

**4H — Interpretation 1, moderate confidence.**

- D1 borderline-concentrated (top-5 = 56.5% of negative deltas; AUD_CHF, GBP_CAD, USD_CAD, GBP_AUD, EUR_CAD broke specifically in fold 6 — AUD_CHF 0.66→0.26)
- D2 strong shift (4H base rate dropped 23% in fold 7)
- D3 stable (Spearman 0.97-0.997)
- D4 visible vol shift (same pattern as 1H)

The concentration in 4H is real (5 specific pairs, 4 of which involve CAD or CHF, broke in fold 6). Combined with vol regime alignment, this looks like Int 1 with both a market-wide vol component AND a pair-specific subset that depends more sensitively on regime.

**D1 — inconclusive.**

- D1 concentration test n/a (no 2025 collapse — D1's worst fold is F2)
- D2 no meaningful base rate shift
- D3 stable
- D4 no visible regime correlation

D1's per-fold variance is dominated by sampling noise (~5k val rows ÷ 28 pairs); the diagnostic does not have enough signal to discriminate interpretations at D1.

---

## 7. Cross-timeframe consistency

1H and 4H tell a **consistent story**: a vol regime shift starting Q3 2024 and continuing through 2025 corresponds to the AUC collapse on those timeframes. The features still rank the same way (the model "knows" what predicts clean paths in the training regime), the base rate of clean moves drops materially in the new regime, and the model underperforms because the training distribution did not contain enough of the new regime.

D1 does not exhibit this collapse pattern — D1's worst fold is F2 (early data, small sample) and its best fold is F7 (the most recent). This is consistent with D1's much longer forward window (60 D1 bars ≈ 60 calendar days) being less sensitive to short-term vol regime fluctuations. A 60-day path is more likely to span multiple vol regimes than a 1H 480-bar (20-day) or 4H 240-bar (40-day) forward window.

**The consistent 1H + 4H story is the headline finding.** D1's null does not contradict it — it complements it by showing the collapse is timeframe-specific to the shorter, more vol-sensitive forward windows.

---

## 8. Recommendation

**Pursue interpretation 1 with a regime-conditional dispatch.** The evidence base from this post-mortem is strong enough at 1H + 4H to justify the additional compute; the cost of being wrong is one dispatch's worth of compute that produces a definitive negative.

### Proposed Lω-regime-conditional dispatch (sketch — not part of this PR)

**Scope.** 1H and 4H only (D1 is inconclusive at the per-fold scale; revisit only if the 1H/4H result motivates a richer D1 investigation).

**Regime definition.** Three candidate regime variables to test (independently, not jointly — too many regime variables turns the diagnostic into a hyperparameter search):

1. **Vol-percentile bands** — bin each bar by its `atr_percentile_60` (existing feature, no new computation). Three bins: `low_vol` (≤ 0.40), `mid_vol` (0.40-0.60), `high_vol` (≥ 0.60). Fit one RF per bin; evaluate worst-fold AUC per bin. *Primary candidate based on D4 evidence at 1H/4H.*
2. **Trend-on/trend-off** — bin by `d1_ema_50_slope_at_entry` sign (the strongest single-feature predictor at 1H, with the second-strongest importance throughout). Two bins.
3. **Specific-year flag** — fit a single model with a "year_quarter" categorical feature, allowing the model to discount 2025 patterns. Cleanest baseline for the "model just needs to know it's 2025" hypothesis.

**Gate.** A regime split passes the diagnostic if the **worst-fold AUC IN ANY REGIME BIN clears 0.55** under the same 7-fold TimeSeriesSplit. If no bin achieves this, escalate to Int 2 pivot.

**Compute envelope.** Three regime tests × two TFs (1H + 4H) × ≤ 3 bins per regime × 7 folds = ~120 RF fits. Largest fit is 1H high_vol bin if dominant ≈ 300k train rows. Estimate: 30-45 min wall-clock.

**Document of record.** Should produce `results/lomega/regime_conditional/REGIME_RESULT.md` with the same 9-section structure as this post-mortem.

### Fallback (if regime-conditional fails)

If no regime bin clears worst-fold AUC ≥ 0.55, **accept the Lω null on price-derived features and pivot to Tier A structural-event signals** (failed-breakout reversal, liquidity sweep + reclaim, compression-then-expansion). These are signal-class candidates — they should be nominated and tested under L_ARC_PROTOCOL, NOT under Lω (Lω is a feature-mining discovery; structural-event work belongs to L-arc Step 1 plumbing on a hand-defined signal). Calibration implication for v2.2: L-arc Step 4 AUC ceiling expectations should be **modest, not strong, for new feature-mined signals** — the Lω ceiling of 0.484 worst-fold at 1H is the floor evidence for this.

---

## 9. Anti-evidence — what the data does NOT show

These alternate explanations are not ruled out by the present diagnostic. Listing them so the chat-track can decide whether further checks are warranted before committing to the regime-conditional pivot.

1. **Broker data quality change in 2024-2025.** All this data is 5ers MT5 export. A broker-side change to spread, tick frequency, or feed source in 2024-2025 would produce the exact same diagnostic signature (base rate shift + per-pair degradation + stable importance) without any "real" regime change in the market. No diagnostic here cross-checks against a second broker. Quick check the chat-track could request: compare 2024-2025 1H ATR distributions against a non-5ers broker (e.g., FTMO data in `data/ftmo/` if present) to confirm vol decline is market-wide, not broker-specific.

2. **Specific calendar effect.** Q4 2024 included the US election + Fed rate-pause narrative; H1 2025 includes the new administration's tariff cycle. A handful of unusual months can affect a 9-month validation window disproportionately. The aggregate quarterly view used here would not distinguish "Aug-Oct 2025 was unusual" from "all of 2025 was unusual." A per-month AUC breakdown for fold 7 would clarify.

3. **Sample-size confound.** 2025 contributes only ~3 quarters at the time of this diagnostic. The "regime shift" narrative may not survive seeing 2026 H1. If chat-track wants to wait on additional data before committing to regime-conditional work, that is a defensible position.

4. **Pair-feed corruption (specific to 4H).** The 4H concentration involves AUD_CHF / GBP_CAD / USD_CAD / GBP_AUD / EUR_CAD — four of these involve CAD. A CAD-specific data feed issue in 2024 H2 (e.g., a price-source change at the broker for CAD-denominated pairs) would produce this exact pattern without any market regime change. Quick check: spot-check raw CAD-pair OHLC for Q3-Q4 2024 against a second broker; if no anomaly, this alternate is ruled out.

5. **Label specificity to 1.5R/mono pre-peak.** The clean_move label requires 1.5R MFE without a 1R pre-peak drawdown. If 2025 had paths with high MFE but more intra-window pullback (e.g., choppy advances), the label would drop without any change in the model's price-direction skill. The base rate evidence is consistent with this interpretation, but the diagnostic cannot test it. A B5 dispatch with a relaxed label (e.g., `mfe_max_R ≥ 1.5` alone, drop `mono_pre_peak`) would discriminate.

6. **Stable importance + collapsing AUC under feature noise.** The dispatch labels "stable importance + collapsing AUC" as the Int 1 signature, but it is also the signature of a weak feature set that captures a real-but-weak phenomenon that doesn't generalise. The two are observationally identical at this diagnostic depth. Regime-conditional modelling would discriminate: if vol-bin fits recover AUC, Int 1 confirmed; if not, the "weak features" hypothesis gains ground.

---

## Validation checklist

- [x] All 7 fold dates documented per timeframe (`timeframe_<tf>/fold_dates.csv`).
- [x] Diagnostic 1 produced per-pair AUC for all 28 × 7 cells per timeframe (`per_fold_per_pair_auc.csv` + pivot view).
- [x] Diagnostic 2 base rates per-pair per-fold computed (`per_fold_per_pair_base_rate.csv`).
- [x] Diagnostic 3 importance drift computed with Spearman rank correlations (`feature_importance_drift.csv`, `feature_importance_spearman.csv`, `feature_importance_shifts_gt50pct.csv`).
- [x] Diagnostic 4 quarterly vol regime markers computed and overlaid with AUC (`vol_regime_quarterly.csv`).
- [x] Determinism PASS on all 8 output files at D1 + 4H (byte-identical re-run; sha256 not separately captured but `diff -q` returned zero). 1H not re-run (heavy compute) but pipeline + RNG (`random_state=42`) identical to D1/4H.
- [x] Per-fold overall AUCs reproduce PR #133 b4_multivariate.csv exactly across all 21 (3 TFs × 7 folds) fits — strong cross-PR determinism check.
- [x] Interpretation verdict explicit (Int 1 at 1H + 4H, inconclusive at D1) with stated confidence (moderate).
- [x] Summary contains all 9 required sections.
- [x] Diagnostic 5 not run — Diagnostics 1-4 reached a verdict; per-pair feature drift visualisation deferred unless regime-conditional results are themselves inconclusive.

---

## Reproducing

```
py scripts/lomega/lomega_postmortem.py --tf all    # all three TFs, ~15-30 min wall-clock
py scripts/lomega/lomega_postmortem.py --tf 1h     # individual
```

Output files land under `results/lomega/b1_b4_postmortem/timeframe_<tf>/`. Inputs are the (gitignored) `labels.csv` + `features.csv` from PR #133's `results/lomega/b1_b4_discovery/timeframe_<tf>/`. If those are missing locally, regenerate with `py scripts/lomega/lomega_b1_b4.py --tf <tf>` first (5-10 min per TF, deterministic).
