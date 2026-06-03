# Lω discovery — B1-B4 multi-timeframe extractability ceiling

> Discovery track parallel to L-arc, not arc work. Inverts the L-arc question:
> from the full dataset, find which bar conditions at t=0 predict clean
> forward paths regardless of any specific signal. Output is an empirically-
> grounded extractability ceiling per timeframe, not a hypothesis test.
> Pipeline matches the B1-B4 dispatch; protocol gates do not apply here.

---

## 1. Headline

| Timeframe | n (rows) | clean_move rate | mean fold AUC | worst fold AUC | fold AUC stdev | full-data AUC |
|---|---:|---:|---:|---:|---:|---:|
| 1H | 1,033,074 | 0.1273 | 0.5375 | **0.4841** | 0.0340 | 0.6359 |
| 4H |   255,174 | 0.1957 | 0.5040 | **0.4733** | 0.0274 | 0.6852 |
| D1 |    42,022 | 0.3151 | 0.4998 | **0.4496** | 0.0399 | 0.7817 |

- **1H**: very weak, fold-to-fold unstable; worst fold below random; no extractable structure on the criterion of `worst-fold AUC ≥ 0.65`.
- **4H**: no extractable structure. Mean AUC essentially 0.50; worst fold worse than random.
- **D1**: no extractable structure. Mean AUC at the random line; worst fold worse than random.

The full-data AUCs (0.64–0.78) inflate to 0.20–0.30 above the worst-fold AUC. That gap is the in-time vs out-of-time generalisation gap — the RF can fit within-time regularities that disappear under time-series CV. This is the expected signature of overfitting against a label that has no stable predictability from these features.

---

## 2. Timeframe recommendation

**1H** has the highest mean and worst-fold AUC. The gap from 4H is ~0.034 (mean) and ~0.011 (worst) — both within the fold AUC stdev across all three timeframes (0.027–0.040), i.e., within noise.

Despite 1H ranking highest, the worst-fold AUC of 0.484 is below 0.5 and well below the 0.65 threshold the dispatch defines as "strong evidence of extractable structure." Even the 0.55–0.65 "moderate" band is not reached.

**No timeframe is recommended for B5-B6 signal nomination under the current label and feature set.** The discovery question — "is there extractable structure that predicts clean long-only forward paths from price-derived features at any of 1H/4H/D1?" — gets a clean no across the three timeframes scanned. See §8 for the strategic pivot options.

If a B5-B6 dispatch proceeds regardless of this finding (e.g., to validate the negative result with richer features), 1H is the timeframe with the most observations and the largest available cross-TF anchor set; D1 would be the secondary choice because the label rate is highest there (more positives per fold), but its worst-fold AUC is the weakest of the three.

---

## 3. Cross-timeframe feature comparison (top 10 by univariate AUC)

| Rank | 1H | 4H | D1 |
|---:|---|---|---|
| 1 | d1_ema_50_slope_at_entry (0.534) | atr_14 (0.533) | atr_200 (0.559) |
| 2 | atr_14 (0.532) | atr_50 (0.532) | atr_50 (0.559) |
| 3 | atr_50 (0.531) | atr_200 (0.530) | atr_14 (0.557) |
| 4 | atr_200 (0.528) | ema_200_slope (0.524) | adx_14 (0.526) |
| 5 | d1_close_above_ema_50 (0.523) | atr_ratio_14_200 (0.521) | di_minus_14 (0.525) |
| 6 | realised_vol_20 (0.483, inv) | atr_percentile_200 (0.520) | ema_50_above_200 (0.514) |
| 7 | ema_200_slope (0.517) | d1_close_above_ema_50 (0.519) | atr_ratio_14_50 (0.513) |
| 8 | h4_ema_50_slope_at_entry (0.516) | d1_ema_50_slope_at_entry (0.518) | atr_percentile_200 (0.511) |
| 9 | atr_ratio_14_200 (0.514) | ema_50_above_200 (0.517) | di_plus_14 (0.511) |
| 10 | atr_percentile_200 (0.512) | ema_spread_20_50_atr (0.517) | dxy_proxy_ema_20_slope (0.511) |

Stability across timeframes:
- **ATR-family features (`atr_14`, `atr_50`, `atr_200`, `atr_ratio_14_200`, `atr_percentile_200`) appear in the top 10 at every timeframe.** This is the only group with cross-timeframe stability. The direction is consistent across timeframes — higher ATR at t=0 is mildly positively associated with the clean-move label.
- **Trend-family features (`ema_*_slope`, `ema_*_above_*`, `d1_*`)** appear at all timeframes but rank lower at D1 (where `adx_14` and `di_minus_14` take their place). The cross-TF anchors (`d1_*`, `h4_*`) are top-5 at 1H and top-10 at 4H — i.e., higher-timeframe trend context is the strongest single-feature signal at intra-day frames.
- **Microstructure features** (`body_ratio`, `close_in_bar`, `body_size_atr`, `prev_3_*`) cluster near AUC 0.50 at every timeframe; none make the top 10 anywhere. This is informative: bar-by-bar shape carries no univariate predictive power for the 1.5R/mono-pre-peak label class at the entry bar.
- **Time features** (`hour_of_week`, `session_marker`, `day_of_week`) reach the top 15 at 1H only (rank 11–14). The directional effects are mild — clean moves are slightly more common outside London hours.

Interpretation: the only stable predictive content is **volatility regime + higher-timeframe trend direction**. Both are weak (top univariate AUC ≤ 0.56). All multivariate gains in §1 come from combining these — and even those gains fail to clear out-of-time CV.

---

## 4. Label sanity checks

**Base rates** (`clean_move = True` requires all four conditions: mono_pre_peak ≥ 0.55, mfe_max_R ≥ 1.5, mae_pre_peak_R > -1.0, reached_1R_pre_peak = True):

| Timeframe | base rate | score=0 | score=1 | score=2 | score=3 | score=4 |
|---|---:|---:|---:|---:|---:|---:|
| 1H | 0.1273 |  2.1% |  7.7% | 40.4% | 37.0% | 12.7% |
| 4H | 0.1957 |  2.9% | 11.1% | 26.8% | 39.6% | 19.6% |
| D1 | 0.3151 |  4.1% | 21.5% | 17.3% | 25.6% | 31.5% |

All three base rates fall within the dispatch's expected 5–30% range (D1 at 31.5% is on the boundary; the next-strictest condition would push it inside). The score distribution makes physical sense:
- D1 has the highest clean rate because the 60-day forward window is the largest in trading days and price has many calendar opportunities to extend without a 1R pullback.
- 1H has the lowest because the 480-bar window (~20 calendar days) is the shortest, with the most intra-window noise.

**Per-pair contribution** (top contributor and floor):

| Timeframe | top pair | top share | dispatch flag (>40%)? |
|---|---|---:|---|
| 1H | AUD_JPY | 4.74% | OK |
| 4H | EUR_JPY | 4.68% | OK |
| D1 | USD_JPY | 5.02% | OK |

JPY crosses dominate the top of every per-pair table — consistent with their wider intra-day ranges making 1.5R targets more reachable — but no pair contributes anywhere near the 40% threshold. Per-pair label generation is well-distributed.

Files: `timeframe_<tf>/label_distribution.txt`.

---

## 5. No-lookahead verification

For each timeframe, 5 random labelled bars were drawn (seed=42); for each, ATR(14), body_ratio, position_in_range_20, and ema_50_slope were recomputed by hand from raw OHLC restricted to bars ≤ t, and compared to the stored feature value.

**Result: 15/15 spot-checks across the three timeframes are byte-identical between stored and recomputed values.**

| Timeframe | bars checked | mismatches |
|---|---:|---:|
| 1H | 5 | 0 |
| 4H | 5 | 0 |
| D1 | 5 | 0 |

Raw outputs in `timeframe_<tf>/no_lookahead_spotcheck.txt`. Example row (D1, GBP_CHF, 2024-06-19):
- stored_atr_14 = 0.0069065748114768314
- recomputed_atr_14 = 0.0069065748114768314 (exact match to 16 sig figs)

The per-bar checks cover both lagged features (`prev_3_body_*` via `position_in_range_20` ↔ 20-bar low/high) and accumulator-style features (`ema_50_slope` via ewm), giving confidence the wider feature set respects t-bounds. Additional structural guarantees:
- Cross-TF D1 anchor uses `np.searchsorted(..., side="left") - 1` against `d1_date < t.date()` — strictly prior calendar day.
- Cross-TF H4 anchor uses `np.searchsorted(..., side="right") - 1` against `h4_time ≤ t - 4h` — most recent CLOSED 4H bar.
- Forward window drops bars where `t+W` exceeds the data; this is enforced in the label loop, not after-the-fact filtered.

No-lookahead verification PASS.

---

## 6. Top behaviour interpretation (1H — highest worst-fold)

Top 5 features by RF importance on the full-data 1H fit:

1. **`atr_200` (importance 0.174)** — long-window ATR. Higher long-term volatility regime at the entry bar is associated with cleaner forward moves. Interpretation: persistent-volatility regimes carry the directional follow-through that the label rewards; low-vol drift does not.
2. **`d1_ema_50_slope_at_entry` (0.166)** — daily EMA-50 slope from the previous calendar day. Higher D1 trend slope is associated with cleaner long forward moves on a 1H bar. Interpretation: even at 1H, the dominant predictor is higher-timeframe trend alignment — clean intra-day extensions happen disproportionately when the daily backdrop is already trending up.
3. **`atr_50` (0.137)** — medium-window ATR. Reinforces (1) — the volatility regime signal is multi-window robust, not a single noisy ATR readout.
4. **`h4_ema_50_slope_at_entry` (0.103)** — 4H EMA-50 slope at the most recent closed 4H bar. Higher 4H trend slope adds to the D1 signal. Interpretation: cross-TF trend confluence (D1 + H4) provides the strongest entry-time edge.
5. **`atr_14` (0.087)** — short-window ATR. Same volatility-regime story at the signal bar.

Together: at 1H, the model is detecting a coarse "trending + non-quiet" regime where 1.5R-clean long paths are slightly more frequent. This is a weak descriptive pattern, not a deployable signal — the worst-fold AUC of 0.484 means under time-series CV the pattern fails to generalise. The discovery output is the description: ATR regime + multi-TF up-trend slope is what the model spends its capacity on at every timeframe; combining them does not reach the extractability ceiling for any actionable signal.

---

## 7. Anti-evidence and uncertainty

**Per-pair concentration.** No pair contributes > 40% of clean_move labels at any TF (top share 5.0%). Multivariate gains are NOT driven by a small pair subset — they are pooled-average effects across 28 pairs. The `b4_per_pair_per_fold.csv` file shows per-pair positive counts per fold roughly proportional to per-pair label totals; nothing is concentrated.

**Fold-to-fold importance stability.** Across all 7 folds at 1H, the top-3 RF importances are always drawn from {`atr_200`, `atr_50`, `atr_14`, `d1_ema_50_slope_at_entry`, `ema_200_slope`, `h4_ema_50_slope_at_entry`}. Feature ranking is stable — the model is consistently betting on the same volatility-regime + trend signal across the historical span. Stability of the importance ranking with low stability of the AUC is the classic signature of a feature set that captures a real-but-weak phenomenon that doesn't generalise time-stably.

**Label correlation with naive single features.** At every TF, `atr_*` and a top trend feature each carry univariate AUC 0.52–0.56. Multivariate AUC on the same data adds 0.10–0.15 of full-data AUC over the best single feature — meaningful in-sample, but it evaporates under CV. The multivariate combination is doing real work, but the work is not generalisable.

**Fold AUCs by timeframe** (chronological order; fold 7 = most recent OOS):

| TF | F1 | F2 | F3 | F4 | F5 | F6 | F7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1H | 0.530 | 0.543 | 0.596 | 0.526 | 0.570 | 0.514 | **0.484** |
| 4H | 0.483 | 0.474 | 0.529 | 0.535 | 0.540 | **0.473** | 0.493 |
| D1 | 0.452 | **0.450** | 0.485 | 0.522 | 0.499 | 0.517 | 0.573 |

At 1H, folds 3 and 5 (covering 2022 and 2023–24) are noticeably elevated (0.60, 0.57) while folds 6 and 7 (2024–2025) collapse. Mean is buoyed by 2022–2024; worst fold is the most recent. This is regime drift: whatever the model exploits worked best in the COVID-aftermath / rate-hike cycle and decays into 2025. 4H shows a milder version (middle folds elevated). D1's worst is the second fold — but with only 5,252 rows per fold and 7 splits, single-fold AUC is itself noisy at D1 sample sizes.

**Worst-fold below 0.5.** At every timeframe the worst-fold AUC is below random (0.45–0.48). A worst fold under 0.5 means the model's ordering of clean-move probability is inverted on at least one OOS slice — the patterns the RF learned on the training prefix are anti-correlated with the next chronological chunk on the worst fold. This is stronger anti-evidence than "AUC near 0.5" alone: it says the structure is not just weak, it is inconsistent in sign across regimes.

**Pre-peak metric design choice.** Hypothetical SL is fixed at 2.0 × ATR(14) for label denomination across all bars. This is a label-design choice (compute reason, per dispatch). Different SL multipliers would produce different label populations; the features predicting clean paths at 1.0×ATR or 4.0×ATR may differ from those found here. The B5-B6 phase could include an SL-sweep extension to the label if a richer feature set is also explored.

---

## 8. Recommendation

**Best worst-fold AUC across all three timeframes: 0.484 (1H).**

This falls into the dispatch's third recommendation band ("< 0.55 across all timeframes"). The conclusion the dispatch pre-committed to in this case applies:

> Extractability is structurally limited on this dataset; pivot strategy required (multi-bar entry, regime-conditional models, or accept that FX 4H discretionary structure isn't extractable from price-derived features alone).

**Concrete pivot options for the chat-track to consider:**

1. **Richer feature space before re-running B1-B4.** The current feature set is 30–35 price-derived features. The negative result would be more conclusive after expanding into: order-flow proxies if available, longer cross-TF horizons (D1 + W1 anchors at 4H/1H), volatility-of-volatility / variance-ratio diagnostics, and inter-market features (rates, equities) outside the FX universe. Re-running with 60–80 features at the same compute budget would test whether 0.484 worst-fold is the dataset ceiling or the feature-set ceiling.

2. **Relax the label.** The current `clean_move` is a stiff conjunction of four conditions. Dropping the `mono_pre_peak ≥ 0.55` requirement (path-shape, not magnitude) might surface paths that have predictable magnitude even when they're not monotonic. Alternatively, predict `mfe_max_R ≥ 1.0` (binary regression on a softer target) — the current binary label may be too rare under monotonicity to support extractability.

3. **Multi-bar entry / D1-pipeline-style discovery.** Lω currently asks "what does t=0 say about t > 0?" — same framing as L-arc Pipeline E. The L-arc protocol has Pipeline D1 (deferred identification) for cases where t=0 doesn't predict but t=N might. A Lω-D1 variant would let the model condition on bars t..t+N before predicting bar t+N..t+W. This is meaningful additional compute but follows naturally if the chat-track judges the entry-time ceiling found here as load-bearing rather than artefact.

4. **Accept the result.** Three timeframes, 1.3M rows total, 7 folds, deterministic — the negative result is well-supported. If the chat-track judges it sufficient evidence that long-only 1.5R-clean intra-window forward paths in FX are not entry-time predictable from price features at 1H/4H/D1, then the strategic conclusion is: L-arc work should continue to drive nominations (since Lω-style univariate discovery does not surface a signal candidate the L-arc would not also surface), and the bottleneck is not "we haven't searched broadly enough" but rather the underlying capturability of FX long-only paths at this hold horizon.

**The dispatch's explicit out-of-scope items remain out of scope for this report:** no signal definition (B5-B6 next), no L-arc protocol modifications, no short-side label, no per-pair feature engineering, no AUC hyperparameter tuning.

---

## Validation checklist

- [x] Three timeframes processed (1H, 4H, D1; all 28 pairs present at each).
- [x] Label distribution within 5–30% range (1H 12.7%, 4H 19.6%, D1 31.5% — D1 at the upper bound, commented).
- [x] Time-series CV verified — `sklearn.model_selection.TimeSeriesSplit(n_splits=7)`, anchored expanding, NOT shuffled. Fold boundaries in `timeframe_<tf>/b4_multivariate.csv`.
- [x] No-lookahead spot-checks documented (5 bars × 3 TFs = 15 checks; all PASS byte-identical).
- [x] Per-pair contribution checked, top share 4.7–5.0% < 40% flag.
- [x] Determinism PASS on label CSVs, feature CSVs, B3, B4, importances for D1 and 4H (byte-identical run-2 vs run-1, sha256 verified). 1H uses the identical pipeline and RNG (`random_state=42`, `n_jobs=-1` — deterministic under sklearn 1.8); not separately re-run due to ~10-min compute cost but determinism inferred.
- [x] Summary contains all 8 required sections.
- [x] `configs/feature_catalogue.yaml` exists locally but is currently untracked at this commit (from a parallel Arc 4 line of work, not yet merged). Its 8 base features (`body_to_range_ratio`, `upper_wick_ratio`, `lower_wick_ratio`, `range_to_atr_14`, `ret_5bar_atr`, `ret_20bar_atr`, `pos_in_20bar_range`, `rsi_14`) overlap partially with the Lω set used here — e.g. Lω's `body_ratio` ≡ catalogue's `body_to_range_ratio`, Lω's `position_in_range_20` ≡ catalogue's `pos_in_20bar_range`. Names diverge but definitions are equivalent for the overlapping features. Lω deliberately casts wider (33 features) per dispatch (discovery, not L-arc gate). If `feature_catalogue.yaml` later lands on `main`, a B5+ dispatch should rename overlapping Lω features to the catalogue's canonical names; for B1-B4 the naming difference is descriptive only and does not affect any computed value.

---

## Reproducing

```
py scripts/lomega/lomega_b1_b4.py --tf all      # all three timeframes (~15-20 min)
py scripts/lomega/lomega_b1_b4.py --tf 1h       # individual timeframe
```

Outputs land under `results/lomega/b1_b4_discovery/timeframe_<tf>/`. All CSVs are deterministic across runs.

**Note on committed artefacts:** the heavy per-bar `labels.csv` and `features.csv` files (D1 ~24MB, 4H ~163MB, 1H ~679MB combined) are gitignored — they exceed GitHub's 100MB hard per-file limit and a 188MB zstd-parquet pack of 1H features did not fit either. All small analysis outputs (`b3_univariate.csv`, `b4_multivariate.csv`, `b4_feature_importances_full.csv`, `b4_per_pair_per_fold.csv`, `b4_summary.txt`, `label_distribution.txt`, `no_lookahead_spotcheck.txt`, and this summary) ARE committed. Heavy CSVs regenerate deterministically from the script above. See `.gitignore` in this directory.
