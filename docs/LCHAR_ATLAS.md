# LCHAR_ATLAS — L Characterization Arc Atlas Synthesis

**Arc:** L characterization arc — bottom-up exploratory data analysis of the 28-pair NNFX universe at 1H, 4H, D1.
**Period covered:** 2020-10-01 → 2025-12-31.
**Phases completed:** L0 (methodology lock), L1 (univariate), L2 (multi-timeframe), L3 (cross-pair), L4 (conditional), L5 (synthesis & registry).
**Methodology lock:** `docs/L0_METHODOLOGY_LOCK.md` (descriptive-not-predictive rule, DSR ranking, N=5 top-N registry).
**Plan source-of-truth:** `docs/L_ARC_PLAN.md`.
**Companion documents:** `LCHAR_TOPN_REGISTRY.md` (top-5 entries), `results/lchar/L_ARC_CLOSURE.md` (arc closure + L6+ pointer), `results/lchar/PHASE_L1_RESULT.md` … `PHASE_L5_RESULT.md` (per-phase result documents).

This atlas is **descriptive**. Per L0 §3 nothing in this document recommends a signal, identifies a "tradeable" pattern, or interprets a high-DSR finding as an opportunity. The mechanical bridge from atlas to candidate signals lives in `LCHAR_TOPN_REGISTRY.md`; what subsequent direction the project takes is summarised in `results/lchar/L_ARC_CLOSURE.md`.

---

## 1. Arc summary

The L arc is a from-scratch bottom-up exploratory analysis of the 28-pair FX universe across three timeframes. Construction began 2026-05-09 with the L0 methodology lock (pre-registered ranking metric, family-budget cap of 300 trials, N=5 top-N registry, DSR > 0.95 threshold for L6+ proceed-flag) and closed 2026-05-10 with this synthesis. Four atlas layers were produced as long-format CSVs under `results/lchar/atlas/`; each phase produced a result document under `results/lchar/PHASE_LN_RESULT.md`. All four CSVs are deterministic (sha256-verified byte-identical on re-run, `numpy.random.default_rng(np.random.PCG64(42))`, `%.10g` float format, sorted output).

| Layer | Output | Rows | sha256 (locked) |
|---|---|---:|---|
| L1 | `results/lchar/atlas/layer1_univariate.csv` | 7,812 | `5fd6cd3d…ed7b86d` |
| L2 | `results/lchar/atlas/layer2_multitimeframe.csv` | 1,176 | `c5eea410…be7b284` |
| L3 | `results/lchar/atlas/layer3_crosspair.csv` | 2,696 | `a4cfcc89…b6393` |
| L4 | `results/lchar/atlas/layer4_conditional.csv` | 6,531 | `c32b1343…36f56` |

---

## 2. L1 — Univariate findings (per-pair × per-TF)

Source: `results/lchar/PHASE_L1_RESULT.md`. Headline cross-pair point-estimate distributions (median across 28 pairs):

| TF | returns std (median) | returns excess kurt (median) | returns ACF lag-1 (median) | ATR ACF lag-1 (median) | ATR ACF lag-60 (median) |
|---|---:|---:|---:|---:|---:|
| 1H | 0.001032 | 23.305 | −0.0311 | 0.9847 | 0.4253 |
| 4H | 0.002042 | 10.253 | −0.0211 | 0.9849 | 0.3810 |
| D1 | 0.004858 | 2.741 | −0.0280 | 0.9862 | 0.3017 |

Headline structural observations (descriptive only):

- **Returns mean ≈ 0** at all three TFs across all pairs, with the bootstrap CI of `mean_r` containing zero in 95.2% of (pair × TF) cases (80 / 84). Consistent with FX bar returns being approximately mean-zero on this window.
- **Excess kurtosis is positive at every (pair × TF)** in 84 / 84 cases — the FX fat-tail signature. Magnitude increases with finer timeframe (1H median ~23, D1 median ~3).
- **ATR ACF at lag 1 > 0.5** at every (pair × TF) in 84 / 84 cases (medians ~0.985 across TFs). Volatility clusters strongly. Persistence falls but stays positive at lag 60 across all TFs (medians 0.30–0.43).
- **Returns ACF at lag 1 is small and negative** on the cross-pair median at all three TFs (median range −0.02 to −0.03), with per-pair extremes in [−0.07, +0.03].
- **Run-length means** for consecutive same-direction closes are ~1.9–2.0 bars in both up and down directions across TFs, with no asymmetry exceeding noise.
- **Distance-from-Kijun std** (in ATR units) decreases from 1H (1.78) to 4H (1.70) to D1 (1.56) on the cross-pair median.
- **Hour-of-day and day-of-week mean |return| effects** are reported per-pair × per-bin in the CSV. Per the descriptive-not-predictive rule the L1 result document does not name "elevated" bins.

Sanity gates passed (per L1 §5): row count, no NaN, returns mean within bootstrap CI of zero "across most pairs" (95.2%), kurtosis > 3 across most pairs (100%), ATR ACF lag 1 > 0.5 across most pairs (100%), determinism.

---

## 3. L2 — Multi-timeframe coherence findings

Source: `results/lchar/PHASE_L2_RESULT.md`. Stat families: `mtf_corr`, `mtf_lead_lag`, `mtf_cond_atr`, `mtf_alignment`. Trend definitions: Kijun-26 sign and SMA-20 sign. No-lookahead via L2 most-recently-completed-higher-TF convention.

Headline observations:

- **Same-bar return aggregation correlation (sum 1H within 4H vs 4H, sum 4H within D1 vs D1) = exactly 1.000** for every one of the 56 stats reported. This is the definitive plumbing check on the merge_asof / floor alignment used by L2 (and inherited by L3 and L4).
- **ATR-within correlation** (mean 1H ATR within a 4H bar vs the 4H bar's ATR; analogous for 4H↔D1) tracks but with cross-pair median 0.819 (1H↔4H) and 0.741 (4H↔D1). 19 / 28 pairs ≥ 0.8 at 1H↔4H; only 4 / 28 at 4H↔D1. This is **not** an alignment bug — it reflects formula-mismatch (different smoothing horizons across TFs); the same-bar return correlation = 1.000 confirms plumbing is correct. (L2 §6 caveat 1.)
- **Lead-lag correlations are all small** in magnitude. Cross-pair medians range from −0.023 to +0.005 across all 16 lead-lag stats (4 directions × 2 trends × 2 lags × 2 pairings, summarised across timeframes). Per-pair extremes widen to ±0.07.
- **MTF alignment state frequencies (sampled at 1H, MECE 6-state taxonomy)** sum to 1.0 within rounding (max deviation 5.4 × 10⁻¹¹) for every (pair × trend). Cross-pair medians (Kijun-based): 22.7% `3_up`, 20.2% `3_down`, 5.2% `2_up_mixed`, 4.8% `2_down_mixed`, 46.4% `opposed`, 0.1% `neutral_present`. SMA-based distribution is qualitatively similar. The `neutral_present` state is rare enough that L4 excludes it from trial enumeration (sample-size reasons).
- **Conditional ATR is monotone in D1 decile** for every one of the 28 pairs (decile_10 mean > decile_1 mean). 26 / 28 are strictly monotone across all 10 deciles; the two non-strict pairs (AUD_CAD, GBP_CAD) have small mid-table dips (≤ 1.5% relative). Cross-pair median ratio decile_10 / decile_1 = 2.19 (range 1.68 – 3.84).

Sanity gates: 7 hard checks PASS (including the same-bar return ID, alignment frequency sum, decile_10 > decile_1, no NaN, determinism). 2 soft misses on the ATR-within > 0.8 threshold, documented as formula-driven not bug-driven.

---

## 4. L3 — Cross-pair findings (28-pair universe)

Source: `results/lchar/PHASE_L3_RESULT.md`. Stat families: `xpair_static_corr`, `xpair_rolling_corr`, `xpair_matrix_stab`, `ccy_strength`, `pairpair_lead_lag`, `risk_basket`. The `pair` column is overloaded by family (per-pair-pair `P1__P2`, currency code, `_UNIVERSE_`, `_RISK_BASKET_`).

Headline observations:

- **Static correlation matrix** is symmetric (max |C − C.T| = 1.11 × 10⁻¹⁶) and diagonal = 1 (max |diag − 1| = 2.22 × 10⁻¹⁶) at all three TFs. Mean off-diagonal correlation is positive at all three TFs: +0.106 (1H), +0.102 (4H), +0.087 (D1) — the FX-shared-market-beta sanity expectation. Per-pair-pair range is wide: [−0.804, +0.904] depending on shared currencies.
- **Rolling 60-bar off-diagonal-mean correlation distribution** (universe-wide):

  | TF | p25 | p50 | p75 | p95 |
  |---|---:|---:|---:|---:|
  | 1H | 0.071 | 0.092 | 0.121 | 0.173 |
  | 4H | 0.064 | 0.083 | 0.112 | 0.163 |
  | D1 | 0.059 | 0.078 | 0.103 | 0.133 |

- **Matrix stability** (Frobenius diff `‖C_t − C_{t−1}‖_F`) p25 / p50 / p75 / p95: 1H 0.272 / 0.435 / 0.709 / 1.506; 4H 0.216 / 0.403 / 0.711 / 1.460; D1 0.304 / 0.467 / 0.676 / 1.194. Distributions are similar in scale across TFs.
- **Risk-on (AUD_USD, NZD_USD) vs risk-off (USD_JPY, USD_CHF) static correlation** is negative at all three TFs: −0.518 (1H), −0.515 (4H), −0.552 (D1). Rolling 60-bar correlation distribution stays negative through p90 at 4H and D1; 1H p90 approaches zero (−0.18).
- **Currency-strength rank persistence** (per-currency, D1) is essentially at the uniform-random baseline of 1/8 = 0.125 at lag 1 across all 8 currencies (cross-currency mean: 0.128 lag 1, 0.134 lag 5, 0.137 lag 20). Day-to-day rank is approximately independent on this window. JPY shows the highest persistence (~0.147 lag 1) and CAD the lowest (~0.121).
- **Top→bottom reversal frequency** (fraction of days where rank ≤ 3 was followed by rank ≥ 6 within 5 days) is 0.88–0.96 across the 8 currencies (mean 0.92).
- **Pair-pair lead-lag survivors** (filter: `|full-sample static corr| > 0.5`): 97 / 103 / 96 unique combos at 1H / 4H / D1. Cross-survivor lead-lag correlation medians at lag ±1 are within ±0.007, asymmetry-lag-1 medians within ±0.002. Per-survivor extremes widen at D1 due to smaller T (~1,360 days).

Sanity gates: 6 hard checks PASS (matrix symmetry, diagonal, currency zero-sum, no NaN, no-future-leak code review, determinism). 2 soft sign checks PASS (positive off-diagonal mean, negative risk-on/off correlation).

---

## 5. L4 — Conditional findings (the bridge to candidate signals)

Source: `results/lchar/PHASE_L4_RESULT.md`. 192 trials across 6 condition families × signal_TF × horizon, all pooled across the 28-pair universe. ATR-normalised next-N-bar log returns; Bailey & López de Prado deflated Sharpe with `N_trials = 300` family-budget correction.

DSR distribution across all 192 trials:

| Bracket | Count |
|---:|---:|
| DSR < 0.50 | 174 |
| 0.50 ≤ DSR < 0.80 | 4 |
| 0.80 ≤ DSR < 0.95 | 2 |
| DSR > 0.95 | 12 |

Per-family DSR summary:

| Family | trials | min DSR | median DSR | max DSR |
|---|---:|---:|---:|---:|
| univariate_extreme | 72 | 1.7 × 10⁻³⁵ | 2.0 × 10⁻⁸ | 1.000 |
| volatility_regime | 18 | 2.9 × 10⁻⁴¹ | 8.9 × 10⁻⁷ | 0.99996 |
| mtf_alignment | 30 | 2.6 × 10⁻¹¹⁵ | 8.9 × 10⁻¹² | 1.000 |
| cross_pair | 36 | 1.6 × 10⁻⁵⁰ | 8.8 × 10⁻⁷ | 0.385 |
| dow | 18 | 1.2 × 10⁻⁵² | 1.3 × 10⁻⁸ | 0.011 |
| run_length | 18 | 9.8 × 10⁻²¹ | 4.5 × 10⁻⁷ | 0.602 |

The bimodal distribution (vast majority at near-zero DSR, a small cluster near 1) is consistent with the design of the deflated Sharpe under a fixed `N_trials = 300` selection-bias correction: small Sharpe deviations from zero produce very small DSR after multiple-comparisons adjustment, while a few large-T trials with non-zero Sharpe saturate `Φ` near 1.

Sanity gates: 6 hard checks PASS (DSR ∈ [0,1], all metrics finite, trial count ≤ 300, all T ≥ 100, no NaN, determinism). 2 soft checks PASS, 1 soft check (h=1 univariate-extreme pos/neg sign opposition) PARTIAL — 5 / 12 strict opposite, documented in L4 §6 caveat 1 as a real descriptive observation about post-extreme mean-reversion at higher TFs.

---

## 6. Where DSR > 0.95 trials emerged

Per L0 §3 this section reports where the mechanical metric clears the locked threshold; it does **not** interpret these as tradeable patterns. All 12 trials are listed below; the L5 top-5 registry sorts and selects from these.

All 12 DSR > 0.95 trials sit at **signal_TF = 1H**. None of the 4H or D1 signal-TF trials cleared the threshold.

| Family | TF | Count of DSR > 0.95 trials |
|---|---|---:|
| mtf_alignment | 1H | 7 |
| univariate_extreme | 1H | 3 |
| volatility_regime | 1H | 2 |
| **total** | | **12** |

Within `mtf_alignment` the 7 high-DSR trials are concentrated on the `2_down_mixed` and `3_down` states (Kijun and SMA trends; horizons 1, 24, 120). Within `univariate_extreme`, both the `bar_range_top_decile` and `abs_return_top_decile` bases produce clearing trials at h=1 with the `neg` sign (and `bar_range pos` at h=1 also clears). Within `volatility_regime`, only the `d1_atr_top_decile` base clears (at horizons 24 and 120 — neither at h=1).

The full 12 trials, sorted by DSR descending then Sharpe descending then T descending (the L0 sort key):

| Rank | trial_id | TF | DSR |
|---:|---|---|---:|
| 1 | `TRIAL__univariate_extreme__abs_return_top_decile__neg__h_001` | 1H | 1.000000 |
| 2 | `TRIAL__mtf_alignment__2_down_mixed__kijun__h_120` | 1H | 0.999999 |
| 3 | `TRIAL__volatility_regime__d1_atr_top_decile__any__h_120` | 1H | 0.999964 |
| 4 | `TRIAL__univariate_extreme__bar_range_top_decile__neg__h_001` | 1H | 0.999951 |
| 5 | `TRIAL__mtf_alignment__2_down_mixed__kijun__h_024` | 1H | 0.999769 |
| 6 | `TRIAL__volatility_regime__d1_atr_top_decile__any__h_024` | 1H | 0.999424 |
| 7 | `TRIAL__mtf_alignment__3_down__sma__h_024` | 1H | 0.997459 |
| 8 | `TRIAL__univariate_extreme__bar_range_top_decile__pos__h_001` | 1H | 0.992599 |
| 9 | `TRIAL__mtf_alignment__3_down__sma__h_001` | 1H | 0.992218 |
| 10 | `TRIAL__mtf_alignment__3_down__kijun__h_001` | 1H | 0.988961 |
| 11 | `TRIAL__mtf_alignment__3_down__kijun__h_024` | 1H | 0.987134 |
| 12 | `TRIAL__mtf_alignment__2_down_mixed__sma__h_024` | 1H | 0.969536 |

The L5 top-5 registry takes the first five rows of this table; entries 6–12 are below the registry cap and do not enter L6+ via the L0 mechanical bridge. Per L0 §9 the only way to revisit these is a future re-planning phase with its own pre-registration.

---

## 7. Where expected edges did not appear

This section is descriptive: where the L4 trials closed below threshold despite carrying L1–L3 structural findings into the conditional-distribution stage.

- **Cross-pair conditions** (`cross_pair` family, 36 trials) cleared none. The per-base maximums are: `risk_basket_corr_bottom_decile_mag` 0.385, `matrix_frobenius_shift_top_decile` 0.0045, `risk_basket_corr_top_decile_mag` 0.0020, `usd_strength_extreme` 0.0032. Despite L3's robust −0.51 to −0.55 risk-on/risk-off static correlation, conditioning on the rolling-correlation extremes did not produce DSR > 0.95 for any of the (TF × horizon × base) combinations in the trial budget.
- **Day-of-week conditions** (`dow` family, 18 trials) cleared none — max DSR 0.011. Consistent with L1's per-pair × per-TF dow_effect distributions which the L1 result document does not flag as anomalous.
- **Currency-strength extremes** (one of the four `cross_pair` bases) cleared none — the L3 finding that daily currency-rank persistence sits at the 1/8 uniform baseline carries through.
- **Run-length conditions** (`run_length` family, 18 trials) cleared none — max DSR 0.602 (one trial near-but-below the threshold).
- **4H and D1 signal_TFs** produced no DSR > 0.95 trial in any family.

---

## 8. Methodology observations

- **Kurtosis convention in DSR** (L4 §3.5 / §6 caveat 3): the Bailey & López de Prado denominator uses raw Pearson kurtosis (= `excess + 3`). The L0 spec text references `scipy.stats.kurtosis(fisher=True)` (which returns excess); a literal application would make the denominator go negative for moderate Sharpe and is clearly not the canonical form. The L4 script computes excess via scipy and adds 3 inside `dsr_value()`. The reported `kurt_r` stat in `layer4_conditional.csv` is **excess** (matches L1 convention). This is the only deliberate convention departure across L1–L4 and is documented in three places (L4 §3.5, L4 §6, this section).
- **iid percentile bootstrap throughout** (L1 §3.3, L2 §3.4, L3 §3.5, L4 §3.6). Same `np.random.default_rng(np.random.PCG64(42))` seed across phases. Resampling unit varies by stat: time-series sample for univariate moments / percentiles, aligned-pair sample for cross-TF correlations, conditional-return observation for L4 trials, etc. **For autocorrelation-style measures the iid-resampled CI describes the estimator distribution under the iid null** (not a CI on the population ACF). Block bootstrap is left for any subsequent layer that requires dependence-aware uncertainty.
- **Day-of-week extremes are retrospective** (L4 §6 caveat 2). The per-pair top / bottom day is determined from the full-sample mean `|log_ret|` per weekday and treated as a structural property of the pair. A walk-forward determination would be required for any L6+ signal arc that adopts a DOW condition; this is L6's problem.
- **Per-bar timestamps are MT5-broker time**, no time-zone normalisation. Hour bins (h00..h23) and weekday bins refer to bar-start times as recorded in the on-disk CSVs. Consistent across pairs (verified by `scripts/audit_data_integrity.py`).
- **W1 not pulled** — per `L_ARC_PLAN.md` §10. If a future arc needs D1↔W1 coherence or a W1-aligned signal filter, it must add the W1 dataset and re-extend.

---

## 9. Caveats summary

Carrying forward from L1–L4:

1. **Inner-join alignment in L3** (and inherited by L4 universe series) drops a small number of bars (≤ 1% per TF) where one or more pairs has a missing bar relative to the rest. Per-pair univariate stats in L1 are unaffected.
2. **JPY-quoted pairs distort raw-price magnitude summaries** (L1, L2 caveats). All correlation- and rank-based statistics are scale-invariant and unaffected. The `mtf_cond_atr` and L1 `atr` family report raw-price values; consumers of those numbers should normalise per-pair before cross-pair comparisons.
3. **ATR-within correlation < 0.8 for many pairs in L2** (ATR formula mismatch across TFs — different smoothing horizons). Same-bar return aggregation = 1.000 confirms alignment plumbing is correct; the < 0.8 result is descriptive, not a bug.
4. **DSR formula is asymptotic in T**. The bootstrap CI on DSR is an iid percentile envelope; for moderate-T trials (smallest T = 843 in L4) the CI may be wider than the asymptotic Bailey-López-de-Prado expression suggests.
5. **The 192-trial L4 budget leaves 108 trials of headroom under the 300-cap**. The DSR formula uses `N_trials = 300` regardless — i.e. the selection-bias correction is conservative relative to the actual run. If a future re-planning phase changes the family budget, the YAML constant in `configs/lchar/layer4.yaml` and the DSR computation must be updated jointly.

---

This atlas closes the descriptive output of the L characterization arc. The mechanical bridge to candidate signals is in `LCHAR_TOPN_REGISTRY.md`. The arc-level closure disposition and L6+ pointer are in `results/lchar/L_ARC_CLOSURE.md`.
