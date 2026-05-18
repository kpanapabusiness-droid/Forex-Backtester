# Lω-v2 — cross-TF feature expansion (interpretation 3 test)

> Discovery dispatch following the B1-B4 post-mortem's anti-evidence flag
> #6: "stable importance + collapsing AUC is observationally identical
> between Int 1 and a weak feature set that captures a real-but-weak
> phenomenon." Two changes vs B1-B4: cross-TF feature expansion (51 total
> features, 18 added on top of B1-B4's 33 — see §4 for inventory) and a
> relaxed clean_move label that drops the reached_1R requirement. The
> full 51-feature set is trained (NOT top-15 by univariate AUC) using
> identical RF hyperparameters and fold-window indexing to PR #133 for
> direct comparability. Scope: 1H + 4H; D1 skipped per dispatch.

---

## 1. Headline

| Timeframe | n (rows) | n (after dropna) | clean_relaxed rate | mean fold AUC | worst fold AUC | Δ worst vs B1-B4 | full-data AUC | n_features |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1H | 1,033,074 | 1,032,603 | 0.1582 | 0.5352 | **0.4987** | **+0.0146** | 0.6816 | 51 |
| 4H |   255,174 |   255,138 | 0.2403 | 0.5065 | **0.4736** | **+0.0003** | 0.7144 | 51 |

**B1-B4 baselines** (from PR #133 `LOMEGA_B1_B4_SUMMARY.md`):
- 1H worst-fold AUC: 0.4841 (fold 7); mean 0.5375; stdev 0.0340; full-data 0.6359
- 4H worst-fold AUC: 0.4733 (fold 6); mean 0.5040; stdev 0.0274; full-data 0.6852

**Gates per dispatch:**
- worst-fold AUC ≥ 0.55 (Step 4 Pipeline E relaxed-for-discovery threshold)
- AND Δ worst-fold AUC ≥ +0.05 vs B1-B4 (real improvement, not noise)

**Verdict: NEITHER GATE CLEARED at either timeframe.**

- 1H: worst-fold 0.4987 < 0.55 (fail by 0.05), Δ +0.0146 < +0.05 (fail by 0.04)
- 4H: worst-fold 0.4736 < 0.55 (fail by 0.08), Δ +0.0003 < +0.05 (fail by 0.05)

The B1-B4 worst-fold ceiling is essentially **unchanged** under v2's richer
cross-TF feature set. The dispatch's pre-committed conclusion applies:
**feature expansion does NOT rescue Lω**. Strong evidence FOR structural
pivot (interpretation 2 dominant) at both timeframes. See §7 for the
combined-dispatch interpretation and §10 for the recommended next move.

Sample-comparability footnote: v2 drops 471 rows at 1H (471/1,033,074 =
0.045%) and 36 rows at 4H (0.014%) due to new cross-TF features' warmup
NaN (w_ema_20_slope needs ~40 W bars; d1_realised_vol_20 ~21 D1 bars).
These rows are at the dataset's chronological start and do not affect
the val_n in any fold materially.

---

## 2. Label sanity

`clean_move_relaxed = True` if both required conditions hold:
- `mono_pre_peak ≥ 0.55`
- `mae_pre_peak_R > -1.0`

AND at least one of:
- `mfe_max_R ≥ 1.0`
- `reached_0.5R_pre_peak` (equivalently `mfe_max_R ≥ 0.5`)

`clean_score_relaxed` counts how many of these four sub-conditions are
met. Range is **[0, 4]**, not the dispatch's literal {0..3} — the two MFE
conditions are listed as separate sub-conditions even though mfe_max_R
≥ 1.0 strictly implies reached_0.5R_pre_peak. Score=4 cells are bars
that pass mono + mae + both MFE thresholds; score=3 typically means
either MFE ≥ 0.5 but not ≥ 1.0 (with mono + mae passing), or other
combinations.

| Timeframe | base rate | score=0 | 1 | 2 | 3 | 4 | within 30-50% expected? | within 15-70% extreme? |
|---|---:|---:|---:|---:|---:|---:|---|---|
| 1H | **0.1582** |  0.98% |  4.73% | 40.56% | 39.47% | 14.26% | **NO** (below) | YES (barely; 15.82% just above the 15% floor) |
| 4H | **0.2403** |  1.31% |  6.90% | 26.68% | 43.34% | 21.78% | **NO** (below) | YES |

The relaxation lifted clean_move from B1-B4's strict label by a modest
amount:

| TF | B1-B4 strict rate | v2 relaxed rate | relative lift |
|---|---:|---:|---:|
| 1H | 0.1273 | 0.1582 | +24% |
| 4H | 0.1957 | 0.2403 | +23% |

Both relaxed rates land **below** the dispatch's expected 30-50% band.
This is informative — it says even the relaxed conjunction is rare at
intra-day timeframes. The principal reason is that `mono_pre_peak ≥ 0.55`
is the binding constraint, not the MFE threshold: dropping `reached_1R`
and lowering `mfe_max_R ≥ 1.5 → ≥ 1.0/0.5R` adds ~3 percentage points
of clean_move; relaxing the mono-pre-peak condition would add much more.
The relaxation chosen by the dispatch holds the path-shape requirement
constant and only loosens the magnitude requirement — which is the right
call for testing whether features predict CLEAN paths, not just any
paths.

**Per-pair contribution** (top contributor + flag check):

| Timeframe | top pair | top share | dispatch flag (>40%)? |
|---|---|---:|---|
| 1H | AUD_JPY | 4.61% | OK |
| 4H | USD_JPY | 4.26% | OK |

JPY crosses dominate the top of every per-pair table (same as B1-B4) but
no pair contributes anywhere near the 40% flag. Per-pair label
distribution is well spread. Full per-pair tables in
`timeframe_<tf>/label_distribution.txt`.

---

## 3. Per-fold AUC table

### 1H — v2 vs B1-B4

| fold | train_n | val_n | val_clean_rate (v2) | val AUC v2 | val AUC B1-B4 | Δ |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 129,078 | 129,075 | 0.1611 | 0.5342 | 0.5298 | +0.004 |
| 2 | 258,153 | 129,075 | 0.1774 | 0.5472 | 0.5432 | +0.004 |
| 3 | 387,228 | 129,075 | 0.1665 | 0.5444 | 0.5962 | **−0.052** |
| 4 | 516,303 | 129,075 | 0.1556 | 0.5232 | 0.5256 | −0.002 |
| 5 | 645,378 | 129,075 | 0.1404 | 0.5506 | 0.5696 | −0.019 |
| 6 | 774,453 | 129,075 | 0.1527 | 0.5481 | 0.5140 | **+0.034** |
| 7 | 903,528 | 129,075 | 0.1254 | **0.4987** | **0.4841** | **+0.015** |

Fold AUC stdev fell from B1-B4's 0.0340 to v2's 0.0174 — **fold-to-fold
AUC variance ~halved**. v2 is more uniform across folds but no higher
on average. The improvement is concentrated in folds 6 + 7 (the 2024-
2025 collapse region), the loss in fold 3 (the COVID-aftermath peak).

### 4H — v2 vs B1-B4

| fold | train_n | val_n | val_clean_rate (v2) | val AUC v2 | val AUC B1-B4 | Δ |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 31,894 | 31,892 | 0.2329 | 0.4923 | 0.4833 | +0.009 |
| 2 | 63,786 | 31,892 | 0.2359 | 0.4922 | 0.4741 | +0.018 |
| 3 | 95,678 | 31,892 | 0.2727 | 0.5348 | 0.5290 | +0.006 |
| 4 | 127,570 | 31,892 | 0.2373 | 0.5222 | 0.5352 | −0.013 |
| 5 | 159,462 | 31,892 | 0.2482 | 0.5383 | 0.5399 | −0.002 |
| 6 | 191,354 | 31,892 | 0.2360 | 0.4917 | 0.4733 | +0.018 |
| 7 | 223,246 | 31,892 | 0.1948 | **0.4736** | 0.4931 | **−0.020** |

**The worst fold MOVED from F6 (B1-B4) to F7 (v2).** v2 improved F6 by
+0.018 but degraded F7 by −0.020. Net: worst-fold AUC essentially flat
(Δ +0.0003). This is a redistribution, not a real lift.

**Concentration check.** No timeframe shows the improvement landing in
1-2 folds with the worst untouched — at 4H the improvement zeroes out
across folds (some up, some down); at 1H the improvement helps folds 6
and 7 but the worst (still F7) remains below 0.5. The headline gate
verdict is the right metric here.

---

## 4. Cross-TF feature contribution

Feature inventory (51 total per TF):
- **33 reused from B1-B4** (compute_pair_features + DXY proxy + B1-B4
  D1/H4 anchors). Imported byte-identical from `lomega_b1_b4.py`.
- **18 added in v2**: 11 D1 panel features (vol regime / swing /
  EMA-stack / momentum), 1 H4 EMA-stack (`h4_ema_stack_up`), 1 H1
  anchor (`h1_ema_50_slope_at_entry`, only at 4H signal TF; absent at
  1H since signal IS H1), 1 W EMA-20 slope (`w_ema_20_slope`), 3
  cross-TF agreement (`cross_tf_trend_alignment`,
  `cross_tf_trend_strong_up`, `d1_h4_slope_agreement`).

Note: at 1H, the H1 slope element of cross_tf_trend_alignment is the
signal-TF's own `ema_50_slope`, so `h1_ema_50_slope_at_entry` is absent
at 1H. At 4H, the H4 slope element is the signal-TF's own `ema_50_slope`
and the H1 anchor is added explicitly. Both TFs end up with 51 features
total.

### 1H — top-15 features by full-data RF importance

| Rank | Feature | Importance | Provenance |
|---:|---|---:|---|
| 1 | d1_ema_50_slope_at_entry | 0.0899 | B1-B4 |
| 2 | **w_ema_20_slope** | 0.0898 | **v2 NEW** |
| 3 | atr_200 | 0.0890 | B1-B4 |
| 4 | **d1_atr_14_normalised** | 0.0660 | **v2 NEW** |
| 5 | **d1_realised_vol_20** | 0.0636 | **v2 NEW** |
| 6 | atr_50 | 0.0593 | B1-B4 |
| 7 | **d1_momentum_20_atr** | 0.0450 | **v2 NEW** |
| 8 | **d1_atr_percentile_60** | 0.0439 | **v2 NEW** |
| 9 | atr_14 | 0.0437 | B1-B4 |
| 10 | ema_200_slope | 0.0415 | B1-B4 |
| 11 | h4_ema_50_slope_at_entry | 0.0396 | B1-B4 |
| 12 | **d1_momentum_5_atr** | 0.0335 | **v2 NEW** |
| 13 | atr_ratio_14_200 | 0.0335 | B1-B4 |
| 14 | **d1_range_compression_20** | 0.0256 | **v2 NEW** |
| 15 | realised_vol_20 | 0.0213 | B1-B4 |

### 4H — top-15 features by full-data RF importance

| Rank | Feature | Importance | Provenance |
|---:|---|---:|---|
| 1 | ema_200_slope | 0.0837 | B1-B4 |
| 2 | **w_ema_20_slope** | 0.0798 | **v2 NEW** |
| 3 | d1_ema_50_slope_at_entry | 0.0763 | B1-B4 |
| 4 | atr_200 | 0.0586 | B1-B4 |
| 5 | **d1_atr_percentile_60** | 0.0569 | **v2 NEW** |
| 6 | atr_50 | 0.0516 | B1-B4 |
| 7 | **d1_atr_14_normalised** | 0.0481 | **v2 NEW** |
| 8 | **d1_realised_vol_20** | 0.0471 | **v2 NEW** |
| 9 | atr_ratio_14_200 | 0.0434 | B1-B4 |
| 10 | ema_50_slope | 0.0405 | B1-B4 |
| 11 | atr_14 | 0.0375 | B1-B4 |
| 12 | dxy_proxy_ema_20_slope | 0.0322 | B1-B4 |
| 13 | **d1_momentum_20_atr** | 0.0316 | **v2 NEW** |
| 14 | atr_ratio_14_50 | 0.0253 | B1-B4 |
| 15 | ema_spread_20_50_atr | 0.0242 | B1-B4 |

### Summary

| TF | new in top-3 | new in top-5 | new in top-15 | highest-rank new feature |
|---|---:|---:|---:|---|
| 1H | 1 (`w_ema_20_slope` @ #2) | 3 | **8** | `w_ema_20_slope` (rank 2, importance 0.0898, univariate AUC 0.5119) |
| 4H | 1 (`w_ema_20_slope` @ #2) | 2 | **5** | `w_ema_20_slope` (rank 2, importance 0.0798, univariate AUC 0.5119) |

**`w_ema_20_slope` is the highest-ranking new feature at both
timeframes**, sitting at rank #2 just behind whichever existing trend
slope leads. The RF is genuinely allocating capacity to the new D1
panel — at 1H, 8 of the top-15 features are v2 additions, and at 4H,
5 of the top-15 are v2 additions. The new features are NOT being
ignored — they are getting tree splits.

**But their univariate AUCs are weak (top new feature at 4H is
`d1_atr_percentile_60` at AUC 0.5368; top new feature at 1H is
`d1_ema_stack_up` and `d1_realised_vol_20` near AUC 0.52).** The full-
data multivariate AUC rose accordingly (1H +0.046, 4H +0.029 vs B1-B4),
but the worst-fold AUC did not — the new features add capacity that
fits in-sample but does not generalise out-of-sample.

---

## 5. Feature stability across folds

Spearman ρ of per-fold RF importance vector across all 51 features
between consecutive folds (F1↔F2 ... F6↔F7) and the extremes (F1↔F7).
Comparable to postmortem D3 — but note that D3 was on top-15 (rank
correlation over 15 items, very tight), while v2 is over all 51 features
(rank correlation more sensitive to small permutations).

| TF | F1↔F2 | F2↔F3 | F3↔F4 | F4↔F5 | F5↔F6 | F6↔F7 | F1↔F7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1H | 0.984 | 0.988 | 0.988 | 0.995 | 0.998 | 0.997 | 0.964 |
| 4H | 0.964 | 0.965 | 0.989 | 0.994 | 0.995 | 0.993 | 0.972 |

**All seven cross-fold correlations are ≥ 0.96 at both TFs**, matching
the B1-B4 postmortem's "STABLE at all timeframes" diagnosis (D3 reported
ρ in 0.96-0.997 for B1-B4 top-15 at 1H/4H). The new features take their
place in the ranking and stay there — the RF does not chase fold-
specific noise after the expansion. This actually **rules out one more
interpretation 3 hypothesis** ("v2 features unstable across folds") and
reinforces interpretation 2 (the underlying ceiling is genuine).

---

## 6. Boruta results

**Not run.** The `boruta` package is not installed in the environment;
installing it and running per-feature shadow-importance over the 51-
feature set across both timeframes would exceed the dispatch's 1hr/TF
compute envelope. The dispatch explicitly permits skipping ("Skip if
Boruta runtime exceeds 1 hour per timeframe. Document skip explicitly").

If a follow-up dispatch wants Boruta validation, run after `pip install
boruta` on the existing `labels_relaxed.csv` + `features_expanded.csv`
intermediates (deterministic inputs — no re-run of the full pipeline
needed).

---

## 7. Comparison interpretation

Three-way verdict per dispatch §"Comparison interpretation":

- **Strong gate cleared** — worst-fold AUC ≥ 0.60 AND Δ ≥ 0.10: strong
  evidence for interpretation 3 (feature set binding). Proceed to B5-B6
  signal nomination on the v2 feature set.
- **Marginal gate cleared** — worst-fold AUC 0.55-0.60 AND Δ 0.05-0.10:
  weak interpretation 3 support. B5-B6 with realistic expectations.
- **Gate not cleared** — worst-fold AUC < 0.55 OR Δ < 0.05: feature
  expansion does not rescue Lω. Combined with the parallel regime-
  conditional dispatch result, determines the final Lω disposition.

**Outcome:** Gate not cleared at either TF. Both worst-fold AUCs sit
below 0.50; both Δ values sit below +0.02 (1H) / essentially zero (4H).
The v2 feature batch's RF importances ARE non-trivial (5-8 of the
top-15 features at each TF are new), but they do not translate into
out-of-fold AUC improvement.

**This is strong evidence for interpretation 2 (structural limit) being
dominant** at the 1H + 4H entry-time / hold-window combinations Lω
covers. The dispatch's anti-evidence flag #6 from the post-mortem ("Int
1 vs weak features is observationally identical") is resolved here: a
specifically richer feature set was tested, RF allocated capacity to
the new features, and the worst-fold AUC remained at the same B1-B4
ceiling. The "weak features" alternative explanation is therefore
substantially weakened — the new features ARE used, they just don't
fix the out-of-fold problem.

Combined with the parallel `discovery/lomega_regime_conditional`
dispatch (separate report), this provides the joint evidence the chat-
track committed to use for the Lω disposition:

- If regime-conditional ALSO null: formal Lω closure; structural pivot
  to Tier A structural-event signals (failed-breakout reversal,
  liquidity sweep + reclaim, compression-then-expansion) nominated via
  L-arc registry.
- If regime-conditional CLEARS gate in any vol-bin: regime-conditional
  modelling is the right framing, and v2 cross-TF features should be
  re-tested within the cleared vol bin (cheap re-run on
  `features_expanded.csv` filtered by `atr_percentile_60` bin).

The combination is a chat-track analytical decision after both
dispatches return; this report does not pre-empt it.

---

## 8. No-lookahead verification

10 bars per timeframe; for each, stored cross-TF feature values were
manually reproduced from raw OHLC restricted to bars ≤ t at each source
TF. Cross-TF lag rules:

- D1 anchor: largest D1 bar with `d1_date < t.date()` (strict prior
  calendar day; `searchsorted side="left" - 1`).
- H4 anchor: largest H4 bar with `h4_time + 4h ≤ t` (most recent CLOSED
  H4 bar; `searchsorted side="right" - 1` against `t - 4h`).
- H1 anchor: largest H1 bar with `h1_time + 1h ≤ t` (only used at 4H
  signal TF; at 1H, signal bar IS H1).
- W anchor: largest W bar with `w_time + 7d ≤ t`.

Per-bar feature checks per row: stored vs recomputed `atr_14`,
`d1_ema_50_slope_at_entry`, `d1_ema_stack_up`, `w_ema_20_slope`, and
`h4_ema_stack_up`. (`h1_ema_50_slope_at_entry` values are displayed but
not asserted in the pass counter.)

Files: `timeframe_<tf>/lookahead_spot_check.txt`.

| TF | bars checked | stored-vs-recomputed matches | comments |
|---|---:|---:|---|
| 1H | 10 | **50/50** | tol 1e-9; bars span 2020-07 to 2025-11 across 10 different pairs |
| 4H | 10 | **50/50** | tol 1e-9; bars span 2020-07 to 2025-01 |

**Status: PASS at both timeframes.** No cross-TF lag boundary violations
detected.

---

## 9. Anti-evidence and uncertainty

What remains open after this dispatch:

1. **Improvement concentration.** At 1H, the +0.015 worst-fold delta is
   genuinely on the worst fold (F7) and does not come from inflating
   other folds (in fact F3 dropped −0.052 and the mean is unchanged).
   That makes the small worst-fold gain less suspect — but it is also
   too small to clear the +0.05 threshold. At 4H, the worst-fold delta
   is essentially zero and the worst fold MOVED from F6 to F7, which is
   a sideways result, not improvement. Neither TF has a "real but
   marginal" signal; both are flat.

2. **Univariate vs multivariate.** Top new features at each TF have
   univariate AUC 0.51-0.54 (1H best new: `d1_realised_vol_20` 0.5236;
   4H best new: `d1_atr_percentile_60` 0.5368). These are univariate
   detectable, not multivariate-only. So the "new features aren't real
   signal" hypothesis is partially rejected — they are real, just
   marginal. The multivariate AUC lift is also marginal.

3. **Pair concentration.** No pair contributes > 40% of relaxed labels
   at either TF (top 4-5% per pair). Multivariate AUC effects are
   pooled across all 28 pairs; no single pair drives the v2 result.
   (Full per-pair × per-fold tables: `b4_per_pair_per_fold.csv`.)

4. **Label-shift.** The relaxed label's score distribution puts most
   of the dataset mass at score=2 (1H 40.6%, 4H 26.7%) and score=3 (1H
   39.5%, 4H 43.3%) — but most score-3 bars are NEGATIVES (e.g., mono +
   mfe1 + mfe05 with mae failing). The binary positives decompose as:
   - 1H: 163,437 total → 147,292 from score=4 (90.1%; hard 1R+ wins)
     + 16,145 from score=3 (9.9%; soft 0.5-1R wins, mono+mae+mfe05 only)
   - 4H: 61,315 total → 55,563 from score=4 (90.6%) + 5,752 from
     score=3 (9.4%)
   So ~10% of v2 positives at each TF are "softer" wins that the
   strict B1-B4 label would have rejected. This is consistent with the
   23-24% relative lift in clean_move rate seen in §2. The bulk of
   v2 positives are still the same 1R+ winners B1-B4 evaluated.
   Interpretation: the gate failure cannot be attributed to "the
   model is being asked to predict a weaker target" — 90% of the
   target is identical to B1-B4's.

5. **Sample of new features.** The 15 (effectively 18) new features
   were chosen by the dispatcher for higher-TF state. A different
   batch (e.g., realized vol of vol, inter-market features outside FX,
   order-flow proxies) could land different results. This dispatch
   tests THIS specific feature batch, not "any possible feature
   expansion".

6. **w_ema_20_slope dominance.** The W EMA-20 slope is rank #2 by RF
   importance at BOTH timeframes despite having univariate AUC of only
   0.5119. The RF is finding it informative in combination — but the
   worst-fold AUC isn't improving. One interpretation: the W slope is
   a regime variable (low-freq trend direction) that the RF uses to
   condition on, but the regime DEFINED BY weekly trend is itself
   non-stationary (2025 had unusual weekly direction patterns vs
   2020-2024 training history). The parallel regime-conditional
   dispatch would test this directly.

7. **Determinism.** Captured sha256 hashes of all 10 deterministic
   outputs per TF after run 1 in `timeframe_<tf>/_determinism_run1_hashes.txt`;
   re-running the full pipeline produces byte-identical outputs (verified
   per `_determinism_results.txt` in each TF dir). Cross-PR
   reproducibility is preserved via fixed `random_state=42` and
   sklearn 1.8 RF determinism.

---

## 10. Recommendation

**The v2 cross-TF expansion does not rescue Lω.** Both worst-fold AUC
gates fail by margins large enough that no plausibly-defensible "this
nearly cleared" framing applies (1H is the closer of the two at 0.4987
vs 0.55 — 0.05 short).

Conditional on the parallel regime-conditional dispatch's result:

- **If regime-conditional also null at both TFs**:
  - Formal Lω closure. The discovery question — "is there extractable
    structure from price-derived features at entry time that predicts
    clean long-only forward paths at 1H/4H?" — gets a definitive no
    after two specifically targeted attempts (relaxed label + cross-TF
    expansion; vol-regime conditional). Pivot to **Tier A structural-
    event signals** (failed-breakout reversal, liquidity sweep +
    reclaim, compression-then-expansion). These get nominated via L-arc
    registry as hand-defined signals, NOT discovered through Lω
    feature-mining.
  - Calibration implication for L-arc protocol v2.2: the L-arc Step 4
    AUC ceiling for new feature-mined signals is empirically modest;
    `worst_fold_auc ≈ 0.49-0.55` is the realistic-best floor on
    discovery-derived signals at intra-day FX timeframes.

- **If regime-conditional clears gate in any vol bin**:
  - Lω-v2's flat result becomes evidence for "the discovery question
    is conditional on regime". Re-test v2 cross-TF features filtered to
    the cleared vol bin using the existing `features_expanded.csv`
    intermediates (no full pipeline re-run needed — ~15-30 min for the
    conditional fit at 1H).

- **Either way, B5-B6 signal nomination from Lω discovery is NOT
  recommended at this dispatch's result** — the v2 worst-fold ceiling
  is too low to support a deployable signal candidate. B5-B6 should be
  paused pending the regime-conditional verdict.

---

## Validation checklist

- [x] 1H + 4H processed; D1 skipped per dispatch scope
- [x] Relaxed label base rate documented per TF; flagged when below 30%
  expected band (1H 15.82%, 4H 24.03% — both below; not a blocker since
  both > 15% extreme floor)
- [x] All 51 features computed (33 reused B1-B4 + 18 new v2; no
  features deferred — W data + D1 data + H1 data + H4 data all present
  for 28/28 pairs)
- [x] No-lookahead spot-check 10 bars per timeframe, including cross-TF
  lag boundaries (D1 / H4 / H1 / W); 50/50 matches at tol 1e-9 each TF
- [x] 7-fold time-series CV with identical fold indexing to B1-B4; row
  count parity within 0.05% (471 rows dropped at 1H, 36 at 4H, all in
  pre-2020 warmup window which is filtered out anyway)
- [x] Determinism PASS on labels and AUC tables (run-1 hashes captured
  in `_determinism_run1_hashes.txt`; re-run byte-identical per
  `_determinism_results.txt`)
- [x] LOMEGA_V2_CROSSTF_SUMMARY.md contains all 10 required sections
- [x] Comparison vs B1-B4 baseline explicit per fold

---

## Reproducing

```
py scripts/lomega/lomega_v2_crosstf.py --tf all      # both TFs (~35-45 min wall, single-threaded I/O)
py scripts/lomega/lomega_v2_crosstf.py --tf 1h       # 1H only (~30 min)
py scripts/lomega/lomega_v2_crosstf.py --tf 4h       # 4H only (~5 min)
```

Outputs land under `results/lomega/v2_crosstf/timeframe_<tf>/`. Heavy
`labels_relaxed.csv` and `features_expanded.csv` are gitignored (1H
features.csv ≈ 700MB, 4H ≈ 160MB; identical scale to B1-B4 due to the
same row count). All small analysis outputs are committed:
`b3_univariate_expanded.csv`, `b4_multivariate_expanded.csv`,
`b4_feature_importances_full.csv`, `b4_per_pair_per_fold.csv`,
`b4_summary.txt`, `feature_importance_drift.csv`,
`feature_importance_spearman.csv`, `label_distribution.txt`,
`lookahead_spot_check.txt`, this summary, and the determinism artefacts.
