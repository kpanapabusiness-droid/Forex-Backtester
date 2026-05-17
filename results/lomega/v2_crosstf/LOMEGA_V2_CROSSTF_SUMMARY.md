# Lω-v2 — cross-TF feature expansion (interpretation 3 test)

> Discovery dispatch following the B1-B4 post-mortem's anti-evidence flag
> #6: "stable importance + collapsing AUC is observationally identical
> between Int 1 and a weak feature set that captures a real-but-weak
> phenomenon." Two changes vs B1-B4: cross-TF feature expansion (~15 new
> features) and a relaxed clean_move label that drops the reached_1R
> condition. The full 45-feature set is trained (NOT top-15 by univariate
> AUC) using identical RF hyperparameters and fold dates to PR #133 for
> direct comparability. Scope: 1H + 4H; D1 skipped per dispatch.

---

## 1. Headline

| Timeframe | n (rows) | clean_move_relaxed rate | mean fold AUC | worst fold AUC | Δ worst vs B1-B4 | full-data AUC | n_features |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1H | _PLACEHOLDER_ | _PLACEHOLDER_ | _PLACEHOLDER_ | _PLACEHOLDER_ | _PLACEHOLDER_ | _PLACEHOLDER_ | _PLACEHOLDER_ |
| 4H | _PLACEHOLDER_ | _PLACEHOLDER_ | _PLACEHOLDER_ | _PLACEHOLDER_ | _PLACEHOLDER_ | _PLACEHOLDER_ | _PLACEHOLDER_ |

**B1-B4 baseline (from PR #133 LOMEGA_B1_B4_SUMMARY.md):**
- 1H worst-fold AUC: 0.4841 (fold 7)
- 4H worst-fold AUC: 0.4733 (fold 6)

**Gates per dispatch:**
- worst-fold AUC ≥ 0.55 (Step 4 Pipeline E relaxed-for-discovery threshold)
- AND Δ worst-fold AUC ≥ +0.05 vs B1-B4 baseline (real improvement, not noise)

**Verdict:** _PLACEHOLDER_

---

## 2. Label sanity

clean_move_relaxed = required {mono_pre_peak ≥ 0.55, mae_pre_peak_R > -1.0}
plus at-least-one {mfe_max_R ≥ 1.0, reached_0.5R_pre_peak}.

clean_score_relaxed counts sub-conditions met. Range is [0, 4] because the
two MFE conditions are listed as separate sub-conditions even though
mfe_max_R ≥ 1.0 strictly implies reached_0.5R_pre_peak — the dispatch's
literal {0..3} range is a minor inconsistency in the dispatch text and
does not change the binary label.

| Timeframe | base rate | score=0 | 1 | 2 | 3 | 4 | within 30-50% band? | within 15-70% extreme? |
|---|---:|---:|---:|---:|---:|---:|---|---|
| 1H | _PLACEHOLDER_ | _PLACEHOLDER_ | _PLACEHOLDER_ | _PLACEHOLDER_ | _PLACEHOLDER_ | _PLACEHOLDER_ | _PLACEHOLDER_ | _PLACEHOLDER_ |
| 4H | _PLACEHOLDER_ | _PLACEHOLDER_ | _PLACEHOLDER_ | _PLACEHOLDER_ | _PLACEHOLDER_ | _PLACEHOLDER_ | _PLACEHOLDER_ | _PLACEHOLDER_ |

Per-pair contribution: _PLACEHOLDER_ (any pair > 40%? — listed in `timeframe_<tf>/label_distribution.txt`).

---

## 3. Per-fold AUC table

### 1H
| fold | train_n | val_n | val_clean_rate | val AUC v2 | val AUC B1-B4 | Δ |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | _PLACEHOLDER_ | _PLACEHOLDER_ | _PLACEHOLDER_ | _PLACEHOLDER_ | 0.5300 | _PLACEHOLDER_ |
| 2 | _PLACEHOLDER_ | _PLACEHOLDER_ | _PLACEHOLDER_ | _PLACEHOLDER_ | 0.5430 | _PLACEHOLDER_ |
| 3 | _PLACEHOLDER_ | _PLACEHOLDER_ | _PLACEHOLDER_ | _PLACEHOLDER_ | 0.5960 | _PLACEHOLDER_ |
| 4 | _PLACEHOLDER_ | _PLACEHOLDER_ | _PLACEHOLDER_ | _PLACEHOLDER_ | 0.5260 | _PLACEHOLDER_ |
| 5 | _PLACEHOLDER_ | _PLACEHOLDER_ | _PLACEHOLDER_ | _PLACEHOLDER_ | 0.5700 | _PLACEHOLDER_ |
| 6 | _PLACEHOLDER_ | _PLACEHOLDER_ | _PLACEHOLDER_ | _PLACEHOLDER_ | 0.5140 | _PLACEHOLDER_ |
| 7 | _PLACEHOLDER_ | _PLACEHOLDER_ | _PLACEHOLDER_ | _PLACEHOLDER_ | **0.4841** | _PLACEHOLDER_ |

### 4H
| fold | train_n | val_n | val_clean_rate | val AUC v2 | val AUC B1-B4 | Δ |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | _PLACEHOLDER_ | _PLACEHOLDER_ | _PLACEHOLDER_ | _PLACEHOLDER_ | 0.4830 | _PLACEHOLDER_ |
| 2 | _PLACEHOLDER_ | _PLACEHOLDER_ | _PLACEHOLDER_ | _PLACEHOLDER_ | 0.4740 | _PLACEHOLDER_ |
| 3 | _PLACEHOLDER_ | _PLACEHOLDER_ | _PLACEHOLDER_ | _PLACEHOLDER_ | 0.5290 | _PLACEHOLDER_ |
| 4 | _PLACEHOLDER_ | _PLACEHOLDER_ | _PLACEHOLDER_ | _PLACEHOLDER_ | 0.5350 | _PLACEHOLDER_ |
| 5 | _PLACEHOLDER_ | _PLACEHOLDER_ | _PLACEHOLDER_ | _PLACEHOLDER_ | 0.5400 | _PLACEHOLDER_ |
| 6 | _PLACEHOLDER_ | _PLACEHOLDER_ | _PLACEHOLDER_ | _PLACEHOLDER_ | **0.4733** | _PLACEHOLDER_ |
| 7 | _PLACEHOLDER_ | _PLACEHOLDER_ | _PLACEHOLDER_ | _PLACEHOLDER_ | 0.4930 | _PLACEHOLDER_ |

**Concentration check:** _PLACEHOLDER_ — did improvement concentrate in
1-2 folds (mean inflated, worst unchanged), or did the worst fold also
improve?

---

## 4. Cross-TF feature contribution

For each timeframe, top-15 features by full-data RF importance.

### 1H
_PLACEHOLDER_ — table of top-15 features with importance, flagging new
(v2) vs reused (B1-B4) provenance.

### 4H
_PLACEHOLDER_

Summary metrics:
- Number of new cross-TF features in top-15: _PLACEHOLDER_
- Highest-ranked new feature: _PLACEHOLDER_ (rank _, importance _) — its
  univariate AUC: _PLACEHOLDER_
- Any new feature in top-3? _PLACEHOLDER_   Top-5? _PLACEHOLDER_

---

## 5. Feature stability across folds

Spearman ρ of feature importance ordering between consecutive folds
(F1↔F2, F2↔F3, ..., F6↔F7) plus F1↔F7. Comparable to postmortem D3,
which showed B1-B4 ρ in 0.96-0.997 (top-15) at 1H/4H.

| TF | F1↔F2 | F2↔F3 | F3↔F4 | F4↔F5 | F5↔F6 | F6↔F7 | F1↔F7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1H | _PLACEHOLDER_ | _PLACEHOLDER_ | _PLACEHOLDER_ | _PLACEHOLDER_ | _PLACEHOLDER_ | _PLACEHOLDER_ | _PLACEHOLDER_ |
| 4H | _PLACEHOLDER_ | _PLACEHOLDER_ | _PLACEHOLDER_ | _PLACEHOLDER_ | _PLACEHOLDER_ | _PLACEHOLDER_ | _PLACEHOLDER_ |

Interpretation: lower ρ than B1-B4 → new features may be overfitting
fold-specific noise. Higher or comparable ρ → new features stably
contribute. _PLACEHOLDER_ overall.

---

## 6. Boruta results

**Not run.** Boruta is not installed in this environment (`pip install
boruta` was outside the dispatch's compute envelope and explicitly
allowed to be skipped). Documented skip per dispatch §"Boruta validation
(optional, time-permitting)".

If a follow-up dispatch wants Boruta, run after installing the package
on the existing labels_relaxed.csv + features_expanded.csv (deterministic
inputs — no rerun needed).

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

**Outcome:** _PLACEHOLDER_ — _PLACEHOLDER_.

---

## 8. No-lookahead verification

Per dispatch: 10 bars per timeframe, with stored cross-TF feature values
manually reproduced from raw OHLC restricted to bars ≤ t at each source
TF. Cross-TF lag rules:

- D1 anchor: largest D1 bar with `d1_date < t.date()` (strict prior
  calendar day; `searchsorted side="left" - 1`).
- H4 anchor: largest H4 bar with `h4_time + 4h ≤ t` (most recent CLOSED
  H4 bar; `searchsorted side="right" - 1` against `t - 4h`).
- H1 anchor: largest H1 bar with `h1_time + 1h ≤ t` (only used at 4H
  signal TF).
- W anchor: largest W bar with `w_time + 7d ≤ t`.

Files: `timeframe_<tf>/lookahead_spot_check.txt`.

| TF | bars checked | stored-vs-recomputed matches | comments |
|---|---:|---:|---|
| 1H | 10 | _PLACEHOLDER_ | _PLACEHOLDER_ |
| 4H | 10 | _PLACEHOLDER_ | _PLACEHOLDER_ |

**Status:** _PLACEHOLDER_.

---

## 9. Anti-evidence and uncertainty

Open questions and potential confounders not ruled out by the present
dispatch:

1. _Concentration of improvement._ _PLACEHOLDER_ — did the improvement
   land in 1-2 folds (mean inflated, worst unchanged), or did the worst
   fold itself improve? If worst-fold stayed flat, the headline gate
   verdict is more important than the mean.
2. _Univariate vs multivariate._ Are the new features predictive at a
   single-feature univariate level (B3), or only multivariate? Multi-
   variate-only signals are less robust. _PLACEHOLDER_.
3. _Pair concentration._ Did improvement come from a specific pair
   subset? _PLACEHOLDER_ — refer to `timeframe_<tf>/b4_per_pair_per_fold.csv`.
4. _Label-shift_. Is the relaxed label measuring something materially
   different from the original B1-B4 label (e.g., majority of new
   positives are weak +0.5R trades)? The graded score_relaxed
   distribution lets readers judge: high mass at score=2 (only mono+mae
   met without any MFE condition) cannot happen — the binary label
   requires at least one MFE condition. So score=2 positives have
   mono+mae+(mfe≥0.5 only). Compare score=2 to score=3 (mfe≥0.5 + mfe≥1
   both, i.e., real 1R+ winners).
5. _Sample of new features._ The 15 new features were chosen by the
   dispatcher for higher-TF state. If a different 15 (e.g., realized
   vol of vol, inter-market) had been chosen, results could differ.
   This is an upper-bound test on this specific feature batch, not
   "all possible features".

---

## 10. Recommendation

Conditional on the gate verdict:

- **Strong gate cleared** → B5-B6 signal nomination using v2 feature set
  (highest-importance v2 features become candidate signal-class
  primitives). Chat-track decides which v2 features warrant standalone
  L-arc registry entries.
- **Marginal gate cleared** → B5-B6 with realistic expectations; expect
  modest Step 4 AUC. Run alongside the regime-conditional dispatch's
  best vol-bin for a head-to-head signal nomination.
- **Gate not cleared** → combined with the regime-conditional dispatch
  result, formal Lω closure. If both dispatches null, structural pivot
  to Tier A structural-event signals (failed-breakout reversal,
  liquidity sweep + reclaim, compression-then-expansion). These get
  nominated as L-arc registry entries, not as Lω discovery.

**Outcome:** _PLACEHOLDER_.

---

## Validation checklist

- [ ] 1H + 4H processed; D1 skipped per scope
- [ ] Relaxed label base rate documented per timeframe; flagged if
  outside 30-50% expected band; blocker if outside 15-70% extreme
- [ ] All ~45 features computed (30 reused + 15 new cross-TF + 3
  derived agreement = ~48 in practice per TF); document any features
  deferred (e.g., w_ema_20_slope if W data unavailable — not the case
  here, all 28 pairs have W data 2010+).
- [ ] No-lookahead spot-check 10 bars per timeframe, including
  cross-TF lag boundaries (D1/H4/H1/W); BLOCKER if any mismatch
- [ ] 7-fold time-series CV with identical fold dates to B1-B4
- [ ] Determinism PASS on labels and AUC tables (byte-identical re-run)
- [ ] LOMEGA_V2_CROSSTF_SUMMARY.md contains all 10 required sections
- [ ] Comparison vs B1-B4 baseline explicit per fold

---

## Reproducing

```
py scripts/lomega/lomega_v2_crosstf.py --tf all      # both TFs
py scripts/lomega/lomega_v2_crosstf.py --tf 1h       # 1H only
py scripts/lomega/lomega_v2_crosstf.py --tf 4h       # 4H only
```

Outputs land under `results/lomega/v2_crosstf/timeframe_<tf>/`. Heavy
`labels_relaxed.csv` and `features_expanded.csv` are gitignored under
the existing `results/lomega/b1_b4_discovery/.gitignore` pattern (extended
to v2_crosstf via a new .gitignore in the v2_crosstf directory). Small
analysis outputs (`b3_univariate_expanded.csv`, `b4_multivariate_expanded.csv`,
`b4_feature_importances_full.csv`, `b4_per_pair_per_fold.csv`,
`b4_summary.txt`, `feature_importance_drift.csv`,
`feature_importance_spearman.csv`, `label_distribution.txt`,
`lookahead_spot_check.txt`, and this summary) ARE committed.
