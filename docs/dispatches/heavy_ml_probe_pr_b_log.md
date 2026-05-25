# heavy_ml_probe PR-B — Log Doc

> **Branch:** `infra/heavy_ml_probe_pr_b` → `infra/heavy_ml_probe_build` (cut from PR-A locally; chat's "PR-A merged" signal honoured at the head level).
> **Dispatch:** chat prompt 2026-05-25 confirming PR-B scope + flag dispositions for PR-A.
> **Intent doc:** [`docs/dispatches/heavy_ml_probe_build_intent.md`](heavy_ml_probe_build_intent.md).
> **PR-A log:** [`docs/dispatches/heavy_ml_probe_pr_a_log.md`](heavy_ml_probe_pr_a_log.md).
> **Sub-protocol spec:** [`docs/sub_protocols/heavy_ml_probe.md`](../sub_protocols/heavy_ml_probe.md) v1.0.
> **Authoring CC session:** worktree `elegant-ride-ad9a57`.

---

## §1 Scope landed (verbatim against dispatch §6)

| Path | Status | Purpose |
|---|---|---|
| `core/heavy_ml_probe/automl.py` | new | FLAML wrapper: 11-fold TimeSeriesSplit + per-fold AutoML.fit() + modelcount accounting + permutation importance + NaN-safe AUC + holdout-guard |
| `core/heavy_ml_probe/pipeline.py` | extended | `run_pipeline` now orchestrates the AutoML stage; emits 3 new artefacts + extended manifest; PR-A skip-paths preserved |
| `core/heavy_ml_probe/metrics.py` | unchanged | `auc_roc` already complete from PR-A; `concordance` / `integrated_brier_score` stay stubs for PR-D |
| `scripts/heavy_ml_probe/run_probe.py` | tweaked | stdout summary surfaces AutoML stats; `HoldoutGuardViolation` + `AllFeaturesRejected` mapped to exit 1 |
| `docs/sub_protocols/heavy_ml_probe.md` | tweaked | one-line CLI exit-code mapping appended per PR-A flag-1 disposition |
| `tests/heavy_ml_probe/test_automl.py` | new | 19 tests (end-to-end, modelcount cap, determinism, NaN-AUC propagation, holdout-guard, lineage-rejects-all) |

Artefact outputs landing per dispatch §6:

- `step_4/heavy_ml/automl_leaderboard.csv` — per-fold FLAML model rankings + per-config AUC + modelcount
- `step_4/heavy_ml/automl_feature_importance.csv` — permutation importance (mean + std + n_folds_present) per feature
- `step_4/heavy_ml/compute_budget_used.md` — per-fold trial counts vs cap + aggregate AUC
- `step_4/heavy_ml/manifest.json` — sha256 per artefact + `automl` extras block

PR-B does NOT modify `core/architectures/**`, `L_PROTOCOL.md`, vanilla Step 1-6 mechanics, or any arc folders.

---

## §2 §3a — FLAML `max_iter` empirical verification (mandatory)

**Finding: `flaml.AutoML.modelcount` equals `max_iter` exactly when FLAML runs to the cap. Ratio 1.0. Locked, proceed.**

### §2.1 Probe methodology

Before any PR-B implementation, FLAML 2.6.0 was installed and probed via three runs against a deterministic synthetic binary-classification pool (`seed=42`, 500 trades, 3 informative features). Probe varied `max_iter` ∈ {5, 10, 30, 100} both with the full default estimator list and with a single-estimator (`lgbm`-only) variant.

### §2.2 Results

```
max_iter=  5  modelcount=  5  best_iteration=  2
max_iter= 10  modelcount= 10  best_iteration=  9
max_iter= 30  modelcount= 30  best_iteration= 22
```

Per dispatch §3a's three possible outcomes:

| Outcome | Disposition |
|---|---|
| Exactly N | **Confirmed.** `modelcount == max_iter` in every probe. Ratio 1.0. Locked. |
| Less than N (early-stop) | Did not materialise on noisy targets at `max_iter ≤ 30`. |
| Greater than N (pre-warm trials uncounted) | Did not materialise. No HALT trigger. |

`automl.best_iteration` is the 0-indexed position of the best trial (always ≤ `modelcount - 1`); it is NOT the trial counter. The trial counter is `automl.modelcount`. This module records `modelcount` per fold into the leaderboard CSV + the `compute_budget_used.md` per-fold table.

### §2.3 Bound enforcement

`core/heavy_ml_probe/automl.py:_run_one_fold` records `modelcount` after each FLAML fit. If `modelcount > max_iter` ever drifts upward in a future FLAML release, a `UserWarning` fires citing this log doc's §3a empirical claim. Test `test_modelcount_does_not_exceed_max_iter` enforces the inequality; `test_modelcount_matches_max_iter_on_learnable_target` asserts equality on the test fixture's noisy target.

### §2.4 Estimator-set drift note

FLAML 2.6.0's default classification estimator list on this install:

```
['lgbm', 'rf', 'xgboost', 'extra_tree', 'xgb_limitdepth', 'sgd', 'catboost', 'lrl1']
```

This differs slightly from the dispatch §1 spec (which listed `lrl2` instead of `xgb_limitdepth` / `sgd`). Per dispatch §1 "If defaults differ, log the actual set and proceed" — captured in `core/heavy_ml_probe/automl.py::DOCUMENTED_FLAML_2_6_0_ESTIMATORS` plus per-fold capture into `FoldResult.estimator_list` (manifest-bound for cross-env audit). Test `test_estimator_list_recorded_per_fold` asserts non-empty + the always-present `lgbm` / `rf` baseline so a benign FLAML minor-version drift doesn't break the test suite.

---

## §3 §3b — Single-class fold AUC handling (mandatory)

**Finding: NaN propagates correctly; aggregate uses `nanmean`; `n_folds_valid` reported alongside.**

### §3.1 Two-layer NaN propagation

Two layers can produce a single-class fold:

1. **Single-class training slice** — TimeSeriesSplit early folds against a temporally-stratified target can yield an all-zero (or all-one) train fold. FLAML's internal CV cannot fit on a single-class input. Per `core/heavy_ml_probe/automl.py:_run_one_fold`, this is detected up-front, AutoML is skipped, and `FoldResult.auc_val` / `auc_train` come back NaN. A `UserWarning` records the skip; the fold's `best_estimator` is recorded as `"<skipped:single_class_train>"` for grep-able audit.

2. **Single-class validation slice** — FLAML fits cleanly but `roc_auc_score` returns NaN (modern sklearn behaviour — verified at PR-B time). `core/heavy_ml_probe/metrics.py:auc_roc` short-circuits to NaN on `len(set(y_true)) < 2` so the per-fold AUC is consistently NaN regardless of which sklearn path is taken.

### §3.2 Aggregate

```python
auc_mean = float(np.nanmean(auc_vals)) if n_valid > 0 else float("nan")
auc_std  = float(np.nanstd(auc_vals, ddof=0)) if n_valid > 0 else float("nan")
n_valid  = int(np.isfinite(auc_vals).sum())
```

`AutoMLResult.n_folds_valid` is reported on the in-memory object, in the manifest's `automl` extras block, and in `compute_budget_used.md` (e.g. "across 7 valid folds; total 11"). Downstream readers know whether the aggregate spans every fold or fewer.

### §3.3 Tests

- `test_nan_auc_propagates_on_single_class_training` — builds a pool with the first 80% all-y=0 so early TimeSeriesSplit folds train on a single class; asserts ≥1 fold NaNs out and `n_folds_valid` reflects the live count.
- `test_nan_auc_aggregate_uses_nanmean` — every fold NaN → `auc_mean` is NaN AND `n_folds_valid == 0`. Boundary case covered.

---

## §4 Verification

### §4.1 Test suite

```
py -3 -m pytest tests/heavy_ml_probe -q -W ignore::UserWarning
..................................................                       [100%]
50 passed in 16.77s
```

50 tests: 31 carried over from PR-A (unchanged behaviour preserved) + 19 new in `test_automl.py`. Wall-clock under 60s per dispatch §4.

Coverage breakdown for the 19 new tests:

| Group | Count | Coverage |
|---|---|---|
| Core mechanics | 3 | end-to-end smoke; modelcount ≤ max_iter; modelcount == max_iter on noisy target |
| NaN propagation | 2 | single-class training; aggregate nanmean semantics |
| Guards | 5 | holdout-guard violation (interior + boundary); all-features-rejected; missing required column; non-binary target |
| Determinism | 1 | two-run AutoMLResult equality (auc_mean, auc_std, modelcount, leaderboard, importance) |
| FLAML capture | 1 | estimator_list recorded per fold |
| Pipeline integration | 7 | full artefact set written; manifest sha256 matches on-disk; two-run determinism (4 artefacts + stable-payload manifest); holdout-guard surfaces with partial manifest; skip reasons (missing_entry_time, missing_y, no_clean_features); AUTOML_REQUIRED_COLUMNS contract |

### §4.2 Sibling regression

```
py -3 -m pytest tests/discovery -q
..........................................                               [100%]
42 passed in 43.71s
```

42/42 — sibling sub-protocol untouched.

### §4.3 Lint

```
py -3 -m ruff check core/heavy_ml_probe scripts/heavy_ml_probe tests/heavy_ml_probe
All checks passed!
```

One initial E402 issue surfaced in `test_automl.py` because `pytest.importorskip("flaml")` precedes the module-level imports; resolved with explicit `# noqa: E402` annotations + a comment explaining the gate.

### §4.4 CLI smoke

End-to-end CLI invocation against a 200-trade synthetic pool + the locked `configs/heavy_ml_probe/default.yaml` (overridden to `n_folds=3 / max_iter_per_fold=5` for the smoke):

```
[heavy_ml_probe] pipeline run complete.
  arc            : smoke
  ...
  lineage gate   : accepted=0 / rejected=5 / input=5
  AutoML         : status=no_clean_features (skipped)
exit_code: 0
manifest artefacts: ['stub_summary']
automl skip_reason: no_clean_features
```

This particular smoke skipped AutoML cleanly because the synthetic pool's feature names (`a`, `b`, `trade_id`, `entry_time`, `y`) aren't in the production feature registry, so the real lineage gate rejected every column. That's the right behaviour — the gate fired exactly as designed, AutoML was correctly bypassed, and the manifest records the reason. Exit code 0 because the pipeline ran to completion; the skip is informational, not a failure.

### §4.5 Wall-clock probe at test scale

```
test-scale: n=400, n_folds=5, max_iter=10 -> 1.55s wall, total_modelcount=50, auc_mean=0.9884, valid_folds=5/5
  per-fold fit times (s): [0.20, 0.28, 0.19, 0.24, 0.24]
  per-fold modelcount  : [10, 10, 10, 10, 10]
```

50 model trials in 1.55s on this hardware → ~32ms per trial. Linear extrapolation to production scale (n_folds=11, max_iter_per_fold=1000, total trials = 11,000) suggests **~6 minutes wall-clock per cluster** under these assumptions. That's well inside the dispatch §4 expectation of "2-6 hours per cluster" — likely orders of magnitude faster, but the extrapolation is on a 4-feature toy problem; production has 20-30 features and FLAML's per-trial cost grows with column count. A production-scale run will surface the true number in PR-E's first end-to-end timing.

### §4.6 Determinism — two-run byte equality

`test_pipeline_automl_two_run_determinism` asserts byte equality for all four artefacts (`stub_summary.md`, `automl_leaderboard.csv`, `automl_feature_importance.csv`, `compute_budget_used.md`) plus the stable-payload manifest sha256 (timestamp excluded). Initially failed because `fit_wall_seconds` leaked into the leaderboard CSV + the markdown — wall-clock varies across runs. Fix: kept `fit_wall_seconds` on the in-memory `AutoMLResult` only; removed it from the on-disk artefacts and the manifest extras. Two-run determinism now holds. See `core/heavy_ml_probe/automl.py:leaderboard_to_dataframe` + `compute_budget_markdown` docstrings.

---

## §5 Deviations from intent doc / dispatch

### §5.1 sklearn / FLAML compatibility shim for permutation_importance (real bug found and fixed)

Initial implementation passed FLAML's outer `AutoML` object to `sklearn.inspection.permutation_importance`. sklearn 1.5+'s estimator-type check refuses the wrapper:

```
ValueError: AutoML should either be a classifier to be used with response_method=predict_proba
or the response_method should be 'predict'. Got a regressor with response_method=predict_proba instead.
```

Fix: extract the inner fitted estimator (`automl.model.estimator`) and pass THAT to `permutation_importance`. The inner estimator is a proper sklearn-compatible classifier (`LGBMClassifier`, `RandomForestClassifier`, etc.). Predictions through the inner estimator are byte-identical to the outer wrapper for standard pipelines per FLAML's docs. Documented inline in `_run_one_fold`.

Flagged here so future PR reviewers aren't surprised by the `getattr(automl, "model", None).estimator` extraction pattern.

### §5.2 Wall-clock dropped from on-disk artefacts (not flagged in dispatch)

Per §4.6 above. The dispatch §4 mentions wall-clock measurement but doesn't specify whether it lands in artefacts; for determinism, it can't. Documented in `compute_budget_markdown` docstring + the artefact body's "Wall-clock measurements live on the in-memory AutoMLResult" line. The in-memory `AutoMLResult.total_fit_wall_seconds` is still computed and surfaced through the CLI's stdout for operator visibility.

### §5.3 Pipeline auto-skip when pool lacks `entry_time` / `y`

Not explicitly in the dispatch but follows from the design — PR-A's smoke tests use 50-row pools without `entry_time` / `y` columns. Rather than break those tests, the pipeline detects missing required columns and skips AutoML with `skip_reason ∈ {missing_entry_time, missing_y}`. The skip reason lands in the manifest's `automl` extras block. Production arcs always have these columns; the skip path is purely a backward-compatibility shim.

`AUTOML_REQUIRED_COLUMNS = ("entry_time", "y")` is exported from `pipeline.py` so consumers can introspect.

### §5.4 Spec touch — `docs/sub_protocols/heavy_ml_probe.md`

Per PR-A flag-1 disposition, added a 5-line "Invocation exit codes" block under "When invoked." Documents existing behaviour (0/1/2 mapping) without protocol surface change.

### §5.5 YAML schema unchanged

Per PR-A flag-2 disposition: no `schema_version` bump in PR-B. The AutoML config keys (`max_iter_per_fold`, `n_folds`, `metric`, etc.) were already locked in PR-A under the `automl` namespace; PR-B just started reading them. No breaking change.

---

## §6 Flags for chat

### §6.1 Non-blocking

- **`automl.model.estimator` extraction** (PR-B §5.1 above). FLAML's API surface for "the underlying sklearn classifier" is documented but not formally typed; a future FLAML major-version bump could rename this attribute. Test `test_estimator_list_recorded_per_fold` catches at least the "FLAML's estimator list became empty" regression; a deeper change to the `model.estimator` attribute would surface as test failure across all `permutation_importance`-using tests (most of `test_automl.py`). Worth pinning FLAML to a known-good major version when the production environment is locked.

- **`xgb_limitdepth` + `sgd` in FLAML's default estimator list** (§2.4). The dispatch spec said `lrl2`; FLAML 2.6.0 actually defaults to a different mix. Captured in `DOCUMENTED_FLAML_2_6_0_ESTIMATORS` constant + per-fold capture. If chat prefers an explicit `estimator_list` override (e.g. force-include `lrl2` for parity with the spec doc), it's a one-line YAML change in `configs/heavy_ml_probe/default.yaml::automl.estimator_list`. PR-B reads from FLAML's default; an explicit override path is wired but commented out.

- **scikit-survival install failure on Python 3.14** (carried over from PR-A risk note + surfaced here for real). `pip install scikit-survival lifelines` on the build machine fails because `ecos` (transitive dep) has no cp314 wheel and the source build requires MSVC. **FLAML + xgboost + catboost installed cleanly** — sufficient for PR-B. Survival deps need to install for PR-D; will be resolved at PR-D time either by (a) MSVC build tools install on the dev machine, (b) pinned older Python (3.11 / 3.12 where the wheels exist), or (c) conda-forge variants. Test `test_automl.py` import-guards FLAML so the module is skipped (not failed) on environments without FLAML, future-proofing for CI matrix.

- **Wall-clock extrapolation** (§4.5) suggests production cost is order-of-minutes per cluster, not the dispatch's order-of-hours estimate. Could be very wrong — depends on feature count + FLAML's per-trial cost growth — but worth knowing the test-scale baseline before PR-E's first real production run.

### §6.2 Surfaced for PR-C intent

- **Meta-labeling target reuses AutoML path**: the spec says "binary target = reach +1R MFE before SL." PR-C will compute this target from the pool's MFE / MAE columns (consumed in the existing Step 1 pool schema, not added) and reuse `automl.run_automl` with `target_col` overridden. The shape contract is already in place — `target_col` is a parameter of `run_automl`. PR-C will add the target-construction logic + a thin wrapper, not a new AutoML loop.

- **Threshold sweep storage**: meta-labeling produces a per-fold threshold sweep (precision/recall at `{0.30, 0.40, 0.50, 0.60, 0.70}` per the locked YAML). New artefact `meta_label_results.csv` per dispatch §7. PR-C intent should specify whether the threshold sweep replaces or augments the leaderboard CSV — current `automl_leaderboard.csv` has no threshold column.

### §6.3 No HALT

No HALT conditions encountered. §3a ratio == 1.0 (well under the 1.5× HALT threshold). §3b NaN propagation works as designed. All non-negotiables enforced. WORKFLOW §6 unused.

---

## §7 Branch state at PR-time

- Build branch (`infra/heavy_ml_probe_build`) on origin is still at `7c238e8` (main's tip at PR-A time). PR-A's GitHub merge has not happened yet at the time PR-B is opened — chat's "PR-A merged" signal was interpreted at the head level (PR-A's code is conceptually in build; PR-B is layered on top locally).
- PR-B branch (`infra/heavy_ml_probe_pr_b`) was cut from `infra/heavy_ml_probe_pr_a` locally so the head of PR-B includes PR-A's commits + PR-B's commit. When chat actually merges PR-A → build on GitHub, the PR-B diff against `infra/heavy_ml_probe_build` will recompute and show only PR-B's changes. If chat prefers a different sequencing (e.g. rebase PR-B onto a freshly-merged build), happy to redo.

---

## §8 What lands next (per dispatch §8)

After PR-B merges into `infra/heavy_ml_probe_build`:

- **PR-C (Meta-labeling)** — `core/heavy_ml_probe/meta_labeling.py` reusing the AutoML path with `target_col="reach_1r"`; new `meta_label_results.csv` artefact. ≤ 1 day.
- **PR-D (Survival)** — Cox PH + RSF + Q5b-resolved adapter wrapping survival predictions into A4's `predict_admit` shape. Requires sksurv install (see §6.1 flag). ≤ 1 day.
- **PR-E (Integration + Step 5 hook)** — gate PR; end-to-end determinism on synthetic data + `build_a2_config_from_heavy_ml` / `build_a6_config_from_heavy_ml` wrappers. ≤ 1 day.
- **PR-F (Docs + polish)** — README / inline doc cleanup. ≤ 0.5 day.

Per chat directive: I will not start PR-C work during the PR-B review window.

---

End of PR-B log.
