# Sub-Protocol — heavy_ml_probe

> **Status:** locked v1.0
> **Predecessor:** L_PROTOCOL v3.0 overseer (this sub-protocol overrides Step 4 and adds to Step 5)
> **Purpose:** apply heavy machine learning techniques to an arc whose vanilla Step 4 didn't extract a usable filter, but where Step 3 evidence suggests genuine cohort edge exists.

---

## When invoked

An arc's `ARC_OPEN.md` declares `sub_protocol: heavy_ml_probe`. Typical use:

- A Phase 1 arc closes FAIL but Step 3 shows clean capturable cohort (V-shape, Stepwise, etc.)
- Vanilla Step 4 (RF + LGBM + Logistic with fixed defaults) returned AUC at chance
- The arc is worth one more pass with heavier ML before being shelved

Not invoked by default. Compute-heavy. Run on selected arcs only.

**Invocation (`scripts/heavy_ml_probe/run_probe.py`) exit codes** — standard Unix mapping; documented here for CI consumers (PR-A flag-1 disposition):

- `0` — success; artefact set written
- `1` — runtime failure (e.g. pool not found, holdout-guard violated, lineage gate rejected every feature)
- `2` — argparse misuse (missing required flag, etc.; emitted by Python's argparse on its own)

---

## What overrides the overseer

### Step 4 — replaced

Vanilla Step 4 (three classifiers with fixed defaults, 5-fold TimeSeriesSplit AUC) is replaced with:

1. **AutoML inside CV.** Per fold of an 11-fold TimeSeriesSplit on the IS window (2010-2020):
   - Run AutoML (autogluon-tabular or FLAML)
   - Compute budget: **1000 model evaluations per fold, hard cap.** Sum across 11 folds = 11,000 evaluations max per cluster
   - Search space: RF, LGBM, XGBoost, CatBoost, ExtraTrees, Logistic Regression
   - Per-model hyperparameter search included in the AutoML budget
   - Feature selection happens INSIDE each fold's training window (not as preprocessing)
   - Ensembling and stacking decided by AutoML automatically

2. **Meta-labeling target.** Binary target = "did this trade reach +1R MFE before hitting SL?" instead of cluster membership. Aligns with the actual deployment question.

3. **Survival model variant for Pipeline D.** When arc declares `pipeline_d_exits` as a target architecture, also train a Cox Proportional Hazards model on time-to-+1R-MFE censored at SL/time-exit (per chat resolution Q1). Predicted hazard at each post-entry bar feeds Step 5 Pipeline D exit policy via the adapter pattern (Q5b).

   **Library:** Cox PH via `statsmodels.duration.hazard_regression.PHReg` (PR-D). The original dispatch named `lifelines`; replaced because lifelines is blocked on Python 3.14 (its transitive dep `ecos` has no cp314 wheel). statsmodels has a clean cp314 wheel and PHReg covers the Cox PH path with the same modelling semantics.

   **Random Survival Forest — DEFERRED.** scikit-survival (RSF) is also blocked on Py 3.14 by the same `ecos` cp314 gap. Per PR-B flag-1 disposition, PR-D ships Cox PH only. RSF can be added later when the wheel ships; the spec section reserves the slot.

4. **Per-feature causal lineage.** Every feature consumed by AutoML must carry a causal lineage tag from Step 1. Features without a clean tag are excluded from AutoML training set. No exceptions — Arc 9 lookahead lesson.

### Step 5 — augmented

The standard six architectures (A1-A6) still run. Additionally:

- **A2 (Classifier filter)** uses the AutoML-best classifier (not vanilla RF default)
- **A4 (Pipeline D)** uses the survival-model-predicted exit timing
- **A6 (Meta-labeling)** uses the AutoML-best meta-labeler at thresholds {0.0, 0.5, 1.0}x risk sizing

Step 5 WFO mechanics, gates, and verdict logic are unchanged from overseer §3.

### Step 6 — applies as standard

Causal audit runs identically to overseer §2 Step 6. Heavy ML doesn't bypass it; if anything, the audit is more important because AutoML's feature interactions are harder to inspect manually.

---

## Output artefacts

`step_4/heavy_ml/`:

- `stub_summary.md` — human-readable per-stage status (skip reasons, AUC / concordance summaries, lineage gate counts). Timestamp-free → reproducible across two runs at byte-identity.
- `compute_budget_used.md` — cross-stage audit. AutoML modelcount vs `max_iter` cap, meta-label AUC / threshold-sweep count, survival fold count / concordance / total events. Single source of truth for "did we honour the 1000-evals-per-fold cap."
- `automl_leaderboard.csv` — per-(fold, estimator) rows from FLAML's `best_loss_per_estimator`, plus the fold's `auc_val` / `auc_train` / `modelcount` / `is_overall_best`.
- `automl_feature_importance.csv` — permutation importance per feature, aggregated across folds (`importance_mean`, `importance_std`, `n_folds_present`).
- `meta_label_results.csv` — threshold sweep on out-of-fold reach-1R-before-SL predictions. Columns: `threshold`, `precision`, `recall`, `f1`, `n_trades_kept`, `n_trades_dropped`, `mean_r_kept_set`, `mean_r_dropped_set`, `edge_lift_r`, plus aggregate diagnostic columns (`target_positive_rate`, `aggregate_oof_auc_mean`, `n_folds_valid`).
- `survival_model_results.csv` — per-(fold, feature) Cox PH coefficient, `p_value`, `std_err`, fold concordance, `n_train`, `n_train_event`, `convergence_warning` flag.
- `manifest.json` — sha256 over every artefact above + per-stage extras blocks (lineage gate, AutoML, meta-label, survival). Top-level `created_at`; per-stage details under their respective keys.
- `classifiers/meta_label/manifest.json` + `classifiers/meta_label/fold_NN.joblib` — PR-C per-fold meta-label classifier pickles + sidecar manifest with sha256 + classifier_type + concordance + joblib version.
- `classifiers/survival/manifest.json` + `classifiers/survival/fold_NN.joblib` — PR-D per-fold Cox PH model pickles + sidecar with sha256 + concordance + n_train_event + convergence_warning. Each pickle is a dict `{results, baseline_hazard, used_features, coefficients}` so PR-E's A4 adapter has everything to compute survival probabilities without refit.

`step_5/heavy_ml_augmented/`:

- `heavy_ml_manifest.json` — adapter consumer manifest. Schema below.

**Scope clarification (PR-E):** `heavy_ml_probe` does NOT produce `heavy_ml_augmented_architectures.csv` directly. PR-E ships **adapters** that translate heavy-ML-trained components into the existing `core/architectures/{A2,A4,A6}*.py` config shapes (per Q5a + Q5b), and emits `step_5/heavy_ml_augmented/heavy_ml_manifest.json` describing what's adapter-buildable. When the overseer's Step 5 loop runs against an arc whose `ARC_OPEN.md` declares `sub_protocol: heavy_ml_probe`, the loop reads this manifest, builds A2/A4/A6 via the adapters, runs them, and writes the augmented-architectures CSV alongside its vanilla Step 5 outputs. This keeps `heavy_ml_probe` narrow-scope (no direct A2/A4/A6 invocation).

### Step 5 adapter manifest schema (locked at `STEP5_MANIFEST_SCHEMA_VERSION = "1.0"`)

```json
{
  "schema_version": "1.0",
  "sub_protocol": "heavy_ml_probe",
  "arc_name": "<arc>",
  "cluster_id": 0,
  "step4_dir": "step_4/heavy_ml",
  "used_features": ["..."],
  "stages": {
    "automl": {
      "status": "ok" | "skipped",
      "skip_reason": "<reason>",
      "classifier_manifest_path": null
    },
    "meta_label": {
      "status": "ok" | "skipped",
      "skip_reason": "<reason>",
      "classifier_manifest_path": "../step_4/heavy_ml/classifiers/meta_label/manifest.json"
    },
    "survival": {
      "status": "ok" | "skipped",
      "skip_reason": "<reason>",
      "classifier_manifest_path": "../step_4/heavy_ml/classifiers/survival/manifest.json"
    }
  },
  "adapters": {
    "a2_buildable": true,
    "a4_buildable": true,
    "a6_buildable": true
  },
  "schema_doc": "core.heavy_ml_probe.adapters.STEP5_MANIFEST_SCHEMA_VERSION"
}
```

Adapter consumers (`build_a{2,4,6}_from_heavy_ml`) refuse a manifest whose `schema_version` doesn't match the current pin. Per-stage `classifier_manifest_path` is recorded RELATIVE to the Step 5 manifest's location (Step 4 and Step 5 dirs are siblings under the arc root; the path includes `..` segments). `adapters.{a2,a4,a6}_buildable` flags reflect whether the upstream stage's status is `ok` — adapters for skipped stages raise `StageUnavailableError` with the skip reason.

---

## Invocation

```
python -m scripts.heavy_ml_probe.run_probe \
    --arc <arc_name> \
    --pool <path/to/step_1/pool.parquet> \
    --cluster-id <int> \
    [--config configs/heavy_ml_probe/default.yaml] \
    [--output-root results/<arc_name>]
```

A single invocation runs AutoML (PR-B) + meta-labeling (PR-C) + Cox PH survival (PR-D) end-to-end. Stages are independent: each auto-skips if the pool lacks its required schema columns; skip reasons land in the manifest. Survival uses `statsmodels.duration.hazard_regression.PHReg` rather than the originally-spec'd `lifelines` (Python 3.14 cp314 wheel gap); RSF is deferred for the same reason.

**Exit codes** (PR-E):

- `0` — every stage succeeded OR every stage cleanly skipped with documented reason
- `1` — runtime failure (pool not found, holdout-guard violated, lineage gate rejected every feature, uncaught exception)
- `2` — argparse misuse (missing required flag — argparse's default)
- `3` — partial success: at least one stage succeeded AND at least one stage was skipped. Adapters for `ok` stages will build cleanly; adapters for skipped stages raise `StageUnavailableError` with the skip reason

**Step 5 adapter manifest** (PR-E) — `step_5/heavy_ml_augmented/heavy_ml_manifest.json` carries per-stage status + paths + buildability flags. Schema locked at `STEP5_MANIFEST_SCHEMA_VERSION = "1.0"`; adapters refuse to consume an unknown schema. The downstream overseer Step 5 loop (and any chat-side analysis script) constructs A2/A4/A6 via:

```python
from core.heavy_ml_probe.adapters import (
    build_a2_from_heavy_ml,
    build_a4_from_heavy_ml,
    build_a6_from_heavy_ml,
    FoldSelectionStrategy,
)

a2_cfg = build_a2_from_heavy_ml("results/<arc>/step_5/heavy_ml_augmented/heavy_ml_manifest.json")
a4_cfg, cox_adapter = build_a4_from_heavy_ml(manifest_path, k_horizon=5, exit_threshold=0.4)
a6_cfg = build_a6_from_heavy_ml(manifest_path, lower_threshold=0.3, upper_threshold=0.7)
```

Fold-selection strategy (per dispatch §3): `LAST_FOLD` (default — fold N classifier / Cox PH coefficients), `ENSEMBLE_MEAN` (average across all valid folds for sensitivity analysis), or `FULL_REFIT` (reserved — raises `NotImplementedError` in PR-E).

---

## A4 hazard math (the `CoxPHAdapter` contract)

A4 (Pipeline D — differentiated exits) consumes Cox PH via the adapter pattern (Q5b option B1). The math the adapter implements per `CoxPHAdapter.hazard_predict(features_at_N, bars_survived_at_N)`:

```
H_0(N)         = baseline_cumulative_hazard at time N        (Breslow estimator step function)
H_0(N+K)       = baseline_cumulative_hazard at time N+K
relative_risk  = exp(features_at_N @ cox_coefficients)
integrated_h   = (H_0(N+K) - H_0(N)) * relative_risk
P(reach +1R
  in next K bars
  | survived N) = 1 - exp(-integrated_h)
```

- `features_at_N`: the per-trade feature dict at bar N post-entry. Keys must match the fold's `used_features` (the column order the Cox PH was fitted on).
- `bars_survived_at_N`: integer count of bars from entry to N.
- `k_horizon`: configured at adapter-build time; the dispatch default in `CoxPHAdapter` is 5 bars.
- The baseline cumulative hazard is a right-continuous step function evaluated by lookup at the unique observed event times: for `t < min(times)` we return 0; for `t ≥ max(times)` we clamp to the largest observed value; between event times we take the largest observed time ≤ t (no linear interpolation — that would invent hazard accrual between events, contra Cox PH's non-parametric semantics).
- Numerical guard: the exp argument for `relative_risk` is clamped to `[-50, 50]` to avoid overflow on extreme linear predictors.
- NaN propagation: missing or non-finite features in `features_at_N` → `hazard_predict` returns `nan`. A4's `predict_admit` treats `nan` as a no-decision.

The adapter ALSO exposes `predict_proba(X)` as a sklearn-shape shim so A4's runtime (which only knows `predict_proba`) can dispatch to the Cox PH machinery without modification. The shim reads `bars_survived` from a named column (`BARS_SURVIVED_FEATURE = "_bars_survived"`, appended to the feature row by A4's predicate at evaluation time) and delegates each row to `hazard_predict`.

Direct callers (chat-side analysis, tests, sensitivity sweeps) should prefer `hazard_predict` over `predict_proba` — same math, cleaner contract.

---

## Cox PH minimum-N discipline

Cox PH coefficients are unstable below `n_train ≈ 200`. The survival stage enforces the policy:

- **Warning, not skip.** When a fold's training slice has `n_train < MIN_N_WARN` (= 200), `run_survival` emits a `UserWarning` AND sets `convergence_warning=True` on the fold's row in `survival_model_results.csv`. The fit still proceeds — partial information is better than no information for small-cluster arcs.
- **No-events folds are skipped cleanly.** If a TimeSeriesSplit early fold has zero events in its training slice, Cox PH cannot fit. The fold is recorded with `fit_succeeded=False`, `fit_message="skipped:no_events_in_training_slice"`, `concordance=NaN`. The loop continues — aggregate concordance uses `np.nanmean` so the bad fold doesn't poison the estimate.
- **statsmodels convergence failures are trapped.** `ConvergenceWarning` and `LinAlgError` are caught at `_safe_phreg_fit`. The fold's `convergence_warning=True` is set; `fit_message` carries the exception type + message. `concordance=NaN` for that fold.
- **Aggregate concordance** uses `nanmean` + `nanstd` with `n_folds_valid` reported alongside `n_folds_total` so downstream readers can tell whether the aggregate spans every fold or fewer.

Closure-doc audit: any heavy_ml_probe arc reporting a survival stage should surface the `convergence_warning` count per fold in its closure narrative. The pattern matters — 1 fold with `n_train < 200` is normal (TimeSeriesSplit early folds always have small training slices); all 11 folds flagging is structural (cluster too small for Cox PH).

---

## RSF deferral status

Per PR-B flag-1 disposition (chat-approved): Random Survival Forest is DEFERRED from heavy_ml_probe PR-D ship-scope.

- `scikit-survival` (the canonical RSF library) is blocked on Python 3.14 because its transitive dep `ecos` has no `cp314` wheel
- `core.heavy_ml_probe.metrics.integrated_brier_score` is a stub that raises `NotImplementedError` with the deferral pointer; its signature stays on the public API surface so a future PR can reinstate without breaking imports
- The spec leaves the RSF slot reserved: when `ecos` ships a `cp314` wheel AND chat re-enables RSF, the addition is `add scikit-survival to requirements-dev.txt` + `implement RandomSurvivalForest path in core/heavy_ml_probe/survival.py` + `wire integrated_brier_score body`. Cox PH semantics in PR-D do not change

This deferral is documented at three levels: this section, `core/heavy_ml_probe/metrics.py::integrated_brier_score` error message, and `requirements-dev.txt` comment block above the survival deps.

---

## Discipline rules specific to this sub-protocol

- Compute budget is HARD. 1000 evaluations per fold, 11,000 per cluster. AutoML must respect the cap.
- Causal lineage tag is binding. A feature without a clean tag CANNOT enter training even if AutoML wants it.
- Holdout (2021-2025) is one-shot per heavy-ML-augmented architecture, same as standard arcs.
- Heavy ML does NOT lower the deployment gates. PASS-DEPLOYABLE and PASS-VIABLE thresholds from L_PROTOCOL §3 apply identically.
- If heavy ML underperforms vanilla Step 4 on the same arc — that's a finding. Record it. The arc closure documents whether heavy ML lifted the ratio or not.

---

## Expected compute cost

Per arc:
- AutoML training: ~2-6 hours per cluster on standard hardware
- Typical arc has 2-4 capturable clusters → ~8-24 hours total Step 4
- Plus Step 5 with augmented architectures: ~2-4 hours additional
- Total: half a day to a day per heavy-ML-augmented arc

Reason to invoke selectively — not on every arc.
