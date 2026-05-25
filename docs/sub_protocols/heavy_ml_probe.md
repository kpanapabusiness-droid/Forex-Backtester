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

## Output (in addition to standard arc outputs)

`step_4/heavy_ml/`:
- `automl_leaderboard.csv` — per-fold model rankings from AutoML
- `automl_feature_importance.csv` — averaged across folds
- `survival_model_results.csv` — if Pipeline D variant invoked
- `meta_label_results.csv` — meta-labeler per-fold performance
- `compute_budget_used.md` — actual model evaluations consumed (audit against 1000/fold cap)

`step_5/heavy_ml_augmented_architectures.csv` — A2/A4/A6 results using heavy-ML-trained components

**Scope clarification (PR-E):** the augmented-architectures CSV is produced by the **overseer Step 5 loop**, not by `heavy_ml_probe` itself. PR-E ships **adapters** that translate heavy-ML-trained components into the existing `core/architectures/{A2,A4,A6}*.py` config shapes (per Q5a + Q5b), and emits `step_5/heavy_ml_augmented/heavy_ml_manifest.json` describing what's adapter-buildable. When the overseer's Step 5 loop runs against an arc whose `ARC_OPEN.md` declares `sub_protocol: heavy_ml_probe`, the loop reads this manifest, builds A2/A4/A6 via the adapters, runs them, and writes the augmented-architectures CSV alongside its vanilla Step 5 outputs. This keeps heavy_ml_probe narrow-scope (no direct A2/A4/A6 invocation).

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
