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

3. **Survival model variant for Pipeline D.** When arc declares `pipeline_d_exits` as a target architecture, also train Cox Proportional Hazards model and Random Survival Forest on time-to-exit. Predicted hazard at each post-entry bar feeds Step 5 Pipeline D exit policy.

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
