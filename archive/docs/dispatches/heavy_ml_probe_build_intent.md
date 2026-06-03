# heavy_ml_probe — Build Intent Doc

> **Authoring CC session:** worktree `elegant-ride-ad9a57` on branch `claude/elegant-ride-ad9a57`.
> **Status:** intent doc only; no code, config, or branch changes this turn. End-of-turn for chat review per WORKFLOW §2 + dispatch §11.
> **Dispatch reference:** `CC Dispatch — Build heavy_ml_probe Sub-Protocol` (this conversation).
> **Source of truth being implemented:** `docs/sub_protocols/heavy_ml_probe.md` v1.0 (locked).
> **Sibling reference consumed:** `core/discovery/**`, `scripts/arc_discovery_01/**`, `configs/arc_discovery_01.yaml`, `docs/dispatches/arc_discovery_01_intent.md`, `tests/discovery/**`.
> **Predecessor docs read in order per §11:** `L_PROTOCOL.md` (full, with focus on §1 / §2 Step 1 / §2 Step 4 / §2 Step 5 incl. Amendment 2 retraining policy + Amendment 3 risk-normalised gates / §3 / §5 sub-protocol mechanism); `docs/sub_protocols/heavy_ml_probe.md` v1.0; `docs/sub_protocols/signal_discovery_probe.md` v1.0; `WORKFLOW.md` §2 + §6; `ARC_TRACKER.md` (read-only — no row to touch).
> **Engine-consumer modules surveyed:** `core/steps/step_4_extraction.py`, `core/steps/classifier_persistence.py`, `core/steps/path_classifier_per_fold.py`, `core/architectures/{a1,a2,a4,a6}*.py`, `core/architectures/_path_classifier.py`, `core/features/{lineage,pipeline,registry}.py`.

---

## §1 Goal restatement (one paragraph)

Build the `heavy_ml_probe` sub-protocol so a future arc can declare `sub_protocol: heavy_ml_probe` in its `ARC_OPEN.md` and run the overseer's Step 4 + Step 5 with: (a) FLAML AutoML in 11-fold TimeSeriesSplit replacing the vanilla three-classifier Step 4, (b) a meta-labeling target (`reach +1R MFE before SL`) trained via the same AutoML path, (c) Cox PH + Random Survival Forest fed into A4 (Pipeline D exits), and (d) Step 5 augmentation hook so A2 / A4 / A6 consume heavy-ML-trained components. No arc dispatched against this; build-now-so-ready-when-needed. Overseer Step 1 / 2 / 3 / 6 + §3 gates untouched.

---

## §2 Prerequisites verified

| Prerequisite | Status | Where |
|---|---|---|
| L_PROTOCOL v3.0 + Amendments 1-3 locked | ✅ | `L_PROTOCOL.md` header + §3 risk-normalised gates locked. heavy_ml_probe does NOT touch gates per spec §"Discipline rules". |
| Sub-protocol v1.0 doc | ✅ | `docs/sub_protocols/heavy_ml_probe.md` (read; spec is the bind). |
| Sibling sub-protocol implemented | ✅ | `core/discovery/**` (10 modules) + `scripts/arc_discovery_01/run_discovery.py` + `tests/discovery/**` (9 test modules). Conventions to mirror: IO layout, manifest schema, lineterminator='\n', sorted JSON, sha256, random_state=42, n_jobs=1, locked YAML config. |
| Feature lineage source-of-truth | ✅ | `core/features/lineage.py::CausalLineage` (CLEAN/SUSPECT/UNVERIFIED) attached at `FeatureSpec` construction. `core/features/pipeline.py::feature_lineage_dataframe()` returns columns `name, feature_class, causal_lineage, needs_panel, description`. Already consumed by `core/discovery/causal_filter.py` — heavy_ml_probe reuses the same read pattern (see §4 Q3 resolution). |
| Step 4 classifier persistence + manifest convention | ✅ | `core/steps/step_4_extraction.py::_persist_best_classifier` + `_write_manifest` (joblib.dump compress=3, sha256, classifier_type/feature_order/best_threshold/auc_oos_cv5 entries). `core/steps/classifier_persistence.py::load_classifier` with SHA256 verification + version-drift warning. A heavy-ML augmented artefact set can land in a parallel directory using the exact same manifest schema. |
| A2 / A6 consumer wiring exists | ✅ | `build_a2_config_from_step4`, `build_a6_config_from_step4` already load a `predict_proba`-shaped classifier and wrap it in `A2Config` / `A6Config`. Heavy-ML AutoML output IS `predict_proba`-shaped (FLAML's best estimator exposes `predict_proba`) — drop-in compatible. |
| A4 consumer wiring exists for binary classifier | ✅ partial — see §4 Q5b | `core/architectures/a4_pipeline_d_exits.py::_A4ExitPredicate` consults `PathClassifierFit` via `predict_admit(...)`. Cox PH / RSF do NOT emit a binary admit decision — they emit hazards / survival probabilities. Spec §"Survival model variant for Pipeline D" says "predicted hazard at each post-entry bar feeds Step 5 Pipeline D exit policy." Translation rule (hazard → exit decision) is not yet specified at the consumer interface. Surface as Q5b open question. |
| WORKFLOW §2 intent → log → PR pattern | ✅ | Followed for sibling (`arc_discovery_01_intent.md` / `_log.md`). This doc is the heavy_ml_probe equivalent. |
| `docs/dispatches/` write convention | ✅ | All prior dispatch artefacts follow `<work>_intent.md` / `<work>_log.md` / `<work>_diagnostic.md` naming. This doc lands at `docs/dispatches/heavy_ml_probe_build_intent.md`. |
| Branch strategy | ✅ | Build branch `infra/heavy_ml_probe_build` cut from main; per-PR branches off it (`infra/heavy_ml_probe_pr_a` ... `_pr_f`). Final merge to main after PR-F. Per dispatch §12. |

---

## §3 Confirmed module structure (dispatch §4)

Dispatch §4's proposed tree is adopted verbatim with no deviations. Mirrors `core/discovery/**` shape so reviewers can navigate by analogy:

```
core/heavy_ml_probe/
  __init__.py                # version + public API surface
  automl.py                  # FLAML wrapper, evaluation-count enforcement, 11-fold CV loop
  meta_labeling.py           # reach-1R-before-SL target construction + AutoML training reuse
  survival.py                # Cox PH (lifelines) + RSF (scikit-survival); n < 200 warning per dispatch §6.10
  pipeline.py                # orchestration: Step 1 pool in → Step 4 heavy-ML artefacts out
  metrics.py                 # AUC + concordance + IBS wrappers
  io.py                      # manifest writer, deterministic CSV/parquet writers, sha256 utility
  causal_lineage.py          # pre-evaluation gate per spec §"Per-feature causal lineage"

scripts/heavy_ml_probe/
  __init__.py
  run_probe.py               # CLI: --arc <name> --pool <path> --config <yaml> --cluster-id <id>

configs/heavy_ml_probe/
  default.yaml               # AutoML budget, model search space, CV folds, thresholds, library versions

tests/heavy_ml_probe/
  __init__.py
  test_automl.py
  test_meta_labeling.py
  test_survival.py
  test_causal_lineage.py
  test_io_manifest.py        # added: same coverage gap noticed in sibling, worth filling proactively
  test_pipeline_integration.py   # renamed from dispatch's `test_integration.py` for clarity
```

**Two minor additions to dispatch §4** (flagged here for chat sign-off rather than silently changing):

1. `tests/heavy_ml_probe/test_io_manifest.py` — sibling has determinism-artefact tests (`tests/discovery/test_determinism_artefacts.py`); same pattern fits here. Keeps PR-A test surface honest without inflating PR-A.
2. `scripts/heavy_ml_probe/__init__.py` — matches `scripts/arc_discovery_01/__init__.py` (an empty marker so the script package imports cleanly). Trivial; included for parity.

Nothing else changes from dispatch §4.

---

## §4 Open question resolutions (dispatch §9)

Dispatch §11 asks me to answer each §9 question from the codebase where possible and explicitly list the rest for chat. Working through them in order:

### Q1 — Survival model target (CHAT, unresolvable from code)

Dispatch §9.1 lists three plausible framings:
- (a) time-to-SL-or-time-exit (realised exit event)
- (b) time-to-MFE-peak (time at which MFE was maximised)
- (c) time-to-reach-1R-MFE (event = +1R hit, censored at SL/time-exit)

The codebase doesn't choose. Each has different downstream A4 implications:

| Framing | What A4 does at each bar | Failure mode |
|---|---|---|
| (a) time-to-exit | Predict survival probability past horizon h; exit if S(h) drops below threshold. Maximally generic. | Lumps profitable runners and SL-flushes into one hazard, blunts signal. |
| (b) time-to-MFE-peak | Predict bars-until-MFE-peak; exit when t > predicted peak. | Requires MFE-peak in the training label (uses full path), which is fine for IS labels but A4 needs to behave at OOS bars where MFE-peak is unknown — only a prediction is used, but the target is path-dependent and noisy. |
| (c) time-to-reach-1R-MFE censored at SL/time-exit | Predict probability of reaching +1R within remaining horizon; exit when probability drops below threshold. Aligns directly with the meta-labeling target (reach-1R-before-SL). | Cleanest semantically; same event definition as the meta-labeling target → fewer concept mismatches. |

**CC recommendation: (c)** — it aligns the survival target with the meta-labeling target (both events = "reach +1R before SL"). One semantic story across heavy_ml_probe. (a) is the safe fallback; (b) is the most novel but most fragile. Chat to confirm before PR-D.

### Q2 — FLAML evaluation-count granularity (PARTIAL — default reading, chat to confirm strict reading)

FLAML's `max_iter` counts hyperparameter combinations evaluated, not "model evaluations" in the abstract. One `max_iter=1000` call typically spans multiple base learners with the budget divided across them by FLAML's bandit-style scheduler. **Default reading: 1 trial = 1 evaluation; `max_iter=1000` per fold = 1000 evaluations per fold = 11,000 per cluster.** This matches FLAML's `max_iter` semantics and is the cleanest mapping to dispatch §6 #6.

Stricter reading (1 base-learner × 1 hyperparam config = 1 evaluation; multi-learner FLAML run with `max_iter=1000` counts as N × 1000 where N = number of base learners) is possible but would force `max_iter ≈ 167` per fold (1000 / 6 learners), which materially reduces FLAML's search depth.

CC recommendation: default reading. Surface to chat; if stricter reading is required, set `max_iter = floor(1000 / n_estimators_searched)` in YAML and document.

HALT trigger: if FLAML cannot enforce a hard upper bound under either reading (e.g. internal warm-start trials are uncounted), HALT per dispatch §5 and surface for library swap.

### Q3 — Causal lineage tag schema (RESOLVED FROM CODEBASE — no chat needed)

Resolved by reading `core/features/lineage.py`, `core/features/pipeline.py`, `core/discovery/causal_filter.py`:

- Lineage tags are attached at `FeatureSpec` registration time via the `lineage: CausalLineage` field (`CLEAN` / `SUSPECT` / `UNVERIFIED`). Registry-resident; NOT embedded in Step 1's `pool.parquet` per-row.
- `feature_lineage_dataframe()` returns a DataFrame with columns `name, feature_class, causal_lineage, needs_panel, description`. This is the canonical lineage table.
- `core/discovery/causal_filter.py::clean_feature_pool(lineage_df, accepted=("clean",), exclude_classes=())` returns the filtered feature-name tuple.
- `core/steps/step_4_extraction.py::_filter_lineage` does the same job inside vanilla Step 4 (also keying on `lineage["causal_lineage"] == "clean"`).

**heavy_ml_probe convention (mirroring sibling):** `core/heavy_ml_probe/causal_lineage.py` consumes `feature_lineage_dataframe()`, filters the feature pool to `causal_lineage == "clean"`, and logs the rejection list with the same reason vocabulary as `core.discovery.causal_filter` (`non_clean_lineage`, `excluded_class`, `unknown_feature`). The CC layer enforces the pre-evaluation gate per dispatch §6 #2 by filtering training-set columns BEFORE the AutoML run starts. No new schema invented; no separate metadata file.

### Q4 — Per-cluster invocation pattern (RESOLVED — CC recommends single-cluster CLI)

Dispatch §9.4's default — `run_probe.py` takes a single cluster ID as input, orchestration over clusters is the caller's responsibility — is what CC will implement unless chat redirects. Reasons:

1. Keeps `run_probe.py` unit-testable in isolation (`test_pipeline_integration.py` only needs one cluster's worth of synthetic data).
2. Compute budget (8-24h per arc per spec §"Expected compute cost") is dominated by the per-cluster AutoML run — coarse-grained parallelism over clusters is best done at the caller layer (a shell loop or future orchestrator hook), not buried inside the script.
3. Matches sibling pattern: `scripts/arc_discovery_01/run_discovery.py` runs one arc's worth of search, not multi-arc auto-iteration.

If chat prefers auto-iteration over all candidate clusters from a Step 3 output, that's a one-flag change (`--all-candidate-clusters` reading `step_3/capturability_summary.md`). Mention in PR-A if chat redirects.

### Q5 — Step 5 architecture-layer interface (PARTIAL — A2/A6 work today, A4 needs a small interface decision)

**Q5a (A2 / A6 — RESOLVED FROM CODEBASE):** `core/steps/classifier_persistence.py` already exposes `build_a2_config_from_step4` / `build_a6_config_from_step4`. Both load a `predict_proba`-shaped fitted estimator from a directory layout `step_4/classifiers/<cluster_id>.pkl` + `step_4/classifiers/manifest.json`. FLAML's best estimator exposes `predict_proba`, so a heavy-ML artefact directory at `step_4/heavy_ml/classifiers/<cluster_id>.pkl` + sibling manifest is plug-and-play. PR-E will provide thin wrappers (`build_a2_config_from_heavy_ml`, `build_a6_config_from_heavy_ml`) that point at the heavy-ML directory instead of the vanilla one; same downstream `A2Config` / `A6Config` consumed.

**Q5b (A4 / survival — NEEDS CHAT DECISION):** A4's consumer interface today expects a `PathClassifierFit` with `.model.predict_proba` and a binary "exit if prob < threshold" rule (`_A4ExitPredicate.__call__` in `core/architectures/a4_pipeline_d_exits.py:110-159`). Survival models (Cox PH, RSF) don't fit this shape — they predict a hazard rate or a survival probability over a horizon, not a single "should-exit" probability.

Three options to bridge:

| Option | What changes | Pros | Cons |
|---|---|---|---|
| **B1.** Wrap survival prediction in a `PathClassifierFit`-shaped adapter that maps `S(h | features) < threshold` to a binary exit at the calling bar. | New `SurvivalPathClassifierAdapter` class with `.predict_admit(features) -> (admit, proba)`. A4 unchanged. | Zero engine surface change. Fits the dispatch §3 "out-of-scope" boundary. | Hides the survival semantics — debuggers see "AUC 0.X" but the underlying object is computing a hazard. Threshold tuning is at one horizon only. |
| **B2.** Extend A4 with an opt-in `SurvivalExitPredicate` alongside `_A4ExitPredicate`. Heavy_ml_probe builds the new predicate; vanilla A4 keeps `_A4ExitPredicate`. | Cleaner separation of survival vs. binary semantics. | Touches `core/architectures/a4_pipeline_d_exits.py` — dispatch §3 says that file is out-of-scope. Engine-team would have to approve. | Wider scope; needs a chat-decided "engine extension allowed?" sign-off. |
| **B3.** Surface the survival model in the Step 4 manifest only; A4 augmentation deferred to a follow-up engine PR; heavy_ml_probe ships with AutoML + meta-labeling consuming A2/A6 only, and A4 closure on heavy-ML stays a TODO. | Cleanest scope-wise. Matches dispatch §3 boundary literally. | Spec §"A4 (Pipeline D) uses the survival-model-predicted exit timing" not honored by this build; A4 heavy-ML augmentation becomes a follow-up. |

**CC recommendation: B1 (adapter pattern).** Keeps A4 untouched per dispatch §3, gives heavy_ml_probe a complete A4 path. The adapter's threshold is the hyperparam chat tunes at Step 5. If chat prefers B2 (cleaner) or B3 (most conservative scope), the impact is on PR-D's design — flag at PR-D intent.

PR-E will flag this in the PR description regardless of which option is chosen, per dispatch §9.5: "if that consumer code doesn't exist yet, this build can't be end-to-end-verified against A2/A4/A6 — only against the Step 4 artefact set."

---

## §5 Library choices (dispatch §5)

| Component | Library | Confirmed? | Notes |
|---|---|---|---|
| AutoML | **FLAML** | ✅ confirmed | Native `max_iter` budget enforcement maps to dispatch §6 #6 cleanly. HALT trigger reaffirmed: if `max_iter` cannot enforce the hard 1000-eval/fold cap under either reading from Q2, HALT and surface to chat for library swap. |
| Cox Proportional Hazards | **lifelines** | ✅ confirmed | Canonical Python implementation. Returns concordance, log-likelihood. |
| Random Survival Forest + IBS | **scikit-survival** (`sksurv`) | ✅ confirmed | Standard RSF + Integrated Brier Score. |
| Base learners inside FLAML search | RF, LightGBM, XGBoost, CatBoost, ExtraTrees, LR | ⚠️ partial — see §6 dep bump | LightGBM already in vanilla Step 4 (via `core.steps._classifier_defaults`); XGBoost + CatBoost NOT installed. |
| Determinism | stdlib + `random_state=42` | ✅ confirmed | n_jobs=1; lineterminator='\n'; sha256 sidecar manifests. Same convention as `core/discovery/io.py`. |

**Dependency bump required in PR-A:**

`requirements-dev.txt` currently lists: `numpy, pandas, pyarrow, pytest, ruff, pyyaml, pydantic, scipy, scikit-learn, joblib`. Missing for heavy_ml_probe: **`flaml`, `lifelines`, `scikit-survival`, `xgboost`, `catboost`, `lightgbm`** (lightgbm imports defensively in `core.steps._classifier_defaults` so it's optional today, but heavy_ml_probe relies on it more centrally and should make it a hard dep).

PR-A will append these to `requirements-dev.txt` and pin them in `configs/heavy_ml_probe/default.yaml` under a `library_versions` block (same convention as `core/steps/step_4_extraction.py:_LGBM_VERSION` tracking) so the manifest can record env drift like vanilla Step 4 already does.

No alternative libraries proposed — FLAML / lifelines / scikit-survival are the canonical choices per the spec doc and the dispatch. Hold the line.

---

## §6 Constraints baked in (dispatch §6)

Each constraint is mapped to the module that enforces it:

| Constraint | Module enforcing | Mechanism |
|---|---|---|
| 1. No lookahead | `causal_lineage.py` (pre-evaluation gate) + reuse of registry tags | feature pool filtered to `CLEAN` before AutoML sees any data |
| 2. Causal lineage gate is pre-evaluation | `causal_lineage.py::filter_training_columns(X, lineage_df)` | column-drop BEFORE `automl.fit` is called |
| 3. Holdout window OFF-LIMITS | `pipeline.py` accepts `train_end: pd.Timestamp` and filters trades to `entry_time < train_end` before passing into AutoML / meta-label / survival fit | mirrors `core/steps/step_4_extraction.py:362-382` |
| 4. 11-fold TimeSeriesSplit on IS | `automl.py::run_per_fold_automl(X, y, n_folds=11)` | `sklearn.model_selection.TimeSeriesSplit(n_splits=11)` |
| 5. Per-fold AutoML training | `automl.py` loops over `(train_idx, test_idx) in cv.split(X)` | NO "train once globally" shortcut path; verified by `test_automl.py` |
| 6. Compute budget hard cap | `automl.py::FlamlBudget` wrapper | `max_iter=1000` per fold; recorded evaluations per fold logged to `compute_budget_used.md`; HALT if cap exceeded |
| 7. Determinism | `core/heavy_ml_probe/io.py` + everywhere | `random_state=42`, `n_jobs=1`, `lineterminator='\n'`, sorted JSON, sha256 in manifest, two-run determinism in `test_pipeline_integration.py` |
| 8. sha256 manifests | `io.py::write_manifest` | mirror of `core/discovery/io.py::write_manifest`; outputs `step_4/heavy_ml/manifest.json` + `step_5/heavy_ml_augmented/manifest.json` |
| 9. r_min=0.15% / r_max=2.0% from Amendment 3 | NOT enforced by heavy_ml_probe | scalability gate lives in overseer Step 5 + §3; heavy_ml_probe emits artefacts, overseer reads them |
| 10. Minimum-N (n < 200) warning | `survival.py::fit_cox_ph(...)` + `fit_rsf(...)` | `warnings.warn(...)` and proceed; per dispatch §6.10 do NOT skip silently |

---

## §7 Output artefact layout (dispatch §7 — unchanged)

`step_4/heavy_ml/`:
- `automl_leaderboard.csv` — per-fold FLAML model rankings + per-config AUC + budget consumed
- `automl_feature_importance.csv` — per-feature permutation importance averaged across folds
- `survival_model_results.csv` — Cox PH coefficients + concordance, RSF feature importance + IBS
- `meta_label_results.csv` — per-fold AUC + threshold-sweep precision/recall for reach-1R-before-SL
- `compute_budget_used.md` — evaluations consumed per fold vs cap (HALT-trigger evidence if breached)
- `classifiers/<cluster_id>.pkl` — joblib-pickled best FLAML estimator (mirrors vanilla `step_4/classifiers/`)
- `classifiers/manifest.json` — sha256 + provenance (joblib / sklearn / flaml / lightgbm / xgboost / catboost versions; `auc_in_sample`, `auc_oos_cv5`, `trained_on_pool_size`, `feature_order`)
- `survival/<cluster_id>_cox.pkl` + `survival/<cluster_id>_rsf.pkl` (added — needed for PR-D's A4 adapter)
- `survival/manifest.json` — sha256 + provenance (lifelines / sksurv versions, target framing per Q1, concordance / IBS)
- `manifest.json` — top-level sha256 per file above

`step_5/heavy_ml_augmented/`:
- `heavy_ml_augmented_architectures.csv` — A2 / A4 / A6 results using heavy-ML-trained components (filled by the augmentation hook, not by this build per dispatch §3)
- `manifest.json`

PR-E emits an empty `step_5/heavy_ml_augmented/manifest.json` template + the hook contract documented for the engine team to consume.

---

## §8 Build order (dispatch §8 — confirmed verbatim)

No reordering. Sized estimates per PR appear below in §10. Each PR is reviewable in isolation per dispatch + WORKFLOW §5.

**PR-A — Scaffolding + causal lineage gate + IO + dep bump**
- `core/heavy_ml_probe/{__init__.py, pipeline.py (skeleton), causal_lineage.py, io.py, metrics.py}`
- `configs/heavy_ml_probe/default.yaml` (config skeleton)
- `scripts/heavy_ml_probe/{__init__.py, run_probe.py (stub running scaffolding only)}`
- `requirements-dev.txt` — append flaml, lifelines, scikit-survival, xgboost, catboost, lightgbm
- Tests: `test_causal_lineage.py`, `test_io_manifest.py`
- End turn for chat review.

**PR-B — AutoML**
- `core/heavy_ml_probe/automl.py` (FLAML wrapper, 11-fold TS-split, eval-count enforcement, leaderboard + importance)
- `pipeline.py` wires AutoML into orchestration
- Tests: `test_automl.py` (synthetic data; respects cap; deterministic)
- End turn.

**PR-C — Meta-labeling**
- `core/heavy_ml_probe/meta_labeling.py` (target = reach-1R-before-SL from pool's MFE/MAE columns; reuse AutoML path for training with new target; threshold sweep)
- `pipeline.py` orchestrates meta-labeling alongside AutoML
- Tests: `test_meta_labeling.py` (target construction matches spec; no lookahead in target; end-to-end runs)
- End turn.

**PR-D — Survival**
- `core/heavy_ml_probe/survival.py` (Cox PH via lifelines, RSF via sksurv; target per Q1 resolution; censoring at time-exit horizon)
- `pipeline.py` orchestrates survival training
- Adapter (Q5b option B1, unless chat overrides): `SurvivalPathClassifierAdapter` exposing `predict_admit(features) -> (admit, proba)` for A4 consumption
- Tests: `test_survival.py` (n ≥ 200 synthetic pool, Cox + RSF run end-to-end; metrics within sane bounds; minimum-N warning fires under n < 200)
- End turn.

**PR-E — Integration + Step 5 augmentation hook (gate PR)**
- `pipeline.py` full orchestration: causal-filter → AutoML → meta-label → survival → write artefact set + manifests
- Step 5 hook: emit `step_5/heavy_ml_augmented/manifest.json` template + consumer documentation for chat to wire engine-side
- `test_pipeline_integration.py` — small synthetic pool end-to-end; two-run determinism check (sha256-byte-identical)
- `compute_budget_used.md` template + writer
- End turn (gate PR).

**PR-F — Documentation + polish**
- README addition or operator notes for invocation
- Docstring / inline doc cleanup surfaced during PR-A → PR-E
- `docs/sub_protocols/heavy_ml_probe.md` itself NOT modified unless a spec gap was found and chat ratified (per dispatch §3)
- End turn (final PR).

---

## §9 Open questions remaining for chat (consolidated)

After §4 resolution, the following remain chat-side:

1. **Q1 — Survival target framing.** CC recommends (c) `time-to-reach-1R-MFE censored at SL/time-exit`. Chat to confirm or pick (a) / (b).
2. **Q2 — FLAML budget granularity.** CC defaults to 1 trial = 1 evaluation (max_iter=1000 per fold). Chat to confirm or impose stricter accounting.
3. **Q5b — A4 survival consumer interface.** CC recommends B1 (adapter pattern, A4 untouched). Chat to confirm or pick B2 (A4 extension) / B3 (defer A4 augmentation).
4. **Minor — module-structure deviations (§3 above).** Two added test files (`test_io_manifest.py`, `scripts/heavy_ml_probe/__init__.py`). Trivial; flag-only.

Q3 (lineage schema) and Q4 (per-cluster invocation) are CC-resolved from codebase reading — no chat action needed unless chat disagrees with the resolution.

---

## §10 Pace estimate (dispatch §10)

| PR | Estimate | Driver |
|---|---|---|
| PR-A | ≤ 1 day | Scaffolding is mostly file-creation + mirroring sibling conventions. Dep bump may surface install issues on Windows for sksurv (Cython-build) — flag if so. |
| PR-B | 1-2 days | FLAML wrapper + CV discipline + eval-count enforcement + permutation importance + first determinism check. Largest single PR. |
| PR-C | ≤ 1 day | Reuses PR-B's AutoML path with a different target. Target construction is mechanical from the pool's MFE/MAE columns. |
| PR-D | ≤ 1 day | Cox PH + RSF are out-of-the-box library calls; the adapter (per Q5b) adds a small wrapper class. |
| PR-E | ≤ 1 day | Mostly orchestration glue + integration test + manifest scaffolding. Step 5 hook is a markdown + JSON-template emission, not engine code. |
| PR-F | ≤ 0.5 day | Doc polish. |
| **Total** | **4-6 working days** | Matches dispatch §10. |

No artificial deadline. HALT per WORKFLOW §6 if:
- FLAML cannot enforce hard eval cap (PR-B)
- sksurv install on Windows fails (PR-A)
- Q1 / Q2 / Q5b are unanswered when their PR opens
- Determinism check fails on PR-E
- Any spec gap surfaces requiring `heavy_ml_probe.md` amendment

---

## §11 Files this dispatch will create / touch

### New
- `core/heavy_ml_probe/{__init__,automl,meta_labeling,survival,pipeline,metrics,io,causal_lineage}.py`
- `scripts/heavy_ml_probe/{__init__,run_probe}.py`
- `configs/heavy_ml_probe/default.yaml`
- `tests/heavy_ml_probe/{__init__,test_automl,test_meta_labeling,test_survival,test_causal_lineage,test_io_manifest,test_pipeline_integration}.py`
- `docs/dispatches/heavy_ml_probe_build_{intent,log}.md` (this doc + PR-F log)

### Modified (existing files touched)
- `requirements-dev.txt` — append heavy-ML deps (PR-A)

### Read-only (consumed but not modified)
- `core/features/{lineage,pipeline,registry}.py` — lineage source-of-truth
- `core/steps/step_4_extraction.py`, `core/steps/classifier_persistence.py` — vanilla Step 4 patterns to mirror
- `core/architectures/{a2,a4,a6}*.py`, `core/architectures/_path_classifier.py` — consumer interfaces for Q5a/Q5b reasoning
- `core/discovery/**`, `scripts/arc_discovery_01/**` — sibling implementation conventions
- `docs/sub_protocols/heavy_ml_probe.md` — the spec being implemented
- `L_PROTOCOL.md`, `WORKFLOW.md` — methodology + operations
- `ARC_TRACKER.md` — no row to touch (build dispatch, not arc)

### Explicitly NOT touched (per dispatch §3)
- `core/steps/**` — overseer Step 1-6 mechanics
- `core/architectures/**` — A1-A6 base implementations (Q5b option B1 keeps this true; B2 would change it, flag at PR-D if chat picks B2)
- `L_PROTOCOL.md`, `docs/sub_protocols/heavy_ml_probe.md` — the spec itself (HALT + surface if gap found)
- Any arc invocation; no `results/<arc>/` lands from this build

---

## §12 Branch + PR mechanics (dispatch §12)

- Build branch: `infra/heavy_ml_probe_build`, cut from `main` at the point this intent doc is approved
- Per-PR branches: `infra/heavy_ml_probe_pr_{a,b,c,d,e,f}` off the build branch
- Each PR merges into the build branch; final PR-F merge + build-branch merge to `main`
- One `docs/dispatches/heavy_ml_probe_build_log.md` written at PR-F time consolidating verification across all 6 PRs (per WORKFLOW §2.4)
- HALT pattern per WORKFLOW §6 produces `docs/dispatches/heavy_ml_probe_build_diagnostic.md`

The current worktree (`elegant-ride-ad9a57` on `claude/elegant-ride-ad9a57`) hosts THIS intent doc only. The implementation branches will be cut fresh from main after chat ratification per dispatch §12.

---

## §13 What I'm waiting on

Per dispatch §11 last line: **do not start PR-A until chat confirms intent.** This turn ends here.

Chat actions needed before PR-A opens:
1. Confirm or redirect §4 Q1 (survival target).
2. Confirm or redirect §4 Q2 (FLAML evaluation granularity).
3. Confirm or redirect §4 Q5b (A4 survival consumer interface).
4. Note (or veto) the two minor module-structure additions in §3.
5. Approve the `requirements-dev.txt` dep bump (PR-A surface).

Once those land, PR-A opens on `infra/heavy_ml_probe_pr_a` cut from `infra/heavy_ml_probe_build` cut from `main`.

---

End of intent doc.
