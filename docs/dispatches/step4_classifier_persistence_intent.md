# step4_classifier_persistence_intent.md

> **Dispatch:** `CC Dispatch — Engine PR: Expose Step 4 Fitted Classifiers for A2/A6 Architectures`
> **Branch (current):** `claude/dreamy-meitner-4d18ac` (worktree off `main`). Dispatch names `engine/step4-classifier-persistence` — will rename / open PR from this branch unless chat directs otherwise.
> **Scope:** engine change. Persist Step 4's per-cluster best-AUC classifier to disk; expose paths through `ClusterExtraction`; add loader + A2/A6 config builders; threshold-sweep without retraining; tests + docs. No algorithmic change. No verdict logic change.
> **Stage:** Read-first complete. Awaiting chat answers on five interpretive calls before executing Tasks 1-10.

---

## Reads completed

1. **[core/steps/step_4_extraction.py:1-399](core/steps/step_4_extraction.py)** — current `Step4Result`, `ClusterExtraction`, classifier training flow.
   - `ClusterExtraction` already has: `cluster_id`, `n_trades`, `excluded_features`, `used_features`, `classifier_fold_results`, `best_classifier`, `best_classifier_mean_auc`, `best_threshold`, `feature_importance` (step_4_extraction.py:60-72). `best_threshold` already present — dispatch's "if present, value MUST match" rule applies.
   - Training flow: per cluster, `_train_one_classifier` is called for RF / LR / (LGBM if available). Each call fits one model per CV fold (5 folds × {rf, lr, [lgbm]} = 10-15 fitted models per cluster), evaluates per-fold AUC, then **discards** the fitted estimators. No single "the best-AUC classifier" object survives the function (step_4_extraction.py:132-187).
   - This means there is **no fitted final estimator** at end of `run_step_4` today — only metrics. Persisting "the best classifier" requires us to fit one extra model after CV selection. **See Interpretive call 1.**
2. **[core/architectures/a2_classifier_filter.py:1-229](core/architectures/a2_classifier_filter.py)** — `A2Config` (lines 42-59) expects `classifier: ClassifierLike`, `threshold: float`, `classifier_feature_order: tuple[str, ...]`. `ClassifierLike = Any` (line 39) — only needs `predict_proba`. Mirrored in [core/architectures/a6_meta_labeling.py:51-68](core/architectures/a6_meta_labeling.py).
3. **[L_PROTOCOL.md:231-283](L_PROTOCOL.md)** Amendment 2. Confirms contract: A2 — "no retraining at Step 5 (use Step 4 output directly)" (line 238). A6 — "Classifier: Step 4's best-AUC classifier (same as A2)" (line 270). **But also (line 282):** "Per-fold WFO trains classifier on IS, evaluates on OOS — no global model trained once and used across folds." That last clause applies to A3/A4 (new-classifier architectures, line 282 lives under "Shared discipline" but in practice A3/A4 retrain per fold by design; A2/A6 reuse Step 4's fit). **See Interpretive call 2.**
4. **[docs/archive/arc_results/ARC_5_RESULT.md](docs/archive/arc_results/ARC_5_RESULT.md)** — Step 6 FAIL closure. Steps 1-5 ran under v2.1.x with bespoke per-fold retraining (line 106: "Strict per-fold classifier retraining"); no §3 follow-up note describes the v3.0 A2/A6 engine gap directly. The gap surfaces in **[scripts/l_arc_11/run.py:1-32](scripts/l_arc_11/run.py)** docstring: "Bypasses `ArcOrchestrator._run_step_5` (which has a wiring gap: doesn't plumb `run_context` through to `ArcFoldRunner`, which means A2/A6/A4 architectures requiring `per_trade_features` cannot be evaluated via the orchestrator's Step 5)." Arc 11 works around the gap by training a fresh classifier in `prefit_classifier` (lines 191-218) on the 2010-2020 training window. **The bypass = the gap this PR is closing.**
5. **Integration points.**
   - `core/arc/arc_orchestrator.py:_run_step_5` (lines 160-189) builds `(arch, conf)` pairs from `cfg.architecture_configs` and runs each through `ArcFoldRunner`. **Does not build A2Config / A6Config from Step 4 output** — caller pre-populates `architecture_configs` with already-built configs (so A2/A6 cannot currently be wired without external classifier prefitting).
   - `core/arc/arc_orchestrator.py:_run_step_5` (line 174-181) creates `ArcFoldRunner` WITHOUT passing `run_context`. Arc 11 documents this as a second gap. **Out of scope for this PR per dispatch ("no verdict logic changes, no algorithmic changes"); flagged in Interpretive call 5 as a follow-up.**
6. **[core/steps/_classifier_defaults.py:1-99](core/steps/_classifier_defaults.py)** — RF (sklearn `RandomForestClassifier`), LR (sklearn `Pipeline(StandardScaler + LogisticRegression)`), LGBM (`lightgbm.LGBMClassifier` if available). All have `predict_proba`. All deterministic at `random_state=42, n_jobs=1`.
7. **[scripts/l_arc_11/run.py:191-218](scripts/l_arc_11/run.py)** — `prefit_classifier`: trains on the **2010-2020 training window subset of the Step 1 pool** (line 200-202), filling NaN/inf with 0.0 (line 213). Returns `(model, tuple(fm_train.columns))`. **This is the de facto pattern A2/A6 currently rely on.** This PR's design choice must decide whether to mirror it (train on training window) or train on the **full Step 1 pool** (what `run_step_4` already sees during CV). **See Interpretive call 1.**
8. **[tests/protocol_runtime/test_step_4_extraction.py:1-79](tests/protocol_runtime/test_step_4_extraction.py)** — existing Step 4 tests use a synthetic 400-trade pool with 3 features. Adds 4 tests (synthetic determinism, AUC > 0.6, lineage exclusion). Pattern is reusable for the new persistence tests.
9. **[tests/protocol_runtime/test_architectures_synthetic.py:70-130](tests/protocol_runtime/test_architectures_synthetic.py)** — A2 / A6 smoke tests construct `A2Config` / `A6Config` with dummy `_AdmitAll` / `_Always06` classifiers. Pattern reusable for the integration test in Task 7.
10. **[requirements-dev.txt:1-9](requirements-dev.txt)** — **no version pins** for sklearn, lightgbm, or joblib (they're transitive). Dispatch Risk #1 (cross-version pickle compatibility) is real but unmitigated by the repo today. **See Interpretive call 4.**
11. **[docs/PROTOCOL_RUNTIME.md:217-247](docs/PROTOCOL_RUNTIME.md)** §7 Step 4 documentation. Update target.
12. **[docs/BACKTESTER_ARCHITECTURE.md](docs/BACKTESTER_ARCHITECTURE.md)** — **no Step 4 section exists** (searched: no `Step 4` / `step_4` / `extraction` matches). Dispatch Task 9 says to update the Step 4 section; **none to update**. Will create one (mirroring the dispatch's stated location: "Step 4 section with the new artefact"). **See Interpretive call 3.**
13. **[WORKFLOW.md](WORKFLOW.md)** — §2 "Arc-close artefact set" (lines 38-52) is closure-doc focused (template v1.2, deployment_spec). No per-step artefact list exists. **No update needed unless chat wants the classifier artefact called out (likely not — it's an internal Step 4 sub-artefact, not a closure artefact).**

---

## Design choice (locked by dispatch, restated)

Persist classifiers to **disk** via `joblib.dump(..., compress=3)`; expose paths through new fields on `ClusterExtraction` (`fitted_classifier_path`, `fitted_classifier_type`, `fitted_classifier_feature_order`). Classifier objects do NOT live in the dataclass. Path-based artefact + SHA256 integrity check on load.

Rationale (dispatch, restated):
- Pickle compatibility avoidance — path indirection keeps Step 4 dataclass picklable / hashable / equality-comparable.
- Mirrors per-day-max-DD parquet pattern from Amendment 3.
- Idempotent: re-running Step 4 overwrites the same path deterministically.

Storage layout (locked by dispatch):
```
results/<arc_name>/step_4/classifiers/
  <cluster_id>.pkl              # joblib-pickled best-AUC fitted estimator
  manifest.json                 # SHA256 + feature_order + threshold + AUC stats per cluster
```

---

## File paths CC will create / modify (in execution order)

| Order | Path | Action | Source |
|---|---|---|---|
| T1 | `core/steps/step_4_extraction.py` | Edit — extend `ClusterExtraction` with 3 new fields (`fitted_classifier_path`, `fitted_classifier_type`, `fitted_classifier_feature_order`); refit best classifier on full lineage-filtered pool after CV; persist to disk; write manifest | Tasks 1, 2 |
| T2 | `core/steps/classifier_persistence.py` | CREATE — `load_classifier(path)` + `ClassifierIntegrityError` + manifest helpers | Task 3 |
| T3 | `core/arc/arc_config.py` | **Does not exist** — will create OR colocate the two builders in `core/arc/arc_orchestrator.py` next to `ArcConfig`. **See Interpretive call 3 for location decision.** Adds `build_a2_config_from_step4(...)`, `build_a6_config_from_step4(...)` | Task 4 |
| T4 | `core/arc/arc_orchestrator.py` | Edit — `_run_step_5` plumbing optionally consumes Step 4 result to auto-instantiate A2/A6 configs (CC scope decision: out of scope; helpers are caller-facing). **See Interpretive call 5.** | Task 4 / 5 |
| T5 | `tests/protocol_runtime/test_step_4_classifier_persistence.py` | CREATE — file existence, manifest sha matches file sha, load round-trip identical predictions, builder constructs A2Config, threshold-sweep semantics, integrity check on corrupt file, determinism across re-runs | Task 6 |
| T6 | `tests/protocol_runtime/test_a2_end_to_end.py` | CREATE — Steps 1+2+3+4 on synthetic fixture → `build_a2_config_from_step4` → Step 5 runs, returns A2 result without engine errors | Task 7 |
| T7 | `docs/PROTOCOL_RUNTIME.md` | Edit §7 — add classifier persistence subsection; reference path + manifest schema | Task 9 |
| T8 | `docs/BACKTESTER_ARCHITECTURE.md` | Edit — **add a Step 4 section** (none currently exists); document the persistence pattern | Task 9 |
| T9 | `L_PROTOCOL.md` | Edit §2 Step 4 description — one sentence about "fitted best-AUC classifier is persisted to `results/<arc>/step_4/classifiers/<cluster>.pkl` with SHA256 manifest; A2/A6 load it via `build_a2_config_from_step4` without retraining" | Task 9 |
| T10 | `docs/dispatches/step4_classifier_persistence_log.md` | CREATE — verification log per WORKFLOW §2 dispatch pattern | end of work |

**Net:** 4 files modified, 5 files created (3 code + 2 doc artefacts).

---

## Interpretive calls flagged for chat

### 1. What data does the persisted best-AUC classifier fit on? — **load-bearing**

Step 4 today fits 5 fold-models per algorithm (RF/LR/[LGBM]) via TimeSeriesSplit, evaluates per-fold OOS AUC, picks the best algorithm by mean OOS AUC, and **discards all 15 fitted models**. To persist "the" classifier, Step 4 must fit one extra model after CV selection. Three options:

| Option | Training data | Pros | Cons |
|---|---|---|---|
| **A — Full Step 1 pool** (recommended) | All trades in `feature_matrix`, lineage-filtered, NaN-dropped | Maximises information content; what `run_step_4` already iterated over for CV evaluation; matches manifest's `trained_on_pool_size` field semantics in the dispatch | Trains on the full WFO window — at first glance reads like lookahead, BUT the classifier is a CLASSIFIER, not a strategy. Predicts cluster membership (a Step 2 label that's already pool-derived). Pool-derived classifier on pool-derived target is unchanged in its lookahead profile — leakage was already adjudicated when pool was built. |
| B — Training window only (2010-2020) | Trades with `entry_time < 2021-01-01` (Arc 11's `prefit_classifier` convention) | Conservative re: lookahead; matches Arc 11's de facto pattern | Engine doesn't know about the train/holdout split (`ArcConfig` has no `train_end` date — split lives in `WfoStructure`); requires plumbing the split into Step 4 OR the persistence helper. Adds a new parameter to `run_step_4`. |
| C — Per-fold persistence | Persist 5 separate fold-models | Honours L_PROTOCOL Amendment 2 line 282 "Per-fold WFO trains classifier on IS" most literally | A2/A6 contract is "no retrain at Step 5"; if 5 fold models exist, the architecture must pick one per WFO fold — implies A2/A6 mechanic change. Out of scope for "engine change, no algorithmic change". |

**Recommendation: Option A.** Justification: dispatch Task 2 mandates "After training the best-AUC classifier per cluster, pickle it to..." singular, deterministic path. The manifest schema (`trained_on_pool_size`, `auc_in_sample`, `auc_oos_cv5`) reads as one classifier per cluster. `auc_in_sample` only makes sense as a resubstitution metric on the full training set. Option C contradicts the dispatch's data model. Option B requires a `train_end` parameter that isn't currently in `run_step_4`'s signature — adds plumbing the dispatch doesn't ask for.

**Risk if Option A is wrong:** the persisted classifier sees all WFO years during training. When A2/A6 load it and run WFO, classifier predictions on out-of-fold years are not strict out-of-sample. **Mitigation:** classifier predicts cluster membership (a structural label), not P&L; structural label leakage was already adjudicated at Step 2. Arc 11's `prefit_classifier` chose Option B to be conservative — chat may want to mandate the same for consistency.

**If chat picks B**: I'll add a `train_window_end: pd.Timestamp | None = None` parameter to `run_step_4` (default `None` = full pool = Option A behaviour, preserves backwards compatibility), and the dispatch's helper `build_a2_config_from_step4` will work either way.

### 2. Reconciling with L_PROTOCOL line 282 ("Per-fold WFO trains classifier on IS")

Line 282 is under "Shared discipline across all ML architectures" and reads: "Per-fold WFO trains classifier on IS, evaluates on OOS — no global model trained once and used across folds."

This contradicts Amendment 2's A2/A6 spec (lines 238 / 270): "no retraining at Step 5 (use Step 4 output directly)".

Reading I'm acting on: line 282 applies to A3/A4 (Pipeline DE and Pipeline D exits) — which explicitly retrain a NEW classifier per fold by design (lines 243, 260). A2/A6 reuse the single Step 4 classifier across folds. The "global model" language at line 282 is at odds with this but I think it's a wording issue, not a design issue — otherwise A2/A6 as specified cannot exist.

**Recommendation:** chat clarifies whether line 282 should be amended to say "applies to A3/A4 only; A2/A6 use one Step-4-fit classifier across folds." This PR proceeds on that interpretation. No L_PROTOCOL text change in scope here — flag for v3.X amendment cycle.

### 3. Builder helper location

Dispatch says "Add helper in arc config layer" referring to `A2Config / A6Config` definitions. But `core/arc/arc_config.py` **does not exist** — `ArcConfig` lives at [core/arc/arc_orchestrator.py:55-77](core/arc/arc_orchestrator.py). `A2Config` lives in `core/architectures/a2_classifier_filter.py`. `A6Config` lives in `core/architectures/a6_meta_labeling.py`.

Three options:
- **A — `core/arc/arc_orchestrator.py`** (next to `ArcConfig`). Pros: dispatch's "arc config layer" reading. Cons: orchestrator file already heavy; coupling.
- **B — Per-architecture file** (`a2_classifier_filter.py`, `a6_meta_labeling.py`). Pros: builder lives next to the config it builds. Cons: imports `Step4Result` from `core.steps` into `core.architectures` — direction reversal from current import graph.
- **C — New `core/steps/classifier_persistence.py`** (where `load_classifier` lives). Pros: persistence + loading + config-building all in one cohesive module. Cons: file does multiple things.

**Recommendation: Option C.** Co-locates persistence-write + persistence-read + config-build (all three consume the same manifest / path / SHA convention). Keeps the import direction `core.arc` / `core.architectures` ← `core.steps` (consumers depend on the persistence module, not vice versa). Calling pattern from a driver: `from core.steps.classifier_persistence import load_classifier, build_a2_config_from_step4`.

### 4. Version pinning for joblib / sklearn / lightgbm (dispatch Risk #1)

`requirements-dev.txt` (lines 1-9) does not pin sklearn, joblib, or lightgbm. Pickle binary compatibility across versions is real. Dispatch flags this and asks to pin in `pyproject.toml` "if not already" — but **there is no `pyproject.toml`**. Options:

- **A — Add pins to `requirements-dev.txt`** in this PR (one-line additions, ~3 packages).
- **B — Skip the pinning sub-task; flag for separate PR** (dispatch Task 6 already includes a determinism test that loads → predicts vs in-memory predictions — catches binary mismatch at test time, just not in production).
- **C — Use joblib + sklearn `__version__` write at manifest creation; loader verifies on read** (defensive; raises if loaded under a different version).

**Recommendation: B + C.** Add version-string fields to the manifest (`joblib_version`, `sklearn_version`, `lightgbm_version`); loader emits a `UserWarning` (not an error) on mismatch; pinning lives in a follow-up PR with broader dependency review. Rationale: the determinism test (Task 6) catches actual behavioural mismatch; version-string surveillance gives the operator a visible signal without breaking valid loads when minor versions move.

### 5. Plumbing `A1RunContext` through `ArcOrchestrator._run_step_5` (the second Arc 11 gap)

Arc 11's docstring flags TWO gaps with the orchestrator:
1. **Step 4 doesn't persist its classifier** ← this PR
2. **`_run_step_5` doesn't construct `A1RunContext` from `cfg.feature_matrix`** — even with builders, the orchestrator's auto-Step-5 still can't run A2/A6 because `ArcFoldRunner` doesn't receive `per_trade_features`.

Dispatch is silent on gap (2). Fixing it is a 5-line change (build `A1RunContext` from `cfg.feature_matrix`, pass to `ArcFoldRunner`). Without it, A2/A6 builders are usable from custom drivers (like `scripts/l_arc_11/run.py`) but **not** from `ArcOrchestrator.run()`.

**Recommendation: include it.** Both gaps are blockers for the same workflow (orchestrator-driven A2/A6). Fixing only gap (1) leaves the integration test in Task 7 ("end-to-end A2 via Step 5") only passable via a hand-rolled driver, not the orchestrator path. Wave 2 dispatches still need a working orchestrator. **If chat declines**: I'll mark the Task 7 integration test as orchestrator-bypass (instantiate `ArcFoldRunner` directly with `A1RunContext`) and log the gap as a follow-up.

---

## Tests I'll add (Tasks 6 + 7 detail)

`tests/protocol_runtime/test_step_4_classifier_persistence.py`:
1. `test_step_4_persists_classifier_file_at_expected_path` — file exists at `results/<arc>/step_4/classifiers/<cid>.pkl` after `run_step_4(..., persistence_dir=tmp_path)`.
2. `test_manifest_sha_matches_file_sha` — recompute SHA256 of pickle bytes; assert equality with manifest entry.
3. `test_load_classifier_round_trip_predictions_identical` — fit in-memory; predict on held-out X; persist; load; predict on same X; assert array equality.
4. `test_build_a2_config_from_step4_constructs_without_exception` — synthetic Step 4 result → `build_a2_config_from_step4(s4, cluster_id=1)` returns an `A2Config` with non-None classifier.
5. `test_threshold_sweep_does_not_retrain` — `build_a2_config_from_step4(..., threshold_override=0.7)` returns config with `threshold == 0.7`; classifier identity (`id()` or `joblib.hash`) matches the underlying loaded estimator across calls.
6. `test_corrupted_pickle_raises_integrity_error` — write 1 byte over a valid pickle; assert `load_classifier(path)` raises `ClassifierIntegrityError`.
7. `test_determinism_across_runs` — two `run_step_4(...)` calls on same synthetic input → matching SHA256 in manifests (joblib binary determinism is suspect across joblib versions, so primary assertion is **post-load classifier `predict_proba` byte-equality on a held-out sample**; raw byte equality asserted as a soft check that warns if differing).

`tests/protocol_runtime/test_a2_end_to_end.py`:
1. Build synthetic Step 1+2+3+4 on the `_fixtures` helpers; assert non-empty `step_4_result.per_cluster`.
2. `build_a2_config_from_step4(step_4_result, cluster_id=<first>)` → `A2Config`.
3. Construct minimal `ArcConfig` with `architectures=(A2Architecture(),)` and `architecture_configs=(a2_cfg,)`; **also `feature_matrix=<the same matrix Step 4 saw>` so A2 can build its run context**.
4. Run `ArcOrchestrator(...).run()`; assert it completes; assert `result.verdict` is a valid enum value; assert Step 5 produced at least one `StrategyResult` for A2.

Dispatch line: "This is the test that would have caught the gap. CRITICAL." — confirmed. Without Interpretive call 5 resolved, this test will be **orchestrator-bypass** (instantiate `ArcFoldRunner` directly) rather than going through `ArcOrchestrator.run()` — flagged.

---

## Risks (additional to dispatch)

- **Joblib determinism (dispatch Risk #2 expanded).** `joblib.dump(...)` writes the joblib protocol version + sklearn class metadata in the binary blob. Across two runs in the same environment / version, bytes are typically deterministic — but not guaranteed at every joblib version. The persistence test mitigates by also asserting prediction-identity (Task 6 test #7 above).
- **Disk size (dispatch Risk #3 quantified).** Per Appendix A: RF `n_estimators=200, max_depth=6`. On a synthetic 400-trade pool, expect ~50-200 KB per RF. On a 12k-trade pool (real arc scale), expect ~5-15 MB per RF (depends on actual depth). LGBM: similar order. 5-10 clusters per arc = 50-150 MB per arc. **Acceptable; well under the 50-500 MB dispatch upper bound.** Will not push to commit threshold for git LFS conversation.
- **`run_step_4` signature change.** Adding a `persistence_dir: Path | None = None` parameter — when `None`, persistence is **skipped** (preserves test fixtures that don't want disk writes). Existing 4 Step 4 tests pass `None` implicitly and continue to work. Documented in Task 8 backwards-compat coverage.
- **Pre-PR arc results.** Existing Arcs 5/7/8/10/11 do not have persisted classifiers. Dispatch Task 8 says: documented in change log, pre-PR arc results need Step 4 rerun to access classifiers. Confirmed: nothing breaks; Step 4 results are not deserialised from disk anywhere (they're rebuilt fresh in each driver run).

---

## Definition of done (echoed from dispatch)

1. ✅ `ClusterExtraction` dataclass extended with classifier-path fields
2. ✅ Step 4 persists per-cluster classifiers to `results/<arc>/step_4/classifiers/` with manifest
3. ✅ `load_classifier` utility validates SHA256 and returns the fitted estimator
4. ✅ `build_a2_config_from_step4` and `build_a6_config_from_step4` helpers exposed (location pending Interpretive call 3)
5. ✅ Threshold sweep works without retraining (one classifier load, multiple A2Config instances)
6. ✅ Unit tests + end-to-end A2 integration test pass
7. ✅ Docs updated (engine architecture, L_PROTOCOL §2 Step 4 mention, PROTOCOL_RUNTIME §7)
8. PR title: `[ENGINE] Step 4 fitted-classifier persistence for A2/A6 architectures`

---

## End-turn questions for chat

1. **Interpretive call 1** — Option A (full Step 1 pool), B (training window only), or C (per-fold)? **Recommend A.**
2. **Interpretive call 2** — confirm A2/A6 use one Step-4-fit classifier across folds (vs L_PROTOCOL line 282 wording)?
3. **Interpretive call 3** — builder helper location: orchestrator file (A), architecture files (B), or `core/steps/classifier_persistence.py` (C)? **Recommend C.**
4. **Interpretive call 4** — version pinning: pin now in `requirements-dev.txt` (A), defer to follow-up + version-string warnings (B+C), or accept dispatch's `pyproject.toml` wording (which doesn't exist here)? **Recommend B+C.**
5. **Interpretive call 5** — also fix the second Arc 11 gap (`_run_step_5` builds `A1RunContext` from `cfg.feature_matrix`)? **Recommend yes** — without it the integration test in Task 7 cannot use the orchestrator path.

Awaiting chat answers before executing Tasks 1-10.
