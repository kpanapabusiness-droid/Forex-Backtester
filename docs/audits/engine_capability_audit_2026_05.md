# Engine Capability Audit — 2026-05-23 (refreshed 2026-05-25 post-PR-#197)

> Read-only enumeration of which L_PROTOCOL v3.0 (+ Amendments 1-6) capabilities are wired end-to-end, partially wired, or missing.
> Methodology: every claim cites the file + arc/test/PR that exercised it (WIRED), or the file + scope estimate (PARTIAL/MISSING).
> **Refreshed 2026-05-25 in CC_22:** post-Phase-1-engine-build sprint (8 PRs: #185 / #186 / #188 / #189 / #193 / #194 / #195 / #197). Most prior MISSING items are now WIRED. Per-section statuses below reflect current `main` (commit `198d78f`). See "Post-Phase-1-engine-build summary" at end for the PR-by-PR resolution map.
> Wave 1 = Arcs 5, 7 closures in flight + 8, 10, 11 closed. Wave 2 (not dispatched yet) = Arcs 4 RERUN, 4, 6, 3, 1, 2. Phase 2 = sub-protocols + 6 new signal classes.

---

## Executive summary (refreshed 2026-05-25)

- **Total capabilities audited:** 68
- **WIRED:** 56 (was 30 at original audit)
- **PARTIAL:** 9 (was 24)
- **MISSING:** 2 (was 13) — A5 portfolio composition + heavy_ml_probe engine path
- **UNKNOWN:** 1 (A4 trail-precedence semantics — original status; downgraded to documented in PROTOCOL_RUNTIME §"A4 same-bar exit precedence" via PR #195)

### Tier 1 — Was "Blocks Wave 2 dispatch"; now ✅ MOSTLY CLEARED

1. **Amendment 3 engine emission:** ✅ **WIRED via PR #186.** `core.wfo.amended_gates`, `core.wfo.chained_dd`, `core.wfo.holdout_rerun`, `core.runners._fold_stats_helpers.compute_per_day_max_dd` + `core.arc.arc_orchestrator._run_amendment_3_evaluation`. Per-day max-DD parquet, chained DD via equity-stitching (v3.0.2 follow-up: full-window sim), scaled-risk holdout re-runs, priority-ordered failure-mode taxonomy — all emitted automatically when `s5.top_k` non-empty.
2. **`core/wfo/gates.py` Amendment-3 rewrite:** ✅ **WIRED via PR #186** — `amended_gates.classify_amended_fold_stats(...)` returns `AmendedGateResult` with every Amendment-3 tracker field. Legacy `classify_fold_stats` preserved for backwards-compatibility; the new gate runs on top.
3. **Step 4 lineage filter column-name mismatch:** ✅ **WIRED via PR #185** — column name reconciled (`lineage` ↔ `causal_lineage`). `_filter_lineage` now enforces lineage tags at training time.
4. **Step 4 holdout-window training filter + classifier persistence:** ✅ **WIRED via PR #185.** `Step4Result.per_cluster[*].fitted_classifier_path` + `core/steps/classifier_persistence.py` with SHA256 + provenance manifest; `train_end` parameter restricts CV + refit to IS-only; A2 / A6 instantiate via `build_a2_config_from_step4` / `build_a6_config_from_step4` with no Step-5 retrain (per Amendment 2 lock).
5. **`ArcOrchestrator._run_step_5` `run_context` plumbing:** ✅ **WIRED via PR #186** — `ArcConfig.auto_arch_specs` tuple drives Step-5 dispatch; orchestrator builds `A1RunContext` from `cfg.feature_matrix` and threads through every `ArcFoldRunner`. A2 / A6 admit gates now operational via orchestrator (was: arc-script bypass only).
6. **Per-fold retrain orchestration for A3 / A4:** ✅ **WIRED via PR #186** — `core/steps/path_classifier_per_fold.py` builds per-fold `PathClassifierFit` (target = cluster membership for A3; `final_r > 0` for A4). Threaded via `A1RunContext.path_classifier_fits`. Cost-decomposition emitted per top-K in `StrategyResult.metadata`.

### Tier 2 — Blocks Phase 2 sub-protocols

1. **`heavy_ml_probe` engine path:** **MISSING — in flight via PR #187 (parallel chat).** Spec at `docs/sub_protocols/heavy_ml_probe.md`. PR #187 PR-A landed; PR-B/C/D/E/F to follow per build plan. Engine still does not import autogluon/FLAML/lifelines/scikit-survival.
2. **`signal_discovery_probe` sub-protocol registry registration:** PARTIAL — engine path operational via PR #175 standalone CLI; sub-protocol registry at `core/arc/sub_protocol.py` still empty. Cosmetic for Phase 2; the standalone CLI invocation pattern works.

### Tier 3 — Was "Blocks specific Wave-1-in-flight arc work"; now mixed

1. **`ArcOrchestrator` auto-feature-matrix construction:** PARTIAL — `ArcConfig.feature_matrix` still required as external input. Wave 1 arcs (5, 7) carry the hand-roll pattern from Arc 11. Cosmetic.
2. **Integrity report 3-of-6 missing checks:** PARTIAL unchanged (D1-lag NaN-perturbation, spread-floor activation, KH-24 co-fire still by-hand at arc-script level).

### Tier 4 — Step 6 framework

✅ **WIRED via PR #188 (Amendment 4).** Full six-category framework at `core/step_6/`. Auto-dispatches on Top-1 PASS candidate per Amendment 4; manual CLI at `scripts/run_step_6.py`. Closure template v1.3 + parser v1.3 + tracker registry. See post-audit append §"Post-audit update — 2026-05-24 (L_PROTOCOL Amendment 4 / CC_17)" below for the per-capability migration.

### Tier 5 — Signal parity + EET session semantics

✅ **WIRED via PRs #189 + #193 + #197 (Amendment 6).**

- PR #189: mid-price features + 5ers EET bar boundaries + worst-case fills. Engine output venue-independent.
- PR #193: signal-module timezone alignment via canonical `core/signals/htf_alignment.py`. 8 modules fixed; pre-commit lint rule blocks regression.
- PR #197 / Amendment 6: 5ers EET broker trading day is the daily-DD measurement boundary. `core/time_utils/session_boundary.utc_to_eet_trading_day` is the single source of truth; `Panel.boundary_convention` carries the choice through orchestrator slicing. Three convention-aware consumers: `distance.py`, `reset_floor.py`, `compute_per_day_max_dd` (load-bearing).

### Tier 6 — Canonical exit-policy registry

✅ **WIRED via PR #195.** `core/sim/exit_policies/` with `ExitPolicy` ABC + six registered policies including `sl_partial_close_1r_runner_trail` (Arc 10's load-bearing exit). Account partial-fill semantics added (`partial_close`, `current_size_of`, `_current_sizes`, `ClosedTrade.parent_position_id`). Driver wiring + per-arc migration (Arc 10 v3, Arc 8) complete; 218 byte-identical reference-parity tests.

### Tier 7 — Architecture selection (Amendment 5)

✅ **DOCUMENTED via PR #194.** Engine unchanged (all six architectures wired post-PR-186); enforcement is dispatch-time. Closure template v1.3.1 adds `architectures_skipped_by_amendment_5` optional field; parser v1.3 validates it on post-cutoff PASS verdicts.

### Housekeeping / non-blocking

- Closure-writer template `core/arc/_closure_template.py` still emits generic skeleton; v1.3.1 `§1 tracker_payload` YAML block + §4 deployment_spec still hand-written. PARTIAL unchanged.
- CI runs `ubuntu-latest` only; no Windows runner. PARTIAL unchanged. Two-run sha256 within Linux IS exercised.
- Arc 8 v3.0 closure §10 correction (`primary_failure_mode` → `step5_not_scalable`) — pending; tracked in TODO.md.

---

## Detailed findings

### Step 1 — Plumbing

#### Pool generation via `build_ex_ante_bounded_population` for all 28 pairs
- **Status:** WIRED (functionally — uses ex-ante construction; the literal `build_ex_ante_bounded_population` function name is from the prior protocol layer)
- **Evidence:** `core.arc.arc_pool_builder.build_arc_pool` ([core/arc/arc_pool_builder.py:267](core/arc/arc_pool_builder.py:267)) iterates sorted pairs in `primary.pairs`, calls `signal_module.evaluate(panels)` once before any per-trade simulation. Exercised by Arc 8 ([scripts/l_arc_8/build_step1_pool.py](scripts/l_arc_8/build_step1_pool.py)), Arc 10 ([scripts/l_arc_10_v3/step_1.py](scripts/l_arc_10_v3/step_1.py)), Arc 11 ([scripts/l_arc_11/run.py:266](scripts/l_arc_11/run.py:266)), and ground-truthed by [tests/protocol_runtime/test_arc_pool_builder.py](tests/protocol_runtime/test_arc_pool_builder.py).
- **Notes:** The "for all 28 pairs" is a property of the panel construction, not the pool builder — pool builder iterates whatever `panels[primary_tf].pairs` contains. Arc 11 lists all 28 pairs ([scripts/l_arc_11/run.py:79-86](scripts/l_arc_11/run.py:79)). All 28 pairs operationally exercised in production.

#### Feature matrix construction with full v3.0 default feature space
- **Status:** WIRED (the module exists and all 7 feature classes register on import; arc scripts compute the matrix end-to-end)
- **Evidence:** `core.features.pipeline.compute_feature_matrix` ([core/features/pipeline.py:74](core/features/pipeline.py:74)) imports the 7 feature-class modules (price_geometry, distance, multi_tf, cross_pair, spread_regime, session, vol_regime) at module load ([core/features/pipeline.py:25-33](core/features/pipeline.py:25)). Arc 11 builds the 27-feature matrix per pair at signal bars ([scripts/l_arc_11/run.py:138-178](scripts/l_arc_11/run.py:138)).
- **Notes:** Producers: 5 price-geometry features ([core/features/price_geometry.py](core/features/price_geometry.py)), 3 distance, 4 multi-TF (D1 slope sign/magnitude, D1 ATR percentile, W1 slope sign), 4 cross-pair, 2 spread-regime, 2 session + ~10 session-dummy registrations, 2 vol-regime. Total ~27 features.

#### Causal lineage tagging per feature
- **Status:** WIRED (declaration); PARTIAL (downstream enforcement)
- **Evidence:** Every `FeatureSpec` declares `lineage: CausalLineage` ([core/features/lineage.py:44-66](core/features/lineage.py:44)). `pipeline.feature_lineage_dataframe` emits a DataFrame with `name, feature_class, lineage, needs_panel, description` ([core/features/pipeline.py:52-71](core/features/pipeline.py:52)).
- **Notes:** Downstream enforcement is **PARTIAL** — Step 4 `_filter_lineage` ([core/steps/step_4_extraction.py:85-100](core/steps/step_4_extraction.py:85)) looks for column name `causal_lineage`, but the pipeline emits column name `lineage`. Silent bypass — see Step 4 §"holdout-window training filtering".
- **Scope to fix the column-name mismatch:** trivial (one-line rename in `_filter_lineage` or `feature_lineage_dataframe`). Blocks any honest lineage enforcement claim on existing arcs.

#### D1 lag-1 rule enforcement
- **Status:** WIRED (at the multi-TF feature producer layer)
- **Evidence:** `core.features.multi_tf._build_d1_lag1_series` ([core/features/multi_tf.py:32-50](core/features/multi_tf.py:32)) uses `pd.merge_asof(direction="backward")` on a key shifted by `-1 day`. Every multi-TF feature producer (D1 slope sign / magnitude / ATR percentile, W1 slope sign) routes through this. Arc 10 Step 6 audit verified the D1 swing-low detector's right-edge offset ([results/l_arc_10/step_6/audit_report.md](results/l_arc_10/step_6/audit_report.md), §"Producer-level causal trace — signal").
- **Notes:** Enforced producer-side. NOT enforced engine-side via an integrity check — see next item.

#### Integrity report emission
- **Status:** PARTIAL — emits 5 of 6 protocol-listed checks; "lookahead spot-check" is shallow
- **Evidence:** `core.arc.integrity` ([core/arc/integrity.py:40-146](core/arc/integrity.py:40)) emits: `pool_size_min`, `per_pair_below_warn`, `per_pair_zero_trades`, `coverage_window`, `lookahead_declared_lineage`, `determinism_two_run`.
- **Missing per L_PROTOCOL §2 Step 1:**
  - D1-lag NaN-perturbation test (3 random trades) — protocol prescribes; integrity layer does NOT implement
  - Spread-floor activation rate per pair (informational) — not in integrity layer
  - KH-24 co-fire rate (informational) — not in integrity layer
  - Lookahead spot-check (5 random trades, manually verified by causal lineage trace) — `check_lookahead_spot` just records declared lineage, doesn't trace 5 trades end-to-end ([core/arc/integrity.py:104-124](core/arc/integrity.py:104))
- **Scope:** 1-2 hr engine PR (each check is a single function emitting one `IntegrityRow`). Arc 10 reports D1-lag NaN-perturbation done by hand at Step 1.
- **Blocks:** Cosmetic — no Wave 2 arc breaks, but the closure docs claim integrity coverage they don't have.

---

### Step 2 — Clustering

#### Path-shape K-means / HDBSCAN over K ∈ {2..6}
- **Status:** WIRED (KMeans only; HDBSCAN not implemented)
- **Evidence:** `core.steps.step_2_clustering.run_step_2` ([core/steps/step_2_clustering.py:141-282](core/steps/step_2_clustering.py:141)) iterates `K_RANGE = (2, 3, 4, 5, 6)` ([core/steps/step_2_clustering.py:42](core/steps/step_2_clustering.py:42)), runs `KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300, algorithm="lloyd")` per K. Tested in [tests/protocol_runtime/test_step_2_clustering.py](tests/protocol_runtime/test_step_2_clustering.py). Exercised by Arc 8/10/11.
- **Notes:** L_PROTOCOL §2 Step 2 mentions "KMeans or HDBSCAN" — engine only ships KMeans. Not a blocker (KMeans is the default).

#### Silhouette computation
- **Status:** WIRED
- **Evidence:** [core/steps/step_2_clustering.py:131-138](core/steps/step_2_clustering.py:131) computes `silhouette_score(features, labels)` per K. Selected K = argmax silhouette over finite values ([core/steps/step_2_clustering.py:208-212](core/steps/step_2_clustering.py:208)).

#### Archetype tag assignment
- **Status:** WIRED
- **Evidence:** `core.steps._shape_tags.assign_shape_tag` ([core/steps/_shape_tags.py:62-92](core/steps/_shape_tags.py:62)) applies quartile rules over `ClusterCentroid` and returns one of `V_SHAPE | STEPWISE | BIMODAL | MONOTONIC_UP | MONOTONIC_DOWN | CHOPPY | UNCLASSIFIED`. Tested in [tests/protocol_runtime/test_shape_tags.py](tests/protocol_runtime/test_shape_tags.py). Tracker row at [ARC_TRACKER.md:120-132](ARC_TRACKER.md:120) shows tags assigned across Arcs 8/10/11.
- **Notes:** Threshold rules locked in code ([core/steps/_shape_tags.py:75-91](core/steps/_shape_tags.py:75)). Per L_PROTOCOL §2 "within-arc thresholds are immutable; cross-arc recalibration only."

#### Cluster assignment artefact
- **Status:** WIRED
- **Evidence:** `Step2Result.cluster_assignments` ([core/steps/step_2_clustering.py:56](core/steps/step_2_clustering.py:56)) — trade_id, cluster_id, k_selected. Orchestrator writes parquet at [core/arc/arc_orchestrator.py:291-293](core/arc/arc_orchestrator.py:291). Arc 11 writes the same ([scripts/l_arc_11/run.py:281-284](scripts/l_arc_11/run.py:281)).

---

### Step 3 — Capturability

#### Per-cluster reach_1R, MFE distribution, ww_pp
- **Status:** WIRED
- **Evidence:** `core.steps.step_3_capturability._cluster_metrics` ([core/steps/step_3_capturability.py:84-150](core/steps/step_3_capturability.py:84)) computes `reach_1r/2r/3r`, `mfe_p25/p50/p75/p90`, `wrong_way_pp` from the path table, plus `ttp_p25/p50/p75`, `mean_r`, `final_r_p25/p50`. Tested in [tests/protocol_runtime/test_step_3_capturability.py](tests/protocol_runtime/test_step_3_capturability.py).
- **Notes:** SL multiplier sweep across {1.5, 2.0, 2.5, 3.0, 3.5, 4.0} via R-multiple rescaling (approximation, documented) ([core/steps/step_3_capturability.py:130-142](core/steps/step_3_capturability.py:130)).

#### Composite capturability score
- **Status:** WIRED
- **Evidence:** Weighted aggregate `0.40·reach_1R + 0.40·(mfe_p50/3) + 0.20·(1-ww_pp)` ([core/steps/step_3_capturability.py:39-41](core/steps/step_3_capturability.py:39), 137-141). Per-cluster `capturability_composite` lands in CSV.

#### Archetype-aware ranking + candidate-cluster flag
- **Status:** WIRED
- **Evidence:** `is_candidate` = `reach_1r >= 0.50 AND ww_pp <= 0.30 AND mfe_p50 >= 1.5R` ([core/steps/step_3_capturability.py:43-45](core/steps/step_3_capturability.py:43), 214-218). Shape tag carried from Step 2 ([core/steps/step_3_capturability.py:200-212](core/steps/step_3_capturability.py:200)). Output CSV sorted by composite descending ([core/steps/step_3_capturability.py:267-268](core/steps/step_3_capturability.py:267)).

---

### Step 4 — Extraction

#### Per-cluster RF / LGBM / Logistic training
- **Status:** WIRED
- **Evidence:** `core.steps.step_4_extraction.run_step_4` ([core/steps/step_4_extraction.py:190](core/steps/step_4_extraction.py:190)) trains `build_rf`, `build_lr`, optionally `build_lgbm` (if `is_lgbm_available()`) via `_train_one_classifier` ([core/steps/step_4_extraction.py:265-274](core/steps/step_4_extraction.py:265)). Builders at [core/steps/_classifier_defaults.py](core/steps/_classifier_defaults.py) (matching L_PROTOCOL Appendix A).

#### 5-fold TimeSeriesSplit CV + per-fold AUC
- **Status:** WIRED
- **Evidence:** `TimeSeriesSplit(n_splits=N_TS_FOLDS=5)` ([core/steps/step_4_extraction.py:44](core/steps/step_4_extraction.py:44), [261](core/steps/step_4_extraction.py:261)). `roc_auc_score` per fold ([core/steps/step_4_extraction.py:159](core/steps/step_4_extraction.py:159)).

#### Threshold sweep (AUC-best / F1-best)
- **Status:** PARTIAL — AUC-best (Youden's J) only; F1-best NOT implemented
- **Evidence:** `_auc_best_threshold` uses ROC curve + Youden's J ([core/steps/step_4_extraction.py:103-114](core/steps/step_4_extraction.py:103)). No `_f1_best_threshold` function. L_PROTOCOL §2 Step 4 mentions "AUC-best and F1-best thresholds, report precision, recall, trade count" — engine emits AUC-best only.
- **Scope:** trivial — add `_f1_best_threshold` (precision_recall_curve + argmax F1). 30 min.
- **Blocks:** No current arc, but Wave 2 arcs whose closure docs cite F1-threshold sweeps would be hand-rolling them.

#### Top features by importance (permutation)
- **Status:** WIRED
- **Evidence:** `permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=1)` ([core/steps/step_4_extraction.py:171-178](core/steps/step_4_extraction.py:171)). Aggregated per cluster + classifier + fold; mean importance ranked descending ([core/steps/step_4_extraction.py:292-302](core/steps/step_4_extraction.py:292)).

#### Holdout-window training filtering
- **Status:** **WIRED** (PR #185)
- **Evidence:** `run_step_4` accepts `train_end: pd.Timestamp | None` parameter; restricts CV + refit to `entry_time < train_end`. Orchestrator threads `train_end` from `WfoStructure.holdout.oos_start`. Persisted classifier manifest records `train_end` field; `trained_on_pool_size` reflects IS-only subset.

#### Fitted classifier persistence
- **Status:** **WIRED** (PR #185)
- **Evidence:** `Step4Result.per_cluster[*].fitted_classifier_path` (pathlib.Path) + `fitted_classifier_type` + `fitted_classifier_feature_order`. Joblib-pickled to `step_4/classifiers/<cluster_id>.pkl` with `step_4/classifiers/manifest.json` SHA256 + provenance (joblib/sklearn/lightgbm versions, `auc_in_sample`, `auc_oos_cv5`, `trained_on_pool_size`, feature_order, `best_threshold`, `classifier_name`). `core/steps/classifier_persistence.py:load_classifier(path)` SHA256-verifies before load and warns on version drift. `build_a2_config_from_step4` + `build_a6_config_from_step4` instantiate A2 / A6 configs directly from Step 4 output (no Step-5 retrain per Amendment 2 lock).

#### Feature-importance pipeline end-to-end
- **Status:** WIRED
- **Evidence:** `Step4Result.feature_importance` aggregates per (cluster, classifier, fold, feature) rows; mean + std emitted; orchestrator writes `step_4/feature_importance.csv` ([core/arc/arc_orchestrator.py:311-313](core/arc/arc_orchestrator.py:311)). Confirmed in Arc 11's `step_4_per_cluster.top_10_features` field ([scripts/l_arc_11/run.py:586-589](scripts/l_arc_11/run.py:586)).

#### Lineage filter (additional finding)
- **Status:** **WIRED** (PR #185 — column-name reconciled)
- **Evidence:** Lineage filter now enforces tags at training time end-to-end. Features tagged anything other than `clean` are excluded from training and logged in `s4.extraction_metrics` + `summary_md`.

---

### Step 5 — WFO architecture search

#### A1 system_level_filter
- **Status:** WIRED
- **Evidence:** `core.architectures.a1_system_level_filter.A1Architecture` ([core/architectures/a1_system_level_filter.py](core/architectures/a1_system_level_filter.py)). KH-24 equivalence test [tests/protocol_runtime/test_kh24_a1_equivalence.py:test_kh24_runs_through_a1_without_error](tests/protocol_runtime/test_kh24_a1_equivalence.py) passes. Exercised by Arcs 5, 8, 10, 11 (CLAUDE.md). A1 is the verdict-carrying architecture for Arc 10's PASS-DEPLOYABLE.
- **Notes:** [scripts/anchor/check_a1_equivalence.py](scripts/anchor/check_a1_equivalence.py) is the workstation-runnable full-data anchor reproduction harness; result documented at CLAUDE.md ("Full-data anchor regression: A1 path byte-identical to legacy `KH24FoldRunner`").

#### A2 classifier_filter
- **Status:** **WIRED** (PR #185 + PR #186)
- **Evidence:** `A2Architecture` respects Amendment 2 (no Step 5 retrain) via `core/steps/classifier_persistence.build_a2_config_from_step4`. Orchestrator gap closed in PR #186: `ArcConfig.auto_arch_specs` drives Step-5 dispatch; orchestrator constructs `A1RunContext` from `cfg.feature_matrix` and threads through every `ArcFoldRunner`. A2 admit gates operational via orchestrator (was: arc-script bypass only). Cross_arc_tag `canonical_orchestrator_step5_run_context_gap` resolved.

#### A3 pipeline_de — per-fold retraining
- **Status:** **WIRED** (PR #186)
- **Evidence:** `core/steps/path_classifier_per_fold.py` builds per-fold `PathClassifierFit` from each fold's IS-only window (target = cluster membership for A3). Threaded via `A1RunContext.path_classifier_fits`. Architecture remains unchanged — receives the per-fold fit through `A3Config`. Cost-decomposition emitted per top-K in `StrategyResult.metadata`.

#### A4 pipeline_d_exits — per-fold exit-policy variation
- **Status:** **WIRED** (PR #186 + PR #195 precedence lock)
- **Evidence:** `core/steps/path_classifier_per_fold.py` also handles A4 (target = `final_r > 0` per fold IS window). Trail-vs-classifier-exit precedence locked in PR #195: when both fire same bar close, **trail-stop wins** (`exit_reason = "trailing_stop"`). Implementation: `core/sim/multipair_backtester.py:_process_bar` step 3 uses direct `_pending_closes[pos_id] = "trailing_stop"`. Regression test at [tests/protocol_runtime/test_multipair_backtester_precedence.py](../../tests/protocol_runtime/test_multipair_backtester_precedence.py).

#### A5 portfolio_composition
- **Status:** PARTIAL — architecture exists; no arc has exercised it
- **Evidence:** `A5Architecture` ([core/architectures/a5_portfolio_composition.py](core/architectures/a5_portfolio_composition.py)) consumes ≥ 2 `StrategyResult` constituents; combines closed trades and equity curves; recomputes DD on combined curve. Synthetic test [tests/protocol_runtime/test_architectures_synthetic.py:test_a5_combines_two_constituents](tests/protocol_runtime/test_architectures_synthetic.py) passes.
- No arc has surfaced 2+ candidate clusters at Step 3, so A5 has never been exercised in production. Per-architecture win rate at [ARC_TRACKER.md:65](ARC_TRACKER.md:65): "A5 portfolio_composition: 0 / 0".
- **Notes:** Amendment 3 §"A5 follow-up flag" defers combined-portfolio DD constraints. A5 itself can run; the gate logic for VIABLE-tier combination is the open work.

#### A6 meta_labeling
- **Status:** **WIRED** (PR #185 + PR #186 — inherits A2's classifier persistence + run_context plumbing)
- **Evidence:** `A6Architecture.run` loads classifier from `arch_config.classifier` (no Step 5 retrain) — matches Amendment 2. `_confidence_to_multiplier` maps `prob < lower → 0×`, `lower ≤ prob < upper → 0.5×`, `prob ≥ upper → 1×`. `core/steps/classifier_persistence.build_a6_config_from_step4` builds the config; orchestrator threads `A1RunContext` through.

#### WFO 11-fold IS evaluation
- **Status:** WIRED
- **Evidence:** `core.wfo.folds.build_v3_folds` produces 11 anchored-expanding-IS folds (2010-01-01 → 2020-12-31) plus a holdout fold ([core/wfo/folds.py:99-160](core/wfo/folds.py:99)). `run_search` iterates eligible folds (≥ 365 IS days) per [core/wfo/orchestrator.py:85-99](core/wfo/orchestrator.py:85). Exercised by Arcs 8, 10, 11.

#### Holdout one-shot evaluation
- **Status:** **WIRED** (mechanics) + **WIRED** (Amendment 3 re-run at scaled risk, PR #186)
- **Evidence:** `run_holdout` evaluates each top-K candidate exactly once on `structure.holdout`. Amendment 3 re-runs at `r_safe` / `r_hard` via `core.wfo.holdout_rerun.rescale_arch_config_risk` — per top-K candidate, two additional sims with `config_id` carrying the scaled-risk suffix.

#### Oracle WFO per cluster
- **Status:** WIRED
- **Evidence:** `core.runners.oracle_fold_runner` exists ([core/runners/oracle_fold_runner.py](core/runners/oracle_fold_runner.py)); Arc 8/10 produced `step_5/wfo_oracle.csv` artefacts ([results/l_arc_10/step_5/wfo_oracle.csv](results/l_arc_10/step_5/wfo_oracle.csv) exists). Arc 10's closure §"Post-closure research findings" cites the oracle Sharpe 4.61 vs base −1.29 gap from this runner.

#### Architecture ranking emission
- **Status:** WIRED
- **Evidence:** `WfoSearchResult.top_k` sorted by `(worst_fold_ratio, mean_fold_ratio)` descending ([core/wfo/orchestrator.py:101-111](core/wfo/orchestrator.py:101)). `step_5/architectures_ranked.md` written by arc scripts (Arc 8/10/11) and by `ArcOrchestrator._build_architectures_table` ([core/arc/arc_orchestrator.py:335-351](core/arc/arc_orchestrator.py:335)).

---

### Amendment 3 — Risk-normalised gates

> **All items in this section transitioned MISSING → WIRED via PR #186 + PR #197.** The tracker parser schema fields ([scripts/tracker_parser/schema.py:71-105](scripts/tracker_parser/schema.py:71)) are now populated by the engine. PR #197 / Amendment 6 swapped the daily-DD bucketing convention from UTC to EET broker day. Arc 10's PASS-DEPLOYABLE retrofit pre-dates the engine emission (re-evaluated via PR #177 retroactively).

#### Chained max DD emission across IS + holdout trajectory at `r_base`
- **Status:** **WIRED** (PR #186, v3.0.1 equity-stitching default)
- **Evidence:** `core.wfo.chained_dd.stitch_per_fold_oos_equity` + `compute_chained_max_dd_from_continuous_equity` produces multiplicative chaining of per-fold OOS returns with continuity adjustment. Emitted per top-K candidate in `AmendedGateResult.chained_max_dd_base_pct`.
- **Notes:** v3.0.2 follow-up (Q6 gold standard: single full-window sim per top-K) deferred — chained_dd_method field `equity_stitching | full_window_sim` introduced in closure template v1.2.1.

#### Per-day max-DD parquet emission
- **Status:** **WIRED** (PR #186 baseline + PR #197 / Amendment 6 EET bucketing)
- **Evidence:** `core.runners._fold_stats_helpers.compute_per_day_max_dd(equity_curve, *, boundary_convention="5ers_eet")` resamples equity to daily bars under the convention and writes `step_5/per_day_max_dd_base__<safe_cid>.parquet` per top-K candidate. Day-start equity is the first equity sample of the trading day under the active convention.
- **Notes:** PR #197 / Amendment 6 changed the default `boundary_convention` to `"5ers_eet"`. UTC opt-in preserved for KH-24 anchor byte-identity.

#### Verdict logic with priority-ordered gate evaluation
- **Status:** **WIRED** (PR #186)
- **Evidence:** `core.wfo.amended_gates.classify_amended_fold_stats(...)` returns `AmendedGateResult` with: `k_safe`, `k_hard`, `r_safe_pct`, `r_hard_pct`, scaled metrics, priority-ordered failure mode, both-tier verdict (DEPLOYABLE / VIABLE / FAIL). Locked thresholds: `R_MIN=0.15%`, `R_MAX=2.0%`, `CHAINED_DD_MAX_PCT=10%`, daily-DD threshold 5%.
- **Notes:** Legacy `classify_fold_stats` preserved for backwards-compatibility. The amended gate evaluates Amendment 3 §"Failure-mode priority" first-fail-wins order.

#### Holdout re-run at scaled risk (`r_safe` / `r_hard`)
- **Status:** **WIRED** (PR #186)
- **Evidence:** `core.wfo.holdout_rerun.rescale_arch_config_risk(arch_config, k_scale)` produces a frozen-dataclass copy with `risk_pct *= k_scale`. `ArcOrchestrator._run_amendment_3_evaluation` runs two additional holdout sims per top-K (one at `r_safe`, one at `r_hard`); `config_id` carries scaled-risk suffix (e.g. `a1_e2e_test_r0.0100`).

#### Sizing-convention gate check
- **Status:** **WIRED** (PR #186)
- **Evidence:** Every arch config carries `sizing_convention: str = "reset_floor"`. `ArcConfig.accept_equity_pct: bool = False` is the chat-approval override. Equity-pct sizing FAILs the scalability gate by default.

#### Failure-mode taxonomy emission
- **Status:** **WIRED** (engine + schema, both via PR #186)
- **Evidence:** `AmendedGateResult.primary_failure_mode` is assigned by `classify_amended_fold_stats` per the Amendment 3 priority order; the tracker schema enum at `scripts/tracker_parser/schema.py` accepts every value. Closures emit it directly from `AmendedGateResult` — no more hand-fill.

---

### Step 6 — Causal audit

> **All items in this section transitioned MISSING → WIRED via PR #188 (Amendment 4 framework).** Step 6 is auto-dispatched on Top-1 PASS candidate post-§3-pass; manual CLI invokable on any closure. Six-category framework at `core/step_6/`.

#### Producer-level feature trace as a runnable check
- **Status:** **WIRED** (PR #188)
- **Evidence:** `core/step_6/lookahead.py` — §6.1 includes `per_feature_lineage_clean` critical check enforcing causal_lineage on every feature consumed.

#### Byte-compare from raw OHLC for feature reproduction
- **Status:** **WIRED** (PR #188)
- **Evidence:** `core/step_6/byte_compare.py` — generic harness factored from Arc 10's `step_6_byte_compare.py`. Invoked by §6.1 `byte_compare_no_drift`.

#### D1 lag rule verification check
- **Status:** **WIRED** (PR #188)
- **Evidence:** `core/step_6/lookahead.py:_check_d1_lag_rule` — static source-inspection check.

#### Automated trigger on PASS verdict
- **Status:** **WIRED** (PR #188 + PR #186)
- **Evidence:** `core/step_6/dispatch.py:maybe_dispatch_step_6` — post-`_run_amendment_3_evaluation`, if at least one top-K has verdict PASS-DEPLOYABLE / PASS-VIABLE, Step 6 runs on the Top-1. `replace_top_1_with_step6_fail` re-classifies on critical failure.

#### Manifest / report artefact format
- **Status:** **WIRED** (PR #188)
- **Evidence:** `core/step_6/manifest.py` (CheckResult, CategoryAuditResult, Step6Result, Step6Manifest, Severity, VerdictImpact) + `core/step_6/artefacts.py` writers. Closure template v1.3 carries the `§1 tracker_payload.step_6` block; parser v1.3 enforces it for v1.3 PASS verdicts.

---

### Closure infrastructure

#### From-scratch v1.3.1 closure existence
- **Status:** PENDING — Wave 1 retries + Wave 2 will be first from-scratch v1.3.1 closures
- **Evidence:** Template current at v1.3.1 (PR #194 added `architectures_skipped_by_amendment_5`); parser v1.3 enforces Phase 2 fields. All existing v1.2 closures in `results/` are retrofits. Wave 1 retries (Arcs 5/7 v3.0.1, Arc 10 signal-parity) + Wave 2 (Arcs 4 RERUN / 4 / 6 / 3 / 1 / 2) will exercise the from-scratch path with full Amendment 3 + 4 + 5 emission.
- **Scope:** Operational discipline — no engine work.

#### Closure-writer auto-generation tooling
- **Status:** PARTIAL — orchestrator's writer produces only a generic skeleton, NOT v1.2 `§1 tracker_payload` YAML
- **Evidence:** `core.arc._closure_template.ARC_CLOSURE_TEMPLATE` ([core/arc/_closure_template.py:37-88](core/arc/_closure_template.py:37)) emits a flat markdown skeleton with `## 1. Headline / 2. Best architecture / 3. All architectures tested / 4. Step-by-step results / ...`. No machine-parseable YAML block. No `best_architecture` keyed YAML fields. No §4 deployment_spec section.
- Arc 11's run script ([scripts/l_arc_11/write_closure.py](scripts/l_arc_11/write_closure.py)) writes the v1.0-flavoured closure manually from `run_summary.json`. Arc 8 and Arc 10 closures were chat-written, then chat-retrofitted to v1.2.
- **Scope:** Multi-day engine PR — extend `_closure_template.py` to emit the v1.2 `§1 tracker_payload` YAML block keyed off `ArcOrchestratorResult` fields. §4 deployment_spec generation requires per-architecture spec emitters (A1 spec ≠ A2 spec ≠ A5 spec).
- **Blocks:** Closure friction is a constant tax on every arc close; not a hard blocker.

#### `config_artefact_path` validation at closure time
- **Status:** WIRED (in the parser)
- **Evidence:** `scripts.tracker_parser.schema` v1.2 validation requires `best_architecture.config_artefact_path` non-null + file-exists + `## §4 deployment_spec` heading present in the closure doc + `deployment_spec_section_present == true` (Section 4-L). Documented in [scripts/tracker_parser/README.md:65-78](scripts/tracker_parser/README.md:65). Tested in [tests/tracker_parser/test_pass_verdict_validation.py](tests/tracker_parser/test_pass_verdict_validation.py).
- **Notes:** Validation is enforced at parser invocation (pre-PR), not at orchestrator write time. A closure-writer that auto-generated v1.2 closures would emit `config_artefact_path` and `deployment_spec_section_present: true` by construction; right now both are filled by hand.

#### Parser v1.2 detection + validation on a from-scratch v1.2 PASS closure
- **Status:** PARTIAL — parser detects v1.2 correctly on retrofits; never exercised on a from-scratch v1.2 PASS closure
- **Evidence:** Parser tested on Arc 8/10/11 retrofits ([tests/tracker_parser/test_golden_arc*.py](tests/tracker_parser/test_golden_arc8.py), [test_v12_golden_arc10.py](tests/tracker_parser/test_v12_golden_arc10.py)). Only Arc 10 is a PASS verdict; it's a retrofit. Section 4-L validation has been exercised against Arc 10 (PASS-DEPLOYABLE) — the v1.2 config + §4 section both present.
- **Notes:** The first Wave 2 arc closing with a from-scratch PASS will be the parser's first never-before-seen-v1.2-PASS run. Idempotency + determinism tests are solid; expect WIRED-grade behaviour.

---

### Tracker

#### Parser handles all schema versions end-to-end
- **Status:** WIRED
- **Evidence:** Detection precedence v1.2 → v1.1 → v1.0 → error ([scripts/tracker_parser/README.md:51-57](scripts/tracker_parser/README.md:51)). Tests in [tests/tracker_parser/test_schema_detection.py](tests/tracker_parser/test_schema_detection.py) + [test_schema_v12.py](tests/tracker_parser/test_schema_v12.py) cover all three versions. 60 tests total per the README.
- **Notes:** v1.0 legacy field names are coerced to v1.1 names internally ([scripts/tracker_parser/README.md:60](scripts/tracker_parser/README.md:60)). Historical closures never rewritten.

#### Idempotency proven on a real Wave 1 closure
- **Status:** WIRED — proven on Arcs 8, 10, 11 (all v1.2 retrofits, but real arc closures)
- **Evidence:** `scripts/tracker_parser/parsed.log` contains sha256 entries for Arcs 8/10/11 (verified inline). Tests at [tests/tracker_parser/test_idempotency.py](tests/tracker_parser/test_idempotency.py) cover re-running same closure → no-op.
- **Caveat per README §"Bootstrap state":** Arc 11 was the seeding closure; Arcs 8 and 10 were manually backfilled to the Closed arcs summary row before parser ran. Per the README, re-running the parser on Arc 8 or 10's closure today would append a duplicate Closed arcs summary row. So idempotency is proven on Arc 11; Arcs 8/10 have a pending cleanup tracked in [results/re_evaluation_2026_05/SUMMARY.md](results/re_evaluation_2026_05/SUMMARY.md).

#### Determinism cross-platform (Linux CI vs Windows dev)
- **Status:** PARTIAL — single-platform two-run determinism proven; cross-platform NOT tested in CI
- **Evidence:** [tests/tracker_parser/test_determinism.py](tests/tracker_parser/test_determinism.py) covers byte-identical re-parse on same platform. CI runs `ubuntu-latest` only ([.github/workflows/ci.yml:7](.github/workflows/ci.yml:7)); no Windows job.
- **Notes:** `LINE_TERMINATOR = "\n"` is enforced ([core/determinism.py:35](core/determinism.py:35)); parser uses `lineterminator="\n"` and UTF-8 ([scripts/tracker_parser/README.md:112-114](scripts/tracker_parser/README.md:112)). The pieces are in place; observed-empirically cross-platform is not asserted.

---

### CI / engine determinism

#### Two-run sha256 reproduction enforced in CI
- **Status:** WIRED — at the unit / mini-pipeline level
- **Evidence:** [tests/test_determinism.py](tests/test_determinism.py) runs a mini pipeline twice and asserts panel + features + equity byte-identity. CI runs `pytest -m "not research"` ([.github/workflows/ci.yml:70](.github/workflows/ci.yml:70)) which includes `test_determinism.py`.
- **Notes:** Not exercised at the full-arc level (mini fixture only). End-to-end byte-identity for a real arc run (Arc 11 etc.) is asserted by the arc scripts themselves at pool-sha emission time, not as a CI gate.

#### Seed pinning across RF / LGBM / sampling / clustering
- **Status:** WIRED
- **Evidence:** `RANDOM_STATE = 42` baked into:
  - `seed_everything()` ([core/determinism.py:39-49](core/determinism.py:39))
  - `KMeans(random_state=42, n_init=10, max_iter=300)` ([core/steps/step_2_clustering.py:127-131](core/steps/step_2_clustering.py:127))
  - RF/LGBM/LR builders ([core/steps/_classifier_defaults.py](core/steps/_classifier_defaults.py))
  - `permutation_importance(random_state=42, n_jobs=1)` ([core/steps/step_4_extraction.py:175](core/steps/step_4_extraction.py:175))
  - All discovery / WFO scripts call `seed_everything(RANDOM_STATE)` at entry

#### `lineterminator='\n'` enforcement in CSV outputs
- **Status:** WIRED
- **Evidence:** `LINE_TERMINATOR = "\n"` ([core/determinism.py:35](core/determinism.py:35)). Every CSV write in the orchestrator + arc scripts passes `lineterminator="\n"` explicitly. Spot-checked: [core/arc/arc_orchestrator.py:295-313](core/arc/arc_orchestrator.py:295), [scripts/l_arc_11/run.py:284-488](scripts/l_arc_11/run.py:284).

#### Cross-platform line-ending stability
- **Status:** PARTIAL
- **Evidence:** Codebase consistently uses `\n` writes ([core/determinism.py:35](core/determinism.py:35), `write_text(..., newline="\n")` throughout). However the CI matrix is `ubuntu-latest` only — Windows-side line-ending behaviour is not asserted by an automated test. Local developer (Windows 11 per `git config`) relies on the explicit `newline="\n"` parameter being respected by Python's file API (it is, by spec).
- **Scope:** 1-2 hr CI addition — `windows-latest` runner matrix entry running `pytest -k determinism`.

---

### Cross-arc registries (tracker G, H, I)

#### Cross-arc cluster registry (G)
- **Status:** WIRED — populated correctly by parser across multiple arcs
- **Evidence:** [ARC_TRACKER.md:120-132](ARC_TRACKER.md:120) — 10 rows across Arcs 8/10/11. `tests/tracker_parser/test_mapping_A_through_K.py` covers the registry G mapping ([scripts/tracker_parser/mapping.py:G_…](scripts/tracker_parser/mapping.py)). Append-only per spec.

#### Cost-decomposition registry (H)
- **Status:** WIRED, but exercised only once
- **Evidence:** [ARC_TRACKER.md:145-147](ARC_TRACKER.md:145) — single row for Arc 11 (only classifier-based winner to date). Mapping logic at [scripts/tracker_parser/mapping.py:H_…](scripts/tracker_parser/mapping.py); tested in `test_mapping_A_through_K.py`.
- **Notes:** Schema correct; coverage is light because few arcs have classifier-based winners.

#### Cross-arc tag registry (I)
- **Status:** WIRED
- **Evidence:** [ARC_TRACKER.md:155-171](ARC_TRACKER.md:155) — 15 tags across Arcs 8/10/11. Increment logic in `mapping.py`; tested in `test_mapping_A_through_K.py`. Per-tag counts and arc lists correctly populated.

---

### Sub-protocols

#### `signal_discovery_probe` — engine path
- **Status:** PARTIAL — engine path exists and is documented as ready; sub-protocol registry is empty
- **Evidence:** `core.discovery` module ([core/discovery/](core/discovery/)) contains grammar, random_search, rule_engine, causal_filter, bonferroni, pool_simulator, quantile_grid, io, metrics — all the spec's required components. `scripts/arc_discovery_01/run_discovery.py` is the CLI ready to run on `configs/arc_discovery_01.yaml`. [tests/discovery/](tests/discovery/) has 8 test files covering search smoke, grammar, rule engine, determinism, etc.
- ARC_TRACKER.md states "infrastructure landed; full 10k run pending compute slot" — engine is ready for the 10k run.
- **PARTIAL:** Sub-protocol registry at [core/arc/sub_protocol.py:38](core/arc/sub_protocol.py:38) is `_REGISTRY: dict = {}`. `signal_discovery_probe` is not registered there. Invocation is via standalone CLI, not via `ArcOrchestrator`'s `sub_protocol` field. Per the protocol §5 spec the registry pattern is the canonical hook.
- **Scope:** Trivial (1-2 hr) — write `core/sub_protocols/signal_discovery_probe.py` exposing `step_1` callable that wraps `run_discovery.py`'s entrypoint; register at import. Optional — direct CLI invocation works today.

#### `heavy_ml_probe` — engine path
- **Status:** MISSING — spec-only at `docs/sub_protocols/heavy_ml_probe.md`
- **Evidence:** Grep `autogluon|FLAML|CoxPH|RandomSurvivalForest|survival` in `core/` returns zero matches. No `core/sub_protocols/heavy_ml_probe.py` exists. The spec defines AutoML inside CV, meta-labeling target, Cox PH / Random Survival Forest for Pipeline D.
- **Scope:** Multi-day engine PR (potentially week+) — bring in `autogluon-tabular` (or `flaml`) + `lifelines` (Cox PH) or `scikit-survival` (RSF) dependencies; implement AutoML-inside-CV wrapper; define meta-labeling target alongside cluster-membership target; wire into Step 4 sub-protocol path; extend A4 to consume survival predictions.
- **Blocks Phase 2:** Yes — Phase 2 expects heavy_ml_probe to be available for arcs whose vanilla Step 4 didn't extract.

---

## Out-of-band findings

### Orphaned / older infrastructure
- `core/backtester.py` + `core/backtester_helpers.py` + `core/d1_pipeline.py` are the pre-v3 KH-24 paths. Not deleted; superseded by `core/sim/multipair_backtester.py` for v3.0 arc work. CLAUDE.md notes the v3 runtime infrastructure landed via `infra/protocol-runtime-v3`. Not strictly dead — KH-24 legacy depends on these for the anchor reproduction harness.
- `attic/` directory — explicitly excluded from pytest via [.github/workflows/ci.yml:70](.github/workflows/ci.yml:70) `--ignore=attic`. Confirmed retired.
- `scripts/phase*.py` — pre-v3 phase scripts (KA, KG, KH, etc.). Not deleted; not exercised by current arc work.

### Documentation drift
- `core/arc/sub_protocol.py` docstring says "CC_07 ships the hook but no actual sub-protocols" ([core/arc/sub_protocol.py:11](core/arc/sub_protocol.py:11)). True for vanilla overseer integration; but discovery + heavy_ml_probe have spec docs at `docs/sub_protocols/`. Engine-side, only discovery has runnable code (via standalone script, not via the registry).
- `core/arc/arc_orchestrator.py:9-14` lists steps 1→5 + lazy Step 6 — matches L_PROTOCOL §2.
- Arc 11's run.py docstring documents the `ArcOrchestrator._run_step_5` wiring gap directly ([scripts/l_arc_11/run.py:6-11](scripts/l_arc_11/run.py:6)). Treat as a load-bearing comment that should outlive Arc 11 — it accurately describes a real engine gap.

### Anchor reproduction
- `scripts/anchor/check_a1_equivalence.py` — workstation-runnable harness for KH-24 A1-equivalence on real 28-pair data. Documented in CLAUDE.md as PASS for the v3 anchor (A1 path byte-identical to legacy `KH24FoldRunner` under matched warmup convention).
- `tests/protocol_runtime/test_kh24_a1_equivalence.py` — small-fixture proof of structural equivalence. CI-gated.
- Anchor preservation invariant per L_PROTOCOL §8 (KH-24 worst-fold ROI +1.92%, DD 6.37% within ±0.5pp / ±1pp tolerance) is operationally validated; no automated CI check asserts the tolerance numerically (the small fixture is too small for a meaningful ROI/DD comparison).

### Risk-sizing primitives
- `LiveBalanceRisk` ([core/sim/risk/live_balance.py](core/sim/risk/live_balance.py)) is equity-pct sizing. `ResetFloorRisk` ([core/sim/risk/reset_floor.py](core/sim/risk/reset_floor.py)) is reset-floor sizing. Architecture configs choose between them — but Amendment 3 §"Sizing convention" mandates FAIL on equity_pct without chat approval, and that gate is not enforced.

### Determinism baseline applied per-arc
- `random_state=42`, `n_jobs=1`, `lineterminator="\n"` per CLAUDE.md "Conventions" are followed throughout the codebase. Arc 11's run script calls `seed_everything(RANDOM_STATE)` at entry. The convention is documented, tested at unit level, and respected at module level.

---

## Recommendations

### Tier 1 — Blocks Wave 2 dispatch (highest priority)
1. **Rewrite `core/wfo/gates.py` for Amendment 3.** Implement scalability bounds, `k_safe/k_hard/r_safe/r_hard`, priority-ordered failure modes, sizing-convention check. Multi-day engine PR. *Depends on per-day max-DD + chained max DD emission landing first.*
2. **Add per-day max-DD parquet emission to engine.** New artefact `step_5/per_day_max_dd_base.parquet` with day-start-equity-derived per-day DD. Half-day engine PR.
3. **Add chained max DD emission across IS + holdout trajectory.** Continuous-equity tracker spanning all folds + holdout. Half-day engine PR.
4. **Add holdout re-run at scaled risk.** Per top-K candidate, re-run holdout at `r_safe` (DEPLOYABLE) and `r_hard` (VIABLE). 1-2 hr engine PR (sits on top of (1)).
5. **Fix Step 4 lineage filter column-name mismatch.** Either rename `feature_lineage_dataframe`'s output column from `lineage` to `causal_lineage`, OR fix `_filter_lineage` to accept `lineage` as the column name. Trivial; should ship before any Wave 2 arc whose closure claims lineage enforcement.
6. **Fix `ArcOrchestrator._run_step_5` to plumb `run_context`.** Extend `ArcConfig` with `run_context: A1RunContext | None`, pass to `ArcFoldRunner`. 1-2 hr engine PR. Unblocks A2/A6 via the orchestrator (today only the arc-script bypass works).
7. **Step 4 holdout-window training filter + classifier persistence.** CC_12 PR in flight covers both. Land it before any Wave 2 arc relies on the orchestrator's Step 4.

### Tier 2 — Blocks Phase 2 sub-protocols
1. **Build `heavy_ml_probe` engine path.** AutoML inside CV (autogluon or FLAML), meta-labeling target, Cox PH / RSF for Pipeline D. Multi-day-to-week engine PR. Phase 2 prerequisite.
2. **Register `signal_discovery_probe` in the sub-protocol registry.** Wrap `scripts/arc_discovery_01/run_discovery.py` as a `step_1` override and register at import. 1-2 hr engine PR; cosmetic for Phase 2 readiness (the standalone CLI already works for the 10k run).

### Tier 3 — Blocks specific Wave-1-in-flight arc work
1. **Per-fold retrain orchestration for A3 / A4.** Required if any Wave 2 arc dispatches Pipeline DE or Pipeline D exits. Half-day to multi-day engine PR.
2. **Verify A4 trail-vs-classifier-exit precedence (UNKNOWN).** Read `MultiPairBacktester`'s bar-evaluation order and confirm SL fires before classifier exit when both trigger same bar. 1-2 hr read.
3. **Add `ArcOrchestrator` auto-feature-matrix construction.** Inside `_run_step_1`, build the feature matrix via `compute_feature_matrix` after pool builds — drop the manual hand-roll from arc scripts. Half-day engine PR. Pure ergonomics.
4. **Complete integrity report: D1-lag NaN-perturbation + spread-floor activation + KH-24 co-fire.** L_PROTOCOL §2 Step 1 lists these. Each is one function. 2-3 hr engine PR total.

### Tier 4 — Step 6 framework
1. **Build generic Step 6 byte-compare harness.** Factor Arc 10's `step_6_byte_compare.py` into a feature-producer-agnostic harness over the registered `FeatureSpec` set. Multi-day engine PR.
2. **Add Step 6 invocation trigger on PASS verdict.** Flip `ArcConfig.invoke_step_6` based on gate outcome; bundle with (1).
3. **Lock Step 6 manifest format.** Define `step_6/audit_report.md` + `manifest.json` schema. Half-day docs PR alongside the engine work.

### Tier 5 — Closure infrastructure
1. **Extend orchestrator's closure-writer to emit v1.2 `§1 tracker_payload` YAML.** Keyed off `ArcOrchestratorResult` fields. Multi-day engine PR. Per-architecture §4 deployment_spec emitters are the bulk of the work.

### Tier 6 — Quality of life / housekeeping
1. **Add `windows-latest` runner to CI matrix** for the determinism test subset. 1-2 hr CI PR.
2. **Add full-data anchor preservation check** at L_PROTOCOL §8 tolerance (±0.5pp ROI, ±1pp DD on KH-24 worst-fold). Currently only structural A1 equivalence is CI-gated; numerical anchor invariant is asserted by chat at major engine PRs. Multi-day data-dependent PR (the small-fixture in CI is too small).
3. **Add F1-best threshold sweep to Step 4.** Trivial.
4. **Backfill cross-arc registries for Arcs 8 and 10.** Per parser README §"Bootstrap state", they have summary rows but not full A-K contributions. Cleanup tracked in `results/re_evaluation_2026_05/SUMMARY.md`. Operational cleanup, not engine work.

---

## Appendix — Capability count breakdown by step (refreshed 2026-05-25)

| Step / area | WIRED | PARTIAL | MISSING | UNKNOWN | Total | Delta vs original |
|---|---:|---:|---:|---:|---:|---|
| Step 1 — Plumbing | 4 | 2 | 0 | 0 | 6 | unchanged |
| Step 2 — Clustering | 4 | 0 | 0 | 0 | 4 | unchanged |
| Step 3 — Capturability | 3 | 0 | 0 | 0 | 3 | unchanged |
| Step 4 — Extraction | 8 | 1 | 0 | 0 | 9 | +3 W, −3 P (PR #185) |
| Step 5 — Architectures + WFO | 10 | 1 | 0 | 0 | 11 | +6 W, −5 P, −1 U (PR #186, #195) |
| Amendment 3 — Risk-normalised gates | 6 | 0 | 0 | 0 | 6 | +6 W, −6 M (PR #186, PR #197 / Amendment 6 boundary) |
| Step 6 — Causal audit | 5 | 0 | 0 | 0 | 5 | +5 W, −5 M (PR #188 / Amendment 4) |
| Closure infrastructure | 1 | 3 | 0 | 0 | 4 | unchanged (operational, not engine) |
| Tracker | 2 | 1 | 0 | 0 | 3 | unchanged |
| CI / determinism | 3 | 1 | 0 | 0 | 4 | unchanged |
| Cross-arc registries | 3 | 0 | 0 | 0 | 3 | unchanged |
| Sub-protocols | 0 | 1 | 1 | 0 | 2 | unchanged (heavy_ml_probe in flight via PR #187 parallel chat) |
| **Other (anchor, sizing primitives)** | 4 | 3 | 1 | 0 | 8 | +3 W via PR #186 sizing-convention gate |
| **NEW** Signal-level EET alignment (PR #193) | 1 | 0 | 0 | 0 | 1 | new capability family |
| **NEW** EET session semantics (PR #197 / Amendment 6) | 3 | 0 | 0 | 0 | 3 | new capability family (distance, reset_floor, compute_per_day_max_dd) |
| **NEW** Canonical exit-policy registry (PR #195) | 4 | 0 | 0 | 0 | 4 | new capability family (registry + Account partial-fill + driver wiring + per-arc migration) |
| **NEW** Amendment 5 dispatch-time selection (PR #194) | 1 | 0 | 0 | 0 | 1 | documented; engine unchanged |
| **Total** | **56** | **9** | **2** | **0** | **77** | +26 W, −15 P, −11 M, −1 U, +9 new caps |

End of audit body. Post-audit append sections below capture per-PR detail.

---

## Post-audit update — 2026-05-24 (L_PROTOCOL Amendment 4 / CC_17)

§"Step 6 — Causal audit" capability statuses transition MISSING → WIRED:

1. **Producer-level feature trace as a runnable check** — WIRED at `core/step_6/lookahead.py`. Per-feature `causal_lineage` enforcement via the §6.1 `per_feature_lineage_clean` critical check.
2. **Byte-compare from raw OHLC for feature reproduction** — WIRED at `core/step_6/byte_compare.py` (generic harness factored from Arc 10's `step_6_byte_compare.py`). Invoked by §6.1 `byte_compare_no_drift`.
3. **D1 lag rule verification check** — WIRED at `core/step_6/lookahead.py:_check_d1_lag_rule` (static source-inspection check).
4. **Automated trigger on PASS verdict** — WIRED at `core/step_6/dispatch.py:maybe_dispatch_step_6` + `core/arc/arc_orchestrator.py` `run()`. Post-gate per chat Q1.
5. **Manifest / report artefact format** — WIRED at `core/step_6/manifest.py` + `core/step_6/artefacts.py`. Closure template v1.3 + parser v1.3 + tracker "Step 6 audit registry" section per dispatch Tasks 4-6.

Capability count delta: Step 6 row updates from `0 WIRED / 0 PARTIAL / 5 MISSING` to `5 WIRED / 0 PARTIAL / 0 MISSING`. Overall: WIRED 30 → 35, MISSING 13 → 8.

Remaining Tier 1 / 2 items (Amendment 3 engine emission, heavy_ml_probe sub-protocol, etc.) unchanged by this PR.

---

## Post-audit update — 2026-05-25 (Signal-level EET timezone alignment)

§"Data / aggregator" capability — **Signal-level EET alignment**: MISSING → **WIRED**.

PR #189 closed engine-level EET aggregation but did not audit signal modules.
Several signal + feature modules used UTC-anchored HTF lookup idioms (`.floor("4h")`,
`.normalize() + Timedelta(days=1) + merge_asof`, `.normalize() + searchsorted`)
that produced State C (empty pool) or State B (silent wrong-value) outputs under
the 5ers EET storage convention.

WIRED via:
- [core/signals/htf_alignment.py](../../core/signals/htf_alignment.py) — canonical utility
  with `get_htf_value_at` / `get_htf_row_at` / `get_htf_index_at`. Byte-identical
  to legacy KH-24 idiom under UTC; correct prior-EET-day alignment under EET.
- Fixed modules: `core/strategies/kh24/{signal,exits/kijun_d1,filters/d1_regime}.py`,
  `core/signals/mtf_alignment_2_down_mixed_kijun.py` (restored from origin/arc/l_arc_5),
  `core/features/multi_tf.py`, `signals/lchar_d1atr_top_decile.py`, `signals/lchar_dlr_long.py`.
- Tests: 19 unit tests + 14 regression tests including a static guard that
  flags `.floor()` / `.normalize()` reintroduction in fixed modules.
- Lint rule: pre-commit `forbid-utc-anchor-in-signal-modules` hook.

Per-arc impact:
- **Arc 3, Arc 5, Arc 10** verdicts suspect — re-run needed under EET post-merge.
- **Arcs 4, 7, 8, 9, 11** unaffected (single-TF signals).
- **KH-24** live deployment unaffected (uses UTC convention); latent landmine fixed.

Full audit: [docs/audits/signal_module_eet_audit_2026_05.md](signal_module_eet_audit_2026_05.md).

Open follow-ups: OPEN-RESET-FLOOR-EET (`core/sim/risk/reset_floor.py`),
OPEN-FEATURES-DISTANCE-EET-SESSION-SEMANTICS (`core/features/distance.py`) — both
resolved by CC_20 (next section).

---

## Post-audit update — 2026-05-25 (CC_20 / Amendment 6 — EET session semantics)

Two EET-fault-class items resolved:

- **`OPEN-FEATURES-DISTANCE-EET-SESSION-SEMANTICS`** — RESOLVED.
  [core/features/distance.py](../../core/features/distance.py)
  `_prior_session_high` / `_prior_session_low` now bucket bars by
  trading day via the canonical `core.time_utils.session_boundary.utc_to_eet_trading_day`
  utility, keyed on `panel.boundary_convention`. Default `panel=None`
  preserves UTC behaviour for legacy callers.

- **`OPEN-RESET-FLOOR-EET`** — RESOLVED (forward hygiene).
  [core/sim/risk/reset_floor.py](../../core/sim/risk/reset_floor.py)
  `ResetFloorAccount.update_at_day_close` uses the same utility;
  default `boundary_convention="5ers_eet"`. Note: the read-first
  phase of CC_20 surfaced that this module is dormant in v3 runtime
  (not instantiated by any architecture). The fix is forward
  hygiene; the load-bearing analogous fix is in
  `compute_per_day_max_dd` (see Amendment 6 below).

**Newly tracked fix (CC_20 chat-decided):**

- **`OPEN-COMPUTE-PER-DAY-MAX-DD-EET`** — RESOLVED.
  [core/runners/_fold_stats_helpers.py](../../core/runners/_fold_stats_helpers.py)
  `compute_per_day_max_dd` now accepts `boundary_convention`
  (default `"5ers_eet"`); the orchestrator forwards
  `panel.boundary_convention`. This is the load-bearing fix —
  `daily_dd_breaches_at_r_safe` / `daily_dd_breaches_at_r_hard`
  now bucket equity by the EET trading day matching 5ers' actual
  reset boundary.

**Amendment 6** (CC_20 PR) supersedes Amendment 3 §"Boundary" —
the locked `UTC broker-day. Locked value.` framing is amended to
`EET broker trading day. Locked value (version-amended).`
Amendment 6 text lands in the parallel L_PROTOCOL.md docs PR;
engine implementation lands here.

**KH-24 anchor preservation:** KH-24 runs `convention="utc"`
end-to-end. Verified byte-identical via
[tests/protocol_runtime/test_kh24_a1_equivalence.py](../../tests/protocol_runtime/test_kh24_a1_equivalence.py)
+ [tests/replays_v2_1_1/](../../tests/replays_v2_1_1/).

**Plumbing:** convention propagates via `Panel.boundary_convention`
(mirrors `Panel.tf`). Default `"utc"` for legacy safety.

See [PROTOCOL_RUNTIME.md §15.5](../PROTOCOL_RUNTIME.md) for the
session-semantics convention contract.

---

## Post-audit update — 2026-05-25 (Canonical exit-policy registry)

§"Backtester / sim" capabilities — **Exit-policy registry**: MISSING → **WIRED**
for the full canonical policy catalogue.

Pre-PR state: only `sl_only` was tacitly handled by the engine (default
behaviour). The other six policies named in L_PROTOCOL §10 lived as
hand-rolled post-hoc simulators inside `scripts/l_arc_*/step_5.py`,
producing silent drift between arcs (the dispatch's "hand-rolled per-arc
exit logic is creating silent drift" mandate). No canonical engine
support existed for partial-fill semantics, so `sl_partial_close_1r_runner_trail`
(Arc 10's load-bearing PASS-DEPLOYABLE exit) had no first-class home.

WIRED via [core/sim/exit_policies/](../../core/sim/exit_policies/):

- `_base.py` — `ExitPolicy` ABC + `ExitPolicyContext` / `ExitPolicyDecision`
  / `ExitAction` / `ExitPolicyState` / `NullPolicyState`.
- `_registry.py` — `build_exit_policy(name)` + `available_policies()`.
- Six policy modules, one each:
  - `sl_only.py`
  - `sl_plus_tp_2r.py`, `sl_plus_tp_3r.py`
  - `sl_plus_trailing_atr.py`, `sl_plus_trailing_swing.py`
  - `sl_partial_close_1r_runner_trail.py`
- `path_simulate.py` — replay surface for legacy Step 5 fast path
  (`simulate_path` + `simulate_pool_approximation`).

Account partial-fill: [core/sim/account.py](../../core/sim/account.py)
gains `partial_close`, `current_size_of`, `ClosedTrade.parent_position_id`,
and `_current_sizes` shadow dict. Position stays frozen.

Driver wiring: [core/sim/multipair_backtester.py](../../core/sim/multipair_backtester.py)
gains `exit_policy_manager: ExitPolicyManager | None` field + per-bar
`evaluate_intrabar_for_all` (before intra-bar SL/TP with same-bar SL
suppression) + `evaluate_at_close_for_all` (after trail-manager ratchet).
Order schema gains `exit_policy: str | None` + `sl_atr_mult: float | None`.

Architecture wiring: A1/A2/A3/A4/A6 configs gain `exit_policy: str | None = None`.
KH-24 (a1_adapter) unchanged — defaults to None.

Per-arc migration: [scripts/l_arc_10_v3/step_5.py](../../scripts/l_arc_10_v3/step_5.py)
and [scripts/l_arc_8/run_step5_wfo.py](../../scripts/l_arc_8/run_step5_wfo.py)
`_apply_exit_policy` bodies replaced by one-line delegates to the canonical
registry. 150 LOC of hand-rolled per-policy branches deleted from Arc 10.
Byte-identical parity asserted by
[tests/sim/exit_policies/test_path_simulate_reference_parity.py](../../tests/sim/exit_policies/test_path_simulate_reference_parity.py)
(218 cases: 6 policies × 4 SL multipliers × 9 scenarios).

KH-24 anchor regression: 37/37 KH-24-specific tests + 105/105
protocol_runtime tests pass on the branch. Full-data HistData anchor
(scripts/anchor/check_a1_equivalence.py) requires chat-side run.

Capability count delta:
- "Live exit policy registry" row updates from `1 WIRED (sl_only implicit) /
  0 PARTIAL / 6 MISSING` to `7 WIRED / 0 PARTIAL / 0 MISSING`.
- "Account partial-fill semantics" row: MISSING → WIRED.
- "Per-arc exit-policy hand-rolling" row (anti-capability): WIRED →
  MISSING (deleted; the goal).

Overall: WIRED 35 → 41 (Steps + registry policies count separately).
Per-arc silent-drift surface area: closed.

---

## Post-Phase-1-engine-build summary — items resolved across PRs #185-#197

> CC_22 docs refresh (2026-05-25) consolidates the eight-PR sprint into a single per-PR resolution map. Use this as the index when reading the per-section status flips above.

| PR | Topic | Items resolved (in this audit) |
|---|---|---|
| **#184** | Engine capability audit (this doc) | Baseline enumeration. Original WIRED/PARTIAL/MISSING counts: 30 / 24 / 13. |
| **#185** | Step 4 fitted-classifier persistence + holdout-window training fix | Step 4 §"Holdout-window training filtering" (PARTIAL → WIRED); §"Fitted classifier persistence" (PARTIAL → WIRED); §"Lineage filter additional finding" (PARTIAL bug → WIRED via column-name reconcile). Unblocks A2 / A6 admit-gate honesty. |
| **#186** | L_PROTOCOL Amendment 3 implementation + A3/A4 wiring + Step 4/5 fixes | All 6 Amendment-3 items (MISSING → WIRED): chained max DD, per-day max-DD parquet, priority-ordered gates, holdout re-run at scaled risk, sizing-convention check, failure-mode taxonomy. Step 5: A2 (PARTIAL → WIRED via orchestrator run_context plumbing), A3 (PARTIAL → WIRED via per-fold retrain orchestration), A4 (PARTIAL → WIRED, same mechanism), A6 (PARTIAL → WIRED). |
| **#188** | Step 6 causal audit framework + Amendment 4 + parser v1.3 + closure template v1.3 | All 5 Step-6 items (MISSING → WIRED): producer-level feature trace, byte-compare harness, D1 lag verification, automated PASS-trigger, manifest/report format. Closure template v1.3 + parser v1.3 Phase 2 tightening. |
| **#189** | Signal parity engine (mid features + 5ers EET bar boundaries + worst-case fills) | New capability family: engine output venue-independent. Mid-anchored features; EET bar boundaries opt-in via `boundary_convention="5ers_eet"`; worst-case fills (long ask / short bid). UTC default preserves KH-24 anchor. |
| **#193** | Signal-level EET timezone audit + canonical alignment utility | New capability family: signal-module timezone correctness. Canonical `core/signals/htf_alignment.py`; 8 modules fixed; 33 new tests + pre-commit lint rule. Per-arc impact: Arcs 3 / 5 / 10 verdicts re-run-eligible under EET. |
| **#194** | Amendment 5 — AUC-gated A2/A6 architecture selection + parser v1.3.1 field | New capability family (documentation only — engine unchanged): four-gate dispatch-time architecture selection (Gate 1 archetype; Gate 2 AUC ≥ 0.65; Gate 3 universal A1; Gate 4 portfolio if ≥ 2 clusters). Closure template v1.3.1 `architectures_skipped_by_amendment_5` field. |
| **#195** | Canonical exit-policy registry + `sl_partial_close_1r_runner_trail` primitive + per-arc migration | New capability family: 6 registered exit policies + Account partial-fill semantics (`partial_close`, `current_size_of`, `ClosedTrade.parent_position_id`) + driver wiring + per-arc migration (Arc 10 v3, Arc 8). 218 byte-identical reference-parity tests. A4 same-bar trail-vs-classifier precedence locked (trail wins). |
| **#197** | EET session semantics: distance.py + reset_floor.py + `compute_per_day_max_dd` + Amendment 6 | New capability family: 3 convention-aware consumers wired via `core/time_utils/session_boundary.py` + `Panel.boundary_convention`. Amendment 6 supersedes Amendment 3 §"Boundary" — daily-DD measurement is now 5ers EET broker trading day (load-bearing fix in `compute_per_day_max_dd`). UTC opt-in preserved for KH-24 anchor byte-identity. |

### What is still missing

After PR #197, only two real engine capabilities remain MISSING:

1. **`heavy_ml_probe` engine path** — spec-only at `docs/sub_protocols/heavy_ml_probe.md`. Phase 2 prerequisite. In flight via PR #187 (parallel chat); PR-B/C/D/E/F to follow.
2. **A5 portfolio composition** (PARTIAL — architecture exists; no arc has surfaced ≥ 2 candidate clusters at Step 3 to exercise it). Deferred until first VIABLE candidate emerges.

Operational housekeeping remains (closure-writer auto-generation, Windows-runner CI matrix, integrity-report check completeness, Arc 8 closure §10 correction) but these don't block Wave 2 dispatch.

### Wave 2 dispatch readiness

✅ **All Tier-1 Wave-2-blocking items cleared.** Amendment 3 engine emission, gate rewrite, classifier persistence + holdout-window training, A3 / A4 per-fold orchestration, A2 / A6 orchestrator plumbing — all WIRED. Wave 2 arcs can dispatch with full Amendment 3 + 4 + 5 + 6 emission from first run.

End of refresh.
