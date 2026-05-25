# Engine Capability Audit — 2026-05-23

> Read-only enumeration of which L_PROTOCOL v3.0 (+ Amendments 1/2/3) capabilities are wired end-to-end, partially wired, or missing.
> Methodology: every claim cites the file + arc/test that exercised it (WIRED), or the file + scope estimate (PARTIAL/MISSING).
> No engine code changed.
> CC_12 (Step 4 holdout-window training filter + classifier persistence) is in flight — items it covers are classified PARTIAL against current `main`.
> Wave 1 = Arcs 5, 7 in flight + 8, 10, 11 closed. Wave 2 (not dispatched yet) = Arcs 4 RERUN, 4, 6, 3, 1, 2. Phase 2 = sub-protocols + 6 new signal classes.

---

## Executive summary

- **Total capabilities audited:** 68
- **WIRED:** 30
- **PARTIAL:** 24
- **MISSING:** 13
- **UNKNOWN:** 1 (A4 trail-precedence semantics under exit predicate — needs runtime inspection)

### Blocks Wave 2 dispatch (must clear before Wave 2 arcs run cleanly)

1. **Amendment 3 engine emission (MISSING).** No engine site emits `step_5/per_day_max_dd_base.parquet`, `chained_max_dd_base_pct`, scaled-risk holdout re-runs, `k_safe` / `r_safe` / `k_hard` / `r_hard`, or the sizing-convention check. The tracker parser schema knows the field names (`scripts/tracker_parser/schema.py:80+`) but no engine writes them. Wave 2 arcs producing PASS candidates will land closures that are PROVISIONAL by construction, as Arc 10's already is ([results/l_arc_10/ARC_CLOSURE.md](results/l_arc_10/ARC_CLOSURE.md): "Engine re-run with continuous-equity tracking + per-day max-DD emission still recommended for definitive measurement").
2. **`core/wfo/gates.py` is pre-Amendment-3.** `classify_fold_stats` evaluates raw-risk only; no scalability bounds, no priority-ordered failure modes, no `step5_not_scalable` / `step5_chained_dd_above_gate` emission ([core/wfo/gates.py:120-193](core/wfo/gates.py:120)). Until this is rewritten, every closure must hand-derive risk-normalised verdicts (Arc 10's §10 re-evaluation table is the manual workaround).
3. **Step 4 lineage filter silent-bypass (PARTIAL).** `_filter_lineage` checks for column `causal_lineage` ([core/steps/step_4_extraction.py:94](core/steps/step_4_extraction.py:94)); `feature_lineage_dataframe` emits column `lineage` ([core/features/pipeline.py:62](core/features/pipeline.py:62)). Result: every Arc using `compute_feature_matrix` (Arc 11 confirmed) silently skips the lineage filter and trains on every feature regardless of tag. Bug; not a CC_12 item.
4. **Step 4 holdout-window training filter + classifier persistence (PARTIAL, CC_12 in flight).** Step 4 trains on full pool via 5-fold TimeSeriesSplit without restricting to `train_end ≤ 2020-12-31` ([core/steps/step_4_extraction.py:222-265](core/steps/step_4_extraction.py:222)); returns only `best_classifier: str`, not a fitted model. Arc 11 works around both by manually prefitting on `entry_time < 2021-01-01` ([scripts/l_arc_11/run.py:191-218](scripts/l_arc_11/run.py:191)). CC_12 PR will land both fixes.
5. **`ArcOrchestrator._run_step_5` doesn't plumb `run_context` (PARTIAL).** Orchestrator's `_runner` constructs `ArcFoldRunner` without passing `run_context` ([core/arc/arc_orchestrator.py:174-181](core/arc/arc_orchestrator.py:174)), so A2/A6 (which require `per_trade_features` via `A1RunContext`) silently run as no-admit baselines. Arc 11 bypasses by invoking `run_search` directly ([scripts/l_arc_11/run.py:1-32](scripts/l_arc_11/run.py:1)). Documented as a closure cross_arc_tag `canonical_orchestrator_step5_run_context_gap`.
6. **Per-fold retrain orchestration for A3/A4 (MISSING).** A3 and A4 accept a pre-fit `PathClassifierFit` via `arch_config` ([core/architectures/a3_pipeline_de.py:78](core/architectures/a3_pipeline_de.py:78), [core/architectures/a4_pipeline_d_exits.py:67](core/architectures/a4_pipeline_d_exits.py:67)). Per Amendment 2 §"A3" / §"A4" a NEW classifier must be trained per fold (not reused across folds). No driver code instantiates the per-fold retrain loop — only the pre-fit-once pattern exists. Any Wave 2 arc dispatching A3 or A4 against the v3.0 architecture set would need this orchestration first.

### Blocks Phase 2 sub-protocols

1. **`heavy_ml_probe` engine path (MISSING).** `docs/sub_protocols/heavy_ml_probe.md` specifies autogluon / FLAML AutoML inside CV, meta-labeling target, survival-models (Cox PH / RSF) for Pipeline D. Grep for `autogluon|FLAML|CoxPH|RandomSurvivalForest|survival` in `core/` returns zero matches. Spec-only.
2. **`signal_discovery_probe` is not integrated via the sub-protocol registry (PARTIAL).** Engine path exists at `core/discovery/` + `scripts/arc_discovery_01/run_discovery.py` and is documented as ready for the 10k run ([ARC_TRACKER.md:14](ARC_TRACKER.md:14): "Step 1 — infrastructure landed; full 10k run pending compute slot"). But the sub-protocol registry at `core/arc/sub_protocol.py` is empty (`_REGISTRY: dict[str, Mapping[str, StepOverride]] = {}` — no `signal_discovery_probe` registered). Sub-protocols invoke via standalone scripts today, not via the `ArcOrchestrator` Protocol §5 hook.

### Blocks specific Wave-1-in-flight arc work

1. **`ArcOrchestrator` doesn't auto-build the feature matrix.** `ArcConfig.feature_matrix` and `feature_lineage` are required external inputs ([core/arc/arc_orchestrator.py:69-70](core/arc/arc_orchestrator.py:69)). Arcs 5 and 7 in flight must hand-roll feature-matrix construction (as Arc 11 did at [scripts/l_arc_11/run.py:138-178](scripts/l_arc_11/run.py:138)). Cosmetic friction, not a hard blocker.
2. **Integrity report missing 3 of 6 protocol checks.** `core/arc/integrity.py` emits pool_size, per_pair_distribution, coverage_window, lookahead_spot (declared-lineage only — no actual trace), determinism. **Missing:** D1-lag NaN-perturbation test, spread-floor activation rate per pair, KH-24 co-fire rate. L_PROTOCOL §2 Step 1 lists all six. Arcs in flight report these by hand (Arc 10 Step 1 banner mentions "3/3 D1-lag NaN-perturbation" — done outside the integrity report).

### Housekeeping / non-blocking

- Closure-writer template `core/arc/_closure_template.py` emits a generic skeleton ([core/arc/_closure_template.py:37-88](core/arc/_closure_template.py:37)) — does NOT emit v1.2 `§1 tracker_payload` YAML block, `best_architecture` fields, or `§4 deployment_spec`. Every closure to date is hand-written or retrofitted by arc scripts (Arcs 8/10/11). PARTIAL.
- CI runs `ubuntu-latest` only (no Windows runner) — cross-platform byte-identical reproduction is not tested in CI ([.github/workflows/ci.yml:7](.github/workflows/ci.yml:7)). Two-run sha256 within a single platform IS exercised by [tests/test_determinism.py](tests/test_determinism.py).
- Step 6 framework is essentially MISSING as protocol-level infrastructure. Arc 10's Step 6 was hand-written + had an arc-specific byte-compare script ([scripts/l_arc_10_v3/step_6_byte_compare.py](scripts/l_arc_10_v3/step_6_byte_compare.py)).

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

#### Holdout-window training filtering (CC_12 in flight)
- **Status:** PARTIAL — being addressed in CC_12 PR (in flight)
- **Evidence:** `run_step_4` does NOT restrict training data to IS window. It accepts trades and runs `TimeSeriesSplit` over the entire sorted-by-`entry_time` pool ([core/steps/step_4_extraction.py:222](core/steps/step_4_extraction.py:222)). The 5-fold split's final fold's TEST partition includes holdout-window trades.
- **Arc-side workaround:** Arc 11 pre-fits the deployment classifier on `entry_time < 2021-01-01` ([scripts/l_arc_11/run.py:191-218](scripts/l_arc_11/run.py:191)) AFTER Step 4 has been run on the full pool. So Step 4 metrics are reported on a CV that includes holdout, but the actual deployed classifier (A2/A6 input) is trained only on IS.
- **Scope:** Half-day engine PR — add `is_end: date | None` parameter to `run_step_4`, filter trades by `entry_time <= is_end`, propagate through call sites. CC_12 already includes this.

#### Fitted classifier persistence (CC_12 in flight)
- **Status:** PARTIAL — being addressed in CC_12 PR (in flight)
- **Evidence:** `Step4Result.per_cluster[*].best_classifier` is `str` ("rf"/"lgbm"/"lr") not the fitted model object ([core/steps/step_4_extraction.py:60-72](core/steps/step_4_extraction.py:60)). Arc 11 manually rebuilds the classifier via `prefit_classifier` ([scripts/l_arc_11/run.py:191-218](scripts/l_arc_11/run.py:191)) — wasteful (re-trains on a separately-constructed IS subset) and a hand-rolled bypass.
- **Scope:** Half-day engine PR — extend `ClusterExtraction` to hold the fitted model + return alongside metrics. CC_12 already includes this.

#### Feature-importance pipeline end-to-end
- **Status:** WIRED
- **Evidence:** `Step4Result.feature_importance` aggregates per (cluster, classifier, fold, feature) rows; mean + std emitted; orchestrator writes `step_4/feature_importance.csv` ([core/arc/arc_orchestrator.py:311-313](core/arc/arc_orchestrator.py:311)). Confirmed in Arc 11's `step_4_per_cluster.top_10_features` field ([scripts/l_arc_11/run.py:586-589](scripts/l_arc_11/run.py:586)).

#### Lineage filter (additional finding)
- **Status:** PARTIAL — bug in current `main` (NOT a CC_12 item)
- **Evidence:** `_filter_lineage` checks `"causal_lineage" not in lineage.columns` ([core/steps/step_4_extraction.py:94](core/steps/step_4_extraction.py:94)). The pipeline emits column name `lineage` ([core/features/pipeline.py:62](core/features/pipeline.py:62)). When the column-name check fails, the filter returns `feature_matrix, cols, ()` — all features accepted, zero excluded.
- The lineage test ([tests/protocol_runtime/test_step_4_extraction.py:58-72](tests/protocol_runtime/test_step_4_extraction.py:58)) passes because it hand-builds a DataFrame with `causal_lineage` column — the test never exercises the production pipeline's column-name shape.
- **Scope:** Trivial (one-line rename in either site). Should land alongside or before CC_12.

---

### Step 5 — WFO architecture search

#### A1 system_level_filter
- **Status:** WIRED
- **Evidence:** `core.architectures.a1_system_level_filter.A1Architecture` ([core/architectures/a1_system_level_filter.py](core/architectures/a1_system_level_filter.py)). KH-24 equivalence test [tests/protocol_runtime/test_kh24_a1_equivalence.py:test_kh24_runs_through_a1_without_error](tests/protocol_runtime/test_kh24_a1_equivalence.py) passes. Exercised by Arcs 5, 8, 10, 11 (CLAUDE.md). A1 is the verdict-carrying architecture for Arc 10's PASS-DEPLOYABLE.
- **Notes:** [scripts/anchor/check_a1_equivalence.py](scripts/anchor/check_a1_equivalence.py) is the workstation-runnable full-data anchor reproduction harness; result documented at CLAUDE.md ("Full-data anchor regression: A1 path byte-identical to legacy `KH24FoldRunner`").

#### A2 classifier_filter
- **Status:** PARTIAL — architecture respects Amendment 2 (no Step 5 retrain), but the orchestrator's `_run_step_5` doesn't pass `run_context` to the runner. CC_12 in flight for the Step 4 classifier-persistence half.
- **Evidence:** `A2Architecture.run` accepts a `run_context: A1RunContext | None` ([core/architectures/a2_classifier_filter.py:168](core/architectures/a2_classifier_filter.py:168)). Classifier loaded from `arch_config.classifier` (no Step 5 retrain) ([core/architectures/a2_classifier_filter.py:111-125](core/architectures/a2_classifier_filter.py:111)) — matches Amendment 2. Synthetic test [tests/protocol_runtime/test_architectures_synthetic.py:test_a2_admit_gate_with_dummy_classifier](tests/protocol_runtime/test_architectures_synthetic.py) passes via direct construction. Arc 11 ran A2 via the bypass driver ([scripts/l_arc_11/run.py](scripts/l_arc_11/run.py)).
- **Orchestrator gap:** `ArcOrchestrator._run_step_5` constructs `ArcFoldRunner` without `run_context` ([core/arc/arc_orchestrator.py:176-181](core/arc/arc_orchestrator.py:176)). Result: A2 has no `per_trade_features` lookup → every signal is rejected (`ctx.per_trade_features is None` short-circuit at [core/architectures/a2_classifier_filter.py:111](core/architectures/a2_classifier_filter.py:111)).
- **Scope:** Trivial (1-2 hr) — extend `ArcConfig` to carry `run_context` and pass through `_run_step_5`. Arc 11's closure surfaces this as cross_arc_tag `canonical_orchestrator_step5_run_context_gap`.

#### A3 pipeline_de — per-fold retraining
- **Status:** PARTIAL — architecture file accepts pre-fit classifier; no engine site builds per-fold `PathClassifierFit` for it
- **Evidence:** `A3Config.classifier_fit: PathClassifierFit` ([core/architectures/a3_pipeline_de.py:78](core/architectures/a3_pipeline_de.py:78)). The architecture's `run(...)` uses the supplied fit verbatim — no retraining at runtime. Amendment 2 §"A3" requires "NEW classifier per fold".
- No driver code instantiates the per-fold retrain orchestration. No arc has exercised A3 end-to-end.
- `core.features_path_so_far.PATH_FEATURE_KEYS` ([core/features_path_so_far.py](core/features_path_so_far.py)) plus `core.architectures._path_classifier.fit_path_classifier` exist for the training side, but no caller composes them into a per-fold loop.
- **Scope:** Half-day to multi-day engine PR — write `A3FoldPrep` that builds per-fold `PathClassifierFit` from the IS pool of each fold's IS window, and a search wrapper that swaps configs per fold. The architecture itself is fine.
- **Blocks Wave 2:** Yes, if any Wave 2 arc dispatches A3.

#### A4 pipeline_d_exits — per-fold exit-policy variation
- **Status:** PARTIAL — same shape as A3; architecture accepts pre-fit, no driver builds per-fold
- **Evidence:** `A4Config.classifier_fit: PathClassifierFit` ([core/architectures/a4_pipeline_d_exits.py:67](core/architectures/a4_pipeline_d_exits.py:67)). Architecture wraps A1 mechanics with an additional `ExitPredicate` that consults the classifier per bar.
- **UNKNOWN:** Trail vs predicate precedence under simultaneous fire — header says "SL precedence is preserved by the driver's intra-bar exits-first ordering" but I have not verified the driver's bar-evaluation order against the predicate path. Needs deeper inspection.
- **Scope:** Same as A3, plus verify trail-vs-classifier-exit precedence in `MultiPairBacktester`.

#### A5 portfolio_composition
- **Status:** PARTIAL — architecture exists; no arc has exercised it
- **Evidence:** `A5Architecture` ([core/architectures/a5_portfolio_composition.py](core/architectures/a5_portfolio_composition.py)) consumes ≥ 2 `StrategyResult` constituents; combines closed trades and equity curves; recomputes DD on combined curve. Synthetic test [tests/protocol_runtime/test_architectures_synthetic.py:test_a5_combines_two_constituents](tests/protocol_runtime/test_architectures_synthetic.py) passes.
- No arc has surfaced 2+ candidate clusters at Step 3, so A5 has never been exercised in production. Per-architecture win rate at [ARC_TRACKER.md:65](ARC_TRACKER.md:65): "A5 portfolio_composition: 0 / 0".
- **Notes:** Amendment 3 §"A5 follow-up flag" defers combined-portfolio DD constraints. A5 itself can run; the gate logic for VIABLE-tier combination is the open work.

#### A6 meta_labeling
- **Status:** PARTIAL — same gap as A2 (orchestrator doesn't plumb `run_context`); architecture respects Amendment 2
- **Evidence:** `A6Architecture.run` ([core/architectures/a6_meta_labeling.py](core/architectures/a6_meta_labeling.py)) loads classifier from `arch_config.classifier` (no Step 5 retrain) — matches Amendment 2. `_confidence_to_multiplier` maps `prob < lower → 0×`, `lower ≤ prob < upper → 0.5×`, `prob ≥ upper → 1×` ([core/architectures/a6_meta_labeling.py:71-78](core/architectures/a6_meta_labeling.py:71)) — matches Amendment 2 sizing spec.
- A6 was exercised by Arc 11 (via direct driver, FAIL on holdout) and Arc 8 (best architecture, FAIL) — Arc 11's closure marks A6 as "best of FAIL" rather than a real win.
- **Scope:** Inherits Arc 11's `canonical_orchestrator_step5_run_context_gap` fix.

#### WFO 11-fold IS evaluation
- **Status:** WIRED
- **Evidence:** `core.wfo.folds.build_v3_folds` produces 11 anchored-expanding-IS folds (2010-01-01 → 2020-12-31) plus a holdout fold ([core/wfo/folds.py:99-160](core/wfo/folds.py:99)). `run_search` iterates eligible folds (≥ 365 IS days) per [core/wfo/orchestrator.py:85-99](core/wfo/orchestrator.py:85). Exercised by Arcs 8, 10, 11.

#### Holdout one-shot evaluation
- **Status:** WIRED (mechanics) / PARTIAL (Amendment 3 re-run at scaled risk)
- **Evidence:** `run_holdout` evaluates each top-K candidate exactly once on `structure.holdout` ([core/wfo/orchestrator.py:114-146](core/wfo/orchestrator.py:114)). One-shot enforced by orchestrator; selection-bias laundering blocked by construction.
- **Amendment 3 holdout re-run at `r_safe`/`r_hard`:** NOT implemented. Arc 10's PASS-DEPLOYABLE relies on hand-derived scaling.
- **Scope:** Half-day engine PR alongside the Amendment 3 gate logic rewrite.

#### Oracle WFO per cluster
- **Status:** WIRED
- **Evidence:** `core.runners.oracle_fold_runner` exists ([core/runners/oracle_fold_runner.py](core/runners/oracle_fold_runner.py)); Arc 8/10 produced `step_5/wfo_oracle.csv` artefacts ([results/l_arc_10/step_5/wfo_oracle.csv](results/l_arc_10/step_5/wfo_oracle.csv) exists). Arc 10's closure §"Post-closure research findings" cites the oracle Sharpe 4.61 vs base −1.29 gap from this runner.

#### Architecture ranking emission
- **Status:** WIRED
- **Evidence:** `WfoSearchResult.top_k` sorted by `(worst_fold_ratio, mean_fold_ratio)` descending ([core/wfo/orchestrator.py:101-111](core/wfo/orchestrator.py:101)). `step_5/architectures_ranked.md` written by arc scripts (Arc 8/10/11) and by `ArcOrchestrator._build_architectures_table` ([core/arc/arc_orchestrator.py:335-351](core/arc/arc_orchestrator.py:335)).

---

### Amendment 3 — Risk-normalised gates

> **All items in this section are MISSING in the engine.** The tracker parser schema knows the field names ([scripts/tracker_parser/schema.py:71-105](scripts/tracker_parser/schema.py:71)) but no engine code computes them. Arc 10's PASS-DEPLOYABLE is the hand-derived workaround.

#### Chained max DD emission across IS + holdout trajectory at `r_base`
- **Status:** MISSING
- **Evidence:** Grep `chained_max_dd|chained_dd` in `core/` returns zero hits. `core/runners/_fold_stats_helpers.py:max_drawdown_pct` ([core/runners/_fold_stats_helpers.py:39-45](core/runners/_fold_stats_helpers.py:39)) computes per-fold DD only; no continuous-equity chained DD across folds.
- Arc 10's closure §10 says this directly: "Chained DD not measured (forwarded out of Step 6 scope per main #178). Per-fold equity reset means cumulative cross-fold DD is unknown."
- **Scope:** Half-day engine PR — modify `run_search` / `run_holdout` to emit a continuous-equity trajectory artefact alongside per-fold; chained max DD = peak-to-trough across the concatenated curve.

#### Per-day max-DD parquet emission
- **Status:** MISSING
- **Evidence:** No code site writes `step_5/per_day_max_dd_base.parquet`. `count_daily_5pct_breaches` ([core/runners/_fold_stats_helpers.py:48-56](core/runners/_fold_stats_helpers.py:48)) counts breaches but does NOT persist the per-day series. Day-start equity definition (00:00 broker-day, NOT reset-floor baseline) per Amendment 3 §"Daily DD measurement" is not implemented anywhere.
- **Scope:** Half-day engine PR — add `emit_per_day_max_dd()` to the fold runner, resample equity to daily bars, compute `day_max_dd_pct = (day_open - day_min) / day_open`, persist parquet at `step_5/per_day_max_dd_base.parquet` per fold. Day-start-equity vs reset-floor distinction requires a separate equity tracker.

#### Verdict logic with priority-ordered gate evaluation
- **Status:** MISSING — `core/wfo/gates.py` is pre-Amendment-3
- **Evidence:** `classify_fold_stats` ([core/wfo/gates.py:65-193](core/wfo/gates.py:65)) evaluates at raw risk only. The order is: trade count → daily breaches → max DD → PASS-DEPLOYABLE / PASS-VIABLE / FAIL. **No scalability check, no `step5_not_scalable`, no `step5_chained_dd_above_gate`, no `step5_daily_dd_breach` (new Amendment 3 mode), no scaled-ROI / scaled-ratio failure modes.**
- The failure-mode taxonomy at Amendment 3 §"Failure-mode taxonomy" is NOT emitted by `classify_fold_stats`. Closures fill `primary_failure_mode` by hand.
- **Scope:** Multi-day engine PR — full rewrite of `gates.py`. Inputs: per-fold metrics at `r_base`, chained max DD trajectory, per-day max-DD series. Outputs: `k_safe`, `k_hard`, `r_safe`, `r_hard`, scaled metrics, priority-ordered failure mode, both-tier verdict.

#### Holdout re-run at scaled risk (`r_safe` / `r_hard`)
- **Status:** MISSING
- **Evidence:** `run_holdout` evaluates top-K at the base risk of their configs; no scaling logic, no second sim at `r_safe` per Amendment 3 §"Engine-side changes" item 4.
- **Scope:** 1-2 hr engine PR — wrap each candidate's config with risk-scaled twin(s) before holdout fold-run.

#### Sizing-convention gate check
- **Status:** MISSING
- **Evidence:** No code site reads `sizing_convention` field. `LiveBalanceRisk` ([core/sim/risk/live_balance.py](core/sim/risk/live_balance.py)) is the live-balance (equity-pct) implementation; `ResetFloorRisk` ([core/sim/risk/reset_floor.py](core/sim/risk/reset_floor.py)) is reset-floor. The choice is config-driven but no gate FAILs on `equity_pct` per Amendment 3 §"Scalability bounds" requirement.
- **Scope:** Trivial — add `sizing_convention` to fold-runner output; gate FAIL on `equity_pct` unless chat-approved override flag.

#### Failure-mode taxonomy emission
- **Status:** MISSING (engine-side); WIRED (tracker schema-side)
- **Evidence:** `scripts/tracker_parser/schema.py` enumerates `step5_not_scalable`, `step5_chained_dd_above_gate`, `step5_daily_dd_breach`, `step5_wf_roi_below_gate_after_scaling`, `step5_ratio_below_gate_after_scaling`, `step5_trade_count_below_gate`, `step5_negative_folds` as valid `primary_failure_mode` enum values. But the engine never assigns any of them — closures fill the field by hand.
- **Scope:** Bundled with the `gates.py` rewrite above.

---

### Step 6 — Causal audit

> Per the dispatch's expectation, this section is mostly MISSING. Step 6 is "lazy" by L_PROTOCOL §2 — runs only on PASS verdicts — but no generic framework exists for it.

#### Producer-level feature trace as a runnable check
- **Status:** MISSING (as protocol-level infrastructure); WIRED (as arc-specific hand-written audit for Arc 10)
- **Evidence:** No `core/step_6/` directory. `core.arc.arc_orchestrator` stubs Step 6 as "(lazy — deferred to chat at PASS verdict)" ([core/arc/arc_orchestrator.py:251](core/arc/arc_orchestrator.py:251)). Arc 10's audit ([results/l_arc_10/step_6/audit_report.md](results/l_arc_10/step_6/audit_report.md), [results/l_arc_10/step_6/causal_audit_report.md](results/l_arc_10/step_6/causal_audit_report.md)) is hand-written.

#### Byte-compare from raw OHLC for feature reproduction
- **Status:** MISSING (as protocol-level infrastructure); WIRED (arc-specific for Arc 10)
- **Evidence:** [scripts/l_arc_10_v3/step_6_byte_compare.py](scripts/l_arc_10_v3/step_6_byte_compare.py) is Arc-10-bespoke. It samples 5 trades, recomputes features from raw H4/D1 OHLC, byte-compares to pool. Output: `byte_compare_log.json`. Generic framework doesn't exist.
- **Scope:** Multi-day engine PR — factor Arc 10's byte-compare into a feature-producer-agnostic harness over the registered `FeatureSpec` set.

#### D1 lag rule verification check
- **Status:** MISSING (as Step 6 invocation); WIRED (in producer-side feature code)
- **Evidence:** D1 lag is enforced at the producer level via `_build_d1_lag1_series` ([core/features/multi_tf.py:32](core/features/multi_tf.py:32)) but no Step 6 check independently verifies it on a random trade sample.

#### Automated trigger on PASS verdict
- **Status:** MISSING
- **Evidence:** `ArcConfig.invoke_step_6: bool = False` ([core/arc/arc_orchestrator.py:74](core/arc/arc_orchestrator.py:74)) — orchestrator does not flip this flag based on Step 5 verdict. Arc 10's PASS-DEPLOYABLE Step 6 was triggered by chat dispatch, not by the engine.
- **Scope:** Trivial post-`gates.py` rewrite — flip `invoke_step_6` when any candidate clears PASS-tier non-Step-6 constraints.

#### Manifest / report artefact format
- **Status:** MISSING (no locked format); WIRED (Arc 10 invented one)
- **Evidence:** Arc 10's reports have a structure (winner inventory, producer-level trace per feature, byte-compare, regime invariance, verdict) but it's hand-written, not template-driven.
- **Scope:** Bundled with the framework PR.

---

### Closure infrastructure

#### From-scratch v1.2 closure existence
- **Status:** MISSING — every v1.2 closure in `results/` is a retrofit
- **Evidence:** All 3 v1.2 closures ([results/l_arc_8/ARC_CLOSURE.md](results/l_arc_8/ARC_CLOSURE.md), [results/l_arc_10/ARC_CLOSURE.md](results/l_arc_10/ARC_CLOSURE.md), [results/l_arc_11/ARC_CLOSURE.md](results/l_arc_11/ARC_CLOSURE.md)) were retrofitted from earlier versions per recent commits (`#181 Closure template v1.2 + Arc 8/10/11 deployment spec retrofit`). No arc has been opened, run, and closed entirely against the v1.2 template — no fresh PASS-DEPLOYABLE has gone through the v1.2 §4 deployment_spec gate as part of normal arc workflow.
- **Scope:** Operational discipline — Wave 2 arcs land as the first from-scratch v1.2 examples. No code work needed.

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

## Appendix — Capability count breakdown by step

| Step / area | WIRED | PARTIAL | MISSING | UNKNOWN | Total |
|---|---:|---:|---:|---:|---:|
| Step 1 — Plumbing | 4 | 2 | 0 | 0 | 6 |
| Step 2 — Clustering | 4 | 0 | 0 | 0 | 4 |
| Step 3 — Capturability | 3 | 0 | 0 | 0 | 3 |
| Step 4 — Extraction | 5 | 4 | 0 | 0 | 9 |
| Step 5 — Architectures + WFO | 4 | 6 | 0 | 1 | 11 |
| Amendment 3 — Risk-normalised gates | 0 | 0 | 6 | 0 | 6 |
| Step 6 — Causal audit | 0 | 0 | 5 | 0 | 5 |
| Closure infrastructure | 1 | 3 | 0 | 0 | 4 |
| Tracker | 2 | 1 | 0 | 0 | 3 |
| CI / determinism | 3 | 1 | 0 | 0 | 4 |
| Cross-arc registries | 3 | 0 | 0 | 0 | 3 |
| Sub-protocols | 0 | 1 | 1 | 0 | 2 |
| **Other (anchor, sizing primitives)** | 1 | 6 | 1 | 0 | 8 |
| **Total** | **30** | **24** | **13** | **1** | **68** |

End of audit.

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
  trading day via the canonical `core.utils.session_boundary.utc_to_eet_trading_day`
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
