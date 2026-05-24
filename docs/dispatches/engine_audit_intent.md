# Engine Capability Audit — Intent

> **Dispatch:** CC — Engine Capability Audit (read-only)
> **Date opened:** 2026-05-23
> **Worktree branch:** `claude/zealous-ardinghelli-401c35` (cut from `main`; harness-named — actual PR title will be `[AUDIT] Engine capability audit 2026-05` per dispatch)
> **Deliverable:** `docs/audits/engine_capability_audit_2026_05.md`
> **Status:** intent drafted — end turn for chat review before audit begins.

---

## §1 Read-first confirmation

Read in full before drafting this intent:

1. **`L_PROTOCOL.md`** (678 lines) — current v3.0 of record. Amendments 1 (Step 5 search policy), 2 (ML mechanics for A2/A3/A4/A6), and 3 (risk-normalised gates) are landed inline; Amendment 3 full text also archived at `archive/L_PROTOCOL_v3_0_AMENDMENT_3.md`. §1 non-negotiables, §2 Steps 1-6, §3 gates, §5 sub-protocol mechanism, §6 closure docs, Appendix A/B all read.
2. **`docs/templates/ARC_CLOSURE_TEMPLATE.md`** v1.2 (locked 2026-05-23). §1 `tracker_payload` schema with v1.0/v1.1/v1.2 field map. §4 `deployment_spec` required for any PASS-* verdict. Parser at `scripts/update_tracker_from_closure.py` does v1.2 PASS-verdict validation per §"Section 4-L".
3. **`archive/L_PROTOCOL_v3_0_AMENDMENT_3.md`** (282 lines) — engine-side actions itemised in §"Engine-side changes" + §"Action items": per-day max-DD parquet, chained max DD emission, verdict priority logic, holdout re-runs at `r_safe`/`r_hard`, sizing-convention check.

No other docs consulted to form this intent. Direct file-system reconnaissance done only to enumerate the capability checklist categories (see §3 below).

---

## §2 Scope and methodology

**Read-only.** No engine code, config, or protocol file is modified. The only file written is the audit report at `docs/audits/engine_capability_audit_2026_05.md` (new file; `docs/audits/` directory does not exist yet — will be created).

**Classification per capability:**

- **WIRED** — code exists AND has been exercised end-to-end through the canonical orchestrator path on a real arc (not just unit-tested in isolation). Evidence: file path + arc name / closure doc / passing integration test.
- **PARTIAL** — code exists but has not been exercised end-to-end through real arc work. Latent bugs likely (the Arc 5 / A2 pattern from CC_12). Evidence: file path + scope estimate to harden.
- **MISSING** — protocol spec exists, engine does not. Evidence: protocol section citation + scope estimate.
- **UNKNOWN — needs deeper inspection** — escape hatch for items that cannot be classified in ~15 min. Flagged with what would clarify them.

**No fixes proposed inline.** PARTIAL / MISSING items get a one-line scope estimate (trivial / 1-2 hr / half-day / multi-day) and what they block (Wave 1 / Wave 2 / Phase 2 / housekeeping). Prioritisation in §"Recommendations" of the audit report.

**Evidence discipline:** every WIRED claim cites a specific file path AND a specific arc / test that exercised it. No claims of the form "looks wired" without a referenced proof point. If reconnaissance can't surface proof in ~15 min, the item drops to UNKNOWN with a flag.

---

## §3 Audit scope (capability checklist, per dispatch)

Twelve capability families, mirroring the dispatch's checklist verbatim. Per-capability findings will live under §"Detailed findings" of the audit report.

### 3.1 Step 1 — Plumbing
- Pool generation via `build_ex_ante_bounded_population` across 28 pairs
- Feature matrix with full v3.0 default feature space (price geometry, session, cross-pair, multi-TF, vol regime, distance, spread regime)
- Causal lineage tagging per feature
- D1 lag-1 enforcement
- Integrity report emission (pool size, gaps, spread-floor activation, D1-lag NaN-perturbation, lookahead spot-check, KH-24 co-fire, determinism sha256)

### 3.2 Step 2 — Clustering
- Path-shape K-means / HDBSCAN across K ∈ {2..6}
- Silhouette per K
- Archetype tag assignment (V-shape, Stepwise, Bimodal, Monotonic_up, Monotonic_down, Choppy, Unclassified)
- `step_2/cluster_assignments.parquet` artefact

### 3.3 Step 3 — Capturability
- Per-cluster reach_1R, MFE distribution, ww_pp
- Composite capturability score
- Archetype-aware ranking + candidate-cluster flag

### 3.4 Step 4 — Extraction
- Per-cluster RF / LGBM / Logistic training
- 5-fold TimeSeriesSplit CV + per-fold AUC
- Threshold sweep (AUC-best / F1-best)
- Top features by importance (permutation)
- Holdout-window training filtering (CC_12 in-flight bug — status post-CC_12)
- Fitted classifier persistence (CC_12 in flight)
- Feature-importance pipeline end-to-end

### 3.5 Step 5 — WFO architecture search
- A1 system_level_filter — confirmed WIRED via Arcs 5/8/10/11 in CLAUDE.md
- A2 classifier_filter — being fixed in CC_12 (post-CC_12 status check)
- A3 pipeline_de — per-fold retraining required; any arc exercised?
- A4 pipeline_d_exits — per-fold exit policy; any arc exercised?
- A5 portfolio_composition — runner existence
- A6 meta_labeling — depends on CC_12 A2 fix
- WFO 11-fold IS evaluation
- Holdout one-shot evaluation
- Oracle WFO per cluster
- Architecture ranking emission

### 3.6 Amendment 3 — Risk-normalised gates
- Chained max DD emission at `r_base` across IS + holdout
- Per-day max-DD parquet emission (`step_5/per_day_max_dd_base.parquet` per Amendment 3 §"Daily DD measurement")
- Verdict logic with priority-ordered gate evaluation (`step5_not_scalable` → scaled DD → scaled chained DD → scaled daily DD → ROI at `r_safe`)
- Holdout re-run at scaled risk (`r_safe` / `r_hard`)
- Sizing-convention gate check (FAIL on `equity_pct` without chat approval)
- Failure-mode taxonomy (`step5_not_scalable`, `step5_daily_dd_breach`, `step5_chained_dd_above_gate`, `step5_wf_roi_below_gate_after_scaling`, `step5_ratio_below_gate_after_scaling`, etc.)

### 3.7 Step 6 — Causal audit
- Producer-level feature trace runnable?
- Byte-compare regeneration from raw OHLC?
- D1 lag rule verification check?
- Automated trigger on PASS verdict?
- Manifest / report artefact format?

(Expected to be mostly MISSING per dispatch. Audit confirms and informs future Amendment 4 scope.)

### 3.8 Closure infrastructure
- Has a from-scratch v1.2 closure been written, or only retrofits (Arcs 8/10/11)?
- Closure-writer auto-generation tooling — any?
- `config_artefact_path` validation at closure time
- Parser v1.2 detection + validation against from-scratch v1.2 PASS closure

### 3.9 Tracker
- Parser handles all schema versions (v1.0, v1.1, v1.2) end-to-end
- Idempotency proven on a real Wave 1 closure (not Arcs 8/10/11 retrofits)
- Determinism cross-platform (Linux CI vs Windows dev)

### 3.10 CI / engine determinism
- Two-run sha256 reproduction enforced in CI
- Seed pinning across RF / LGBM / sampling / clustering
- `lineterminator='\n'` enforcement in CSV outputs
- Cross-platform line-ending stability

### 3.11 Cross-arc registries (tracker G, H, I)
- Cross-arc cluster registry (G) — populated across multiple arcs?
- Cost-decomposition registry (H) — populated for any classifier-based winning arch?
- Cross-arc tag registry (I) — increment logic verified?

### 3.12 Sub-protocols
- `signal_discovery_probe.md` engine path — ready for 10k local run?
- `heavy_ml_probe.md` — AutoML / meta-labeling / survival-models built, or spec-only?

---

## §4 Directories that will be inspected (read-only)

Confirmed present from initial reconnaissance:

```
core/
  arc/            arc_orchestrator.py, arc_pool_builder.py, integrity.py, signal_protocol.py, sub_protocol.py, _closure_template.py
  architectures/  a1..a6 + _path_classifier.py, _protocol.py
  steps/          step_2_clustering.py, step_3_capturability.py, step_4_extraction.py, _shape_tags.py, _classifier_defaults.py
  runners/        arc_fold_runner.py, oracle_fold_runner.py, _fold_stats_helpers.py
  wfo/            fold_runner.py, folds.py, gates.py, orchestrator.py
  features/       pipeline.py, registry.py, lineage.py, cache.py + 7 feature-class modules
  sim/            multipair_backtester.py, account.py, panel.py, risk/{reset_floor,live_balance}.py, exit_hooks.py, fill.py, trailing_stop.py
  discovery/      grammar.py, random_search.py, rule_engine.py, causal_filter.py, bonferroni.py, pool_simulator.py, io.py, metrics.py, quantile_grid.py
  spread/, data/, signals/, strategies/, manifest.py, determinism.py, parallel.py

scripts/
  l_arc_8/        run_step{1..5}_*.py (Arc 8 reference orchestration)
  l_arc_10_v3/    step_{1..5,6_byte_compare}.py + run_all.py (Arc 10 reference)
  l_arc_11/       run.py, build_summary.py, write_closure.py
  tracker_parser/ extract.py, mapping.py, schema.py, registry.py, rolling_state.py, tracker_io.py, parsed.log
  update_tracker_from_closure.py
  arc_discovery_01/  run_discovery.py, determinism_check.py (signal_discovery_probe exercise?)
  anchor/         run_anchor.py, check_a1_equivalence.py, bisect_warmup.py (anchor preservation)

tests/
  protocol_runtime/  test_arc_orchestrator_e2e.py, test_arc_pool_builder.py, test_architectures_synthetic.py,
                     test_kh24_a1_equivalence.py, test_step_{2,3,4}_*.py, test_signal_protocol.py, test_sub_protocol.py
  discovery/         test_search_smoke.py, test_grammar.py, test_rule_engine.py, test_determinism_artefacts.py
  (143 test files total — full directory will be scanned for capability proofs)

configs/             [will be enumerated]
docs/                BACKTESTER_ARCHITECTURE.md, PROTOCOL_RUNTIME.md, KH24_SYSTEM_LOCK.md, etc.
results/             l_arc_*/ closure docs and step artefacts for proof-of-exercise
archive/             L_PROTOCOL_v3_0_AMENDMENT_3.md (referenced)
ARC_TRACKER.md       parser output target
.github/             CI config (sha256 reproduction enforcement check)
```

Audit will read; will not modify any of the above.

---

## §5 Report structure

`docs/audits/engine_capability_audit_2026_05.md` will follow the dispatch's prescribed format:

```
# Engine Capability Audit — 2026-05-22
> Read-only enumeration. No code changes.

## Executive summary
- Total capabilities audited: <N>
- WIRED / PARTIAL / MISSING / UNKNOWN counts
- Blocks Wave 2 dispatch: <list>
- Blocks any current arc work: <list>

## Detailed findings
### 3.1 Step 1 — Plumbing
  Pool generation: STATUS / Evidence / Notes
  Feature matrix: ...
  (repeat per capability)
### 3.2 Step 2 ...
(through 3.12 Sub-protocols)

## Recommendations
Prioritised:
1. Blocks Wave 2 — highest
2. Blocks Phase 2 sub-protocols — high
3. Blocks specific future arc — medium
4. Quality-of-life / housekeeping — low

## Orphans / dead code / undocumented deps
(anything found outside the checklist; per dispatch "surface anything else the audit finds")
```

---

## §6 Discipline confirmations

- **No code changes.** Only file written: `docs/audits/engine_capability_audit_2026_05.md`.
- **No protocol amendments.** No edits to `L_PROTOCOL.md`, `CLAUDE.md`, `STATUS.md`, etc.
- **No closure docs touched.** Existing arc closures read for evidence only.
- **No PR merging.** PR opened titled `[AUDIT] Engine capability audit 2026-05`. Chat reviews + merges.
- **Evidence-anchored claims only.** Every WIRED gets file + arc/test. Every PARTIAL/MISSING gets file + scope.
- **UNKNOWN escape hatch.** Items not classifiable in ~15 min inspection flagged for deeper follow-up rather than guessed.
- **Out-of-band findings surfaced.** Orphaned modules, dead code, undocumented dependencies noted in a final report section.

---

## §7 Estimated effort

- Reading + cross-referencing across ~12 capability families: ~3-4 hours.
- Report drafting: ~1-2 hours.
- Iteration on chat review of intent: variable.

Audit will be done by reading code + tests + arc closure docs, NOT by running anything. No engine execution required.

---

## §8 Open questions for chat before audit begins

1. **CC_12 status.** Dispatch notes CC_12 (Step 4 holdout contamination fix + A2 wiring) is "in flight". Should A2 / Step 4 holdout-filter / classifier persistence be classified against:
   - the codebase as it stands on `main` right now (likely still PARTIAL pre-CC_12), or
   - the expected post-CC_12 state (anticipated WIRED)?
   Default reading: classify against current `main` state, with a flag noting "CC_12 lands → reclassify expected to WIRED". Confirm if different.

2. **Branch name.** Worktree is on harness-named branch `claude/zealous-ardinghelli-401c35`. Dispatch specifies `audit/engine-capability-2026-05`. Default reading: keep worktree branch (single-commit branch, throwaway post-merge per `§7` of L_PROTOCOL); PR title is `[AUDIT] Engine capability audit 2026-05` regardless. Confirm if you'd prefer me to rename / reset to the dispatched branch name.

3. **Wave 1 / Wave 2 definition.** Dispatch references "Wave 1 / Wave 2 / Phase 2" but I don't have a direct file pointer. Best understanding from CLAUDE.md + STATUS.md context: Wave 1 = Arcs 8/9/10/11 (Steps 1-3 closures); Wave 2 = whatever follows the v2.4 calibration packet (Arc 8/9/10/11 follow-ups + new Pipeline-D1 / classifier-rerun arcs). If a definitive Wave 1 / Wave 2 manifest exists elsewhere (TODO.md, RE_RUN_PLAN.md), point me to it before audit so the "blocks Wave 2" classification is correctly anchored.

4. **A2 / A6 lineage clarification.** Amendment 2 §"A2 (classifier_filter)" states "No retraining at Step 5 (use Step 4 output directly)". Amendment 2 §"A3" and §"A4" require NEW classifier training at Step 5 per-fold. Audit will mark per-fold-retraining requirement against each architecture and note whether engine respects it. Confirm: this matches your reading?

---

End of intent. Awaiting chat sign-off before any audit-report work begins.
