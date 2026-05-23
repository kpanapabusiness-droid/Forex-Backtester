# Combined Engine PR — Intent

> **Dispatch:** CC — Combined Engine PR (Amendment 3 implementation + A3/A4 wiring + Step 4/5 fixes)
> **Date opened:** 2026-05-23
> **Branch:** `engine/amendment-3-implementation-and-architectures` (cut from `main` at `e60ba26` — PR #185 base)
> **PR title at end:** `[ENGINE] Amendment 3 implementation + A3/A4 wiring + Step 4/5 fixes`
> **Expected size:** ~30-50 files modified, ~1500-2500 LOC. ~3-5 days of CC work.
> **Status:** intent drafted — end turn for chat review before code lands.

---

## §1 Read-first confirmation

Read in full before writing this intent:

1. **`docs/audits/engine_capability_audit_2026_05.md`** (the audit landed by PR #184). Scope confirmed against the audit's MISSING items in Amendment 3 (Step 5 §"Amendment 3 — Risk-normalised gates") + Step 6 (out of scope per dispatch — separate dispatch later) + A3/A4 PARTIAL items (Step 5 §"A3 pipeline_de" / §"A4 pipeline_d_exits") + the small fixes (Step 4 lineage column-name, A4 trail-vs-classifier-exit UNKNOWN). PR #185 already moved A2/A6 + orchestrator `_run_step_5` `run_context` plumbing PARTIAL → WIRED — confirmed via `core/arc/arc_orchestrator.py:260-328` post-PR-185.
2. **`L_PROTOCOL.md`** §2 Step 4 output list (post-PR-185 includes `classifiers/` artefacts) + §2 Step 5 ML mechanics + Amendment 3 inline + §3 gates. Read.
3. **`archive/L_PROTOCOL_v3_0_AMENDMENT_3.md`** §"Engine-side changes" enumerates the six MISSING items + §"Action items" §"Risks" §"Daily DD measurement". Read.
4. **`docs/templates/ARC_CLOSURE_TEMPLATE.md`** v1.2 §1 `tracker_payload.best_architecture` schema — every Amendment 3 risk-normalised field. Read.
5. **`core/steps/step_4_extraction.py`** post-PR-185 — `persistence_dir`, `arc_name`, `train_end` kwargs landed; `_filter_lineage` line 119 still checks for `"causal_lineage"` column (PR #185 did NOT touch lineage filter; bug confirmed still present per audit).
6. **`core/arc/arc_orchestrator.py`** post-PR-185 — `_AUTO_BUILDERS`, `AutoArchSpec`, `_resolve_train_end`, `_run_step_5(signal_eval, pool, s4)` with `A1RunContext(per_trade_features=…)` plumbed through. A3/A4 not yet in `_AUTO_BUILDERS` (intentional — they retrain per fold).
7. **`core/architectures/`** — `a1_system_level_filter.py` defines `A1RunContext` (used by A1/A2/A6); `a3_pipeline_de.py` + `a4_pipeline_d_exits.py` accept `PathClassifierFit` via `arch_config` (per-fold-train infrastructure NOT YET present); `_path_classifier.py` has `fit_path_classifier()` + `predict_admit()` but no per-fold orchestration; `a5_portfolio_composition.py` exists (admit-only composition, no classifier).
8. **`core/sim/multipair_backtester.py`** `_process_bar` order documented (steps 1-6). Group 4 (A4 trail-vs-classifier-exit precedence) — see §"Group 4" below for findings.

Branch base: cut from `origin/main` at `e60ba26 [ENGINE] Step 4 fitted-classifier persistence + holdout-window training fix (#185)`. PR #185 is the immediate prerequisite. Confirmed via `git log`.

---

## §2 Scope confirmation (matches dispatch)

Eight task groups. Order:

| Group | Title | Audit item | Status |
|---|---|---|---|
| 1 | Step 4 lineage column-name fix | PARTIAL bug, not CC_12 | Trivial |
| 2 | `_run_step_5` run_context plumbing for A3/A4 | PARTIAL (A1/A2/A6 done in PR #185) | Define `A3RunContext` + `A4RunContext`; extend `_AUTO_BUILDERS` OR new per-fold-prep hook |
| 3 | A3 / A4 architecture orchestration (per-fold retrain) | PARTIAL → WIRED | Per-fold `fit_path_classifier` orchestration + cost decomposition emission |
| 4 | A4 trail-vs-classifier-exit precedence | UNKNOWN → resolved | Current behaviour: predicate wins on tie. Per dispatch recommendation: flip to trail-first. Change is ~5 LOC + regression test |
| 5 | Amendment 3 engine emissions (6 items) | MISSING × 6 | Chained DD, per-day max-DD parquet, holdout re-runs at scaled risk, sizing-convention gate check, scaled gate logic, failure-mode taxonomy |
| 6 | Daily DD per-day re-evaluation | MISSING (part of Group 5) | Loader + per-day scaling + breach count at `r_safe` / `r_hard` |
| 7 | Integration tests | new | 6 named tests + full-pipeline test |
| 8 | Documentation | new | 5 doc updates + audit footer |
| 9 | Verify against Arcs 8/10/11 manual re-evaluation | new | Engine vs manual reconciliation; engine wins on tiebreak |

**Out of scope (per dispatch + audit):**
- Step 6 framework (separate dispatch after Amendment 4 spec)
- `heavy_ml_probe` build (Phase 2)
- Closure-writer auto-generation of v1.2 YAML block (Tier 5 housekeeping)
- Cross-platform CI (Tier 6 housekeeping)
- F1-best threshold sweep (Step 4 Tier 3 housekeeping)

---

## §3 Group-by-group findings + file paths + LOC estimates

### Group 1 — Step 4 lineage column-name fix

**Finding.** Confirmed bug still present post-PR-185: `core/steps/step_4_extraction.py:119` checks `"causal_lineage" not in lineage.columns`; `core/features/pipeline.py:62` (`feature_lineage_dataframe`) emits column `lineage` (not `causal_lineage`).

Result: any arc using `compute_feature_matrix(...).lineage` → `run_step_4(feature_lineage=...)` silently falls through the filter — every feature accepted, zero excluded. The existing test `tests/protocol_runtime/test_step_4_extraction.py:test_step_4_lineage_exclusion` passes only because it hand-builds a DataFrame with `causal_lineage` column.

**Resolution.** Pick the protocol verb: rename `feature_lineage_dataframe`'s output column from `lineage` → `causal_lineage`. L_PROTOCOL §1 non-negotiables read "causal lineage hint"; the column name `causal_lineage` matches the protocol term. The pre-existing test stays correct (it already uses `causal_lineage`); production callers (Arc 11's driver) get correct filtering.

**Files:**
- `core/features/pipeline.py` — rename column `lineage` → `causal_lineage` in `feature_lineage_dataframe()` (~5 LOC change)
- Grep + update any callers reading `lineage_df["lineage"]` (audit predicts 0-2 sites; confirm at implementation time)
- `tests/protocol_runtime/test_step_4_extraction.py:test_step_4_lineage_exclusion` — unchanged (already uses `causal_lineage`)
- New regression test: `tests/protocol_runtime/test_step_4_extraction.py:test_step_4_lineage_pipeline_integration` — calls `feature_lineage_dataframe()` → `run_step_4(...)` end-to-end with a SUSPECT feature; asserts exclusion. Catches the column-name mismatch.

**LOC:** ~10 engine + ~30 test.

### Group 2 — `_run_step_5` run_context plumbing for A3/A4

**Finding.** A1/A2/A6 already wired via `A1RunContext(per_trade_features=…)` in PR #185. A3/A4 currently consume `per_trade_entry_features` as a field on their `arch_config` (see `A3Config.per_trade_entry_features` and `A4Config.per_trade_entry_features`). They also consume a pre-fit `PathClassifierFit` via `arch_config.classifier_fit` — per-fold retraining is NOT in the architecture file; it must be orchestrated externally.

**Design call.** Two options:

A. **Per-architecture RunContext type** (`A3RunContext`, `A4RunContext`) that carries (a) per-fold `PathClassifierFit` + (b) `per_trade_entry_features`. ArcFoldRunner threads the appropriate `RunContext` per architecture; orchestrator builds them.

B. **`A1RunContext` becomes the universal context type** with optional fields; A3/A4 read `path_classifier_fit_per_fold` from it.

Recommendation: option B. Avoids context-type proliferation; keeps `ArcFoldRunner` signature stable. Add to `A1RunContext`: `path_classifier_fits: Mapping[int, PathClassifierFit] | None` (keyed by fold_id) and `per_trade_entry_features: Mapping[(pair, signal_time), Mapping[str, float]] | None`. A3/A4 read their data; A1/A2/A6 ignore.

**Files:**
- `core/architectures/a1_system_level_filter.py` — `A1RunContext` extended with `path_classifier_fits` + `per_trade_entry_features` optional fields (~10 LOC)
- `core/architectures/a3_pipeline_de.py` — runner reads `run_context.path_classifier_fits[fold.fold_id]` instead of `arch_config.classifier_fit`; reads `run_context.per_trade_entry_features` instead of `arch_config.per_trade_entry_features`. Both fields deprecated on the config (keep for backwards-compat with synthetic test) (~30 LOC)
- `core/architectures/a4_pipeline_d_exits.py` — same pattern (~30 LOC)
- `core/arc/arc_orchestrator.py` — `_AUTO_BUILDERS` extended with `"A3"` and `"A4"` entries; new per-fold-prep helpers `_build_per_trade_entry_features()` and `_build_path_classifier_fits_per_fold()` (~80 LOC)

**LOC:** ~150 engine + ~60 test.

**Question — chat decision-point:**
A3/A4 currently take `per_trade_entry_features` and `classifier_fit` via `arch_config`. The dispatch says "No silent fallback. If an architecture's required input is missing (e.g., A2 needs fitted classifier; if Step4Result has no classifier for the target cluster, error explicitly)." Open: should the deprecated `arch_config.classifier_fit` / `arch_config.per_trade_entry_features` paths be REMOVED entirely (cleaner), or kept for synthetic-test backwards-compat (less churn)? Default reading: remove from prod path; keep as `None`-default fields on the config for the synthetic test fixtures that still pass them directly. Confirm if different.

### Group 3 — A3 / A4 architecture orchestration (per-fold retrain)

**Finding.** `core/architectures/_path_classifier.py:fit_path_classifier` exists with `random_state=42`, `n_jobs=1` (via `build_rf()`), AUC-best threshold via Youden's J. Schema: `ENTRY_FEATURE_KEYS` (8) + `PATH_FEATURE_KEYS` (7) = 15 features.

What's missing:
- Per-fold orchestration that builds `(X, y)` for each fold's IS pool, fits, and emits `PathClassifierFit` keyed by `fold.fold_id`
- Cost decomposition (admit / reject / early-exit pool fractions + mean R)
- Per-fold seeds derived from `(arc_seed, fold_index, architecture_id)` — current `fit_path_classifier` always uses seed 42; need a determinism path that lets different folds get different seeds without losing reproducibility

**Design.** New module `core/steps/path_classifier_per_fold.py`:
```python
def build_path_classifier_fits_per_fold(
    *,
    pool_trades: pd.DataFrame,
    pool_paths: pd.DataFrame,
    cluster_assignments: pd.DataFrame,
    candidate_cluster_id: int,
    folds: tuple[Fold, ...],
    arch: Literal["A3", "A4"],
    n_defer: int = 5,  # A3 only
    arc_seed: int = 42,
) -> Mapping[int, PathClassifierFit]:
    """Build per-fold path-so-far classifiers for A3 / A4.

    For each fold:
      1. Restrict pool to trades with entry_time ∈ fold IS window
      2. Build (X, y): X = entry-features + path-so-far-at-n_defer (A3)
         or full-trade-path-features (A4); y = cluster_membership (A3)
         or final_r > 0 (A4)
      3. Seed = hash((arc_seed, fold.fold_id, arch))
      4. fit_path_classifier(X, y, ...) → PathClassifierFit
      5. Key result by fold.fold_id

    Holdout: separate one-shot call with IS = full training window.
    """
```

Cost decomposition emission (new field on `StrategyResult.metadata`):
- `admit_pool`: {n_fraction, mean_r}
- `reject_pool`: {n_fraction, mean_r}
- `early_exit_pool`: {n_fraction, mean_r}

Calculated post-fold in `core/architectures/a3_pipeline_de.py` and `a4_pipeline_d_exits.py` runners.

**Files:**
- `core/steps/path_classifier_per_fold.py` (NEW) — orchestration helper (~200 LOC)
- `core/architectures/a3_pipeline_de.py` — emit cost decomposition in `StrategyResult.metadata` (~50 LOC)
- `core/architectures/a4_pipeline_d_exits.py` — same (~50 LOC)
- `core/arc/arc_orchestrator.py` — wire `_AUTO_BUILDERS["A3"]` + `_AUTO_BUILDERS["A4"]` (~50 LOC)

**LOC:** ~400 engine + ~200 test (Group 7.2 + 7.3).

### Group 4 — A4 trail-vs-classifier-exit precedence

**Finding.** Read `core/sim/multipair_backtester.py:_process_bar()` (lines 295-324):

```python
def _process_bar(self, t, snapshot):
    # 1a. fill any closes queued at the prior bar's close
    self._fill_pending_closes(t, snapshot)
    # 1b. fill any entries pending from prior bar
    self._fill_pending_entries(t, snapshot)
    # 2. check intra-bar SL/TP exits + bar-close predicate exits
    #    (predicate hits go to _pending_closes for next-bar-open fill)
    self._check_exits(t, snapshot)            # ← predicate runs HERE, sets _pending_closes[pos_id] = reason
    # 3. update trailing stops at bar close AND queue trail-triggered
    #    closes for next-bar-open fill
    if self.trail_manager is not None:
        self.trail_manager.update_all_at_close(snapshot, self.account)
        trail_hits = self.trail_manager.trail_exit_triggers_at_close(snapshot, self.account)
        for pos_id in trail_hits:
            # Don't overwrite a predicate-driven exit that fired this bar
            self._pending_closes.setdefault(pos_id, "trailing_stop")   # ← setdefault WINS predicate
```

**Current behaviour:** intra-bar SL/TP fires FIRST (hard SL always wins). On same-bar tie between A4 classifier predicate + trail-stop, the **predicate currently wins** because step 2 runs first AND step 3's `setdefault` does not overwrite. The comment at line 309 ("Don't overwrite a predicate-driven exit that fired this bar") makes this an intentional design choice in the current code.

**Dispatch recommendation:** flip to trail-first ("matches typical real-world execution: SL hits before manual close on a fast move").

**Resolution.** Change `setdefault` → direct assignment at line 310. Document precedence as:
1. Intra-bar SL/TP (highest precedence)
2. Trail-stop at bar close (overrides predicate if both trigger)
3. Classifier exit predicate at bar close (only if no SL/TP/trail fired)

Add to `L_PROTOCOL.md` §2 Step 5 "Architecture-specific retraining policy" subsection: explicit precedence statement.

**Files:**
- `core/sim/multipair_backtester.py` — 1-line change (`setdefault` → `__setitem__`); update comment (~5 LOC)
- `L_PROTOCOL.md` — add precedence statement (~3 LOC)
- `tests/protocol_runtime/test_multipair_backtester.py` (NEW or extend existing) — synthetic bar with simultaneous trail-stop + A4-classifier-exit triggers; assert trail-stop wins (~80 LOC test)

**LOC:** ~10 engine + ~80 test.

### Group 5 — Amendment 3 engine emissions (the heavy group)

**Finding.** `core/wfo/gates.py:classify_fold_stats` is the pre-Amendment-3 gate logic. It evaluates only at `r_base` (no scaling), has no `step5_not_scalable` / `step5_chained_dd_above_gate` / `step5_daily_dd_breach` failure modes, no `sizing_convention` check, no `r_safe` / `r_hard` computation, no chained max DD, no per-day max-DD. The tracker parser schema knows the field names (`scripts/tracker_parser/schema.py:71-105`) but no engine writes them.

**Six sub-items per Amendment 3 §"Engine-side changes":**

#### 5.1 Chained max DD emission

**Design.** New module `core/wfo/chained_dd.py`:
```python
def compute_chained_max_dd(
    fold_equity_curves: Sequence[pd.Series],   # one per IS fold, OOS slice
    holdout_equity_curve: pd.Series | None,    # holdout OOS slice
    starting_balance: float,
) -> float:
    """Concatenate equity curves chronologically into one continuous
    curve; compute peak-to-trough max DD as positive percent.

    Per Amendment 3 §"Chained max DD measurement". When per-fold equity
    resets at fold start (current behavior), reconstruct continuous
    equity by chaining: end_equity_fold_k → start_equity_fold_k+1.
    """
```

Wire into `StrategyResult` aggregator at search-completion time. Emit `chained_max_dd_base_pct` per candidate.

**Files:**
- `core/wfo/chained_dd.py` (NEW) (~80 LOC)
- `core/wfo/orchestrator.py` — `run_search` returns chained DD per candidate (~30 LOC)
- `core/architectures/_protocol.py` — extend `StrategyResult` (~5 LOC)

#### 5.2 Per-day max-DD parquet emission

**Design.** New helper in `core/runners/_fold_stats_helpers.py`:
```python
def compute_per_day_max_dd(
    equity: pd.Series,
    starting_balance: float,
) -> pd.DataFrame:
    """For each UTC trading day in equity.index, compute:
      date, day_start_equity, day_min_equity, day_max_dd_base_pct,
      n_trades_open_start_of_day

    day_start_equity = equity at 00:00 UTC of that day (NOT reset-floor
    sizing baseline — that's the per-trade sizing reference; day-start
    equity is the daily-DD reference per Amendment 3 §"Daily DD
    measurement"/"Day-start equity definition").

    Boundary: UTC broker-day (locked).
    """
```

Persist to `results/<arc>/step_5/per_day_max_dd_base.parquet` (full IS + holdout trajectory). Columns: `date`, `pair_set`, `day_max_dd_base_pct`, `n_trades_open_start_of_day`, `day_start_equity`.

**Files:**
- `core/runners/_fold_stats_helpers.py` — add `compute_per_day_max_dd()` (~80 LOC)
- `core/wfo/orchestrator.py` — emit parquet at search completion (~30 LOC)

#### 5.3 Holdout re-run at scaled risk

**Design.** For each top-K candidate that's reached gate-stage:
1. Compute `k_safe = 8.0 / worst_fold_dd_base`, `k_hard = 10.0 / worst_fold_dd_base`
2. Build scaled-risk config: `r_safe = r_base × k_safe`, `r_hard = r_base × k_hard`
3. Re-run holdout sim at each scaled risk
4. Emit `holdout_roi_at_r_safe_pct`, `holdout_dd_at_r_safe_pct`, `holdout_roi_at_r_hard_pct`, `holdout_dd_at_r_hard_pct`
5. Two separate manifest entries — separate sha256 each (per Amendment 3 §5.5)

Scaling mechanism: change `risk_pct` in `arch_config` from `r_base` to scaled value. For per-architecture configs that use `LiveBalanceRisk(risk_pct=cfg.risk_pct)`, this propagates naturally.

**Files:**
- `core/wfo/orchestrator.py` — extend `run_holdout` to optionally do re-runs at scaled risks (~80 LOC)
- `core/wfo/holdout_rerun.py` (NEW) — config-rescaling helper (~50 LOC)

#### 5.4 Sizing-convention gate check

**Design.** New field `sizing_convention: Literal["reset_floor", "equity_pct"]` on every arch config (A1Config, A2Config, ..., A6Config). Gate logic FAILs the arc with `step5_not_scalable` if `equity_pct` is selected without an `accept_equity_pct: bool = False` chat-approval override on the arc config.

Default for L arc work: `reset_floor`.

**Files:**
- `core/architectures/a1_system_level_filter.py` ... `a6_meta_labeling.py` — add `sizing_convention` field to each config (~5 LOC × 6 = 30 LOC)
- `core/arc/arc_orchestrator.py` — `ArcConfig` gains `accept_equity_pct: bool = False` (~5 LOC)
- Gate logic (Group 5.5) consumes these fields

#### 5.5 Scaled gate logic — `core/wfo/amended_gates.py` (NEW)

**Design.** Replaces `core/wfo/gates.py:classify_fold_stats` (keep old fn for backwards-compat; new fn is `classify_amended_fold_stats`). Inputs:

- Per-fold `FoldStats` (existing — at `r_base`)
- `chained_max_dd_base_pct: float` (Group 5.1)
- `per_day_max_dd_df: pd.DataFrame` (Group 5.2)
- `holdout_stats_at_r_safe: FoldStats | None`, `holdout_stats_at_r_hard: FoldStats | None` (Group 5.3)
- `sizing_convention: str` + `accept_equity_pct: bool` (Group 5.4)
- `r_base: float`

Logic per Amendment 3 §"Failure-mode priority":

```python
def classify_amended_fold_stats(...) -> AmendedGateResult:
    worst_fold_dd_base = max(f.max_dd_pct for f in folds)

    # Compute scaling factors
    if worst_fold_dd_base <= 0:
        return _fail("step5_not_scalable", ...)  # zero DD → k = ∞

    k_safe = 8.0 / worst_fold_dd_base
    k_hard = 10.0 / worst_fold_dd_base
    r_safe = r_base * k_safe
    r_hard = r_base * k_hard

    R_MIN, R_MAX = 0.0015, 0.02
    scalable_to_safe = R_MIN <= r_safe <= R_MAX
    scalable_to_hard = R_MIN <= r_hard <= R_MAX

    if sizing_convention == "equity_pct" and not accept_equity_pct:
        return _fail("step5_not_scalable", ...)

    if not scalable_to_safe and not scalable_to_hard:
        return _fail("step5_not_scalable", ...)

    # Priority-ordered constraints per Amendment 3 §"Failure-mode priority":
    # 1. pool_too_small — caller-level (not here)
    # 2. step5_not_scalable — above
    # 3. step5_dd_above_gate — defensive (should not occur post-scaling)
    # 4. step5_chained_dd_above_gate
    # 5. step5_daily_dd_breach
    # 6. holdout_fail_after_is_pass
    # 7. step5_sign_consistency_fail / step5_negative_folds
    # 8. step5_trade_count_below_gate
    # 9. step5_wf_roi_below_gate_after_scaling
    # 10. step5_ratio_below_gate_after_scaling
    # (Step 6 causal_audit_fail = lazy, deferred)

    # Evaluate each in priority order; first fail wins
    ...
```

Emits new `AmendedGateResult` dataclass with all fields the tracker payload requires (`k_safe`, `k_hard`, `r_safe_pct`, `r_hard_pct`, `worst_fold_roi_at_r_safe_pct`, `chained_max_dd_at_r_safe_pct`, `daily_dd_breaches_at_r_safe`, `holdout_roi_at_r_safe_pct`, etc.).

**Files:**
- `core/wfo/amended_gates.py` (NEW) — verdict logic (~300 LOC)
- `core/wfo/gates.py` — keep `classify_fold_stats` for backwards compat; mark legacy in docstring (no LOC change)
- `core/wfo/orchestrator.py` — `run_search` + `run_holdout` use `classify_amended_fold_stats` instead of `classify_fold_stats` when `wfo_structure.holdout is not None` (~30 LOC)

#### 5.6 Failure-mode taxonomy emission

Enum extension. Tracker schema already knows the names — engine just needs to emit them.

**Files:**
- `core/wfo/amended_gates.py` — `Verdict` enum extended with all Amendment 3 modes (~10 LOC, bundled with Group 5.5)

**LOC for Group 5 total:** ~700 engine + ~300 test (covered in Group 7).

### Group 6 — Daily DD per-day re-evaluation

**Finding.** Per Amendment 3 §"Daily DD measurement", count-scaling is wrong. Correct procedure: load per-day parquet (Group 5.2), multiply each day's `day_max_dd_base_pct` by `k_safe` (or `k_hard`), count days where scaled DD ≥ 5%.

**Files:**
- `core/wfo/amended_gates.py` (in Group 5.5 module) — `_count_daily_breaches_at_scaled_risk(per_day_df, k) -> int` helper (~30 LOC)
- Verdict logic calls this helper for both tiers; emits `daily_dd_breaches_at_r_safe` / `_at_r_hard`

**LOC:** ~50 engine + ~50 test (Group 7.5).

### Group 7 — Integration tests

Per dispatch §"Group 7":

1. `tests/integration/test_amendment_3_gate_evaluation.py` — synthetic `FoldStats` + chained DD + per-day DD → verdict + `primary_failure_mode`. Covers all PASS / FAIL paths and every new failure mode (~250 LOC).
2. `tests/integration/test_a3_end_to_end.py` — Steps 1-5 on small fixture with A3; per-fold retraining (not classifier persistence); cost decomposition emitted (~150 LOC).
3. `tests/integration/test_a4_end_to_end.py` — same for A4; includes trail-vs-classifier precedence regression (~150 LOC).
4. `tests/integration/test_holdout_rerun_at_scaled_risk.py` — verdict triggers re-run; sha256 manifest has separate entries for `r_base`, `r_safe`, `r_hard` (~100 LOC).
5. `tests/integration/test_per_day_dd_breach_evaluation.py` — synthetic per-day parquet; verdict counts breaches at scaled risk; under known fixture assertions (~100 LOC).
6. `tests/integration/test_steps_1_through_5_end_to_end.py` — one command, complete v1.2 `tracker_payload` populated (~250 LOC).

**LOC:** ~1000 test (in addition to per-group tests above).

Note: `tests/integration/` is a NEW directory. Conftest setup is needed; will mirror `tests/protocol_runtime/conftest.py` pattern.

### Group 8 — Documentation

Per dispatch §"Group 8":

1. **`L_PROTOCOL.md`** §2 Step 5 — confirm Amendment 3 + retraining policy subsection accurate post-implementation. Add A4 trail-vs-classifier-exit precedence statement (Group 4). Verify Amendment 3 §"Failure-mode priority" ordering is implementation-accurate.
2. **`docs/PROTOCOL_RUNTIME.md`** — new section §"Amendment 3 emissions" documenting: chained max DD location + computation, per-day max-DD parquet schema + day-start-equity definition, holdout re-runs at scaled risk (file paths + manifest entries), sizing-convention check semantics.
3. **`docs/BACKTESTER_ARCHITECTURE.md`** — update Step 5 section pointing to amended gates + A3/A4 wiring; mirror PR #185's "Out of scope" → "see PROTOCOL_RUNTIME §X" pattern.
4. **`WORKFLOW.md`** §7 — confirm v1.2 closure template references accurate; no change expected.
5. **`docs/audits/engine_capability_audit_2026_05.md`** — APPEND footer: "## Post-PR-XXX update (2026-05-XX): items moved MISSING → WIRED" listing the six Amendment 3 items + A3 + A4 + Step 4 lineage. Original audit body untouched.
6. **`scripts/update_tracker_from_closure.py`** — v1.2 schema PASS-verdict validation extended: require Amendment 3 fields (`chained_max_dd_base_pct`, `k_safe`, `r_safe_pct`, `scalable_to_safe`, `worst_fold_roi_at_r_safe_pct`, `chained_max_dd_at_r_safe_pct`, `daily_dd_breaches_at_r_safe`, `holdout_roi_at_r_safe_pct`, `holdout_dd_at_r_safe_pct`, `sizing_convention`) to be non-null. v1.0 / v1.1 unchanged.

**LOC:** ~200 doc lines + ~30 parser test.

### Group 9 — Verify against Arcs 8/10/11 manual re-evaluation

**Approach.** Per `results/re_evaluation_2026_05/SUMMARY.md` (already exists per audit), Arcs 8/10/11 have manual re-evaluation §10 blocks in their closure docs.

Build a small verification script `scripts/verify_amended_gates_against_arcs.py`:

```python
"""Verify the new amended-gate logic against Arcs 8/10/11 manual
re-evaluation outcomes.

For each closure doc:
  1. Read §1 tracker_payload (post-PR-185 / pre-this-PR data).
  2. Synthesise the inputs to classify_amended_fold_stats from the
     metrics in the closure (worst_fold_roi_base_pct, worst_fold_dd_base_pct,
     n_folds, holdout_roi_base_pct, etc.).
  3. Call classify_amended_fold_stats.
  4. Compare verdict + primary_failure_mode to closure's
     re_evaluated_verdict.
  5. Print a reconciliation table; non-zero exit if any mismatch.
"""
```

Arc 10: expect `PASS-DEPLOYABLE-PROVISIONAL`. Arc 8: expect `FAIL` with `step5_ratio_below_gate_after_scaling`. Arc 11: expect `FAIL` with `step5_not_scalable`.

Engine wins on tiebreak; closure docs NOT mutated.

**Files:**
- `scripts/verify_amended_gates_against_arcs.py` (NEW) (~150 LOC)
- `tests/integration/test_verify_arcs_8_10_11.py` (NEW) — calls the verification script + asserts no mismatch (~80 LOC)

**LOC:** ~230 script + test.

---

## §4 Aggregate file inventory

### Files modified (estimated)

| File | Group | LOC |
|---|---|---:|
| `core/features/pipeline.py` | 1 | ~5 |
| `core/steps/step_4_extraction.py` | 1 (audit) | ~0 (audit only; `causal_lineage` check already correct) |
| `core/architectures/a1_system_level_filter.py` | 2 | ~10 |
| `core/architectures/a3_pipeline_de.py` | 2, 3 | ~80 |
| `core/architectures/a4_pipeline_d_exits.py` | 2, 3 | ~80 |
| `core/architectures/a5_portfolio_composition.py` | 5.4 | ~5 |
| `core/architectures/a6_meta_labeling.py` | 5.4 | ~5 |
| `core/architectures/a2_classifier_filter.py` | 5.4 | ~5 |
| `core/architectures/_protocol.py` | 5.1 | ~5 |
| `core/arc/arc_orchestrator.py` | 2, 3, 5 | ~150 |
| `core/sim/multipair_backtester.py` | 4 | ~5 |
| `core/wfo/orchestrator.py` | 5 | ~150 |
| `core/wfo/gates.py` | 5.5 (legacy mark only) | ~10 |
| `core/runners/_fold_stats_helpers.py` | 5.2 | ~80 |
| `L_PROTOCOL.md` | 4, 8 | ~20 |
| `docs/PROTOCOL_RUNTIME.md` | 8 | ~100 |
| `docs/BACKTESTER_ARCHITECTURE.md` | 8 | ~20 |
| `docs/audits/engine_capability_audit_2026_05.md` | 8 | ~80 (footer) |
| `scripts/update_tracker_from_closure.py` | 8 | ~30 |

### Files created (estimated)

| File | Group | LOC |
|---|---|---:|
| `core/wfo/chained_dd.py` | 5.1 | ~80 |
| `core/wfo/holdout_rerun.py` | 5.3 | ~50 |
| `core/wfo/amended_gates.py` | 5.5 | ~300 |
| `core/steps/path_classifier_per_fold.py` | 3 | ~200 |
| `tests/integration/__init__.py` | 7 | ~0 |
| `tests/integration/conftest.py` | 7 | ~80 |
| `tests/integration/test_amendment_3_gate_evaluation.py` | 7.1 | ~250 |
| `tests/integration/test_a3_end_to_end.py` | 7.2 | ~150 |
| `tests/integration/test_a4_end_to_end.py` | 7.3 | ~150 |
| `tests/integration/test_holdout_rerun_at_scaled_risk.py` | 7.4 | ~100 |
| `tests/integration/test_per_day_dd_breach_evaluation.py` | 7.5 | ~100 |
| `tests/integration/test_steps_1_through_5_end_to_end.py` | 7.6 | ~250 |
| `tests/integration/test_verify_arcs_8_10_11.py` | 9 | ~80 |
| `tests/protocol_runtime/test_multipair_backtester_precedence.py` | 4 | ~80 |
| `tests/protocol_runtime/test_amended_gates.py` | 5.5 | ~250 |
| `tests/protocol_runtime/test_chained_dd.py` | 5.1 | ~80 |
| `tests/protocol_runtime/test_per_day_max_dd.py` | 5.2 | ~80 |
| `scripts/verify_amended_gates_against_arcs.py` | 9 | ~150 |
| `docs/dispatches/combined_engine_intent.md` | (this file) | (this file) |
| `docs/dispatches/combined_engine_log.md` | end of PR | TBD |

**Estimated total:** ~2900 LOC across ~40 files (within the dispatch's 1500-3000 estimate).

---

## §5 Commit organisation

Per dispatch discipline rule: "commits organised by Task Group; each group's commits should be reviewable in isolation."

Plan:

| Commit | Title | Group |
|---|---|---|
| 1 | `[STEP 4] Fix lineage column name in feature_lineage_dataframe` | 1 |
| 2 | `[STEP 5] Extend A1RunContext for A3/A4 per-fold prep` | 2 |
| 3 | `[STEP 5] A3/A4 per-fold classifier orchestration + cost decomp` | 3 |
| 4 | `[SIM] A4 trail-vs-classifier-exit precedence: trail wins` | 4 |
| 5 | `[AMENDMENT 3] Chained max DD emission` | 5.1 |
| 6 | `[AMENDMENT 3] Per-day max-DD parquet emission` | 5.2 |
| 7 | `[AMENDMENT 3] Sizing-convention gate check` | 5.4 |
| 8 | `[AMENDMENT 3] Scaled gate logic + failure-mode taxonomy` | 5.5 + 5.6 |
| 9 | `[AMENDMENT 3] Holdout re-run at scaled risk` | 5.3 |
| 10 | `[AMENDMENT 3] Daily DD per-day re-evaluation` | 6 |
| 11 | `[TESTS] Integration tests for Amendment 3 + A3/A4` | 7 |
| 12 | `[VERIFY] Reconcile new gate logic with Arcs 8/10/11 manual re-eval` | 9 |
| 13 | `[DOCS] L_PROTOCOL + PROTOCOL_RUNTIME + audit footer` | 8 |

Each commit independently buildable + tested. Final PR rebases against `main` head; squash merge optional per chat preference.

---

## §6 Risk register

Per dispatch §"Risks":

1. **Scope size (3-5 days).** Mitigation: commit-per-group; intent doc update every 2-3 commits with progress; chat sees daily status.
2. **A3/A4 architectural complexity** — per-fold retrain + admit/reject/early-exit dynamics is the highest-novelty work. Mitigation: build Group 3 BEFORE Group 5; if Group 3 surfaces a structural blocker, surface to chat before committing further engine work.
3. **Per-day max-DD parquet schema stability** — new artefact. Mitigation: lock schema in Group 5.2 commit; document in `docs/PROTOCOL_RUNTIME.md` as part of the same commit; do not iterate schema during this PR.
4. **Manual re-evaluation reproduction (Group 9)** may surface manual errors. Mitigation: per dispatch §"Risks" #4, engine wins on tiebreak; manual values get patched as separate housekeeping after this PR lands. Discrepancies surfaced explicitly to chat.
5. **NEW — A3/A4 backwards-compat with synthetic tests.** Existing `tests/protocol_runtime/test_architectures_synthetic.py` constructs A3Config / A4Config with `classifier_fit` directly. Decision-point in §"Group 2" — confirm: keep field on config for synthetic test path, deprecate in production via the run_context path. Mitigation: keep `Optional` field; emit DeprecationWarning when set; production callers route through orchestrator's auto-builders.
6. **NEW — `core/wfo/orchestrator.py:run_search` already returns a fully-formed `WfoSearchResult`.** Extending it to also emit chained-DD + per-day-DD per candidate may bloat the return type. Mitigation: introduce `AmendedWfoSearchResult` extension dataclass; existing callers (Arc 11's driver) keep working with `WfoSearchResult`; orchestrator routes new fields through the extension.

---

## §7 Open questions for chat before code lands

1. **Lineage column-name decision (Group 1).** Confirm: rename `feature_lineage_dataframe` output column `lineage` → `causal_lineage` (matches protocol + `_filter_lineage` check + existing test). Alternative: change `_filter_lineage` to accept `lineage` column. Default: rename to match protocol verb.

2. **A3/A4 config backwards-compat (Group 2).** Confirm: keep `classifier_fit` + `per_trade_entry_features` as optional fields on `A3Config` / `A4Config` for direct-construction backward-compat (synthetic tests), but route the production path through `A1RunContext`-extended fields. Emit DeprecationWarning when config fields are set directly. Alternative: full removal — break synthetic tests, force migration to context. Default: keep with deprecation.

3. **A4 precedence flip (Group 4).** Confirm: per dispatch recommendation, flip current `setdefault` to direct assignment so trail-stop wins over classifier exit on same-bar tie. The current code's comment "Don't overwrite a predicate-driven exit that fired this bar" is an intentional design choice and will be reversed. Alternative: keep current behaviour and document it. Default: flip per dispatch.

4. **`AmendedWfoSearchResult` vs extending `WfoSearchResult` (Group 5 Risk #6).** Confirm: new extension dataclass keeps backwards-compat with existing callers. Alternative: amend `WfoSearchResult` in-place — risks breaking Arc 11's bypass driver. Default: extension dataclass.

5. **`tests/integration/` directory creation.** Confirm: new top-level test directory is acceptable, or should tests go under `tests/protocol_runtime/`? Default: new `tests/integration/` directory for tests that span multiple steps + the full pipeline; mirrors the dispatch §"Group 7" naming.

6. **Continuous-equity reconstruction for chained DD (Group 5.1).** Per-fold equity currently resets at fold start. Chaining requires either (a) reconstructing continuous equity by anchoring fold k+1's start to fold k's end, OR (b) running a single continuous sim spanning all folds + holdout. Option (a) is cheaper but lossy (loses the per-fold equity-reset semantics that L arc work uses for sizing). Option (b) is expensive (one extra full-window sim per candidate). Default: option (a) — concatenate per-fold OOS equity curves chronologically with continuity adjustment (multiplicative chaining); document the approximation. Confirm if (b) preferred for accuracy.

7. **Per-arc seed vs deterministic-default for A3/A4 per-fold retrain (Group 3).** Dispatch §"Group 3" item 3.4: "Per-fold seeds derived from `(arc_seed, fold_index, architecture_id)`." Confirm: derived seed is `hash((arc_seed, fold.fold_id, arch_name)) & 0xFFFFFFFF`; default `arc_seed = 42` per `RANDOM_STATE`. Reproducibility guaranteed.

---

## §8 Discipline confirmations

- Single PR per chat decision; commits organised by group per dispatch.
- No engine changes outside the audit's MISSING/PARTIAL list for this PR's scope.
- Determinism: all new engine paths use `random_state=42`, `n_jobs=1`, `lineterminator='\n'`. Per-fold derived seeds per Group 3.
- Backwards compatibility: Arcs 5/7 in-flight closures (none currently in `results/`) work under the new engine; closed Arcs 8/10/11 closure docs stay as-is (Group 9 verifies; doesn't mutate).
- Tests gate merge (Group 7 deliverable).
- Daily intent doc update at significant progress points (per dispatch §"Risk #1").

---

## §9 End turn

Awaiting chat sign-off on:
- §7 open questions (especially #1, #4, #6)
- Overall scope + commit organisation
- Branch name confirmation: `engine/amendment-3-implementation-and-architectures` matches dispatch

After sign-off, code lands in commit order per §5. Intent doc gets a progress addendum at the end of each group. Log doc opens at PR-creation time.

End of intent.
