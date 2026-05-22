# CC_07 — Protocol Runtime Infrastructure: Intent

> **Branch:** `claude/inspiring-mendeleev-8720f7` (cut from `main` at `f3bc078`).
> Final branch name on push will be `infra/protocol-runtime-v3` per dispatch.
> **Status:** intent doc only. No code yet. End-of-turn for chat review.
> **Dispatch source:** chat-supplied CC_07 dispatch (verbatim above).

This document fulfils the dispatch's "Read-first" requirement. It surveys
the current state of the repo, lists every file CC plans to touch or
create across PRs 7.A–7.G, confirms the PR sequencing, and flags
interpretive calls that need a chat decision before code lands.

---

## §1 Current-state survey

### §1.1 What CC_06 left behind (the certified substrate)

**Backtester layer is complete** per [BACKTESTER_ARCHITECTURE.md](docs/BACKTESTER_ARCHITECTURE.md). The
following modules exist and are tested:

- `core/data/{histdata_loader,aggregator,cache_keys}.py` — M1 bid+ask
  loader + deterministic TF aggregation.
- `core/spread/real_spread.py` — per-bar spread + tradability mask.
- `core/sim/{multipair_backtester,panel,account,fill,exit_hooks,trailing_stop}.py` —
  the simulator and its building blocks.
- `core/sim/risk/{reset_floor,live_balance}.py` — two risk-sizing modes
  (KH-24 uses live-balance).
- `core/features/{pipeline,registry,lineage}.py` + 7 feature class
  modules — 27 features, lineage-tagged.
- `core/features/cache.py` — feature-matrix cache keyed on
  `signal_def × pool_sha × feature_set_version`.
- `core/wfo/{folds,gates,orchestrator,fold_runner}.py`:
  - `folds.py` — `build_v3_folds()` (11+1 anchored) and
    `build_kh24_anchor_folds()` (7 rolling).
  - `gates.py` — §3 PASS-DEPLOYABLE / PASS-VIABLE / FAIL classifier on
    `FoldStats`.
  - `orchestrator.py` — `run_search` + `run_holdout` with top-K
    ranking, signal-agnostic.
  - `fold_runner.py` — `KH24FoldRunner` (KH-24-specific) plus generic
    helpers (`_max_drawdown_pct`, `_count_daily_5pct_breaches`,
    `_build_fold_stats`, `_equity_to_daily_returns`).
- `core/strategies/kh24/` — KH-24 assembly. `KH24Config` +
  `build_kh24_runtime(panel_h4, panel_d1, panel_h1, config)` returns
  the bundle (`account`, `risk`, `trail_manager`, `exit_predicates`,
  `strategy`) consumed by `MultiPairBacktester`.
- `core/features_path_so_far.py` — **already exists**, defines the 8
  entry + 7 path-so-far feature schema for Pipeline D1 / DE work.
  Locked schema, ready for A3/A4 to reuse.
- `core/parallel.py` + `core/determinism.py` — parallelism harness and
  RNG / line-terminator invariants.
- `scripts/anchor/run_anchor.py` — drives KH-24 end-to-end through
  the 7-fold (mode A) or 11+1-fold (mode B) WFO, writes per-fold
  parquet + summary + sha256 manifest.

**KH-24 anchor reproduction is closed at "Path B partial"**: F7
reproduces inside the real-spread band; F2 sign recovered; F1/F4/F5
remain sign-reversed against the live-broker numbers, attributable
to documented data-source drift + two deferred EA-correction items
(Sections G + H of the EA diff doc). The live deployment on Contabo /
5ers MT5 is unaffected. CC_07 inherits this as the anchor baseline —
"v3 anchor" hereafter means the per-fold numbers from
`scripts/anchor/run_anchor.py` mode A, not the published 5ers numbers.

### §1.2 What does NOT yet exist (CC_07's scope)

None of the four target directories are populated:

- `core/architectures/` — **does not exist.** No A1–A6 modules.
  `BACKTESTER_ARCHITECTURE.md:331` explicitly notes these get added
  "incrementally with each arc" — CC_07 builds them once, properly.
- `core/runners/` — **does not exist.** `KH24FoldRunner` lives at
  `core/wfo/fold_runner.py:135` and is KH-24-specific. No generic
  `arc_fold_runner` or `oracle_fold_runner`.
- `core/steps/` — **does not exist.** Step 2/3/4 logic only exists
  signal-coupled inside `scripts/arc_kh24_v2/step{2,3}/` (v2-schema
  reference, KH-24's exhaustion-bar signal) and
  `scripts/replays_v2_1_1/kh24_v2_c4{,_step4,_step5}/` (replays under
  the v2.1.x amendment proposal — also KH-24-coupled and v2-schema).
- `core/arc/` — **does not exist.** No `arc_pool_builder` or
  `arc_orchestrator`. Each historical arc reinvented Step 1 via
  per-arc scripts in `scripts/arc_*/` or `scripts/l_arc_*/`. That's
  the duplication CC_07 closes.

### §1.3 v2-schema reference (read for shape, do not reuse)

- `scripts/arc_kh24_v2/step1/run_step1.py` — Step 1 CLI: YAML config →
  per-pair signal evaluation → simulator → `trades_all.csv` +
  `trades_paths.csv` + `plumbing_report.md`. Pattern: deterministic
  CSV writes (`lineterminator='\n'`), sha256 logging, sample-checks
  for spread / SL / D1-lag invariants, gate summary at top of
  report. The simulator (`_simulate.py`) is a hard-SL-only +
  240-bar-cap loop — explicitly NOT the v3 simulator.
- `scripts/arc_kh24_v2/step2/run_step2.py` — KMeans K∈{3..7} sweep on
  StandardScaler-normalised path features, silhouette per K,
  archetype matching. Outputs `clusters_K{N}.csv` +
  `centroids_K{N}.csv` + `archetype_assignments.csv`.
- `scripts/arc_kh24_v2/step3/run_step3.py` — per-cluster forward
  geometry + distribution shape + capturability composite +
  archetype finalisation. Outputs `archetype_summaries.csv` +
  `capturability_pass_list.csv`.
- `scripts/replays_v2_1_1/kh24_v2_c4_step4/step4.py` — Pipeline E +
  Pipeline D1 classifier production (RF + Logistic, StratifiedKFold,
  AUC, threshold sweep, joblib + YAML outputs). Useful pattern but
  signal-coupled to KH-24 c1/c4 and v2.1.1 protocol semantics.
- `scripts/replays_v2_1_1/kh24_v2_c4_step5/step5.py` — cross-fold
  stability check (per-fold retrain, admit-only metrics, §9 gate
  evaluation). Also v2-schema and signal-coupled.

These are not deleted by CC_07. They're frozen historical reference;
the new `core/steps/` modules port the **shape** (deterministic IO,
gate-as-ranking output) but not the signal-coupled code. PR 7.G's
"cleanup orphaned files" applies to arcs 5/7/8/10/11 partial-run
detritus, NOT to these v2-schema canonical references.

### §1.4 KH-24 anchor lineage (what PRs 7.A/7.D/7.F validate against)

The anchor baseline for every CC_07 check is the output of:

```python
from scripts.anchor.run_anchor import run
from core.wfo.folds import build_kh24_anchor_folds

run(
    mode="A",
    structure=build_kh24_anchor_folds(),
    histdata_root=<repo>/data/histdata,
    cache_root=<temp>/pr_e2_cache,
    out_root="results/anchor_kh24_7fold_v3",
)
```

This produces `fold_by_fold.parquet` with the v3 reproduction's
per-fold ROI / DD / trade-count / win% / PF / ratio. The
`BACKTESTER_ARCHITECTURE.md` table at §B is the published reference
("Pub" columns); the "v3" columns are the same as `fold_by_fold.parquet`.

CC_07 anchor checks compare the new orchestrator-driven KH-24 run
against the "v3" numbers, NOT against "Pub". Tolerance per dispatch:
±0.5pp ROI / ±1pp DD per fold.

---

## §2 Required-first reads — confirmation

Per dispatch §Read-first, CC has read in full:

- ✅ [L_PROTOCOL.md](L_PROTOCOL.md) — all 9 sections + Appendix A
  (classifier defaults) + Appendix B (search space). Amendments 1 + 2
  on Step 5 search policy / ML mechanics noted and incorporated
  into A2/A3/A4/A6 specs below.
- ✅ [WORKFLOW.md](WORKFLOW.md) — §2 dispatch artefact pattern (intent
  → log → PR), §4 diff-before-code, §5 staged PRs, §6 HALT discipline.
  CC_07 follows the staged-PR pattern with PRs 7.A through 7.G.
- ✅ [CLAUDE.md](CLAUDE.md) — non-negotiables, eliminated approaches,
  current research state. Loaded via system context at session open
  (no separate read needed).
- ✅ [docs/BACKTESTER_ARCHITECTURE.md](docs/BACKTESTER_ARCHITECTURE.md) —
  current scope statement, layer diagram, anchor reproduction band.
- ✅ Surveyed `core/` layout (§1.1), `scripts/arc_kh24_v2/step1/`
  (§1.3), `scripts/anchor/run_anchor.py` (§1.4), `core/strategies/kh24/`
  KH-24 assembly mechanics, `core/wfo/{folds,orchestrator,fold_runner,gates}.py`.

---

## §3 Files CC will touch — full enumeration by staged PR

Every path below is repo-relative. New files are marked `[new]`,
edits are marked `[edit]`. CC has not yet touched any file.

### §3.1 PR 7.A — Arc-pool plumbing + Step 1 generalisation

**Goal:** generic, signal-agnostic Step 1 runner that
arc-config → pool.parquet + integrity_report.md + manifest.json.

New / edited:

- `core/arc/__init__.py` `[new]` — package marker.
- `core/arc/arc_pool_builder.py` `[new]` — `build_arc_pool(arc_config,
  signal_module, panels) -> ArcPoolResult` returning trade pool +
  lineage + integrity-check report. Wraps signal application,
  ex-ante population construction, per-trade simulation against
  the v3 multipair backtester (or a generic ex-ante simulation if
  the arc's SL/exit policy is set at open and unchanged across Step 1).
- `core/arc/signal_protocol.py` `[new]` — defines the `SignalModule`
  Protocol that arc configs reference (must expose `evaluate(pair_df,
  panel) -> SignalResult` returning `signal_mask + atr_series +
  per_trade_features`).
- `core/arc/integrity.py` `[new]` — pool integrity checks (pool size,
  per-pair n, coverage window, gap report, spread-floor activation,
  D1-lag NaN perturbation, 5-random-trade lookahead spot-check,
  KH-24 co-fire rate, determinism sha256). Output → markdown table
  per L_PROTOCOL §2 Step 1.
- `configs/protocol_runtime/__init__.py` `[new]` + `configs/protocol_runtime/example_arc.yaml`
  `[new]` — schema reference YAML for the runtime.
- `tests/protocol_runtime/__init__.py` `[new]`.
- `tests/protocol_runtime/test_arc_pool_builder.py` `[new]` — synthetic
  signal end-to-end, determinism sha256, integrity-check shape.
- `tests/protocol_runtime/test_arc_pool_builder_kh24.py` `[new]` —
  KH-24 signal through `arc_pool_builder` → pool consistency check
  (per §6 below: trade count + entry-time set vs anchor-runner
  reference; documented residual band).
- `docs/PROTOCOL_RUNTIME.md` `[new]` — §1 "Step 1 — Arc Pool Builder"
  section + output schema for `step_1/pool.parquet`.

PR 7.A landing condition (per dispatch, with §6.1 interpretive note):
KH-24 Step 1 via `arc_pool_builder` matches the anchor-runner KH-24
output on a documented invariant (entry-time set + per-pair count
within bound).

### §3.2 PR 7.B — Step 2 + Step 3 runners

New / edited:

- `core/steps/__init__.py` `[new]`.
- `core/steps/step_2_clustering.py` `[new]` — KMeans K∈{2..6}
  (`random_state=42, n_init=10, max_iter=300`), silhouette per K,
  shape-tag assignment per L_PROTOCOL §2 Step 2 quartile rules
  (V-shape recovery / Stepwise climber / Bimodal / Monotonic up /
  Monotonic down / Choppy / Unclassified). Output:
  `cluster_assignments.parquet` + `cluster_summary.md` +
  `manifest.json`.
- `core/steps/step_3_capturability.py` `[new]` — per-cluster reach
  rates (≥1R/2R/3R), MFE distribution (p25/p50/p75/p90), `ww_pp`,
  time-to-peak, mean R, SL sweep {1.5,2.0,2.5,3.0,3.5,4.0}×ATR,
  capturability composite, archetype label, candidate-cluster flag.
  Output: `capturability.csv` + `capturability_summary.md` +
  `manifest.json`.
- `core/steps/_shape_tags.py` `[new]` — quartile rules + tag assignment
  (factored out of step 2 so tests can exercise tagging on synthetic
  data without re-running clustering).
- `tests/protocol_runtime/test_step_2_clustering.py` `[new]` — synthetic
  path data with known cluster structure; assert K-sweep output,
  silhouette ranking, archetype labels.
- `tests/protocol_runtime/test_step_3_capturability.py` `[new]` —
  synthetic clusters with controlled MFE / ww_pp / time-to-peak;
  assert reach-rate formula, composite formula, candidate-flag
  threshold (reach_1R ≥ 0.50 ∧ ww_pp ≤ 0.30 ∧ mfe_p50 ≥ 1.5R).
- `tests/protocol_runtime/test_shape_tags.py` `[new]` — quartile rule
  coverage across all 7 tag classes.
- `docs/PROTOCOL_RUNTIME.md` `[edit]` — §2 / §3 sections added.

### §3.3 PR 7.C — Step 4 runner

New / edited:

- `core/steps/step_4_extraction.py` `[new]` — RF + LightGBM + Logistic
  at L_PROTOCOL Appendix A defaults, 5-fold TimeSeriesSplit per-cluster.
  AUC + threshold sweep (AUC-best + F1-best) + permutation importance.
  Lineage-tag enforcement: features tagged `suspect` / `unverified`
  excluded from training; assertion logs the exclusion list.
  Outputs: `extraction_metrics.csv` + `feature_importance.csv` +
  `extraction_summary.md` + `manifest.json`.
- `core/steps/_classifier_defaults.py` `[new]` — Appendix A hyperparams
  as a `dataclass`, with helper to instantiate sklearn estimators.
  Mirrors `core/features/pipeline.py` style for reusability.
- `tests/protocol_runtime/test_step_4_extraction.py` `[new]` —
  synthetic data with known AUC ceiling; assert classifier choice
  matches AUC-best, threshold sweep monotonicity, permutation
  importance ranking, lineage-tag exclusion test.
- `tests/protocol_runtime/test_step_4_determinism.py` `[new]` — two-run
  sha256 match on `extraction_metrics.csv`.
- `docs/PROTOCOL_RUNTIME.md` `[edit]` — §4 section added.

PR 7.C landing condition: synthetic data with controlled
signal-to-noise produces AUC matching analytical expectation within
sampling tolerance (±0.02 AUC, given the small synthetic N).

### §3.4 PR 7.D — Architectures A1, A2, A6 (non-retrain)

New / edited:

- `core/architectures/__init__.py` `[new]`.
- `core/architectures/_protocol.py` `[new]` — uniform `Architecture`
  Protocol: `run(signal_pool, features, config, wfo_fold) -> StrategyResult`.
  Defines `StrategyResult` (per-trade decisions, per-fold metrics,
  equity curve, sha256 manifest entry).
- `core/architectures/a1_system_level_filter.py` `[new]` — base signal +
  rule-based filter from Step 4 (or "no filter") + exposure cap +
  searched SL/exit policy. **Critically, KH-24 must be expressible as
  an A1 config.** See §6.2 for the open question on how to factor
  `build_kh24_runtime` such that KH-24 becomes one A1 instantiation
  rather than its own runtime.
- `core/architectures/a2_classifier_filter.py` `[new]` — base signal +
  Step 4's best AUC classifier at AUC-best threshold + exposure cap +
  searched SL/exit policy. No retraining at A2 — uses the joblib /
  pickled classifier from Step 4. Per Amendment 2 mechanics.
- `core/architectures/a6_meta_labeling.py` `[new]` — Step 4 best
  classifier confidence → 0x/0.5x/1x sizing per Amendment 2 threshold
  pairs {(0.3,0.5), (0.4,0.6), (0.5,0.7)}. Sizing implemented as a
  per-order risk-multiplier injected at strategy-fn level (see §6.4).
- `core/runners/__init__.py` `[new]`.
- `core/runners/arc_fold_runner.py` `[new]` — generic
  `ArcFoldRunner(architecture, config, panels)` callable returning
  `FoldStats`. Replaces the KH-24-specific path inside
  `core/wfo/fold_runner.py:KH24FoldRunner`. Internally delegates to
  the supplied architecture's `run()`.
- `core/wfo/fold_runner.py` `[edit]` — `KH24FoldRunner` becomes a thin
  wrapper that constructs an `ArcFoldRunner(A1, KH24Config-as-A1Config,
  panels)`. **The old behaviour is preserved as a regression path**: if
  the wrapper produces different per-fold numbers than the direct path,
  HALT.
- `tests/protocol_runtime/test_a1_kh24_anchor.py` `[new]` — runs KH-24
  through A1 via `ArcFoldRunner`, asserts per-fold ROI / DD / trade
  count match the v3 anchor within ±0.5pp / ±1pp / ±5 trades.
- `tests/protocol_runtime/test_a2_classifier_filter.py` `[new]` —
  synthetic pool + injected classifier; assert admit/reject decisions
  match expected threshold sweep behaviour.
- `tests/protocol_runtime/test_a6_meta_labeling.py` `[new]` — synthetic
  pool + injected classifier confidence; assert sizing decisions match
  the threshold-pair table.
- `docs/PROTOCOL_RUNTIME.md` `[edit]` — A1/A2/A6 API reference.

PR 7.D landing condition: KH-24 reproduced through A1 within the
documented anchor band. This is **the strongest gate of the dispatch**.

### §3.5 PR 7.E — Architectures A3, A4 (new-classifier-per-fold)

New / edited:

- `core/architectures/a3_pipeline_de.py` `[new]` — NEW classifier per
  fold, target = cluster membership, features = path-so-far at bar
  N ∈ {3, 5} (reuses `core/features_path_so_far.py:PATH_FEATURE_KEYS`).
  RF defaults from `_classifier_defaults`. Decision rule: at bar N
  post-signal, classify → if ≥ threshold enter at next bar open, else
  cancel.
- `core/architectures/a4_pipeline_d_exits.py` `[new]` — NEW classifier
  per fold, target = "final R > 0" binary. Same path-so-far feature
  set, evaluated at each bar post-entry. Decision rule: if confidence
  < exit threshold (tested at {0.3, 0.4, 0.5}), queue exit at next bar
  open. Initial SL still binds.
- `core/architectures/_path_classifier.py` `[new]` — shared per-fold
  retrain helper. Both A3 and A4 instantiate it; lives separately so
  the contract (per-fold training, no global model) is centralised
  and testable.
- `tests/protocol_runtime/test_a3_pipeline_de.py` `[new]` — synthetic
  path data where early bars distinguish in-cluster vs out-of-cluster;
  assert DE rejects clearly-bad early paths.
- `tests/protocol_runtime/test_a4_pipeline_d_exits.py` `[new]` —
  synthetic path data with clearly-failing trades; assert D-exits cut
  them before time-exit horizon. SL precedence test (SL fires before
  classifier exit if both would trigger same bar).
- `tests/protocol_runtime/test_a3_a4_per_fold_retrain.py` `[new]` —
  assert each fold trains its own classifier (test by injecting
  fold-index-marker into training data and verifying the marker
  changes across folds).
- `docs/PROTOCOL_RUNTIME.md` `[edit]` — A3/A4 API reference with
  per-fold-retrain emphasis.

PR 7.E landing condition: synthetic data only (no KH-24 anchor for A3
or A4 — KH-24 doesn't use deferred entry or differentiated exits).

### §3.6 PR 7.F — Architecture A5 + arc orchestrator

New / edited:

- `core/architectures/a5_portfolio_composition.py` `[new]` — combines
  ≥ 2 candidate clusters into a portfolio with combined account state.
  Admit-only economics (per L_PROTOCOL §2 Step 5 multi-cluster). The
  combined account uses a shared `Account` instance with summed
  exposure rules. Top-level invariant: `A5.run([A1_cluster_a,
  A1_cluster_b]) ≡ run both, sum equity curves, recompute DD on
  combined curve`.
- `core/arc/arc_orchestrator.py` `[new]` — `ArcOrchestrator(arc_config)`
  → runs Steps 1→5 sequentially, invokes Step 6 conditionally per
  L_PROTOCOL §2 Step 6 trigger, drafts closure doc skeleton with
  results populated per §6 ARC_CLOSURE.md format.
- `core/arc/_closure_template.py` `[new]` — Jinja-free string templates
  for ARC_OPEN.md and ARC_CLOSURE.md per L_PROTOCOL §6 (Sections 1–7).
- `tests/protocol_runtime/test_arc_orchestrator.py` `[new]` — synthetic
  arc with known structure (3-cluster synthetic pool, known capturability,
  known extractability); assert orchestrator runs to completion and
  produces correct top-K + verdict + closure doc.
- `tests/protocol_runtime/test_arc_orchestrator_kh24_e2e.py` `[new]` —
  **the final certification of the runtime.** KH-24 through
  `ArcOrchestrator` end-to-end; assert per-fold numbers match v3
  anchor within ±0.5pp / ±1pp; assert closure doc populated.
- `core/wfo/fold_runner.py` `[edit]` — optional: tighten the thin
  wrapper if PR 7.D left rough edges.
- `docs/PROTOCOL_RUNTIME.md` `[edit]` — A5 + ArcOrchestrator section.

PR 7.F landing condition: KH-24 through `ArcOrchestrator` end-to-end
within anchor band. Closure doc skeleton properly populated.

### §3.7 PR 7.G — Documentation + cleanup

New / edited:

- `docs/PROTOCOL_RUNTIME.md` `[edit]` — final pass: full TOC, API
  reference for every public function, three usage examples
  (KH-24 reproduction, vanilla new-signal arc, sub-protocol-extended
  arc — last one stubbed since no sub-protocols ship in CC_07).
- `docs/BACKTESTER_ARCHITECTURE.md` `[edit]` — line 331 ("added
  incrementally with each arc") removed; reference to
  `core/architectures/` added with link.
- `docs/features_reference.md` `[edit-if-needed]` — only if a feature
  contract changed during PR 7.C lineage-tag enforcement (CC will
  flag in the log doc if so).
- `CLAUDE.md` `[edit-if-needed]` — header line updated to reflect
  v3.0 runtime now exists. CC will draft, chat approves wording.
- Cleanup of arcs 5/7/8/10/11 partial-run orphaned files: per dispatch
  §G, CC will catalog any unowned files committed during prior HALTs
  and propose deletions in the PR description (NOT delete branches).
  Branches/worktrees stay intact.
- `docs/dispatches/cc_07_log.md` `[new]` — final log doc per WORKFLOW §2.

---

## §4 PR sequencing — confirmation

The dispatch's ordering (7.A → 7.G) is correct and CC confirms it.
Specifically:

- 7.A first: every subsequent PR consumes `pool.parquet` and
  `feature_matrix.parquet`. Step 2/3/4 cannot be tested without a
  generic pool builder.
- 7.B before 7.C: Step 4 trains on clusters from Step 2 — Step 2 must
  ship first.
- 7.D before 7.E: A1/A2/A6 reuse Step 4 outputs without new training,
  so they're simpler to land + anchor-validate. Landing A1 first
  establishes the StrategyResult interface that A3/A4 inherit.
- 7.E before 7.F: A5 portfolio composition combines architectures from
  D and E. Orchestrator end-to-end test in F requires all six
  architectures available.
- 7.G last: docs + cleanup after all code lands.

**Proposed minor sequencing tweak (flag for chat):**

PRs 7.D and 7.E are large (3 architectures each in D, 2 in E with
per-fold retrain plumbing). If size becomes unwieldy, CC may propose
splitting 7.D into 7.D.1 (A1 only — the anchor-critical one) and
7.D.2 (A2 + A6). This would be decided at the start of 7.D after
the StrategyResult interface is locked. If split, CC documents in
the 7.D intent micro-doc and flags before continuing.

Final PR count: **7 (no split)** or **8 (with split)**.

---

## §5 Anchor preservation strategy

CC_07 has three anchor checkpoints, each progressively tightening:

1. **PR 7.A** — KH-24 Step 1 pool via `arc_pool_builder` must
   reproduce the anchor-runner's trade selection. Per §6.1 below, the
   strict "byte-identical pool.parquet" criterion may not be hittable
   on the first attempt (the anchor runner emits per-fold trades, not
   a single pool); CC will document the invariant actually checked and
   the residual band.

2. **PR 7.D** — KH-24 as an A1 config through `ArcFoldRunner`
   produces per-fold numbers within ±0.5pp ROI / ±1pp DD of the v3
   anchor. **This is the gate.** Failure HALTs at 7.D per WORKFLOW §6.

3. **PR 7.F** — KH-24 through `ArcOrchestrator` end-to-end matches
   7.D's output (orchestrator is a thin loop over fold runner; should
   be exact). Closure doc skeleton populated correctly.

Determinism check at every PR: two-run sha256 match on the new
artefacts produced by that PR. CC writes the determinism test
alongside the runner code, not as an afterthought.

---

## §6 Interpretive calls flagged for chat

These need a chat decision before code lands. CC will not invent
answers.

### §6.1 Arc-pool byte-identity criterion at PR 7.A

The dispatch says: "KH-24 Step 1 via new builder matches existing
KH-24 Step 1 byte-identical (sha256 match on pool.parquet) OR within
documented Path B residual band with written explanation."

But: **there is no pre-existing `pool.parquet` for KH-24 in the v3
world**. The anchor runner emits per-fold trades + equity, not a
single Step 1 pool. The closest analogue is
`scripts/arc_kh24_v2/step1/trades_all.csv` — but that uses
hard-SL-only + 240-bar-cap exits, not KH-24's full trail + kijun_d1.
So byte-identity is structurally impossible.

**Proposed substitute** (chat to confirm):

The new `arc_pool_builder` produces a pool whose trade set is
**identical** to the union of `scripts/anchor/run_anchor.py` mode A's
per-fold trades when run over the full 2010-2026 window in a single
pass. Determinism is checked via two-run sha256 on `pool.parquet`
within the builder itself (no external comparison).

The "byte-identical vs anchor runner" criterion is restated as: "trade
count + (pair, entry_time) set identical" — robust to minor schema
differences (the new pool has more columns: features, lineage, etc.).

**Decision needed:** approve substitute, or specify a different
invariant.

### §6.2 KH-24 → A1 config factoring

Dispatch landing condition for PR 7.D: "KH-24 reproducible as an A1
config (system filter + exposure + SL + exit) within documented
anchor band."

Currently `build_kh24_runtime` hardcodes:
- C1–C6/C8/C9 signal evaluation (in `core/strategies/kh24/signal.py`)
- H1 CIR T=0.28 filter (in `core/strategies/kh24/filters/h1_cir.py`)
- Per-pair pre-computed signal masks + ATR + Kijun + lag-1 D1
- TrailManager + LiveBalanceRisk + per-currency exposure cap = 2
- kijun_d1 exit predicate (in `core/strategies/kh24/exits/kijun_d1.py`)

Two options:

**Option (a) — A1 generalises, KH-24 becomes config-only.**
A1 takes a `SignalModule + filter_rules + exposure_rules + sl_atr_mult +
exit_policy_id + trail_params + risk_pct`. The KH-24 signal /
filter / exit modules stay where they are; `build_kh24_runtime`
becomes a thin convenience constructor that wires them into an A1
config. This is the dispatch's literal reading.

**Option (b) — A1 is a generic config-driven scaffold, KH-24 is the
canonical instantiation.** A1 directly imports `build_kh24_runtime`
when its config says `signal=kh24_signal`; for other arcs A1 needs a
different signal module but the broader scaffold stays the same.
This is faster to land but couples A1 to KH-24.

**Decision needed:** (a) is cleaner long-term but takes ~2x the LOC
in PR 7.D. (b) is pragmatic; future arcs would need a similar
plug-in for their own signal. CC recommends (a). If chat agrees, PR
7.D writes a `KH24SignalModule` adapter in `core/strategies/kh24/`
that conforms to the `SignalModule` Protocol, and A1's signature
becomes purely `SignalModule + rule-based filters + system params`.

### §6.3 v2-schema scripts: keep, port, or archive

`scripts/arc_kh24_v2/` and `scripts/replays_v2_1_1/` are historical
KH-24-coupled v2-protocol references. CC_07 does not need them
functionally — the new runtime supersedes their role.

**Proposed handling:** leave alone. They produce v2-schema outputs
under `results/arc_kh24_v2/` and `results/replays_v2_1_1/` that
historical closure docs reference. Migrating them to v3 schema is
out of scope. Add a one-line note to their README-equivalent (or in
PROTOCOL_RUNTIME.md §"historical references") that v3 supersedes
them.

**Decision needed:** confirm no migration / archival. If chat wants
them deleted or moved to `archive/`, CC does so in PR 7.G.

### §6.4 A6 meta-labeling sizing mechanics

L_PROTOCOL §2 Step 5 ML mechanics (Amendment 2) specifies:
"prob < lower → 0x; lower ≤ prob < upper → 0.5x risk; prob ≥ upper →
1.0x risk."

Currently `LiveBalanceRisk.size_for_trade(...)` returns a single size
given a fixed risk_pct. A6 needs per-order size multiplier injection.

**Proposed mechanic:** `Order` dataclass gains an optional
`risk_multiplier: float = 1.0` field. The driver multiplies the
computed position size by this multiplier before placing the trade.
A6's strategy-fn emits orders with `risk_multiplier ∈ {0.5, 1.0}`
based on classifier confidence (0x is implemented by simply not
emitting an order). This is a single-line change to `Order` and
~3 lines in the driver.

**Decision needed:** approve mechanic, or specify a different
sizing channel (e.g. a separate `MetaLabelingSizer` class consumed by
the architecture, leaving `Order` untouched).

### §6.5 Sub-protocol mechanism (out of scope for CC_07?)

L_PROTOCOL §5 supports sub-protocols (heavy_ml_probe,
signal_discovery_probe). The dispatch does not mention them. CC
proposes:

- ArcOrchestrator's API supports optional sub-protocol injection
  via `arc_config.sub_protocol: str | None`. If non-None, the
  orchestrator looks up `docs/sub_protocols/<name>.md` and reads
  override hooks from a sibling `core/sub_protocols/<name>.py`.
- No sub-protocols actually ship in CC_07. The hook is wired
  but every step uses vanilla overseer.

**Decision needed:** approve the hooked-but-empty approach, or defer
sub-protocol scaffolding entirely to a later dispatch.

### §6.6 Closure doc auto-update of ARC_TRACKER.md

L_PROTOCOL §6 ARC_TRACKER.md auto-update describes overseer parsing
Sections 1-3 of every closure doc and updating the tracker
mechanically.

Currently `ARC_TRACKER.md` exists but is manually maintained. CC_07
could add a tracker-updater module under `core/arc/` invoked at the
end of `ArcOrchestrator.run()`. This is scope creep — possibly
warranting a CC_07.5 dispatch.

**Decision needed:** in scope or defer? CC recommends defer. If
deferred, document in PR 7.G that the tracker remains
chat-maintained for the v3.0 runtime's first wave of arcs.

---

## §7 Discipline rules acknowledged

Per dispatch §"Discipline rules", CC commits to:

- ✅ Anchor-reproducing at PR 7.A, 7.D, 7.F. HALT per WORKFLOW §6 on
  any failure outside explainability.
- ✅ KH-24 anchor as the gate, per the band stated in §5 above
  (±0.5pp ROI / ±1pp DD, v3-anchor reference not 5ers-published).
- ✅ No protocol logic invented. Runtime implements L_PROTOCOL v3.0 +
  Amendments 1+2 exactly. Any ambiguity → HALT and surface per §6.
- ✅ No backtester modification. CC_06 substrate stays as-is. If a
  bug surfaces during anchor checks (e.g. fold runner edge case),
  HALT and surface before patching.
- ✅ Determinism binding. Two-run sha256 match required for every
  artefact produced by every PR. Tests written alongside.
- ✅ WORKFLOW §2 intent → log → PR for every PR. CC will produce
  per-PR micro-intent (one paragraph + file list) at the start of each
  PR and a log doc at the end.

---

## §8 Definition of done (CC's reading)

After this dispatch:

1. PRs 7.A → 7.G merged in order to `main`. (Per WORKFLOW §3, branch
   name is `infra/protocol-runtime-v3`; CC will rename the current
   branch from `claude/inspiring-mendeleev-8720f7` at first push.)
2. `core/architectures/`, `core/runners/`, `core/steps/`, `core/arc/`
   populated with tested modules.
3. `docs/PROTOCOL_RUNTIME.md` complete (TOC + API ref + 3 usage
   examples).
4. KH-24 reproduces through `arc_orchestrator.py` end-to-end within
   ±0.5pp ROI / ±1pp DD of v3 anchor.
5. Two-run sha256 determinism for full orchestrator run.
6. All tests pass under `pytest`.
7. Final turn ends after PR 7.G merge. Chat re-dispatches Phase 1
   Wave 1 arcs against the new runtime.

---

## §9 End of intent — turn ending

Per WORKFLOW §2 step 2 ("CC produces intent doc FIRST [...] end
turn for chat review"), CC stops here. No code written. No PR opened.

Chat decisions needed (in priority order):

1. **§6.1** — substitute byte-identity criterion at PR 7.A: approve
   or override.
2. **§6.2** — KH-24 → A1 factoring: option (a) generalise (CC
   recommends) or (b) couple.
3. **§6.4** — A6 sizing mechanic: `Order.risk_multiplier` or
   separate `MetaLabelingSizer`.
4. **§6.3** — v2-schema scripts: leave alone (CC recommends) or
   archive.
5. **§6.5** — sub-protocol scaffolding: hook-but-empty (CC
   recommends) or defer.
6. **§6.6** — ARC_TRACKER.md auto-update: defer (CC recommends) or
   in scope.

CC awaits chat direction on these six items, plus any other
adjustments to the PR sequence, file list, or anchor strategy. Once
chat responds, CC opens PR 7.A as the first executable step.
