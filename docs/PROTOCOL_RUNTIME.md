# PROTOCOL_RUNTIME.md — L_PROTOCOL v3.0 Runtime Reference

> **Status:** v3.0 launch. The runtime sits between the backtester
> ([BACKTESTER_ARCHITECTURE.md](BACKTESTER_ARCHITECTURE.md)) and arc
> execution. Built by CC_07.
>
> Closely tracks [L_PROTOCOL.md](../L_PROTOCOL.md) §2 (Steps 1–5) +
> Amendments 1 + 2. Any divergence is a bug in this doc.

---

## §1 What the runtime is

`core/arc/`, `core/steps/`, `core/architectures/`, `core/runners/`
turn an arc's signal definition + window + pair set into:

1. A trade pool (Step 1)
2. Path-shape clusters (Step 2)
3. Per-cluster capturability metrics (Step 3)
4. Per-cluster entry-time extractability classifiers (Step 4)
5. WFO architecture search across A1..A6 (Step 5)
6. A closure doc skeleton populated with results (per §6 of the
   overseer protocol)

The runtime is **signal-agnostic** — every arc plugs in via the
:class:`SignalModule` Protocol. KH-24 is the canonical first user; new
arcs (Wave 1: arc_8/9/11 etc.) write their own SignalModule and
configure A1..A6.

---

## §2 Module map

```
core/
├── arc/
│   ├── signal_protocol.py        # SignalModule Protocol + validate_panels
│   ├── arc_pool_builder.py       # Step 1: build_arc_pool, write_arc_pool
│   ├── integrity.py              # Pool integrity checks + report writer
│   ├── sub_protocol.py           # Hook for sub-protocols (empty at v3.0)
│   ├── arc_orchestrator.py       # ArcOrchestrator (Steps 1→5)
│   └── _closure_template.py      # ARC_OPEN.md / ARC_CLOSURE.md templates
│
├── steps/
│   ├── _shape_tags.py            # Archetype quartile rules
│   ├── _classifier_defaults.py   # L_PROTOCOL Appendix A hyperparams
│   ├── step_2_clustering.py      # KMeans K-sweep + silhouette + tags
│   ├── step_3_capturability.py   # Reach/MFE/ww_pp + SL sweep + composite
│   └── step_4_extraction.py      # RF + LGBM + LR + AUC + lineage gate
│
├── architectures/
│   ├── _protocol.py              # Architecture Protocol + StrategyResult
│   ├── _path_classifier.py       # Shared per-fold retrain helper (A3, A4)
│   ├── a1_system_level_filter.py
│   ├── a2_classifier_filter.py
│   ├── a3_pipeline_de.py         # Deferred entry (per-fold retrain)
│   ├── a4_pipeline_d_exits.py    # Differentiated exits (per-fold retrain)
│   ├── a5_portfolio_composition.py
│   └── a6_meta_labeling.py
│
├── runners/
│   ├── arc_fold_runner.py        # Generic (Architecture, config) -> FoldStats
│   ├── oracle_fold_runner.py     # True-label cluster admit (upper bound)
│   └── _fold_stats_helpers.py
│
└── strategies/kh24/
    ├── signal_module.py          # KH24SignalModule (KH-24 → SignalModule)
    └── a1_adapter.py             # kh24_to_a1(KH24Config) -> (A1Config, ...)
```

---

## §3 The SignalModule contract

```python
class SignalModule(Protocol):
    signal_name: str
    primary_tf: str
    auxiliary_tfs: tuple[str, ...]
    causal_lineage: str  # "clean" | "suspect" | "unverified"

    def evaluate(self, panels: Mapping[str, Panel]) -> SignalEvaluation: ...
```

The module owns:
  - Signal evaluation
  - Signal-class-inherent filters (e.g. KH-24's H1 CIR)
  - Signal-class-inherent exit predicates (e.g. KH-24's kijun_d1)
  - List of TFs it needs

It does NOT own SL multiplier, trail, exposure, risk, or external rule
filters — those are A1Config (and beyond).

`SignalEvaluation` returns a `PerPairSignalState` per pair:
  - `signal_mask`: bool Series on the primary TF
  - `atr`: float Series on the primary TF
  - `additional_gates`: dict of per-bar boolean Series (ANDed with mask)
  - `exit_predicate`: optional ExitPredicate for signal-class exits

---

## §4 Step 1 — Arc Pool Builder

```python
from core.arc.arc_pool_builder import ArcPoolConfig, build_arc_pool

pool = build_arc_pool(
    signal_module,           # implements SignalModule
    panels,                  # {"H4": panel, "D1": panel, "H1": panel}
    ArcPoolConfig(
        arc_name="my_arc",
        sl_atr_mult=2.0,
        hold_bars=240,
        window_start=date(2010, 1, 1),
        window_end=date(2026, 1, 1),
    ),
)
```

Returns `ArcPool`:
  - `trades`: DataFrame with the schema below
  - `paths`: per-bar forward-window R-multiples
  - `integrity`: tuple of IntegrityRow (pass/fail/info per check)
  - `signal_evaluation`: the SignalEvaluation that produced the pool

`trades.parquet` schema:
```
pair                 string
trade_id             int64
signal_time          timestamp
entry_time           timestamp
entry_price          float
atr_at_signal        float
sl_at_entry_price    float
exit_time            timestamp
exit_price           float
exit_reason          string  # "hard_sl" | "time_exit"
bars_held            int
final_r              float
mfe_r                float
mae_r                float
```

`paths.parquet` schema:
```
trade_id, bar_offset, timestamp, close_r, mfe_so_far_r, mae_so_far_r
```

Determinism: `pool.pool_sha256` is the sha256 of the canonical CSV
representation of trades. Two runs from the same panel + signal +
config produce identical sha256.

---

## §5 Step 2 — Clustering

```python
from core.steps.step_2_clustering import run_step_2

s2 = run_step_2(pool.trades, pool.paths)
# s2.cluster_assignments: trade_id, cluster_id, k_selected
# s2.cluster_summary:     cluster_id, n_trades, shape_tag, centroid_features
# s2.silhouette_per_k:    {2: 0.32, 3: 0.41, ...}
# s2.k_selected:          argmax silhouette
# s2.centroids:           {cluster_id: ClusterCentroid}
```

Path-shape features (5): monotonicity, local_peaks, mfe_p50_proxy,
time_to_peak_rel, wrong_way_first.

K-sweep range: {2, 3, 4, 5, 6}. Determinism: `random_state=42`,
`n_init=10`, `max_iter=300`.

Shape-tag rules (locked at v3.0; cross-arc recalibration only):

| Tag | Rule |
|---|---|
| Monotonic up | m ≥ 0.85, mfe_p50 ≥ 2.0 |
| Monotonic down | m ≤ 0.15, mfe_p50 ≤ 0.5 |
| Stepwise climber | m ≥ 0.60, mfe_p50 ≥ 1.5, ww_pp < 0.20 |
| V-shape recovery | m < 0.50, ttp_rel < 0.40, mfe_p50 > 0.5, ww_pp ≥ 0.20 |
| Bimodal | 0.30 ≤ m < 0.70, peaks ≥ 2.0, mfe_p50 > 1.0 |
| Choppy | m < 0.40, peaks ≥ 1.5, mfe_p50 < 1.0 |
| Unclassified | (none of the above) |

---

## §6 Step 3 — Capturability

```python
from core.steps.step_3_capturability import run_step_3

s3 = run_step_3(
    pool.trades, pool.paths, s2.cluster_assignments,
    declared_sl_mult=2.0, cluster_centroids=s2.centroids,
)
```

Per-cluster metrics: reach_1R/2R/3R, MFE p25/p50/p75/p90, ww_pp,
ttp_p25/p50/p75, mean_R, final_r p25/p50.

SL sweep: {1.5, 2.0, 2.5, 3.0, 3.5, 4.0} × ATR — composite per
multiplier, argmax selects the cluster's preferred SL.

Capturability composite (w_reach=0.40, w_mfe_p50=0.40, w_1minus_ww=0.20):
```
composite = 0.40·reach_1R + 0.40·(mfe_p50/3) + 0.20·(1 − ww_pp)
```

Candidate-cluster flag (L_PROTOCOL §2 Step 3):
```
reach_1R ≥ 0.50  AND  ww_pp ≤ 0.30  AND  mfe_p50 ≥ 1.5R
```

---

## §7 Step 4 — Extraction

```python
from core.steps.step_4_extraction import run_step_4

s4 = run_step_4(
    pool.trades,
    feature_matrix,                    # per-trade entry-time features
    s2.cluster_assignments,
    feature_lineage=lineage_df,        # optional, drives suspect-exclusion
    candidate_cluster_ids=(c1.cluster_id for c1 in s3.per_cluster if c1.is_candidate),
)
```

Per candidate cluster: RF + LightGBM + LogisticRegression at
L_PROTOCOL Appendix A defaults. 5-fold TimeSeriesSplit. AUC +
threshold sweep + permutation importance.

**Lineage enforcement**: features tagged anything other than `clean`
in `feature_lineage` are excluded from training. Excluded list logged
in `s4.extraction_metrics` + `summary_md`.

`s4.per_cluster[i]`:
  - `best_classifier`: "rf" | "lgbm" | "lr"
  - `best_classifier_mean_auc`: float
  - `best_threshold`: float (AUC-best, mean across folds)
  - `feature_importance`: top features (mean ± std across folds)
  - `fitted_classifier_path`: `Path | None` — joblib pickle of the
    best-AUC classifier, refit on the full lineage-filtered pool
    after CV evaluation (`None` if `persistence_dir` was not supplied
    or the cluster produced no viable classifier)
  - `fitted_classifier_type`: e.g. `"RandomForestClassifier"`
  - `fitted_classifier_feature_order`: the column order the persisted
    classifier expects at `predict_proba` time

A2 and A6 consume `best_classifier` + `best_threshold` directly. A3
and A4 retrain their own classifier per fold (different feature set:
path-so-far, not entry-time). See
[§"Architecture-specific retraining policy" in L_PROTOCOL §2 Step 5](../L_PROTOCOL.md)
for the locked policy.

### Classifier persistence

When `run_step_4` is called with `persistence_dir`, the best-AUC
algorithm is refit on the full lineage-filtered pool for each
candidate cluster and pickled via joblib (compression level 3) to
`<persistence_dir>/<cluster_id>.pkl`. A `manifest.json` is written
alongside with SHA256 + provenance:

```json
{
  "arc_name": "l_arc_X",
  "generated_at": "2026-05-23T...",
  "train_end": "2021-01-01T00:00:00Z",
  "joblib_version": "1.5.3",
  "sklearn_version": "1.8.0",
  "lightgbm_version": "4.6.0",
  "classifiers": {
    "1": {
      "path": "1.pkl",
      "sha256": "...",
      "classifier_type": "RandomForestClassifier",
      "classifier_name": "rf",
      "feature_order": ["feat_a", "feat_b", ...],
      "best_threshold": 0.62,
      "auc_in_sample": 0.95,
      "auc_oos_cv5": 0.66,
      "trained_on_pool_size": 1288
    }
  }
}
```

A2 / A6 instantiate themselves from Step 4 output via the persistence
helpers — no retraining at Step 5:

```python
from core.steps.classifier_persistence import (
    build_a2_config_from_step4,
    build_a6_config_from_step4,
)

a2 = build_a2_config_from_step4(s4, cluster_id=1)            # uses best_threshold
a2_sweep = build_a2_config_from_step4(s4, cluster_id=1,
                                      threshold_override=0.65)
a6 = build_a6_config_from_step4(s4, cluster_id=1,
                                lower_threshold=0.4, upper_threshold=0.6)
```

`load_classifier(path)` (also in
`core.steps.classifier_persistence`) SHA256-verifies the pickle
against the manifest before loading and raises
`ClassifierIntegrityError` on mismatch. It emits a `UserWarning` on
joblib / sklearn / lightgbm version drift relative to the manifest's
recorded environment.

**Holdout exclusion.** Step 4's CV and the persisted-classifier
refit both restrict to trades with `entry_time < train_end` when a
holdout window is configured. The orchestrator threads `train_end`
from `WfoStructure.holdout.oos_start`; arcs running outside the
orchestrator pass `train_end=…` to `run_step_4` directly. The
manifest's top-level `train_end` field declares the IS cutoff (ISO
timestamp, or `null` when no holdout is configured). A2 / A6 can
then be evaluated cleanly on the holdout window because the
classifier they consume has not seen it. Per-cluster
`trained_on_pool_size` in the manifest reflects the IS-only subset.

### Orchestrator wiring for A2 / A6

`ArcOrchestrator._run_step_5` accepts an `auto_arch_specs` field on
`ArcConfig` (tuple of `AutoArchSpec(architecture, cluster_id,
builder_kwargs)`) — at Step 5 dispatch, the orchestrator looks up
each spec's cluster in the Step 4 result, calls the appropriate
`build_*_config_from_step4` helper, and wires the resulting config
into the WFO search. A1 `filter_rules` and the A2 / A6 admit gates
all read the per-trade feature dict via `A1RunContext`; the
orchestrator builds it once from `cfg.feature_matrix` and passes it
to every `ArcFoldRunner` (A1 / A5 ignore it).

---

## §8 Step 5 — Architectures

All six expose:

```python
class Architecture(Protocol):
    architecture_name: str  # "A1".."A6"

    def run(
        self, *,
        signal_evaluation: SignalEvaluation,
        panels: Mapping[str, Panel],
        fold: Fold,
        arch_config: Any,
        config_id: str,
        run_context: A1RunContext | None = None,
    ) -> StrategyResult: ...
```

Returns `StrategyResult` with `architecture`, `config_id`, `fold`,
`run_result`, `fold_stats`, `equity_curve`, `closed_trades`,
`metadata`.

### A1 — System-level filter

Base signal + optional rule-based filter from Step 4 + exposure + SL +
trail + exit. **KH-24 is the canonical A1 instantiation.**

```python
from core.architectures.a1_system_level_filter import A1Architecture, A1Config
from core.strategies.kh24.a1_adapter import kh24_to_a1
from core.strategies.kh24.kh24 import KH24Config

a1_cfg, kh24_signal = kh24_to_a1(KH24Config(), config_id="kh24_canonical")
result = A1Architecture().run(
    signal_evaluation=kh24_signal.evaluate(panels),
    panels=panels,
    fold=fold,
    arch_config=a1_cfg,
    config_id=a1_cfg.config_id,
)
```

### A2 — Classifier filter

Same as A1 plus a classifier admit gate. Uses Step 4's best
classifier + AUC-best threshold. No retrain.

### A3 — Pipeline DE (deferred entry)

NEW classifier per fold, target = cluster membership, features =
path-so-far at bar N ∈ {3, 5}. Decision at bar N; enter at bar N+1
open if admitted.

### A4 — Pipeline D (differentiated exits)

NEW classifier per fold, target = "final R > 0", features = path-so-far
each bar. If classifier confidence < `exit_threshold` ∈ {0.3, 0.4, 0.5},
queue bar-close exit. Initial SL still binds.

### A5 — Portfolio composition

Combines ≥ 2 constituents' StrategyResults at admit-only economics. No
new backtest; just merges trades + equity.

### A6 — Meta-labeling

Same classifier as A2 (Step 4 best). Confidence maps to size:
- `prob < lower`: 0x (skip)
- `lower ≤ prob < upper`: 0.5x risk
- `prob ≥ upper`: 1.0x risk

Sized via `Order.risk_multiplier` (driver multiplies position size).
Threshold pair sweep: {(0.3, 0.5), (0.4, 0.6), (0.5, 0.7)}.

---

## §8a Architecture selection — Amendment 5 four-gate (dispatch-time)

Per L_PROTOCOL Amendment 5 (ratified 2026-05-23, PR #194), the architectures
admitted for each surviving cluster are the UNION over four independent gates:

- **Gate 1 — Shape-required:** archetype-driven (V-shape → A3; Stepwise → A4;
  Bimodal → A4; Monotonic up/down / Unclassified → none).
- **Gate 2 — Classifier-driven:** if Step 4 mean OOS AUC ≥ 0.65 → add A2 and A6,
  REGARDLESS of archetype.
- **Gate 3 — Universal:** always add A1.
- **Gate 4 — Portfolio:** if ≥ 2 candidate clusters survive Step 3 → add A5.

Choppy clusters skip all architectures.

**Engine impact: zero.** All six architectures are wired post-PR-186; Amendment
5 enforcement is **dispatch-time** — the arc's `ArcConfig.architecture_configs`
tuple is curated by the dispatcher to reflect the four-gate union per cluster.
The orchestrator + `ArcFoldRunner` receive a fully-resolved set and do not
re-evaluate the gating rule. Closures record any architecture admissible under
the prior Amendment 1 rule but skipped under Amendment 5 in the optional
top-level `architectures_skipped_by_amendment_5` field (template v1.3.1).

Full rule text: L_PROTOCOL §2 Step 5 "Architecture selection (Amendment 5)"
and `archive/L_PROTOCOL_v3_0_AMENDMENT_5.md`.

---

## §8b Amendment 3 emissions (risk-normalised gates)

Per `archive/L_PROTOCOL_v3_0_AMENDMENT_3.md` + audit
`docs/audits/engine_capability_audit_2026_05.md`, the engine now emits
the full risk-normalised-gate evaluation as part of `ArcOrchestrator.run()`.

**Trigger:** automatic when `s5.top_k` is non-empty. For each top-K
candidate the orchestrator runs an Amendment 3 evaluation pass after
search + holdout complete.

**Module map:**

| Module | Responsibility |
|---|---|
| `core.wfo.amended_gates` | Pure gate logic: scaling factors, scaled DEPLOYABLE / VIABLE gates, priority-ordered failure-mode taxonomy. `classify_amended_fold_stats(...)` returns `AmendedGateResult` with every Amendment 3 tracker payload field. |
| `core.wfo.chained_dd` | Chained max DD across IS + holdout. `stitch_per_fold_oos_equity` + `compute_chained_max_dd_from_continuous_equity`. v3.0.1 uses equity stitching; v3.0.2 follow-up replaces with full-window sim per chat directive Q6. |
| `core.runners._fold_stats_helpers.compute_per_day_max_dd` | Per-day max-DD series at r_base. **5ers EET broker-day boundary** under Amendment 6 (PR #197); `boundary_convention="utc"` opt-in preserved for KH-24. Day-start equity = first equity sample of the trading day under the active convention. See §15.5 for the propagation pattern. |
| `core.wfo.holdout_rerun` | `rescale_arch_config_risk(arch_config, k_scale)` — frozen-dataclass copy with `risk_pct *= k_scale`. Used for r_safe / r_hard holdout re-runs. |
| `core.arc.arc_orchestrator._run_amendment_3_evaluation` | Per-top-K orchestration: equity stitching → chained DD → per-day DD parquet → holdout re-runs at scaled risks → amended gate classify. |

**Locked thresholds (per `core/wfo/amended_gates`):**

- `R_MIN = 0.15%`, `R_MAX = 2.00%` — scalability bounds (both r_safe and r_hard must fall in this range; failure → `step5_not_scalable`)
- `CHAINED_DD_MAX_PCT = 10%` — scaled chained DD ceiling
- Daily-DD breach threshold: 5% of day-start equity at scaled risk (exactly 0 breaches permitted in both tiers)

**Scaling rule:**

```
k_safe  = 8.0  / worst_fold_dd_base_pp     # 8 = 8 percentage-point DEPLOYABLE cap
k_hard  = 10.0 / worst_fold_dd_base_pp     # 10 = 10pp VIABLE / 5ers hard cap
r_safe  = r_base × k_safe
r_hard  = r_base × k_hard
```

**Failure-mode priority** (first-fail wins, per Amendment 3 §"Failure-mode priority"):

1. `pool_too_small` (Step 1)
2. `step5_not_scalable`
3. `step5_dd_above_gate` (defensive — should not occur post-scaling)
4. `step5_chained_dd_above_gate`
5. `step5_daily_dd_breach`
6. `holdout_fail_after_is_pass`
7. `step5_negative_folds` / `step5_sign_consistency_fail`
8. `step5_trade_count_below_gate`
9. `step5_wf_roi_below_gate_after_scaling`
10. `step5_ratio_below_gate_after_scaling`
11. `step6_causal_audit_fail`

**Artefacts emitted:**

- `results/<arc>/step_5/per_day_max_dd_base__<safe_cid>.parquet` per top-K candidate. Schema: `date, pair_set, day_start_equity, day_max_dd_base_pct, n_trades_open_start_of_day`.
- `AmendedWfoSearchResult.amended_results[*]` per top-K. Read via `ArcOrchestratorResult.amended_wfo`. Extension dataclass — does NOT amend `WfoSearchResult` in place (Q4 backwards-compat directive).

**Holdout re-runs at scaled risk:**

Per Amendment 3 §"Engine-side changes" #4 + §5.5: for each top-K
candidate the orchestrator runs TWO additional holdout sims — one
at `r_safe` (for DEPLOYABLE evaluation), one at `r_hard` (for VIABLE
evaluation). Scaling done via `rescale_arch_config_risk`. Each sim's
`config_id` carries the scaled-risk suffix (e.g.
`a1_e2e_test_r0.0100`), so downstream artefacts can distinguish them.

**Sizing convention:**

Every arch config has a `sizing_convention: str = "reset_floor"` field;
`ArcConfig.accept_equity_pct: bool = False` is the chat-approval
override. Equity-pct sizing FAILs the scalability gate by default —
linear DD scaling holds only under reset-floor sizing.

**A4 same-bar exit precedence (per L_PROTOCOL §2 Step 5 lock):** when
trailing-stop and A4 classifier-exit fire on the same bar close, the
trail-stop wins. Intra-bar SL/TP remains the highest-priority exit.
Implementation: `core/sim/multipair_backtester.py:_process_bar` step 3
uses direct `_pending_closes[pos_id] = "trailing_stop"` (the prior
`setdefault` behaviour gave priority to the predicate by accident of
evaluation order — corrected per chat directive Q3).

**Equity-stitching caveat (v3.0.2 follow-up):** chained DD is computed
from per-fold OOS equity series multiplicatively chained with
continuity adjustment in v3.0.1. Per chat directive Q6 the gold
standard is a single full-window sim spanning IS + holdout per
top-K candidate; deferred to a v3.0.2 follow-up (concern: per-fold
classifier-selection at fold boundaries for A3 / A4 adds complexity
outside this PR's scope contract). Documented in
`core/wfo/chained_dd.py` module docstring.

---

## §8c Exit-policy registry

`core/sim/exit_policies/` holds the canonical exit-policy registry for
the v3 multipair backtester. Each registered policy defines the
post-SL behaviour of a position (TP placement, trailing logic,
partial-close lifecycle) and is consumable by every architecture
(A1..A6) via `arch_config.exit_policy = "<name>"`. Default `None`
preserves prior behaviour (SL + optional `TrailManager` + signal-class
exit predicates only — e.g. KH-24's `kijun_d1` path).

### Registered policies

| Name | Semantics summary | Reference |
|---|---|---|
| `sl_only` | Baseline. SL + (optional) signal-class predicates only. | [sl_only.py](../core/sim/exit_policies/sl_only.py) / [step_5.py:143-151](../scripts/l_arc_10_v3/step_5.py) |
| `sl_plus_tp_2r` | SL + intra-bar TP at `entry + 2 × R_atr`. Decorates `Order.tp_price`; uses existing intra-bar TP infrastructure. | [sl_plus_tp_2r.py](../core/sim/exit_policies/sl_plus_tp_2r.py) / [step_5.py:153-161](../scripts/l_arc_10_v3/step_5.py) |
| `sl_plus_tp_3r` | Same as TP_2R at `+3R`. | [sl_plus_tp_3r.py](../core/sim/exit_policies/sl_plus_tp_3r.py) / [step_5.py:163-171](../scripts/l_arc_10_v3/step_5.py) |
| `sl_plus_trailing_atr` | Activate at MFE ≥ +1R; trail at `peak_high − R_atr`; bar-close eval; exit at next-bar open. | [sl_plus_trailing_atr.py](../core/sim/exit_policies/sl_plus_trailing_atr.py) / [step_5.py:173-193](../scripts/l_arc_10_v3/step_5.py) |
| `sl_plus_trailing_swing` | Activate at MFE ≥ +1R; trail = running max of `min(prev_close, entry)`; capped at entry by design. | [sl_plus_trailing_swing.py](../core/sim/exit_policies/sl_plus_trailing_swing.py) / [step_5.py:195-217](../scripts/l_arc_10_v3/step_5.py) |
| `sl_partial_close_1r_runner_trail` | Intra-bar partial close 50% at `entry + R_atr`; runner trails at `peak_high − R_atr` (path-wide peak); SL still binds on runner. | [sl_partial_close_1r_runner_trail.py](../core/sim/exit_policies/sl_partial_close_1r_runner_trail.py) / [step_5.py:219-250](../scripts/l_arc_10_v3/step_5.py) |

`R_atr = sl_atr_mult × atr_at_entry` — the size of 1R in price units,
computed at trade fill against the actual entry price and the
mid-anchored ATR per PR #189 §15.1. The registry is SL-multiplier
agnostic: policies anchor in R-units; the live engine carries the
fill price through the policy state machine.

### Architecture wiring

```python
A1Config(
    config_id="kh24_with_partial_close",
    sl_atr_mult=2.0,
    trail_enabled=False,             # canonical trail off
    exit_policy="sl_partial_close_1r_runner_trail",
    ...,
)
```

The architecture's `run(...)` instantiates an `ExitPolicyManager()`
when `exit_policy is not None` and passes it to `MultiPairBacktester(
exit_policy_manager=...)`. The driver:

  1. Calls `policy.apply_to_order(ctx)` at trade fill time (anchored to
     actual fill price). TP-style policies populate `Order.tp_price`
     here so the existing intra-bar TP infrastructure handles fire+fill
     at the exact TP level (realised R = exactly +2R / +3R).
  2. Calls `manager.evaluate_intrabar_for_all(...)` BEFORE intra-bar
     SL/TP — partial-close at +1R fires on bar high before SL is
     evaluated against bar low. The manager's
     `has_intrabar_partial_this_bar(pos_id)` flag SUPPRESSES same-bar
     intra-bar SL/TP for the position that just partial-closed
     (matches reference's `sl_breach > tp1_i` constraint so the runner
     survives the same-bar low).
  3. Calls `manager.evaluate_at_close_for_all(...)` AFTER trail-manager
     ratchet. Trailing-style policies and partial-close runner-trail
     fire here; queued at next-bar open per existing pattern.

Account partial-fill semantics: see `core/sim/account.py:partial_close`
+ `current_size_of` + `ClosedTrade.parent_position_id`. Position is
frozen; Account owns the live size via a shadow dict.

### Same-bar precedence

Intra-bar SL/TP > intra-bar policy (partial-close) > signal-class
predicates (bar-close) > trail-manager (bar-close ratchet) > at-close
policy (last-write-wins). Documented per architecture in each
`AnConfig.exit_policy` field's docstring.

### Path-replay (legacy Step 5 fast path)

`core/sim/exit_policies/path_simulate.py` exposes the SAME canonical
semantics applied to recorded path data (`mae_so_far_r / mfe_so_far_r /
close_r / is_held` per-trade columns). Used by `scripts/l_arc_*/step_5.py`
post-hoc evaluators (Arc 10 v3 = full path replay; Arc 8 = the coarser
`simulate_pool_approximation` legacy fast path).

```python
from core.sim.exit_policies import simulate_path
final_r_new, bars_held = simulate_path(
    "sl_partial_close_1r_runner_trail",
    trade_row,
    path_rows,
    sl_mult=2.5,
)
```

Both execution paths (live bar-by-bar AND replay) share ONE definition
per policy. Reference-parity test
[tests/sim/exit_policies/test_path_simulate_reference_parity.py](../tests/sim/exit_policies/test_path_simulate_reference_parity.py)
asserts byte-identical behaviour between the canonical replay module
and the historical hand-rolled simulator
([scripts/l_arc_10_v3/step_5.py:99-253](../scripts/l_arc_10_v3/step_5.py))
across all 6 policies × 4 SL multipliers × 9 synthetic path scenarios
(218 cases).

### KH-24 invariant

KH-24 uses `exit_policy=None` (its production parameters are SL +
trail + `kijun_d1` predicate, all unchanged). The canonical exit-policy
code paths are dormant on the KH-24 anchor. Anchor regression
unaffected.

---

## §9 Step 5 fold runners

```python
from core.runners.arc_fold_runner import ArcFoldRunner
from core.runners.oracle_fold_runner import OracleFoldRunner
from core.wfo.orchestrator import run_search, run_holdout

runner = ArcFoldRunner(
    architecture=A1Architecture(),
    signal_evaluation=signal_eval,
    panels=panels,
)
search = run_search(wfo_struct, candidates, fold_runner=runner)
holdout = run_holdout(wfo_struct, search.top_k, fold_runner=runner)
```

Oracle runner: admits only trades belonging to a target cluster using
post-hoc Step 2 cluster_id (NON-DEPLOYABLE — diagnostic upper bound only).

---

## §10 ArcOrchestrator (Steps 1→5 end-to-end)

```python
from core.arc.arc_orchestrator import ArcConfig, ArcOrchestrator
from core.architectures.a1_system_level_filter import A1Architecture, A1Config
from core.strategies.kh24.a1_adapter import kh24_to_a1
from core.strategies.kh24.kh24 import KH24Config

a1_cfg, kh24_signal = kh24_to_a1(KH24Config())
arc_cfg = ArcConfig(
    arc_name="kh24_canonical",
    signal_class="kb_exhaustion_bar",
    pair_set=("EURUSD", "GBPUSD", ...),
    sl_atr_mult=2.0,
    hold_bars=240,
    architectures=(A1Architecture(),),
    architecture_configs=(a1_cfg,),
)
orch = ArcOrchestrator(arc_cfg, kh24_signal, panels)
result = orch.run()
orch.write(result, "results/kh24_v3_e2e")
```

The orchestrator:

1. Runs Step 1 via `build_arc_pool`
2. Runs Step 2 / 3 / 4 via the step runners
3. Runs Step 5 by passing (arch, config) candidates through
   `core.wfo.orchestrator.run_search` + `run_holdout`
4. Writes ARC_OPEN.md + ARC_CLOSURE.md skeleton + step artefacts

Step 6 (causal audit) is **post-gate auto-dispatched** per
Amendment 4 — see §10b below.

---

## §10b Step 6 — causal audit framework (Amendment 4)

`core/step_6/` implements the six-category causal-audit framework that
L_PROTOCOL Amendment 4 (2026-05-24) ratifies. Step 6 dispatches AFTER
the amended gate (Amendment 3) clears §3 constraints #1-9 on at least
one top-K candidate — per §3 "Evaluation order" item 2.

**Module map:**

```
core/step_6/
├── __init__.py
├── inputs.py                 # Step6Inputs bundle
├── manifest.py               # CheckResult, CategoryAuditResult, Step6Result,
│                             # Step6Manifest, Severity, VerdictImpact
├── orchestrator.py           # run_step_6() dispatcher
├── byte_compare.py           # Generic feature-producer byte-compare harness
├── artefacts.py              # markdown/manifest writers
├── io.py                     # from_arc_orchestrator_result + from_closure_dir
├── dispatch.py               # maybe_dispatch_step_6 + replace_top_1_with_step6_fail
├── lookahead.py              # §6.1
├── selection_bias.py         # §6.2
├── execution_realism.py      # §6.3
├── statistical.py            # §6.4
├── determinism.py            # §6.5
└── deployment_readiness.py   # §6.6
```

**Trigger semantics** (chat resolution Q1):

1. `_run_amendment_3_evaluation` produces per-top-K
   :class:`AmendedGateResult` with `causal_audit_clean=True` (default).
2. If at least one top-K candidate has verdict PASS-DEPLOYABLE / PASS-VIABLE,
   `maybe_dispatch_step_6` runs Step 6 on the **Top-1** candidate (chat
   resolution Q2).
3. If Step 6 produces ≥ 1 critical failure, `replace_top_1_with_step6_fail`
   re-classifies the Top-1's gate with `causal_audit_clean=False` →
   `primary_failure_mode = step6_causal_audit_fail`.
4. Top-2 / Top-3 candidates are NOT downgraded; feature-set divergence
   surfaces as a `top_k_feature_set_divergence` warning on the lookahead
   category.

**Severity rules** (chat resolution Q4):

- `critical` failure → category FAIL → Step 6 FAIL → verdict downgrade
- `warning` failure → category PASS but flagged; counted in `n_warnings`
- `info` failure → recorded; no effect on category outcome

`CategoryAuditResult.passed = AND over critical-severity checks only`.

**Manual CLI** — `scripts/run_step_6.py`:

```bash
# Run all six categories on a closure
python scripts/run_step_6.py results/l_arc_10/ARC_CLOSURE.md

# Restrict to one category
python scripts/run_step_6.py results/l_arc_10/ARC_CLOSURE.md --category lookahead

# Demote critical failures to warnings in the report (still no verdict modification)
python scripts/run_step_6.py results/l_arc_10/ARC_CLOSURE.md --no-block

# Validate inputs without running audits
python scripts/run_step_6.py results/l_arc_10/ARC_CLOSURE.md --dry-run
```

Manual invocations write to `results/<arc>/step_6_manual_<timestamp>/`
and **never modify the verdict** (chat resolution Q6).

**Artefact layout** (per auto-dispatched run):

```
results/<arc>/step_6/
├── manifest.json               # Step6Manifest (per dispatch Task 4 schema)
├── summary.md                  # one-page roll-up
├── lookahead_report.md
├── selection_bias_report.md
├── execution_realism_report.md
├── statistical_report.md
├── determinism_report.md
├── deployment_readiness_report.md
└── sha256_manifest.json        # per-file sha256 (excludes itself)
```

**Closure integration** — closure template v1.3 carries a new
`§1 tracker_payload.step_6` block keyed off `Step6Manifest`. Parser
v1.3 detection precedence: explicit `template_version: v1.3` → top-level
`step_6` field → fall-through to v1.2. Phase 2 tightening: for any PASS
verdict with `closed_timestamp > 2026-05-23T06:20:59Z` (PR-186 merge),
Amendment 3 fields required; v1.3 PASS verdicts additionally require
`step_6.overall_passed: true`.

**Backwards compatibility:**

- v1.0 / v1.1 / v1.2 / v1.2.1 closures grandfathered. Parser handles
  them via existing detection.
- Arc 10's hand-written Step 6 at `results/l_arc_10/step_6/` stays
  canonical; manual CLI re-runs land in `step_6_manual_<ts>/` and do
  NOT clobber.
- Wave 2 onward closes at v1.3; Step 6 auto-dispatches.

---

## §11 Sub-protocol mechanism

`core/arc/sub_protocol.py` exposes a registry-style hook. At v3.0 no
sub-protocols are registered; the registry is empty. Each step
consults `resolve_step_override(sub_protocol_name, "step_N")`; if
None, vanilla overseer runs.

To register a new sub-protocol (e.g. `heavy_ml_probe`):

1. Add the methodology doc at `docs/sub_protocols/heavy_ml_probe.md`
2. Add the Python module at `core/sub_protocols/heavy_ml_probe.py`
3. Call `register_sub_protocol("heavy_ml_probe", {"step_4": replace_step_4})`
   (typically at import time)
4. Set `arc_config.sub_protocol = "heavy_ml_probe"` to invoke

CC_07 ships the hook but no sub-protocols. Phase 2 work adds them.

---

## §12 Determinism contract

Every step output is reproducible:

- `core/arc/arc_pool_builder.py` — `pool.pool_sha256` byte-identical
  across runs
- `core/steps/step_2_clustering.py` — `step_2_sha256()` matches
- `core/steps/step_3_capturability.py` — `step_3_sha256()` matches
- `core/steps/step_4_extraction.py` — `step_4_sha256()` matches

`random_state=42` everywhere; `n_jobs=1`; `lineterminator='\n'`.

Smoke tests under `tests/protocol_runtime/` enforce two-run sha256
identity at every step.

---

## §13 Warmup convention (kijun-class features)

Per the CC_07 anchor bisect (2026-05-22, `scripts/anchor/bisect_warmup.py`),
the runtime adopts a **full-history warmup convention** for any feature
that requires a multi-bar rolling lookback on an auxiliary TF (kijun,
ATR, multi-TF lookbacks).

Specifically: the SignalModule evaluates signal-class features
(signal_mask, atr, additional_gates, exit_predicate) ONCE on the
**full panel** (entire available history before the OOS window). For
each WFO fold, the driver iterates only the OOS-window slice but reads
pre-computed series via `.reindex(sliced_index)`.

This differs from the legacy `KH24FoldRunner` convention which sliced
the panel to `[oos_start − 30 days, oos_end]` before signal evaluation.
The 30-day slice was buffer-management heuristic, NOT a methodology
choice — it NaN-masked `kijun(26)` on the first OOS bars when
weekends/holidays consumed the warmup buffer. CC_07's bisect (see
`results/anchor_kh24_bisect_warmup/`) confirmed: when the legacy path
runs with `warmup_days=365`, it converges to the A1 numbers
byte-identically across all 7 KH-24 anchor folds.

The full-history warmup is the correct version. The legacy
`KH24FoldRunner` is retained in `core/wfo/fold_runner.py` as a
regression baseline; new arcs use `ArcFoldRunner` (full-history
warmup).

## §14 KH-24 anchor preservation

KH-24 is the canonical A1 instantiation. The equivalence chain:

```
KH24FoldRunner(panels, KH24Config())  with warmup_days=365 internal
  ≡  ArcFoldRunner(A1Architecture(), kh24_signal_evaluation, panels)(fold, kh24_to_a1(KH24Config())[0])
```

The two paths produce byte-identical per-fold ROI / DD / trade count
on the full-history-warmup convention (verified 2026-05-22 across all
7 KH-24 anchor folds, see `results/anchor_kh24_bisect_warmup/summary.md`).

The default `KH24FoldRunner` `warmup_days=30` produces slightly
different F2/F3 numbers (see [BACKTESTER_ARCHITECTURE.md §B.1](BACKTESTER_ARCHITECTURE.md))
because the short warmup NaN-masks `kijun(26)` at fold boundaries
where weekends/holidays consume the buffer. Both paths are deterministic;
they differ only in their warmup-window convention.

`tests/protocol_runtime/test_kh24_a1_equivalence.py` runs the
structural equivalence check on the synthetic mini-fixture
(`tests/fixtures/histdata_mini/`). Full-data verification uses
`scripts/anchor/check_a1_equivalence.py` against
`scripts/anchor/bisect_warmup.py` (chat-side workstation runs).

---

## §15 Signal parity (mid-price + 5ers EET + worst-case fills) — PR #189

The v3 engine produces venue-independent signals. Three invariants:

### §15.1 Mid-price feature computation

All Step-1 price-derived features (price_geometry, vol_regime, multi_tf,
cross_pair, distance, plus the load-bearing Arc 10 feature
`L1_minus_L0_atr` in [core/features/multi_tf.py](../core/features/multi_tf.py))
compute on **mid price** = `(close_bid + close_ask) / 2` per OHLC field.
Spread is treated as pure cost, not as feature input.

Exempt by design: `spread_regime.*` features read `spread_close` as a
structural regime indicator (preserved per dispatch B.5). These do not
leak bid/ask asymmetry into per-bar price-derived computations — they
report the regime, they don't shape it.

D1 lag rule (L_PROTOCOL §1 non-negotiable) is unchanged: features that
use D1 lag-1 close now use D1 lag-1 **mid** close. The lag itself —
`iClose(D1, 1)` semantics, enforced by
[`_build_d1_lag1_series`](../core/features/multi_tf.py) via
`merge_asof(direction="backward")` — is unaffected.

Verification: [tests/test_feature_parity_mid.py](../tests/test_feature_parity_mid.py)
locks the contract — same mid OHLC + different bid/ask spread → identical
mid-price features.

### §15.2 Worst-case fill execution

[core/sim/fill.py](../core/sim/fill.py) and
[core/sim/multipair_backtester.py](../core/sim/multipair_backtester.py)
already implemented worst-case fills in V3:
- Long entry: fills at `bar.open_ask`
- Short entry: fills at `bar.open_bid`
- Long SL hit: `bar.low_bid ≤ sl_price` → fills at `sl_price`
- Short SL hit: `bar.high_ask ≥ sl_price` → fills at `sl_price`
- Long TP hit: `bar.high_bid ≥ tp_price` → fills at `tp_price`
- Short TP hit: `bar.low_ask ≤ tp_price` → fills at `tp_price`

Spread cost is implicit in `(open_ask − open_bid)` and the bid/ask wing
of SL/TP — no separate "spread deduction" step is applied.

[core/sim/trailing_stop.py](../core/sim/trailing_stop.py): **trail
activation + ratchet read MID close** (PR #189 reverses PR-E.1.6's bid-only
trail); **trail hit detection still reads BID close** (worst-case-fill
realism — long exits when its bid falls to the trail level). This
asymmetric model (mid-anchored decision, bid-anchored fill) is the
dispatch's strict reading of B.3. The live EA must be updated to match
mid activation in a parallel deployment PR — until then, EA divergence
from backtest is expected on the trail-activation side.

### §15.3 5ers EET bar boundary

[core/data/aggregator.py](../core/data/aggregator.py) supports two
boundary conventions via the `boundary_convention` parameter:

- `"utc"` (default, legacy) — bars anchored to UTC midnights / hours.
- `"5ers_eet"` — bars anchored to the 5ers broker EET/EEST trading day
  using the IANA `Europe/Athens` zone (functionally identical to
  `Asia/Nicosia` for the 2010+ HistData range). DST handled automatically.

Steady-state anchors:

| TF | Winter (EET = UTC+2) | Summer (EEST = UTC+3) |
|---|---|---|
| H4 | UTC 22, 02, 06, 10, 14, 18 | UTC 21, 01, 05, 09, 13, 17 |
| D1 | UTC 22:00 (= EET 00:00 next day) | UTC 21:00 (= EEST 00:00 next day) |
| W1 | UTC Sun 22:00 (= EET Mon 00:00) | UTC Sun 21:00 (= EEST Mon 00:00) |

Sub-hourly TFs (M5/M15/M30/H1) have bin widths smaller than the DST
shift; the UTC bin SET is identical to local-anchored bins (only the
display label differs). The 5ers_eet path stores them under the same
schema as UTC for storage parity.

**DST transition handling:**
- Spring forward (last Sunday March): EET 03:00 → EEST 04:00. The local
  day has 23 wall-clock hours. The H4 bar starting at local 00:00 on
  the DST day spans 3 wall-clock hours (4 UTC hours). Subsequent bars
  re-anchor to EEST.
- Autumn fall-back (last Sunday October): EEST 04:00 → EET 03:00. The
  local day has 25 wall-clock hours. The H4 bar starting at local
  00:00 on the DST day spans 5 wall-clock hours (4 UTC hours). The
  duplicate EET 02:00-03:00 wall-clock hour is unambiguous in the
  underlying UTC index. An extra short bar (1 UTC hour) appears at
  local "00:00 EET of next day" within the DST day's groupby — this
  is an internally consistent artefact of per-local-day re-anchoring
  and is documented in
  [docs/calibration/histdata_mt5_aggregation_parity_2026_05.md](calibration/histdata_mt5_aggregation_parity_2026_05.md).

Implementation: H4 uses per-local-date groupby + per-day
`origin="start_day"` resample because pandas' `origin="start_day"` on
a tz-aware multi-day index does NOT re-anchor at DST. D1 and W1 use
pandas' built-in `resample("1D")` / `resample("W-MON")` on the
tz-converted index, which is DST-aware out of the box.

**Cache layout:**

```
data/cache/<TF>/<PAIR>.parquet            # UTC (legacy)
data/cache/<TF>_5ers_eet/<PAIR>.parquet   # 5ers EET (new)
```

The two convention caches coexist; the convention is encoded in both
the directory name and the cache_key sidecar so cross-pollination is
impossible. UTC caches built pre-PR-#187 remain valid.

Verification:
- [tests/test_aggregator_5ers_eet.py](../tests/test_aggregator_5ers_eet.py)
  — anchor + DST + cache namespace tests
- [tests/test_aggregator.py](../tests/test_aggregator.py) — legacy UTC
  byte-identity preserved

### §15.4 Signal-module timezone-invariant HTF alignment

[core/signals/htf_alignment.py](../core/signals/htf_alignment.py) is the canonical
way for signal modules and multi-TF feature producers to look up an HTF column
value (e.g. prior-day D1 close) at an LTF anchor timestamp (e.g. each H4 bar).
Replaces the legacy UTC-anchored idioms that broke under the 5ers EET storage
convention in §15.3:

| Legacy idiom | Failure mode under EET | Canonical replacement |
|---|---|---|
| `ltf_index.floor("4h").map(idx_h4)` | State **C** — `.map()` exact-match returns NaN → empty signal pool | `get_htf_index_at(..., require_fully_closed=True)` |
| `ltf_index.normalize().map(idx_d1)` | State **C** | `get_htf_index_at(...)` or `get_htf_value_at(...)` |
| `df.index.normalize() - pd.Timedelta(days=1)` + `merge_asof(backward)` | State **B** — silently picks same-EET-day HTF (lookahead) | `get_htf_value_at(..., require_fully_closed=True)` or `get_htf_row_at(...)` |
| `d1_ts.normalize()` + `np.searchsorted` | State **B** — picks neighbouring EET-day HTF | `get_htf_index_at(...)` |

**Public API:**

```python
get_htf_value_at(current_timestamps, htf_panel, column, *, require_fully_closed=True) -> pd.Series
get_htf_row_at(current_timestamps, htf_panel, *, require_fully_closed=True) -> pd.DataFrame
get_htf_index_at(current_timestamps, htf_panel, *, require_fully_closed=True, invalid_sentinel=-1) -> np.ndarray
```

**`require_fully_closed` semantics:**

- `True` (default) — returns the most-recently *fully closed* HTF bar at each
  LTF ts (matches L_PROTOCOL §1 one-day-lag rule; byte-identical to the legacy
  KH-24 `.normalize() - Timedelta(days=1) + merge_asof(backward)` idiom under
  UTC convention). Used by KH-24, Arc 3, Arc 5, `core/features/multi_tf.py`.
- `False` — returns the HTF bar *containing* each LTF ts (no fully-closed
  back-off; caller applies its own freshness offset downstream). Used by Arc
  10 DLR, which derives `d_t - 4` for the swing-low search constraint.

**Timezone-awareness contract:** both `current_timestamps` and `htf_panel.index`
must share the same tz-awareness (both tz-aware or both tz-naive). A mismatch
raises `ValueError`. The engine guarantees tz-aware UTC output under both
`boundary_convention="utc"` and `"5ers_eet"` (per §15.3), so the contract is
auto-satisfied for any panels flowing through `core.data.aggregator`.

**Verification:**
- [tests/signals/test_htf_alignment.py](../tests/signals/test_htf_alignment.py) — 19 unit tests including byte-identical-to-legacy-KH-24 under UTC
- [tests/signals/test_htf_alignment_timezone_invariance.py](../tests/signals/test_htf_alignment_timezone_invariance.py) — 14 regression tests including a static guard that flags reintroduction of `.floor()` / `.normalize()` in fixed modules

**Audit report:** [docs/audits/signal_module_eet_audit_2026_05.md](audits/signal_module_eet_audit_2026_05.md)

### §15.5 EET session semantics — distance.py + reset_floor.py + compute_per_day_max_dd

PR #189 fixed engine-level bar aggregation under EET. PR #193 (§15.4 above)
fixed HTF-alignment lookups in signal modules. The CC_20 PR closes the
remaining session-semantics fault class — three modules where
day-bucketing was still UTC-anchored despite EET-aggregated bars. This is
a distinct fault class from §15.4: session bucketing vs HTF lookup. The
two utilities are intentionally separate.

**Single source of truth:** [core/time_utils/session_boundary.py](../core/time_utils/session_boundary.py)
exposes `utc_to_eet_trading_day(ts, *, convention="5ers_eet")` —
the only DST-aware mapping callers need. `convention="utc"` falls
back to `.normalize()` byte-identically (legacy KH-24 safety).

**Propagation pattern:** `Panel.boundary_convention` (default `"utc"`
for legacy safety) is the convention carrier. Aggregator-driven
constructors (`Panel.from_pairs`, `build_panel_parallel`) thread
the caller's choice into the Panel. Per-fold slicing
(`_slice_panels_to_fold` in [core/architectures/a1_system_level_filter.py](../core/architectures/a1_system_level_filter.py),
`_slice_panel_by_dates` in [core/wfo/fold_runner.py](../core/wfo/fold_runner.py))
preserves it via `Panel.from_frames(..., boundary_convention=panel.boundary_convention)`.

**Three convention-aware consumers:**

1. **[core/features/distance.py](../core/features/distance.py)** —
   `_prior_session_high` / `_prior_session_low` bucket bars by
   trading day via `utc_to_eet_trading_day(df.index, convention=...)`
   keyed on `panel.boundary_convention`. UTC fallback when
   `panel=None` preserves legacy behaviour for callers that don't
   provide a panel.

2. **[core/sim/risk/reset_floor.py](../core/sim/risk/reset_floor.py)** —
   `ResetFloorAccount` takes a `boundary_convention` kwarg
   (default `"5ers_eet"`). The daily-close ratchet uses the
   utility. Forward-hygiene fix — the module is dormant in v3
   runtime (no architecture currently instantiates one) but the
   convention surface is wired for future callers.

3. **[core/runners/_fold_stats_helpers.py](../core/runners/_fold_stats_helpers.py)** —
   `compute_per_day_max_dd` takes `boundary_convention` (default
   `"5ers_eet"`). This is the **load-bearing fix per Amendment 6**:
   the orchestrator's per-day max-DD parquet (which feeds
   `daily_dd_breaches_at_r_safe` / `daily_dd_breaches_at_r_hard`
   gates in [core/wfo/amended_gates.py](../core/wfo/amended_gates.py))
   now buckets equity by the EET trading day, matching 5ers'
   actual daily-DD reset boundary. The orchestrator
   ([core/arc/arc_orchestrator.py](../core/arc/arc_orchestrator.py)
   `_run_step_5`) reads `panel.boundary_convention` and forwards
   it explicitly.

**Amendment 6** (supersedes Amendment 3 §"Boundary"): the
daily-DD measurement boundary is the EET broker trading day
post-PR-189. The locked value in Amendment 3 §"Boundary"
(`UTC broker-day. Locked value.`) was authored under the pre-PR-189
UTC-bar engine assumption and is amended to EET to keep the
boundary consistent with the bars.

**KH-24 anchor preservation:** KH-24 runs `convention="utc"`
end-to-end. Under UTC convention every consumer above takes the
legacy code path (`.normalize()`-equivalent bucketing) — byte-
identical to pre-CC_20 output. Anchor regression verified by
[tests/protocol_runtime/test_kh24_a1_equivalence.py](../tests/protocol_runtime/test_kh24_a1_equivalence.py)
+ [tests/replays_v2_1_1/](../tests/replays_v2_1_1/).

Verification:
- [tests/utils/test_session_boundary.py](../tests/utils/test_session_boundary.py)
  — DST, winter/summer anchors, UTC pass-through (16 tests)
- [tests/features/test_distance_eet_session_semantics.py](../tests/features/test_distance_eet_session_semantics.py)
  — prior-session bucket-shift visible at EET boundary bar; sha256-deterministic
- [tests/sim/risk/test_reset_floor_eet_daily_bucket.py](../tests/sim/risk/test_reset_floor_eet_daily_bucket.py)
  — same-EET-day idempotency vs UTC-day double-ratchet; sha256-deterministic
- [tests/runners/test_per_day_max_dd_eet.py](../tests/runners/test_per_day_max_dd_eet.py)
  — drawdown spanning UTC midnight unifies into one EET day → one breach
  (zero under UTC bucketing); sha256-deterministic

---

## §16 What lives elsewhere

- Backtester (M1 loader, TF aggregator, fill, sim, account, panel,
  WFO folds + gates + orchestrator) — [BACKTESTER_ARCHITECTURE.md](BACKTESTER_ARCHITECTURE.md)
- Methodology (gates, verdicts, anchor invariant) — [L_PROTOCOL.md](../L_PROTOCOL.md)
- Operations (dispatch pattern, branches, halts) — [WORKFLOW.md](../WORKFLOW.md)
- KH-24 system lock — [KH24_SYSTEM_LOCK.md](KH24_SYSTEM_LOCK.md)
- Per-arc closure docs — `docs/archive/arc_results/`

---

End.
