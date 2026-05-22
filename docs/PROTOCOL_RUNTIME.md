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

A2 and A6 consume `best_classifier` + `best_threshold` directly. A3
and A4 retrain their own classifier per fold (different feature set:
path-so-far, not entry-time).

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

Step 6 (causal audit) is **lazy**: not invoked by the orchestrator
unless a candidate clears PASS-DEPLOYABLE / PASS-VIABLE — at v3.0
launch it's a stub that says "deferred to chat."

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

## §13 KH-24 anchor preservation

KH-24 is the canonical A1 instantiation. The equivalence chain:

```
KH24FoldRunner(panels, KH24Config())
  ≡  ArcFoldRunner(A1Architecture(), kh24_signal_evaluation, panels)(fold, kh24_to_a1(KH24Config())[0])
```

The two paths should produce identical per-fold ROI / DD / trade
count. Tolerance per L_PROTOCOL §8: ±0.5pp ROI / ±1pp DD against the
v3 anchor reference (`scripts/anchor/run_anchor.py` mode A output).

`tests/protocol_runtime/test_kh24_a1_equivalence.py` runs the
structural equivalence check on the synthetic mini-fixture
(`tests/fixtures/histdata_mini/`). Full-data verification needs real
HistData — chat runs `scripts/anchor/check_a1_equivalence.py` on the
workstation.

---

## §14 What lives elsewhere

- Backtester (M1 loader, TF aggregator, fill, sim, account, panel,
  WFO folds + gates + orchestrator) — [BACKTESTER_ARCHITECTURE.md](BACKTESTER_ARCHITECTURE.md)
- Methodology (gates, verdicts, anchor invariant) — [L_PROTOCOL.md](../L_PROTOCOL.md)
- Operations (dispatch pattern, branches, halts) — [WORKFLOW.md](../WORKFLOW.md)
- KH-24 system lock — [KH24_SYSTEM_LOCK.md](KH24_SYSTEM_LOCK.md)
- Per-arc closure docs — `docs/archive/arc_results/`

---

End.
