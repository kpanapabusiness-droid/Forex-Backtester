# TOOL_REGISTRY — what discovery arcs CALL vs BUILD

> Two tiers of code, two trust rules. This registry is the single place an arc checks before
> writing a line of measurement or experiment code. **[`DISCOVERY_PROTOCOL.md`](./DISCOVERY_PROTOCOL.md)
> is authoritative**; this file is the operational lookup it points at.

## The load-bearing distinction (governs everything below)

- **MEASUREMENT tools** = what *does* the testing: the WFO runners, cost application, scoring /
  `FoldStats`, the pool/population build, the fold structures, the discovery judge, and the
  SL-honest engine. **A bug here is invisible to the gate — the gate IS this code** (exactly the
  Arc-10 failure class). These are **CANONICAL / LOCKED: arcs CALL them, never reimplement.** If
  one looks wrong, an arc FLAGS it in its arc doc (code is human-gated, protocol §9) — it does not
  patch it mid-run.
- **EXPERIMENT tools** = what is *being* tested: filters, clustering methods, transforms, exit
  policies, signal logic, feature constructions, soundness controls (e.g. a random-entry null).
  **A bug here fails the WFO and dies loudly.** CC builds these FREELY, registers them in the BUILT
  section, and reuses them across arcs.

The rule is NOT "CC can't code." It is: **CC builds experiment tools freely; CC calls measurement
tools always** (never re-rolls the apparatus).

## Why this registry exists (the Arc-0 survey)

Arc 0 (the supervised trial) was surveyed against the in-tree core. The finding:

- **Arc 0 CALLED canonical measurement for essentially everything** — `build_arc_pool`,
  `run_step_2` / `run_step_3`, `ArcFoldRunner`, `OracleFoldRunner`, `A1Architecture` / `A1Config`,
  `build_v3_folds`, `Panel.from_pairs`. It did **not** re-roll the engine / WFO / cost / scoring
  core. Good.
- **What it re-rolled in scratch (`_arc0_work/`) was avoidable:** (1) per-stage *driver
  boilerplate* (load → build → loop → summarize), (2) **per-year OOS folds** (2021-present) — hand
  built because `build_v3_folds` exposes 2021+ only as one locked holdout, (3) the **all-folds-
  positive discovery judge** — distinct from the L_PROTOCOL dual-tier gate, so it had no canonical
  home, and (4) a **random-entry NULL baseline** — written *twice* (in `wfo_validate.py` and
  `null_compare.py`), with no canonical or reusable equivalent (the only prior one is retired-era,
  archived under `attic/`).
- **Fix landed by this registry's PR:** (2) and (3) are MEASUREMENT and are now canonical
  (`core/wfo/discovery_measure.py`); (1) is removed by the **standard entry point** below; (4) is an
  EXPERIMENT tool → the first expected **BUILT** entry (built by the first arc that needs it, under
  `discovery/tools/`, then registered here).

---

## The standard measurement entry point (call this — do not re-roll a driver)

A discovery arc runs its measurement by CALLING the canonical pieces in this sequence. Copy/adapt;
do not rewrite the apparatus. (`PAIRS`, the signal module, and config are the arc's own; everything
imported below is LOCKED.)

```python
from datetime import date

# (1) load the panel — canonical loader, real bid/ask, EET sessions
from core.sim.panel import Panel
panel = Panel.from_pairs(
    PAIRS, tf="H4",
    histdata_root=r"C:\Users\panap\histdata_backup",  # the recovered 65 GB corpus (see README)
    cache_root="data/cache", boundary_convention="5ers_eet",
)

# (2) ex-ante population  (protocol alias: "build_ex_ante_bounded_population" → build_arc_pool)
from core.arc.arc_pool_builder import ArcPoolConfig, build_arc_pool
pool = build_arc_pool(my_signal, {"H4": panel}, ArcPoolConfig(
    arc_name="arc_<id>_<slug>", sl_atr_mult=2.0, hold_bars=120, risk_pct=0.005,
    window_start=date(2010, 1, 1), window_end=date(2020, 12, 31),
))

# (3) characterize — cluster + capturability
from core.steps.step_2_clustering import run_step_2
from core.steps.step_3_capturability import run_step_3
s2 = run_step_2(pool.trades, pool.paths)
s3 = run_step_3(pool.trades, pool.paths, s2.cluster_assignments, cluster_centroids=s2.centroids)

# (4) cheap kills — oracle-best-cluster ceiling + raw triage on representative folds
from core.architectures.a1_system_level_filter import A1Architecture, A1Config
from core.runners.arc_fold_runner import ArcFoldRunner
from core.runners.oracle_fold_runner import OracleFoldRunner
from core.wfo.folds import build_v3_folds
sig_eval = my_signal.evaluate({"H4": panel})
cfg = A1Config(config_id="arc_<id>", exit_policy="sl_partial_close_1r_runner_trail", sl_atr_mult=2.0)
raw_runner = ArcFoldRunner(architecture=A1Architecture(), signal_evaluation=sig_eval, panels={"H4": panel})
oracle = OracleFoldRunner(signal_evaluation=sig_eval, panels={"H4": panel},
                          cluster_assignments=s2.cluster_assignments,
                          candidate_cluster_id=best_cid, trades=pool.trades)

# (5) validate — full honest WFO over IS folds + per-year OOS, then the discovery judge
from core.wfo.discovery_measure import (
    build_oos_year_folds, judge_all_folds_positive, run_config_over_folds,
)
is_folds = [f for f in build_v3_folds().folds if f.is_days >= 365]   # 2010-2020 search folds
oos_folds = build_oos_year_folds(start_year=2021)                    # per-year holdout 2021-present
is_verdict = judge_all_folds_positive(run_config_over_folds(raw_runner, is_folds, cfg))
oos_verdict = judge_all_folds_positive(run_config_over_folds(raw_runner, oos_folds, cfg))
# PASS the discovery judge  ⇔  is_verdict.all_folds_positive AND oos_verdict.all_folds_positive
```

Costs (FundedNext: 1.5× spread, 0.5 pip/fill slippage, $5/lot RT, no swaps), the SL-first
take-the-loss invariant and EET daily-DD bucketing are all applied *inside* the runner — the arc
never re-derives them.

---

## CANONICAL — LOCKED (call, never reimplement)

> Measurement apparatus. The gate IS this code. Calling only; FLAG, never patch, mid-run.

| Tool | What it does | Path | How to call |
|---|---|---|---|
| `Panel.from_pairs` | Load multi-pair OHLCV panel (real bid/ask, EET sessions, parquet cache) | `core/sim/panel.py` | `Panel.from_pairs(PAIRS, tf="H4", histdata_root=..., cache_root="data/cache", boundary_convention="5ers_eet")` |
| `build_arc_pool` + `ArcPoolConfig` | Ex-ante population build (the protocol's "ex-ante bounded population"); returns `ArcPool(.trades, .paths, .signal_evaluation, .pool_sha256, ...)` | `core/arc/arc_pool_builder.py` | `build_arc_pool(signal_module, {"H4": panel}, ArcPoolConfig(arc_name=..., sl_atr_mult=2.0, hold_bars=120, risk_pct=0.005, window_start=..., window_end=...))` |
| `run_step_2` | KMeans path-shape clustering (K∈2..6, silhouette pick); returns `Step2Result(.cluster_assignments, .centroids, .k_selected, ...)` | `core/steps/step_2_clustering.py` | `run_step_2(pool.trades, pool.paths)` |
| `run_step_3` | Per-cluster capturability + `is_candidate` flag; returns `Step3Result(.per_cluster, ...)` | `core/steps/step_3_capturability.py` | `run_step_3(pool.trades, pool.paths, s2.cluster_assignments, cluster_centroids=s2.centroids)` |
| `ArcFoldRunner` | Per-fold IS/OOS run → `FoldStats`; routes through the architecture → `MultiPairBacktester` → cost netting | `core/runners/arc_fold_runner.py` | `ArcFoldRunner(architecture=A1Architecture(), signal_evaluation=sig_eval, panels={"H4": panel})(fold, cfg)` |
| `OracleFoldRunner` | Oracle-best-cluster CEILING (perfect-hindsight cluster membership) → `FoldStats`; diagnostic upper bound, NOT deployable | `core/runners/oracle_fold_runner.py` | `OracleFoldRunner(signal_evaluation=sig_eval, panels={"H4": panel}, cluster_assignments=..., candidate_cluster_id=..., trades=pool.trades)(fold, cfg)` |
| `A1Architecture` + `A1Config` | System-level rule-filter architecture (no ML training). Other wired: A2 classifier, A3/A4 pipeline, A6 meta-labeling | `core/architectures/` | `A1Config(config_id=..., exit_policy="sl_partial_close_1r_runner_trail", sl_atr_mult=2.0, risk_pct=0.005, ...)` |
| `build_v3_folds` + `Fold` | 11 expanding-IS search folds (1-yr OOS each, 2010-2020) + 1 locked holdout | `core/wfo/folds.py` | `[f for f in build_v3_folds().folds if f.is_days >= 365]` |
| `build_oos_year_folds` | **Per-year OOS folds (2021-present)** — IS pinned to the 2010-2020 dev window, one fold per holdout year (the discovery OOS judge set) | `core/wfo/discovery_measure.py` | `build_oos_year_folds(start_year=2021)` |
| `judge_all_folds_positive` | **The DISCOVERY judge** — all-folds-positive (every fold ROI > 0); returns `DiscoveryVerdict`. SEPARATE from the L_PROTOCOL dual-tier gate | `core/wfo/discovery_measure.py` | `judge_all_folds_positive(fold_stats_seq)` |
| `run_config_over_folds` | Thin loop: run one config across folds via the canonical runner → `tuple[FoldStats]` | `core/wfo/discovery_measure.py` | `run_config_over_folds(runner, folds, cfg)` |
| `run_search` | Multi-config / multi-fold search orchestrator (wraps a per-fold runner; ranks via the L_PROTOCOL gate) — use when sweeping configs | `core/wfo/orchestrator.py` | `run_search(structure, candidates, fold_runner, min_is_days=365, top_k=3)` |
| `build_fold_stats_from_run` + `FoldStats` | **Cost chokepoint** — nets FundedNext costs (`apply_cost_model`) and emits per-fold `FoldStats(roi_pct, max_dd_pct, n_trades, days_breaching_daily_5pct, roi_dd_ratio, fold_id)` | `core/runners/_fold_stats_helpers.py` (+ `core/wfo/gates.py`) | called *inside* the runner; arcs read the returned `FoldStats` |
| `CostModel.fundednext()` | The gate-default broker profile (1.5× spread, $5/lot RT, 0.5 pip/fill, swaps off). `CostModel.zero()` only for explicit diagnostics | `core/sim/costs/model.py` | default — do not pass `zero()` for a gate |
| `MultiPairBacktester` | **The sole SL-honest gate engine** (bar-by-bar, SL-first take-the-loss). Reached via the architecture; arcs do not instantiate it directly | `core/sim/multipair_backtester.py` | (via `A1Architecture.run` inside the runner) |
| `build_exit_policy` (registry) | SL-honest exit-policy registry. Names: `sl_only`, `sl_plus_tp_2r`, `sl_plus_tp_3r`, `sl_plus_trailing_atr`, `sl_plus_trailing_swing`, `sl_partial_close_1r_runner_trail` | `core/sim/exit_policies/_registry.py` | pass the name as `A1Config(exit_policy="...")` |
| `SignalModule` / `SignalEvaluation` / `PerPairSignalState` | The signal contract every arc signal conforms to (the arc's signal is an EXPERIMENT tool, but it must implement this LOCKED Protocol) | `core/arc/signal_protocol.py` | implement `signal_name`, `primary_tf`, `causal_lineage`, `evaluate(panels) -> SignalEvaluation` |

---

## BUILT — CC-created reusable experiment tools (add + reuse freely)

> Filters, clusterers, transforms, exits, signals, soundness controls. CC builds these freely; a
> bug fails the WFO loudly. **No human gate to build one** — but it must be committed and registered
> so it compounds.

**Usage rule (do this every arc):**
1. **Before** building any filter / clusterer / transform / exit / signal / null-baseline, **check
   this section.** If it exists, **call it** (the row points at the script).
2. If it does not exist, **build it under `discovery/tools/`** (committed — persists and is reusable,
   NOT in per-arc scratch), use it in the arc, and **APPEND a row here at arc end** with: name |
   what it does | path | how to call | which arc created it.

This compounds reusable tooling the same way the log + LESSONS compound knowledge.

| Tool | What it does | Path | How to call | Created by |
|---|---|---|---|---|
| _(empty — the first arc to need one appends here)_ | | | | |

**First expected entry — the random-entry NULL baseline.** The council mandates a random-entry
soundness control (does the real signal beat random entry under the same exit?). No canonical or
reusable one exists (Arc 0 hand-rolled it twice in scratch; the only prior is retired-era under
`attic/`). The first arc that needs it builds it under `discovery/tools/` — it constructs a random
`SignalEvaluation` (random masks at a matched fire-rate, deterministic seed) and runs it through the
**canonical** `ArcFoldRunner` (the apparatus stays canonical; only the random-mask generation is the
experiment part) — then registers it in the table above.
