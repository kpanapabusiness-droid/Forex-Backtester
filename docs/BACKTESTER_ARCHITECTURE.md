# Backtester Architecture (v3.0)

> Single source of truth for the v3.0 backtester after CC_06 lands.
> Updated through each staged PR. Final consolidation lands in PR-E
> alongside the KH-24 anchor reproduction results.

This document covers what the engine *is*; the higher-level **why**
lives in [L_PROTOCOL.md](../L_PROTOCOL.md), and the per-arc *what*
lives in arc closure docs under `docs/archive/arc_results/`.

This document supersedes the v2.0.0 architecture write-up; that
content lived in this file at the time of `09319bb` (2025-09-30) and
is fully obsolete after the CC_06 reconfig.

---

## Layers (bottom-up)

```
                ┌────────────────────────────────────────────┐
   PR-E         │   KH-24 anchor reproduction + final docs   │
                ├────────────────────────────────────────────┤
   PR-D         │   parallelism + determinism harness        │
                ├────────────────────────────────────────────┤
   PR-C         │   WFO + broader features                   │
                ├────────────────────────────────────────────┤
   PR-B         │   real spread + fill + multi-pair sim      │
                ├────────────────────────────────────────────┤
   PR-A         │   HistData loader + TF aggregator + cache  │
                ├────────────────────────────────────────────┤
                │   HistData M1 bid+ask layer (52 GB tick,   │
                │   18 GB derived M1, 28 pairs, 2010-2026)   │
                └────────────────────────────────────────────┘
```

Each layer is independently testable and depends only on the layers
below.

---

## Data layer (PR-A)

- **Loader:** `core.data.histdata_loader.load_m1(pair, ...)` reads
  per-pair-month M1 bid + ask CSVs and joins them into a single
  DataFrame with columns `open/high/low/close_{bid,ask} + volume +
  spread_close + bid_ask_data_quality`.
- **Aggregator:** `core.data.aggregator.aggregate(pair, tf, ...)`
  deterministically aggregates M1 → {M5, M15, M30, H1, H4, D1, W1}.
  Bid and ask are aggregated independently; data quality is
  recomputed per aggregated bar.
- **Parquet cache layout:**

  ```
  data/cache/
    m1/<PAIR>.parquet           # joined bid+ask M1 (per pair)
    {M5,M15,M30,H1,H4,D1,W1}/<PAIR>.parquet   # aggregated TFs
    *.parquet.meta.json         # sidecar: cache_key + source manifest sha
    features/<arc_id>/<feature_set_hash>.parquet   # PR-D feature cache
    features/<arc_id>/<feature_set_hash>.parquet.meta.json
  ```

- **Cache invalidation:** keyed on
  `data/histdata/m1_manifest.json`. The per-pair cache key is
  `sha256(sorted (relpath, sha256) pairs)`; any change in the upstream
  M1 layer regenerates the manifest and cascades through.
- **Spread:** `spread_close = close_ask - close_bid`. Zero/negative
  and NaN bars are flagged in `bid_ask_data_quality`; no fallback to
  any external floor file (L_PROTOCOL §1 non-negotiable, enforced
  since PR-B).

## Spread + sim (PR-B)

- **`core.spread.real_spread`** — per-bar spread + tradability mask +
  data-quality summary.
- **`core.sim.fill`** — 8 bar-level fill primitives. Long entry =
  `open_ask`, long exit = `close_bid`, intra-bar SL/TP triggered
  against `low_bid`/`high_bid`. Short symmetric.
- **`core.sim.panel.Panel`** — multi-pair wrapper over
  `dict[pair, DataFrame]` with union-of-timestamps iteration and
  `snapshot_at(t)` for cross-pair access.
- **`core.sim.account.Account`** — single account state across all 28
  pairs: balance, equity curve, max-DD tracker, exposure caps
  (total/per-pair/per-currency).
- **`core.sim.multipair_backtester.MultiPairBacktester`** — bar-by-bar
  driver with deferred next-bar-open entry fills and intra-bar SL/TP
  exits. Single equity curve output.

## WFO + features (PR-C)

- **`core.wfo.folds`** — two builders. `build_v3_folds()` produces
  11-fold expanding-IS 2010-2020 + one-shot holdout 2021-present (per
  L_PROTOCOL §2 Step 5). `build_kh24_anchor_folds()` produces 7-fold
  rolling Oct 2020 → Jan 2026 (matching the published KH-24 lineage).
- **`core.wfo.gates`** — §3 PASS-DEPLOYABLE / PASS-VIABLE / FAIL
  classifier on per-fold stats.
- **`core.wfo.orchestrator`** — `run_search` runs candidates across
  the fold list and ranks by worst-fold ratio; `run_holdout`
  evaluates top-K candidates ONCE on the locked holdout. Holdout is
  provably untouched during search (asserted by
  `tests/test_wfo_orchestrator.py`).
- **`core.features`** — 27 features across 7 classes
  (price_geometry, session, vol_regime, distance, spread_regime,
  multi_tf, cross_pair). Every feature carries a `causal_lineage`
  tag ∈ {clean, suspect, unverified} that drives the Step 6 producer
  audit. Full reference: [features_reference.md](features_reference.md).
- **`core.features.pipeline.compute_feature_matrix(pair, pair_df,
  panel)`** — walks the registry, emits a DataFrame + lineage table.

## Parallelism + determinism (PR-D)

### Parallelism

- **`core.parallel.parallel_pair_map(func, pairs, pool_size)`** —
  `multiprocessing.Pool` over per-pair work. NOT threading — Python's
  GIL blocks numpy/pandas.
- **`core.parallel.build_panel_parallel(pairs, tf, ...)`** — builds a
  multi-pair Panel by aggregating each pair in parallel. Cold-cache
  build hits the dispatch's "≥ 10× faster across 28 pairs" target.
- **Pool size default:** `min(n_pairs, max(1, cpu_count() - 1))`.
  Configurable via `configs/data_v3.yaml`'s `parallelism.pool_size`.
- **Aggregation order:** results are always returned sorted by pair
  name. The aggregation step (`pd.concat`, dict insertion, etc.)
  iterates that sorted order, so output is byte-identical across pool
  sizes.

### Determinism contract

- **`core.determinism`** centralises invariants:
  - `RANDOM_STATE = 42`
  - `N_JOBS = 1` (inside any per-row computation)
  - `LINE_TERMINATOR = "\n"` (every text artefact)
  - `seed_everything(seed)` seeds numpy + python `random` + sets
    `PYTHONHASHSEED` for subprocess spawn.
- **Two-run sha256 reproducibility** is the load-bearing contract.
  `tests/test_determinism.py` runs the same mini pipeline twice and
  asserts every output (panel CSV, feature matrix CSV, backtester
  equity curve CSV) is byte-identical. The same test asserts
  `pool_size=1 == pool_size=4` (parallelism doesn't change output).

### Feature matrix cache

- **Path:** `data/cache/features/<arc_id>/<feature_set_hash>.parquet`
- **Cache key:** `sha256(signal_def + pool_sha256 + feature_set_version)`.
  Any change in any component → new key → cache invalidates.
- **Sidecar `.meta.json`** records all three input components in
  plaintext so an audit can reconstruct what produced the cached
  matrix.
- **Workflow:**
  ```python
  from core.features.cache import get_or_compute, pool_sha_from_dataframe

  pool_sha = pool_sha_from_dataframe(my_trade_pool)
  features = get_or_compute(
      arc_id="my_arc",
      signal_def="kb_exhaustion_bar(c1-c6,c8,c9)",
      pool_sha=pool_sha,
      feature_set_version="v3.0",
      compute_fn=lambda: compute_feature_matrix(pair, pair_df, panel),
  )
  ```

## Anchor preservation (PR-E)

Per L_PROTOCOL §8: any cross-arc evaluation framework must reproduce
KH-24's documented worst-fold numbers within tolerance (±0.5pp ROI,
±1pp DD). PR-E runs the KH-24 config through the v3 engine on the
7-fold KH-24-anchor WFO structure and compares.

Numbers to reproduce (from `ARC_HISTORY.md`):
- Worst-fold ROI: +1.92% (fold 7, 2025-04-01 → 2026-01-01)
- Worst-fold DD: 6.37% (fold 1, 2020-10-01 → 2021-07-01)
- 214 trades across 7 OOS folds, all 7 positive

Real-spread reconciliation already documented: F7 ROI drops to ~+1.28%
under HistData spreads (vs +1.92% on the original 5ers MT5 data).
PR-E's tolerance band includes both points.

---

## Out of scope for v3.0 backtester docs

- Per-arc signal definitions — live in arc-specific docs / configs
- Architecture A1..A6 implementations — registered in
  `core/architectures/` (added incrementally with each arc)
- Step 6 producer-audit procedures — defined in L_PROTOCOL §2 Step 6
- ML / classifier choices — sub-protocol per arc
- Live deployment EA — `EA/KH24_EA.mq5` is the only deployed system;
  ports of v3 candidates open only when a candidate clears
  PASS-DEPLOYABLE per §3.
