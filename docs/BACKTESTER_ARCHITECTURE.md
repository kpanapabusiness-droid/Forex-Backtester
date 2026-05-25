# Backtester Architecture (v3.0)

> Single source of truth for the v3.0 backtester. Closed at PR-E.1.7
> (2026-05-22) with the KH-24 anchor reproduction documented as
> partial (per Path B verdict). v3.0 is certified Phase 0 ready for
> forward arc work; the live KH-24 deployment on 5ers MT5 is
> unaffected.

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
   PR-E.1.7     │   v3.0 closure docs (Phase 0 GO)           │
                ├────────────────────────────────────────────┤
   PR-E.1.6     │   EA-correction round 3 (signal bid OHLC,  │
                │   trail bid/close/next-bar, live-balance   │
                │   risk, per-currency exposure cap)         │
                ├────────────────────────────────────────────┤
   PR-E.1.5     │   EA-correction round 2 (kijun_d1 lag-1    │
                │   D1, h1_cir bid OHLC, D1 regime no-op)    │
                ├────────────────────────────────────────────┤
   PR-E.2       │   KH-24 anchor reproduction (HALT round 1) │
                ├────────────────────────────────────────────┤
   PR-E.1       │   KH-24 strategy + engine extensions       │
                │   (trailing stop, exit hooks, risk)        │
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

## Data layer (PR-A; PR #187 EET extension)

Under PR #187, the M1→TF aggregator supports two bar-boundary
conventions via `boundary_convention=` parameter:

- `"utc"` (default, legacy) — UTC-anchored bins; existing caches
  byte-identical to pre-PR #187.
- `"5ers_eet"` — 5ers broker EET/EEST-anchored; per-day re-anchored
  for H4 to handle DST transitions. Cache namespace
  `data/cache/<TF>_5ers_eet/<PAIR>.parquet`.

See [PROTOCOL_RUNTIME.md §15.3](PROTOCOL_RUNTIME.md) and
[docs/calibration/histdata_mt5_aggregation_parity_2026_05.md](calibration/histdata_mt5_aggregation_parity_2026_05.md)
for full convention specs and DST handling.

**Signal-module timezone responsibility:** Signal modules and multi-TF feature
producers MUST use [`core/signals/htf_alignment.py`](../core/signals/htf_alignment.py)
(`get_htf_value_at` / `get_htf_row_at` / `get_htf_index_at`) for any
lookup of an HTF column at LTF anchor timestamps — NOT raw `.floor()`
or `.normalize()` against UTC anchors. The engine handles bar
*aggregation* tz-correctly (above); signal modules are responsible for
the *alignment* step and must use the canonical utility to remain
timezone-invariant under both UTC and 5ers EET conventions. See
[PROTOCOL_RUNTIME.md §15.4](PROTOCOL_RUNTIME.md) and
[docs/audits/signal_module_eet_audit_2026_05.md](audits/signal_module_eet_audit_2026_05.md)
for the bug class this avoids.

**Session-bucketing responsibility:** distinct fault class from HTF lookup
— covers prior-session HL features, reset-floor daily ratchet, and
Amendment 6 daily-DD bucketing. `Panel` carries a `boundary_convention`
attribute that downstream consumers consult for trading-day-aware
bucketing. Three engine modules use it:

- `core.features.distance._prior_session_*` — prior-session HL features
  bucket bars by `panel.boundary_convention`.
- `core.sim.risk.reset_floor.ResetFloorAccount` — daily floor ratchet
  uses the same boundary (forward hygiene; dormant in v3 runtime).
- `core.runners._fold_stats_helpers.compute_per_day_max_dd` — daily-DD
  bucketing for Amendment 6 (load-bearing; feeds
  `daily_dd_breaches_at_r_safe` / `daily_dd_breaches_at_r_hard`).

The shared utility is `core.utils.session_boundary.utc_to_eet_trading_day`.
Default `Panel.boundary_convention="utc"` preserves KH-24 anchor
byte-identity; aggregator-driven constructors (`Panel.from_pairs`,
`build_panel_parallel`) thread the caller's choice through.
See [PROTOCOL_RUNTIME.md §15.5](PROTOCOL_RUNTIME.md).

### Pre-PR-#187 layer (unchanged for UTC)

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

## Spread + sim (PR-B; PR #187 updates)

- **`core.spread.real_spread`** — per-bar spread + tradability mask +
  data-quality summary.
- **`core.sim.fill`** — 8 bar-level fill primitives. Long entry =
  `open_ask`, long exit = `close_bid`, intra-bar SL/TP triggered
  against `low_bid`/`high_bid`. Short symmetric. Worst-case fills
  satisfy PR #187 Sub-change B — spread is implicit in the bid/ask
  wings, no separate deduction step.
- **`core.sim.trailing_stop.TrailManager`** — trail activation +
  ratchet operate on **mid close** under PR #187 (signal-parity
  convention; reverses PR-E.1.6's bid-only trail). Trail hit detection
  remains bid-side for worst-case-fill realism. See
  [PROTOCOL_RUNTIME.md §15.2](PROTOCOL_RUNTIME.md).
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

## Anchor reproduction (PR-E.2 / E.1.5 / E.1.6 — closed PR-E.1.7)

Per L_PROTOCOL §8, any cross-arc evaluation framework should
reproduce KH-24's documented worst-fold numbers within tolerance.
PR-E ran the locked KH-24 config through the v3 engine on the 7-fold
KH-24 anchor WFO structure and compared against the published
lineage. The reproduction PARTIALLY held — F7 reproduced inside the
documented real-spread reconciliation band; F2 sign was recoverable;
F1/F4/F5 remained sign-reversed.

Per chat's Path B verdict (2026-05-22): v3.0 is the canonical
backtester for Phase 0+ forward work. Published KH-24 numbers stand
as the live-system record on 5ers MT5; v3 reproduction is approximate,
not exact, and the divergence is attributable to data-source change +
two deferred EA-correction items (Sections G + H of the EA diff doc).
The live EA on the Contabo VPS / 5ers MT5 broker feed is UNAFFECTED.

### A. Reproduction methodology

- **Runner:** `scripts.anchor.run_anchor.run(...)` builds H4/D1/H1
  panels from HistData M1 (PR-A loader + aggregator), runs
  `build_kh24_runtime(...)` per fold, and dispatches the v3
  multi-pair backtester. Output: per-fold parquet + summary.md +
  comparison.md.
- **Folds:** 7 anchored rolling Oct 2020 → Jan 2026 (3-year IS,
  9-month OOS) — matches the published lineage's WFO structure.
- **Config:** `KH24Config()` defaults — signal C1-C6/C8/C9 on
  bid-side single OHLC, H1 CIR at T=0.28, 2×ATR SL, 2×ATR trail
  activation + 1.5×ATR trail distance (close-driven trigger,
  next-bar-open fill), per-currency exposure cap = 2, per-pair = 1,
  no total cap, 1% live-balance risk.
- **Engine extensions in PR-E.1.6:** `TrailManager` switched to
  `close_bid` (was mid-OHLC); trail exits queue at bar-close and fill
  at next-bar `open_bid` (was intra-bar wick); `LiveBalanceRisk`
  replaces `ResetFloorAccount` for KH-24 (compounds with realised
  PnL); SL anchor remains at signal-bar `close_ask` proxy
  (Section H deferred).

### B. v3 vs published — fold-by-fold (post-CC_07 anchor)

The v3 anchor below is the **A1 architecture path** (CC_07's
`ArcFoldRunner(A1, kh24_to_a1)` over the full-history-warmup signal
evaluation). Per the CC_07 anchor regression (`scripts/anchor/check_a1_equivalence.py`
+ `scripts/anchor/bisect_warmup.py`, 2026-05-22) the A1 path matches
the legacy `KH24FoldRunner` byte-identically on 5 of 7 folds and
diverges on F2/F3 due to a warmup-buffer effect in the legacy
`KH24FoldRunner` (its 30-day pre-OOS slice NaN-masks `kijun(26)` on
the first OOS bars when calendar weekends/holidays consume the
buffer). Running the legacy path with `warmup_days=365` recovers the
A1 numbers byte-identically. The A1 numbers below are therefore the
**correct full-history reproduction**; legacy 30-day-warmup numbers
are preserved in §B.1 for historical reference. KH-24 live
deployment numbers are unchanged.

| Fold | OOS window | Pub ROI | v3 ROI (A1) | Pub DD | v3 DD (A1) | Pub trades | v3 trades (A1) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 2020-10 → 2021-07 | +13.35% | -1.43% | 6.37% | 5.13% | 41 | 32 |
| 2 | 2021-07 → 2022-04 | +9.63%  | +3.17% | 4.45% | 4.25% | 36 | 18 |
| 3 | 2022-04 → 2023-01 | +11.90% | +6.69% | 4.43% | 3.95% | 25 | 29 |
| 4 | 2023-01 → 2023-10 | +3.32%  | -5.34% | 3.80% | 5.41% | 32 | 19 |
| 5 | 2023-10 → 2024-07 | +6.23%  | -6.51% | 3.09% | 9.22% | 23 | 20 |
| 6 | 2024-07 → 2025-04 | +3.24%  | -1.13% | 5.03% | 11.51% | 30 | 29 |
| 7 | 2025-04 → 2026-01 | +1.92%  | +2.31% | 4.06% | 3.88% | 27 | 19 |

**Aggregates (post-CC_07):** total trades 214 (pub) vs 166 (v3 A1);
positive folds 7/7 (pub) vs 3/7 (v3 A1); worst-fold ROI +1.92% (pub)
vs -6.51% (v3 A1, F5); worst-fold DD 6.37% (pub) vs 11.51% (v3 A1, F6).

**F7 reproduces inside the documented real-spread band.** Published
F7 +1.92% → expected ~+1.28% under HistData spread audit
(ARC_HISTORY.md); v3 A1 produced +2.31%. F7 is the only fold the
original ±0.5pp tolerance criterion passes unambiguously against pub.

**F2 sign recovered.** PR-E.1.5's kijun_d1 fix moved F2 from -1.78%
(initial) to +4.21% / +3.17% (PR-E.1.6 / CC_07 A1) vs published
+9.63% — sign-consistent, direct evidence the engine's exit semantics
now match the EA on the fold where they previously diverged most.
CC_07 A1's +3.17% is preferred over PR-E.1.6's +4.21% because the
latter benefitted from the legacy warmup's NaN-masked kijun on the
first OOS bars; A1 evaluates the signal/kijun on the full panel, so
no NaN buffer.

### B.1. Pre-CC_07 anchor (legacy `KH24FoldRunner`, `warmup_days=30`)

Preserved for reference. These numbers were produced by the legacy
fold-runner with a 30-day slice before signal evaluation — the cause
of the F2/F3 divergence diagnosed at CC_07 anchor regression. They
are no longer the canonical anchor.

| Fold | Pub ROI | legacy v3 ROI | Pub DD | legacy v3 DD | Pub trades | legacy v3 trades |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | +13.35% | -1.43% | 6.37% | 5.13% | 41 | 32 |
| 2 | +9.63%  | +4.58% | 4.45% | 2.89% | 36 | 18 |
| 3 | +11.90% | +5.64% | 4.43% | 4.27% | 25 | 30 |
| 4 | +3.32%  | -5.34% | 3.80% | 5.41% | 32 | 19 |
| 5 | +6.23%  | -6.51% | 3.09% | 9.22% | 23 | 20 |
| 6 | +3.24%  | -1.13% | 5.03% | 11.51% | 30 | 29 |
| 7 | +1.92%  | +2.31% | 4.06% | 3.88% | 27 | 19 |

The `KH24FoldRunner` class is retained in `core/wfo/fold_runner.py`
to preserve this reproduction path for regression purposes; the
canonical anchor going forward is the A1 path.

### C. Attributed sources of residual divergence

1. **HistData vs 5ers MT5 spread.** Arc 4's spread audit found
   HistData spreads 3-48× higher than the per-pair floors KH-24's
   published numbers were measured against. Direction: v3 ROI ≤
   published ROI on every fold. Extrapolated magnitude across 165
   trades over 5+ years: 2-4 ROI points per fold downward shift.
2. **MTM vs closed-trade DD convention.** The published lineage used
   closed-trade DD, which understates real account DD by 14-63% per
   ARC_HISTORY. v3 uses MTM equity DD. Direction: v3 DD ≥ published
   DD on folds with material open positions (visible on F5/F6).
3. **Section H deferred — SL anchor post-fill.** The EA computes
   `sl_price = realised_entry_price − 2×ATR` AFTER the fill; v3
   computes `sl_price = signal_bar.close_ask − 2×ATR` BEFORE the
   fill. SL distance differs by `(next_bar.open_ask −
   signal_bar.close_ask)`. On news/gaps the discrepancy can reach
   ~10% of SL distance. Estimated cumulative impact: 3-5pp aggregate
   per fold downward.
4. **Section G deferred — news filter.** EA delays entries during
   high-impact news windows (per `IsNewsBlackout` in
   `reference/kh24_ea/KH24_EA.mq5:499`); v3 fills immediately. Effect
   varies by fold; macro-event-heavy folds (F2 Russia-Ukraine,
   F4 banking-crisis) show 23-50%+ trade-count gap consistent with
   delayed-entry effects compounding through the per-pair cap.
5. **Cross-currency sizing simplification.** v3 sizes non-USD-quote
   pairs (USDJPY, AUDCAD, EURGBP, ...) treating account balance as
   quote-denominated; EA uses `SYMBOL_TRADE_TICK_VALUE` for proper
   cross-rate conversion. Minor contributor.

Detail in [docs/dispatches/anchor_diagnostic_round_3.md](dispatches/anchor_diagnostic_round_3.md).

### D. Phase 0 readiness

v3.0 is certified Phase 0 ready (per Path B). The criteria:

- **Engine correctness is established.** F7 reproduces within the
  documented real-spread band; F2 sign is recoverable; every
  EA-correction in PR-E.1.5 + PR-E.1.6 produced its expected
  isolated-test result.
- **Divergences are attributable, not unknown.** Sections C.1-C.5
  above each have a documented root cause; none are "the engine
  produces wrong numbers we can't explain."
- **Forward work uses v3 on HistData, not 5ers MT5.** L arcs already
  ran under HistData via the v2 backtester; v3 is the next-generation
  engine for the same data substrate. No retroactive obligation to
  reproduce 5ers-MT5-era numbers exactly.
- **The live system is unaffected.** KH-24 stays deployed on the
  Contabo VPS / 5ers MT5 feed unchanged. v3 reproduction is a
  research-engine validation, not a deployment-system replacement.

If a future arc needs tighter reproduction, the path is to land
Sections G (news filter) and H (post-fill SL anchor) per the diff
doc — estimated 150-250 LOC + tests, with declining returns on
closing the residual gap (likely 2-3pp each on the most-affected
folds, not 10pp).

### Re-running the anchor

```python
from scripts.anchor.run_anchor import run

result = run(
    structure="kh24_anchor",   # 7-fold rolling Oct 2020 → Jan 2026
    pairs=None,                # default: 28 FX universe
    output_dir="results/anchor_kh24_7fold_v3",
)
```

`result` includes per-fold trades + equity parquet, a comparison
table against the published numbers, and a manifest with sha256s of
every artefact. The runner is deterministic (PR-D guarantees) — two
invocations produce byte-identical output.

### Anchor reference artefacts

- `reference/kh24_ea/KH24_EA.mq5` — deployed EA source; ground
  truth for KH-24 mechanics.
- `docs/dispatches/kh24_ea_full_diff.md` — EA-vs-v3 corrections
  catalog (Sections A-H).
- `docs/dispatches/anchor_diagnostic_round_3.md` — final HALT
  diagnostic with three-paths analysis (chat picked Path B).
- `docs/dispatches/anchor_diagnostic_round_2.md` — post-PR-E.1.5
  HALT diagnostic.
- `results/anchor_kh24_7fold_v3/` — reproduction output (gitignored;
  reproducible from runner).

---

## Out of scope for v3.0 backtester docs

- Per-arc signal definitions — live in arc-specific docs / configs
- Architecture A1..A6 implementations — live in `core/architectures/`,
  with step runners under `core/steps/` and the orchestrator at
  `core/arc/arc_orchestrator.py`. See
  [PROTOCOL_RUNTIME.md](PROTOCOL_RUNTIME.md) for the API reference.
  Built by CC_07.
- Step 6 producer-audit procedures — defined in L_PROTOCOL §2 Step 6
- ML / classifier choices — sub-protocol per arc (hook empty at v3.0;
  see [PROTOCOL_RUNTIME.md §11](PROTOCOL_RUNTIME.md))
- Step 4 classifier persistence — the best-AUC classifier per
  candidate cluster is refit on the full lineage-filtered pool and
  pickled to `results/<arc>/step_4/classifiers/<cluster_id>.pkl`
  with a SHA256 + provenance `manifest.json`. A2 / A6 architectures
  load it without retraining via
  `core.steps.classifier_persistence.build_a2_config_from_step4` /
  `build_a6_config_from_step4`. Full reference:
  [PROTOCOL_RUNTIME.md §7](PROTOCOL_RUNTIME.md).
- A3 / A4 per-fold classifier orchestration — `core/steps/path_classifier_per_fold.py`
  builds per-fold `PathClassifierFit` (target = cluster membership for
  A3, `final_r > 0` for A4) on each fold's IS-only window. Threaded
  via `A1RunContext.path_classifier_fits`. Cost-decomposition emitted
  per top-K candidate in `StrategyResult.metadata`. Reference:
  [PROTOCOL_RUNTIME.md §8](PROTOCOL_RUNTIME.md).
- Amendment 3 risk-normalised gates — engine emits the full scaled-risk
  evaluation (chained max DD, per-day max-DD parquet, scaled
  DEPLOYABLE / VIABLE gates, holdout re-runs at r_safe / r_hard,
  sizing-convention check, priority-ordered failure-mode taxonomy).
  Modules: `core.wfo.amended_gates`, `core.wfo.chained_dd`,
  `core.wfo.holdout_rerun`. Full reference:
  [PROTOCOL_RUNTIME.md §8b](PROTOCOL_RUNTIME.md).
- Live deployment EA — `EA/KH24_EA.mq5` is the only deployed system;
  ports of v3 candidates open only when a candidate clears
  PASS-DEPLOYABLE per §3.
