# Backtester v3.0 Reconfiguration — Log

Running log of each staged PR under the `infra/backtester-v3-reconfig` umbrella.
Intent doc: [`backtester_reconfig_intent.md`](backtester_reconfig_intent.md).

Chat resolved the four interpretive calls on 2026-05-22:
1. Staged 5-PR cadence (PR-A through PR-E); PR-E anchor is the critical review gate.
2. Hard cut on MT5 — no `data_source` flag, no fallback branches. `spread_floors_5ers.yaml` deleted in PR-B.
3. Report both the 7-fold Oct 2020–Jan 2026 (KH-24 anchor preservation) AND the 11-fold 2010-2020 + 1-shot 2021-2025 holdout in PR-E.
4. Strict spread deprecation — zero/negative-spread bars dropped, no fallback, file deleted.

---

## PR-A — data loader + aggregator + parquet cache (Tasks 1, 2)

**Branch:** `claude/agitated-hellman-c92fb6` (worktree).

**Scope:**
- HistData M1 bid+ask loader: per-pair, joined on `timestamp_utc`, written to `data/cache/m1/<PAIR>.parquet`.
- Deterministic OHLC aggregation M1 → {M5, M15, M30, H1, H4, D1, W1}, cached per TF at `data/cache/<TF>/<PAIR>.parquet`.
- sha256 cache-key derived from `data/histdata/m1_manifest.json`; sidecar `<parquet>.meta.json` records the key for cache validity checks.
- Strict spread handling baked into the schema: `spread_close = close_ask - close_bid`, with per-bar `bid_ask_data_quality ∈ {ok, zero_or_negative_spread, nan_bid_or_ask}`. No fallback to any external floor file.

**Added:**
- [`core/data/__init__.py`](core/data/__init__.py)
- [`core/data/cache_keys.py`](core/data/cache_keys.py)
- [`core/data/histdata_loader.py`](core/data/histdata_loader.py)
- [`core/data/aggregator.py`](core/data/aggregator.py)
- [`core/manifest.py`](core/manifest.py)
- [`configs/data_v3.yaml`](configs/data_v3.yaml)
- [`tests/fixtures/histdata_mini/__init__.py`](tests/fixtures/histdata_mini/__init__.py) + [`build.py`](tests/fixtures/histdata_mini/build.py) — synthetic two-pair × two-month layout for test isolation
- [`tests/test_cache_keys.py`](tests/test_cache_keys.py) — 15 tests on key derivation, sidecar IO, isolation
- [`tests/test_histdata_loader.py`](tests/test_histdata_loader.py) — 13 tests covering schema, bid≤ask invariant, cache hit/miss, manifest-driven invalidation, data-quality flagging
- [`tests/test_aggregator.py`](tests/test_aggregator.py) — 25 tests covering OHLC rules, TF anchoring (H4 @ 00/04/.../20, D1 @ midnight, W1 @ Monday), byte-identical two-run parquet, cache cascade from M1

**Status:** 53/53 tests pass; ruff clean.

**Deferred to PR-B+:**
- Real-spread / fill mechanics integration with `core/backtester.py` (PR-B).
- Multi-pair simultaneous simulation (PR-B).
- Per-pair `multiprocessing.Pool` parallelism for cache warm-up (PR-D). For now, callers warm caches single-pair; serial loop over 28 pairs is the intended pattern until PR-D.
- Feature-matrix cache (PR-C, dispatch Task 9b).
- KH-24 anchor reproduction (PR-E, dispatch Task 8).

**Notes / interpretive choices in PR-A:**

1. **Cache invalidation source.** The dispatch says "Invalidate cache if HistData `manifest.json` sha256s change" but `manifest.json` is the tick-zip manifest, upstream of the M1 layer the loader actually reads. PR-A keys off `m1_manifest.json` (the directly-relevant manifest); any tick-layer change that re-derives M1 also rewrites `m1_manifest.json` and cascades through. Documented in [`core/data/__init__.py`](core/data/__init__.py) module docstring.

2. **Volume column reconciliation.** Per [DATA_FOUNDATION.md §Aggregation](docs/DATA_FOUNDATION.md), bid and ask M1 CSVs both carry per-minute tick counts; they should match minute-for-minute. PR-A asserts a 0.1% tolerance and takes `max(volume_bid, volume_ask)` when they agree, raises `ValueError` on widespread mismatch (would indicate aggregation drift in the upstream M1 layer).

3. **H4 anchoring.** Pandas 3 honours `origin='start_day'` only for Tick-like freqs. H4 (`4h`) is tick-like, so `start_day` anchors bars at 00:00 / 04:00 / 08:00 / 12:00 / 16:00 / 20:00 UTC. D1 (`1D`) and W1 (`W-MON`) are non-tick-like; their default anchoring (midnight UTC for D1, Monday start for W1 via `closed=left, label=left`) is correct without an `origin` argument. Asserted in three TF-anchoring tests.

4. **Cache files are gitignored.** `data/cache/` is covered by the existing `data/*` exclude in `.gitignore` with no re-include. Caches are rebuilt locally on first run; the cache root structure is documented in [`configs/data_v3.yaml`](configs/data_v3.yaml).

5. **Synthetic test fixture.** Real HistData is 70 GB and gitignored, so PR-A ships a [`build.py`](tests/fixtures/histdata_mini/build.py) that materialises a tiny matching layout (2 pairs × 2 months × 12 minutes) into `tmp_path` per test. Deterministic by construction; produces matching sha256s in its fixture manifest.

---

## PR-B — spread / fill mechanics + multi-pair sim + spread_floors purge (Tasks 3, 6)

**Branch:** `infra/backtester-v3-pr-b` (worktree at `.claude/worktrees/pr-b-backtester-v3`).

**Scope:**
- Real per-bar bid/ask spread accounting on top of PR-A's data layer; strict drop of zero/NaN-spread bars per L_PROTOCOL §1.
- Bar-level fill primitives — long/short entry, market exit, intra-bar SL/TP with correct bid/ask reference.
- Multi-pair `Panel` over the v3 data layer; union-of-timestamps iteration with per-bar snapshot for cross-pair access.
- Single-account `Account` with realised PnL, mark-to-market equity curve, max-DD tracker, and exposure rules (total / per-pair / per-currency).
- `MultiPairBacktester` driver: bar-by-bar loop, deferred next-bar-open entry fills, intra-bar SL/TP exits, single-equity-curve output.
- Strict `spread_floors_5ers.yaml` purge: the YAML, the loader (`core/spread_floor.py`), the body-hash lock test, and `.gitattributes` line all deleted. MT5-era arc scripts and configs that referenced the floor moved en bloc to `attic/` (per chat call #2 hard-cut on MT5).

**Added:**
- [`core/spread/__init__.py`](core/spread/__init__.py) + [`core/spread/real_spread.py`](core/spread/real_spread.py) — `per_bar_spread`, `is_tradable_bar`, `data_quality_summary`, `collate_summaries`
- [`core/sim/__init__.py`](core/sim/__init__.py) re-exports
- [`core/sim/fill.py`](core/sim/fill.py) — 8 fill primitives, long + short symmetric
- [`core/sim/panel.py`](core/sim/panel.py) — `Panel` over `dict[pair, DataFrame]` with `from_pairs(pairs, tf, ...)` and `from_frames(...)`
- [`core/sim/account.py`](core/sim/account.py) — `Direction`, `Position`, `ClosedTrade`, `ExposureRules`, `Account`
- [`core/sim/multipair_backtester.py`](core/sim/multipair_backtester.py) — `Order`, `RunResult`, `MultiPairBacktester` driver
- [`scripts/data/histdata_quality_scan.py`](scripts/data/histdata_quality_scan.py) + [`__init__.py`](scripts/data/__init__.py) — full HistData DQ tally script
- [`tests/test_real_spread.py`](tests/test_real_spread.py) — 9 tests
- [`tests/test_fill.py`](tests/test_fill.py) — 14 tests (long + short + boundary)
- [`tests/test_panel.py`](tests/test_panel.py) — 10 tests
- [`tests/test_account.py`](tests/test_account.py) — 14 tests covering exposure caps + PnL + equity curve
- [`tests/test_multipair_backtester.py`](tests/test_multipair_backtester.py) — 10 tests including 3-pair sanity sim, exposure-cap enforcement, two-run determinism, SL trigger, single-equity-curve invariant
- [`docs/dispatches/pr_b_data_quality.md`](docs/dispatches/pr_b_data_quality.md) + sidecar JSON — quality scan output

**Removed (purge):**
- `configs/spread_floors_5ers.yaml` (the lock file)
- `core/spread_floor.py` (the loader module)
- `tests/test_spread_floors_lock.py` (the body-sha256 lock test)
- `.gitattributes` (whose only entry pinned the lock file's line endings)

**Moved to `attic/` (MT5-era code/configs/tests with spread_floor references or KH-era engine dependencies):**
- `scripts/{arc_2_redo,arc_2_redo2,arc_3,arc_5,arc_7,l_arc_1,l_arc_2,l_arc_3,l_arc_4,l_arc_8,l_arc_9,l_arc_10,l_arc_11,lchar,spread_validation}/`
- `scripts/phase_kgl_v2_4h_wfo.py`
- `configs/{arc_2_redo,arc_2_redo2}/`
- 21 arc config YAMLs (`configs/wfo_l_*.yaml`, `wfo_l6_*.yaml`, `l_arc_4*.yaml`, `l4_characterisation.yaml`)
- `core/signals/l4_{mtf_alignment_2_down_mixed_kijun,univariate_extreme,volatility_regime_d1_atr_top_decile_any}.py` — L_char atlas signal modules
- KH-era integration tests: `tests/test_{d1_pipeline,engine_generalisation,exit_policies,kh24_trades_all_regression,kh24_trades_paths_no_lookahead,kh24_trades_paths_schema,arc_3_step1_lookahead,phase_kgl_v2_context_columns,concurrent_filter_no_lookahead}.py`

**Status:** 57/57 PR-B tests pass; full repo sweep **773 passed / 305 skipped / 0 fail / 0 error**; ruff clean.

**Verification per dispatch:**
1. ✅ Tests pass for new spread/fill/sim code (57/57 PR-B-specific, full suite clean).
2. ✅ Two-run determinism preserved (asserted by `test_backtester_two_run_determinism` and `test_backtester_equity_curve_sha256_stable`).
3. ✅ `grep -r spread_floor core/ scripts/ configs/` returns **zero matches**.
4. ✅ Multi-pair sanity: 3-pair × 2-month × M5 panel under per-currency cap=1 keeps total USD-bearing open positions ≤ 1; single equity curve produced; per-pair cap=1 keeps any given pair to ≤ 1 open.
5. ✅ Data-quality log: `docs/dispatches/pr_b_data_quality.md` summarises zero/NaN-spread bar counts across all 28 pairs and 167.8M total bars; report numbers below.

**Notes / interpretive choices in PR-B:**

1. **Hard-cut MT5 = wholesale attic move.** ~85 MT5-era files (scripts, configs, dependent tests, L4 atlas signal modules) moved en bloc to `attic/`. The alternative — surgically remove the spread-floor import lines from each — was rejected as hours of dead-code edits for code no longer runnable (MT5 data dirs gone). The `attic/` quarantine preserves history without polluting the active tree; ruff already excludes it.

2. **`core/signal_adapter.py` and KH-24 signal modules stay active.** They're imported by `scripts/arc_kh24_v2/*` (the v2.0 self-test scripts) and `signals/kb_exhaustion_bar*` — both still serve as the reference KH-24 implementation that PR-E will replay against the v3 engine.

3. **No fallback for zero-spread bars.** Per L_PROTOCOL §1 non-negotiable on real bid/ask, `bid_ask_data_quality != ok` bars are skipped by every trade-sim path in PR-B (`core.sim.multipair_backtester._check_exits` and `._fill_pending_entries` both gate on `is_tradable_bar`). No silent backfill.

4. **PnL convention.** Positions sized in *units of base currency*; PnL realised in quote-currency units per `direction.sign * (mark - entry) * size`. Lot-sizing layers live above this module (will plug in via PR-C/E configs).

5. **Per-pair cap default = 1.** `ExposureRules.max_concurrent_per_pair=1` by default — matches KH-24's published convention. Per-currency and total caps are uncapped by default; arcs set them via the rules constructor.

---

## PR-C — v3.0 WFO + KH-24 anchor WFO + broader feature space (Tasks 4, 5)

**Branch:** `infra/backtester-v3-pr-c` (worktree at `.claude/worktrees/pr-c-backtester-v3`).

**Scope:**
- **WFO Task 4** — two fold structures sharing one orchestrator. v3.0 mode: 11-fold expanding-IS 2010-2020 + one-shot holdout 2021-present, search and holdout strictly separated. KH-24 anchor mode: 7-fold rolling Oct-2020 → Jan-2026 with 9-month OOS + 3-year rolling IS, matching the published lineage.
- **WFO §3 gates** — `classify_fold_stats` applies PASS-DEPLOYABLE / PASS-VIABLE / FAIL per L_PROTOCOL §3.
- **Feature engineering Task 5** — registry-driven Step-1 feature pipeline. 27 features across 7 classes (price_geometry, session, vol_regime, distance, spread_regime, multi_tf, cross_pair). Each feature carries a `causal_lineage` tag (clean / suspect / unverified) for the Step 6 producer audit.

**Added:**
- [`core/wfo/__init__.py`](core/wfo/__init__.py) + [`folds.py`](core/wfo/folds.py) + [`gates.py`](core/wfo/gates.py) + [`orchestrator.py`](core/wfo/orchestrator.py) — WFO core (4 modules)
- [`core/features/__init__.py`](core/features/__init__.py) + [`lineage.py`](core/features/lineage.py) + [`registry.py`](core/features/registry.py) + [`_helpers.py`](core/features/_helpers.py) + [`pipeline.py`](core/features/pipeline.py) — feature engine (5 modules)
- [`core/features/price_geometry.py`](core/features/price_geometry.py) — 5 features
- [`core/features/session.py`](core/features/session.py) — 7 features
- [`core/features/vol_regime.py`](core/features/vol_regime.py) — 2 features
- [`core/features/distance.py`](core/features/distance.py) — 3 features
- [`core/features/spread_regime.py`](core/features/spread_regime.py) — 2 features
- [`core/features/multi_tf.py`](core/features/multi_tf.py) — 4 features (panel-dependent, D1 lag-1 + W1)
- [`core/features/cross_pair.py`](core/features/cross_pair.py) — 4 features (panel-dependent, lineage=suspect)
- [`docs/features_reference.md`](docs/features_reference.md) — 27 features documented per class
- [`tests/test_wfo_folds.py`](tests/test_wfo_folds.py) — 18 tests (fold counts, anchoring, no-overlap, holdout strictly after search)
- [`tests/test_wfo_gates.py`](tests/test_wfo_gates.py) — 10 tests (each §3 gate path)
- [`tests/test_wfo_orchestrator.py`](tests/test_wfo_orchestrator.py) — 8 tests (search/holdout separation, top-K ranking, deployable flag)
- [`tests/test_features_individual.py`](tests/test_features_individual.py) — 13 tests (per-feature invariants)
- [`tests/test_features_pipeline.py`](tests/test_features_pipeline.py) — 14 tests (registry shape, lookahead spot-check, two-run determinism, sha256 stability)

**Status:** 63/63 PR-C tests pass; full repo sweep **836 passed / 305 skipped / 0 fail / 0 error**; ruff clean.

**Verification per dispatch:**
1. ✅ Tests pass for all new feature classes (per-class unit tests + pipeline integration).
2. ✅ WFO orchestrator produces correct fold counts in both modes: **11** v3.0 folds + 1 holdout, **7** KH-24 anchor folds.
3. ✅ Holdout provably untouched during search — `test_run_search_does_not_touch_holdout` asserts no holdout fold_id or holdout-year OOS appears in the search call log.
4. ✅ Cross-pair features compute on multi-pair panel from PR-B (`test_panel_dependent_features_compute_with_panel` verifies non-NaN output for cross-pair features when panel is supplied).
5. ✅ Two-run determinism preserved on the feature matrix — `test_pipeline_two_run_determinism` + `test_pipeline_sha256_stable` (CSV serialisation sha256 stable across runs).
6. ✅ Lookahead spot-check: `test_lookahead_spotcheck_5_random_trades` truncates pair_df at 5 random timestamps and asserts feature values at those timestamps are identical to the full-history compute (no future-bar influence).
7. ✅ `docs/features_reference.md` complete — every feature documented with definition, computation method, lineage tag, inputs.

**Notes / interpretive choices in PR-C:**

1. **WFO v3.0 fold 1 has empty IS.** The protocol says "11-fold WFO on 2010-2020". To get 11 OOS years from an 11-year window with anchored expanding IS, fold 1's IS = ∅ (no data strictly before 2010-01-01 in the training window). The orchestrator's `min_is_days=365` default skips fold 1; callers can pass `min_is_days=0` to evaluate every fold. This is documented in [`folds.py`](core/wfo/folds.py) and tested.

2. **KH-24 anchor IS = 3-year rolling.** Published numbers used a rolling IS; I default `is_months=36`. Callers can override per arc. The 7-fold span is 63 months from 2020-10-01 → 2025-12-31 (last fold ends Dec 2025 inclusive); the dispatch shorthand "Oct 2020 → Jan 2026" refers to the exclusive upper bound.

3. **§3 PASS-VIABLE gate interpretation.** The protocol allows "a single negative fold permitted" alongside "worst-fold ratio ≥ 2.0". With one negative fold the ratio gate is unsatisfiable on that fold (ROI < 0 → ratio < 0). I read this as: the worst-fold-ratio constraint applies to the worst **non-negative** fold's ratio (the negative fold is the documented exception). The `mean_fold_ratio ≥ 2.5` constraint still uses every fold. Tested in `test_negative_fold_blocks_deployable_but_allows_viable`.

4. **Cross-pair features default to `suspect` lineage.** Cross-pair alignment via panel snapshots is non-trivial (ffill semantics, multi-pair time gaps). Step 6 producer audit is required before any cross-pair feature can ship; until then they're tagged suspect and surface in the lineage report. Individual cross-pair producers can be promoted to clean after audit (no code change to other modules).

5. **D1 / W1 lag rule baked into multi-TF producers.** L_PROTOCOL §1 mandates one-day D1 lag (4H bar at day T sees D1 day T-1). Multi-TF features call `merge_asof(direction="backward")` on a shifted-by-1-day key. W1 alignment uses `allow_exact_matches=False` so week N's bar is invisible during week N. Both rules are tested for lookahead invariance in the spot-check.

6. **Registry is global side-effect at import.** Each feature module registers its specs at import time. The pipeline imports every class module via `core.features.pipeline` to populate the registry; callers don't need to import each class individually. Tests use `registry.clear()` only if they need a clean slate (none currently do).

---

## PR-D — parallelism + determinism harness (Tasks 7, 9c, 9d) — pending

## PR-E — KH-24 anchor reproduction + docs (Tasks 8, 10) — pending
