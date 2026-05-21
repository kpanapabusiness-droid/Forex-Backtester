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

## PR-B — spread / fill mechanics + multi-pair sim (Tasks 3, 6) — pending

## PR-C — WFO + features (Tasks 4, 5) — pending

## PR-D — parallelism + determinism harness (Tasks 7, 9c, 9d) — pending

## PR-E — KH-24 anchor reproduction + docs (Tasks 8, 10) — pending
