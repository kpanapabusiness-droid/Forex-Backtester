# PR-A Pre-Merge Verification

PR: [Forex-Backtester#161](https://github.com/kpanapabusiness-droid/Forex-Backtester/pull/161)
Branch: `claude/agitated-hellman-c92fb6` (worktree)
Date: 2026-05-22
Mode: read-only verification per CC dispatch. No code modifications. Output is this single report file.

---

## Check 1 — Parquet cache works as expected

**What was tested.** First and second `load_m1('EURUSD', ...)` calls timed against real HistData (5,964,931 M1 rows / 16 years × 12 months × 2 sides). First should parse CSVs; second should hit the parquet cache. Ratio target: ≥10×.

**Command.**

```python
from core.data.histdata_loader import load_m1
HISTDATA = Path('C:/Users/panap/Documents/Forex-Backtester/data/histdata')
CACHE = Path('C:/Users/panap/AppData/Local/Temp/pr_a_check1_cache')
shutil.rmtree(CACHE, ignore_errors=True)
# t1: first call (cache miss); t2: second call (cache hit)
```

**Result.**

| Metric | Value |
|---|---:|
| First run (cache miss) | **41.6207 s** |
| Second run (cache hit) | **0.4223 s** |
| Ratio (first / second) | **98.56×** |
| Parquet file present | **Y** — `data/cache/m1/EURUSD.parquet`, 131.96 MB |
| Sidecar `.meta.json` present | **Y**, 492 bytes |
| Sidecar `cache_key` (sha256, 64 hex chars) | **Y** — `16b7daaa…7a660e96` |
| Sidecar `source_m1_manifest_sha256` (64 hex chars) | **Y** — `a2d19a9c…2caa679d` |
| Sidecar fields populated | `pair=EURUSD`, `layer=m1`, `n_rows=5964931`, all 11 canonical columns recorded |

**Verdict: PASS.** Speedup 98.56× — almost 10× the 10× bar.

**Concerns:** none.

---

## Check 2 — TF aggregation byte-identical (determinism)

**What was tested.** Locate the byte-identical determinism test for M1→TF aggregation, run it in isolation, then manually re-confirm on real EURUSD data (not just the synthetic fixture) by aggregating to H4 twice and sha256-comparing the two parquet outputs.

**Test located.** `tests/test_aggregator.py::test_aggregate_byte_identical_two_runs` — implementation aggregates twice, writes both to tmp parquets, asserts `sha256(a) == sha256(b)`.

**Command (isolated test).**

```bash
py -m pytest tests/test_aggregator.py::test_aggregate_byte_identical_two_runs -v
```

**Result (isolated test).** `1 passed in 0.25s`.

**Command (manual on real EURUSD H4).**

```python
df_m1 = load_m1('EURUSD', histdata_root=HISTDATA, cache_root=CACHE)
df_h4_a = aggregate_m1_to_tf(df_m1, 'H4')
df_h4_b = aggregate_m1_to_tf(df_m1, 'H4')
# write each to tmp parquet, sha256 compare
```

**Result (manual).**

| Metric | Value |
|---|---|
| H4 rows | 26,170 |
| sha256(a) | `2d3b62ef31c35a7b417671887dd15bdd97a429e3be2f41d0980b2538451484e9` |
| sha256(b) | `2d3b62ef31c35a7b417671887dd15bdd97a429e3be2f41d0980b2538451484e9` |
| Byte-identical | **Y** |
| `pd.testing.assert_frame_equal(a, b)` | **passes** |

**Verdict: PASS.** Two-run byte-identical determinism holds at both fixture scale and real-data scale (16 years of EURUSD H4).

**Concerns:** none.

---

## Check 3 — Zero `spread_floors_5ers` references in new code

**Commands + results.**

```
grep -r "spread_floors_5ers" core/data/        → No matches found
grep -r "spread_floors"       configs/data_v3.yaml → No matches found
grep -r "spread_floor"        core/data/        → No matches found
```

All three return zero matches.

**Verdict: PASS.** PR-A new code is clean of any spread-floor references. The existing `configs/spread_floors_5ers.yaml` file is untouched in PR-A and will be deleted in PR-B per the chat-resolved scope.

**Concerns:** none.

---

## Check 4 — Legacy MT5 tests skip (not silently failing)

**Command.**

```bash
py -m pytest -rs --tb=no -q
py -m pytest -rs --tb=no -q | grep "^SKIPPED"  # for distinct reasons
```

**Result.**

| Metric | Value |
|---|---:|
| Passed | 859 |
| Skipped | 364 |
| Failed | 0 |
| Errors | 0 |

All 364 skips are explicit `pytest.skip(...)` calls with informative reasons (no import errors, no setup errors masquerading as skips). Distinct skip-reason categories observed:

| Category | Pattern | Notes |
|---|---|---|
| Missing legacy MT5 data | `data/daily missing or empty; need OHLCV CSV for <X> tests`, `data/4hr and data/daily not present` | Expected — dirs deleted with v3.0 reorg. Will be replaced by HistData-fed equivalents in PR-C+. |
| Missing pre-built result artefacts | `No artifact at .../trades_all.csv; run <config> first`, `Step 1 + Step 2 outputs must exist; run those first.`, `results/kh24/trades_all.csv not present — run the KH-24 backtester first` | Tests guard against running without the upstream artefact; cheap and correct. |
| Missing legacy config | `configs/wfo_phase5.yaml not found`, `configs/wfo_v2.yaml not found`, `configs/v1_system.yaml not found` | Older configs not on main. |
| Explicit "real arc data unavailable" | `Real arc data not available in this environment` | Same as above, with a friendlier message. |
| Logical / design skips (not data-related) | `binary does not output 0 by design`, `flip comparison is for binary vs naive`, `Full perturbation test deferred to v1.3.1 — protocol §10.1 lookahead invariant covers entry/exit decisions; per-bar invariant check is a planned follow-up extension.` | These are by-design skips, not data-availability skips. Three of these in total. |

**Verdict: PASS.** Every skip is intentional and well-labelled. No silent failures, no import errors, no setup errors.

**Concerns:** none. (Note for chat awareness: many of the 361 data-availability skips will need to be either rewired to the HistData layer or marked obsolete during PR-C+. That cleanup is out of scope for PR-A.)

---

## Check 5 — Test coverage of new code

**Command (collection).**

```bash
py -m pytest tests/test_cache_keys.py tests/test_histdata_loader.py tests/test_aggregator.py --collect-only -q
```

**Result.** **53 tests collected** across the three files. Coverage summary:

| File | Test count | Coverage areas |
|---|---:|---|
| `tests/test_cache_keys.py` | 15 | sha256 derivation (stability under dict ordering, change on file mutation, change on file added, isolation per pair), TF key cascade from M1 key, sidecar meta IO (roundtrip, missing, malformed), `cache_valid` semantics, LF line-terminator invariant, `manifest_self_sha256` content-sensitivity |
| `tests/test_histdata_loader.py` | 13 | canonical schema, row count vs fixture, bid≤ask invariant, `spread_close` consistency, all-OK data-quality on clean fixture, parquet write, second-call cache hit, manifest-change → cache invalidation, `use_cache=False` rebuild, two-pair cache isolation, missing-pair raises, missing-manifest raises, zero-spread data-quality flag |
| `tests/test_aggregator.py` | 25 (incl. 7-way + 7-way parametrize) | canonical schema per TF (× 7 TFs), M5 OHLC rules concrete, H4 anchoring at `:00/:04/.../:20`, D1 at midnight, W1 starting Monday, unsupported-TF error, **byte-identical two-run parquet**, DataFrame equality two-run, volume sum preserved (no dropped minutes), parquet cache per TF (× 7 TFs), cache hit avoids recompute, TF cache invalidation cascade from M1, `use_cache=False` idempotence |

Areas explicitly covered per the dispatch's coverage requirement:
- Data loader paths ✅
- Missing-month handling ✅ (`_enumerate_pair_months` keeps only paired months; `test_load_m1_missing_pair_raises` for absent pair; `test_load_m1_missing_manifest_raises` for absent manifest)
- Cache invalidation ✅ (`test_load_m1_cache_invalidates_on_manifest_change`, `test_aggregate_tf_cache_invalidates_with_m1`)
- M1→TF aggregation ✅ (7 TFs × parametrize, OHLC rules, anchoring, volume preservation)
- Bid-ask schema ✅ (canonical columns, bid≤ask invariant, spread_close consistency, data-quality flags)
- Parquet I/O ✅ (write on miss, read on hit, mtime preserved on hit, byte-identical two-run)

**Command (spot-check 5 random tests individually).**

```bash
py -m pytest \
  "tests/test_cache_keys.py::test_m1_cache_key_isolated_per_pair" \
  "tests/test_histdata_loader.py::test_load_m1_cache_invalidates_on_manifest_change" \
  "tests/test_histdata_loader.py::test_load_m1_data_quality_flag_on_zero_spread" \
  "tests/test_aggregator.py::test_aggregate_h4_anchors_to_zero_oclock" \
  "tests/test_aggregator.py::test_aggregate_volume_sum_preserved" \
  -v
```

**Result.** `5 passed in 0.44s`. Per-test:

| # | Test | Result |
|---|---|---|
| 1 | `test_cache_keys.py::test_m1_cache_key_isolated_per_pair` | PASSED |
| 2 | `test_histdata_loader.py::test_load_m1_cache_invalidates_on_manifest_change` | PASSED |
| 3 | `test_histdata_loader.py::test_load_m1_data_quality_flag_on_zero_spread` | PASSED |
| 4 | `test_aggregator.py::test_aggregate_h4_anchors_to_zero_oclock` | PASSED |
| 5 | `test_aggregator.py::test_aggregate_volume_sum_preserved` | PASSED |

**Verdict: PASS.** 53 tests cover every required area; spot-checks all green individually.

**Concerns:** none.

---

## Check 6 — Manifest and meta files

**Cache directory listing (after Check 1's `load_m1('EURUSD')`).**

```
data/cache/m1/
  EURUSD.parquet            138,367,909 bytes
  EURUSD.parquet.meta.json          492 bytes
```

One parquet + one `.meta.json` per pair attempted (one pair attempted in this check). Naming convention `<file>.parquet.meta.json` per spec.

**Full sidecar contents (`EURUSD.parquet.meta.json`).**

```json
{
  "cache_key": "16b7daaa7ee1550b826d9d0047334248552599c326c7815053bba0197a660e96",
  "columns": [
    "open_bid",
    "high_bid",
    "low_bid",
    "close_bid",
    "open_ask",
    "high_ask",
    "low_ask",
    "close_ask",
    "volume",
    "spread_close",
    "bid_ask_data_quality"
  ],
  "created_at": "2026-05-21T23:55:32Z",
  "layer": "m1",
  "n_rows": 5964931,
  "pair": "EURUSD",
  "source_m1_manifest_sha256": "a2d19a9cfc4c8ff2ff1e5edbb51e97b144267ccdbc019d3193cdac432caa679d"
}
```

Both sha256 fields are 64 hex characters (well-formed). `cache_key` derives from the sorted `(relpath, sha256)` pairs of EURUSD's entries in `m1_manifest.json` per `core.data.cache_keys.m1_cache_key_for_pair`. `source_m1_manifest_sha256` is the sha256 of the full `m1_manifest.json` file itself (coarse provenance pointer; the load-bearing invariant is `cache_key`).

**Cache invalidation on upstream manifest change.**

Re-ran the two unit tests that exercise this directly:

```bash
py -m pytest \
  "tests/test_histdata_loader.py::test_load_m1_cache_invalidates_on_manifest_change" \
  "tests/test_aggregator.py::test_aggregate_tf_cache_invalidates_with_m1" -v
```

Result: `2 passed in 0.32s`. Both tests:
1. Build a fixture, load the pair, capture the sidecar `cache_key`.
2. Mutate one CSV's sha256 in the fixture's `m1_manifest.json` (writing the file back with the standard sort_keys + LF terminator).
3. Reload the pair, capture the new sidecar `cache_key`.
4. Assert old ≠ new.

The aggregator test additionally proves that changing the M1 layer's manifest entry cascades into the **H1** TF cache key (because `tf_cache_key(m1_key, tf)` consumes the upstream key).

**Note on manual re-confirmation on real data.** A direct manual repro (mutate a fork of the real `m1_manifest.json` and reload the real EURUSD) was attempted but blocked by Windows' lack of admin-required symlink privilege — copying the actual 5,488-row EURUSD pair-dir into tmp would have moved ~0.5 GB of CSVs which violates the "read-only" verification scope. The two unit tests above are the canonical proof; both pass.

**Verdict: PASS.** Cache files present and well-formed; sidecar fields all populated with valid sha256 keys; invalidation-on-manifest-change verified via the unit tests.

**Concerns:** none material. The Windows symlink limitation is environmental, not a PR-A defect.

---

## VERDICT: PR-A safe to merge

All six checks pass. Specific highlights worth recording:

- **Real-data cache speedup is 98.56×** (dispatch target: ≥10×).
- **Real-data H4 byte-identical** across two-run aggregation (26,170 H4 bars on 16 years of EURUSD).
- **Zero `spread_floor*` references** in any PR-A new code (modules + new config).
- **Zero failures, zero errors, zero silent skips** across 1,223 collected tests in the full repo sweep.

No anomalies, no concerns flagged. Recommending merge of PR-A; PR-B (real-spread + fill mechanics + multi-pair sim, including deletion of `configs/spread_floors_5ers.yaml`) can proceed immediately on top.
