# Data Foundation

> Canonical statement of where this project's price data lives, how it was
> sourced, how it's verified, and how it's backed up. The previous data loss
> event made this contract necessary: any downstream protocol or backtest
> work that depends on price data MUST cite this doc.

## Canonical source

**HistData ASCII tick data** — bid+ask quotes per tick, one zip per pair-month.

Source: `https://www.histdata.com/download-free-forex-historical-data/?/ascii/tick-data-quotes/...`

Chosen over HistData's M1 bar product because the latter is one-sided (bid
OHLC only — verified directly against histdata.com, see `download_intent.md`).
Tick gives bid+ask at maximum granularity, from which M1 (and any higher TF)
can be aggregated with both sides preserved.

This pivot overrode the original dispatch's "tick not in scope" rule; the
"M1 OHLC bid+ask" deliverable in the dispatch turned out to be incompatible
with HistData's actual product line.

## Coverage

- **28 FX pairs** — the KH-24 / L-arc canonical universe.
- **2010-01 → 2026-04** — 196 months per pair, all pairs complete (see
  `data/histdata/gap_report.md`). HistData had no real coverage gaps for
  any pair in the universe across the full window.
- **Tick granularity** — one tick = one quote update (bid, ask, volume=0).
  Volume is always 0 in HistData tick data (synthetic).
- **EST without DST timezone** — fixed UTC−5 year-round, with one known
  exception: October files may show a single-hour backward jump at the EU DST
  "fall back". The aggregator handles this transparently (minute buckets are
  order-independent within the bucket).

### Final dataset stats

- **Tick layer**: 5,488 files, **52.26 GB compressed**, **8.79 billion ticks**.
- **M1 derived layer**: 10,976 CSV files (5,488 bid + 5,488 ask),
  **18.22 GB**, **333.5 million minute bars** (~167.8M per side).
- **Heaviest pair**: EURJPY (2.97 GB tick, 509M ticks).
- **Lightest pair**: NZDUSD (1.21 GB tick, 207M ticks).
- **Integrity**: 0 sha mismatches, 0 per-row bid>ask violations, 0 parse
  failures. 265 row-count anomalies, all positive z-score (COVID March 2020,
  2022 macro shocks, recent 2025 vol — expected high-activity months).

## Location

```
data/histdata/
  <PAIR>/
    tick/
      <YYYY>/
        DAT_ASCII_<PAIR>_T_<YYYYMM>.zip      # canonical, raw
    m1/
      bid/
        <YYYY>/
          <PAIR>_M1_BID_<YYYYMM>.csv         # derived, aggregated from tick
      ask/
        <YYYY>/
          <PAIR>_M1_ASK_<YYYYMM>.csv         # derived, aggregated from tick
  manifest.json           # tick zip integrity (sha256, size, rows)
  m1_manifest.json        # derived M1 CSV integrity (sha256, size, rows)
  gap_report.md           # per-pair coverage + bid/ask pairing + real gaps
  download.log            # full download history
  download_intent.md      # plan / read-first deliverable
  integrity_failures.md   # post-pass full-file integrity failures (if any)
  row_count_anomalies.md  # files outside per-pair row-count band
```

## Integrity

Three layers, each with its own artefact:

### 1. Download-time validation (per file, fast)

Built into `scripts/histdata_download.py`:
- HTTP 3-retry with exponential backoff (5s, 10s, 20s).
- Non-ZIP responses classified `MISSING_ON_SOURCE` (real gap from upstream).
- ZIP magic check (`PK` header).
- `zipfile.ZipFile` open + CSV member present.
- Streaming line count (newline scan, no parse).
- Sample bid<=ask + median spread on first 5k + last 5k rows.

Files failing any of these are quarantined to
`<PAIR>/tick/<YYYY>/quarantine/` with a sibling `.reason.txt`.

### 2. Per-file sha256

Computed at download time over the raw zip bytes, recorded in
`data/histdata/manifest.json` keyed by relative path under `data/histdata/`.
Re-running the downloader is idempotent: any file whose on-disk sha256
matches the manifest is skipped (`CACHED`).

### 3. Post-pass full-row integrity

`scripts/histdata_verify.py` (runs after the download completes):
- Recomputes sha256 of every file on disk; compares to `manifest.json`.
- Parses every row of every CSV; checks bid<=ask on every row.
- Records full row count (writes back to manifest if previously zero).
- Flags row-count anomalies (files >2σ from per-pair mean) to
  `row_count_anomalies.md`.
- Writes any failure to `integrity_failures.md` — downloader picks up these
  files on next run for re-fetch.

### Gap detection

`scripts/histdata_gap_report.py`:
- Walks the manifest and the expected pair × month grid.
- For each pair, distinguishes:
  - **Real coverage start** — HistData simply doesn't have early months for
    this pair; recorded as `earliest_month` in the report.
  - **Missing download** — a month inside the pair's coverage band that has
    no file. Re-attempted by the downloader on next run.
- Bid/ask pairing: under the tick source, bid+ask are combined in one CSV
  per pair-month, so this check reduces to "every expected pair-month tick
  zip is present". Documented in `gap_report.md`.

## Backup

**Script:** `scripts/histdata_backup.py`

**Default destination:** `~/histdata_backup/` (= `C:\Users\panap\histdata_backup`
on this host).

**⚠️ This is INSUFFICIENT long-term — same physical disk as the project,
no off-machine redundancy. The previous data loss event is the reference
case. Chat must provide a real off-machine destination (external drive,
NAS, cloud bucket mount) before any protocol work uses this data.**

Backup script properties:
- Idempotent (re-running compares sha256, copies only missing/changed files).
- Recomputes destination sha256 after copy; flags mismatches.
- `scripts/verify_backup.py` is a separate read-only verification pass.

**Last backup:** 2026-05-21T15:40:19 UTC — 16,464 files copied (52.26 GB tick + 18.22 GB M1), 0 failures, 0 sha mismatches, elapsed 279s. Backup report: `data/histdata/backup_report.md`.

**Last verified:** 2026-05-21 (verify_backup.py pass — see `data/histdata/backup_verify_report.md`).

## Spread handling

HistData tick data carries real bid and ask prices per quote update. Once the
M1 derived layer is built, the actual spread at each minute is
`ask[close] - bid[close]`.

**Strict deprecation of `spread_floors_5ers.yaml`** (locked in PR-B,
2026-05-22). The file has been deleted along with `core/spread_floor.py`,
the body-hash lock test, and the `.gitattributes` pin. v3 uses ONLY the
per-bar real spread from the HistData M1 layer. Bars with zero/negative or
NaN spread (`bid_ask_data_quality != ok`) are flagged and silently dropped
from trade simulation — no fallback to any external floor file. This
matches L_PROTOCOL §1 non-negotiable on real bid/ask.

The full data-quality scan across all 28 pairs is in
[docs/dispatches/pr_b_data_quality.md](dispatches/pr_b_data_quality.md).
Per-bar gating is implemented in `core/spread/real_spread.py` and consulted
by `core.sim.multipair_backtester._check_exits` /
`._fill_pending_entries`.

## Aggregation

M1 is the canonical derived granularity (one row per minute, two CSVs per
pair-month: one bid OHLC, one ask OHLC). Higher TFs (M5, M15, H1, H4, D1, W1)
are derived from M1 by the backtester at run time, not pre-materialised. This
keeps storage bounded and avoids the lookahead risks of off-line
multi-resolution caches.

`scripts/histdata_aggregate_m1.py` produces the M1 layer. It is re-runnable;
the M1 CSVs are derived artefacts (gitignored alongside the zips), but
`m1_manifest.json` is committed so re-derivation can be verified for
determinism.

Aggregation logic:
- Parse tick ts as EST (UTC−5).
- Convert to UTC by adding 5 hours.
- Floor to minute → bucket key.
- For each minute: emit one bid OHLC row (O=first bid, H=max bid, L=min bid,
  C=last bid) and one ask OHLC row.
- Volume column = count of ticks in that minute (HistData's source volume is
  always 0, so we substitute the tick count which is informative).

CSV format (both bid and ask files identical schema):
```
timestamp_utc,open,high,low,close,volume
```

## Lessons applied from previous data loss

1. **Source-of-truth diversification** — HistData is free, public, and
   re-downloadable. If we lose local storage we can rebuild in ~12-24
   hours of wall-clock download.
2. **Per-file sha256 manifest** — corruption detectable without re-download.
3. **Local primary + at least one off-machine backup** — *backup destination
   still pending; default same-disk path is a temporary stopgap.*
4. **All data text artefacts in git** — manifest, gap report, intent doc,
   download log are tracked. Re-derivation is reproducible from script +
   manifest + source URL.

## Cache layer (added in v3 backtester PR-A)

The HistData raw layer (zips + M1 CSVs) is read-only input. The v3
backtester maintains its own parquet cache for fast iteration:

```
data/cache/
  m1/<PAIR>.parquet           # joined bid+ask M1 (per pair)
  {M5,M15,M30,H1,H4,D1,W1}/<PAIR>.parquet   # aggregated TFs
  *.parquet.meta.json         # sidecar: cache_key + source manifest sha
  features/<arc_id>/<feature_set_hash>.parquet   # PR-D feature cache
  features/<arc_id>/<feature_set_hash>.parquet.meta.json
```

- **Cache keys** derive from `data/histdata/m1_manifest.json` via
  `core.data.cache_keys`. Any change to the upstream M1 layer
  rewrites the manifest and cascades through.
- **Aggregator** is `core.data.aggregator.aggregate(pair, tf, ...)`.
  Higher TFs are derived from M1 at run time, not pre-materialised.
  See [BACKTESTER_ARCHITECTURE.md](BACKTESTER_ARCHITECTURE.md).
- The cache directory is gitignored; it is rebuilt locally on first
  run.

## Out of scope (for this foundation dispatch)

- Running the backtester or any L-arc step.
- Modifying any L-arc or protocol document.
- Cleaning up prior data directories (`data/external/...` etc.) — chat
  decides separately.
- Touching the live KH-24 system or its VPS.
