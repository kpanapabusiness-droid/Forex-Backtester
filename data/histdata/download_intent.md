# HistData download intent

Read-first deliverable. Captures plan, scope, and assumptions before any download executes.

## Scope revision against original dispatch (chat-approved)

Original dispatch specified "M1 OHLC bid+ask" as the canonical source, with directory layout
`data/histdata/<PAIR>/bid/<YYYY>/...` and `data/histdata/<PAIR>/ask/<YYYY>/...`.

Probe of histdata.com confirms HistData's free ASCII product line provides only:
1. **`1-minute-bar-quotes`** — single OHLC stream (bid-side). Form has one `timeframe=M1` field
   and no bid/ask selector. Page contains zero mentions of "bid" or "ask" for this product.
2. **`tick-data-quotes`** — bid+ask per tick (single combined CSV per pair-month).

There is no native HistData M1 ask product. Separate bid/ask bar files only exist in the
NinjaTrader format at 1-second granularity (tick-derived).

Chat resolution: pivot to **tick data as canonical source**, with M1 bid OHLC + M1 ask OHLC as
a derived layer produced by aggregation. Overrides the dispatch's "Tick data is NOT in scope"
rule. The aggregation lands as part of this dispatch (it is sample preparation, not analysis).

## Canonical source

HistData ASCII tick data, one zip per pair-month at the URL pattern
`https://www.histdata.com/download-free-forex-historical-data/?/ascii/tick-data-quotes/<pair_lower>/<year>/<month>`.
POST to `/get.php` with `timeframe=T`, `platform=ASCII`. CSV inside each zip has format
`YYYYMMDD HHMMSSnnn,bid,ask,volume` (volume is always 0 — synthetic). Timezone is EST without
DST, i.e. fixed UTC−5 year-round per HistData's own docs, with one known exception: October
months may show a single-hour backward jump at the EU DST "fall back" transition (legitimate
upstream feed behavior; harmless for bar aggregation since buckets are order-independent
within the bucket).

## Pairs (28)

```
AUDCAD, AUDCHF, AUDJPY, AUDNZD, AUDUSD,
CADCHF, CADJPY,
CHFJPY,
EURAUD, EURCAD, EURCHF, EURGBP, EURJPY, EURNZD, EURUSD,
GBPAUD, GBPCAD, GBPCHF, GBPJPY, GBPNZD, GBPUSD,
NZDCAD, NZDCHF, NZDJPY, NZDUSD,
USDCAD, USDCHF, USDJPY
```

Source coverage start dates vary per pair — actual earliest available month is recorded
per-pair in `gap_report.md` after the download. Pairs with partial coverage are still pulled.

## Window

```
Start:  2010-01-01 (or earliest HistData coverage per pair, whichever is later)
End:    most recent complete month at run time (currently 2026-04)
```

`iter_yearmonths(2010, 1, 2026, 4)` yields 196 months × 28 pairs = 5,488 file specs.

## Directory layout

```
data/
  histdata/
    AUDCAD/
      tick/
        2010/
          DAT_ASCII_AUDCAD_T_201001.zip
          DAT_ASCII_AUDCAD_T_201002.zip
          ...
        2011/
          ...
      m1/                          # derived, written by aggregator
        bid/
          2010/
            AUDCAD_M1_BID_201001.csv
            ...
        ask/
          2010/
            AUDCAD_M1_ASK_201001.csv
            ...
    AUDCHF/
      ...
    manifest.json                  # tick zip integrity (sha256, size, rows, validate result)
    m1_manifest.json               # derived M1 CSV integrity (sha256, size, rows)
    gap_report.md                  # per-pair coverage start, real gaps, bid/ask pairing
    download.log                   # full download history
    download_progress.md           # resume marker if session interrupted
    integrity_failures.md          # sha256/parse/bid<=ask failures
    row_count_anomalies.md         # outlier file row counts
    download_intent.md             # this file
```

Tick zips are the canonical artefact. M1 CSVs are derived and re-creatable from tick + the
aggregation script, but committed (well — their manifest is committed; the CSVs themselves
are gitignored along with the zips).

## Integrity protocol

Applied per file at download time:

1. **HTTP-level**: 3 retries with exponential backoff (5s, 10s, 20s). Non-ZIP bodies (typically
   HTML error pages from the server) are classified `MISSING_ON_SOURCE` rather than retried —
   this happens for months HistData genuinely doesn't have.
2. **ZIP-level**: `BadZipFile` exceptions are caught and the file is quarantined.
3. **CSV-level**: pandas C-engine parse with explicit dtypes. Failures are quarantined.
4. **Semantic**: bid ≤ ask on every row (any violation quarantines the file). Median spread
   in [0.05, 50] pips (gross-corruption sanity, not a tight quality check).
5. **sha256**: computed on the raw zip bytes and recorded in `manifest.json` keyed by relative
   path under `data/histdata/`.

Quarantine: failures are moved to `data/histdata/<PAIR>/tick/<YEAR>/quarantine/` with a
sibling `.reason.txt` file. Quarantined files are recorded in `integrity_failures.md`.

Gap detection (post-download):
- Per pair, identify missing pair-months in the expected range.
- Distinguish "real gap" (HistData doesn't have this month for this pair — coverage starts
  later) from "missing download" (months either side are present). Real gaps are documented
  in `gap_report.md`; missing-downloads are re-attempted.

## Backup

Default destination: `~/histdata_backup/` (= `C:\Users\panap\histdata_backup\` on this host).
**This is INSUFFICIENT long-term — same physical disk, no off-machine redundancy.** Chat must
provide a real off-machine destination (external drive, NAS, cloud bucket) before any
downstream protocol work uses this data as the source of truth.

Backup script behavior:
- Mirrors `data/histdata/` to the destination preserving the directory tree.
- Idempotent (re-running compares sha256 and only copies missing/changed files).
- Recomputes sha256 of the destination and compares against `manifest.json`. Any mismatch is
  flagged in the run report.
- `verify_backup.py` is a separate verification-only pass that does not modify the backup.

## Existing partial download from prior spread audit

Per dispatch: "any existing partial HistData download from prior work (e.g. spread audit) is
to be checked but NOT trusted — re-download or re-verify to integrate cleanly."

Probed: `data/external/histdata/` does NOT exist in this worktree. The previous data loss
event referenced by chat appears to have removed it. Fresh start; nothing to integrate. The
existing downloader script at `scripts/spread_validation/01c_download_histdata_extended.py`
is read for pattern reference only — its `data/external/histdata/` cache path and its
combined-file naming do not match the new `data/histdata/<PAIR>/tick/<YEAR>/` layout, so a
new downloader is built rather than retrofitting the old one.

## TLS bypass scope

HistData's TLS certificate is expired or otherwise invalid (`WebFetch` returns
"certificate has expired" against `https://www.histdata.com`). Per established project
methodology, `verify=False` is applied **only** to the `requests.Session` instance used for
histdata.com — no other HTTP call in this dispatch bypasses verification. The bypass is
scoped to the source-of-truth provider, not blanket.

## Rate limiting

2 seconds between requests (sequential), configurable via CLI flag. HistData has been
observed to throttle aggressive downloaders; 2s is a safe default. Total throttle floor:
5,488 file specs × 2s ≈ 3 hours. Per-file download time adds another 2–10 seconds depending
on pair-month size (older crosses are smaller, recent majors larger). Realistic wall-clock:
**6–14 hours for the full tick download**, plus 1–3 hours for M1 aggregation and 30–60
minutes for the first backup.

If the session window doesn't permit the full run in one session: the downloader is
resumable. `manifest.json` is written incrementally per file. On restart, files whose
recorded sha256 still matches what's on disk are skipped (`CACHED` status); only missing or
corrupted files are re-fetched.

## Definition of done

1. `data/histdata/` populated with tick zips per the layout above.
2. `data/histdata/manifest.json` lists every file with sha256, size, and row count.
3. `data/histdata/<PAIR>/m1/bid/<YYYY>/...` and `.../m1/ask/...` populated by the aggregator;
   `m1_manifest.json` records each derived CSV.
4. `data/histdata/gap_report.md` documents per-pair coverage and any real gaps.
5. `data/histdata/download.log` records the full download history.
6. `scripts/histdata_download.py` committed.
7. `scripts/histdata_aggregate_m1.py` committed.
8. `scripts/histdata_backup.py` + `scripts/verify_backup.py` committed.
9. At least one backup executed to the default destination, with the insufficiency caveat
   flagged in `docs/DATA_FOUNDATION.md`.
10. `docs/DATA_FOUNDATION.md` committed.
11. `.gitignore` updated to exclude `data/histdata/**/*.csv` and `data/histdata/**/*.zip`
    while re-including manifests, gap reports, and the download log.
12. Final commit: `data: HistData tick foundation (M1 bid+ask derived) — 28 pairs, 2010-current, backed up`.

## Out of scope (per dispatch)

- Touching `configs/spread_floors_5ers.yaml`. Real spread becomes derivable from the new data
  but this dispatch does not modify the floor file.
- Running the backtester.
- Validating any downstream pipeline (spread floor → backtester, etc.).
- Cleaning up prior data dirs (`data/external/...` etc.) — chat decides separately.
- Modifying any L-arc or protocol document.
