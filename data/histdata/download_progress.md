# Download progress marker

Live state of the HistData tick download. Resume context for any future
session that picks up where this one left off.

## Run state

- **First launched:** 2026-05-19T20:58:37+00:00 UTC (killed after AUDCAD completed)
- **Relaunched with session rotation:** 2026-05-19T22:32:07+00:00 UTC
- **Spec count:** 5,488 (28 pairs × 196 months, 201001 → 202604)
- **Rate:** 2.0 s between requests
- **Session rotation:** every 50 files (and on initial Session creation), reason
  recorded in download.log as `event=session_rotate`.
- **Manifest:** `data/histdata/manifest.json` (incremental, updated every 20 files
  and at end of every pair).
- **Log:** `data/histdata/download.log` (one line per per-file event +
  start/progress/pair_done/finish summary lines).

## Throttle observation (informational)

HistData has a progressive throttle that does NOT fully reset on Session
rotation. Per-file response time within a pair escalates from ~13s in the
first quartile to ~30s in the last quartile, even with Session rotation every
50 files. Suggests an IP-level throttle layer in addition to session-level
state. Realistic per-file mean at 2s rate: ~25s.

Effective throughput on this host/IP: ~25s/file = ~144 files/hour.
5,488 specs total → ~38 hours worst-case wall-clock.

Possible mitigations if needed (NOT applied yet — currently letting the
multi-day download proceed since the dispatch budget allows session
resume):
- Increase `--rate` from 2s to 4-5s. Loses ~3h to pure waits; may save more
  in throttle-induced response slowdown if HistData throttles less
  aggressively at lower request rates.
- Parallelise across pairs from a single host (risk: IP-throttle gets worse,
  may net-out negative).
- Run from a different network egress (no infra to do this currently).

## Resume protocol

If this session ends mid-download:

1. Re-run the downloader: `python scripts/histdata_download.py --start 201001`
2. Any file with on-disk sha256 matching the manifest is skipped (`CACHED`).
3. Missing or sha-changed files are re-fetched.
4. Manifest writes are atomic (.tmp + rename); incomplete manifest writes are
   not possible — at worst you lose the last 0-19 file entries since the last
   incremental save, and those are re-fetched on resume.
5. Quarantined files stay quarantined; the next run will try once more to
   re-download (skipping if already cached).

## Operational notes

- TLS bypass is scoped to histdata.com only; no other call in this codebase
  is verify=False.
- HistData may serve a non-zip body for months it genuinely lacks (recorded as
  `event=missing_on_source` in download.log). These count as "real coverage
  gaps" in `gap_report.md` and are not retried.
- The first request to HistData per session can be slow (~50 s); subsequent
  requests are ~5-10 s.

## After download completes

In order:
1. `python scripts/histdata_verify.py --update-manifest --fix`
   - Recomputes sha256 on disk vs manifest, parses full CSV, checks bid<=ask
     on every row. Deletes sha-mismatched files so the downloader re-fetches
     them. Updates manifest row counts.
2. `python scripts/histdata_gap_report.py`
   - Writes `gap_report.md` with per-pair coverage + missing-month classification.
3. `python scripts/histdata_aggregate_m1.py`
   - Aggregates every tick zip to M1 bid + M1 ask CSVs under
     `<PAIR>/m1/{bid,ask}/<YYYY>/`.
4. `python scripts/histdata_backup.py`
   - Mirrors to `~/histdata_backup/` (default). Writes `backup_report.md`.
5. `python scripts/verify_backup.py`
   - Read-only verification against `manifest.json`. Writes
     `backup_verify_report.md`.

If any verifier step fails: re-run downloader (resumes), then re-run verifier.
