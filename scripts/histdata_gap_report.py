"""histdata_gap_report.py — emit data/histdata/gap_report.md from manifest state.

Walks the tick manifest + expected pair × month grid; for each pair distinguishes
"real coverage gap" (HistData has no data for that pair-month) from "missing
download" (a hole between two observed months — usually a transient HTTP failure
that should be re-attempted on the next downloader run).

Reads the download.log (if present) to attribute MISSING_ON_SOURCE vs
FAIL_AFTER_RETRY for each missing month, since download.log records the
upstream outcome.

Output: data/histdata/gap_report.md, plain markdown, one section per pair.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PAIRS: tuple[str, ...] = (
    "AUDCAD", "AUDCHF", "AUDJPY", "AUDNZD", "AUDUSD",
    "CADCHF", "CADJPY", "CHFJPY",
    "EURAUD", "EURCAD", "EURCHF", "EURGBP", "EURJPY", "EURNZD", "EURUSD",
    "GBPAUD", "GBPCAD", "GBPCHF", "GBPJPY", "GBPNZD", "GBPUSD",
    "NZDCAD", "NZDCHF", "NZDJPY", "NZDUSD",
    "USDCAD", "USDCHF", "USDJPY",
)

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_TARGET: Path = REPO_ROOT / "data" / "histdata"


def iter_yyyymm(y0: int, m0: int, y1: int, m1: int) -> list[str]:
    out: list[str] = []
    y, m = y0, m0
    while (y, m) <= (y1, m1):
        out.append(f"{y:04d}{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


def parse_yyyymm(s: str) -> tuple[int, int]:
    return int(s[:4]), int(s[4:])


def last_complete_month(now: datetime | None = None) -> str:
    n = now or datetime.now(timezone.utc)
    y, m = (n.year, n.month - 1) if n.month > 1 else (n.year - 1, 12)
    return f"{y:04d}{m:02d}"


def parse_download_log(log_path: Path) -> dict[tuple[str, str], str]:
    """Return latest-event-per-(pair, ym) from download.log.

    Lines look like:
        2026-05-19T20:47:04+00:00  event=ok  pair=EURUSD  ym=201501  ...
    """
    out: dict[tuple[str, str], str] = {}
    if not log_path.exists():
        return out
    for ln in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m_pair = re.search(r"\bpair=(\w+)\b", ln)
        m_ym = re.search(r"\bym=(\d{6})\b", ln)
        m_evt = re.search(r"\bevent=(\w+)\b", ln)
        if not (m_pair and m_ym and m_evt):
            continue
        out[(m_pair.group(1), m_ym.group(1))] = m_evt.group(1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=str, default=str(DEFAULT_TARGET))
    ap.add_argument("--start", type=str, default="201001")
    ap.add_argument("--end", type=str, default=None)
    args = ap.parse_args()

    target = Path(args.target).resolve()
    manifest_path = target / "manifest.json"
    if not manifest_path.exists():
        print(f"BLOCKER: no manifest at {manifest_path}", flush=True)
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    start = args.start
    end = args.end or last_complete_month()
    sy, sm = parse_yyyymm(start)
    ey, em = parse_yyyymm(end)
    expected_months = iter_yyyymm(sy, sm, ey, em)

    log_events = parse_download_log(target / "download.log")

    # Build per-pair file dict (observed months).
    observed: dict[str, set[str]] = defaultdict(set)
    sizes: dict[str, int] = defaultdict(int)
    rows: dict[str, int] = defaultdict(int)
    for pair, pe in manifest.get("pairs", {}).items():
        for rel, entry in pe.get("files", {}).items():
            # rel = <PAIR>/tick/<YYYY>/DAT_ASCII_<PAIR>_T_<YYYYMM>.zip
            ym = rel.split("_")[-1].rstrip(".zip")
            if len(ym) == 6 and ym.isdigit():
                observed[pair].add(ym)
                sizes[pair] += int(entry.get("size_bytes", 0))
                rows[pair] += int(entry.get("rows", 0))

    # Per-pair coverage table
    lines: list[str] = []
    lines.append("# HistData coverage and gap report")
    lines.append("")
    lines.append(
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}  ")
    lines.append(f"Window: {start} → {end} ({len(expected_months)} months)")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        "| pair | observed | expected | coverage | first | last | size_MB | total_rows |")
    lines.append(
        "|------|---------:|---------:|---------:|-------|------|--------:|-----------:|")
    grand_obs = 0
    grand_size = 0
    grand_rows = 0
    for pair in PAIRS:
        obs = sorted(observed.get(pair, set()))
        if obs:
            first = obs[0]
            last = obs[-1]
        else:
            first = last = "—"
        coverage = f"{len(obs) / len(expected_months) * 100:5.1f}%"
        lines.append(
            f"| {pair} | {len(obs)} | {len(expected_months)} | {coverage} | "
            f"{first} | {last} | {sizes[pair]/1e6:7.1f} | {rows[pair]:>10d} |"
        )
        grand_obs += len(obs)
        grand_size += sizes[pair]
        grand_rows += rows[pair]
    lines.append(
        f"| **TOTAL** | **{grand_obs}** | **{len(PAIRS) * len(expected_months)}** | "
        f"**{grand_obs / (len(PAIRS) * len(expected_months)) * 100:5.1f}%** | — | — | "
        f"**{grand_size/1e6:7.1f}** | **{grand_rows:>10d}** |"
    )
    lines.append("")

    # Per-pair detail: missing months, real-gap vs missing-download attribution
    lines.append("## Per-pair detail")
    lines.append("")
    for pair in PAIRS:
        obs = sorted(observed.get(pair, set()))
        missing = [ym for ym in expected_months if ym not in observed.get(pair, set())]
        first_obs = obs[0] if obs else None

        # Classify each missing month
        real_gap: list[str] = []         # before first observed month
        missing_download: list[str] = [] # after first observed, no zip
        for ym in missing:
            if first_obs is None or ym < first_obs:
                real_gap.append(ym)
            else:
                missing_download.append(ym)

        # Cross-check missing_download against download.log
        upstream_missing = []
        retry_failed = []
        unattempted = []
        for ym in missing_download:
            evt = log_events.get((pair, ym))
            if evt == "missing_on_source":
                upstream_missing.append(ym)
            elif evt == "fail_after_retry":
                retry_failed.append(ym)
            else:
                unattempted.append(ym)

        lines.append(f"### {pair}")
        lines.append("")
        if obs:
            lines.append(f"- First observed month: **{obs[0]}**  (real coverage start)")
            lines.append(f"- Last observed month:  **{obs[-1]}**")
            lines.append(f"- Observed months: **{len(obs)} / {len(expected_months)}**")
        else:
            lines.append("- **No months observed.**")
        if real_gap:
            head = real_gap[0]
            tail = real_gap[-1]
            lines.append(
                f"- Real coverage gap (before HistData has data): "
                f"{len(real_gap)} months, {head}–{tail}"
            )
        if upstream_missing:
            lines.append(
                f"- Upstream-missing months (HistData returned non-zip — likely "
                f"genuine source gap): {len(upstream_missing)}"
            )
            lines.append("    - " + ", ".join(upstream_missing[:40])
                         + (" …" if len(upstream_missing) > 40 else ""))
        if retry_failed:
            lines.append(
                f"- Download failed after retries (re-attempt on next run): "
                f"{len(retry_failed)}"
            )
            lines.append("    - " + ", ".join(retry_failed[:40])
                         + (" …" if len(retry_failed) > 40 else ""))
        if unattempted:
            lines.append(
                f"- Unattempted months (no log entry — re-run downloader): "
                f"{len(unattempted)}"
            )
            lines.append("    - " + ", ".join(unattempted[:40])
                         + (" …" if len(unattempted) > 40 else ""))
        lines.append("")

    # Bid/ask pairing — under the tick source it's trivially satisfied (one zip
    # = bid+ask). But still record the explicit statement so a future reader
    # can see this dimension was considered.
    lines.append("## Bid/ask pairing")
    lines.append("")
    lines.append(
        "Under the tick source, each pair-month zip contains a single CSV with "
        "bid AND ask columns per row. The bid/ask pairing check is therefore "
        "satisfied tautologically by the existence of the tick zip. The "
        "downstream M1 aggregator (`scripts/histdata_aggregate_m1.py`) emits "
        "matched bid/ask CSV pairs per pair-month; any discrepancy there is "
        "an aggregator bug, not a source-data gap."
    )
    lines.append("")

    out_path = target / "gap_report.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
