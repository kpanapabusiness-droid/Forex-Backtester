"""DST-aware mini fixture for 5ers EET boundary tests.

Synthesises HistData M1 bid+ask CSVs spanning both EU DST transitions
within a single year so the aggregator's 5ers_eet boundary code is
exercised at:

  - spring forward (last Sunday of March, EET 03:00 → EEST 04:00)
  - autumn fall back (last Sunday of October, EEST 04:00 → EET 03:00)

The fixture covers Mar-2024 and Oct-2024 minutes only — enough to label
H4 / D1 bars at EET 00:00/04:00/.../20:00 and confirm DST is handled
without producing duplicate or missing bars.
"""

from __future__ import annotations

import calendar
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path


@dataclass(frozen=True)
class DstFixtureSpec:
    pairs: tuple[str, ...] = ("EURUSD",)
    # (yyyymm, day_start_inclusive, day_end_inclusive) windows. Picked so
    # each window straddles a DST transition.
    windows: tuple[tuple[str, int, int], ...] = (
        ("202403", 29, 31),  # 2024-03-31 = spring forward; include 29 pre + 31 transition
        ("202404", 1, 2),  # 2024-04-01..02 = post spring-forward steady state (EEST)
        ("202410", 25, 27),  # 2024-10-27 = autumn fall-back; include 25 pre + 27 transition
        ("202410", 28, 29),  # 2024-10-28..29 = post fall-back steady state (EET)
    )
    base_bid: float = 1.0850
    spread_pips: float = 0.5


def _gen_rows(pair: str, side: str, yyyymm: str, day_lo: int, day_hi: int, spec: DstFixtureSpec):
    year = int(yyyymm[:4])
    month = int(yyyymm[4:])
    _, last_day = calendar.monthrange(year, month)
    day_hi_capped = min(day_hi, last_day)
    pair_offset = sum(ord(c) for c in pair) % 17 / 10_000
    start_bid = spec.base_bid + pair_offset
    spread_price = spec.spread_pips * 0.0001

    rows: list[tuple[str, float, float, float, float, int]] = []
    walk = 0
    cur = datetime(year, month, day_lo, 0, 0, tzinfo=timezone.utc)
    end = datetime(year, month, day_hi_capped, 23, 59, tzinfo=timezone.utc)
    while cur <= end:
        b = start_bid + walk * 0.00002
        o = round(b, 5)
        h = round(b + 0.00015, 5)
        lo = round(b - 0.00005, 5)
        c = round(b + 0.00010 + (walk % 3) * 0.00001, 5)
        if side == "ask":
            o, h, lo, c = (round(x + spread_price, 5) for x in (o, h, lo, c))
        ts = cur.strftime("%Y-%m-%dT%H:%M:%SZ")
        volume = 5 + (walk % 4)
        rows.append((ts, o, h, lo, c, volume))
        walk += 1
        cur += timedelta(minutes=1)
    return rows


def _write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("timestamp_utc,open,high,low,close,volume\n")
        for ts, o, h, lo, c, v in rows:
            f.write(f"{ts},{o},{h},{lo},{c},{v}\n")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_dst_fixture(root: Path, spec: DstFixtureSpec | None = None) -> Path:
    """Materialise the DST mini fixture under ``root``."""
    spec = spec or DstFixtureSpec()
    root.mkdir(parents=True, exist_ok=True)

    manifest_pairs: dict[str, dict] = {}
    for pair in spec.pairs:
        files: dict[str, dict] = {}
        # Aggregate per (yyyymm, side) into one CSV file containing all windows
        # for that month, so the manifest has one file per pair-month per side.
        by_month: dict[str, dict[str, list]] = {}
        for yyyymm, day_lo, day_hi in spec.windows:
            for side in ("bid", "ask"):
                by_month.setdefault(yyyymm, {}).setdefault(side, []).extend(
                    _gen_rows(pair, side, yyyymm, day_lo, day_hi, spec)
                )
        for yyyymm, sides in by_month.items():
            year = yyyymm[:4]
            for side, rows in sides.items():
                rel = f"{pair}/m1/{side}/{year}/{pair}_M1_{side.upper()}_{yyyymm}.csv"
                path = root / rel
                rows.sort(key=lambda r: r[0])
                _write_csv(path, rows)
                files[rel] = {
                    "n_ticks_source": 0,
                    "rows": len(rows),
                    "sha256": _sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
        manifest_pairs[pair] = {"files": files}

    manifest = {
        "aggregated_at": datetime(2026, 1, 1, tzinfo=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S+00:00"
        ),
        "pairs": manifest_pairs,
    }
    (root / "m1_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return root
