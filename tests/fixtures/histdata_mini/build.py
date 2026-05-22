"""Builder for the synthetic HistData mini-fixture used by data-layer tests.

The real HistData layer is 70 GB and gitignored; we can't ship it with the
repo. Instead each data-layer test constructs a tiny in-tmp version of the
exact layout the loader expects, then exercises the loader against it.

Layout produced (rooted at ``root``):

    root/
      m1_manifest.json
      <PAIR>/m1/bid/<YYYY>/<PAIR>_M1_BID_<YYYYMM>.csv
      <PAIR>/m1/ask/<YYYY>/<PAIR>_M1_ASK_<YYYYMM>.csv

CSV schema mirrors production: ``timestamp_utc,open,high,low,close,volume``.

The fixture is deterministic: same ``root`` + ``pairs`` + ``months`` →
byte-identical CSVs and identical sha256s in the manifest.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class FixtureSpec:
    pairs: tuple[str, ...] = ("EURUSD", "GBPUSD")
    months: tuple[str, ...] = ("201001", "201002")
    minutes_per_month: int = 12  # 12 sequential minutes — enough for OHLC tests at every TF
    base_bid: float = 1.4300
    spread_pips: float = 0.5  # constant 0.5-pip spread → ask = bid + 0.00005


def _gen_side_rows(
    pair: str, side: str, yyyymm: str, spec: FixtureSpec
) -> list[tuple[str, float, float, float, float, int]]:
    """Generate deterministic minute rows for one (pair, side, month).

    The price walk is a simple deterministic ramp from base_bid varying with
    pair index and month index — enough to make different pairs/months
    distinguishable in test assertions, while keeping bid ≤ ask everywhere.
    """
    year = int(yyyymm[:4])
    month = int(yyyymm[4:])
    pair_offset = sum(ord(c) for c in pair) % 17 / 10_000  # ≤ 0.0017
    month_offset = (month - 1) * 0.0001
    start_bid = spec.base_bid + pair_offset + month_offset
    spread_price = spec.spread_pips * 0.0001  # pip = 0.0001 for non-JPY pairs

    rows: list[tuple[str, float, float, float, float, int]] = []
    from calendar import monthrange

    _, last_day = monthrange(year, month)
    minutes_per_day = 24 * 60
    cap = min(spec.minutes_per_month, last_day * minutes_per_day)
    for i in range(cap):
        # Bid OHLC: small zigzag — open = base, high = base + 1.5pip,
        # low = base - 0.5pip, close = base + 1pip + i%3 * 0.1pip
        b = start_bid + i * 0.00002
        o = round(b, 5)
        h = round(b + 0.00015, 5)
        lo = round(b - 0.00005, 5)
        c = round(b + 0.00010 + (i % 3) * 0.00001, 5)
        if side == "ask":
            o, h, lo, c = (round(x + spread_price, 5) for x in (o, h, lo, c))
        # Walk forward minute-by-minute within the month: day rolls every 1440 min.
        day = i // minutes_per_day + 1
        within_day = i % minutes_per_day
        hour = within_day // 60
        minute = within_day % 60
        ts = f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:00Z"
        volume = 5 + (i % 4)
        rows.append((ts, o, h, lo, c, volume))
    return rows


def _write_side_csv(path: Path, rows: list[tuple]) -> None:
    """Write one side CSV with locked formatting for byte-identical reproduction."""
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


def build_fixture(root: Path, spec: FixtureSpec | None = None) -> Path:
    """Materialise the mini HistData layout under ``root``.

    Returns the root path (same as input). After the call:

        root/m1_manifest.json
        root/<PAIR>/m1/bid/<YYYY>/<PAIR>_M1_BID_<YYYYMM>.csv  (per pair × month)
        root/<PAIR>/m1/ask/<YYYY>/<PAIR>_M1_ASK_<YYYYMM>.csv
    """
    spec = spec or FixtureSpec()
    root.mkdir(parents=True, exist_ok=True)

    manifest_pairs: dict[str, dict] = {}
    for pair in spec.pairs:
        files: dict[str, dict] = {}
        for yyyymm in spec.months:
            year = yyyymm[:4]
            for side in ("bid", "ask"):
                rel = f"{pair}/m1/{side}/{year}/{pair}_M1_{side.upper()}_{yyyymm}.csv"
                path = root / rel
                rows = _gen_side_rows(pair, side, yyyymm, spec)
                _write_side_csv(path, rows)
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
