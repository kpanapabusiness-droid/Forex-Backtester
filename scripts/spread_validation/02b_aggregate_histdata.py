"""02b_aggregate_histdata.py — HistData tick CSV → per-hour aggregate.

Parses the ASCII tick CSV inside each HistData ZIP under
    data/external/histdata/{PAIR}_{YYYYMM}.zip
and emits, per pair, a per-hour aggregate CSV at
    data/external/dukascopy_processed/{PAIR}_hourly.csv

HistData ASCII tick CSV format (one row per tick, no header):
    YYYYMMDD HHMMSSnnn,bid,ask,volume
- Timestamp is EST without DST → fixed UTC-5 year-round.
- bid/ask are float prices (5-decimal for non-JPY, 3-decimal for JPY).
- volume is always 0; ignored.

Spread conversion to pips:
    non-JPY:  spread_pips = (ask - bid) * 10000
    JPY:      spread_pips = (ask - bid) * 100
"""

from __future__ import annotations

import csv
import hashlib
import io
import multiprocessing as mp
import os
import sys
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PAIRS: tuple[str, ...] = (
    "AUDCAD", "AUDCHF", "AUDJPY", "AUDNZD", "AUDUSD",
    "CADCHF", "CADJPY", "CHFJPY",
    "EURAUD", "EURCAD", "EURCHF", "EURGBP", "EURJPY", "EURNZD", "EURUSD",
    "GBPAUD", "GBPCAD", "GBPCHF", "GBPJPY", "GBPNZD", "GBPUSD",
    "NZDCAD", "NZDCHF", "NZDJPY", "NZDUSD",
    "USDCAD", "USDCHF", "USDJPY",
)

REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent
CACHE_DIR: Path = REPO_ROOT / "data" / "external" / "histdata"
OUTPUT_DIR: Path = REPO_ROOT / "data" / "external" / "dukascopy_processed"

EST_OFFSET_HOURS: int = 5
CHUNK_SIZE: int = 200_000
FIRST_WINDOW_MS: int = 5 * 60 * 1000
PERCENTILES: tuple[int, ...] = (10, 25, 50, 75, 90, 95, 99)
DEFAULT_WORKERS: int = 8


def is_jpy(pair: str) -> bool:
    return pair.endswith("JPY")


def pip_multiplier(pair: str) -> float:
    return 100.0 if is_jpy(pair) else 10000.0


def classify_session(weekday: int, hour: int) -> str:
    if weekday == 4 and hour == 21:
        return "weekend_edge"
    if weekday == 6 and hour in (22, 23):
        return "weekend_edge"
    if 12 <= hour < 16:
        return "overlap"
    if 7 <= hour < 12:
        return "london"
    if 16 <= hour < 21:
        return "ny"
    return "off_hours"


def fmt(x) -> str:
    if isinstance(x, float):
        if np.isnan(x):
            return ""
        return format(x, ".10g")
    return str(x)


def iter_pair_zips(pair: str) -> list[Path]:
    return sorted(CACHE_DIR.glob(f"{pair}_*.zip"))


def parse_zip_to_chunks(zip_path: Path):
    with zipfile.ZipFile(zip_path) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            return
        for nm in csv_names:
            with zf.open(nm) as fh:
                for chunk in pd.read_csv(
                    fh,
                    header=None,
                    names=["ts", "bid", "ask", "vol"],
                    dtype={"ts": str, "bid": np.float64, "ask": np.float64, "vol": np.int64},
                    chunksize=CHUNK_SIZE,
                    engine="c",
                    low_memory=True,
                ):
                    yield chunk


def aggregate_pair(pair: str) -> dict[pd.Timestamp, dict]:
    pip_m = pip_multiplier(pair)
    bars: dict[pd.Timestamp, dict] = {}

    for zip_path in iter_pair_zips(pair):
        for chunk in parse_zip_to_chunks(zip_path):
            spread_pips_chunk = (chunk["ask"].to_numpy() - chunk["bid"].to_numpy()) * pip_m
            spread_pips_chunk = np.clip(spread_pips_chunk, 0.0, None)

            ts = pd.to_datetime(
                chunk["ts"],
                format="%Y%m%d %H%M%S%f",
                errors="coerce",
                utc=False,
            )
            ts_utc = (ts + pd.Timedelta(hours=EST_OFFSET_HOURS)).dt.tz_localize("UTC")
            hour_open = ts_utc.dt.floor("h")
            ms_into_hour = ((ts_utc - hour_open).dt.total_seconds() * 1000).astype(np.int64)

            df = pd.DataFrame({
                "hour_open": hour_open,
                "spread_pips": spread_pips_chunk,
                "ms_into_hour": ms_into_hour.to_numpy(),
            })
            for hour, sub in df.groupby("hour_open", sort=False):
                arr = sub["spread_pips"].to_numpy()
                ms_arr = sub["ms_into_hour"].to_numpy()
                first_mask = ms_arr < FIRST_WINDOW_MS

                bucket = bars.setdefault(
                    hour,
                    {"spread_vals": [], "spread_vals_first5min": []},
                )
                bucket["spread_vals"].append(arr)
                if first_mask.any():
                    bucket["spread_vals_first5min"].append(arr[first_mask])

    return bars


def finalise_pair_rows(pair: str, bars: dict[pd.Timestamp, dict]) -> list[list[str]]:
    rows: list[list[str]] = []
    for hour in sorted(bars.keys()):
        bucket = bars[hour]
        all_spreads = np.concatenate(bucket["spread_vals"]) if bucket["spread_vals"] else np.array([])
        if all_spreads.size == 0:
            continue
        first5 = (
            np.concatenate(bucket["spread_vals_first5min"])
            if bucket["spread_vals_first5min"]
            else np.array([])
        )

        weekday = int(hour.weekday())
        hour_int = int(hour.hour)
        session = classify_session(weekday, hour_int)
        pcts = np.percentile(all_spreads, PERCENTILES, method="linear")

        if first5.size > 0:
            first5_med = float(np.median(first5))
            first5_mean = float(first5.mean())
        else:
            first5_med = float("nan")
            first5_mean = float("nan")

        rows.append([
            hour.isoformat(),
            str(weekday),
            str(hour_int),
            session,
            str(int(all_spreads.size)),
            str(int(first5.size)),
            fmt(float(pcts[0])),
            fmt(float(pcts[1])),
            fmt(float(pcts[2])),
            fmt(float(pcts[3])),
            fmt(float(pcts[4])),
            fmt(float(pcts[5])),
            fmt(float(pcts[6])),
            fmt(first5_med),
            fmt(first5_mean),
            fmt(float(all_spreads.mean())),
        ])
    return rows


def write_pair_csv(pair: str, rows: list[list[str]]) -> tuple[Path, str]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{pair}_hourly.csv"
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow([
        "hour_utc", "weekday", "hour", "session",
        "n_ticks", "n_ticks_first5min",
        "p10_spread_pips", "p25_spread_pips", "p50_spread_pips",
        "p75_spread_pips", "p90_spread_pips", "p95_spread_pips",
        "p99_spread_pips",
        "first5min_median_spread_pips", "first5min_mean_spread_pips",
        "mean_spread_pips",
    ])
    for r in rows:
        w.writerow(r)
    body = buf.getvalue().encode("utf-8")
    out_path.write_bytes(body)
    return out_path, hashlib.sha256(body).hexdigest()


def process_one_pair(pair: str) -> tuple[str, int, int, str, float]:
    zips = iter_pair_zips(pair)
    if not zips:
        return (pair, 0, 0, "", 0.0)
    pt0 = time.time()
    bars = aggregate_pair(pair)
    rows = finalise_pair_rows(pair, bars)
    _, digest = write_pair_csv(pair, rows)
    return (pair, len(rows), len(zips), digest, time.time() - pt0)


def _workers_from_env() -> int:
    raw = os.environ.get("SPREAD_VALIDATION_AGG_WORKERS")
    if raw:
        try:
            n = int(raw)
            if n > 0:
                return n
        except ValueError:
            pass
    return DEFAULT_WORKERS


def main() -> int:
    if not CACHE_DIR.exists():
        print(f"BLOCKER: cache root not found: {CACHE_DIR}", flush=True)
        return 1

    workers = _workers_from_env()
    print(f"Processing {len(PAIRS)} pairs from {CACHE_DIR}", flush=True)
    print(f"Output dir: {OUTPUT_DIR}", flush=True)
    print(f"Workers: {workers}", flush=True)
    t0 = time.time()

    summary: list[tuple[str, int, int, str]] = []
    if workers <= 1:
        for pair in PAIRS:
            pair_out, n_rows, n_zips, digest, dt = process_one_pair(pair)
            if n_rows == 0 and n_zips == 0:
                print(f"  {pair_out}: no zips cached, skipping", flush=True)
            else:
                print(
                    f"  {pair_out}: {n_rows:>6d} hour rows from {n_zips:>2d} zips  "
                    f"({dt:.1f}s)  sha256={digest[:16]}...",
                    flush=True,
                )
            summary.append((pair_out, n_rows, n_zips, digest))
    else:
        with mp.Pool(processes=workers) as pool:
            for pair_out, n_rows, n_zips, digest, dt in pool.imap_unordered(
                process_one_pair, PAIRS
            ):
                if n_rows == 0 and n_zips == 0:
                    print(f"  {pair_out}: no zips cached, skipping", flush=True)
                else:
                    print(
                        f"  {pair_out}: {n_rows:>6d} hour rows from {n_zips:>2d} zips  "
                        f"({dt:.1f}s)  sha256={digest[:16]}...",
                        flush=True,
                    )
                summary.append((pair_out, n_rows, n_zips, digest))

    summary.sort(key=lambda s: s[0])

    manifest_path = OUTPUT_DIR / "_manifest.csv"
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["pair", "n_rows", "n_zips", "sha256"])
    for s in summary:
        w.writerow(s)
    manifest_path.write_bytes(buf.getvalue().encode("utf-8"))

    elapsed = time.time() - t0
    print(f"DONE  elapsed={elapsed:.0f}s ({elapsed/60:.1f}min)  manifest={manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
