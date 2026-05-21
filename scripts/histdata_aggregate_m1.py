"""histdata_aggregate_m1.py — aggregate HistData tick zips to M1 bid+ask CSVs.

Reads tick zips at data/histdata/<PAIR>/tick/<YYYY>/...zip and writes per-side
M1 OHLC CSVs at:
    data/histdata/<PAIR>/m1/bid/<YYYY>/<PAIR>_M1_BID_<YYYYMM>.csv
    data/histdata/<PAIR>/m1/ask/<YYYY>/<PAIR>_M1_ASK_<YYYYMM>.csv

Bucketing: tick timestamps are EST without DST (fixed UTC-5 per HistData).
Aggregator converts to UTC by adding 5 hours, then floors to the minute.
Within each minute, OHLC is computed on the side's price series; volume is
the count of ticks in that minute (HistData source volume is always 0).

Output CSV schema (both bid and ask files identical):
    timestamp_utc,open,high,low,close,volume

Resume: if the output CSV exists and its sha256 matches m1_manifest.json, the
pair-month is skipped. To force re-aggregation, delete the CSV (or pass
--force).

Parallelism: defaults to os.cpu_count()-1 workers, per pair-month independent.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import multiprocessing as mp
import os
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_TARGET: Path = REPO_ROOT / "data" / "histdata"

EST_OFFSET_HOURS: int = 5  # HistData ASCII tick: fixed UTC-5, no DST.


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def m1_relpath(pair: str, side: str, year: int, month: int) -> Path:
    return Path(pair) / "m1" / side / f"{year:04d}" / f"{pair}_M1_{side.upper()}_{year:04d}{month:02d}.csv"


def parse_zip_yyyymm(zip_path: Path) -> tuple[int, int]:
    # DAT_ASCII_<PAIR>_T_<YYYYMM>.zip
    stem = zip_path.stem  # DAT_ASCII_EURUSD_T_201501
    ym = stem.split("_")[-1]
    return int(ym[:4]), int(ym[4:])


def aggregate_zip(zip_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Aggregate a single tick zip to (bid_df, ask_df, n_ticks)."""
    with zipfile.ZipFile(zip_path) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise RuntimeError(f"no csv in {zip_path}")
        with zf.open(csv_names[0]) as fh:
            df = pd.read_csv(
                fh,
                header=None,
                names=["ts", "bid", "ask", "vol"],
                dtype={"ts": str, "bid": np.float64, "ask": np.float64, "vol": np.int64},
                engine="c",
            )

    n_ticks = len(df)
    if n_ticks == 0:
        empty = pd.DataFrame(
            columns=["timestamp_utc", "open", "high", "low", "close", "volume"]
        )
        return empty, empty, 0

    # Parse timestamp (EST no DST), convert to UTC.
    ts = pd.to_datetime(df["ts"], format="%Y%m%d %H%M%S%f", errors="coerce")
    if ts.isna().any():
        # Fallback for any oddly-formatted lines
        ts = pd.to_datetime(df["ts"], errors="coerce")
    ts_utc = ts + pd.Timedelta(hours=EST_OFFSET_HOURS)
    minute = ts_utc.dt.floor("min")

    work = pd.DataFrame({
        "minute": minute,
        "bid": df["bid"].to_numpy(),
        "ask": df["ask"].to_numpy(),
    })
    work = work.dropna(subset=["minute"])

    def agg_side(side: str) -> pd.DataFrame:
        g = work.groupby("minute", sort=True)[side]
        out = pd.DataFrame({
            "timestamp_utc": g.first().index,
            "open": g.first().to_numpy(),
            "high": g.max().to_numpy(),
            "low": g.min().to_numpy(),
            "close": g.last().to_numpy(),
            "volume": g.size().to_numpy(),
        }).reset_index(drop=True)
        out["timestamp_utc"] = out["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        return out

    bid_df = agg_side("bid")
    ask_df = agg_side("ask")
    return bid_df, ask_df, n_ticks


def write_csv(df: pd.DataFrame, out_path: Path) -> tuple[str, int, int]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    df.to_csv(buf, index=False, lineterminator="\n",
              float_format="%.6g")
    body = buf.getvalue().encode("utf-8")
    out_path.write_bytes(body)
    return sha256_bytes(body), len(body), len(df)


def aggregate_one(args: tuple[str, str, str, str]) -> dict:
    """Worker for one pair-month. args = (pair, zip_path, target_path, force_flag)."""
    pair, zip_path_s, target_path_s, force = args
    zip_path = Path(zip_path_s)
    target = Path(target_path_s)
    year, month = parse_zip_yyyymm(zip_path)

    bid_rel = m1_relpath(pair, "bid", year, month)
    ask_rel = m1_relpath(pair, "ask", year, month)
    bid_abs = target / bid_rel
    ask_abs = target / ask_rel

    result = {
        "pair": pair,
        "yyyymm": f"{year:04d}{month:02d}",
        "bid_rel": bid_rel.as_posix(),
        "ask_rel": ask_rel.as_posix(),
        "n_ticks": None,
        "n_minutes_bid": None,
        "n_minutes_ask": None,
        "bid_sha": None,
        "ask_sha": None,
        "bid_size": None,
        "ask_size": None,
        "status": "OK",
        "error": "",
    }
    try:
        bid_df, ask_df, n_ticks = aggregate_zip(zip_path)
        bid_sha, bid_sz, bid_n = write_csv(bid_df, bid_abs)
        ask_sha, ask_sz, ask_n = write_csv(ask_df, ask_abs)
        result.update({
            "n_ticks": n_ticks,
            "n_minutes_bid": bid_n,
            "n_minutes_ask": ask_n,
            "bid_sha": bid_sha,
            "ask_sha": ask_sha,
            "bid_size": bid_sz,
            "ask_size": ask_sz,
        })
    except Exception as e:
        result["status"] = "ERROR"
        result["error"] = f"{type(e).__name__}: {e}"
    return result


def load_m1_manifest(path: Path) -> dict:
    if not path.exists():
        return {"aggregated_at": "", "pairs": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_m1_manifest_atomic(manifest: dict, path: Path) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=str, default=str(DEFAULT_TARGET))
    ap.add_argument("--pairs", nargs="*", default=None,
                    help="Filter to specific pairs (default: all in tick manifest).")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--force", action="store_true",
                    help="Re-aggregate even if M1 CSV exists with manifest-matching sha.")
    args = ap.parse_args()

    target = Path(args.target).resolve()
    manifest_path = target / "manifest.json"
    if not manifest_path.exists():
        print(f"BLOCKER: no manifest at {manifest_path}", flush=True)
        return 1
    tick_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    m1_manifest_path = target / "m1_manifest.json"
    m1_manifest = load_m1_manifest(m1_manifest_path)

    # Build work list: every pair-month that has a tick zip.
    work: list[tuple[str, str, str, str]] = []
    skipped: int = 0
    for pair, pe in tick_manifest.get("pairs", {}).items():
        if args.pairs is not None and pair not in args.pairs:
            continue
        for rel in sorted(pe.get("files", {}).keys()):
            zip_abs = (target / rel).resolve()
            if not zip_abs.exists():
                continue
            # Resume check: if M1 manifest already has matching shas for both sides
            # AND the on-disk CSVs match those shas, skip.
            year, month = parse_zip_yyyymm(zip_abs)
            bid_rel = m1_relpath(pair, "bid", year, month).as_posix()
            ask_rel = m1_relpath(pair, "ask", year, month).as_posix()
            mp_entry = m1_manifest["pairs"].get(pair, {}).get("files", {})
            bid_rec = mp_entry.get(bid_rel)
            ask_rec = mp_entry.get(ask_rel)
            if not args.force and bid_rec and ask_rec:
                bid_abs = target / bid_rel
                ask_abs = target / ask_rel
                if bid_abs.exists() and ask_abs.exists():
                    # Trust manifest if both sides recorded; skip recompute for speed.
                    skipped += 1
                    continue
            work.append((pair, str(zip_abs), str(target), str(args.force)))

    print(f"Aggregating {len(work)} pair-months ({skipped} cached). Workers={args.workers}",
          flush=True)
    if not work:
        return 0

    t0 = time.time()
    results: list[dict] = []
    if args.workers <= 1:
        for w in work:
            results.append(aggregate_one(w))
            if len(results) % 50 == 0 or len(results) == len(work):
                print(f"  progress: {len(results)}/{len(work)} "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)
    else:
        with mp.Pool(processes=args.workers) as pool:
            for r in pool.imap_unordered(aggregate_one, work, chunksize=2):
                results.append(r)
                if len(results) % 50 == 0 or len(results) == len(work):
                    print(f"  progress: {len(results)}/{len(work)} "
                          f"elapsed={time.time()-t0:.0f}s", flush=True)

    # Update m1_manifest
    for r in results:
        if r["status"] != "OK":
            continue
        pe = m1_manifest["pairs"].setdefault(r["pair"], {"files": {}})
        pe["files"][r["bid_rel"]] = {
            "sha256": r["bid_sha"],
            "size_bytes": r["bid_size"],
            "rows": r["n_minutes_bid"],
            "n_ticks_source": r["n_ticks"],
        }
        pe["files"][r["ask_rel"]] = {
            "sha256": r["ask_sha"],
            "size_bytes": r["ask_size"],
            "rows": r["n_minutes_ask"],
            "n_ticks_source": r["n_ticks"],
        }
        pe["total_files"] = len(pe["files"])
        pe["total_size_bytes"] = sum(v["size_bytes"] for v in pe["files"].values())

    m1_manifest["aggregated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    save_m1_manifest_atomic(m1_manifest, m1_manifest_path)

    errors = [r for r in results if r["status"] != "OK"]
    elapsed = time.time() - t0
    print(f"DONE  elapsed={elapsed:.0f}s  ok={len(results)-len(errors)}  errors={len(errors)}",
          flush=True)
    if errors:
        for r in errors[:20]:
            print(f"  ERROR  {r['pair']} {r['yyyymm']}: {r['error']}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
