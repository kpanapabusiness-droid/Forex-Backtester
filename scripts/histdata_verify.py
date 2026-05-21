"""histdata_verify.py — post-pass full integrity verification of HistData tick zips.

Runs after histdata_download.py completes. For every file on disk:
    * Recomputes sha256 of the raw zip bytes; compares to manifest.json.
    * Opens the zip, extracts the CSV, parses every row.
    * Verifies bid <= ask on every row (any violation flags the file).
    * Computes full row count (writes back to manifest if previously 0).
    * Flags row-count anomalies (>2σ from per-pair mean across observed
      months) to row_count_anomalies.md.

Failures (sha mismatch, parse error, bid>ask) are recorded in
integrity_failures.md. The downloader picks these up on next run by quarantine
or re-download (sha-mismatched files are unlinked so they re-download).

Parallelism: defaults to os.cpu_count() workers. Per-file work is independent.

Read-only on the data tree EXCEPT for the manifest write-back of row counts
and removal of sha-mismatched files (for re-download).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_TARGET: Path = REPO_ROOT / "data" / "histdata"

MEDIAN_SPREAD_PIPS_LO: float = 0.05
MEDIAN_SPREAD_PIPS_HI: float = 50.0


def is_jpy(pair: str) -> bool:
    return pair.endswith("JPY")


def pip_multiplier(pair: str) -> float:
    return 100.0 if is_jpy(pair) else 10000.0


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_manifest(path: Path) -> dict:
    if not path.exists():
        return {"downloaded_at": "", "pairs": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest_atomic(manifest: dict, path: Path) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def pair_from_relpath(rel: str) -> str:
    # Path layout: <PAIR>/tick/<YYYY>/<file>
    return rel.split("/", 1)[0]


def verify_one(args: tuple[str, str, dict, str]) -> dict:
    """Worker function. args = (rel_path, abs_path, manifest_entry, pair).

    Returns a dict of findings. No I/O side effects on the data tree from
    here; the parent collates and applies.
    """
    rel, abs_path_s, entry, pair = args
    abs_path = Path(abs_path_s)
    result = {
        "rel": rel,
        "pair": pair,
        "sha_on_disk": None,
        "sha_manifest": entry.get("sha256"),
        "sha_match": None,
        "rows": None,
        "median_spread_pips": None,
        "bid_gt_ask_rows": 0,
        "parse_error": None,
        "size_bytes": None,
    }
    if not abs_path.exists():
        result["parse_error"] = "missing on disk"
        return result
    try:
        size = abs_path.stat().st_size
        result["size_bytes"] = size
        sha = sha256_file(abs_path)
        result["sha_on_disk"] = sha
        result["sha_match"] = (sha == entry.get("sha256"))
        # Parse full CSV
        with zipfile.ZipFile(abs_path) as zf:
            csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                result["parse_error"] = "no csv in zip"
                return result
            with zf.open(csv_names[0]) as fh:
                df = pd.read_csv(
                    fh,
                    header=None,
                    names=["ts", "bid", "ask", "vol"],
                    dtype={"ts": str, "bid": np.float64, "ask": np.float64,
                           "vol": np.int64},
                    engine="c",
                )
        result["rows"] = int(len(df))
        if df[["bid", "ask"]].isna().any().any():
            result["parse_error"] = "bid or ask NaN"
            return result
        bad = (df["bid"] > df["ask"])
        result["bid_gt_ask_rows"] = int(bad.sum())
        pip_m = pip_multiplier(pair)
        spread = (df["ask"].to_numpy() - df["bid"].to_numpy()) * pip_m
        result["median_spread_pips"] = float(np.median(spread))
    except zipfile.BadZipFile:
        result["parse_error"] = "bad zip"
    except Exception as e:
        result["parse_error"] = f"{type(e).__name__}: {e}"
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=str, default=str(DEFAULT_TARGET))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--fix", action="store_true",
                    help="Delete files with sha mismatch so the downloader re-fetches them on next run.")
    ap.add_argument("--update-manifest", action="store_true",
                    help="Write back full row counts into manifest.json (only filled rows are added).")
    args = ap.parse_args()

    target = Path(args.target).resolve()
    manifest_path = target / "manifest.json"
    if not manifest_path.exists():
        print(f"BLOCKER: no manifest at {manifest_path}", flush=True)
        return 1
    manifest = load_manifest(manifest_path)

    # Build verification work list: every file in every pair's manifest entry.
    work: list[tuple[str, str, dict, str]] = []
    for pair, pe in manifest["pairs"].items():
        for rel, entry in pe.get("files", {}).items():
            abs_path = (target / rel).resolve()
            work.append((rel, str(abs_path), entry, pair))

    print(f"Verifying {len(work)} files across {len(manifest['pairs'])} pairs "
          f"with {args.workers} workers", flush=True)
    if not work:
        print("nothing to verify", flush=True)
        return 0

    t0 = time.time()
    results: list[dict] = []
    if args.workers <= 1:
        for w in work:
            results.append(verify_one(w))
            if len(results) % 100 == 0:
                print(f"  progress: {len(results)}/{len(work)}", flush=True)
    else:
        with mp.Pool(processes=args.workers) as pool:
            for r in pool.imap_unordered(verify_one, work, chunksize=4):
                results.append(r)
                if len(results) % 100 == 0:
                    print(f"  progress: {len(results)}/{len(work)}", flush=True)

    # Categorize
    failures: list[dict] = []
    for r in results:
        if r["parse_error"] is not None or r["sha_match"] is False or r["bid_gt_ask_rows"] > 0:
            failures.append(r)

    # Row count anomalies: >2σ from per-pair mean
    rows_by_pair: dict[str, list[tuple[str, int]]] = {}
    for r in results:
        if r["rows"] is None or r["parse_error"]:
            continue
        rows_by_pair.setdefault(r["pair"], []).append((r["rel"], r["rows"]))
    anomalies: list[tuple[str, str, int, float]] = []  # rel, pair, rows, z-score
    for pair, items in rows_by_pair.items():
        if len(items) < 6:
            continue
        arr = np.array([n for _, n in items], dtype=float)
        mu = float(arr.mean())
        sd = float(arr.std(ddof=0))
        if sd == 0:
            continue
        for rel, n in items:
            z = (n - mu) / sd
            if abs(z) > 2.0:
                anomalies.append((rel, pair, n, z))

    # Write reports
    fail_path = target / "integrity_failures.md"
    if failures:
        lines = ["# Integrity failures", ""]
        lines.append("Files that failed sha256 match, full parse, or per-row bid<=ask.")
        lines.append("")
        lines.append("| pair | rel | sha_match | parse_error | bid>ask_rows | rows |")
        lines.append("|------|-----|-----------|-------------|--------------|------|")
        for r in sorted(failures, key=lambda x: x["rel"]):
            lines.append(
                f"| {r['pair']} | {r['rel']} | {r['sha_match']} | "
                f"{r['parse_error'] or ''} | {r['bid_gt_ask_rows']} | {r['rows']} |"
            )
        fail_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Wrote {len(failures)} failures to {fail_path}", flush=True)
    else:
        if fail_path.exists():
            fail_path.unlink()
        print("No integrity failures.", flush=True)

    anom_path = target / "row_count_anomalies.md"
    if anomalies:
        lines = ["# Row count anomalies", ""]
        lines.append("Files whose row count is >2σ from the per-pair mean.")
        lines.append("Likely benign — quiet months, holidays, illiquid crosses — but worth a glance.")
        lines.append("")
        lines.append("| pair | rel | rows | z-score |")
        lines.append("|------|-----|------|---------|")
        for rel, pair, n, z in sorted(anomalies, key=lambda x: (x[1], x[0])):
            lines.append(f"| {pair} | {rel} | {n} | {z:+.2f} |")
        anom_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Wrote {len(anomalies)} anomalies to {anom_path}", flush=True)
    else:
        if anom_path.exists():
            anom_path.unlink()
        print("No row-count anomalies.", flush=True)

    # Manifest updates (row counts) — only if --update-manifest
    if args.update_manifest:
        n_updated = 0
        for r in results:
            if r["rows"] is None or r["parse_error"]:
                continue
            entry = manifest["pairs"][r["pair"]]["files"].get(r["rel"])
            if entry is None:
                continue
            if entry.get("rows", 0) != r["rows"]:
                entry["rows"] = r["rows"]
                n_updated += 1
        if n_updated:
            save_manifest_atomic(manifest, manifest_path)
            print(f"Updated {n_updated} row counts in manifest.", flush=True)

    # Fix: delete sha-mismatched files so the downloader re-fetches
    if args.fix:
        n_deleted = 0
        for r in failures:
            if r["sha_match"] is False:
                p = target / r["rel"]
                if p.exists():
                    p.unlink()
                    n_deleted += 1
        print(f"Deleted {n_deleted} sha-mismatched files (re-fetch on next download run).", flush=True)

    elapsed = time.time() - t0
    print(f"DONE  elapsed={elapsed:.0f}s  failures={len(failures)}  anomalies={len(anomalies)}",
          flush=True)
    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(main())
