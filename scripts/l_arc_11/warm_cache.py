"""Warm M1/H4/D1/W1 cache for all 28 pairs by launching a subprocess per
pair-TF. Avoids memory accumulation that triggers `Could not open Parquet
input source '<Buffer>'` on later pairs in the v3 engine.

Usage:
    py scripts/l_arc_11/warm_cache.py --tfs H4 D1 W1
"""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
import time
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _log(msg: str) -> None:
    ts = dt.datetime.now().strftime("%H:%M:%S")
    print(f"[warm_cache {ts}] {msg}", flush=True)


def warm_one(pair: str, tf: str, histdata_root: str, cache_root: str) -> tuple[int, float, str]:
    """Run a subprocess to aggregate one pair+TF; return (exit_code, seconds, stderr_tail)."""
    code = (
        f"import sys; sys.path.insert(0, r'{_REPO_ROOT}');"
        f"from core.data.aggregator import aggregate;"
        f"df = aggregate('{pair}', '{tf}', histdata_root=r'{histdata_root}', cache_root=r'{cache_root}');"
        f"print(f'{pair} {tf} rows={{len(df)}}')"
    )
    t0 = time.time()
    p = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=600
    )
    elapsed = time.time() - t0
    stderr_tail = (p.stderr or "").splitlines()[-5:]
    return p.returncode, elapsed, "\n".join(stderr_tail)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="configs/wfo_l_arc_11.yaml")
    ap.add_argument("--tfs", nargs="+", default=["H4", "D1", "W1"])
    args = ap.parse_args(argv)

    cfg = yaml.safe_load((_REPO_ROOT / args.config).read_text(encoding="utf-8"))
    pairs = sorted(list(cfg["pairs"]))
    histdata_root = cfg["data"]["histdata_root"]
    cache_root = cfg["data"]["cache_root"]

    failures: list[tuple[str, str, str]] = []
    for tf in args.tfs:
        for i, pair in enumerate(pairs, 1):
            rc, elapsed, stderr = warm_one(pair, tf, histdata_root, cache_root)
            status = "OK" if rc == 0 else f"FAIL rc={rc}"
            _log(f"{tf} {pair} ({i}/{len(pairs)}): {status} in {elapsed:.1f}s")
            if rc != 0:
                _log(f"  stderr: {stderr}")
                failures.append((tf, pair, stderr))
    if failures:
        _log(f"{len(failures)} failures")
        for tf, pair, st in failures:
            _log(f"  {tf} {pair}: {st}")
        return 1
    _log("All caches warm.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
