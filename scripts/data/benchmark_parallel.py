"""Benchmark serial vs parallel 28-pair Step-1 cache build.

Runs ``build_panel_parallel`` for all 28 pairs at three pool sizes
(serial, pool=8, pool=cpu-1) and prints wall-clock timings. Used to
populate PR-D's headline speedup numbers.

Each run uses a separate ``--cache-root`` so each measures a cold-cache
build (the expensive first-load path). Pass ``--reuse-cache`` to measure
cache-hit timings instead.

Usage::

    py -m scripts.data.benchmark_parallel \\
        --config configs/data_v3.yaml \\
        --histdata-root "C:/Users/panap/Documents/Forex-Backtester/data/histdata" \\
        --tf M5
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import shutil
import time
from pathlib import Path

import yaml

from core.parallel import build_panel_parallel, default_pool_size


def _bench_one(
    pairs: list[str], tf: str, histdata_root: Path, cache_root: Path, pool_size: int
) -> float:
    if cache_root.exists():
        shutil.rmtree(cache_root)
    t0 = time.perf_counter()
    build_panel_parallel(
        pairs, tf, histdata_root=histdata_root, cache_root=cache_root, pool_size=pool_size
    )
    return time.perf_counter() - t0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/data_v3.yaml"))
    parser.add_argument("--histdata-root", type=Path, default=None)
    parser.add_argument("--cache-base", type=Path, default=None,
                        help="Base dir for per-bench cache roots; default is tmp")
    parser.add_argument("--tf", type=str, default="M5")
    parser.add_argument("--pool-sizes", type=int, nargs="+", default=None,
                        help="Pool sizes to benchmark (default: 1, 8, cpu-1)")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    pairs = sorted(cfg["pairs"])
    histdata_root = args.histdata_root or Path(cfg["data"]["histdata_root"])

    n_cpu_minus_one = max(1, mp.cpu_count() - 1)
    pool_sizes = args.pool_sizes or sorted(set([1, 8, n_cpu_minus_one]))

    base = args.cache_base or Path(
        f"C:/Users/panap/AppData/Local/Temp/pr_d_bench_{args.tf}"
    )
    base.mkdir(parents=True, exist_ok=True)

    print(f"Benchmark: 28-pair {args.tf} cold-cache build")
    print(f"  CPUs: {mp.cpu_count()}; default_pool_size(28)={default_pool_size(28)}")
    print(f"  HistData: {histdata_root}")
    print(f"  Pairs: {len(pairs)}")
    print()
    print(f"{'pool_size':>10s}  {'elapsed_sec':>12s}  {'speedup':>10s}")
    print("-" * 40)

    serial = None
    for ps in pool_sizes:
        cache_root = base / f"pool_{ps}"
        elapsed = _bench_one(pairs, args.tf, histdata_root, cache_root, ps)
        if serial is None:
            serial = elapsed
            speedup = "1.00x (ref)"
        else:
            speedup = f"{serial / elapsed:.2f}x"
        print(f"{ps:>10d}  {elapsed:>12.2f}  {speedup:>10s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
