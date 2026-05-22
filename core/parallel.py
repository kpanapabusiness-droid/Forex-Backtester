"""Per-pair parallelism via ``multiprocessing.Pool``.

Per CC_06 Tasks 9c + 9d:

  - The 28 pairs are independent at Step 1 plumbing and at most feature
    computations. Run them in parallel using ``multiprocessing.Pool``
    (NOT threading — Python's GIL blocks numpy/pandas).
  - Each worker is single-threaded internally (``n_jobs=1``,
    ``random_state=42``) so per-row results are reproducible.
  - Aggregation back into a single panel is deterministic — results
    are returned sorted by pair name regardless of completion order.
  - ``pool_size`` defaults to ``min(28, cpu_count() - 1)``; callers
    can pass ``pool_size=1`` to force serial execution (mostly for
    tests asserting parallel = serial output equality).

Determinism contract: ``parallel_pair_map(func, pairs, pool_size=N)``
returns byte-identical output (after sorted aggregation) for any
``N ≥ 1``. Asserted by ``tests/test_parallel.py``.
"""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import pandas as pd

from core.data.aggregator import aggregate
from core.data.histdata_loader import load_m1
from core.sim.panel import Panel

T = TypeVar("T")


def default_pool_size(n_items: int | None = None) -> int:
    """Return the default pool size.

    ``min(n_items or 28, max(1, cpu_count() - 1))`` — capped at the
    number of pairs being processed (no point spawning more workers
    than items) and leaves one CPU free for the parent / OS.
    """
    n = mp.cpu_count() or 1
    cap = n_items if n_items is not None else 28
    return max(1, min(cap, n - 1))


def parallel_pair_map(
    func: Callable[[str], T],
    pairs: list[str] | tuple[str, ...],
    pool_size: int | None = None,
) -> dict[str, T]:
    """Apply ``func(pair)`` to each pair in parallel via ``multiprocessing.Pool``.

    Returns ``dict[pair, result]`` keyed by pair name in sorted order.
    The dict insertion order is deterministic across runs, regardless
    of worker completion order — sorted by pair name.

    ``pool_size=1`` runs serially in the parent process (no Pool spawn),
    which is the path used by determinism tests asserting parallel ==
    serial output equality.

    ``func`` MUST be a top-level (picklable) callable. Lambdas and
    closures don't survive the spawn/fork boundary on Windows; use
    ``functools.partial`` for kwargs.
    """
    sorted_pairs = sorted(pairs)
    if pool_size is None:
        pool_size = default_pool_size(len(sorted_pairs))

    if pool_size <= 1:
        results = [func(p) for p in sorted_pairs]
    else:
        with mp.Pool(processes=pool_size) as pool:
            # imap preserves input order; pair k's result comes back at
            # position k regardless of completion order. We pre-sorted
            # ``sorted_pairs`` so the iteration order is deterministic.
            results = list(pool.imap(func, sorted_pairs, chunksize=1))

    return {p: r for p, r in zip(sorted_pairs, results)}


# ── convenience: per-pair data-layer parallelism ────────────────────────


def _load_m1_for(args: tuple[str, Path, Path]) -> pd.DataFrame:
    """Worker-side: load M1 for one pair. Top-level → picklable on spawn."""
    pair, histdata_root, cache_root = args
    return load_m1(pair, histdata_root=histdata_root, cache_root=cache_root)


def _aggregate_for(args: tuple[str, str, Path, Path]) -> pd.DataFrame:
    """Worker-side: aggregate one pair to one TF. Picklable."""
    pair, tf, histdata_root, cache_root = args
    return aggregate(pair, tf, histdata_root=histdata_root, cache_root=cache_root)


def parallel_load_m1(
    pairs: list[str],
    histdata_root: Path | str = "data/histdata",
    cache_root: Path | str = "data/cache",
    pool_size: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Load M1 for many pairs in parallel.

    Each worker runs ``load_m1`` standalone — first call writes the
    parquet cache; subsequent calls hit the cache. The cache write is
    per-pair so there's no cross-worker contention.
    """
    histdata_root = Path(histdata_root)
    cache_root = Path(cache_root)
    args = [(p, histdata_root, cache_root) for p in sorted(pairs)]
    if pool_size is None:
        pool_size = default_pool_size(len(args))
    if pool_size <= 1:
        results = [_load_m1_for(a) for a in args]
    else:
        with mp.Pool(processes=pool_size) as pool:
            results = list(pool.imap(_load_m1_for, args, chunksize=1))
    return {p: r for (p, _, _), r in zip(args, results)}


def build_panel_parallel(
    pairs: list[str],
    tf: str,
    histdata_root: Path | str = "data/histdata",
    cache_root: Path | str = "data/cache",
    pool_size: int | None = None,
) -> Panel:
    """Build a multi-pair ``Panel`` at ``tf`` by aggregating each pair in parallel.

    The first call warms the per-TF parquet cache; subsequent calls
    against the same ``cache_root`` are fast even at ``pool_size=1``.
    Aggregation order is deterministic (sorted by pair name).
    """
    histdata_root = Path(histdata_root)
    cache_root = Path(cache_root)
    args = [(p, tf, histdata_root, cache_root) for p in sorted(pairs)]
    if pool_size is None:
        pool_size = default_pool_size(len(args))
    if pool_size <= 1:
        results = [_aggregate_for(a) for a in args]
    else:
        with mp.Pool(processes=pool_size) as pool:
            results = list(pool.imap(_aggregate_for, args, chunksize=1))

    pair_dfs = {p: df for (p, _, _, _), df in zip(args, results)}
    return Panel.from_frames(pair_dfs, tf=tf)
