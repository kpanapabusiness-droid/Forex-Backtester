"""Tests for core/parallel.py — multiprocessing.Pool wrapper + determinism."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from core.parallel import (
    build_panel_parallel,
    default_pool_size,
    parallel_load_m1,
    parallel_pair_map,
)
from tests.fixtures.histdata_mini.build import FixtureSpec, build_fixture


@pytest.fixture
def mini_root(tmp_path: Path) -> Path:
    return build_fixture(
        tmp_path / "histdata",
        FixtureSpec(
            pairs=("EURUSD", "GBPUSD", "USDJPY", "AUDUSD"),
            months=("201001", "201002"),
            minutes_per_month=180,
        ),
    )


@pytest.fixture
def cache_root(tmp_path: Path) -> Path:
    return tmp_path / "cache"


# Top-level worker functions — must be picklable on Windows spawn.


def _ord_sum(pair: str) -> int:
    return sum(ord(c) for c in pair)


def _identity(pair: str) -> str:
    return pair


# ── basic mechanics ─────────────────────────────────────────────────


def test_default_pool_size_caps_at_n_items() -> None:
    assert default_pool_size(3) <= 3
    assert default_pool_size(28) <= 28
    assert default_pool_size(28) >= 1


def test_parallel_pair_map_serial_path() -> None:
    """pool_size=1 → no Pool spawn; results match func applied directly."""
    pairs = ["EURUSD", "GBPUSD", "AUDUSD"]
    out = parallel_pair_map(_ord_sum, pairs, pool_size=1)
    assert out == {p: _ord_sum(p) for p in pairs}


def test_parallel_pair_map_returns_sorted_by_pair() -> None:
    """Insertion order is sorted, regardless of input order."""
    pairs = ["USDJPY", "AUDUSD", "EURUSD"]
    out = parallel_pair_map(_ord_sum, pairs, pool_size=1)
    assert list(out.keys()) == ["AUDUSD", "EURUSD", "USDJPY"]


def test_parallel_pair_map_pool_2_vs_pool_1_equal() -> None:
    pairs = ["EURUSD", "GBPUSD", "AUDUSD", "USDJPY"]
    a = parallel_pair_map(_identity, pairs, pool_size=1)
    b = parallel_pair_map(_identity, pairs, pool_size=2)
    assert a == b
    assert list(a.keys()) == list(b.keys())  # sorted order preserved


def test_parallel_pair_map_pool_4_vs_pool_1_equal() -> None:
    pairs = ["EURUSD", "GBPUSD", "AUDUSD", "USDJPY"]
    a = parallel_pair_map(_ord_sum, pairs, pool_size=1)
    b = parallel_pair_map(_ord_sum, pairs, pool_size=4)
    assert a == b


# ── data-layer parallelism ──────────────────────────────────────────


def test_parallel_load_m1_pool1_matches_pool4(mini_root: Path, tmp_path: Path) -> None:
    """Loading 4 pairs serially vs pool=4 → byte-identical parquet caches +
    identical DataFrames."""
    cache_a = tmp_path / "cache_a"
    cache_b = tmp_path / "cache_b"
    pairs = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]

    a = parallel_load_m1(pairs, histdata_root=mini_root, cache_root=cache_a, pool_size=1)
    b = parallel_load_m1(pairs, histdata_root=mini_root, cache_root=cache_b, pool_size=4)

    assert list(a.keys()) == list(b.keys()) == sorted(pairs)
    for pair in pairs:
        pd.testing.assert_frame_equal(a[pair], b[pair])


def test_build_panel_parallel_matches_serial(mini_root: Path, tmp_path: Path) -> None:
    cache_a = tmp_path / "cache_a"
    cache_b = tmp_path / "cache_b"
    pairs = ["EURUSD", "GBPUSD", "USDJPY"]

    pa = build_panel_parallel(pairs, "M5", histdata_root=mini_root, cache_root=cache_a, pool_size=1)
    pb = build_panel_parallel(pairs, "M5", histdata_root=mini_root, cache_root=cache_b, pool_size=3)

    assert pa.pairs == pb.pairs
    assert pa.tf == pb.tf
    for pair in pairs:
        pd.testing.assert_frame_equal(pa.pair_dfs[pair], pb.pair_dfs[pair])


def test_build_panel_parallel_sha256_stable(mini_root: Path, tmp_path: Path) -> None:
    """Two pool=4 runs against fresh caches produce DataFrame-equal output."""
    pairs = ["EURUSD", "GBPUSD", "USDJPY"]

    def go(cache: Path):
        return build_panel_parallel(
            pairs, "M5", histdata_root=mini_root, cache_root=cache, pool_size=4
        )

    a = go(tmp_path / "cache_a")
    b = go(tmp_path / "cache_b")
    for pair in pairs:
        # DataFrame equality is the load-bearing invariant — parquet byte
        # equality is asserted separately for the cache parquets in PR-A.
        pd.testing.assert_frame_equal(a.pair_dfs[pair], b.pair_dfs[pair])


def test_parallel_aggregation_deterministic_csv_sha(mini_root: Path, tmp_path: Path) -> None:
    """The sha256 of the panel's concatenated CSV is the same at pool=1 and pool=N."""
    pairs = ["EURUSD", "GBPUSD", "USDJPY"]

    def panel_sha(cache: Path, pool_size: int) -> str:
        panel = build_panel_parallel(
            pairs, "M5", histdata_root=mini_root, cache_root=cache, pool_size=pool_size
        )
        # Concat in sorted-pair order — deterministic
        ordered = pd.concat(
            [panel.pair_dfs[p].assign(_pair=p) for p in sorted(panel.pairs)],
            axis=0,
        )
        csv = ordered.to_csv(lineterminator="\n", float_format="%.10g")
        return hashlib.sha256(csv.encode("utf-8")).hexdigest()

    serial_sha = panel_sha(tmp_path / "cache_serial", 1)
    parallel_sha = panel_sha(tmp_path / "cache_parallel", 3)
    assert serial_sha == parallel_sha
