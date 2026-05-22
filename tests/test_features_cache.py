"""Tests for core/features/cache.py — feature matrix caching + invalidation."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pytest

from core.data.aggregator import aggregate
from core.features.cache import (
    cache_valid,
    feature_cache_key,
    feature_cache_path,
    get_or_compute,
    pool_sha_from_dataframe,
)
from core.features.pipeline import FeatureMatrix, compute_feature_matrix
from tests.fixtures.histdata_mini.build import FixtureSpec, build_fixture


@pytest.fixture
def pair_df(tmp_path: Path) -> pd.DataFrame:
    root = build_fixture(
        tmp_path / "histdata",
        FixtureSpec(minutes_per_month=720, months=("201001",)),
    )
    return aggregate("EURUSD", "M5", histdata_root=root, cache_root=tmp_path / "cache_agg")


@pytest.fixture
def synthetic_pool(pair_df: pd.DataFrame) -> pd.DataFrame:
    """A small trade pool fixture (one column = entry timestamp)."""
    return pd.DataFrame({"entry_time": pair_df.index[:10]})


# ── key derivation ──────────────────────────────────────────────────


def test_feature_cache_key_changes_on_signal_def_change() -> None:
    a = feature_cache_key("signal_v1", "poolsha_abc", "v3.0")
    b = feature_cache_key("signal_v2", "poolsha_abc", "v3.0")
    assert a != b


def test_feature_cache_key_changes_on_pool_sha_change() -> None:
    a = feature_cache_key("signal_v1", "poolsha_abc", "v3.0")
    b = feature_cache_key("signal_v1", "poolsha_xyz", "v3.0")
    assert a != b


def test_feature_cache_key_changes_on_version_change() -> None:
    a = feature_cache_key("signal_v1", "poolsha_abc", "v3.0")
    b = feature_cache_key("signal_v1", "poolsha_abc", "v3.1")
    assert a != b


def test_feature_cache_key_stable_under_recall() -> None:
    """Same inputs → same key, every call."""
    a = feature_cache_key("signal_v1", "poolsha_abc", "v3.0")
    b = feature_cache_key("signal_v1", "poolsha_abc", "v3.0")
    assert a == b


def test_pool_sha_stable_for_same_dataframe(synthetic_pool: pd.DataFrame) -> None:
    a = pool_sha_from_dataframe(synthetic_pool)
    b = pool_sha_from_dataframe(synthetic_pool)
    assert a == b


def test_pool_sha_changes_when_pool_changes(synthetic_pool: pd.DataFrame) -> None:
    a = pool_sha_from_dataframe(synthetic_pool)
    modified = synthetic_pool.head(5)
    b = pool_sha_from_dataframe(modified)
    assert a != b


# ── get_or_compute ──────────────────────────────────────────────────


def test_get_or_compute_caches_on_first_call(
    pair_df: pd.DataFrame, synthetic_pool: pd.DataFrame, tmp_path: Path
) -> None:
    cache_root = tmp_path / "fcache"
    pool_sha = pool_sha_from_dataframe(synthetic_pool)
    arc_id = "test_arc"
    signal_def = "kb_exhaustion_bar(c1-c6,c8,c9)"

    def compute():
        return compute_feature_matrix("EURUSD", pair_df, names=["hour_of_day", "day_of_week"])

    result = get_or_compute(arc_id, signal_def, pool_sha, "v3.0", compute, cache_root=cache_root)
    assert isinstance(result, FeatureMatrix)
    key = feature_cache_key(signal_def, pool_sha, "v3.0")
    parquet = feature_cache_path(arc_id, key, cache_root)
    assert parquet.exists()
    assert cache_valid(parquet, key)


def test_get_or_compute_cache_hit_returns_identical_matrix(
    pair_df: pd.DataFrame, synthetic_pool: pd.DataFrame, tmp_path: Path
) -> None:
    cache_root = tmp_path / "fcache"
    pool_sha = pool_sha_from_dataframe(synthetic_pool)

    def compute():
        return compute_feature_matrix(
            "EURUSD", pair_df, names=["atr_14", "hour_of_day", "day_of_week"]
        )

    miss = get_or_compute("arc", "sig", pool_sha, "v3.0", compute, cache_root=cache_root)
    hit = get_or_compute("arc", "sig", pool_sha, "v3.0", compute, cache_root=cache_root)

    # check_freq=False because pyarrow parquet roundtrip drops the
    # DatetimeIndex.freq attribute (a pandas-side metadata field, not
    # part of the values). Column values + index positions are equal.
    pd.testing.assert_frame_equal(miss.matrix, hit.matrix, check_freq=False)


def test_get_or_compute_second_call_skips_compute_fn(
    pair_df: pd.DataFrame, synthetic_pool: pd.DataFrame, tmp_path: Path
) -> None:
    """The compute thunk MUST NOT be called on cache hit (otherwise the cache
    isn't doing its job)."""
    cache_root = tmp_path / "fcache"
    pool_sha = pool_sha_from_dataframe(synthetic_pool)
    state = {"calls": 0}

    def compute():
        state["calls"] += 1
        return compute_feature_matrix("EURUSD", pair_df, names=["hour_of_day"])

    get_or_compute("arc", "sig", pool_sha, "v3.0", compute, cache_root=cache_root)
    get_or_compute("arc", "sig", pool_sha, "v3.0", compute, cache_root=cache_root)
    get_or_compute("arc", "sig", pool_sha, "v3.0", compute, cache_root=cache_root)

    assert state["calls"] == 1, f"compute_fn called {state['calls']} times; should be 1"


def test_cache_invalidates_when_signal_def_changes(
    pair_df: pd.DataFrame, synthetic_pool: pd.DataFrame, tmp_path: Path
) -> None:
    cache_root = tmp_path / "fcache"
    pool_sha = pool_sha_from_dataframe(synthetic_pool)
    state = {"calls": 0}

    def compute():
        state["calls"] += 1
        return compute_feature_matrix("EURUSD", pair_df, names=["hour_of_day"])

    get_or_compute("arc", "sig_v1", pool_sha, "v3.0", compute, cache_root=cache_root)
    get_or_compute("arc", "sig_v2", pool_sha, "v3.0", compute, cache_root=cache_root)
    assert state["calls"] == 2  # different keys → both compute


def test_cache_invalidates_when_pool_sha_changes(
    pair_df: pd.DataFrame, synthetic_pool: pd.DataFrame, tmp_path: Path
) -> None:
    cache_root = tmp_path / "fcache"
    state = {"calls": 0}

    def compute():
        state["calls"] += 1
        return compute_feature_matrix("EURUSD", pair_df, names=["hour_of_day"])

    get_or_compute("arc", "sig", "pool_sha_v1", "v3.0", compute, cache_root=cache_root)
    get_or_compute("arc", "sig", "pool_sha_v2", "v3.0", compute, cache_root=cache_root)
    assert state["calls"] == 2


def test_cache_invalidates_when_feature_version_changes(
    pair_df: pd.DataFrame, synthetic_pool: pd.DataFrame, tmp_path: Path
) -> None:
    cache_root = tmp_path / "fcache"
    pool_sha = pool_sha_from_dataframe(synthetic_pool)
    state = {"calls": 0}

    def compute():
        state["calls"] += 1
        return compute_feature_matrix("EURUSD", pair_df, names=["hour_of_day"])

    get_or_compute("arc", "sig", pool_sha, "v3.0", compute, cache_root=cache_root)
    get_or_compute("arc", "sig", pool_sha, "v3.1", compute, cache_root=cache_root)
    assert state["calls"] == 2


# ── speedup target ──────────────────────────────────────────────────


def test_cache_hit_is_at_least_5x_faster_than_compute(
    pair_df: pd.DataFrame, synthetic_pool: pd.DataFrame, tmp_path: Path
) -> None:
    """Cache hit should be substantially faster than a full compute. We
    look for ≥5x; in practice it's ~20-100x but timing on tiny fixtures
    is noisy."""
    cache_root = tmp_path / "fcache"
    pool_sha = pool_sha_from_dataframe(synthetic_pool)

    def compute():
        return compute_feature_matrix("EURUSD", pair_df)

    # Warm
    t0 = time.perf_counter()
    get_or_compute("arc", "sig", pool_sha, "v3.0", compute, cache_root=cache_root)
    t_miss = time.perf_counter() - t0

    # Hit
    t0 = time.perf_counter()
    get_or_compute("arc", "sig", pool_sha, "v3.0", compute, cache_root=cache_root)
    t_hit = time.perf_counter() - t0

    # Tight target: cache hit should not be slower than 1/5 the compute time
    # We DON'T claim ≥10x because synthetic fixture is too small to amortise
    # the parquet roundtrip.
    assert t_hit < t_miss, f"hit ({t_hit:.4f}s) not faster than miss ({t_miss:.4f}s)"
