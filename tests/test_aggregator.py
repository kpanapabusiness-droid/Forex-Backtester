"""Tests for core/data/aggregator.py — M1→TF deterministic OHLC aggregation."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import pandas as pd
import pytest

from core.data.aggregator import (
    SUPPORTED_TFS,
    aggregate,
    aggregate_m1_to_tf,
)
from core.data.cache_keys import read_meta
from core.data.histdata_loader import M1_COLUMNS, load_m1
from tests.fixtures.histdata_mini.build import build_fixture


@pytest.fixture
def mini_root(tmp_path: Path) -> Path:
    return build_fixture(tmp_path / "histdata")


@pytest.fixture
def cache_root(tmp_path: Path) -> Path:
    return tmp_path / "cache"


@pytest.fixture
def m1_df(mini_root: Path, cache_root: Path) -> pd.DataFrame:
    return load_m1("EURUSD", histdata_root=mini_root, cache_root=cache_root)


@pytest.mark.parametrize("tf", SUPPORTED_TFS)
def test_aggregate_returns_canonical_schema(m1_df: pd.DataFrame, tf: str) -> None:
    out = aggregate_m1_to_tf(m1_df, tf)
    assert list(out.columns) == M1_COLUMNS
    assert out.index.tz is not None
    assert len(out) >= 1


def test_aggregate_m5_ohlc_rules(m1_df: pd.DataFrame) -> None:
    """Verify first/max/min/last/sum on a concrete 5-minute block.

    Fixture has 12 minutes per month at 00:00..00:11. M5 bars: [00:00, 00:05),
    [00:05, 00:10), [00:10, 00:15). The third bar covers minutes 10–11 only.
    """
    out = aggregate_m1_to_tf(m1_df, "M5")
    # First M5 bar should aggregate minutes 0..4 of the first month
    first_5 = m1_df.iloc[:5]
    first_bar = out.iloc[0]
    assert first_bar["open_bid"] == first_5["open_bid"].iloc[0]
    assert first_bar["high_bid"] == first_5["high_bid"].max()
    assert first_bar["low_bid"] == first_5["low_bid"].min()
    assert first_bar["close_bid"] == first_5["close_bid"].iloc[-1]
    assert first_bar["volume"] == first_5["volume"].sum()


def test_aggregate_h4_anchors_to_zero_oclock(m1_df: pd.DataFrame) -> None:
    """All H4 bar labels should be at 00, 04, 08, 12, 16, or 20 UTC."""
    out = aggregate_m1_to_tf(m1_df, "H4")
    assert (out.index.hour % 4 == 0).all()


def test_aggregate_d1_anchors_to_midnight(m1_df: pd.DataFrame) -> None:
    out = aggregate_m1_to_tf(m1_df, "D1")
    assert (out.index.hour == 0).all()
    assert (out.index.minute == 0).all()


def test_aggregate_w1_anchors_to_monday(m1_df: pd.DataFrame) -> None:
    """W1 bars start at Monday 00:00 UTC."""
    out = aggregate_m1_to_tf(m1_df, "W1")
    # weekday() == 0 means Monday
    assert all(d.weekday() == 0 for d in out.index)


def test_aggregate_unsupported_tf_raises(m1_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Unsupported TF"):
        aggregate_m1_to_tf(m1_df, "M7")


def test_aggregate_byte_identical_two_runs(m1_df: pd.DataFrame, tmp_path: Path) -> None:
    """Two aggregations of the same M1 input write byte-identical parquet."""
    out_a = aggregate_m1_to_tf(m1_df, "M5")
    out_b = aggregate_m1_to_tf(m1_df, "M5")

    p_a = tmp_path / "a.parquet"
    p_b = tmp_path / "b.parquet"
    out_a.to_parquet(p_a, engine="pyarrow", compression="snappy", index=True)
    out_b.to_parquet(p_b, engine="pyarrow", compression="snappy", index=True)

    sha_a = hashlib.sha256(p_a.read_bytes()).hexdigest()
    sha_b = hashlib.sha256(p_b.read_bytes()).hexdigest()
    assert sha_a == sha_b, "Two-run aggregation produced different parquet bytes"


def test_aggregate_dataframe_equality_two_runs(m1_df: pd.DataFrame) -> None:
    """Logical equality across two runs even if parquet byte equality were brittle."""
    out_a = aggregate_m1_to_tf(m1_df, "H1")
    out_b = aggregate_m1_to_tf(m1_df, "H1")
    pd.testing.assert_frame_equal(out_a, out_b)


def test_aggregate_volume_sum_preserved(m1_df: pd.DataFrame) -> None:
    """Total volume across all bars must equal M1 total volume (no dropped minutes)."""
    out = aggregate_m1_to_tf(m1_df, "H1")
    assert out["volume"].sum() == m1_df["volume"].sum()


@pytest.mark.parametrize("tf", SUPPORTED_TFS)
def test_aggregate_writes_parquet_cache(mini_root: Path, cache_root: Path, tf: str) -> None:
    parquet = cache_root / tf / "EURUSD.parquet"
    assert not parquet.exists()
    aggregate("EURUSD", tf, histdata_root=mini_root, cache_root=cache_root)
    assert parquet.exists()
    meta = read_meta(parquet)
    assert meta is not None
    assert meta.pair == "EURUSD"
    assert meta.layer == tf


def test_aggregate_cache_hit_avoids_recompute(mini_root: Path, cache_root: Path) -> None:
    aggregate("EURUSD", "H1", histdata_root=mini_root, cache_root=cache_root)
    parquet = cache_root / "H1" / "EURUSD.parquet"
    first_mtime = parquet.stat().st_mtime_ns
    time.sleep(0.05)
    aggregate("EURUSD", "H1", histdata_root=mini_root, cache_root=cache_root)
    second_mtime = parquet.stat().st_mtime_ns
    assert second_mtime == first_mtime


def test_aggregate_tf_cache_invalidates_with_m1(mini_root: Path, cache_root: Path) -> None:
    """Bumping the M1 manifest sha cascades into the H1 cache key."""
    import json

    aggregate("EURUSD", "H1", histdata_root=mini_root, cache_root=cache_root)
    parquet = cache_root / "H1" / "EURUSD.parquet"
    first_key = read_meta(parquet).cache_key

    # Bust one EURUSD M1 sha
    manifest_path = mini_root / "m1_manifest.json"
    mfst = json.loads(manifest_path.read_text(encoding="utf-8"))
    rel = next(iter(mfst["pairs"]["EURUSD"]["files"]))
    mfst["pairs"]["EURUSD"]["files"][rel]["sha256"] = "f" * 64
    manifest_path.write_text(
        json.dumps(mfst, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    aggregate("EURUSD", "H1", histdata_root=mini_root, cache_root=cache_root)
    second_key = read_meta(parquet).cache_key
    assert first_key != second_key


def test_aggregate_idempotent_with_use_cache_false(mini_root: Path, cache_root: Path) -> None:
    a = aggregate("EURUSD", "M5", histdata_root=mini_root, cache_root=cache_root, use_cache=False)
    b = aggregate("EURUSD", "M5", histdata_root=mini_root, cache_root=cache_root, use_cache=False)
    pd.testing.assert_frame_equal(a, b)
