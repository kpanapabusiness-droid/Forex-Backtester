"""Tests for core/data/histdata_loader.py — M1 bid+ask loader with parquet cache."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import pytest

from core.data.cache_keys import read_meta
from core.data.histdata_loader import (
    DQ_OK,
    DQ_ZERO_OR_NEG_SPREAD,
    M1_COLUMNS,
    load_m1,
)
from tests.fixtures.histdata_mini.build import FixtureSpec, build_fixture


@pytest.fixture
def mini_root(tmp_path: Path) -> Path:
    """Materialise a tiny HistData layer under tmp_path. Returns its root."""
    return build_fixture(tmp_path / "histdata")


@pytest.fixture
def cache_root(tmp_path: Path) -> Path:
    return tmp_path / "cache"


def test_load_m1_returns_canonical_schema(mini_root: Path, cache_root: Path) -> None:
    df = load_m1("EURUSD", histdata_root=mini_root, cache_root=cache_root)

    assert list(df.columns) == M1_COLUMNS
    assert df.index.name == "timestamp_utc"
    assert df.index.tz is not None  # UTC
    assert str(df.index.tz) == "UTC"


def test_load_m1_row_count_matches_fixture(mini_root: Path, cache_root: Path) -> None:
    """Two months × 12 minutes each = 24 rows."""
    df = load_m1("EURUSD", histdata_root=mini_root, cache_root=cache_root)
    assert len(df) == 2 * FixtureSpec().minutes_per_month


def test_load_m1_bid_le_ask(mini_root: Path, cache_root: Path) -> None:
    df = load_m1("EURUSD", histdata_root=mini_root, cache_root=cache_root)
    # bid ≤ ask holds at every OHLC corner by fixture construction
    for col_b, col_a in [
        ("open_bid", "open_ask"),
        ("high_bid", "high_ask"),
        ("low_bid", "low_ask"),
        ("close_bid", "close_ask"),
    ]:
        assert (df[col_b] <= df[col_a]).all(), f"{col_b} > {col_a} on some row"


def test_load_m1_spread_close_consistent(mini_root: Path, cache_root: Path) -> None:
    df = load_m1("EURUSD", histdata_root=mini_root, cache_root=cache_root)
    expected = df["close_ask"] - df["close_bid"]
    pd.testing.assert_series_equal(
        df["spread_close"], expected, check_names=False, check_dtype=False
    )


def test_load_m1_data_quality_all_ok(mini_root: Path, cache_root: Path) -> None:
    """Fixture has a positive spread on every bar."""
    df = load_m1("EURUSD", histdata_root=mini_root, cache_root=cache_root)
    assert (df["bid_ask_data_quality"] == DQ_OK).all()


def test_load_m1_caches_to_parquet(mini_root: Path, cache_root: Path) -> None:
    parquet = cache_root / "m1" / "EURUSD.parquet"
    assert not parquet.exists()
    load_m1("EURUSD", histdata_root=mini_root, cache_root=cache_root)
    assert parquet.exists()
    # Sidecar meta is present and well-formed
    meta = read_meta(parquet)
    assert meta is not None
    assert meta.pair == "EURUSD"
    assert meta.layer == "m1"
    assert meta.n_rows == 2 * FixtureSpec().minutes_per_month


def test_load_m1_second_call_uses_cache(mini_root: Path, cache_root: Path) -> None:
    """Second call returns same data without re-reading CSVs (mtime check)."""
    load_m1("EURUSD", histdata_root=mini_root, cache_root=cache_root)
    parquet = cache_root / "m1" / "EURUSD.parquet"
    first_mtime = parquet.stat().st_mtime_ns
    # Tiny sleep so a re-write would show a different mtime
    time.sleep(0.05)
    df = load_m1("EURUSD", histdata_root=mini_root, cache_root=cache_root)
    second_mtime = parquet.stat().st_mtime_ns
    assert second_mtime == first_mtime, "Parquet was rewritten on cache hit"
    assert len(df) == 2 * FixtureSpec().minutes_per_month


def test_load_m1_cache_invalidates_on_manifest_change(mini_root: Path, cache_root: Path) -> None:
    """Mutating m1_manifest.json's sha for EURUSD forces a rebuild."""
    load_m1("EURUSD", histdata_root=mini_root, cache_root=cache_root)
    parquet = cache_root / "m1" / "EURUSD.parquet"
    first_key = read_meta(parquet).cache_key

    # Mutate one EURUSD file sha in the manifest
    manifest_path = mini_root / "m1_manifest.json"
    mfst = json.loads(manifest_path.read_text(encoding="utf-8"))
    eur_files = mfst["pairs"]["EURUSD"]["files"]
    first_relpath = next(iter(eur_files))
    eur_files[first_relpath]["sha256"] = "f" * 64
    manifest_path.write_text(
        json.dumps(mfst, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    # Loader should detect the change and rebuild cache; new key differs
    load_m1("EURUSD", histdata_root=mini_root, cache_root=cache_root)
    second_key = read_meta(parquet).cache_key
    assert first_key != second_key


def test_load_m1_use_cache_false_rebuilds(mini_root: Path, cache_root: Path) -> None:
    """use_cache=False ignores existing parquet and rebuilds."""
    load_m1("EURUSD", histdata_root=mini_root, cache_root=cache_root)
    parquet = cache_root / "m1" / "EURUSD.parquet"
    first_mtime = parquet.stat().st_mtime_ns
    time.sleep(0.05)
    load_m1("EURUSD", histdata_root=mini_root, cache_root=cache_root, use_cache=False)
    second_mtime = parquet.stat().st_mtime_ns
    assert second_mtime > first_mtime


def test_load_m1_two_pairs_independent(mini_root: Path, cache_root: Path) -> None:
    """Loading EURUSD must not affect GBPUSD's cache."""
    load_m1("EURUSD", histdata_root=mini_root, cache_root=cache_root)
    assert (cache_root / "m1" / "EURUSD.parquet").exists()
    assert not (cache_root / "m1" / "GBPUSD.parquet").exists()
    load_m1("GBPUSD", histdata_root=mini_root, cache_root=cache_root)
    assert (cache_root / "m1" / "GBPUSD.parquet").exists()


def test_load_m1_missing_pair_raises(mini_root: Path, cache_root: Path) -> None:
    with pytest.raises(KeyError, match="not present in M1 manifest"):
        load_m1("XXXYYY", histdata_root=mini_root, cache_root=cache_root)


def test_load_m1_missing_manifest_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_m1("EURUSD", histdata_root=tmp_path, cache_root=tmp_path / "cache")


def test_load_m1_data_quality_flag_on_zero_spread(mini_root: Path, cache_root: Path) -> None:
    """Manually inject a bad ask CSV row and verify the flag.

    Rebuilds the fixture so the bad row is in the actual on-disk CSV (loader
    re-parses on cache miss).
    """
    # Overwrite one ask CSV with a bad row (ask == bid) on its first minute,
    # then bust the manifest sha to force a re-read.
    bad = mini_root / "EURUSD/m1/ask/2010/EURUSD_M1_ASK_201001.csv"
    bid = mini_root / "EURUSD/m1/bid/2010/EURUSD_M1_BID_201001.csv"
    bid_lines = bid.read_text(encoding="utf-8").splitlines()
    header = bid_lines[0]
    bid_first_row = bid_lines[1]
    # Replace ask first row with the bid first row → close_ask == close_bid → spread=0
    ask_lines = bad.read_text(encoding="utf-8").splitlines()
    ask_lines[1] = bid_first_row
    bad.write_text("\n".join([header] + ask_lines[1:]) + "\n", encoding="utf-8", newline="\n")

    # Bust the manifest sha for that file so the loader regenerates
    manifest_path = mini_root / "m1_manifest.json"
    mfst = json.loads(manifest_path.read_text(encoding="utf-8"))
    rel = "EURUSD/m1/ask/2010/EURUSD_M1_ASK_201001.csv"
    mfst["pairs"]["EURUSD"]["files"][rel]["sha256"] = "deadbeef" + "0" * 56
    manifest_path.write_text(
        json.dumps(mfst, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    df = load_m1("EURUSD", histdata_root=mini_root, cache_root=cache_root)
    assert (df["bid_ask_data_quality"] == DQ_ZERO_OR_NEG_SPREAD).sum() == 1
