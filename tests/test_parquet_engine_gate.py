"""
Test the parquet engine gate helper used to skip parquet-IO tests when no engine is available.
"""

from __future__ import annotations

from tests.conftest import PARQUET_SKIP, has_parquet_engine


def test_parquet_skip_marker_importable() -> None:
    """PARQUET_SKIP is importable and is a pytest skipif mark."""
    assert PARQUET_SKIP is not None
    assert hasattr(PARQUET_SKIP, "mark")


def test_has_parquet_engine_returns_bool() -> None:
    """has_parquet_engine() returns a boolean."""
    result = has_parquet_engine()
    assert isinstance(result, bool)
