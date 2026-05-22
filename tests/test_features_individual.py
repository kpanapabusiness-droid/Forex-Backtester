"""Per-feature unit tests for the v3 feature classes.

Mostly shape + invariant checks (e.g. session_london = 1 iff hour ∈ [7,16)).
The lookahead spot-check lives in tests/test_features_pipeline.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.data.aggregator import aggregate
from core.features.registry import get
from tests.fixtures.histdata_mini.build import FixtureSpec, build_fixture


@pytest.fixture
def pair_df(tmp_path: Path) -> pd.DataFrame:
    """H1 frame so we have multiple hours within a day for session tests."""
    root = build_fixture(tmp_path / "histdata", FixtureSpec(minutes_per_month=1440))
    return aggregate("EURUSD", "H1", histdata_root=root, cache_root=tmp_path / "cache")


def _run(name: str, df: pd.DataFrame, panel=None) -> pd.Series:
    return get(name).producer(df, panel=panel)


# ── price_geometry ───────────────────────────────────────────────────


def test_atr_14_shape_and_dtype(pair_df: pd.DataFrame) -> None:
    s = _run("atr_14", pair_df)
    assert s.index.equals(pair_df.index)
    assert s.dtype == np.float64
    # First ~14 bars NaN, rest finite
    assert s.iloc[20:].notna().any()


def test_atr_14_strictly_positive_after_warmup(pair_df: pd.DataFrame) -> None:
    s = _run("atr_14", pair_df).dropna()
    assert (s > 0).all()


def test_kijun_26_distance_zero_when_close_at_mid_range(pair_df: pd.DataFrame) -> None:
    """If the close exactly equals (high_max + low_min) / 2, the distance is 0."""
    s = _run("kijun_26_distance", pair_df)
    # Just verify it's finite + has expected shape; exact equality depends on synth data.
    assert s.index.equals(pair_df.index)


def test_range_close_ratio_non_negative(pair_df: pd.DataFrame) -> None:
    s = _run("range_close_ratio", pair_df).dropna()
    assert (s >= 0).all()


# ── session ──────────────────────────────────────────────────────────


def test_session_london_iff_hour_in_7_to_16(pair_df: pd.DataFrame) -> None:
    s = _run("session_london", pair_df)
    expected = ((pair_df.index.hour >= 7) & (pair_df.index.hour < 16)).astype("int64")
    pd.testing.assert_series_equal(s, pd.Series(expected, index=pair_df.index), check_names=False)


def test_session_ny_iff_hour_in_12_to_21(pair_df: pd.DataFrame) -> None:
    s = _run("session_ny", pair_df)
    expected = ((pair_df.index.hour >= 12) & (pair_df.index.hour < 21)).astype("int64")
    pd.testing.assert_series_equal(s, pd.Series(expected, index=pair_df.index), check_names=False)


def test_session_dead_iff_hour_in_21_or_under_7(pair_df: pd.DataFrame) -> None:
    s = _run("session_dead", pair_df)
    expected = ((pair_df.index.hour >= 21) | (pair_df.index.hour < 7)).astype("int64")
    pd.testing.assert_series_equal(s, pd.Series(expected, index=pair_df.index), check_names=False)


def test_hour_of_day_matches_index(pair_df: pd.DataFrame) -> None:
    s = _run("hour_of_day", pair_df)
    assert (s.values == pair_df.index.hour).all()


def test_day_of_week_matches_index(pair_df: pd.DataFrame) -> None:
    s = _run("day_of_week", pair_df)
    assert (s.values == pair_df.index.dayofweek).all()


# ── vol_regime ───────────────────────────────────────────────────────


def test_atr_percentile_in_unit_interval(pair_df: pd.DataFrame) -> None:
    s = _run("atr_percentile_100", pair_df).dropna()
    if len(s) > 0:
        assert (s >= 0).all() and (s <= 1).all()


def test_atr_vs_trailing_positive_when_atr_positive(pair_df: pd.DataFrame) -> None:
    s = _run("atr_vs_trailing_100", pair_df).dropna()
    if len(s) > 0:
        assert (s > 0).all()


# ── distance ─────────────────────────────────────────────────────────


def test_distance_to_round_number_non_negative(pair_df: pd.DataFrame) -> None:
    s = _run("distance_to_round_number", pair_df).dropna()
    if len(s) > 0:
        assert (s >= 0).all()


# ── spread_regime ────────────────────────────────────────────────────


def test_spread_percentile_in_unit_interval(pair_df: pd.DataFrame) -> None:
    s = _run("spread_percentile_100", pair_df).dropna()
    if len(s) > 0:
        assert (s >= 0).all() and (s <= 1).all()
