"""
Phase D-6F: Regression tests for clean label summaries.
"""
from __future__ import annotations

import pandas as pd
import pytest

from analytics.phaseD6F_clean_label_summaries import _counts_clean_zoneC_by_year_pair


def _make_synthetic_zonec_df() -> pd.DataFrame:
    """Synthetic DataFrame with clean_zoneC columns across two years."""
    return pd.DataFrame({
        "pair": ["EUR_USD"] * 4 + ["GBP_USD"] * 2,
        "date": pd.to_datetime([
            "2020-01-01", "2020-06-15", "2021-03-01", "2021-12-31",
            "2020-02-01", "2021-07-01",
        ]),
        "valid_h40": [True] * 6,
        "clean_zoneC_long_x1": [True, False, True, False, True, False],
        "clean_zoneC_long_x2": [False, True, False, True, False, True],
        "clean_zoneC_short_x3": [True, True, False, False, True, False],
    })


def test_counts_clean_zoneC_returns_dataframe_no_index_error() -> None:
    """_counts_clean_zoneC_by_year_pair must not raise IndexError."""
    df = _make_synthetic_zonec_df()
    result = _counts_clean_zoneC_by_year_pair(df)
    assert isinstance(result, pd.DataFrame)
    assert not result.empty


def test_counts_clean_zoneC_expected_counts() -> None:
    """Counts per (pair, year, dir, x) are correct."""
    df = _make_synthetic_zonec_df()
    result = _counts_clean_zoneC_by_year_pair(df)
    required_cols = {"pair", "year", "dir", "x", "count"}
    assert required_cols.issubset(set(result.columns))

    r = result.set_index(["pair", "year", "dir", "x"])["count"]
    assert r.get(("EUR_USD", 2020, "long", 1), 0) == 1
    assert r.get(("EUR_USD", 2021, "long", 1), 0) == 1
    assert r.get(("EUR_USD", 2020, "long", 2), 0) == 1
    assert r.get(("EUR_USD", 2021, "long", 2), 0) == 1
    assert r.get(("EUR_USD", 2020, "short", 3), 0) == 2
    assert r.get(("GBP_USD", 2020, "long", 1), 0) == 1
    assert r.get(("GBP_USD", 2021, "short", 3), 0) == 0


def test_counts_clean_zoneC_no_matching_cols_raises() -> None:
    """Raises ValueError when no zoneC columns exist."""
    df = pd.DataFrame({
        "pair": ["X"],
        "date": pd.to_datetime(["2020-01-01"]),
        "valid_h40": [True],
    })
    with pytest.raises(ValueError, match="No clean_zoneC columns found"):
        _counts_clean_zoneC_by_year_pair(df)
