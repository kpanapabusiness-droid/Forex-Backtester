"""Tests for core/spread/real_spread.py — DQ helpers + summary collation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from core.data.histdata_loader import (
    DQ_NAN_BID_OR_ASK,
    DQ_OK,
    DQ_ZERO_OR_NEG_SPREAD,
    load_m1,
)
from core.spread.real_spread import (
    DataQualitySummary,
    collate_summaries,
    data_quality_summary,
    is_tradable_bar,
    per_bar_spread,
)
from tests.fixtures.histdata_mini.build import build_fixture


@pytest.fixture
def m1_df(tmp_path: Path) -> pd.DataFrame:
    root = build_fixture(tmp_path / "histdata")
    return load_m1("EURUSD", histdata_root=root, cache_root=tmp_path / "cache")


def test_per_bar_spread_returns_spread_close(m1_df: pd.DataFrame) -> None:
    s = per_bar_spread(m1_df)
    pd.testing.assert_series_equal(s, m1_df["spread_close"])


def test_is_tradable_bar_all_true_on_clean_fixture(m1_df: pd.DataFrame) -> None:
    """Synthetic fixture has positive spread everywhere → all bars tradable."""
    assert is_tradable_bar(m1_df).all()


def test_is_tradable_bar_false_on_zero_spread() -> None:
    df = pd.DataFrame(
        {
            "spread_close": [0.0001, 0.0, -0.0001],
            "bid_ask_data_quality": [DQ_OK, DQ_ZERO_OR_NEG_SPREAD, DQ_ZERO_OR_NEG_SPREAD],
        }
    )
    mask = is_tradable_bar(df)
    assert mask.tolist() == [True, False, False]


def test_is_tradable_bar_false_on_nan() -> None:
    df = pd.DataFrame(
        {
            "spread_close": [0.0001, float("nan")],
            "bid_ask_data_quality": [DQ_OK, DQ_NAN_BID_OR_ASK],
        }
    )
    mask = is_tradable_bar(df)
    assert mask.tolist() == [True, False]


def test_data_quality_summary_clean_fixture(m1_df: pd.DataFrame) -> None:
    summ = data_quality_summary(m1_df, pair="EURUSD", tf="M1")
    assert isinstance(summ, DataQualitySummary)
    assert summ.pair == "EURUSD"
    assert summ.tf == "M1"
    assert summ.total_bars == len(m1_df)
    assert summ.ok == len(m1_df)
    assert summ.zero_or_neg_spread == 0
    assert summ.nan_bid_or_ask == 0
    assert summ.bad_total == 0
    assert summ.bad_pct == 0.0


def test_data_quality_summary_with_bad_bars() -> None:
    df = pd.DataFrame(
        {
            "spread_close": [0.0001, 0.0, float("nan"), 0.0001],
            "bid_ask_data_quality": [
                DQ_OK,
                DQ_ZERO_OR_NEG_SPREAD,
                DQ_NAN_BID_OR_ASK,
                DQ_OK,
            ],
        }
    )
    summ = data_quality_summary(df, pair="X", tf="M5")
    assert summ.total_bars == 4
    assert summ.ok == 2
    assert summ.zero_or_neg_spread == 1
    assert summ.nan_bid_or_ask == 1
    assert summ.bad_total == 2
    assert summ.bad_pct == 0.5


def test_data_quality_summary_as_dict_roundtrip() -> None:
    summ = DataQualitySummary("X", "M5", 10, 8, 1, 1)
    d = summ.as_dict()
    assert d["pair"] == "X"
    assert d["tf"] == "M5"
    assert d["bad_total"] == 2
    assert d["bad_pct"] == 0.2


def test_collate_summaries_aggregates() -> None:
    a = DataQualitySummary("A", "H1", 100, 95, 3, 2)
    b = DataQualitySummary("B", "H1", 200, 180, 10, 10)
    out = collate_summaries([a, b])
    assert out["n_summaries"] == 2
    assert out["total_bars"] == 300
    assert out["ok"] == 275
    assert out["zero_or_neg_spread"] == 13
    assert out["nan_bid_or_ask"] == 12
    assert out["bad_total"] == 25
    assert out["bad_pct"] == pytest.approx(25 / 300)


def test_collate_summaries_empty_safe() -> None:
    out = collate_summaries([])
    assert out["n_summaries"] == 0
    assert out["total_bars"] == 0
    assert out["bad_pct"] == 0.0
