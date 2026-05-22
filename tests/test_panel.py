"""Tests for core/sim/panel.py — multi-pair OHLC panel."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from core.data.aggregator import aggregate
from core.sim.panel import Panel
from tests.fixtures.histdata_mini.build import build_fixture


@pytest.fixture
def mini_root(tmp_path: Path) -> Path:
    return build_fixture(tmp_path / "histdata")


@pytest.fixture
def cache_root(tmp_path: Path) -> Path:
    return tmp_path / "cache"


@pytest.fixture
def two_pair_panel(mini_root: Path, cache_root: Path) -> Panel:
    return Panel.from_pairs(
        ["EURUSD", "GBPUSD"], "M5", histdata_root=mini_root, cache_root=cache_root
    )


def test_panel_pairs_in_insertion_order(two_pair_panel: Panel) -> None:
    assert two_pair_panel.pairs == ("EURUSD", "GBPUSD")


def test_panel_tf_recorded(two_pair_panel: Panel) -> None:
    assert two_pair_panel.tf == "M5"


def test_panel_timestamps_union(two_pair_panel: Panel) -> None:
    ts = two_pair_panel.timestamps
    assert ts.is_monotonic_increasing
    assert str(ts.tz) == "UTC"
    assert len(ts) > 0


def test_panel_snapshot_at_returns_bars(two_pair_panel: Panel) -> None:
    t = two_pair_panel.timestamps[0]
    snap = two_pair_panel.snapshot_at(t)
    assert set(snap.keys()) == {"EURUSD", "GBPUSD"}
    for pair, bar in snap.items():
        assert bar is not None
        assert "close_bid" in bar.index
        assert "close_ask" in bar.index


def test_panel_iter_bars_yields_every_timestamp(two_pair_panel: Panel) -> None:
    seen = list(two_pair_panel.iter_bars())
    assert len(seen) == len(two_pair_panel.timestamps)
    timestamps_seen = [t for t, _ in seen]
    assert timestamps_seen == list(two_pair_panel.timestamps)


def test_panel_rejects_empty_pair_dict() -> None:
    with pytest.raises(ValueError, match="at least one pair"):
        Panel(pair_dfs={}, tf="M5")


def test_panel_rejects_column_mismatch(mini_root: Path, cache_root: Path) -> None:
    eur = aggregate("EURUSD", "M5", histdata_root=mini_root, cache_root=cache_root)
    gbp = aggregate("GBPUSD", "M5", histdata_root=mini_root, cache_root=cache_root)
    gbp_renamed = gbp.rename(columns={"close_bid": "close_bid_X"})
    with pytest.raises(ValueError, match="column mismatch"):
        Panel(pair_dfs={"EURUSD": eur, "GBPUSD": gbp_renamed}, tf="M5")


def test_panel_rejects_tznaive_index(mini_root: Path, cache_root: Path) -> None:
    eur = aggregate("EURUSD", "M5", histdata_root=mini_root, cache_root=cache_root)
    eur_naive = eur.copy()
    eur_naive.index = eur_naive.index.tz_localize(None)
    with pytest.raises(ValueError, match="tz-naive"):
        Panel(pair_dfs={"EURUSD": eur_naive}, tf="M5")


def test_panel_bar_for_missing_timestamp_returns_none(two_pair_panel: Panel) -> None:
    far_future = pd.Timestamp("2099-01-01", tz="UTC")
    assert two_pair_panel.bar_for("EURUSD", far_future) is None


def test_panel_from_frames_constructor(mini_root: Path, cache_root: Path) -> None:
    eur = aggregate("EURUSD", "M5", histdata_root=mini_root, cache_root=cache_root)
    p = Panel.from_frames({"EURUSD": eur}, tf="M5")
    assert p.pairs == ("EURUSD",)
