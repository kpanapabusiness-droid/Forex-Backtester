"""Tests for core/sim/fill.py — bar-level fill primitives."""

from __future__ import annotations

import pandas as pd

from core.sim.fill import (
    long_entry_fill_price,
    long_exit_market_price,
    long_sl_triggered,
    long_tp_triggered,
    short_entry_fill_price,
    short_exit_market_price,
    short_sl_triggered,
    short_tp_triggered,
)


def _bar(
    open_bid, high_bid, low_bid, close_bid, open_ask, high_ask, low_ask, close_ask
) -> pd.Series:
    return pd.Series(
        {
            "open_bid": open_bid,
            "high_bid": high_bid,
            "low_bid": low_bid,
            "close_bid": close_bid,
            "open_ask": open_ask,
            "high_ask": high_ask,
            "low_ask": low_ask,
            "close_ask": close_ask,
        }
    )


# ── long side ──────────────────────────────────────────────────────────


def test_long_entry_fills_at_open_ask() -> None:
    bar = _bar(1.1000, 1.1010, 1.0990, 1.1005, 1.1002, 1.1012, 1.0992, 1.1007)
    assert long_entry_fill_price(bar) == 1.1002


def test_long_exit_market_fills_at_close_bid() -> None:
    bar = _bar(1.1000, 1.1010, 1.0990, 1.1005, 1.1002, 1.1012, 1.0992, 1.1007)
    assert long_exit_market_price(bar) == 1.1005


def test_long_sl_triggered_when_low_bid_reaches() -> None:
    bar = _bar(1.1000, 1.1010, 1.0990, 1.1005, 1.1002, 1.1012, 1.0992, 1.1007)
    hit, px = long_sl_triggered(bar, 1.0995)  # SL within the low_bid range
    assert hit is True
    assert px == 1.0995


def test_long_sl_not_triggered_when_low_bid_above() -> None:
    bar = _bar(1.1000, 1.1010, 1.0990, 1.1005, 1.1002, 1.1012, 1.0992, 1.1007)
    hit, px = long_sl_triggered(bar, 1.0980)
    assert hit is False
    assert px != px  # NaN


def test_long_tp_triggered_when_high_bid_reaches() -> None:
    bar = _bar(1.1000, 1.1010, 1.0990, 1.1005, 1.1002, 1.1012, 1.0992, 1.1007)
    hit, px = long_tp_triggered(bar, 1.1008)
    assert hit is True
    assert px == 1.1008


def test_long_tp_not_triggered_when_high_bid_below() -> None:
    bar = _bar(1.1000, 1.1010, 1.0990, 1.1005, 1.1002, 1.1012, 1.0992, 1.1007)
    hit, _ = long_tp_triggered(bar, 1.1020)
    assert hit is False


# ── short side ─────────────────────────────────────────────────────────


def test_short_entry_fills_at_open_bid() -> None:
    bar = _bar(1.1000, 1.1010, 1.0990, 1.1005, 1.1002, 1.1012, 1.0992, 1.1007)
    assert short_entry_fill_price(bar) == 1.1000


def test_short_exit_market_fills_at_close_ask() -> None:
    bar = _bar(1.1000, 1.1010, 1.0990, 1.1005, 1.1002, 1.1012, 1.0992, 1.1007)
    assert short_exit_market_price(bar) == 1.1007


def test_short_sl_triggered_when_high_ask_reaches() -> None:
    bar = _bar(1.1000, 1.1010, 1.0990, 1.1005, 1.1002, 1.1012, 1.0992, 1.1007)
    hit, px = short_sl_triggered(bar, 1.1011)
    assert hit is True
    assert px == 1.1011


def test_short_sl_not_triggered_when_high_ask_below() -> None:
    bar = _bar(1.1000, 1.1010, 1.0990, 1.1005, 1.1002, 1.1012, 1.0992, 1.1007)
    hit, _ = short_sl_triggered(bar, 1.1020)
    assert hit is False


def test_short_tp_triggered_when_low_ask_reaches() -> None:
    bar = _bar(1.1000, 1.1010, 1.0990, 1.1005, 1.1002, 1.1012, 1.0992, 1.1007)
    hit, px = short_tp_triggered(bar, 1.0995)
    assert hit is True
    assert px == 1.0995


def test_short_tp_not_triggered_when_low_ask_above() -> None:
    bar = _bar(1.1000, 1.1010, 1.0990, 1.1005, 1.1002, 1.1012, 1.0992, 1.1007)
    hit, _ = short_tp_triggered(bar, 1.0980)
    assert hit is False


# ── boundary cases ─────────────────────────────────────────────────────


def test_long_sl_exact_touch_triggers() -> None:
    """Equality on the boundary triggers (low_bid <= sl_price)."""
    bar = _bar(1.1000, 1.1010, 1.0990, 1.1005, 1.1002, 1.1012, 1.0992, 1.1007)
    hit, px = long_sl_triggered(bar, 1.0990)
    assert hit is True
    assert px == 1.0990


def test_long_tp_exact_touch_triggers() -> None:
    bar = _bar(1.1000, 1.1010, 1.0990, 1.1005, 1.1002, 1.1012, 1.0992, 1.1007)
    hit, px = long_tp_triggered(bar, 1.1010)
    assert hit is True
    assert px == 1.1010
