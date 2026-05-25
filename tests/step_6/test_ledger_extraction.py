"""Tests for the ledger-extraction helper used by Step 6's spread P&L wiring."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import SimpleNamespace

import pandas as pd

from core.step_6.ledger_extraction import (
    build_top_1_fold_assignments,
    build_top_1_trade_ledger,
    extract_top_1_ledger_bundle,
)


class _Dir(Enum):
    LONG = "long"
    SHORT = "short"


@dataclass(frozen=True)
class _FakeTrade:
    position_id: int
    pair: str
    direction: _Dir
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    size: float
    pnl: float
    exit_reason: str
    parent_position_id: int | None = None
    entry_bid: float = 1.0
    entry_ask: float = 1.0001
    exit_bid: float = 1.005
    exit_ask: float = 1.0051
    sl_price: float | None = None


def _mk_trade(i: int) -> _FakeTrade:
    return _FakeTrade(
        position_id=i,
        pair="EURUSD",
        direction=_Dir.LONG,
        entry_time=pd.Timestamp("2020-01-01") + pd.Timedelta(days=i),
        entry_price=1.0,
        exit_time=pd.Timestamp("2020-01-02") + pd.Timedelta(days=i),
        exit_price=1.005,
        size=10000.0,
        pnl=50.0,
        exit_reason="stop_loss",
        sl_price=0.995,
    )


def test_build_ledger_returns_none_for_unknown_config():
    out = build_top_1_trade_ledger("does_not_exist", {}, ())
    assert out is None


def test_build_ledger_flattens_per_fold_trades():
    strategy = {
        "A1::cfg_x": {
            1: SimpleNamespace(closed_trades=(_mk_trade(1), _mk_trade(2))),
            2: SimpleNamespace(closed_trades=(_mk_trade(3),)),
        }
    }
    ledger = build_top_1_trade_ledger("A1::cfg_x", strategy, ())
    assert ledger is not None
    assert len(ledger) == 3
    assert set(["entry_bid", "exit_ask", "sl_price"]).issubset(ledger.columns)
    # direction Enum should be serialised to lowercase name
    assert (ledger["direction"] == "long").all()


def test_build_fold_assignments_row_order_matches_ledger():
    strategy = {
        "A1::cfg_x": {
            1: SimpleNamespace(closed_trades=(_mk_trade(1), _mk_trade(2))),
            2: SimpleNamespace(closed_trades=(_mk_trade(3),)),
        }
    }
    fa = build_top_1_fold_assignments("A1::cfg_x", strategy, (), None)
    assert fa is not None
    assert list(fa["leg_id"]) == [0, 1, 2]
    assert list(fa["fold_id"]) == [1, 1, 2]


def test_extract_bundle_returns_none_when_no_top1():
    ledger, fa, hfid = extract_top_1_ledger_bundle(None, {}, (), holdout_fold_id=10)
    assert ledger is None and fa is None and hfid == 10


def test_extract_bundle_returns_none_when_strategy_results_empty():
    ledger, fa, hfid = extract_top_1_ledger_bundle("A1::cfg_x", None, (), holdout_fold_id=10)
    assert ledger is None and fa is None and hfid == 10


def test_extract_bundle_happy_path():
    strategy = {
        "A1::cfg_x": {
            1: SimpleNamespace(closed_trades=(_mk_trade(1),)),
        }
    }
    ledger, fa, hfid = extract_top_1_ledger_bundle(
        "A1::cfg_x", strategy, (), holdout_fold_id=99,
    )
    assert ledger is not None and len(ledger) == 1
    assert fa is not None and list(fa["fold_id"]) == [1]
    assert hfid == 99
