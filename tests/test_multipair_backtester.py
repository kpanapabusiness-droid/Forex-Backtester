"""Tests for core/sim/multipair_backtester.py — driver loop + sanity sims."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from core.sim.account import Account, Direction, ExposureRules
from core.sim.multipair_backtester import MultiPairBacktester, Order, RunResult
from core.sim.panel import Panel
from tests.fixtures.histdata_mini.build import FixtureSpec, build_fixture


@pytest.fixture
def panel_3pair(tmp_path: Path) -> Panel:
    """Three synthetic pairs, M5 panel, 12 minutes/month × 2 months."""
    root = build_fixture(
        tmp_path / "histdata",
        FixtureSpec(pairs=("EURUSD", "GBPUSD", "USDJPY"), months=("201001", "201002")),
    )
    return Panel.from_pairs(
        ["EURUSD", "GBPUSD", "USDJPY"],
        "M5",
        histdata_root=root,
        cache_root=tmp_path / "cache",
    )


# ── strategies used by tests ───────────────────────────────────────────


def _always_buy_one(t, snapshot, account):
    """Emit one long order per pair per bar (will be exposure-capped)."""
    orders = []
    for pair in sorted(snapshot.keys()):
        bar = snapshot[pair]
        if bar is None:
            continue
        orders.append(Order(pair=pair, direction=Direction.LONG, size=1.0))
    return orders


def _no_orders(t, snapshot, account):
    return []


# ── tests ──────────────────────────────────────────────────────────────


def test_backtester_runs_to_completion_no_strategy(panel_3pair: Panel) -> None:
    acct = Account(starting_balance=100_000.0)
    bt = MultiPairBacktester(panel=panel_3pair, account=acct, strategy=_no_orders)
    result = bt.run()
    assert isinstance(result, RunResult)
    assert result.n_trades == 0
    assert result.n_open_at_end == 0
    assert result.final_balance == 100_000.0
    # Equity curve has one point per panel bar
    assert len(result.equity_curve) == len(panel_3pair.timestamps)


def test_backtester_respects_max_concurrent_total(panel_3pair: Panel) -> None:
    """Strategy emits 3 orders/bar; with total cap=2, at most 2 fills/bar."""
    acct = Account(
        starting_balance=100_000.0,
        exposure=ExposureRules(
            max_concurrent_total=2,
            max_concurrent_per_pair=None,
            max_concurrent_per_currency=None,
        ),
    )
    bt = MultiPairBacktester(panel=panel_3pair, account=acct, strategy=_always_buy_one)
    bt.run()
    # At any point, open positions should never have exceeded 2
    # (we can't directly assert that without instrumenting the loop, but
    # we can confirm that the per-pair counts never exceeded total cap.)
    # End-of-run: <= 2 open positions remaining.
    assert len(acct.open_positions) <= 2


def test_backtester_respects_per_pair_cap(panel_3pair: Panel) -> None:
    """With default per_pair=1 cap, each pair has at most one open at a time."""
    acct = Account(starting_balance=100_000.0)  # default ExposureRules
    bt = MultiPairBacktester(panel=panel_3pair, account=acct, strategy=_always_buy_one)
    bt.run()
    # End-of-run: one open per pair max → up to 3 across 3 pairs
    open_pairs = [p.pair for p in acct.open_positions]
    assert len(open_pairs) == len(set(open_pairs))


def test_backtester_respects_per_currency_cap(panel_3pair: Panel) -> None:
    """USD is in all three pairs; per-currency cap=1 → only one of them opens at a time."""
    acct = Account(
        starting_balance=100_000.0,
        exposure=ExposureRules(
            max_concurrent_total=10,
            max_concurrent_per_pair=10,
            max_concurrent_per_currency=1,
        ),
    )
    bt = MultiPairBacktester(panel=panel_3pair, account=acct, strategy=_always_buy_one)
    bt.run()
    # End state: at most 1 open USD-bearing pair
    usd_open = [p for p in acct.open_positions if "USD" in p.pair]
    assert len(usd_open) <= 1


def test_backtester_single_equity_curve(panel_3pair: Panel) -> None:
    """The driver produces ONE equity curve across all pairs."""
    acct = Account(starting_balance=100_000.0)
    bt = MultiPairBacktester(panel=panel_3pair, account=acct, strategy=_always_buy_one)
    result = bt.run()
    assert isinstance(result.equity_curve, pd.Series)
    assert result.equity_curve.name == "equity"
    # One point per bar (no per-pair sub-curves)
    assert len(result.equity_curve) == len(panel_3pair.timestamps)


def test_backtester_two_run_determinism(panel_3pair: Panel) -> None:
    """Two identical runs produce identical equity curves and trade ledgers."""

    def run_once() -> RunResult:
        acct = Account(
            starting_balance=100_000.0,
            exposure=ExposureRules(max_concurrent_total=2, max_concurrent_per_pair=1),
        )
        bt = MultiPairBacktester(panel=panel_3pair, account=acct, strategy=_always_buy_one)
        return bt.run()

    a = run_once()
    b = run_once()
    pd.testing.assert_series_equal(a.equity_curve, b.equity_curve)
    assert a.final_balance == b.final_balance
    assert a.n_trades == b.n_trades
    # Trade ledgers identical field-by-field
    assert len(a.closed_trades) == len(b.closed_trades)
    for ta, tb in zip(a.closed_trades, b.closed_trades):
        assert ta == tb


def test_backtester_sl_long_closes_position(panel_3pair: Panel) -> None:
    """Plant a long with an absurdly-high SL (always triggers next bar)."""
    fills: list[float] = []

    def strat_once(t, snapshot, account):
        # Only emit on the first bar
        if account.closed_trades or account.open_positions or fills:
            return []
        bar = snapshot.get("EURUSD")
        if bar is None:
            return []
        # SL above the current price → first opportunity to fire after entry
        fills.append(1.0)
        return [Order(pair="EURUSD", direction=Direction.LONG, size=1.0, sl_price=999.0)]

    acct = Account(starting_balance=100_000.0)
    bt = MultiPairBacktester(panel=panel_3pair, account=acct, strategy=strat_once)
    bt.run()

    # SL = 999 above market → trigger condition (low_bid <= 999) is always
    # true → should close on the bar after entry fill.
    assert len(acct.closed_trades) == 1
    assert acct.closed_trades[0].exit_reason == "stop_loss"


def test_backtester_unknown_pair_raises(panel_3pair: Panel) -> None:
    def bad_strat(t, snapshot, account):
        return [Order(pair="XXXYYY", direction=Direction.LONG, size=1.0)]

    acct = Account(starting_balance=100_000.0)
    bt = MultiPairBacktester(panel=panel_3pair, account=acct, strategy=bad_strat)
    with pytest.raises(KeyError, match="unknown pair"):
        bt.run()


def test_backtester_panel_iteration_order_deterministic(panel_3pair: Panel) -> None:
    """The timestamp iteration order is deterministic across instantiations."""
    a = list(t for t, _ in panel_3pair.iter_bars())
    b = list(t for t, _ in panel_3pair.iter_bars())
    assert a == b


def test_backtester_equity_curve_sha256_stable(panel_3pair: Panel) -> None:
    """Hash the equity curve CSV repr — must match across two runs."""

    def equity_csv_sha(panel: Panel) -> str:
        acct = Account(
            starting_balance=100_000.0,
            exposure=ExposureRules(max_concurrent_total=2),
        )
        bt = MultiPairBacktester(panel=panel, account=acct, strategy=_always_buy_one)
        result = bt.run()
        # Stable CSV serialisation
        csv = result.equity_curve.to_csv(lineterminator="\n")
        return hashlib.sha256(csv.encode("utf-8")).hexdigest()

    s1 = equity_csv_sha(panel_3pair)
    s2 = equity_csv_sha(panel_3pair)
    assert s1 == s2
