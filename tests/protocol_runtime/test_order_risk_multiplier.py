"""Driver-level tests for Order.risk_multiplier wiring.

Confirms:
  - Default risk_multiplier=1.0 produces identical behaviour to pre-change.
  - risk_multiplier=0.5 halves position size at fill.
  - risk_multiplier=0.0 skips the fill (no position opened).
"""

from __future__ import annotations

from core.sim.account import Account, Direction
from core.sim.multipair_backtester import MultiPairBacktester, Order
from tests.protocol_runtime._fixtures import build_synthetic_panel


def _make_strategy(risk_multiplier: float):
    """Strategy that emits one order at bar 100 for EURUSD."""
    fired = {"done": False}
    def strategy(t, snapshot, acct):
        if fired["done"]:
            return []
        idx = snapshot.get("EURUSD")
        if idx is None:
            return []
        fired["done"] = True
        return [Order(
            pair="EURUSD",
            direction=Direction.LONG,
            size=10000.0,
            sl_price=float(idx["close_ask"]) * 0.99,
            risk_multiplier=risk_multiplier,
        )]
    return strategy


def test_risk_multiplier_default_unchanged() -> None:
    panel = build_synthetic_panel(pairs=("EURUSD",), n_bars=200)
    acct = Account(starting_balance=100_000.0)
    bt = MultiPairBacktester(panel=panel, account=acct, strategy=_make_strategy(1.0))
    bt.run()
    open_or_closed = list(acct.open_positions) + list(acct.closed_trades)
    assert len(open_or_closed) >= 1
    pos_or_trade = open_or_closed[0]
    assert pos_or_trade.size == 10000.0


def test_risk_multiplier_half_halves_size() -> None:
    panel = build_synthetic_panel(pairs=("EURUSD",), n_bars=200)
    acct = Account(starting_balance=100_000.0)
    bt = MultiPairBacktester(panel=panel, account=acct, strategy=_make_strategy(0.5))
    bt.run()
    open_or_closed = list(acct.open_positions) + list(acct.closed_trades)
    assert len(open_or_closed) >= 1
    assert open_or_closed[0].size == 5000.0


def test_risk_multiplier_zero_skips_fill() -> None:
    panel = build_synthetic_panel(pairs=("EURUSD",), n_bars=200)
    acct = Account(starting_balance=100_000.0)
    bt = MultiPairBacktester(panel=panel, account=acct, strategy=_make_strategy(0.0))
    bt.run()
    assert len(acct.open_positions) == 0
    assert len(acct.closed_trades) == 0
