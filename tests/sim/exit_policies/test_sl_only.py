"""``sl_only`` baseline: pure no-op. Apply yields no Order mods; both
evaluate hooks always return None.
"""

from __future__ import annotations

import pandas as pd

from core.sim.account import Account, Direction
from core.sim.exit_policies import (
    ExitPolicyContext,
    NullPolicyState,
    SlOnlyPolicy,
)


def _bar() -> pd.Series:
    return pd.Series(
        dict(
            open_bid=1.10, open_ask=1.1001,
            high_bid=1.20, high_ask=1.2001,
            low_bid=1.00, low_ask=1.0001,
            close_bid=1.15, close_ask=1.1501,
        )
    )


def test_apply_to_order_returns_empty() -> None:
    ctx = ExitPolicyContext(
        entry_price=1.10, atr_at_entry=0.001, sl_atr_mult=2.0,
        direction=Direction.LONG,
    )
    assert SlOnlyPolicy().apply_to_order(ctx) == {}


def test_make_state_returns_null() -> None:
    ctx = ExitPolicyContext(
        entry_price=1.10, atr_at_entry=0.001, sl_atr_mult=2.0,
        direction=Direction.LONG,
    )
    assert isinstance(SlOnlyPolicy().make_state(ctx), NullPolicyState)


def test_evaluate_intrabar_always_none() -> None:
    from core.sim.account import Position
    ctx = ExitPolicyContext(
        entry_price=1.10, atr_at_entry=0.001, sl_atr_mult=2.0,
        direction=Direction.LONG,
    )
    p = SlOnlyPolicy()
    state = p.make_state(ctx)
    pos = Position(1, "EURUSD", Direction.LONG, pd.Timestamp("2026-01-01", tz="UTC"), 1.10, 10_000)
    assert p.evaluate_intrabar(pos, _bar(), state, ctx) is None


def test_evaluate_at_close_always_none() -> None:
    from core.sim.account import Position
    ctx = ExitPolicyContext(
        entry_price=1.10, atr_at_entry=0.001, sl_atr_mult=2.0,
        direction=Direction.LONG,
    )
    p = SlOnlyPolicy()
    state = p.make_state(ctx)
    pos = Position(1, "EURUSD", Direction.LONG, pd.Timestamp("2026-01-01", tz="UTC"), 1.10, 10_000)
    acct = Account(starting_balance=100_000.0)
    assert p.evaluate_at_close(pos, _bar(), state, ctx, acct) is None
