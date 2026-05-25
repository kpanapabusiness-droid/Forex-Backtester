"""``sl_plus_tp_2r`` / ``sl_plus_tp_3r`` — Order-decoration only. Both
policies set ``tp_price`` at register; the per-bar hooks are no-ops
(the existing intra-bar TP infrastructure handles fire/fill).
"""

from __future__ import annotations

import pandas as pd
import pytest

from core.sim.account import Account, Direction, Position
from core.sim.exit_policies import (
    ExitPolicyContext,
    NullPolicyState,
    SlPlusTp2RPolicy,
    SlPlusTp3RPolicy,
)


def _ctx(direction: Direction = Direction.LONG) -> ExitPolicyContext:
    # entry=1.10, atr=0.001, sl_mult=2.0 → R_atr = 0.002
    return ExitPolicyContext(
        entry_price=1.10, atr_at_entry=0.001, sl_atr_mult=2.0, direction=direction
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


@pytest.mark.parametrize(
    "policy_cls,tp_r,expected_long_tp,expected_short_tp",
    [
        (SlPlusTp2RPolicy, 2.0, 1.104, 1.096),
        (SlPlusTp3RPolicy, 3.0, 1.106, 1.094),
    ],
)
def test_apply_to_order_sets_tp_price(
    policy_cls, tp_r: float, expected_long_tp: float, expected_short_tp: float
) -> None:
    p = policy_cls()
    assert p.tp_r == tp_r
    assert p.apply_to_order(_ctx(Direction.LONG))["tp_price"] == pytest.approx(expected_long_tp)
    assert p.apply_to_order(_ctx(Direction.SHORT))["tp_price"] == pytest.approx(expected_short_tp)


@pytest.mark.parametrize("policy_cls", [SlPlusTp2RPolicy, SlPlusTp3RPolicy])
def test_state_is_null(policy_cls) -> None:
    assert isinstance(policy_cls().make_state(_ctx()), NullPolicyState)


@pytest.mark.parametrize("policy_cls", [SlPlusTp2RPolicy, SlPlusTp3RPolicy])
def test_evaluate_hooks_are_noop(policy_cls) -> None:
    p = policy_cls()
    state = p.make_state(_ctx())
    pos = Position(1, "EURUSD", Direction.LONG, pd.Timestamp("2026-01-01", tz="UTC"), 1.10, 10_000)
    acct = Account(starting_balance=100_000.0)
    assert p.evaluate_intrabar(pos, _bar(), state, _ctx()) is None
    assert p.evaluate_at_close(pos, _bar(), state, _ctx(), acct) is None


def test_tp_3r_is_subclass_of_tp_2r() -> None:
    """SlPlusTp3R reuses SlPlusTp2R's apply_to_order via tp_r override."""
    assert isinstance(SlPlusTp3RPolicy(), SlPlusTp2RPolicy)
    assert SlPlusTp3RPolicy.name == "sl_plus_tp_3r"
    assert SlPlusTp2RPolicy.name == "sl_plus_tp_2r"
