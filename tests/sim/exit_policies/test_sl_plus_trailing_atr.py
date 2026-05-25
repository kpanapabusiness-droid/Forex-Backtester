"""``sl_plus_trailing_atr``: activate at MFE ≥ +1R, trail at peak − R_atr,
exit when close ≤ trail. Long + short symmetry.
"""

from __future__ import annotations

import pandas as pd
import pytest

from core.sim.account import Account, Direction, Position
from core.sim.exit_policies import (
    ExitAction,
    ExitPolicyContext,
    SlPlusTrailingAtrPolicy,
)


def _pos(direction: Direction = Direction.LONG) -> Position:
    return Position(
        position_id=1,
        pair="EURUSD",
        direction=direction,
        entry_time=pd.Timestamp("2026-01-01", tz="UTC"),
        entry_price=1.1000,
        size=10_000,
    )


def _ctx(direction: Direction = Direction.LONG) -> ExitPolicyContext:
    # R_atr = 2.0 * 0.001 = 0.002
    return ExitPolicyContext(
        entry_price=1.1000, atr_at_entry=0.001, sl_atr_mult=2.0, direction=direction
    )


def _bar(**kw) -> pd.Series:
    base = dict(
        open_bid=1.10, open_ask=1.1001,
        high_bid=1.10, high_ask=1.1001,
        low_bid=1.10, low_ask=1.1001,
        close_bid=1.10, close_ask=1.1001,
    )
    base.update(kw)
    return pd.Series(base)


def test_long_no_activation_below_one_r() -> None:
    p = SlPlusTrailingAtrPolicy()
    state = p.make_state(_ctx())
    pos = _pos()
    acct = Account(starting_balance=100_000.0)
    bar = _bar(high_bid=1.1019, close_bid=1.1015)  # just below +1R = 1.1020
    assert p.evaluate_at_close(pos, bar, state, _ctx(), acct) is None
    assert not state.activated


def test_long_activates_at_one_r() -> None:
    p = SlPlusTrailingAtrPolicy()
    state = p.make_state(_ctx())
    pos = _pos()
    acct = Account(starting_balance=100_000.0)
    bar = _bar(high_bid=1.1020, close_bid=1.1018)  # at +1R, close above trail (1.100)
    assert p.evaluate_at_close(pos, bar, state, _ctx(), acct) is None
    assert state.activated
    assert state.peak_price == pytest.approx(1.1020)


def test_long_trail_ratchets_with_new_high() -> None:
    p = SlPlusTrailingAtrPolicy()
    state = p.make_state(_ctx())
    pos = _pos()
    acct = Account(starting_balance=100_000.0)
    # Activation bar: peak=1.1020
    p.evaluate_at_close(pos, _bar(high_bid=1.1020, close_bid=1.1018), state, _ctx(), acct)
    # Next bar: higher high, peak ratchets
    p.evaluate_at_close(pos, _bar(high_bid=1.1050, close_bid=1.1040), state, _ctx(), acct)
    assert state.peak_price == pytest.approx(1.1050)
    # Next bar: lower high, peak stays
    p.evaluate_at_close(pos, _bar(high_bid=1.1040, close_bid=1.1035), state, _ctx(), acct)
    assert state.peak_price == pytest.approx(1.1050)


def test_long_fires_when_close_drops_to_trail() -> None:
    p = SlPlusTrailingAtrPolicy()
    state = p.make_state(_ctx())
    pos = _pos()
    acct = Account(starting_balance=100_000.0)
    # Activation: peak=1.105 → trail = 1.105 - 0.002 = 1.103
    p.evaluate_at_close(pos, _bar(high_bid=1.105, close_bid=1.104), state, _ctx(), acct)
    # Close drops to 1.103 → fires (close <= trail)
    bar = _bar(high_bid=1.104, close_bid=1.103)
    decision = p.evaluate_at_close(pos, bar, state, _ctx(), acct)
    assert decision is not None
    assert decision.action is ExitAction.FULL_CLOSE
    assert decision.exit_reason == "trailing_stop_atr"
    assert decision.timing == "at_close"


def test_long_does_not_fire_when_close_above_trail() -> None:
    p = SlPlusTrailingAtrPolicy()
    state = p.make_state(_ctx())
    pos = _pos()
    acct = Account(starting_balance=100_000.0)
    p.evaluate_at_close(pos, _bar(high_bid=1.105, close_bid=1.104), state, _ctx(), acct)
    bar = _bar(high_bid=1.105, close_bid=1.1031)  # close just above 1.103 trail
    assert p.evaluate_at_close(pos, bar, state, _ctx(), acct) is None


def test_short_activates_at_minus_one_r() -> None:
    p = SlPlusTrailingAtrPolicy()
    state = p.make_state(_ctx(Direction.SHORT))
    pos = _pos(Direction.SHORT)
    acct = Account(starting_balance=100_000.0)
    # Short tp1: low_ask <= entry - R_atr = 1.0980
    bar = _bar(low_ask=1.0980, close_ask=1.0982)
    p.evaluate_at_close(pos, bar, state, _ctx(Direction.SHORT), acct)
    assert state.activated
    assert state.peak_price == pytest.approx(1.0980)


def test_short_fires_when_close_rises_to_trail() -> None:
    p = SlPlusTrailingAtrPolicy()
    state = p.make_state(_ctx(Direction.SHORT))
    pos = _pos(Direction.SHORT)
    acct = Account(starting_balance=100_000.0)
    # Activation: trough=1.095 → trail = 1.095 + 0.002 = 1.097
    p.evaluate_at_close(pos, _bar(low_ask=1.095, close_ask=1.096), state, _ctx(Direction.SHORT), acct)
    # Close rises to 1.097 → fires
    bar = _bar(low_ask=1.096, close_ask=1.097)
    decision = p.evaluate_at_close(pos, bar, state, _ctx(Direction.SHORT), acct)
    assert decision is not None
    assert decision.action is ExitAction.FULL_CLOSE
