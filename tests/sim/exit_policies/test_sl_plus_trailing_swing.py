"""``sl_plus_trailing_swing``: swing trail capped at entry, activated at
+1R MFE. Trail level = running max of min(prev_close, entry).
"""

from __future__ import annotations

import pandas as pd
import pytest

from core.sim.account import Account, Direction, Position
from core.sim.exit_policies import (
    ExitAction,
    ExitPolicyContext,
    SlPlusTrailingSwingPolicy,
    TrailingSwingState,
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
    p = SlPlusTrailingSwingPolicy()
    state = p.make_state(_ctx())
    pos = _pos()
    acct = Account(starting_balance=100_000.0)
    bar = _bar(high_bid=1.1019, close_bid=1.1015)
    assert p.evaluate_at_close(pos, bar, state, _ctx(), acct) is None
    assert not state.activated


def test_long_activates_at_one_r_but_no_trail_yet() -> None:
    """First activation bar: trail can't yet have a candidate
    (no prev_close stored yet). No fire."""
    p = SlPlusTrailingSwingPolicy()
    state = p.make_state(_ctx())
    pos = _pos()
    acct = Account(starting_balance=100_000.0)
    bar = _bar(high_bid=1.1020, close_bid=1.1015)
    decision = p.evaluate_at_close(pos, bar, state, _ctx(), acct)
    assert decision is None
    assert state.activated
    # state.prev_close set for next bar
    assert state.prev_close == pytest.approx(1.1015)
    # trail_level still NaN
    assert pd.isna(state.trail_level)


def test_long_trail_capped_at_entry() -> None:
    """Prev close ABOVE entry → candidate = min(prev_close, entry) = entry.
    Trail level becomes entry (= 1.1000).
    """
    p = SlPlusTrailingSwingPolicy()
    state = p.make_state(_ctx())
    pos = _pos()
    acct = Account(starting_balance=100_000.0)
    # Bar 1: activates, prev_close = 1.1015 (above entry 1.1000)
    p.evaluate_at_close(pos, _bar(high_bid=1.1020, close_bid=1.1015), state, _ctx(), acct)
    # Bar 2: trail candidate = min(1.1015, 1.1000) = 1.1000. trail_level becomes 1.1000.
    # close 1.1010 > 1.1000 → no fire
    bar2 = _bar(high_bid=1.103, close_bid=1.1010)
    decision = p.evaluate_at_close(pos, bar2, state, _ctx(), acct)
    assert decision is None
    assert state.trail_level == pytest.approx(1.1000)
    # close=1.1010 ≤ 1.1000? No, 1.1010 > 1.1000 → no fire. Correct.


def test_long_fires_when_close_drops_to_entry_trail() -> None:
    """After trail establishes at entry (1.1000), close drops to 1.1000 → fire."""
    p = SlPlusTrailingSwingPolicy()
    state = p.make_state(_ctx())
    pos = _pos()
    acct = Account(starting_balance=100_000.0)
    # Bar 1: activate, prev_close=1.1015
    p.evaluate_at_close(pos, _bar(high_bid=1.1020, close_bid=1.1015), state, _ctx(), acct)
    # Bar 2: trail becomes 1.1000, close 1.1010, no fire, prev_close=1.1010
    p.evaluate_at_close(pos, _bar(high_bid=1.103, close_bid=1.1010), state, _ctx(), acct)
    assert state.trail_level == pytest.approx(1.1000)
    # Bar 3: trail candidate = min(1.1010, 1.1000) = 1.1000. trail_level = max(1.1000, 1.1000) = 1.1000.
    # close 1.1000 ≤ 1.1000 → FIRE
    bar3 = _bar(close_bid=1.1000)
    decision = p.evaluate_at_close(pos, bar3, state, _ctx(), acct)
    assert decision is not None
    assert decision.action is ExitAction.FULL_CLOSE
    assert decision.exit_reason == "trailing_stop_swing"


def test_long_trail_ratchets_up_with_higher_prev_close_below_entry() -> None:
    """Sequence where prev_close goes BELOW entry — candidate is prev_close,
    and trail_level ratchets up to whichever is highest.

    Bar 1: activate (high≥+1R), prev_close = 1.0995 (below entry).
    Bar 2: candidate = min(1.0995, 1.1000) = 1.0995. trail_level = 1.0995.
    Bar 3: prev_close = 1.0998 → candidate = 1.0998. trail = max(1.0995, 1.0998) = 1.0998.
    Bar 4: close = 1.0998 → fire (close ≤ trail).
    """
    p = SlPlusTrailingSwingPolicy()
    state = p.make_state(_ctx())
    pos = _pos()
    acct = Account(starting_balance=100_000.0)
    # Bar 1
    p.evaluate_at_close(pos, _bar(high_bid=1.1020, close_bid=1.0995), state, _ctx(), acct)
    assert state.activated
    assert state.prev_close == pytest.approx(1.0995)
    # Bar 2: trail establishes at 1.0995
    p.evaluate_at_close(pos, _bar(high_bid=1.101, close_bid=1.0998), state, _ctx(), acct)
    assert state.trail_level == pytest.approx(1.0995)
    # Bar 3: trail ratchets up to 1.0998
    p.evaluate_at_close(pos, _bar(high_bid=1.1010, close_bid=1.1005), state, _ctx(), acct)
    assert state.trail_level == pytest.approx(1.0998)
    # Bar 4: close drops to 1.0998 → fire
    decision = p.evaluate_at_close(
        pos, _bar(close_bid=1.0998), state, _ctx(), acct
    )
    assert decision is not None


def test_short_mirror() -> None:
    """Short side: activate at low_ask ≤ entry - R_atr. Trail = running min of
    max(prev_close_ask, entry). Capped at entry going UP (worse for short)."""
    p = SlPlusTrailingSwingPolicy()
    state = p.make_state(_ctx(Direction.SHORT))
    pos = _pos(Direction.SHORT)
    acct = Account(starting_balance=100_000.0)
    # Bar 1: low_ask=1.0980 = entry - R_atr → activates. prev_close_ask=1.0985.
    p.evaluate_at_close(pos, _bar(low_ask=1.0980, close_ask=1.0985), state, _ctx(Direction.SHORT), acct)
    assert state.activated
    # Bar 2: candidate = max(1.0985, 1.1000) = 1.1000 (capped at entry). trail = 1.1000.
    # close 1.0990 < 1.1000 → no fire
    p.evaluate_at_close(pos, _bar(low_ask=1.099, close_ask=1.099), state, _ctx(Direction.SHORT), acct)
    assert state.trail_level == pytest.approx(1.1000)
    # Bar 3: close rises to 1.1000 → fires
    decision = p.evaluate_at_close(pos, _bar(close_ask=1.1000), state, _ctx(Direction.SHORT), acct)
    assert decision is not None
    assert decision.action is ExitAction.FULL_CLOSE
