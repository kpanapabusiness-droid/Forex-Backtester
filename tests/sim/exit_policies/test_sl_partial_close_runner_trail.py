"""``sl_partial_close_1r_runner_trail``: intra-bar partial at +1R, at-close
runner trail at peak − R_atr. Reference: scripts/l_arc_10_v3/step_5.py:219-250.
"""

from __future__ import annotations

import pandas as pd
import pytest

from core.sim.account import Account, Direction, Position
from core.sim.exit_policies import (
    ExitAction,
    ExitPolicyContext,
    PartialCloseRunnerTrailState,
    SlPartialClose1RRunnerTrailPolicy,
)


def _pos(direction: Direction = Direction.LONG, entry: float = 1.1000) -> Position:
    return Position(
        position_id=1,
        pair="EURUSD",
        direction=direction,
        entry_time=pd.Timestamp("2026-01-01", tz="UTC"),
        entry_price=entry,
        size=10_000,
    )


def _ctx(direction: Direction = Direction.LONG, entry: float = 1.1000) -> ExitPolicyContext:
    # R_atr = 2.0 * 0.001 = 0.002
    return ExitPolicyContext(
        entry_price=entry, atr_at_entry=0.001, sl_atr_mult=2.0, direction=direction
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


def test_apply_to_order_is_empty() -> None:
    """Partial-close uses state-machine fire, not Order.tp_price."""
    assert SlPartialClose1RRunnerTrailPolicy().apply_to_order(_ctx()) == {}


def test_make_state_initial() -> None:
    state = SlPartialClose1RRunnerTrailPolicy().make_state(_ctx())
    assert isinstance(state, PartialCloseRunnerTrailState)
    assert state.tp1_fired is False
    assert state.tp1_bar_ordinal is None
    assert pd.isna(state.peak_price)
    assert state.bar_ordinal == 0


# ────────────────────────────────────────────────────────────────────────
# intra-bar: +1R partial close fires
# ────────────────────────────────────────────────────────────────────────


def test_intrabar_long_partial_fires_at_one_r() -> None:
    p = SlPartialClose1RRunnerTrailPolicy()
    state = p.make_state(_ctx())
    pos = _pos()
    bar = _bar(high_bid=1.1020)
    d = p.evaluate_intrabar(pos, bar, state, _ctx())
    assert d is not None
    assert d.action is ExitAction.PARTIAL_CLOSE
    assert d.fill_price == pytest.approx(1.1020)  # entry + R_atr
    assert d.partial_fraction == 0.5
    assert d.timing == "intrabar"
    assert state.tp1_fired is True


def test_intrabar_long_does_not_fire_below_one_r() -> None:
    p = SlPartialClose1RRunnerTrailPolicy()
    state = p.make_state(_ctx())
    pos = _pos()
    assert p.evaluate_intrabar(pos, _bar(high_bid=1.1019), state, _ctx()) is None
    assert state.tp1_fired is False


def test_intrabar_long_only_fires_once() -> None:
    p = SlPartialClose1RRunnerTrailPolicy()
    state = p.make_state(_ctx())
    pos = _pos()
    p.evaluate_intrabar(pos, _bar(high_bid=1.1020), state, _ctx())
    # Subsequent bar with high well above +1R → no fire (already done)
    assert p.evaluate_intrabar(pos, _bar(high_bid=1.1100), state, _ctx()) is None


def test_intrabar_short_partial_fires() -> None:
    p = SlPartialClose1RRunnerTrailPolicy()
    state = p.make_state(_ctx(Direction.SHORT))
    pos = _pos(Direction.SHORT)
    bar = _bar(low_ask=1.0980)  # entry - R_atr
    d = p.evaluate_intrabar(pos, bar, state, _ctx(Direction.SHORT))
    assert d is not None
    assert d.fill_price == pytest.approx(1.0980)
    assert state.tp1_fired is True


# ────────────────────────────────────────────────────────────────────────
# at-close: runner-trail
# ────────────────────────────────────────────────────────────────────────


def test_at_close_no_action_before_tp1() -> None:
    """Pre-tp1: at_close only ratchets peak; never fires."""
    p = SlPartialClose1RRunnerTrailPolicy()
    state = p.make_state(_ctx())
    pos = _pos()
    acct = Account(starting_balance=100_000.0)
    # bar with high < +1R, close drops sharply
    bar = _bar(high_bid=1.1010, close_bid=1.0500)
    assert p.evaluate_at_close(pos, bar, state, _ctx(), acct) is None
    # Peak still ratcheted (per reference: peak runs from bar 0)
    assert state.peak_price == pytest.approx(1.1010)


def test_at_close_no_trail_fire_on_tp1_bar() -> None:
    """Reference: i > tp1_i. Same-bar trail-exit forbidden."""
    p = SlPartialClose1RRunnerTrailPolicy()
    state = p.make_state(_ctx())
    pos = _pos()
    acct = Account(starting_balance=100_000.0)
    # Intra-bar: tp1 fires on this bar (high=1.1030, well above +1R=1.1020)
    bar = _bar(high_bid=1.1030, close_bid=1.0900)  # close drops to 1.0900 (below trail)
    p.evaluate_intrabar(pos, bar, state, _ctx())
    assert state.tp1_fired
    assert state.tp1_bar_ordinal == 0
    # At-close on the SAME bar: bar_ordinal = 0 = tp1_bar_ordinal → no fire
    assert p.evaluate_at_close(pos, bar, state, _ctx(), acct) is None
    assert state.bar_ordinal == 1


def test_at_close_trail_fires_on_subsequent_bar() -> None:
    """tp1 on bar 0. Bar 1: peak=1.103 (from bar 0 high). Trail=1.101.
    Close on bar 1 drops to 1.101 → fires.
    """
    p = SlPartialClose1RRunnerTrailPolicy()
    state = p.make_state(_ctx())
    pos = _pos()
    acct = Account(starting_balance=100_000.0)
    # Bar 0: tp1 fires intra-bar; high=1.1030 → peak=1.1030 (at_close)
    bar0 = _bar(high_bid=1.1030, close_bid=1.1025)
    p.evaluate_intrabar(pos, bar0, state, _ctx())
    p.evaluate_at_close(pos, bar0, state, _ctx(), acct)
    assert state.peak_price == pytest.approx(1.1030)
    assert state.bar_ordinal == 1
    # Bar 1: close drops to 1.1010 = trail = peak - R_atr = 1.1030 - 0.002
    bar1 = _bar(high_bid=1.1020, close_bid=1.1010)
    decision = p.evaluate_at_close(pos, bar1, state, _ctx(), acct)
    assert decision is not None
    assert decision.action is ExitAction.FULL_CLOSE
    assert decision.exit_reason == "runner_trail_stop"
    assert decision.timing == "at_close"


def test_at_close_trail_does_not_fire_with_close_above_trail() -> None:
    p = SlPartialClose1RRunnerTrailPolicy()
    state = p.make_state(_ctx())
    pos = _pos()
    acct = Account(starting_balance=100_000.0)
    # Bar 0: tp1
    bar0 = _bar(high_bid=1.1030, close_bid=1.1025)
    p.evaluate_intrabar(pos, bar0, state, _ctx())
    p.evaluate_at_close(pos, bar0, state, _ctx(), acct)
    # Bar 1: close 1.1015 > trail 1.1010 → no fire
    bar1 = _bar(high_bid=1.1020, close_bid=1.1015)
    assert p.evaluate_at_close(pos, bar1, state, _ctx(), acct) is None


def test_at_close_trail_ratchets_with_higher_peak() -> None:
    """Peak goes higher; trail also rises."""
    p = SlPartialClose1RRunnerTrailPolicy()
    state = p.make_state(_ctx())
    pos = _pos()
    acct = Account(starting_balance=100_000.0)
    # Bar 0: tp1, peak=1.103
    p.evaluate_intrabar(pos, _bar(high_bid=1.1030), state, _ctx())
    p.evaluate_at_close(pos, _bar(high_bid=1.1030, close_bid=1.1025), state, _ctx(), acct)
    # Bar 1: higher high 1.106, close 1.1055
    p.evaluate_at_close(pos, _bar(high_bid=1.106, close_bid=1.1055), state, _ctx(), acct)
    assert state.peak_price == pytest.approx(1.106)
    # Bar 2: trail now = 1.106 - 0.002 = 1.104; close 1.104 → fires
    decision = p.evaluate_at_close(pos, _bar(close_bid=1.104), state, _ctx(), acct)
    assert decision is not None


def test_short_at_close_runner_trail() -> None:
    p = SlPartialClose1RRunnerTrailPolicy()
    state = p.make_state(_ctx(Direction.SHORT))
    pos = _pos(Direction.SHORT)
    acct = Account(starting_balance=100_000.0)
    # Bar 0: short tp1 at low_ask 1.0980; trough=1.0980; close 1.0985
    p.evaluate_intrabar(pos, _bar(low_ask=1.0980), state, _ctx(Direction.SHORT))
    p.evaluate_at_close(pos, _bar(low_ask=1.0980, close_ask=1.0985), state, _ctx(Direction.SHORT), acct)
    # Bar 1: trail = trough + R_atr = 1.0980 + 0.002 = 1.1000. close 1.1000 → fires
    decision = p.evaluate_at_close(pos, _bar(close_ask=1.1000), state, _ctx(Direction.SHORT), acct)
    assert decision is not None
    assert decision.action is ExitAction.FULL_CLOSE


# ────────────────────────────────────────────────────────────────────────
# Bar ordinal advancement
# ────────────────────────────────────────────────────────────────────────


def test_bar_ordinal_increments_on_each_at_close() -> None:
    p = SlPartialClose1RRunnerTrailPolicy()
    state = p.make_state(_ctx())
    pos = _pos()
    acct = Account(starting_balance=100_000.0)
    for i in range(5):
        p.evaluate_at_close(pos, _bar(), state, _ctx(), acct)
    assert state.bar_ordinal == 5
