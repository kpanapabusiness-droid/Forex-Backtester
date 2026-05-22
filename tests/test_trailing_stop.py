"""Tests for core/sim/trailing_stop.py — trailing stop state machine."""

from __future__ import annotations

import pandas as pd
import pytest

from core.sim.account import Account, Direction
from core.sim.trailing_stop import TrailManager, TrailState


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


# ── TrailState mechanics ───────────────────────────────────────────────


def test_trail_state_pre_activation_inactive() -> None:
    s = TrailState(position_id=1, entry_price=1.1000, atr_at_entry=0.0010, current_sl_price=1.0980)
    # Close below activation threshold — no change
    moved = s.update_at_close(1.1010)
    assert moved is False
    assert s.activated is False
    assert s.current_sl_price == 1.0980


def test_trail_state_activates_at_threshold() -> None:
    """Activation: close >= entry + 2.0 × ATR."""
    s = TrailState(position_id=1, entry_price=1.1000, atr_at_entry=0.0010, current_sl_price=1.0980)
    # entry + 2.0 × ATR = 1.1020
    moved = s.update_at_close(1.1020)
    assert s.activated is True
    assert s.highest_close_since_activation == 1.1020
    # Trail proposal: 1.1020 - 1.5 × 0.0010 = 1.1005 > 1.0980 → update
    assert moved is True
    assert s.current_sl_price == 1.1005


def test_trail_state_ratchets_on_new_high() -> None:
    s = TrailState(position_id=1, entry_price=1.1000, atr_at_entry=0.0010, current_sl_price=1.0980)
    s.update_at_close(1.1020)  # activate, sl → 1.1005
    s.update_at_close(1.1050)  # new high → sl → 1.1050 - 0.0015 = 1.1035
    assert s.highest_close_since_activation == 1.1050
    assert s.current_sl_price == 1.1035


def test_trail_state_does_not_ratchet_on_retracement() -> None:
    s = TrailState(position_id=1, entry_price=1.1000, atr_at_entry=0.0010, current_sl_price=1.0980)
    s.update_at_close(1.1050)  # activate + ratchet → sl 1.1035
    sl_before = s.current_sl_price
    s.update_at_close(1.1030)  # retracement, no new high
    assert s.current_sl_price == sl_before
    assert s.highest_close_since_activation == 1.1050


def test_trail_state_activation_below_threshold_no_op() -> None:
    """Activation is gated on close ≥ threshold; equal is allowed."""
    s = TrailState(position_id=1, entry_price=1.1000, atr_at_entry=0.0010, current_sl_price=1.0980)
    # 1.1019 < 1.1020 → no activation
    s.update_at_close(1.1019)
    assert s.activated is False


def test_trail_state_never_lowers_sl() -> None:
    """If a proposed trail is BELOW the current SL, current SL holds."""
    # Edge: entry SL is set very tight; the first activation move proposes
    # a trail that's BELOW current SL. Trail must not lower SL.
    s = TrailState(position_id=1, entry_price=1.1000, atr_at_entry=0.0010, current_sl_price=1.1010)
    s.update_at_close(1.1020)  # activate; proposed trail = 1.1005 < 1.1010
    assert s.current_sl_price == 1.1010  # unchanged


# ── TrailManager + Account integration ─────────────────────────────────


def test_trail_manager_register_long() -> None:
    acct = Account(starting_balance=100_000.0)
    pos = acct.open(
        pair="EURUSD",
        direction=Direction.LONG,
        entry_time=_ts("2026-01-01"),
        entry_price=1.1000,
        size=10_000,
        sl_price=1.0980,
    )
    tm = TrailManager()
    state = tm.register(pos, atr_at_entry=0.0010)
    assert tm.get(pos.position_id) is state
    assert state.current_sl_price == 1.0980


def test_trail_manager_rejects_short() -> None:
    acct = Account(starting_balance=100_000.0)
    pos = acct.open(
        pair="EURUSD",
        direction=Direction.SHORT,
        entry_time=_ts("2026-01-01"),
        entry_price=1.1000,
        size=10_000,
        sl_price=1.1020,
    )
    tm = TrailManager()
    with pytest.raises(NotImplementedError, match="long-only"):
        tm.register(pos, atr_at_entry=0.0010)


def test_trail_manager_requires_sl() -> None:
    acct = Account(starting_balance=100_000.0)
    pos = acct.open(
        pair="EURUSD",
        direction=Direction.LONG,
        entry_time=_ts("2026-01-01"),
        entry_price=1.1000,
        size=10_000,
        sl_price=None,
    )
    tm = TrailManager()
    with pytest.raises(ValueError, match="starting fixed SL"):
        tm.register(pos, atr_at_entry=0.0010)


def test_trail_manager_deregister_silent_on_missing() -> None:
    tm = TrailManager()
    tm.deregister(999)  # no-op, must not raise


def test_trail_manager_effective_sl_fallback() -> None:
    """When no trail registered, ``effective_sl`` returns position.sl_price."""
    acct = Account(starting_balance=100_000.0)
    pos = acct.open(
        pair="EURUSD",
        direction=Direction.LONG,
        entry_time=_ts("2026-01-01"),
        entry_price=1.1,
        size=1,
        sl_price=1.095,
    )
    tm = TrailManager()
    assert tm.effective_sl(pos) == 1.095


def test_trail_manager_effective_sl_uses_trail() -> None:
    acct = Account(starting_balance=100_000.0)
    pos = acct.open(
        pair="EURUSD",
        direction=Direction.LONG,
        entry_time=_ts("2026-01-01"),
        entry_price=1.1,
        size=1,
        sl_price=1.095,
    )
    tm = TrailManager()
    state = tm.register(pos, atr_at_entry=0.001)
    state.update_at_close(1.105)  # activate + ratchet
    assert tm.effective_sl(pos) == pytest.approx(state.current_sl_price)
    assert state.current_sl_price > 1.095  # trail lifted SL


def test_trail_manager_update_all_at_close_uses_close_mid() -> None:
    """update_all_at_close reads (close_bid + close_ask) / 2 from snapshot."""
    acct = Account(starting_balance=100_000.0)
    pos = acct.open(
        pair="EURUSD",
        direction=Direction.LONG,
        entry_time=_ts("2026-01-01"),
        entry_price=1.1,
        size=1,
        sl_price=1.095,
    )
    tm = TrailManager()
    tm.register(pos, atr_at_entry=0.001)

    bar = pd.Series(
        {
            "open_bid": 1.099,
            "high_bid": 1.105,
            "low_bid": 1.099,
            "close_bid": 1.1025,
            "open_ask": 1.0995,
            "high_ask": 1.1055,
            "low_ask": 1.0995,
            "close_ask": 1.1035,
        }
    )
    # mid_close = (1.1025 + 1.1035) / 2 = 1.103 (rounded). With entry 1.1,
    # ATR 0.001 → activation threshold 1.102. 1.103 ≥ 1.102 → activates.
    updates = tm.update_all_at_close({"EURUSD": bar}, acct)
    assert pos.position_id in updates
    state = tm.get(pos.position_id)
    assert state.activated is True
