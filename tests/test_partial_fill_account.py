"""Tests for Account partial-fill primitives — partial_close(),
current_size_of(), mark-to-market and exposure semantics with reduced
positions, multi-leg ClosedTrade linkage via parent_position_id.

Anchor: dispatch §"Task 2" / intent doc §6 (Account refactor). Position
stays frozen; Account owns the shadow ``_current_sizes`` dict.

These tests do not exercise any ExitPolicy / ExitPolicyManager wiring —
those are covered by Tasks 3-5 + their test files. Here we are only
proving the Account-level mechanics.
"""

from __future__ import annotations

import pandas as pd
import pytest

from core.sim.account import (
    Account,
    ClosedTrade,
    Direction,
    ExposureRules,
    Position,
)


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


# ────────────────────────────────────────────────────────────────────────
# ClosedTrade schema
# ────────────────────────────────────────────────────────────────────────


def test_closed_trade_parent_position_id_defaults_none() -> None:
    """Backwards-compat: existing callers passing positional args (no
    parent_position_id) get None — standalone full close.
    """
    ct = ClosedTrade(
        position_id=1,
        pair="EURUSD",
        direction=Direction.LONG,
        entry_time=_ts("2026-01-01"),
        entry_price=1.10,
        exit_time=_ts("2026-01-02"),
        exit_price=1.11,
        size=10_000,
        pnl=100.0,
        exit_reason="market",
    )
    assert ct.parent_position_id is None


# ────────────────────────────────────────────────────────────────────────
# current_size_of()
# ────────────────────────────────────────────────────────────────────────


def test_current_size_of_returns_original_when_no_partial() -> None:
    acct = Account(starting_balance=100_000.0)
    pos = acct.open(
        pair="EURUSD",
        direction=Direction.LONG,
        entry_time=_ts("2026-01-01"),
        entry_price=1.10,
        size=10_000,
    )
    assert acct.current_size_of(pos.position_id) == 10_000


def test_current_size_of_raises_on_unknown_id() -> None:
    acct = Account(starting_balance=100_000.0)
    with pytest.raises(KeyError, match="No open position"):
        acct.current_size_of(999)


def test_current_size_of_raises_after_full_close() -> None:
    acct = Account(starting_balance=100_000.0)
    pos = acct.open("EURUSD", Direction.LONG, _ts("2026-01-01"), 1.10, 10_000)
    acct.close(pos.position_id, _ts("2026-01-02"), 1.11, "market")
    with pytest.raises(KeyError):
        acct.current_size_of(pos.position_id)


# ────────────────────────────────────────────────────────────────────────
# partial_close() — happy path
# ────────────────────────────────────────────────────────────────────────


def test_partial_close_realises_partial_pnl_long() -> None:
    acct = Account(starting_balance=100_000.0)
    pos = acct.open("EURUSD", Direction.LONG, _ts("2026-01-01"), 1.1000, 10_000)
    trade = acct.partial_close(
        pos.position_id,
        exit_time=_ts("2026-01-02"),
        exit_price=1.1020,
        exit_reason="partial_close_1r",
        size_to_close=5_000,
    )
    # PnL on 5k units @ +20 pips = 0.002 * 5000 = 10
    assert trade.pnl == pytest.approx(0.002 * 5_000)
    assert trade.size == 5_000
    assert trade.parent_position_id == pos.position_id
    # Balance updated for partial
    assert acct.balance == pytest.approx(100_000.0 + 0.002 * 5_000)
    # Position still open
    assert pos.position_id in {p.position_id for p in acct.open_positions}
    # Live size reduced
    assert acct.current_size_of(pos.position_id) == 5_000


def test_partial_close_realises_partial_pnl_short() -> None:
    acct = Account(starting_balance=100_000.0)
    pos = acct.open("EURUSD", Direction.SHORT, _ts("2026-01-01"), 1.1000, 10_000)
    trade = acct.partial_close(
        pos.position_id,
        exit_time=_ts("2026-01-02"),
        exit_price=1.0980,
        exit_reason="partial_close_1r",
        size_to_close=5_000,
    )
    # Short, price dropped 20 pips → +PnL on 5k units
    assert trade.pnl == pytest.approx(0.002 * 5_000)
    assert acct.current_size_of(pos.position_id) == 5_000


def test_two_sequential_partials() -> None:
    acct = Account(starting_balance=100_000.0)
    pos = acct.open("EURUSD", Direction.LONG, _ts("2026-01-01"), 1.1000, 10_000)
    acct.partial_close(pos.position_id, _ts("2026-01-02"), 1.1010, "p1", 3_000)
    acct.partial_close(pos.position_id, _ts("2026-01-03"), 1.1020, "p2", 4_000)
    assert acct.current_size_of(pos.position_id) == 3_000
    # Cumulative balance: 3000 * 0.001 + 4000 * 0.002 = 3 + 8 = 11
    assert acct.balance == pytest.approx(100_000.0 + 3.0 + 8.0)
    # Both legs recorded; both linked
    closed = acct.closed_trades
    assert len(closed) == 2
    assert all(ct.parent_position_id == pos.position_id for ct in closed)


def test_final_close_after_partial_uses_remaining_size() -> None:
    acct = Account(starting_balance=100_000.0)
    pos = acct.open("EURUSD", Direction.LONG, _ts("2026-01-01"), 1.1000, 10_000)
    acct.partial_close(pos.position_id, _ts("2026-01-02"), 1.1020, "partial", 5_000)
    final = acct.close(pos.position_id, _ts("2026-01-03"), 1.1050, "trail")
    # Final leg sized at 5000 (remainder); +50 pips on 5k = 0.005 * 5000 = 25
    assert final.size == 5_000
    assert final.pnl == pytest.approx(0.005 * 5_000)
    # Final leg also tagged as parent_position_id (this is a multi-leg close)
    assert final.parent_position_id == pos.position_id


def test_full_close_no_prior_partial_has_none_parent() -> None:
    """Backwards-compat: positions that never had a partial close get
    ``parent_position_id=None`` on the full-close ClosedTrade.
    """
    acct = Account(starting_balance=100_000.0)
    pos = acct.open("EURUSD", Direction.LONG, _ts("2026-01-01"), 1.10, 10_000)
    ct = acct.close(pos.position_id, _ts("2026-01-02"), 1.11, "market")
    assert ct.parent_position_id is None


# ────────────────────────────────────────────────────────────────────────
# partial_close() — validation
# ────────────────────────────────────────────────────────────────────────


def test_partial_close_unknown_id_raises() -> None:
    acct = Account(starting_balance=100_000.0)
    with pytest.raises(KeyError, match="No open position"):
        acct.partial_close(999, _ts("2026-01-01"), 1.0, "x", 100)


def test_partial_close_zero_size_raises() -> None:
    acct = Account(starting_balance=100_000.0)
    pos = acct.open("EURUSD", Direction.LONG, _ts("2026-01-01"), 1.10, 10_000)
    with pytest.raises(ValueError, match="size_to_close must be > 0"):
        acct.partial_close(pos.position_id, _ts("2026-01-02"), 1.11, "x", 0)


def test_partial_close_negative_size_raises() -> None:
    acct = Account(starting_balance=100_000.0)
    pos = acct.open("EURUSD", Direction.LONG, _ts("2026-01-01"), 1.10, 10_000)
    with pytest.raises(ValueError, match="size_to_close must be > 0"):
        acct.partial_close(pos.position_id, _ts("2026-01-02"), 1.11, "x", -100)


def test_partial_close_size_equal_to_current_raises() -> None:
    """Equal-to-current is a full close; caller should use close()."""
    acct = Account(starting_balance=100_000.0)
    pos = acct.open("EURUSD", Direction.LONG, _ts("2026-01-01"), 1.10, 10_000)
    with pytest.raises(ValueError, match="use close\\(\\) for full-out"):
        acct.partial_close(pos.position_id, _ts("2026-01-02"), 1.11, "x", 10_000)


def test_partial_close_size_exceeds_current_raises() -> None:
    acct = Account(starting_balance=100_000.0)
    pos = acct.open("EURUSD", Direction.LONG, _ts("2026-01-01"), 1.10, 10_000)
    with pytest.raises(ValueError, match="use close\\(\\) for full-out"):
        acct.partial_close(pos.position_id, _ts("2026-01-02"), 1.11, "x", 20_000)


def test_partial_close_size_equal_to_remaining_after_prior_partial_raises() -> None:
    """After a partial, equal-to-remaining is still a full close."""
    acct = Account(starting_balance=100_000.0)
    pos = acct.open("EURUSD", Direction.LONG, _ts("2026-01-01"), 1.10, 10_000)
    acct.partial_close(pos.position_id, _ts("2026-01-02"), 1.11, "p1", 4_000)
    # remaining = 6000; partial 6000 should reject
    with pytest.raises(ValueError, match="use close\\(\\) for full-out"):
        acct.partial_close(pos.position_id, _ts("2026-01-03"), 1.12, "p2", 6_000)


# ────────────────────────────────────────────────────────────────────────
# mark_to_market with partial state
# ────────────────────────────────────────────────────────────────────────


def test_mark_to_market_uses_current_size_after_partial() -> None:
    acct = Account(starting_balance=100_000.0)
    pos = acct.open("EURUSD", Direction.LONG, _ts("2026-01-01"), 1.1000, 100_000)
    acct.partial_close(pos.position_id, _ts("2026-01-02"), 1.1020, "p1", 50_000)
    # Realised: 0.002 * 50000 = 100. Balance now 100_100.
    # Mark @ 1.1030 → unrealised on remaining 50k = 0.003 * 50000 = 150
    eq = acct.mark_to_market(_ts("2026-01-02 12:00"), {"EURUSD": 1.1030})
    assert eq == pytest.approx(100_000.0 + 100.0 + 150.0)


def test_mark_to_market_uses_current_size_after_partial_short() -> None:
    acct = Account(starting_balance=100_000.0)
    pos = acct.open("EURUSD", Direction.SHORT, _ts("2026-01-01"), 1.1000, 100_000)
    acct.partial_close(pos.position_id, _ts("2026-01-02"), 1.0980, "p1", 60_000)
    # Realised: 0.002 * 60000 = 120 (short profits on drop). Balance 100_120.
    # Mark @ 1.0970 → unrealised on remaining 40k = (1.10 - 1.097) * 40000 = 120
    eq = acct.mark_to_market(_ts("2026-01-02 12:00"), {"EURUSD": 1.0970})
    assert eq == pytest.approx(100_000.0 + 120.0 + 120.0)


# ────────────────────────────────────────────────────────────────────────
# exposure-cap semantics: position counts as 1 until fully closed
# ────────────────────────────────────────────────────────────────────────


def test_exposure_cap_counts_position_as_one_after_partial() -> None:
    """A partial-closed position still occupies its exposure-cap slot."""
    acct = Account(
        starting_balance=100_000.0,
        exposure=ExposureRules(max_concurrent_total=1, max_concurrent_per_pair=1),
    )
    pos = acct.open("EURUSD", Direction.LONG, _ts("2026-01-01"), 1.10, 10_000)
    assert acct.exposure_check("GBPUSD") is False  # total cap = 1, full
    acct.partial_close(pos.position_id, _ts("2026-01-02"), 1.11, "p1", 5_000)
    # Position still open → exposure cap still binding
    assert acct.exposure_check("GBPUSD") is False
    assert acct.exposure_check("EURUSD") is False
    # Full close frees the slot
    acct.close(pos.position_id, _ts("2026-01-03"), 1.12, "trail")
    assert acct.exposure_check("GBPUSD") is True
    assert acct.exposure_check("EURUSD") is True


def test_per_currency_cap_unchanged_by_partial() -> None:
    acct = Account(
        starting_balance=100_000.0,
        exposure=ExposureRules(
            max_concurrent_total=10,
            max_concurrent_per_pair=10,
            max_concurrent_per_currency=1,
        ),
    )
    pos = acct.open("EURUSD", Direction.LONG, _ts("2026-01-01"), 1.10, 10_000)
    acct.partial_close(pos.position_id, _ts("2026-01-02"), 1.11, "p1", 5_000)
    # USD count still 1 → USDJPY blocked
    assert acct.exposure_check("USDJPY") is False


# ────────────────────────────────────────────────────────────────────────
# trade log linkage / grouping
# ────────────────────────────────────────────────────────────────────────


def test_trade_log_groupby_position_id_reconstructs_multi_leg() -> None:
    acct = Account(starting_balance=100_000.0)
    pos1 = acct.open("EURUSD", Direction.LONG, _ts("2026-01-01"), 1.10, 10_000)
    acct.partial_close(pos1.position_id, _ts("2026-01-02"), 1.11, "partial_1r", 5_000)
    acct.close(pos1.position_id, _ts("2026-01-03"), 1.12, "runner_trail")
    # Independent, never-partialled position
    pos2 = acct.open("GBPUSD", Direction.LONG, _ts("2026-01-04"), 1.30, 10_000)
    acct.close(pos2.position_id, _ts("2026-01-05"), 1.31, "market")

    closed = acct.closed_trades
    assert len(closed) == 3
    # pos1 legs both tagged with parent_position_id
    p1_legs = [ct for ct in closed if ct.position_id == pos1.position_id]
    assert len(p1_legs) == 2
    assert all(ct.parent_position_id == pos1.position_id for ct in p1_legs)
    # pos2 standalone close: parent_position_id None
    p2_legs = [ct for ct in closed if ct.position_id == pos2.position_id]
    assert len(p2_legs) == 1
    assert p2_legs[0].parent_position_id is None
    # Filter for whole-position closes only
    standalone = [ct for ct in closed if ct.parent_position_id is None]
    assert len(standalone) == 1
    assert standalone[0].position_id == pos2.position_id


def test_partial_history_cleared_on_full_close() -> None:
    """A re-used position_id (in practice can't happen — _next_position_id
    is monotone) wouldn't leak partial state. Verifies internal cleanup.
    """
    acct = Account(starting_balance=100_000.0)
    pos = acct.open("EURUSD", Direction.LONG, _ts("2026-01-01"), 1.10, 10_000)
    acct.partial_close(pos.position_id, _ts("2026-01-02"), 1.11, "p1", 5_000)
    acct.close(pos.position_id, _ts("2026-01-03"), 1.12, "trail")
    # noqa: SLF001 — internal-state assertions are the point of this test
    assert pos.position_id not in acct._current_sizes  # noqa: SLF001
    assert pos.position_id not in acct._partial_history  # noqa: SLF001


# ────────────────────────────────────────────────────────────────────────
# Position frozen invariant (anchor — must stay frozen)
# ────────────────────────────────────────────────────────────────────────


def test_position_stays_frozen() -> None:
    """Position immutability is a contract. Partial-close must not mutate it."""
    p = Position(1, "EURUSD", Direction.LONG, _ts("2026-01-01"), 1.10, 10_000)
    with pytest.raises((AttributeError, Exception)):
        p.size = 5_000  # type: ignore[misc]
