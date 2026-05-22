"""Tests for core/sim/account.py — Account state + ExposureRules + Position."""

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


def test_direction_sign() -> None:
    assert Direction.LONG.sign == 1
    assert Direction.SHORT.sign == -1


def test_position_pnl_long_positive() -> None:
    p = Position(1, "EURUSD", Direction.LONG, _ts("2026-01-01"), 1.1000, 100_000)
    assert p.pnl_at(1.1010) == pytest.approx(0.001 * 100_000)


def test_position_pnl_short_positive_on_drop() -> None:
    p = Position(1, "EURUSD", Direction.SHORT, _ts("2026-01-01"), 1.1000, 100_000)
    assert p.pnl_at(1.0990) == pytest.approx(0.001 * 100_000)


def test_position_currencies() -> None:
    p = Position(1, "EURUSD", Direction.LONG, _ts("2026-01-01"), 1.1, 1.0)
    assert p.base_currency == "EUR"
    assert p.quote_currency == "USD"


def test_account_open_close_realises_pnl() -> None:
    acct = Account(starting_balance=100_000.0)
    pos = acct.open(
        pair="EURUSD",
        direction=Direction.LONG,
        entry_time=_ts("2026-01-01 10:00"),
        entry_price=1.1000,
        size=10_000,
        sl_price=1.0980,
        tp_price=1.1050,
    )
    assert acct.balance == 100_000.0  # unrealised, not banked
    assert len(acct.open_positions) == 1

    trade = acct.close(pos.position_id, _ts("2026-01-01 14:00"), 1.1020, "market")
    assert isinstance(trade, ClosedTrade)
    assert trade.pnl == pytest.approx(0.002 * 10_000)
    assert acct.balance == pytest.approx(100_000.0 + 0.002 * 10_000)
    assert len(acct.open_positions) == 0
    assert len(acct.closed_trades) == 1


def test_account_close_unknown_id_raises() -> None:
    acct = Account(starting_balance=100_000.0)
    with pytest.raises(KeyError, match="No open position"):
        acct.close(999, _ts("2026-01-01"), 1.0, "market")


def test_account_mark_to_market_updates_equity_and_dd() -> None:
    acct = Account(starting_balance=100_000.0)
    acct.open(
        pair="EURUSD",
        direction=Direction.LONG,
        entry_time=_ts("2026-01-01 10:00"),
        entry_price=1.1000,
        size=100_000,  # 1 lot
    )

    # First mark — slightly under water
    eq1 = acct.mark_to_market(_ts("2026-01-01 11:00"), {"EURUSD": 1.0990})
    assert eq1 == pytest.approx(100_000.0 - 0.001 * 100_000)
    assert acct.max_drawdown_pct == pytest.approx(0.001)

    # Recover above entry
    eq2 = acct.mark_to_market(_ts("2026-01-01 12:00"), {"EURUSD": 1.1010})
    assert eq2 == pytest.approx(100_000.0 + 0.001 * 100_000)
    # Peak now higher; max DD still set from previous low
    assert acct.max_drawdown_pct == pytest.approx(0.001)


def test_account_equity_curve_returns_indexed_series() -> None:
    acct = Account(starting_balance=100_000.0)
    acct.mark_to_market(_ts("2026-01-01"), {})
    acct.mark_to_market(_ts("2026-01-02"), {})
    eq = acct.equity_curve()
    assert len(eq) == 2
    assert eq.name == "equity"
    assert str(eq.index.tz) == "UTC"


def test_exposure_rules_default_caps_only_per_pair() -> None:
    """Default rules: total/per-currency uncapped, per-pair=1."""
    acct = Account(starting_balance=100_000.0)
    assert acct.exposure_check("EURUSD") is True
    acct.open("EURUSD", Direction.LONG, _ts("2026-01-01"), 1.1, 1)
    assert acct.exposure_check("EURUSD") is False  # per-pair cap
    assert acct.exposure_check("GBPUSD") is True  # other pair allowed


def test_exposure_max_concurrent_total() -> None:
    acct = Account(
        starting_balance=100_000.0,
        exposure=ExposureRules(max_concurrent_total=2, max_concurrent_per_pair=10),
    )
    acct.open("EURUSD", Direction.LONG, _ts("2026-01-01"), 1.1, 1)
    acct.open("GBPUSD", Direction.LONG, _ts("2026-01-01"), 1.3, 1)
    assert acct.exposure_check("USDJPY") is False  # total cap would be exceeded
    assert acct.exposure_check("EURUSD") is False  # total cap dominates even though per-pair allows
    # Drop one back below the cap → opens up again
    pos_id = next(iter(acct._open))  # noqa: SLF001
    acct.close(pos_id, _ts("2026-01-02"), 1.1, "market")
    assert acct.exposure_check("USDJPY") is True


def test_exposure_max_concurrent_per_currency() -> None:
    """USD exposure should cap at the per-currency limit."""
    acct = Account(
        starting_balance=100_000.0,
        exposure=ExposureRules(
            max_concurrent_total=10,
            max_concurrent_per_pair=10,
            max_concurrent_per_currency=2,
        ),
    )
    acct.open("EURUSD", Direction.LONG, _ts("2026-01-01"), 1.1, 1)  # USD count = 1
    acct.open("GBPUSD", Direction.LONG, _ts("2026-01-01"), 1.3, 1)  # USD count = 2
    # AUDUSD would push USD to 3 → blocked
    assert acct.exposure_check("AUDUSD") is False
    # NZDCAD has no USD → allowed
    assert acct.exposure_check("NZDCAD") is True


def test_exposure_per_currency_counts_base_and_quote() -> None:
    acct = Account(
        starting_balance=100_000.0,
        exposure=ExposureRules(
            max_concurrent_total=10,
            max_concurrent_per_pair=10,
            max_concurrent_per_currency=1,
        ),
    )
    acct.open("EURUSD", Direction.LONG, _ts("2026-01-01"), 1.1, 1)
    # USD touched once; EUR touched once.
    # GBPJPY shares nothing → allowed
    assert acct.exposure_check("GBPJPY") is True
    # USDJPY shares USD → blocked
    assert acct.exposure_check("USDJPY") is False
    # EURGBP shares EUR → blocked
    assert acct.exposure_check("EURGBP") is False


def test_exposure_uncapped_when_all_none() -> None:
    acct = Account(
        starting_balance=100_000.0,
        exposure=ExposureRules(
            max_concurrent_total=None,
            max_concurrent_per_pair=None,
            max_concurrent_per_currency=None,
        ),
    )
    acct.open("EURUSD", Direction.LONG, _ts("2026-01-01"), 1.1, 1)
    acct.open("EURUSD", Direction.LONG, _ts("2026-01-02"), 1.1, 1)
    assert acct.exposure_check("EURUSD") is True


def test_open_positions_for_filters_by_pair() -> None:
    acct = Account(
        starting_balance=100_000.0,
        exposure=ExposureRules(max_concurrent_per_pair=None),
    )
    acct.open("EURUSD", Direction.LONG, _ts("2026-01-01"), 1.1, 1)
    acct.open("GBPUSD", Direction.LONG, _ts("2026-01-01"), 1.3, 1)
    acct.open("EURUSD", Direction.SHORT, _ts("2026-01-02"), 1.1, 1)
    eurs = list(acct.open_positions_for("EURUSD"))
    assert len(eurs) == 2
    assert all(p.pair == "EURUSD" for p in eurs)
