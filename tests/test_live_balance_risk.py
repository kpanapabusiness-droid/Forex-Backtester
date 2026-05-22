"""Tests for core/sim/risk/live_balance.py — PR-E.1.6 sizing fix."""

from __future__ import annotations

import pandas as pd
import pytest

from core.sim.account import Account, Direction
from core.sim.risk.live_balance import LiveBalanceRisk


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


def test_size_proportional_to_live_balance() -> None:
    """1% of $100k on a 0.002 SL → 500_000 units."""
    acct = Account(starting_balance=100_000.0)
    r = LiveBalanceRisk(risk_pct=0.01)
    size = r.risk_size(acct, entry_price=1.1000, sl_price=1.0980)
    assert size == pytest.approx(500_000.0)


def test_size_compounds_with_realised_pnl() -> None:
    """Realising +10k profit lifts size on the next trade."""
    acct = Account(starting_balance=100_000.0)
    r = LiveBalanceRisk(risk_pct=0.01)
    # Open + close a profitable trade to bump balance from 100k to 110k
    pos = acct.open(
        pair="EURUSD",
        direction=Direction.LONG,
        entry_time=_ts("2026-01-01"),
        entry_price=1.1000,
        size=1_000_000,
        sl_price=1.0980,
    )
    acct.close(pos.position_id, _ts("2026-01-02"), exit_price=1.1100, exit_reason="market")
    assert acct.balance == pytest.approx(110_000.0)
    # Next trade sizes off the lifted balance
    size = r.risk_size(acct, entry_price=1.1000, sl_price=1.0980)
    assert size == pytest.approx(110_000.0 * 0.01 / 0.002)  # 550_000


def test_size_shrinks_after_loss() -> None:
    """Realising a loss shrinks size on the next trade (compounds both ways)."""
    acct = Account(starting_balance=100_000.0)
    r = LiveBalanceRisk(risk_pct=0.01)
    pos = acct.open(
        pair="EURUSD",
        direction=Direction.LONG,
        entry_time=_ts("2026-01-01"),
        entry_price=1.1000,
        size=1_000_000,
        sl_price=1.0980,
    )
    acct.close(pos.position_id, _ts("2026-01-02"), exit_price=1.0980, exit_reason="stop_loss")
    # Lost 0.002 × 1_000_000 = 2_000 → balance = 98_000
    assert acct.balance == pytest.approx(98_000.0)
    size = r.risk_size(acct, entry_price=1.1000, sl_price=1.0980)
    assert size == pytest.approx(98_000.0 * 0.01 / 0.002)  # 490_000


def test_zero_sl_distance_raises() -> None:
    acct = Account(starting_balance=100_000.0)
    r = LiveBalanceRisk()
    with pytest.raises(ValueError, match="zero SL distance"):
        r.risk_size(acct, entry_price=1.1, sl_price=1.1)


def test_per_call_risk_pct_override() -> None:
    acct = Account(starting_balance=100_000.0)
    r = LiveBalanceRisk(risk_pct=0.01)
    size = r.risk_size(acct, entry_price=1.1, sl_price=1.098, risk_pct=0.005)
    assert size == pytest.approx(250_000.0)
