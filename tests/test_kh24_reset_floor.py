"""Tests for core/sim/risk/reset_floor.py."""

from __future__ import annotations

import pandas as pd
import pytest

from core.sim.risk.reset_floor import ResetFloorAccount


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


def test_floor_starts_at_starting_balance() -> None:
    f = ResetFloorAccount(starting_balance=100_000.0)
    assert f.floor == 100_000.0


def test_floor_ratchets_on_winning_day() -> None:
    f = ResetFloorAccount(starting_balance=100_000.0)
    moved = f.update_at_day_close(_ts("2026-01-01"), 101_500.0)
    assert moved is True
    assert f.floor == 101_500.0


def test_floor_does_not_retreat_on_losing_day() -> None:
    f = ResetFloorAccount(starting_balance=100_000.0)
    f.update_at_day_close(_ts("2026-01-01"), 102_000.0)
    moved = f.update_at_day_close(_ts("2026-01-02"), 99_500.0)
    assert moved is False
    assert f.floor == 102_000.0


def test_floor_idempotent_within_day() -> None:
    """Subsequent same-day calls are ignored (only the first updates)."""
    f = ResetFloorAccount(starting_balance=100_000.0)
    f.update_at_day_close(_ts("2026-01-01 00:00:00"), 101_000.0)
    moved_again = f.update_at_day_close(_ts("2026-01-01 12:00:00"), 105_000.0)
    assert moved_again is False
    assert f.floor == 101_000.0  # NOT 105_000


def test_floor_advances_across_multiple_days() -> None:
    f = ResetFloorAccount(starting_balance=100_000.0)
    f.update_at_day_close(_ts("2026-01-01"), 101_000.0)
    f.update_at_day_close(_ts("2026-01-02"), 103_000.0)
    f.update_at_day_close(_ts("2026-01-03"), 102_500.0)  # below day-2 floor
    assert f.floor == 103_000.0


def test_risk_size_uses_floor_not_balance() -> None:
    f = ResetFloorAccount(starting_balance=100_000.0, risk_pct=0.01)
    f.update_at_day_close(_ts("2026-01-01"), 110_000.0)
    # Floor = 110_000; risk_pct = 1% → 1100 risk
    # SL distance = 0.002 → size = 1100 / 0.002 = 550_000
    size = f.risk_size(entry_price=1.1000, sl_price=1.0980)
    assert size == pytest.approx(550_000.0)


def test_risk_size_zero_distance_raises() -> None:
    f = ResetFloorAccount(starting_balance=100_000.0)
    with pytest.raises(ValueError, match="zero SL distance"):
        f.risk_size(entry_price=1.1, sl_price=1.1)


def test_risk_size_per_call_override() -> None:
    """The risk_pct kwarg overrides the instance default."""
    f = ResetFloorAccount(starting_balance=100_000.0, risk_pct=0.01)
    # 0.5% L_arc convention
    size = f.risk_size(entry_price=1.1, sl_price=1.098, risk_pct=0.005)
    # Floor = 100_000; risk = 500; SL distance = 0.002 → 250_000
    assert size == pytest.approx(250_000.0)
