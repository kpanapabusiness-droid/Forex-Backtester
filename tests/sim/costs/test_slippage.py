"""Unit tests for core.sim.costs.slippage."""

from __future__ import annotations

import pytest

from core.sim.costs.slippage import compute_slippage_pips


def test_tp1_hit_three_fills():
    """TP1 hit → 3 fills (entry + TP1 partial + final exit)."""
    total, n = compute_slippage_pips(0.5, tp1_hit=True)
    assert n == 3
    assert total == pytest.approx(1.5)


def test_no_tp1_two_fills():
    """TP1 NOT hit → 2 fills (entry + final exit only)."""
    total, n = compute_slippage_pips(0.5, tp1_hit=False)
    assert n == 2
    assert total == pytest.approx(1.0)


def test_zero_slip_zero_total():
    total, n = compute_slippage_pips(0.0, tp1_hit=True)
    assert n == 3
    assert total == 0.0


def test_one_pip_slip_tp1_hit():
    """1 pip/fill × 3 fills = 3 pips total adverse."""
    total, n = compute_slippage_pips(1.0, tp1_hit=True)
    assert n == 3
    assert total == 3.0


def test_negative_slip_raises():
    with pytest.raises(ValueError, match="must be >= 0"):
        compute_slippage_pips(-0.5, tp1_hit=True)
