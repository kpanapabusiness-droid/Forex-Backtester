"""Unit tests for core.sim.costs.spread_multiplier."""

from __future__ import annotations

import pytest

from core.sim.costs.spread_multiplier import compute_extra_spread_price


def test_mult_1_returns_zero():
    """No widening at baseline multiplier 1.0."""
    assert compute_extra_spread_price(0.0001, 0.00012, 1.0) == 0.0


def test_mult_2_doubles_extra_cost_equals_original():
    """At mult=2, extra = original spread total (× (2-1) = ×1)."""
    se = 0.0001
    sx = 0.00012
    extra = compute_extra_spread_price(se, sx, 2.0)
    assert extra == pytest.approx(0.00022)


def test_mult_4_triples_extra_cost():
    """At mult=4, extra = (entry + exit) × 3."""
    extra = compute_extra_spread_price(0.0001, 0.00012, 4.0)
    assert extra == pytest.approx(0.00022 * 3)


def test_zero_spread_trades_stay_zero():
    """Per dispatch §3.3: zero-spread trades (data gaps) are NOT floored."""
    extra = compute_extra_spread_price(0.0, 0.0, 4.0)
    assert extra == 0.0


def test_one_side_zero_other_positive():
    """If entry spread = 0 but exit spread > 0, only exit half contributes."""
    # spread_total = 0 + 0.00012 = 0.00012
    extra = compute_extra_spread_price(0.0, 0.00012, 3.0)
    # extra = 0.00012 × 2 = 0.00024
    assert extra == pytest.approx(0.00024)


def test_mult_below_one_returns_zero():
    """Multiplier < 1 (tightening spread): treated as 0 — primitive is for stress only."""
    assert compute_extra_spread_price(0.0001, 0.0001, 0.5) == 0.0


def test_realistic_eurusd_spread_at_mult_2():
    """EURUSD: ~0.5 pip spread at entry, ~0.5 pip at exit, mult 2× → 1.0 pip extra cost."""
    pip_size = 0.0001
    se = 0.5 * pip_size  # 0.00005
    sx = 0.5 * pip_size
    extra = compute_extra_spread_price(se, sx, 2.0)
    # extra = (0.00005 + 0.00005) × 1 = 0.0001 (= 1 pip)
    assert extra == pytest.approx(0.0001)
