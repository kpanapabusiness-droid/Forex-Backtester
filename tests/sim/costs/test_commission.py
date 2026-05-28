"""Unit tests for core.sim.costs.commission — G4 correctness gate."""

from __future__ import annotations

import pytest

from core.sim.costs.commission import compute_commission_usd


def test_G4_one_lot_round_turn_is_four_dollars():
    """G4 per dispatch §4: closed trade at 1.0 lot → exactly $4 commission."""
    assert compute_commission_usd(1.0) == 4.0


def test_half_lot_is_two_dollars():
    assert compute_commission_usd(0.5) == 2.0


def test_ten_lots_is_forty_dollars():
    """Linear scaling with lot size."""
    assert compute_commission_usd(10.0) == 40.0


def test_zero_lot_is_zero():
    assert compute_commission_usd(0.0) == 0.0


def test_alternate_rate():
    """Caller can override rate (e.g. for 5ers premium account or different broker)."""
    assert compute_commission_usd(1.0, rate_per_lot_rt=6.0) == 6.0


def test_negative_lots_raises():
    with pytest.raises(ValueError, match="must be >= 0"):
        compute_commission_usd(-1.0)


def test_realistic_arc_10_lot_size():
    """At risk_amount=$500, sl_distance_pips=200, pip_value=$10/lot:
       lots = 500/(200*10) = 0.25
       commission = 0.25 * 4 = $1.0
    """
    risk_amount_usd = 500.0
    sl_pips = 200.0
    pip_value_usd_per_lot = 10.0
    lots = risk_amount_usd / (sl_pips * pip_value_usd_per_lot)
    assert lots == 0.25
    assert compute_commission_usd(lots) == 1.0
