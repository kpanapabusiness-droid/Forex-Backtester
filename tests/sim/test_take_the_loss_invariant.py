"""Locked regression: the take-the-loss invariant for the SOLE gate engine.

The SL-honest ``MultiPairBacktester`` is the only engine that scores a
trade (the fast replay was retired 2026-06-02 — see RESET_MANIFEST.md and
``docs/ARC_10_GATE_FIDELITY_DEFECT.md``). This file pins the invariant that
the retired replay violated:

    Any stop breach AT or BEFORE the +1R partial bar resolves to -1R
    (full position). On the SAME bar a stop breach and the +1R partial
    both would fire, the STOP wins (SL-first) and the partial never fires.
    Ambiguity never resolves to a win.

These tests intentionally use only numpy/pandas so they run in the
minimal CI environment (``pytest -m "not research"``).
"""

from __future__ import annotations

import pandas as pd
import pytest

from core.sim.account import Account, Direction, ExposureRules
from core.sim.exit_policy_manager import ExitPolicyManager
from core.sim.multipair_backtester import MultiPairBacktester, Order, StrategyFn
from core.sim.panel import Panel

# Entry fills at 1.100; atr=0.001, sl_mult=2.0 -> 1R = 0.002 in price.
# SL = entry - 1R = 1.098; +1R partial level = entry + 1R = 1.102.
_ENTRY = 1.100
_SL = 1.098
_R = 0.002


def _make_panel(rows: list[dict]) -> Panel:
    df = pd.DataFrame(rows)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df = df.set_index("timestamp_utc")
    return Panel.from_frames({"EURUSD": df}, tf="H1")


def _bar(t: str, *, o: float, h: float, lo: float, c: float) -> dict:
    return {
        "timestamp_utc": t,
        "open_bid": o, "open_ask": o,
        "high_bid": h, "high_ask": h,
        "low_bid": lo, "low_ask": lo,
        "close_bid": c, "close_ask": c,
        "spread_close": 0.0,
        "bid_ask_data_quality": "ok",
    }


def _one_shot_long(t_target: str, **order_kwargs) -> StrategyFn:
    target = pd.Timestamp(t_target, tz="UTC")
    emitted = {"done": False}

    def strategy(t, snapshot, account):
        if emitted["done"] or t != target:
            return []
        emitted["done"] = True
        return [Order(pair="EURUSD", direction=Direction.LONG, size=10_000.0, **order_kwargs)]

    return strategy


def _run(rows: list[dict]):
    panel = _make_panel(rows)
    acct = Account(
        starting_balance=100_000.0,
        exposure=ExposureRules(max_concurrent_per_pair=1),
    )
    strategy = _one_shot_long(
        "2026-01-01 00:00",
        sl_price=_SL,
        atr_at_entry=0.001,
        sl_atr_mult=2.0,
        exit_policy="sl_partial_close_1r_runner_trail",
    )
    bt = MultiPairBacktester(
        panel=panel, account=acct, strategy=strategy,
        exit_policy_manager=ExitPolicyManager(),
    )
    return bt.run()


def _assert_single_minus_1r_stop(result, *, exit_bar: str) -> None:
    legs = list(result.closed_trades)
    assert len(legs) == 1, f"expected one full -1R stop leg, got {legs}"
    leg = legs[0]
    assert leg.exit_reason == "stop_loss"
    assert leg.parent_position_id is None, "partial must NOT have fired"
    assert leg.size == pytest.approx(10_000.0), "full position must close"
    assert leg.exit_price == pytest.approx(_SL)
    assert leg.entry_price == pytest.approx(_ENTRY)
    # Realised R = (exit - entry) / 1R == -1.0 exactly.
    assert (leg.exit_price - leg.entry_price) / _R == pytest.approx(-1.0)
    assert leg.exit_time == pd.Timestamp(exit_bar, tz="UTC")


def test_stop_before_partial_bar_is_minus_1r() -> None:
    """Stop breach on a bar STRICTLY BEFORE +1R is ever reached -> -1R."""
    result = _run([
        _bar("2026-01-01 00:00", o=1.099, h=1.100, lo=1.098, c=1.099),
        # entry fills here at open=1.100
        _bar("2026-01-01 01:00", o=1.100, h=1.101, lo=1.099, c=1.100),
        # stop breach (low<=1.098); high 1.1005 never reaches +1R (1.102)
        _bar("2026-01-01 02:00", o=1.100, h=1.1005, lo=1.098, c=1.099),
        _bar("2026-01-01 03:00", o=1.099, h=1.100, lo=1.099, c=1.099),
    ])
    _assert_single_minus_1r_stop(result, exit_bar="2026-01-01 02:00")


def test_same_bar_stop_and_partial_is_sl_first_minus_1r() -> None:
    """Same bar reaches +1R high AND breaches the stop low -> SL-first -> -1R.
    The partial must never fire; the full position takes the loss.
    """
    result = _run([
        _bar("2026-01-01 00:00", o=1.099, h=1.100, lo=1.098, c=1.099),
        _bar("2026-01-01 01:00", o=1.100, h=1.101, lo=1.099, c=1.100),
        # high 1.102 (=+1R) AND low 1.097 (<=SL) on the SAME bar
        _bar("2026-01-01 02:00", o=1.100, h=1.102, lo=1.097, c=1.099),
        _bar("2026-01-01 03:00", o=1.099, h=1.100, lo=1.099, c=1.099),
    ])
    _assert_single_minus_1r_stop(result, exit_bar="2026-01-01 02:00")
