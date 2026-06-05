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


def test_stop_then_recover_is_still_minus_1r() -> None:
    """Stop breaches on bar 02:00, THEN a later bar would reach +1R.

    The Arc-10 bug class booked such a trade as a win because the path
    scan saw the eventual +1R MFE and skipped the earlier stop. The
    SL-honest engine must close the FULL position at -1R on the breach
    bar; the subsequent recovery (high 1.103 >= +1R 1.102) must have NO
    effect — the position is already closed.
    """
    result = _run([
        _bar("2026-01-01 00:00", o=1.099, h=1.100, lo=1.098, c=1.099),
        _bar("2026-01-01 01:00", o=1.100, h=1.101, lo=1.099, c=1.100),
        # stop breach (low 1.097 <= SL 1.098); high 1.1005 never hits +1R
        _bar("2026-01-01 02:00", o=1.100, h=1.1005, lo=1.097, c=1.099),
        # full recovery PAST +1R (high 1.103) — must NOT resurrect the trade
        _bar("2026-01-01 03:00", o=1.101, h=1.103, lo=1.101, c=1.103),
        _bar("2026-01-01 04:00", o=1.103, h=1.104, lo=1.102, c=1.103),
    ])
    _assert_single_minus_1r_stop(result, exit_bar="2026-01-01 02:00")


def test_clean_win_partial_then_runner_trail_books_a_win() -> None:
    """No stop is ever touched: +1R partial fires, runner trails to a profit.

    The mirror of the take-the-loss cases — proves the engine books a
    genuine winner correctly (two profitable legs, NO stop_loss leg). A
    bug that spuriously injected a stop here would be just as dishonest
    as one that skipped a real stop.
    """
    result = _run([
        _bar("2026-01-01 00:00", o=1.099, h=1.100, lo=1.098, c=1.099),
        # entry fills at open=1.100; no stop, no +1R yet
        _bar("2026-01-01 01:00", o=1.100, h=1.101, lo=1.0995, c=1.1008),
        # +1R partial fires intra-bar (high 1.103 >= 1.102); low 1.1005 > SL
        _bar("2026-01-01 02:00", o=1.101, h=1.103, lo=1.1005, c=1.1025),
        # runner trails out: peak 1.103 -> trail level 1.101; close_bid
        # 1.1005 <= 1.101 AND strictly after the tp1 bar -> queue full close
        _bar("2026-01-01 03:00", o=1.102, h=1.1025, lo=1.100, c=1.1005),
        # runner fills at next-bar open_bid = 1.1005
        _bar("2026-01-01 04:00", o=1.1005, h=1.101, lo=1.100, c=1.1005),
    ])
    legs = list(result.closed_trades)
    assert len(legs) == 2, f"expected partial + runner legs, got {legs}"
    # Both legs belong to the same multi-leg close-out.
    assert all(leg.parent_position_id is not None for leg in legs)
    assert {leg.exit_reason for leg in legs} == {"partial_close_1r", "runner_trail_stop"}
    # No leg may be a stop — this trade never touched its SL.
    assert all(leg.exit_reason != "stop_loss" for leg in legs)
    partial = next(leg for leg in legs if leg.exit_reason == "partial_close_1r")
    runner = next(leg for leg in legs if leg.exit_reason == "runner_trail_stop")
    assert partial.size == pytest.approx(5_000.0)
    assert runner.size == pytest.approx(5_000.0)
    assert partial.exit_price == pytest.approx(_ENTRY + _R)  # +1R level
    assert partial.pnl > 0.0
    assert runner.pnl > 0.0
    assert (partial.pnl + runner.pnl) > 0.0  # a genuine, honestly-booked win


def test_stop_after_partial_books_partial_plus_runner_minus_1r() -> None:
    """+1R partial banks 50%; the runner is later stopped at -1R.

    Covers the Part A item-2 third bullet: a stop AFTER the partial closes
    the remaining 50% at the SL (runner -1R on half), with the partial
    already banked at +1R on the other half. The partial leg must survive;
    the runner leg must realise exactly the SL price.
    """
    result = _run([
        _bar("2026-01-01 00:00", o=1.099, h=1.100, lo=1.098, c=1.099),
        _bar("2026-01-01 01:00", o=1.100, h=1.101, lo=1.0995, c=1.1008),
        # +1R partial fires (high 1.103 >= 1.102); no stop this bar
        _bar("2026-01-01 02:00", o=1.101, h=1.103, lo=1.1005, c=1.1025),
        # runner stop: low 1.097 <= SL 1.098 -> runner closes at SL (-1R)
        _bar("2026-01-01 03:00", o=1.101, h=1.1015, lo=1.097, c=1.098),
        _bar("2026-01-01 04:00", o=1.098, h=1.099, lo=1.098, c=1.098),
    ])
    legs = list(result.closed_trades)
    assert len(legs) == 2, f"expected partial + runner-stop legs, got {legs}"
    assert all(leg.parent_position_id is not None for leg in legs)
    partial = next(leg for leg in legs if leg.exit_reason == "partial_close_1r")
    runner = next(leg for leg in legs if leg.exit_reason == "stop_loss")
    assert partial.size == pytest.approx(5_000.0)
    assert runner.size == pytest.approx(5_000.0)
    # Partial banked +1R on its half; runner realised exactly -1R on its half.
    assert partial.exit_price == pytest.approx(_ENTRY + _R)
    assert runner.exit_price == pytest.approx(_SL)
    assert (runner.exit_price - runner.entry_price) / _R == pytest.approx(-1.0)


# ── Direction.SHORT mirror (short-side enablement) ──────────────────────────
#
# The SAME take-the-loss invariant, mirrored for a short: the stop sits ABOVE
# entry and the +1R partial BELOW it. Every fixture below is the corresponding
# long fixture reflected around the entry (p' = 2*_ENTRY - p, with high<->low
# swapped), so a green short test is a direct proof of long/short symmetry on
# the SOLE gate engine. Short entry fills at open_bid; the stop fires on
# high_ask >= sl (core.sim.fill.short_sl_triggered), still SL-first.

_SL_SHORT = 1.102  # entry + 1R (stop ABOVE entry); short +1R partial = 1.098.


def _one_shot_short(t_target: str, **order_kwargs) -> StrategyFn:
    target = pd.Timestamp(t_target, tz="UTC")
    emitted = {"done": False}

    def strategy(t, snapshot, account):
        if emitted["done"] or t != target:
            return []
        emitted["done"] = True
        return [Order(pair="EURUSD", direction=Direction.SHORT, size=10_000.0, **order_kwargs)]

    return strategy


def _run_short(rows: list[dict]):
    panel = _make_panel(rows)
    acct = Account(
        starting_balance=100_000.0,
        exposure=ExposureRules(max_concurrent_per_pair=1),
    )
    strategy = _one_shot_short(
        "2026-01-01 00:00",
        sl_price=_SL_SHORT,
        atr_at_entry=0.001,
        sl_atr_mult=2.0,
        exit_policy="sl_partial_close_1r_runner_trail",
    )
    bt = MultiPairBacktester(
        panel=panel, account=acct, strategy=strategy,
        exit_policy_manager=ExitPolicyManager(),
    )
    return bt.run()


def _assert_single_minus_1r_stop_short(result, *, exit_bar: str) -> None:
    legs = list(result.closed_trades)
    assert len(legs) == 1, f"expected one full -1R short stop leg, got {legs}"
    leg = legs[0]
    assert leg.direction == Direction.SHORT
    assert leg.exit_reason == "stop_loss"
    assert leg.parent_position_id is None, "partial must NOT have fired"
    assert leg.size == pytest.approx(10_000.0), "full position must close"
    assert leg.exit_price == pytest.approx(_SL_SHORT)
    assert leg.entry_price == pytest.approx(_ENTRY)
    # Short realised R = (entry - exit) / 1R == -1.0 exactly.
    assert (leg.entry_price - leg.exit_price) / _R == pytest.approx(-1.0)
    # A short stop is a loss, never flattered: pnl = -size * 1R.
    assert leg.pnl == pytest.approx(-10_000.0 * _R)  # -20.0
    assert leg.exit_time == pd.Timestamp(exit_bar, tz="UTC")


def test_short_stop_before_partial_bar_is_minus_1r() -> None:
    """Short stop (above entry) breaches STRICTLY BEFORE +1R is reached -> -1R."""
    result = _run_short([
        _bar("2026-01-01 00:00", o=1.101, h=1.102, lo=1.100, c=1.101),
        # entry fills here at open=1.100 (short sells the bid)
        _bar("2026-01-01 01:00", o=1.100, h=1.101, lo=1.099, c=1.100),
        # stop breach (high>=1.102); low 1.0995 never reaches +1R (1.098)
        _bar("2026-01-01 02:00", o=1.100, h=1.102, lo=1.0995, c=1.101),
        _bar("2026-01-01 03:00", o=1.101, h=1.101, lo=1.100, c=1.101),
    ])
    _assert_single_minus_1r_stop_short(result, exit_bar="2026-01-01 02:00")


def test_short_same_bar_stop_and_partial_is_sl_first_minus_1r() -> None:
    """THE dispatch case: a short whose stop ABOVE entry is breached on the
    SAME bar its +1R partial BELOW entry would fire -> SL-first -> full -1R,
    the partial never fires."""
    result = _run_short([
        _bar("2026-01-01 00:00", o=1.101, h=1.102, lo=1.100, c=1.101),
        _bar("2026-01-01 01:00", o=1.100, h=1.101, lo=1.099, c=1.100),
        # high 1.103 (>=SL 1.102) AND low 1.098 (=+1R) on the SAME bar
        _bar("2026-01-01 02:00", o=1.100, h=1.103, lo=1.098, c=1.101),
        _bar("2026-01-01 03:00", o=1.101, h=1.101, lo=1.100, c=1.101),
    ])
    _assert_single_minus_1r_stop_short(result, exit_bar="2026-01-01 02:00")


def test_short_stop_then_recover_is_still_minus_1r() -> None:
    """Short stop breaches on bar 02:00; a later bar dropping past +1R must
    NOT resurrect the already-closed trade (the Arc-10 bug class, mirrored)."""
    result = _run_short([
        _bar("2026-01-01 00:00", o=1.101, h=1.102, lo=1.100, c=1.101),
        _bar("2026-01-01 01:00", o=1.100, h=1.101, lo=1.099, c=1.100),
        # stop breach (high 1.103 >= SL 1.102); low 1.0995 never hits +1R
        _bar("2026-01-01 02:00", o=1.100, h=1.103, lo=1.0995, c=1.101),
        # full recovery PAST +1R (low 1.097) — must NOT resurrect the trade
        _bar("2026-01-01 03:00", o=1.099, h=1.099, lo=1.097, c=1.097),
        _bar("2026-01-01 04:00", o=1.097, h=1.098, lo=1.096, c=1.097),
    ])
    _assert_single_minus_1r_stop_short(result, exit_bar="2026-01-01 02:00")


def test_short_clean_win_partial_then_runner_trail_books_a_win() -> None:
    """No stop ever touched: short +1R partial fires (50% at 1.098), the runner
    trails (trough + 1R) to a profit. The mirror of the long clean win — two
    profitable legs, NO stop_loss leg."""
    result = _run_short([
        _bar("2026-01-01 00:00", o=1.101, h=1.102, lo=1.100, c=1.101),
        # entry fills at open=1.100; no stop, no +1R yet
        _bar("2026-01-01 01:00", o=1.100, h=1.1005, lo=1.099, c=1.0992),
        # +1R partial fires intra-bar (low 1.097 <= 1.098); high 1.0995 < SL
        _bar("2026-01-01 02:00", o=1.099, h=1.0995, lo=1.097, c=1.0975),
        # runner trails out: trough 1.097 -> trail level 1.099; close_ask
        # 1.0995 >= 1.099 AND strictly after the tp1 bar -> queue full close
        _bar("2026-01-01 03:00", o=1.098, h=1.100, lo=1.0975, c=1.0995),
        # runner fills at next-bar open_ask = 1.0995
        _bar("2026-01-01 04:00", o=1.0995, h=1.100, lo=1.099, c=1.0995),
    ])
    legs = list(result.closed_trades)
    assert len(legs) == 2, f"expected partial + runner legs, got {legs}"
    assert all(leg.direction == Direction.SHORT for leg in legs)
    assert all(leg.parent_position_id is not None for leg in legs)
    assert {leg.exit_reason for leg in legs} == {"partial_close_1r", "runner_trail_stop"}
    assert all(leg.exit_reason != "stop_loss" for leg in legs)
    partial = next(leg for leg in legs if leg.exit_reason == "partial_close_1r")
    runner = next(leg for leg in legs if leg.exit_reason == "runner_trail_stop")
    assert partial.size == pytest.approx(5_000.0)
    assert runner.size == pytest.approx(5_000.0)
    assert partial.exit_price == pytest.approx(_ENTRY - _R)  # short +1R level (below)
    assert partial.pnl > 0.0
    assert runner.pnl > 0.0
    assert (partial.pnl + runner.pnl) > 0.0  # a genuine, honestly-booked short win


def test_short_stop_after_partial_books_partial_plus_runner_minus_1r() -> None:
    """Short +1R partial banks 50% at 1.098; the runner is later stopped at the
    SL above entry (-1R on the runner half). Mirror of the long counterpart."""
    result = _run_short([
        _bar("2026-01-01 00:00", o=1.101, h=1.102, lo=1.100, c=1.101),
        _bar("2026-01-01 01:00", o=1.100, h=1.1005, lo=1.099, c=1.0992),
        # +1R partial fires (low 1.097 <= 1.098); no stop this bar
        _bar("2026-01-01 02:00", o=1.099, h=1.0995, lo=1.097, c=1.0975),
        # runner stop: high 1.103 >= SL 1.102 -> runner closes at SL (-1R)
        _bar("2026-01-01 03:00", o=1.099, h=1.103, lo=1.0985, c=1.102),
        _bar("2026-01-01 04:00", o=1.102, h=1.102, lo=1.101, c=1.102),
    ])
    legs = list(result.closed_trades)
    assert len(legs) == 2, f"expected partial + runner-stop legs, got {legs}"
    assert all(leg.direction == Direction.SHORT for leg in legs)
    assert all(leg.parent_position_id is not None for leg in legs)
    partial = next(leg for leg in legs if leg.exit_reason == "partial_close_1r")
    runner = next(leg for leg in legs if leg.exit_reason == "stop_loss")
    assert partial.size == pytest.approx(5_000.0)
    assert runner.size == pytest.approx(5_000.0)
    # Partial banked +1R on its half; runner realised exactly -1R on its half.
    assert partial.exit_price == pytest.approx(_ENTRY - _R)
    assert runner.exit_price == pytest.approx(_SL_SHORT)
    assert (runner.entry_price - runner.exit_price) / _R == pytest.approx(-1.0)
