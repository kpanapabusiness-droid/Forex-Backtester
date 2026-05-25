"""Integration tests for ExitPolicyManager + MultiPairBacktester wiring.

Synthetic-bar fixtures that exercise:
  * exit_policy=None preserves prior driver behaviour
  * sl_partial_close_1r_runner_trail fires partial intra-bar + runner trail
  * Same-bar SL suppression on the tp1 bar (reference's sl_breach > tp1_i)
  * sl_plus_tp_2r uses existing intra-bar TP infrastructure (no new code path)
  * Fail-loud RuntimeError when Order carries exit_policy but driver lacks manager
"""

from __future__ import annotations

import pandas as pd
import pytest

from core.sim.account import Account, Direction, ExposureRules
from core.sim.exit_policy_manager import ExitPolicyManager
from core.sim.multipair_backtester import MultiPairBacktester, Order, StrategyFn
from core.sim.panel import Panel


# ────────────────────────────────────────────────────────────────────────
# Synthetic 1-pair panel constructor — full bid/ask schema
# ────────────────────────────────────────────────────────────────────────


def _make_panel(rows: list[dict]) -> Panel:
    df = pd.DataFrame(rows)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df = df.set_index("timestamp_utc")
    # is_tradable_bar reads bar.spread implicitly via real_spread.is_tradable_bar;
    # synthetic bars must include all bid/ask cols so the function returns True.
    return Panel.from_frames({"EURUSD": df}, tf="H1")


def _bar(
    t: str,
    *,
    o: float, h: float, l: float, c: float,
    spread: float = 0.0,
) -> dict:
    """One bar row with bid/ask derived from a mid + spread.

    Default spread=0.0 (zero-spread / mid-only fixture). The DQ flag
    is forced to 'ok' so ``is_tradable_bar`` returns True regardless
    of the actual spread — useful for synthetic fixtures where we want
    deterministic mid-anchored fills.
    """
    half = spread / 2.0
    return {
        "timestamp_utc": t,
        "open_bid": o - half, "open_ask": o + half,
        "high_bid": h - half, "high_ask": h + half,
        "low_bid": l - half, "low_ask": l + half,
        "close_bid": c - half, "close_ask": c + half,
        "spread_close": spread,
        "bid_ask_data_quality": "ok",
    }


# ────────────────────────────────────────────────────────────────────────
# Strategies for tests
# ────────────────────────────────────────────────────────────────────────


def _one_shot_long_at(t_target: str, **order_kwargs) -> StrategyFn:
    """Strategy that emits exactly one long order at ``t_target``."""
    target = pd.Timestamp(t_target, tz="UTC")
    emitted = {"done": False}

    def strategy(t, snapshot, account):
        if emitted["done"] or t != target:
            return []
        emitted["done"] = True
        return [Order(pair="EURUSD", direction=Direction.LONG, size=10_000.0, **order_kwargs)]

    return strategy


def _no_op_strategy(t, snapshot, account):
    return []


# ────────────────────────────────────────────────────────────────────────
# exit_policy=None preserves prior behaviour
# ────────────────────────────────────────────────────────────────────────


def test_driver_without_manager_runs_unchanged_for_plain_orders() -> None:
    """When no Order carries exit_policy, the driver behaves exactly as
    before (no exit-policy code paths touched)."""
    panel = _make_panel([
        _bar("2026-01-01 00:00", o=1.100, h=1.101, l=1.099, c=1.100),
        _bar("2026-01-01 01:00", o=1.100, h=1.102, l=1.099, c=1.101),
        _bar("2026-01-01 02:00", o=1.101, h=1.103, l=1.100, c=1.102),
        _bar("2026-01-01 03:00", o=1.102, h=1.104, l=1.101, c=1.103),
    ])
    acct = Account(starting_balance=100_000.0)
    bt = MultiPairBacktester(panel=panel, account=acct, strategy=_no_op_strategy)
    result = bt.run()
    assert result.n_trades == 0


# ────────────────────────────────────────────────────────────────────────
# Fail-loud wiring errors
# ────────────────────────────────────────────────────────────────────────


def test_order_with_exit_policy_but_no_manager_raises() -> None:
    panel = _make_panel([
        _bar("2026-01-01 00:00", o=1.100, h=1.101, l=1.099, c=1.100),
        _bar("2026-01-01 01:00", o=1.100, h=1.102, l=1.099, c=1.101),
        _bar("2026-01-01 02:00", o=1.101, h=1.103, l=1.100, c=1.102),
    ])
    acct = Account(starting_balance=100_000.0)
    strategy = _one_shot_long_at(
        "2026-01-01 00:00",
        sl_price=1.098, atr_at_entry=0.001, sl_atr_mult=2.0,
        exit_policy="sl_only",
    )
    bt = MultiPairBacktester(
        panel=panel, account=acct, strategy=strategy
    )  # no exit_policy_manager
    with pytest.raises(RuntimeError, match="exit_policy_manager"):
        bt.run()


def test_order_with_exit_policy_but_missing_atr_raises() -> None:
    panel = _make_panel([
        _bar("2026-01-01 00:00", o=1.100, h=1.101, l=1.099, c=1.100),
        _bar("2026-01-01 01:00", o=1.100, h=1.102, l=1.099, c=1.101),
        _bar("2026-01-01 02:00", o=1.101, h=1.103, l=1.100, c=1.102),
    ])
    acct = Account(starting_balance=100_000.0)
    strategy = _one_shot_long_at(
        "2026-01-01 00:00",
        sl_price=1.098,
        # missing atr_at_entry + sl_atr_mult
        exit_policy="sl_only",
    )
    bt = MultiPairBacktester(
        panel=panel, account=acct, strategy=strategy,
        exit_policy_manager=ExitPolicyManager(),
    )
    with pytest.raises(RuntimeError, match="atr_at_entry or sl_atr_mult"):
        bt.run()


# ────────────────────────────────────────────────────────────────────────
# sl_partial_close_1r_runner_trail end-to-end on synthetic bars
# ────────────────────────────────────────────────────────────────────────


def test_partial_close_fires_intra_bar_and_runner_trails() -> None:
    """Long entry 1.100 at bar t=1 open. R_atr = 0.002 (sl_mult=2, atr=0.001).
    Bar 2 high reaches 1.102 → partial fires intra-bar at 1.102.
    Bar 3 high 1.103 → peak ratchets to 1.103, trail level = 1.101.
    Bar 4 close 1.101 → runner trail-hit fires; queues exit at bar 5 open.
    Bar 5 opens at 1.101 → runner closes at open_bid 1.101.

    Resulting trade log: 1 partial + 1 final, both linked.
    """
    panel = _make_panel([
        # t=0: signal/no-order bar
        _bar("2026-01-01 00:00", o=1.099, h=1.100, l=1.098, c=1.099),
        # t=1: entry fills at open=1.100
        _bar("2026-01-01 01:00", o=1.100, h=1.101, l=1.099, c=1.100),
        # t=2: high reaches +1R (1.102) → intra-bar partial fires at 1.102
        _bar("2026-01-01 02:00", o=1.100, h=1.102, l=1.099, c=1.101),
        # t=3: high 1.103, close 1.1015 (peak 1.103, trail = 1.101). close 1.1015 > 1.101 → no fire
        _bar("2026-01-01 03:00", o=1.101, h=1.103, l=1.100, c=1.1015),
        # t=4: close 1.101 = trail → at-close fires; queued for t=5 open
        _bar("2026-01-01 04:00", o=1.1015, h=1.1020, l=1.1010, c=1.101),
        # t=5: runner closes at open_bid = 1.101
        _bar("2026-01-01 05:00", o=1.101, h=1.102, l=1.100, c=1.101),
    ])
    acct = Account(
        starting_balance=100_000.0,
        exposure=ExposureRules(max_concurrent_per_pair=1),
    )
    strategy = _one_shot_long_at(
        "2026-01-01 00:00",  # signal at t=0; fills next bar t=1
        sl_price=1.098,
        atr_at_entry=0.001,
        sl_atr_mult=2.0,
        exit_policy="sl_partial_close_1r_runner_trail",
    )
    bt = MultiPairBacktester(
        panel=panel,
        account=acct,
        strategy=strategy,
        exit_policy_manager=ExitPolicyManager(),
    )
    result = bt.run()
    # Expect 2 ClosedTrade events: partial + final, linked by parent_position_id
    assert result.n_trades == 2, f"got {result.n_trades} trades: {result.closed_trades}"
    legs = list(result.closed_trades)
    assert all(leg.parent_position_id is not None for leg in legs)
    assert legs[0].parent_position_id == legs[1].parent_position_id
    # First leg = partial at 1.102 with size 5000
    assert legs[0].exit_reason == "partial_close_1r"
    assert legs[0].size == pytest.approx(5_000.0)
    assert legs[0].exit_price == pytest.approx(1.102)
    # Second leg = runner trail at next-bar open 1.101 with remaining 5000
    assert legs[1].exit_reason == "runner_trail_stop"
    assert legs[1].size == pytest.approx(5_000.0)
    assert legs[1].exit_price == pytest.approx(1.101)


def test_partial_close_runner_sl_hits_after_partial() -> None:
    """+1R partial fires bar 2; bar 3's bid touches original SL → runner
    closes at SL.
    """
    panel = _make_panel([
        _bar("2026-01-01 00:00", o=1.099, h=1.100, l=1.098, c=1.099),
        _bar("2026-01-01 01:00", o=1.100, h=1.101, l=1.099, c=1.100),
        # t=2: partial at +1R
        _bar("2026-01-01 02:00", o=1.100, h=1.102, l=1.099, c=1.101),
        # t=3: low touches SL=1.098 → runner closes at SL
        _bar("2026-01-01 03:00", o=1.101, h=1.102, l=1.097, c=1.099),
        _bar("2026-01-01 04:00", o=1.099, h=1.100, l=1.097, c=1.099),
    ])
    acct = Account(
        starting_balance=100_000.0,
        exposure=ExposureRules(max_concurrent_per_pair=1),
    )
    strategy = _one_shot_long_at(
        "2026-01-01 00:00",
        sl_price=1.098,
        atr_at_entry=0.001,
        sl_atr_mult=2.0,
        exit_policy="sl_partial_close_1r_runner_trail",
    )
    bt = MultiPairBacktester(
        panel=panel, account=acct, strategy=strategy,
        exit_policy_manager=ExitPolicyManager(),
    )
    result = bt.run()
    assert result.n_trades == 2
    legs = list(result.closed_trades)
    assert legs[0].exit_reason == "partial_close_1r"
    assert legs[0].exit_price == pytest.approx(1.102)
    assert legs[1].exit_reason == "stop_loss"
    assert legs[1].exit_price == pytest.approx(1.098)
    assert legs[1].size == pytest.approx(5_000.0)


def test_partial_close_same_bar_sl_suppression() -> None:
    """The tp1 bar has BOTH high >= +1R AND low <= SL. Reference forbids
    runner SL-out on tp1 bar (``sl_breach > tp1_i`` constraint). Canonical
    engine honours this via the manager's suppression flag.

    The salient assertion: the runner does NOT exit on the tp1 bar (t=2)
    via stop_loss. Without suppression, intra-bar SL on the partial-bar's
    low (1.097 ≤ SL 1.098) would close the runner at SL_price=1.098 with
    exit_time=t=2. With suppression, the runner survives bar 2 and exits
    on a subsequent bar via runner_trail or actual stop_loss.

    Bar layout: bar 2 has high=1.102 (tp1) AND low=1.097 (would breach SL).
    Bar 3 close is engineered above the runner-trail level so the trail
    doesn't fire on bar 3 either. Bar 4 has a clean SL touch.
    """
    panel = _make_panel([
        _bar("2026-01-01 00:00", o=1.099, h=1.100, l=1.098, c=1.099),
        _bar("2026-01-01 01:00", o=1.100, h=1.101, l=1.099, c=1.100),
        # t=2: tp1 bar. high=1.102 fires tp1; low=1.097 would breach SL
        # but is suppressed. Peak after bar 2 = 1.102 → trail = 1.100.
        # close=1.1015 > trail 1.100 so trail wouldn't fire even if we
        # got an evaluate_at_close on this bar (we don't: i > tp1_i).
        _bar("2026-01-01 02:00", o=1.100, h=1.102, l=1.097, c=1.1015),
        # t=3: peak ratchets up; trail rises. close above trail → no fire.
        _bar("2026-01-01 03:00", o=1.1015, h=1.1030, l=1.1010, c=1.1025),
        # t=4: clean SL hit (low=1.097 ≤ SL=1.098); not tp1 bar.
        _bar("2026-01-01 04:00", o=1.1025, h=1.1025, l=1.097, c=1.098),
    ])
    acct = Account(
        starting_balance=100_000.0,
        exposure=ExposureRules(max_concurrent_per_pair=1),
    )
    strategy = _one_shot_long_at(
        "2026-01-01 00:00",
        sl_price=1.098,
        atr_at_entry=0.001,
        sl_atr_mult=2.0,
        exit_policy="sl_partial_close_1r_runner_trail",
    )
    bt = MultiPairBacktester(
        panel=panel, account=acct, strategy=strategy,
        exit_policy_manager=ExitPolicyManager(),
    )
    result = bt.run()
    legs = list(result.closed_trades)
    assert len(legs) == 2
    # Leg 1: partial at +1R on tp1 bar
    assert legs[0].exit_reason == "partial_close_1r"
    assert legs[0].exit_price == pytest.approx(1.102)
    assert legs[0].exit_time == pd.Timestamp("2026-01-01 02:00", tz="UTC")
    # Leg 2: NOT closed on tp1 bar at SL — suppression worked
    assert legs[1].exit_time != pd.Timestamp("2026-01-01 02:00", tz="UTC")
    # Specifically: SL hit on bar 4
    assert legs[1].exit_reason == "stop_loss"
    assert legs[1].exit_price == pytest.approx(1.098)
    assert legs[1].exit_time == pd.Timestamp("2026-01-01 04:00", tz="UTC")


# ────────────────────────────────────────────────────────────────────────
# sl_plus_tp_2r via Order.tp_price (existing infra)
# ────────────────────────────────────────────────────────────────────────


def test_sl_plus_tp_2r_fires_via_intrabar_tp() -> None:
    """sl_plus_tp_2r decorates Order.tp_price; existing infra fires the
    TP intra-bar. No exit-policy manager state evaluation needed.

    Test: entry 1.100, R_atr=0.002, tp_price=1.104. Bar high 1.104 → fires.
    """
    panel = _make_panel([
        _bar("2026-01-01 00:00", o=1.099, h=1.100, l=1.098, c=1.099),
        _bar("2026-01-01 01:00", o=1.100, h=1.101, l=1.099, c=1.100),
        # high reaches tp=1.104
        _bar("2026-01-01 02:00", o=1.100, h=1.104, l=1.099, c=1.103),
        _bar("2026-01-01 03:00", o=1.103, h=1.104, l=1.102, c=1.103),
    ])
    acct = Account(
        starting_balance=100_000.0,
        exposure=ExposureRules(max_concurrent_per_pair=1),
    )
    strategy = _one_shot_long_at(
        "2026-01-01 00:00",
        sl_price=1.098,
        tp_price=1.104,  # caller responsible for merging policy.apply_to_order result
        atr_at_entry=0.001,
        sl_atr_mult=2.0,
        exit_policy="sl_plus_tp_2r",
    )
    bt = MultiPairBacktester(
        panel=panel, account=acct, strategy=strategy,
        exit_policy_manager=ExitPolicyManager(),
    )
    result = bt.run()
    assert result.n_trades == 1
    leg = result.closed_trades[0]
    assert leg.exit_reason == "take_profit"
    assert leg.exit_price == pytest.approx(1.104)
    assert leg.parent_position_id is None  # standalone full close
