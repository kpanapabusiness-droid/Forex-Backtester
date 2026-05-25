"""ExitPolicyManager: registration, evaluation hooks, intra-bar partial
suppression set, deregistration. Mirrors TrailManager test surface.
"""

from __future__ import annotations

import pandas as pd
import pytest

from core.sim.account import Account, Direction
from core.sim.exit_policies import (
    ExitAction,
    ExitPolicyContext,
    NullPolicyState,
    SlOnlyPolicy,
    SlPartialClose1RRunnerTrailPolicy,
    SlPlusTrailingAtrPolicy,
    build_exit_policy,
)
from core.sim.exit_policy_manager import ExitPolicyManager


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


def _bar(
    *,
    open_bid: float = 1.10,
    open_ask: float = 1.1001,
    high_bid: float = 1.11,
    high_ask: float = 1.1101,
    low_bid: float = 1.09,
    low_ask: float = 1.0901,
    close_bid: float = 1.105,
    close_ask: float = 1.1051,
) -> pd.Series:
    return pd.Series(
        {
            "open_bid": open_bid,
            "open_ask": open_ask,
            "high_bid": high_bid,
            "high_ask": high_ask,
            "low_bid": low_bid,
            "low_ask": low_ask,
            "close_bid": close_bid,
            "close_ask": close_ask,
        }
    )


def _seed_account_with_one_long_position(entry_price: float = 1.10) -> Account:
    acct = Account(starting_balance=100_000.0)
    acct.open(
        pair="EURUSD",
        direction=Direction.LONG,
        entry_time=_ts("2026-01-01"),
        entry_price=entry_price,
        size=10_000,
    )
    return acct


# ────────────────────────────────────────────────────────────────────────
# Registration / deregistration
# ────────────────────────────────────────────────────────────────────────


def test_register_creates_state() -> None:
    acct = _seed_account_with_one_long_position()
    pos = acct.open_positions[0]
    mgr = ExitPolicyManager()
    mgr.register(pos, SlOnlyPolicy(), atr_at_entry=0.0010, sl_atr_mult=2.0)
    assert mgr.is_registered(pos.position_id)
    assert isinstance(mgr.get_state(pos.position_id), NullPolicyState)
    assert mgr.get_policy_name(pos.position_id) == "sl_only"
    ctx = mgr.get_context(pos.position_id)
    assert ctx is not None
    assert ctx.r_atr == pytest.approx(0.0020)
    assert ctx.direction is Direction.LONG


def test_register_is_idempotent_on_replace() -> None:
    acct = _seed_account_with_one_long_position()
    pos = acct.open_positions[0]
    mgr = ExitPolicyManager()
    mgr.register(pos, SlOnlyPolicy(), atr_at_entry=0.0010, sl_atr_mult=2.0)
    mgr.register(pos, SlPlusTrailingAtrPolicy(), atr_at_entry=0.0015, sl_atr_mult=2.5)
    assert mgr.get_policy_name(pos.position_id) == "sl_plus_trailing_atr"
    assert mgr.get_context(pos.position_id).r_atr == pytest.approx(0.00375)


def test_deregister_removes_state() -> None:
    acct = _seed_account_with_one_long_position()
    pos = acct.open_positions[0]
    mgr = ExitPolicyManager()
    mgr.register(pos, SlOnlyPolicy(), atr_at_entry=0.0010, sl_atr_mult=2.0)
    mgr.deregister(pos.position_id)
    assert not mgr.is_registered(pos.position_id)
    assert mgr.get_state(pos.position_id) is None


def test_deregister_unknown_is_idempotent() -> None:
    mgr = ExitPolicyManager()
    mgr.deregister(999)  # no error


# ────────────────────────────────────────────────────────────────────────
# evaluate_intrabar_for_all
# ────────────────────────────────────────────────────────────────────────


def test_evaluate_intrabar_empty_when_no_regs() -> None:
    acct = Account(starting_balance=100_000.0)
    mgr = ExitPolicyManager()
    out = mgr.evaluate_intrabar_for_all({}, acct)
    assert out == {}
    assert mgr.positions_with_intrabar_partial_this_bar == set()


def test_evaluate_intrabar_sl_only_emits_nothing() -> None:
    acct = _seed_account_with_one_long_position()
    pos = acct.open_positions[0]
    mgr = ExitPolicyManager()
    mgr.register(pos, SlOnlyPolicy(), atr_at_entry=0.0010, sl_atr_mult=2.0)
    bar = _bar(high_bid=1.15)  # well above any threshold
    out = mgr.evaluate_intrabar_for_all({"EURUSD": bar}, acct)
    assert out == {}


def test_evaluate_intrabar_partial_close_fires_at_one_r() -> None:
    """Long entry 1.1000, R_atr = 2 * 0.001 = 0.002 → tp1 at 1.102.
    Bar with high_bid >= 1.102 fires partial close at fill_price=1.102.
    """
    acct = _seed_account_with_one_long_position(entry_price=1.1000)
    pos = acct.open_positions[0]
    mgr = ExitPolicyManager()
    mgr.register(
        pos,
        SlPartialClose1RRunnerTrailPolicy(),
        atr_at_entry=0.0010,
        sl_atr_mult=2.0,
    )
    bar = _bar(high_bid=1.1025, low_bid=1.099, close_bid=1.101)
    out = mgr.evaluate_intrabar_for_all({"EURUSD": bar}, acct)
    assert pos.position_id in out
    d = out[pos.position_id]
    assert d.action is ExitAction.PARTIAL_CLOSE
    assert d.exit_reason == "partial_close_1r"
    assert d.timing == "intrabar"
    assert d.fill_price == pytest.approx(1.1020)
    assert d.partial_fraction == 0.5
    # Manager set populated
    assert mgr.has_intrabar_partial_this_bar(pos.position_id)


def test_evaluate_intrabar_partial_does_not_fire_below_one_r() -> None:
    acct = _seed_account_with_one_long_position(entry_price=1.1000)
    pos = acct.open_positions[0]
    mgr = ExitPolicyManager()
    mgr.register(
        pos,
        SlPartialClose1RRunnerTrailPolicy(),
        atr_at_entry=0.0010,
        sl_atr_mult=2.0,
    )
    bar = _bar(high_bid=1.1019)  # just below +1R = 1.1020
    out = mgr.evaluate_intrabar_for_all({"EURUSD": bar}, acct)
    assert out == {}
    assert not mgr.has_intrabar_partial_this_bar(pos.position_id)


def test_evaluate_intrabar_does_not_fire_twice() -> None:
    """Once tp1 fires, subsequent bars don't re-fire."""
    acct = _seed_account_with_one_long_position(entry_price=1.1000)
    pos = acct.open_positions[0]
    mgr = ExitPolicyManager()
    mgr.register(
        pos,
        SlPartialClose1RRunnerTrailPolicy(),
        atr_at_entry=0.0010,
        sl_atr_mult=2.0,
    )
    bar1 = _bar(high_bid=1.1025)
    bar2 = _bar(high_bid=1.1050)
    out1 = mgr.evaluate_intrabar_for_all({"EURUSD": bar1}, acct)
    out2 = mgr.evaluate_intrabar_for_all({"EURUSD": bar2}, acct)
    assert pos.position_id in out1
    assert out2 == {}


def test_evaluate_intrabar_clears_partial_set_each_call() -> None:
    """Per-bar transient: the partial-set is fresh per evaluate call."""
    acct = _seed_account_with_one_long_position(entry_price=1.1000)
    pos = acct.open_positions[0]
    mgr = ExitPolicyManager()
    mgr.register(
        pos,
        SlPartialClose1RRunnerTrailPolicy(),
        atr_at_entry=0.0010,
        sl_atr_mult=2.0,
    )
    # Bar 1: tp1 fires
    mgr.evaluate_intrabar_for_all({"EURUSD": _bar(high_bid=1.1025)}, acct)
    assert mgr.has_intrabar_partial_this_bar(pos.position_id)
    # Bar 2: no fire (already fired) → set cleared, not repopulated
    mgr.evaluate_intrabar_for_all({"EURUSD": _bar(high_bid=1.1100)}, acct)
    assert not mgr.has_intrabar_partial_this_bar(pos.position_id)


def test_evaluate_intrabar_short_side() -> None:
    acct = Account(starting_balance=100_000.0)
    acct.open("EURUSD", Direction.SHORT, _ts("2026-01-01"), 1.1000, 10_000)
    pos = acct.open_positions[0]
    mgr = ExitPolicyManager()
    mgr.register(
        pos,
        SlPartialClose1RRunnerTrailPolicy(),
        atr_at_entry=0.0010,
        sl_atr_mult=2.0,
    )
    # Short tp1: low_ask <= entry - R_atr = 1.0980
    bar = _bar(low_ask=1.0975, high_bid=1.10, close_bid=1.099)
    out = mgr.evaluate_intrabar_for_all({"EURUSD": bar}, acct)
    assert pos.position_id in out
    assert out[pos.position_id].fill_price == pytest.approx(1.0980)


# ────────────────────────────────────────────────────────────────────────
# evaluate_at_close_for_all
# ────────────────────────────────────────────────────────────────────────


def test_evaluate_at_close_sl_only_emits_nothing() -> None:
    acct = _seed_account_with_one_long_position()
    pos = acct.open_positions[0]
    mgr = ExitPolicyManager()
    mgr.register(pos, SlOnlyPolicy(), atr_at_entry=0.0010, sl_atr_mult=2.0)
    out = mgr.evaluate_at_close_for_all({"EURUSD": _bar()}, acct)
    assert out == {}


def test_evaluate_at_close_trailing_atr_fires_after_activation() -> None:
    """Long 1.10, R_atr=0.002 → activation at high>=1.102.
    After activation peak ratchets; trail = peak - 0.002.
    When close_bid drops to trail → fire.
    """
    acct = _seed_account_with_one_long_position(entry_price=1.1000)
    pos = acct.open_positions[0]
    mgr = ExitPolicyManager()
    mgr.register(pos, SlPlusTrailingAtrPolicy(), atr_at_entry=0.0010, sl_atr_mult=2.0)
    # Bar 1: high reaches 1.104 (activates + peak=1.104, trail=1.102). Close 1.103 > 1.102 → no fire.
    bar1 = _bar(high_bid=1.104, low_bid=1.099, close_bid=1.103)
    out1 = mgr.evaluate_at_close_for_all({"EURUSD": bar1}, acct)
    assert out1 == {}
    # Bar 2: close drops to 1.101 ≤ trail 1.102 → fire
    bar2 = _bar(high_bid=1.103, low_bid=1.100, close_bid=1.101)
    out2 = mgr.evaluate_at_close_for_all({"EURUSD": bar2}, acct)
    assert pos.position_id in out2
    d = out2[pos.position_id]
    assert d.action is ExitAction.FULL_CLOSE
    assert d.exit_reason == "trailing_stop_atr"
    assert d.timing == "at_close"
    assert d.fill_price is None  # at_close timing → driver fills at next-bar open


# ────────────────────────────────────────────────────────────────────────
# Position-gone handling
# ────────────────────────────────────────────────────────────────────────


def test_evaluate_skips_positions_already_closed() -> None:
    """If driver closed a position (e.g. via predicate or SL) without
    calling manager.deregister, evaluate_* skips it cleanly.
    """
    acct = _seed_account_with_one_long_position()
    pos = acct.open_positions[0]
    mgr = ExitPolicyManager()
    mgr.register(pos, SlOnlyPolicy(), atr_at_entry=0.0010, sl_atr_mult=2.0)
    # Externally close
    acct.close(pos.position_id, _ts("2026-01-02"), 1.105, "external")
    # Manager evaluates without error; emits nothing
    out_i = mgr.evaluate_intrabar_for_all({"EURUSD": _bar()}, acct)
    out_c = mgr.evaluate_at_close_for_all({"EURUSD": _bar()}, acct)
    assert out_i == {}
    assert out_c == {}


# ────────────────────────────────────────────────────────────────────────
# Determinism
# ────────────────────────────────────────────────────────────────────────


def test_determinism_iteration_order() -> None:
    """Positions evaluated in sorted(position_id) order regardless of
    register order. Tested via no-raise on multi-position eval.
    """
    from core.sim.account import ExposureRules

    acct = Account(
        starting_balance=100_000.0,
        exposure=ExposureRules(max_concurrent_per_pair=None),
    )
    pos1 = acct.open("EURUSD", Direction.LONG, _ts("2026-01-01"), 1.10, 10_000)
    pos2 = acct.open("EURUSD", Direction.LONG, _ts("2026-01-01"), 1.10, 10_000)
    mgr = ExitPolicyManager()
    # Register in reverse order
    mgr.register(pos2, build_exit_policy("sl_only"), 0.001, 2.0)
    mgr.register(pos1, build_exit_policy("sl_only"), 0.001, 2.0)
    out = mgr.evaluate_intrabar_for_all({"EURUSD": _bar()}, acct)
    assert out == {}
