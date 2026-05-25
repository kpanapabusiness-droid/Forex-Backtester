"""Registry surface contract: name resolution, error message on unknown,
fresh-instance-per-call semantics, full inventory matches expectations.
"""

from __future__ import annotations

import pytest

from core.sim.account import Direction
from core.sim.exit_policies import (
    ExitPolicyContext,
    SlOnlyPolicy,
    SlPartialClose1RRunnerTrailPolicy,
    SlPlusTp2RPolicy,
    SlPlusTp3RPolicy,
    SlPlusTrailingAtrPolicy,
    SlPlusTrailingSwingPolicy,
    available_policies,
    build_exit_policy,
)


EXPECTED_REGISTRY = {
    "sl_only": SlOnlyPolicy,
    "sl_plus_tp_2r": SlPlusTp2RPolicy,
    "sl_plus_tp_3r": SlPlusTp3RPolicy,
    "sl_plus_trailing_atr": SlPlusTrailingAtrPolicy,
    "sl_plus_trailing_swing": SlPlusTrailingSwingPolicy,
    "sl_partial_close_1r_runner_trail": SlPartialClose1RRunnerTrailPolicy,
}


def test_available_policies_matches_expected_inventory() -> None:
    assert available_policies() == tuple(sorted(EXPECTED_REGISTRY))


@pytest.mark.parametrize("name,cls", list(EXPECTED_REGISTRY.items()))
def test_build_exit_policy_returns_correct_class(name: str, cls: type) -> None:
    p = build_exit_policy(name)
    assert isinstance(p, cls)
    assert p.name == name


def test_build_exit_policy_unknown_raises_keyerror_with_available_list() -> None:
    with pytest.raises(KeyError) as excinfo:
        build_exit_policy("not_a_policy")
    msg = str(excinfo.value)
    assert "not_a_policy" in msg
    assert "Available:" in msg
    # Every registered name appears in the error
    for name in EXPECTED_REGISTRY:
        assert name in msg


def test_build_exit_policy_returns_fresh_instance_per_call() -> None:
    p1 = build_exit_policy("sl_only")
    p2 = build_exit_policy("sl_only")
    assert p1 is not p2


def test_tp_policies_apply_to_order_sets_tp_price_long() -> None:
    ctx = ExitPolicyContext(
        entry_price=1.1000,
        atr_at_entry=0.0010,
        sl_atr_mult=2.0,
        direction=Direction.LONG,
    )
    # R_atr = 2.0 * 0.0010 = 0.0020
    assert build_exit_policy("sl_plus_tp_2r").apply_to_order(ctx) == {
        "tp_price": 1.1000 + 2.0 * 0.0020
    }
    assert build_exit_policy("sl_plus_tp_3r").apply_to_order(ctx) == {
        "tp_price": 1.1000 + 3.0 * 0.0020
    }


def test_tp_policies_apply_to_order_sets_tp_price_short() -> None:
    ctx = ExitPolicyContext(
        entry_price=1.1000,
        atr_at_entry=0.0010,
        sl_atr_mult=2.0,
        direction=Direction.SHORT,
    )
    assert build_exit_policy("sl_plus_tp_2r").apply_to_order(ctx) == {
        "tp_price": 1.1000 - 2.0 * 0.0020
    }


def test_stateful_policies_have_non_null_state() -> None:
    """Trailing + partial-close policies must build real state objects."""
    ctx = ExitPolicyContext(
        entry_price=1.1, atr_at_entry=0.0010, sl_atr_mult=2.0, direction=Direction.LONG
    )
    for name in (
        "sl_plus_trailing_atr",
        "sl_plus_trailing_swing",
        "sl_partial_close_1r_runner_trail",
    ):
        p = build_exit_policy(name)
        state = p.make_state(ctx)
        # State must NOT be NullPolicyState (it must carry per-position fields)
        from core.sim.exit_policies import NullPolicyState

        assert not isinstance(state, NullPolicyState), name


def test_stateless_policies_have_null_state() -> None:
    ctx = ExitPolicyContext(
        entry_price=1.1, atr_at_entry=0.0010, sl_atr_mult=2.0, direction=Direction.LONG
    )
    from core.sim.exit_policies import NullPolicyState

    for name in ("sl_only", "sl_plus_tp_2r", "sl_plus_tp_3r"):
        p = build_exit_policy(name)
        assert isinstance(p.make_state(ctx), NullPolicyState), name


def test_r_atr_helper() -> None:
    ctx = ExitPolicyContext(
        entry_price=1.1, atr_at_entry=0.0015, sl_atr_mult=2.5, direction=Direction.LONG
    )
    assert ctx.r_atr == pytest.approx(0.00375)
