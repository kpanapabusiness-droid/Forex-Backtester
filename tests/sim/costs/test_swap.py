"""Unit tests for core.sim.costs.swap — G1, G2, G3 correctness gates per
dispatch §4 of the Arc 10 v3.0.2 cost-realism sweep (REVISED Option B).
"""

from __future__ import annotations

import datetime as dt

import pytest

from core.sim.costs.swap import compute_swap_usd, rollover_instants_utc

_UTC = dt.timezone.utc


def _utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> dt.datetime:
    return dt.datetime(year, month, day, hour, minute, 0, tzinfo=_UTC)


# ---------------------------------------------------------------------------
# G1 — swap day-count: rollover crossings, not calendar-day-diff
# ---------------------------------------------------------------------------


def test_G1_mon_14_to_wed_10_is_two_rollovers_summer():
    """Mon 14:00 UTC → Wed 10:00 UTC, summer (EDT in effect).

    NY 17:00 EDT = 21:00 UTC. Rollovers expected:
      - Mon 21:00 UTC (Mon 17:00 EDT)
      - Tue 21:00 UTC (Tue 17:00 EDT)
    Wed 21:00 UTC is past exit (10:00 UTC); not counted.
    Total = 2, matches dispatch §4 G1 expectation 'not 3'.
    """
    # 2024-06-10 is a Monday, deep in EDT
    entry = _utc(2024, 6, 10, 14, 0)
    exit_ = _utc(2024, 6, 12, 10, 0)
    ros = rollover_instants_utc(entry, exit_)
    assert len(ros) == 2, f"expected 2 rollovers, got {len(ros)}: {ros}"
    # Both should be Monday and Tuesday at 21:00 UTC (EDT season)
    assert ros[0][0] == _utc(2024, 6, 10, 21, 0)
    assert ros[1][0] == _utc(2024, 6, 11, 21, 0)
    # Neither is Friday
    assert not ros[0][1] and not ros[1][1]


def test_G1_mon_14_to_wed_10_is_two_rollovers_winter():
    """Same trade in deep winter (EST in effect): rollover at 22:00 UTC."""
    # 2024-01-15 is a Monday, deep in EST
    entry = _utc(2024, 1, 15, 14, 0)
    exit_ = _utc(2024, 1, 17, 10, 0)
    ros = rollover_instants_utc(entry, exit_)
    assert len(ros) == 2
    assert ros[0][0] == _utc(2024, 1, 15, 22, 0)
    assert ros[1][0] == _utc(2024, 1, 16, 22, 0)


def test_intra_day_trade_has_zero_rollovers():
    """Trade entirely within one trading day, before 17:00 NY."""
    # Mon 14:00 UTC → Mon 19:00 UTC (= 15:00 EDT in summer)
    entry = _utc(2024, 6, 10, 14, 0)
    exit_ = _utc(2024, 6, 10, 19, 0)
    assert rollover_instants_utc(entry, exit_) == []


def test_trade_open_exactly_at_rollover_not_counted():
    """Rollover at entry instant (closed-open interval): NOT counted."""
    # Open exactly at Mon 21:00 UTC = 17:00 EDT rollover
    entry = _utc(2024, 6, 10, 21, 0)
    exit_ = _utc(2024, 6, 11, 19, 0)
    ros = rollover_instants_utc(entry, exit_)
    # Mon rollover at exact entry boundary → excluded; Tue 21:00 > exit 19:00 → excluded.
    assert len(ros) == 0


def test_friday_close_to_monday_open_has_one_friday_rollover():
    """Trade open Thursday, closed Monday 09:00. Crosses Friday 17:00 NY only.

    Sat and Sun rollovers are skipped (market closed). Mon 17:00 NY = past exit.
    """
    # 2024-06-13 is Thursday, 2024-06-17 is Monday
    entry = _utc(2024, 6, 13, 14, 0)
    exit_ = _utc(2024, 6, 17, 9, 0)
    ros = rollover_instants_utc(entry, exit_)
    # Expect: Thu 21:00 UTC, Fri 21:00 UTC. Mon 21:00 > exit 09:00, so excluded.
    assert len(ros) == 2
    # Thu rollover (not Friday)
    assert not ros[0][1]
    # Fri rollover IS Friday → flagged for 3×
    assert ros[1][1] is True
    assert ros[1][0] == _utc(2024, 6, 14, 21, 0)


# ---------------------------------------------------------------------------
# G2 — Friday 3× multiplier
# ---------------------------------------------------------------------------


def test_G2_friday_rollover_applies_3x():
    """Synthetic 1.0-lot trade crossing only Friday rollover → swap = 3× single nightly."""
    # 2024-06-14 is Friday. Open Thu 16:00 UTC, close Sat 12:00 UTC → crosses Thu + Fri rollovers
    entry = _utc(2024, 6, 13, 16, 0)
    exit_ = _utc(2024, 6, 15, 12, 0)
    # Use a positive swap_long_points for clarity (real values are negative)
    # at 1.0 lot, $10 pip value, -10 swap_points/night → -$1/night × Fri 3× = -$3
    total, detail = compute_swap_usd(
        entry_time_utc=entry,
        exit_time_utc=exit_,
        tp1_hit_time_utc=None,
        swap_long_points=-10.0,
        pip_value_usd=10.0,
        lots_original=1.0,
    )
    # 2 rollovers: Thu (1×) and Fri (3×).
    # Thu nightly = -10/10 × 10 × 1 × 1 = -$10
    # Fri nightly = -10/10 × 10 × 1 × 3 = -$30
    assert detail["n_rollovers"] == 2
    assert detail["n_friday_rollovers"] == 1
    assert total == pytest.approx(-10.0 + -30.0)


def test_no_friday_in_window_friday_count_zero():
    """Trade Mon-Wed has no Friday crossing."""
    entry = _utc(2024, 6, 10, 14, 0)
    exit_ = _utc(2024, 6, 12, 10, 0)
    total, detail = compute_swap_usd(
        entry_time_utc=entry,
        exit_time_utc=exit_,
        tp1_hit_time_utc=None,
        swap_long_points=-10.0,
        pip_value_usd=10.0,
        lots_original=1.0,
    )
    assert detail["n_friday_rollovers"] == 0
    # 2 nightly × -$10 each = -$20
    assert total == pytest.approx(-20.0)


# ---------------------------------------------------------------------------
# G3 — runner lot post-TP1 (reduced, not original)
# ---------------------------------------------------------------------------


def test_G3_runner_lot_post_tp1_reduced():
    """Trade with TP1 hit mid-trade: rollovers before TP1 use full lot, after use runner."""
    # Mon 14 → Thu 10. Crosses Mon, Tue, Wed rollovers (3 nights, no Friday).
    entry = _utc(2024, 6, 10, 14, 0)
    exit_ = _utc(2024, 6, 13, 10, 0)
    # TP1 hit Wed 12:00 UTC — Mon and Tue rollovers BEFORE TP1, Wed rollover AFTER.
    tp1 = _utc(2024, 6, 12, 12, 0)
    total, detail = compute_swap_usd(
        entry_time_utc=entry,
        exit_time_utc=exit_,
        tp1_hit_time_utc=tp1,
        swap_long_points=-10.0,
        pip_value_usd=10.0,
        lots_original=1.0,
        tp1_runner_fraction=0.5,
    )
    assert detail["n_rollovers"] == 3
    assert detail["n_with_full_lot"] == 2
    assert detail["n_with_runner_lot"] == 1
    # Mon: -$10, Tue: -$10, Wed (runner 0.5 lot): -$5
    assert total == pytest.approx(-10.0 - 10.0 - 5.0)


def test_G3_tp1_before_entry_all_runner():
    """Edge: tp1_hit_time_utc equal to entry_time_utc → all rollovers use runner lot.

    (Not a realistic case — TP1 can't hit at entry — but the primitive should
    handle it without crashing.)
    """
    entry = _utc(2024, 6, 10, 14, 0)
    exit_ = _utc(2024, 6, 12, 10, 0)
    tp1 = entry  # all rollovers ≥ tp1 → runner
    total, detail = compute_swap_usd(
        entry_time_utc=entry,
        exit_time_utc=exit_,
        tp1_hit_time_utc=tp1,
        swap_long_points=-10.0,
        pip_value_usd=10.0,
        lots_original=1.0,
        tp1_runner_fraction=0.5,
    )
    # 2 rollovers, both runner. Each = -$10/2 = -$5
    assert detail["n_with_runner_lot"] == 2
    assert detail["n_with_full_lot"] == 0
    assert total == pytest.approx(-10.0)


def test_G3_tp1_after_all_rollovers_all_full():
    """tp1 hits AFTER the last rollover in window → all swap charged at full lot.

    Trade Mon 14:00 UTC → Wed 20:00 UTC. Window covers Mon + Tue rollovers
    at 21:00 UTC each. Wed 21:00 UTC rollover is past exit (20:00 UTC), excluded.
    TP1 at Wed 19:00 UTC is AFTER both Mon/Tue rollovers → all 2 are full lot.
    """
    entry = _utc(2024, 6, 10, 14, 0)
    exit_ = _utc(2024, 6, 12, 20, 0)
    tp1 = _utc(2024, 6, 12, 19, 0)
    total, detail = compute_swap_usd(
        entry_time_utc=entry,
        exit_time_utc=exit_,
        tp1_hit_time_utc=tp1,
        swap_long_points=-10.0,
        pip_value_usd=10.0,
        lots_original=1.0,
    )
    # 2 rollovers (Mon 21:00, Tue 21:00) both before tp1=Wed 22:00 → full
    assert detail["n_with_full_lot"] == 2
    assert detail["n_with_runner_lot"] == 0
    assert total == pytest.approx(-20.0)


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def test_naive_datetime_raises():
    naive = dt.datetime(2024, 6, 10, 14, 0)
    with pytest.raises(ValueError, match="tz-aware"):
        rollover_instants_utc(naive, _utc(2024, 6, 12, 10, 0))


def test_exit_before_entry_returns_empty():
    """Defensive: bogus inputs (exit ≤ entry) → empty rollover list."""
    entry = _utc(2024, 6, 12, 10, 0)
    exit_ = _utc(2024, 6, 10, 14, 0)
    assert rollover_instants_utc(entry, exit_) == []


def test_dst_spring_forward_boundary():
    """US-DST starts 2024-03-10 (2nd Sun in March). Trade spanning the transition
    still resolves rollovers correctly via zoneinfo (no manual DST table).

    Open Fri 2024-03-08 16:00 UTC (EST in effect), close Tue 2024-03-12 16:00 UTC
    (EDT in effect post-switch). Rollovers in window:
      - Fri 03-08 22:00 UTC (Fri 17:00 EST, winter)
      - Mon 03-11 21:00 UTC (Mon 17:00 EDT, summer post-switch)
    Tue 03-12 21:00 UTC is past exit 16:00 UTC, excluded.
    Sat/Sun skipped.
    """
    entry = _utc(2024, 3, 8, 16, 0)
    exit_ = _utc(2024, 3, 12, 16, 0)
    ros = rollover_instants_utc(entry, exit_)
    assert len(ros) == 2
    assert ros[0][0] == _utc(2024, 3, 8, 22, 0)  # winter time
    assert ros[1][0] == _utc(2024, 3, 11, 21, 0)  # summer time
    # First rollover IS Friday — should flag 3×
    assert ros[0][1] is True
    # Mon rollover is NOT Friday
    assert ros[1][1] is False
