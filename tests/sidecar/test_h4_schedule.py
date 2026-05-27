"""Test compute_next_utc_h4_close boundary handling."""

from __future__ import annotations

from datetime import datetime, timezone

from deployment.sidecar.sidecar import compute_next_utc_h4_close


def test_just_after_close_returns_next_anchor():
    now = datetime(2026, 5, 27, 12, 0, 1, tzinfo=timezone.utc)
    assert compute_next_utc_h4_close(now) == datetime(2026, 5, 27, 16, 0, tzinfo=timezone.utc)


def test_exactly_on_close_returns_next_anchor():
    # "Strictly greater than now" semantics — at 12:00 exactly, the next
    # close is 16:00 (we just emitted the 12:00 bar).
    now = datetime(2026, 5, 27, 12, 0, 0, tzinfo=timezone.utc)
    assert compute_next_utc_h4_close(now) == datetime(2026, 5, 27, 16, 0, tzinfo=timezone.utc)


def test_just_before_anchor():
    now = datetime(2026, 5, 27, 11, 59, 59, tzinfo=timezone.utc)
    assert compute_next_utc_h4_close(now) == datetime(2026, 5, 27, 12, 0, tzinfo=timezone.utc)


def test_after_last_anchor_rolls_to_next_day():
    now = datetime(2026, 5, 27, 20, 30, 0, tzinfo=timezone.utc)
    assert compute_next_utc_h4_close(now) == datetime(2026, 5, 28, 0, 0, tzinfo=timezone.utc)


def test_year_rollover():
    now = datetime(2026, 12, 31, 23, 30, 0, tzinfo=timezone.utc)
    assert compute_next_utc_h4_close(now) == datetime(2027, 1, 1, 0, 0, tzinfo=timezone.utc)


def test_naive_datetime_treated_as_utc():
    now = datetime(2026, 5, 27, 11, 0, 0)  # naive
    assert compute_next_utc_h4_close(now) == datetime(2026, 5, 27, 12, 0, tzinfo=timezone.utc)
