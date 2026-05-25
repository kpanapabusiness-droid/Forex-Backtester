"""ResetFloorAccount daily-ratchet bucketing under EET vs UTC.

Forward-hygiene coverage: ``ResetFloorAccount`` is not currently
instantiated at runtime (per CC_20 read-first phase) but is convention-
aware so a future caller gets correct behaviour by default.

Default convention is ``"5ers_eet"`` post-Amendment-6. Tests cover both
conventions plus determinism.
"""

from __future__ import annotations

import hashlib

import pandas as pd
import pytest

from core.sim.risk.reset_floor import ResetFloorAccount


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


# ── EET-day boundary case ─────────────────────────────────────────


def test_eet_default_two_utc_days_same_eet_day_one_ratchet():
    """UTC 2026-01-14 22:30 (EET 00:30 Jan 15) and UTC 2026-01-15 10:00
    (EET 12:00 Jan 15) sit in the SAME EET trading day. Under EET
    convention only the first call ratchets the floor."""
    f = ResetFloorAccount(starting_balance=100_000.0)  # default 5ers_eet
    moved_1 = f.update_at_day_close(_ts("2026-01-14 22:30"), 101_000.0)
    moved_2 = f.update_at_day_close(_ts("2026-01-15 10:00"), 105_000.0)
    assert moved_1 is True
    assert moved_2 is False  # same EET trading day → idempotent
    assert f.floor == 101_000.0


def test_utc_convention_two_utc_days_two_ratchets():
    """Same timestamps under UTC convention: two distinct UTC days → two ratchets."""
    f = ResetFloorAccount(starting_balance=100_000.0, boundary_convention="utc")
    moved_1 = f.update_at_day_close(_ts("2026-01-14 22:30"), 101_000.0)
    moved_2 = f.update_at_day_close(_ts("2026-01-15 10:00"), 105_000.0)
    assert moved_1 is True
    assert moved_2 is True
    assert f.floor == 105_000.0


# ── DST handling propagates through the utility ────────────────────


def test_eet_dst_summer_two_utc_days_same_eet_day_one_ratchet():
    """Summer (EEST UTC+3): UTC 2026-07-14 21:30 (EEST 00:30 Jul 15)
    and UTC 2026-07-15 10:00 (EEST 13:00 Jul 15) share EEST day Jul 15."""
    f = ResetFloorAccount(starting_balance=100_000.0)
    moved_1 = f.update_at_day_close(_ts("2026-07-14 21:30"), 102_000.0)
    moved_2 = f.update_at_day_close(_ts("2026-07-15 10:00"), 110_000.0)
    assert moved_1 is True
    assert moved_2 is False
    assert f.floor == 102_000.0


# ── Validation ────────────────────────────────────────────────────


def test_unsupported_convention_raises():
    with pytest.raises(ValueError, match="Unsupported boundary_convention"):
        ResetFloorAccount(starting_balance=100_000.0, boundary_convention="bad")


# ── sha256 determinism gate ───────────────────────────────────────


def _floor_trajectory_sha256(events: list[tuple[str, float]], convention: str) -> str:
    f = ResetFloorAccount(starting_balance=100_000.0, boundary_convention=convention)
    trajectory: list[str] = []
    for ts, bal in events:
        moved = f.update_at_day_close(_ts(ts), bal)
        trajectory.append(f"{ts}|{bal}|{int(moved)}|{f.floor:.4f}")
    return hashlib.sha256("\n".join(trajectory).encode("utf-8")).hexdigest()


def test_eet_trajectory_sha256_deterministic():
    events = [
        ("2026-01-14 22:30", 101_000.0),
        ("2026-01-15 10:00", 105_000.0),
        ("2026-01-15 23:00", 110_000.0),  # EET Jan 16 starts at UTC 22:00 Jan 15
        ("2026-01-16 14:00", 108_000.0),  # still EET Jan 16
    ]
    h1 = _floor_trajectory_sha256(events, "5ers_eet")
    h2 = _floor_trajectory_sha256(events, "5ers_eet")
    assert h1 == h2


def test_utc_vs_eet_trajectories_diverge():
    events = [
        ("2026-01-14 22:30", 101_000.0),
        ("2026-01-15 10:00", 105_000.0),
        ("2026-01-15 23:00", 110_000.0),
        ("2026-01-16 14:00", 108_000.0),
    ]
    assert _floor_trajectory_sha256(events, "utc") != _floor_trajectory_sha256(events, "5ers_eet")
