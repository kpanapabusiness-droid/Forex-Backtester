"""Convention-aware H4 boundary math — regression tests for both conventions.

Covers the four sidecar layers' shared primitive (``deployment.sidecar.boundary``):
entry projection, the wake clock (``next_h4_close``), the anchor predicate
(``is_h4_anchor``), and the broker-offset expectation. The EET path is exercised
across BOTH EU DST regimes plus a spring-forward and a fall-back weekend
crossing, per the dispatch requirement (≥1 spring-forward, ≥1 fall-back).

Ground truth for the EET cases is the locked v3.0.2 EET pool: the spring-forward
GBPJPY row (2019-03-29 18:00 → 2019-03-31 21:00 UTC) and a representative winter
row (2013-01-11 18:00 → 2013-01-13 22:00 UTC) are taken verbatim from
``results/l_arc_10_v3.0.2/step_1/pool.parquet``; the summer / fall-back cases
are derived from the same convention rule that reproduces all 3,152 pool rows
(4 documented advisory residuals: 1 holiday + 3 US/EU DST-mismatch weeks).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from deployment.sidecar.boundary import (
    CONVENTION_EET,
    CONVENTION_UTC,
    expected_utc_offset_hours,
    is_h4_anchor,
    next_h4_close,
    project_entry_bar_open,
)


def _utc(y, m, d, h, mi=0, s=0):
    return datetime(y, m, d, h, mi, s, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# UTC convention — must remain byte-identical to the validated UTC behaviour.
# --------------------------------------------------------------------------- #


def test_utc_entry_weekday_plus_4h():
    assert project_entry_bar_open(_utc(2018, 10, 8, 8), CONVENTION_UTC) == _utc(2018, 10, 8, 12)
    assert project_entry_bar_open(_utc(2018, 10, 11, 20), CONVENTION_UTC) == _utc(2018, 10, 12, 0)


def test_utc_entry_friday_2000_snaps_to_sunday_reopen():
    got = project_entry_bar_open(_utc(2018, 10, 12, 20), CONVENTION_UTC)
    assert got == _utc(2018, 10, 14, 20)
    assert got.weekday() == 6 and got.hour == 20


def test_utc_next_close_matches_legacy_grid():
    assert next_h4_close(_utc(2026, 5, 27, 12, 0, 1), CONVENTION_UTC) == _utc(2026, 5, 27, 16)
    assert next_h4_close(_utc(2026, 5, 27, 12, 0, 0), CONVENTION_UTC) == _utc(2026, 5, 27, 16)
    assert next_h4_close(_utc(2026, 5, 27, 20, 30), CONVENTION_UTC) == _utc(2026, 5, 28, 0)
    assert next_h4_close(_utc(2026, 12, 31, 23, 30), CONVENTION_UTC) == _utc(2027, 1, 1, 0)


def test_utc_is_anchor_and_offset():
    assert is_h4_anchor(_utc(2024, 1, 15, 16), CONVENTION_UTC)
    assert not is_h4_anchor(_utc(2024, 1, 15, 17), CONVENTION_UTC)
    # 22:00 UTC is an EET anchor but NOT a UTC anchor.
    assert not is_h4_anchor(_utc(2013, 1, 13, 22), CONVENTION_UTC)
    assert expected_utc_offset_hours(CONVENTION_UTC, _utc(2024, 7, 1, 0)) == 0.0


# --------------------------------------------------------------------------- #
# EET convention — normal weekday projection (DST-resolved anchors).
# --------------------------------------------------------------------------- #


def test_eet_entry_weekday_plus_one_anchor_winter():
    # Winter (UTC+2): 08:00 EET = 06:00 UTC → next anchor 12:00 EET = 10:00 UTC.
    assert project_entry_bar_open(_utc(2024, 1, 8, 6), CONVENTION_EET) == _utc(2024, 1, 8, 10)


def test_eet_entry_weekday_plus_one_anchor_summer():
    # Summer (UTC+3): 08:00 EEST = 05:00 UTC → next anchor 12:00 EEST = 09:00 UTC.
    assert project_entry_bar_open(_utc(2024, 7, 8, 5), CONVENTION_EET) == _utc(2024, 7, 8, 9)


# --------------------------------------------------------------------------- #
# EET convention — weekend gap, both regimes + DST crossings.
# --------------------------------------------------------------------------- #


def test_eet_winter_friday_snaps_to_monday_open_pool_groundtruth():
    """Pool row USDCAD 2013-01-11 18:00 → 2013-01-13 22:00 UTC.

    Friday 20:00 EET-local signal → Monday 00:00 EET-local reopen = Sun 22:00 UTC.
    """
    assert project_entry_bar_open(_utc(2013, 1, 11, 18), CONVENTION_EET) == _utc(2013, 1, 13, 22)


def test_eet_summer_saturday_0000_local_is_tradeable():
    """EEST summer: the Saturday 00:00 EEST-local bar (= Fri 21:00 UTC) is the
    week's last tradeable bar, so a Fri 20:00 EEST signal projects +1 anchor,
    NOT a weekend snap."""
    # Fri 2024-07-05 20:00 EEST = 17:00 UTC → Sat 00:00 EEST = 21:00 UTC.
    assert project_entry_bar_open(_utc(2024, 7, 5, 17), CONVENTION_EET) == _utc(2024, 7, 5, 21)


def test_eet_summer_saturday_signal_snaps_to_monday():
    """A signal AT Saturday 00:00 EEST-local (= Fri 21:00 UTC) has no further
    tradeable bar until Monday 00:00 EEST = Sun 21:00 UTC."""
    assert project_entry_bar_open(_utc(2024, 7, 5, 21), CONVENTION_EET) == _utc(2024, 7, 7, 21)


def test_eet_spring_forward_weekend_pool_groundtruth():
    """DST SPRING-FORWARD crossing. Pool row GBPJPY 2019-03-29 18:00 → 2019-03-31
    21:00 UTC. Friday signal is in EET winter (+2); the Monday 00:00 reopen falls
    after the last-Sunday-March transition, so it is in EEST summer (+3) →
    Sun 21:00 UTC, NOT Sun 22:00. DST resolved by the tz db."""
    got = project_entry_bar_open(_utc(2019, 3, 29, 18), CONVENTION_EET)
    assert got == _utc(2019, 3, 31, 21)


def test_eet_fall_back_weekend_crossing():
    """DST FALL-BACK crossing. Signal at Saturday 00:00 EEST-local 2024-10-26
    (= Fri 2024-10-25 21:00 UTC, summer +3); the Monday 00:00 reopen falls after
    the last-Sunday-October transition, so it is in EET winter (+2) →
    Sun 2024-10-27 22:00 UTC."""
    got = project_entry_bar_open(_utc(2024, 10, 25, 21), CONVENTION_EET)
    assert got == _utc(2024, 10, 27, 22)


# --------------------------------------------------------------------------- #
# EET convention — wake clock, anchor predicate, broker offset.
# --------------------------------------------------------------------------- #


def test_eet_next_close_winter():
    # Winter UTC anchors 22/02/06/10/14/18. 12:30 UTC → 14:00 UTC.
    assert next_h4_close(_utc(2013, 1, 15, 12, 30), CONVENTION_EET) == _utc(2013, 1, 15, 14)


def test_eet_next_close_summer():
    # Summer UTC anchors 21/01/05/09/13/17. 12:30 UTC → 13:00 UTC.
    assert next_h4_close(_utc(2024, 7, 8, 12, 30), CONVENTION_EET) == _utc(2024, 7, 8, 13)


def test_eet_next_close_across_spring_forward():
    # Sat 2019-03-30 19:00 UTC, still winter → next winter anchor 22:00 UTC.
    assert next_h4_close(_utc(2019, 3, 30, 19), CONVENTION_EET) == _utc(2019, 3, 30, 22)


def test_eet_is_anchor():
    # 22:00 UTC = Mon 00:00 EET-local (winter) → EET anchor.
    assert is_h4_anchor(_utc(2013, 1, 13, 22), CONVENTION_EET)
    # 21:00 UTC = Mon 00:00 EEST-local (summer) → EET anchor.
    assert is_h4_anchor(_utc(2024, 7, 7, 21), CONVENTION_EET)
    # 20:00 UTC = 22:00 EET-local (winter) — NOT an anchor.
    assert not is_h4_anchor(_utc(2013, 1, 13, 20), CONVENTION_EET)


def test_eet_expected_offset_winter_and_summer():
    assert expected_utc_offset_hours(CONVENTION_EET, _utc(2024, 1, 15, 0)) == 2.0
    assert expected_utc_offset_hours(CONVENTION_EET, _utc(2024, 7, 15, 0)) == 3.0


def test_unsupported_convention_raises():
    with pytest.raises(ValueError, match="Unsupported boundary_convention"):
        project_entry_bar_open(_utc(2024, 1, 8, 6), "tokyo")
