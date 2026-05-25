"""Tests for core.utils.session_boundary — UTC ↔ EET trading-day bucketing.

Coverage:
    - Winter (UTC+2 EET) anchor: EET 00:00 = UTC 22:00 prior day
    - Summer (UTC+3 EEST) anchor: EET 00:00 = UTC 21:00 prior day
    - DST transitions in both directions (last Sunday March / October)
    - UTC convention byte-identical to .normalize() fallback
    - Both Timestamp scalar and DatetimeIndex inputs
    - Tz-naive inputs (interpreted as UTC)
"""

from __future__ import annotations

import pandas as pd
import pytest

from core.utils.session_boundary import (
    SUPPORTED_CONVENTIONS,
    utc_to_eet_trading_day,
)


# ── Anchor cases (winter / summer) ──────────────────────────────────


def test_winter_eet_midnight_anchors_to_utc_22_prior_day():
    """EET winter: UTC 22:00 Jan 14 → EET 00:00 Jan 15 → key = UTC 22:00 Jan 14."""
    ts = pd.Timestamp("2026-01-14 22:00", tz="UTC")
    key = utc_to_eet_trading_day(ts, convention="5ers_eet")
    assert key == pd.Timestamp("2026-01-14 22:00", tz="UTC")


def test_winter_eet_morning_anchors_to_prior_utc_22():
    """UTC 04:00 Jan 15 (EET 06:00 Jan 15) → key = UTC 22:00 Jan 14 (EET-day start)."""
    ts = pd.Timestamp("2026-01-15 04:00", tz="UTC")
    key = utc_to_eet_trading_day(ts, convention="5ers_eet")
    assert key == pd.Timestamp("2026-01-14 22:00", tz="UTC")


def test_summer_eet_midnight_anchors_to_utc_21_prior_day():
    """EEST summer: UTC 21:00 Jul 14 → EEST 00:00 Jul 15 → key = UTC 21:00 Jul 14."""
    ts = pd.Timestamp("2026-07-14 21:00", tz="UTC")
    key = utc_to_eet_trading_day(ts, convention="5ers_eet")
    assert key == pd.Timestamp("2026-07-14 21:00", tz="UTC")


def test_summer_eet_morning_anchors_to_prior_utc_21():
    """UTC 04:00 Jul 15 (EEST 07:00 Jul 15) → key = UTC 21:00 Jul 14."""
    ts = pd.Timestamp("2026-07-15 04:00", tz="UTC")
    key = utc_to_eet_trading_day(ts, convention="5ers_eet")
    assert key == pd.Timestamp("2026-07-14 21:00", tz="UTC")


# ── EET-day boundary disambiguation ─────────────────────────────────


def test_same_eet_day_evening_and_morning_share_key_winter():
    """Bars at EET 23:00 day X (UTC 21:00 day X) and EET 02:00 day X
    (UTC 00:00 day X) — same EET trading day, share key. Under UTC
    bucketing they would (correctly) be in UTC day X too — no shift here.
    The interesting case is below."""
    evening = pd.Timestamp("2026-01-15 21:00", tz="UTC")  # EET 23:00 Jan 15
    morning = pd.Timestamp("2026-01-15 00:00", tz="UTC")  # EET 02:00 Jan 15
    key_evening = utc_to_eet_trading_day(evening, convention="5ers_eet")
    key_morning = utc_to_eet_trading_day(morning, convention="5ers_eet")
    assert key_evening == key_morning
    assert key_evening == pd.Timestamp("2026-01-14 22:00", tz="UTC")


def test_eet_day_spans_two_utc_calendar_days_winter():
    """The load-bearing case: EET day X spans UTC 22:00 (X-1) through
    UTC 21:59:59 (X). A bar at UTC 22:30 (X-1) and a bar at UTC 12:00 (X)
    share the same EET day X — but pre-fix UTC bucketing puts them in
    different UTC days. Post-fix EET bucketing unifies them."""
    early_eet = pd.Timestamp("2026-01-14 22:30", tz="UTC")   # EET 00:30 Jan 15
    midday_eet = pd.Timestamp("2026-01-15 12:00", tz="UTC")  # EET 14:00 Jan 15
    key_early = utc_to_eet_trading_day(early_eet, convention="5ers_eet")
    key_mid = utc_to_eet_trading_day(midday_eet, convention="5ers_eet")
    assert key_early == key_mid
    assert key_early == pd.Timestamp("2026-01-14 22:00", tz="UTC")
    # Sanity: under UTC convention these would be in DIFFERENT day buckets.
    assert utc_to_eet_trading_day(early_eet, convention="utc") != utc_to_eet_trading_day(
        midday_eet, convention="utc"
    )


def test_eet_day_spans_two_utc_calendar_days_summer():
    """Same shift case in EEST summer (UTC+3). EEST day X spans
    UTC 21:00 (X-1) through UTC 20:59:59 (X)."""
    early_eest = pd.Timestamp("2026-07-14 21:30", tz="UTC")   # EEST 00:30 Jul 15
    midday_eest = pd.Timestamp("2026-07-15 10:00", tz="UTC")  # EEST 13:00 Jul 15
    key_early = utc_to_eet_trading_day(early_eest, convention="5ers_eet")
    key_mid = utc_to_eet_trading_day(midday_eest, convention="5ers_eet")
    assert key_early == key_mid
    assert key_early == pd.Timestamp("2026-07-14 21:00", tz="UTC")
    assert utc_to_eet_trading_day(early_eest, convention="utc") != utc_to_eet_trading_day(
        midday_eest, convention="utc"
    )


# ── DST transitions ────────────────────────────────────────────────


def test_dst_spring_forward_2026_03_29():
    """Last Sunday in March 2026: EET 03:00 → EEST 04:00.
    Bars before the switch (UTC < 01:00) bucket to the EET-day starting
    UTC 22:00 of Mar 28; bars after the switch (UTC ≥ 01:00) bucket to
    the EEST-day starting UTC 21:00 of Mar 29."""
    before = pd.Timestamp("2026-03-29 00:30", tz="UTC")  # EET 02:30 Mar 29
    after = pd.Timestamp("2026-03-29 02:00", tz="UTC")   # EEST 05:00 Mar 29
    key_before = utc_to_eet_trading_day(before, convention="5ers_eet")
    key_after = utc_to_eet_trading_day(after, convention="5ers_eet")
    assert key_before == pd.Timestamp("2026-03-28 22:00", tz="UTC")
    assert key_after == pd.Timestamp("2026-03-28 22:00", tz="UTC")


def test_dst_fall_back_2026_10_25():
    """Last Sunday in October 2026: EEST 04:00 → EET 03:00.
    Bars in the ambiguous wall-clock window still bucket correctly via
    UTC indexing (zoneinfo handles ambiguity through the canonical UTC
    timeline)."""
    before = pd.Timestamp("2026-10-24 22:00", tz="UTC")  # EEST 01:00 Oct 25
    after = pd.Timestamp("2026-10-25 22:00", tz="UTC")   # EET 00:00 Oct 26
    key_before = utc_to_eet_trading_day(before, convention="5ers_eet")
    key_after = utc_to_eet_trading_day(after, convention="5ers_eet")
    assert key_before == pd.Timestamp("2026-10-24 21:00", tz="UTC")  # EEST day Oct 25
    assert key_after == pd.Timestamp("2026-10-25 22:00", tz="UTC")   # EET day Oct 26


# ── UTC convention byte-identical fallback ─────────────────────────


def test_utc_convention_matches_normalize_scalar():
    ts = pd.Timestamp("2026-01-14 22:00", tz="UTC")
    assert utc_to_eet_trading_day(ts, convention="utc") == ts.normalize()


def test_utc_convention_matches_normalize_index():
    idx = pd.DatetimeIndex(
        ["2026-01-14 22:00", "2026-07-14 21:00", "2026-03-29 02:00"],
        tz="UTC",
    )
    out = utc_to_eet_trading_day(idx, convention="utc")
    expected = idx.normalize()
    assert (out == expected).all()


# ── DatetimeIndex parity ────────────────────────────────────────────


def test_datetimeindex_matches_scalar_evaluation():
    idx = pd.DatetimeIndex(
        [
            "2026-01-14 22:00",
            "2026-01-15 04:00",
            "2026-07-14 21:00",
            "2026-07-15 04:00",
        ],
        tz="UTC",
    )
    expected = pd.DatetimeIndex(
        [utc_to_eet_trading_day(t, convention="5ers_eet") for t in idx]
    )
    out = utc_to_eet_trading_day(idx, convention="5ers_eet")
    assert (out == expected).all()


# ── Tz-naive input ─────────────────────────────────────────────────


def test_tz_naive_scalar_interpreted_as_utc():
    ts_naive = pd.Timestamp("2026-01-14 22:00")
    ts_aware = pd.Timestamp("2026-01-14 22:00", tz="UTC")
    assert utc_to_eet_trading_day(ts_naive, convention="5ers_eet") == utc_to_eet_trading_day(
        ts_aware, convention="5ers_eet"
    )


def test_tz_naive_index_interpreted_as_utc():
    idx_naive = pd.DatetimeIndex(["2026-01-14 22:00", "2026-07-14 21:00"])
    idx_aware = idx_naive.tz_localize("UTC")
    out_naive = utc_to_eet_trading_day(idx_naive, convention="5ers_eet")
    out_aware = utc_to_eet_trading_day(idx_aware, convention="5ers_eet")
    assert (out_naive == out_aware).all()


# ── Validation ─────────────────────────────────────────────────────


def test_unsupported_convention_raises():
    with pytest.raises(ValueError, match="Unsupported convention"):
        utc_to_eet_trading_day(pd.Timestamp("2026-01-14", tz="UTC"), convention="invalid")  # type: ignore[arg-type]


def test_supported_conventions_advertised():
    assert "utc" in SUPPORTED_CONVENTIONS
    assert "5ers_eet" in SUPPORTED_CONVENTIONS
