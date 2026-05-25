"""compute_per_day_max_dd under EET vs UTC bucketing (Amendment 6).

This is the load-bearing EET fix: the daily-DD bucket boundary feeds
``daily_dd_breaches_at_r_safe`` / ``daily_dd_breaches_at_r_hard`` gates
in ``core.wfo.amended_gates``. Under EET bucketing a drawdown event at
EET 23:00 (UTC 21:00 winter) is attributed to the correct EET trading
day matching 5ers' actual reset boundary.

Coverage:
    - Fixture demonstrates bucket-shift between UTC and EET (different
      breach count for the same equity series)
    - Default convention is "5ers_eet" (post-Amendment-6)
    - Explicit "utc" convention preserves legacy bucketing byte-identically
    - sha256-deterministic output
"""

from __future__ import annotations

import datetime as dt
import hashlib

import pandas as pd

from core.runners._fold_stats_helpers import compute_per_day_max_dd


def _equity_spanning_eet_boundary_winter() -> pd.Series:
    """Synthetic equity that drops 6% inside one EET day (Jan 15) but
    that 6% drop sits across TWO UTC calendar days (Jan 14 and Jan 15).

    Under UTC bucketing the 6% drop splits → no UTC day shows a single
    breach. Under EET bucketing the 6% drop lands inside EET day Jan 15
    → one breach at the standard 5% threshold."""
    idx = pd.DatetimeIndex(
        [
            "2026-01-14 21:00",  # EET 23:00 Jan 14, EET day Jan 14
            "2026-01-14 22:00",  # EET 00:00 Jan 15, EET day Jan 15 START
            "2026-01-14 23:00",  # EET 01:00 Jan 15
            "2026-01-15 02:00",  # EET 04:00 Jan 15
            "2026-01-15 12:00",  # EET 14:00 Jan 15
            "2026-01-15 20:00",  # EET 22:00 Jan 15
            "2026-01-15 22:00",  # EET 00:00 Jan 16, EET day Jan 16 START
            "2026-01-16 06:00",  # EET 08:00 Jan 16
        ],
        tz="UTC",
    )
    # Day-by-day equity values designed so that:
    # - UTC day Jan 14 (samples at 21:00, 22:00, 23:00): starts 100k, dips to 98k → 2% DD (no breach)
    # - UTC day Jan 15 (samples at 02:00, 12:00, 20:00, 22:00): starts 96k, dips to 95k → ~1% DD
    # - UTC day Jan 16 (sample at 06:00): starts 95k, no change → 0% DD
    # vs
    # - EET day Jan 14 (sample at UTC 21:00): starts 100k, no change → 0%
    # - EET day Jan 15 (samples at UTC 22:00 Jan 14 .. UTC 20:00 Jan 15):
    #       starts 100k, dips to 94k → 6% DD (BREACH at 5%)
    # - EET day Jan 16 (samples at UTC 22:00 Jan 15 + UTC 06:00 Jan 16):
    #       starts 95k, no change → 0%
    equity_values = [
        100_000.0,  # UTC 21:00 Jan 14 — EET day Jan 14 sample
        100_000.0,  # UTC 22:00 Jan 14 — EET day Jan 15 START
        98_000.0,   # UTC 23:00 Jan 14 — EET day Jan 15 (close UTC 14)
        96_000.0,   # UTC 02:00 Jan 15 — EET day Jan 15 (and UTC 15 start)
        95_000.0,   # UTC 12:00 Jan 15 — EET day Jan 15
        94_000.0,   # UTC 20:00 Jan 15 — EET day Jan 15 deepest dip
        95_000.0,   # UTC 22:00 Jan 15 — EET day Jan 16 START
        95_000.0,   # UTC 06:00 Jan 16 — EET day Jan 16
    ]
    return pd.Series(equity_values, index=idx, name="equity")


# ── Bucket-shift visibility ────────────────────────────────────────


def test_utc_bucketing_distributes_drop_across_days():
    equity = _equity_spanning_eet_boundary_winter()
    df = compute_per_day_max_dd(equity, boundary_convention="utc")
    # Three UTC days
    assert len(df) == 3
    # No single UTC day shows ≥5% DD because the drop is split
    assert (df["day_max_dd_base_pct"] < 0.05).all()


def test_eet_bucketing_unifies_drop_into_one_day():
    equity = _equity_spanning_eet_boundary_winter()
    df = compute_per_day_max_dd(equity, boundary_convention="5ers_eet")
    # Three EET days
    assert len(df) == 3
    # Exactly one EET day breaches 5%
    breach_count = int((df["day_max_dd_base_pct"] >= 0.05).sum())
    assert breach_count == 1
    # And it's EET day Jan 15 with start=100k min=94k = 6% DD
    eet_jan15 = df[df["date"] == dt.date(2026, 1, 15)].iloc[0]
    assert eet_jan15["day_start_equity"] == 100_000.0
    assert eet_jan15["day_max_dd_base_pct"] == pytest.approx(0.06, abs=1e-9)


def test_default_convention_is_5ers_eet():
    """Default param matches Amendment 6 engine convention."""
    equity = _equity_spanning_eet_boundary_winter()
    df_default = compute_per_day_max_dd(equity)
    df_explicit = compute_per_day_max_dd(equity, boundary_convention="5ers_eet")
    pd.testing.assert_frame_equal(df_default, df_explicit)


# ── UTC byte-identical legacy fallback ─────────────────────────────


def test_utc_convention_preserves_legacy_bucketing_structure():
    """Under utc convention, dates equal the UTC calendar dates of the
    equity samples — same semantics as pre-Amendment-6."""
    equity = _equity_spanning_eet_boundary_winter()
    df = compute_per_day_max_dd(equity, boundary_convention="utc")
    assert list(df["date"]) == [
        dt.date(2026, 1, 14),
        dt.date(2026, 1, 15),
        dt.date(2026, 1, 16),
    ]


def test_eet_convention_labels_eet_calendar_dates():
    equity = _equity_spanning_eet_boundary_winter()
    df = compute_per_day_max_dd(equity, boundary_convention="5ers_eet")
    assert list(df["date"]) == [
        dt.date(2026, 1, 14),
        dt.date(2026, 1, 15),
        dt.date(2026, 1, 16),
    ]


# ── sha256 determinism gate ───────────────────────────────────────


def _df_sha256(df: pd.DataFrame) -> str:
    csv_bytes = df.to_csv(lineterminator="\n", index=False).encode("utf-8")
    return hashlib.sha256(csv_bytes).hexdigest()


def test_eet_output_sha256_deterministic():
    equity = _equity_spanning_eet_boundary_winter()
    h1 = _df_sha256(compute_per_day_max_dd(equity, boundary_convention="5ers_eet"))
    h2 = _df_sha256(compute_per_day_max_dd(equity, boundary_convention="5ers_eet"))
    assert h1 == h2


def test_utc_vs_eet_outputs_differ():
    equity = _equity_spanning_eet_boundary_winter()
    h_utc = _df_sha256(compute_per_day_max_dd(equity, boundary_convention="utc"))
    h_eet = _df_sha256(compute_per_day_max_dd(equity, boundary_convention="5ers_eet"))
    assert h_utc != h_eet


# Late import to avoid leaking pytest into the module namespace before
# fixtures are defined (unused at module top).
import pytest  # noqa: E402
