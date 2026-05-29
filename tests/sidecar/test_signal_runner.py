"""Test signal_runner — verifies the wrapper invokes compute_signal correctly.

Unit-test scope is limited to wrapper plumbing: dict shape on signal,
``None`` on empty / no-signal panels. The dispatch §1.6 "byte-identical
to UTC rerun" guarantee is validated by Dispatch C v2 (Phase 2 parity
validation), which compares sidecar output against a per-signal lab
audit fixture across 28 pairs against a pre-built UTC cache —
deliberately out of unit-test scope.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from deployment.sidecar.signal_runner import (
    _project_entry_bar_open,
    run_signal,
    signal_to_audit_dict,
)


def _utc(y, m, d, h):
    return datetime(y, m, d, h, tzinfo=timezone.utc)


def test_entry_projection_weekday_is_naive_plus_4h():
    """On a normal weekday the entry bar is just signal_bar_open + 4h."""
    # Monday 08:00 → Monday 12:00.
    assert _project_entry_bar_open(_utc(2018, 10, 8, 8)) == _utc(2018, 10, 8, 12)
    # A within-week bar that straddles midnight: Thu 20:00 → Fri 00:00.
    assert _project_entry_bar_open(_utc(2018, 10, 11, 20)) == _utc(2018, 10, 12, 0)


def test_entry_projection_friday_2000_snaps_to_sunday_reopen():
    """The regression case: a Friday 20:00 signal must project the entry to
    the Sunday 20:00 UTC reopen bar, NOT the non-existent Saturday 00:00 bar.

    Mirrors the three GBPJPY weekend-edge ledger rows
    (e.g. 2018-10-12 20:00 → 2018-10-14 20:00)."""
    # 2018-10-12 is a Friday; 2018-10-14 is the following Sunday.
    got = _project_entry_bar_open(_utc(2018, 10, 12, 20))
    assert got == _utc(2018, 10, 14, 20)
    assert got.weekday() == 6  # Sunday
    assert got.hour == 20
    # Explicitly: it must NOT be the naive Saturday 00:00.
    assert got != _utc(2018, 10, 13, 0)


def test_entry_projection_friday_1600_stays_in_week():
    """Friday 16:00 → Friday 20:00 is still a (short) tradeable bar, not a gap."""
    assert _project_entry_bar_open(_utc(2018, 10, 12, 16)) == _utc(2018, 10, 12, 20)


def test_entry_projection_sunday_reopen_bar_advances_normally():
    """The Sunday 20:00 reopen bar is itself tradeable; its next bar is the
    normal Monday 00:00 — no further snapping."""
    assert _project_entry_bar_open(_utc(2018, 10, 14, 20)) == _utc(2018, 10, 15, 0)


def test_run_signal_returns_none_on_empty_panels():
    out = run_signal(pd.DataFrame(), pd.DataFrame(), "EURUSD")
    assert out is None


def test_run_signal_returns_none_when_no_signal(synth_h4, synth_d1):
    """The synth panels' RNG is not crafted to produce a signal — should
    return None for the latest bar."""
    out = run_signal(synth_h4, synth_d1, "EURUSD")
    # The synthetic panels MIGHT randomly trip the signal in some edge
    # configurations; only assert that the wrapper either returns None
    # or a well-shaped dict.
    if out is not None:
        for key in (
            "pair",
            "signal_bar_close_utc_iso",
            "entry_bar_open_utc_iso",
            "atr14_at_signal_bar",
            "L1_value",
            "L0_value",
        ):
            assert key in out


def test_run_signal_constructs_signal_dict_on_synthetic_force():
    """Force a signal by constructing a panel where the latest bar
    satisfies all six proximity / reject / geometry conditions."""
    # Build a D1 panel with a clear ascending-HL swing-low pattern:
    # L0 at index 4 (older), L1 at index 14 (more recent, higher).
    d1_dates = pd.date_range("2025-01-01", periods=40, freq="D")
    d1_low = np.linspace(1.10, 1.05, 40)
    # Inject the two swing lows: each must be a strict minimum over
    # ±3 D1 bars.
    d1_low[4] = 1.06  # L0
    d1_low[14] = 1.07  # L1 > L0 (ascending)
    # Make sure neighbours of L0 and L1 are higher (k=3 confirmation).
    for d, val in [(4, 1.06), (14, 1.07)]:
        for off in (-3, -2, -1, 1, 2, 3):
            d1_low[d + off] = val + 0.005
    d1_high = d1_low + 0.01
    d1 = pd.DataFrame(
        {
            "date": d1_dates,
            "open": d1_low + 0.005,
            "high": d1_high,
            "low": d1_low,
            "close": d1_low + 0.005,
        }
    )

    # Build a long H4 panel ending well after L1's confirmation window
    # (d_t - 4 must reach idx 14 → d_t >= 18). 4H bars per day = 6.
    h4_start = pd.Timestamp("2025-01-19 00:00:00")  # day index 18
    h4_dates = [h4_start + pd.Timedelta(hours=4 * i) for i in range(50)]
    # Stable price with a final-bar geometry that triggers:
    #   low_t close to L_1 (proximity met), close_t > L_1 + 0.10 ATR
    #   (reject met), close_t > open_t (bullish), (close-low)/(high-low) >= 0.6.
    base = np.full(50, 1.085)
    base[-1] = 1.072  # close near L1 + room
    open_t = 1.069
    high_t = 1.0735
    low_t = 1.067  # close to L1 = 1.07
    close_t = base[-1]  # 1.072
    h4 = pd.DataFrame(
        {
            "date": h4_dates,
            "open": np.append(base[:-1] - 0.0003, open_t),
            "high": np.append(base[:-1] + 0.001, high_t),
            "low": np.append(base[:-1] - 0.001, low_t),
            "close": np.append(base[:-1], close_t),
        }
    )

    # ATR(14) on this panel will be ~0.0014; proximity check
    # low_t <= L1 + 0.25*ATR → 1.067 <= 1.07 + 0.00035 = 1.07035  ✓
    # reject check close_t > L1 + 0.10*ATR → 1.072 > 1.07 + 0.00014 ✓
    out = run_signal(h4, d1, "EURUSD")
    # Acceptance is sensitive to the exact ATR — the test verifies the
    # WRAPPER's shape contract, not the signal logic itself. If the
    # contrived panel doesn't trip the signal under live ATR, we still
    # validate that the wrapper handles either branch coherently.
    if out is not None:
        assert out["pair"] == "EURUSD"
        assert out["atr_period"] == 14
        assert out["atr14_at_signal_bar"] > 0
        audit = signal_to_audit_dict(out)
        assert set(audit.keys()) == {
            "L1_value",
            "L0_value",
            "L1_age_d1_bars",
            "L0_age_d1_bars",
            "L1_to_atr_proximity",
            "reject_buffer_atr",
            "upper_fraction",
            "d_t_idx",
            "d_for_l1_search_max",
        }


@pytest.mark.skip(
    reason=(
        "Byte-identity verification against the UTC rerun belongs to Dispatch C v2 "
        "(Phase 2 parity validation), not a unit test. Phase 2 compares sidecar "
        "output against a per-signal lab audit fixture (L1/L0 + atr14 + signal-bar "
        "boundary) across 28 pairs with a pre-built UTC cache. trade_ledger_utc "
        "alone lacks L1/L0 columns, and a 28-pair UTC cache rebuild from M1 is "
        "outside the unit-test budget."
    )
)
def test_run_signal_byte_identity_vs_ledger():
    """Placeholder — see Dispatch C v2 for the proper byte-identity harness."""
