"""Tests for core/strategies/kh24/filters/{d1_regime,h1_cir}.py."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.strategies.kh24.filters.d1_regime import evaluate_d1_regime
from core.strategies.kh24.filters.h1_cir import evaluate_h1_cir


def _bidask_frame(rows: list[tuple]) -> pd.DataFrame:
    idx = pd.DatetimeIndex([r[0] for r in rows], tz="UTC")
    o = [r[1] for r in rows]
    h = [r[2] for r in rows]
    lo = [r[3] for r in rows]
    c = [r[4] for r in rows]
    df = pd.DataFrame(
        {
            "open_bid": o,
            "high_bid": h,
            "low_bid": lo,
            "close_bid": c,
            "open_ask": o,
            "high_ask": h,
            "low_ask": lo,
            "close_ask": c,
            "volume": [1] * len(rows),
            "spread_close": [0.0] * len(rows),
            "bid_ask_data_quality": ["ok"] * len(rows),
        },
        index=idx,
    )
    df.index.name = "timestamp_utc"
    return df


# ── D1 regime filter ─────────────────────────────────────────────────


def test_d1_regime_empty_inputs() -> None:
    empty = pd.DataFrame(
        columns=[
            "open_bid",
            "high_bid",
            "low_bid",
            "close_bid",
            "open_ask",
            "high_ask",
            "low_ask",
            "close_ask",
        ],
        index=pd.DatetimeIndex([], tz="UTC"),
    )
    out = evaluate_d1_regime(empty, empty)
    assert len(out) == 0


def test_d1_regime_lookahead_invariance() -> None:
    """Perturbing same-day D1 must not affect today's H4 regime output."""
    h4_rows = []
    d1_a = []
    d1_b = []
    # 31 days in January 2026; H4 bars at 12:00 each day, D1 daily.
    for day in range(1, 31):
        h4_rows.append((f"2026-01-{day:02d} 12:00:00", 1.10, 1.11, 1.09, 1.105))
        # Day 25 differs between the two D1 series
        if day == 25:
            d1_a.append((f"2026-01-{day:02d}", 1.10, 1.11, 1.09, 1.105))
            d1_b.append((f"2026-01-{day:02d}", 1.10, 1.11, 1.09, 99.99))
        else:
            d1_a.append((f"2026-01-{day:02d}", 1.10, 1.11, 1.09, 1.105))
            d1_b.append((f"2026-01-{day:02d}", 1.10, 1.11, 1.09, 1.105))

    df_h4 = _bidask_frame(h4_rows)
    out_a = evaluate_d1_regime(df_h4, _bidask_frame(d1_a))
    out_b = evaluate_d1_regime(df_h4, _bidask_frame(d1_b))
    # Same-day-25 perturbation does NOT affect the H4 bar on day-25
    # (which reads D1 from day-24)
    h4_dates = df_h4.index.normalize()
    day25 = pd.Timestamp("2026-01-25", tz="UTC")
    same_day_mask = h4_dates == day25
    assert (out_a[same_day_mask] == out_b[same_day_mask]).all()


def test_d1_regime_returns_boolean_array() -> None:
    h4 = [
        (f"2026-01-{(i % 28) + 1:02d} {(i % 24):02d}:00:00", 1.10, 1.11, 1.09, 1.105)
        for i in range(50)
    ]
    d1 = [(f"2026-01-{d + 1:02d}", 1.10, 1.11, 1.09, 1.105) for d in range(28)]
    out = evaluate_d1_regime(_bidask_frame(h4), _bidask_frame(d1))
    assert out.dtype == np.bool_
    assert len(out) == 50


# ── H1 CIR filter ────────────────────────────────────────────────────


def test_h1_cir_below_threshold_passes() -> None:
    """An H1 bar with CIR = 0.10 passes the T=0.28 gate."""
    # H4 bar at 12:00 UTC; reference H1 is at 15:00 (12 + 3 hours)
    h4 = _bidask_frame([("2026-01-15 12:00:00", 1.10, 1.105, 1.095, 1.10)])
    # H1 at 15:00 with CIR = 0.10: high=1.10, low=1.09, close=1.091
    # (close - low) / (high - low) = 0.001 / 0.010 = 0.10
    h1 = _bidask_frame([("2026-01-15 15:00:00", 1.095, 1.10, 1.09, 1.091)])
    out = evaluate_h1_cir(h4, h1)
    assert out[0] == True  # noqa: E712


def test_h1_cir_above_threshold_fails() -> None:
    """An H1 bar with CIR = 0.85 fails T=0.28."""
    h4 = _bidask_frame([("2026-01-15 12:00:00", 1.10, 1.105, 1.095, 1.10)])
    # H1 at 15:00 with close near high
    h1 = _bidask_frame([("2026-01-15 15:00:00", 1.095, 1.10, 1.09, 1.0985)])
    # CIR = 0.0085 / 0.01 = 0.85
    out = evaluate_h1_cir(h4, h1)
    assert out[0] == False  # noqa: E712


def test_h1_cir_zero_range_excluded() -> None:
    """High == low → undefined CIR → filter returns False (no signal)."""
    h4 = _bidask_frame([("2026-01-15 12:00:00", 1.10, 1.105, 1.095, 1.10)])
    h1 = _bidask_frame([("2026-01-15 15:00:00", 1.10, 1.10, 1.10, 1.10)])
    out = evaluate_h1_cir(h4, h1)
    assert out[0] == False  # noqa: E712


def test_h1_cir_at_boundary_passes() -> None:
    """CIR exactly at threshold passes (≤ is inclusive)."""
    h4 = _bidask_frame([("2026-01-15 12:00:00", 1.10, 1.105, 1.095, 1.10)])
    # CIR = 0.28 exactly: close - low = 0.0028, high - low = 0.01
    h1 = _bidask_frame([("2026-01-15 15:00:00", 1.095, 1.10, 1.09, 1.0928)])
    out = evaluate_h1_cir(h4, h1)
    assert out[0] == True  # noqa: E712
