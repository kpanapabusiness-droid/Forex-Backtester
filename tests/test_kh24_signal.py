"""Tests for core/strategies/kh24/signal.py — KH-24 signal on bid+ask."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.strategies.kh24.signal import evaluate_kh24_signal


def _bidask_frame(rows: list[tuple]) -> pd.DataFrame:
    """Build a bid+ask H4/D1 frame from (ts, open, high, low, close) tuples.

    Spread is 0 (bid == ask) — keeps test arithmetic simple. Mid OHLC
    equals input OHLC.
    """
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


# ── empty-input behaviour ─────────────────────────────────────────────


def test_empty_inputs_return_empty_result() -> None:
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
    res = evaluate_kh24_signal(empty, empty)
    assert len(res.signal_mask) == 0
    assert len(res.atr_h4) == 0


# ── c1-c3 (bearish exhaustion bar) ────────────────────────────────────


def test_c1_rejects_bullish_bar() -> None:
    """A bar with close > open should NEVER signal (fails C1)."""
    # Synthetic: 50 bars rising, then one bullish exhaustion candidate
    h4 = []
    d1 = []
    for i in range(50):
        base = 1.10 + i * 0.001
        h4.append(
            (f"2026-01-01 {i % 24:02d}:00:00", base, base + 0.003, base - 0.001, base + 0.002)
        )
    # Last bar bullish (close > open) — should not signal
    bullish = ("2026-01-15 12:00:00", 1.20, 1.205, 1.195, 1.204)
    h4.append(bullish)
    # D1: stub uptrend
    for d in range(15):
        d1.append((f"2026-01-{d + 1:02d}", 1.10 + d * 0.001, 1.11, 1.09, 1.10 + d * 0.001 + 0.005))

    df_h4 = _bidask_frame(h4)
    df_d1 = _bidask_frame(d1)
    res = evaluate_kh24_signal(df_h4, df_d1)
    # The bullish last bar must be False
    assert res.signal_mask[-1] == False  # noqa: E712


def test_c3_rejects_close_near_high() -> None:
    """Bearish bar with close near high (not low) fails C3."""
    # Build 50 warmup bars + one bearish bar that closed near high
    h4 = []
    for i in range(50):
        base = 1.10 + i * 0.001
        h4.append(
            (
                f"2026-01-{(i % 28) + 1:02d} {(i % 24):02d}:00:00",
                base,
                base + 0.001,
                base - 0.001,
                base + 0.0005,
            )
        )
    # Bearish bar: open=1.20, high=1.205, low=1.195, close=1.2025 (top 1/3 of range)
    # cp = (1.2025 - 1.195) / (1.205 - 1.195) = 0.0075 / 0.01 = 0.75 > 0.24
    h4.append(("2026-02-01 12:00:00", 1.20, 1.205, 1.195, 1.2025))
    df_h4 = _bidask_frame(h4)
    d1 = [(f"2026-01-{d + 1:02d}", 1.10, 1.11, 1.09, 1.105) for d in range(28)]
    df_d1 = _bidask_frame(d1)
    res = evaluate_kh24_signal(df_h4, df_d1)
    assert res.signal_mask[-1] == False  # noqa: E712


# ── lookahead invariance for D1 lag-1 ─────────────────────────────────


def test_signal_uses_d1_lag1_not_same_day() -> None:
    """Perturbing the D1 row for the SAME calendar day as the H4 bar must
    have ZERO impact on the signal — D1 alignment is strictly prior."""
    # Build 30 days of D1 + corresponding H4
    h4_rows = []
    d1_rows_a = []
    d1_rows_b = []
    for day in range(1, 31):
        # H4 bars at 12:00 UTC each day
        h4_rows.append((f"2026-01-{day:02d} 12:00:00", 1.10, 1.11, 1.09, 1.105))
        # D1 a: normal close
        d1_rows_a.append((f"2026-01-{day:02d}", 1.10, 1.11, 1.09, 1.105))
        # D1 b: same as a EXCEPT day 15's close is wildly different
        if day == 15:
            d1_rows_b.append((f"2026-01-{day:02d}", 1.10, 1.11, 1.09, 99.99))
        else:
            d1_rows_b.append((f"2026-01-{day:02d}", 1.10, 1.11, 1.09, 1.105))

    df_h4 = _bidask_frame(h4_rows)
    df_d1_a = _bidask_frame(d1_rows_a)
    df_d1_b = _bidask_frame(d1_rows_b)

    res_a = evaluate_kh24_signal(df_h4, df_d1_a)
    res_b = evaluate_kh24_signal(df_h4, df_d1_b)

    # Day-15 H4 bar reads D1 from day 14. Day-15 D1 is invisible.
    # Day-16 H4 bar reads D1 from day 15 — THAT might differ.
    day_15_idx = 14  # zero-indexed
    day_16_idx = 15
    assert res_a.signal_mask[day_15_idx] == res_b.signal_mask[day_15_idx], (
        "Same-day D1 perturbation should not affect day-15 H4 signal"
    )
    # Sanity: the day-15 D1 IS visible at day-16 H4
    assert res_a.d1_close_lag1[day_16_idx] != res_b.d1_close_lag1[day_16_idx]


# ── auxiliary outputs ──────────────────────────────────────────────────


def test_atr_h4_aligned_to_input_index() -> None:
    h4 = [
        (f"2026-01-{(i % 28) + 1:02d} {(i % 24):02d}:00:00", 1.10, 1.105, 1.095, 1.10)
        for i in range(50)
    ]
    d1 = [(f"2026-01-{d + 1:02d}", 1.10, 1.11, 1.09, 1.105) for d in range(28)]
    res = evaluate_kh24_signal(_bidask_frame(h4), _bidask_frame(d1))
    assert len(res.atr_h4) == 50
    # First 13 bars have NaN ATR (Wilder warmup); rest finite
    assert np.isnan(res.atr_h4[:13]).all()
    assert np.isfinite(res.atr_h4[20:]).all()


def test_signal_mask_returns_bool_array() -> None:
    h4 = [
        (f"2026-01-{(i % 28) + 1:02d} {(i % 24):02d}:00:00", 1.10, 1.105, 1.095, 1.10)
        for i in range(50)
    ]
    d1 = [(f"2026-01-{d + 1:02d}", 1.10, 1.11, 1.09, 1.105) for d in range(28)]
    res = evaluate_kh24_signal(_bidask_frame(h4), _bidask_frame(d1))
    assert res.signal_mask.dtype == np.bool_
