"""Tests for core/strategies/kh24/exits/kijun_d1.py — fixed in PR-E.1.5.

Per the diff doc Section A, the EA's kijun_d1 exit fires when
``prev D1 close < prev D1 Kijun`` — both quantities on the just-closed
D1 bar (shift=1, lag-1). It does NOT compare the H4 close to the D1
Kijun. These tests pin the corrected semantics.
"""

from __future__ import annotations

import pandas as pd
import pytest

from core.sim.account import Direction, Position
from core.strategies.kh24.exits.kijun_d1 import (
    _build_d1_lag1_close_and_kijun,
    make_kijun_d1_exit_predicate,
)


def _bidask_frame(rows: list[tuple]) -> pd.DataFrame:
    """Build a bid+ask H4/D1 frame from (ts, bid_o, bid_h, bid_l, bid_c, ask_o, ask_h, ask_l, ask_c) tuples."""
    idx = pd.DatetimeIndex([r[0] for r in rows], tz="UTC")
    df = pd.DataFrame(
        {
            "open_bid": [r[1] for r in rows],
            "high_bid": [r[2] for r in rows],
            "low_bid": [r[3] for r in rows],
            "close_bid": [r[4] for r in rows],
            "open_ask": [r[5] for r in rows],
            "high_ask": [r[6] for r in rows],
            "low_ask": [r[7] for r in rows],
            "close_ask": [r[8] for r in rows],
            "volume": [1] * len(rows),
            "spread_close": [r[8] - r[4] for r in rows],
            "bid_ask_data_quality": ["ok"] * len(rows),
        },
        index=idx,
    )
    df.index.name = "timestamp_utc"
    return df


def _h4_one_bar(ts: str, bid_c: float, ask_c: float) -> pd.DataFrame:
    """Single H4 bar for predicate evaluation."""
    return _bidask_frame([(ts, bid_c, bid_c, bid_c, bid_c, ask_c, ask_c, ask_c, ask_c)])


def _d1_window(start_day: int, days: int, bid_close: float, bid_high: float, bid_low: float) -> pd.DataFrame:
    """D1 bars (bid==ask for simplicity) over a stretch of days."""
    rows = []
    for i in range(days):
        day = start_day + i
        rows.append((
            f"2026-01-{day:02d}", bid_close, bid_high, bid_low, bid_close,
            bid_close, bid_high, bid_low, bid_close,
        ))
    return _bidask_frame(rows)


# ── corrected semantics ───────────────────────────────────────────


def test_predicate_fires_when_prev_d1_close_below_prev_d1_kijun() -> None:
    """Build a D1 series where the most recent D1 close is BELOW that D1's
    Kijun(26). Predicate at the next-day H4 bar should fire."""
    # 26 D1 bars of warmup with high=1.20, low=1.00 → Kijun = 1.10
    # Then one extra D1 bar with close=1.05 — same Kijun (high/low unchanged) → close < Kijun
    days_warmup = 27
    rows = []
    for i in range(days_warmup):
        rows.append((
            f"2026-01-{i + 1:02d}", 1.10, 1.20, 1.00, 1.10, 1.10, 1.20, 1.00, 1.10,
        ))
    # Day 28 closes at 1.05 — below the running Kijun of ~1.10
    rows.append((
        "2026-01-28", 1.05, 1.20, 1.00, 1.05, 1.05, 1.20, 1.00, 1.05,
    ))
    df_d1 = _bidask_frame(rows)

    # H4 bar on day 29 (day after the lag-1 day-28 D1)
    df_h4 = _h4_one_bar("2026-01-29 12:00:00", 1.10, 1.10)

    predicate = make_kijun_d1_exit_predicate("EURUSD", df_h4, df_d1)
    pos = Position(
        position_id=1, pair="EURUSD", direction=Direction.LONG,
        entry_time=pd.Timestamp("2026-01-29", tz="UTC"),
        entry_price=1.10, size=1.0, sl_price=1.08, tp_price=None,
    )
    bar = df_h4.iloc[0]
    decision = predicate(pos, {"EURUSD": bar}, df_h4.index[0])
    assert decision is not None
    assert decision.exit_reason == "kijun_d1"


def test_predicate_does_not_fire_when_prev_d1_close_above_prev_d1_kijun() -> None:
    """Same setup but day-28 close stays above Kijun → no exit."""
    days_warmup = 27
    rows = []
    for i in range(days_warmup):
        rows.append((
            f"2026-01-{i + 1:02d}", 1.10, 1.20, 1.00, 1.10, 1.10, 1.20, 1.00, 1.10,
        ))
    rows.append((
        "2026-01-28", 1.15, 1.20, 1.00, 1.15, 1.15, 1.20, 1.00, 1.15,
    ))
    df_d1 = _bidask_frame(rows)
    df_h4 = _h4_one_bar("2026-01-29 12:00:00", 1.05, 1.05)  # H4 < D1 Kijun ≈ 1.10

    predicate = make_kijun_d1_exit_predicate("EURUSD", df_h4, df_d1)
    pos = Position(
        position_id=1, pair="EURUSD", direction=Direction.LONG,
        entry_time=pd.Timestamp("2026-01-29", tz="UTC"),
        entry_price=1.10, size=1.0, sl_price=1.08, tp_price=None,
    )
    bar = df_h4.iloc[0]
    decision = predicate(pos, {"EURUSD": bar}, df_h4.index[0])
    # H4 close is 1.05 — would have fired under the OLD impl that compared H4
    # mid-close vs D1 Kijun. Under the fixed impl, D1 close (1.15) > Kijun
    # (~1.10) → no exit.
    assert decision is None


def test_predicate_ignores_h4_mid_close_when_d1_above_kijun() -> None:
    """Direct regression test against the old (buggy) behaviour.

    The OLD v3 impl compared H4 mid-close to D1 Kijun. We construct a
    scenario where:
      - H4 mid-close is BELOW D1 Kijun (old impl would fire)
      - But D1 close is ABOVE D1 Kijun (new impl should NOT fire)
    """
    days_warmup = 27
    rows = []
    for i in range(days_warmup):
        rows.append((
            f"2026-01-{i + 1:02d}", 1.10, 1.20, 1.00, 1.10, 1.10, 1.20, 1.00, 1.10,
        ))
    # D1 close at 1.15 (well above Kijun ~1.10)
    rows.append((
        "2026-01-28", 1.15, 1.20, 1.00, 1.15, 1.15, 1.20, 1.00, 1.15,
    ))
    df_d1 = _bidask_frame(rows)
    # H4 bar with mid-close 1.07 (below Kijun ~1.10)
    df_h4 = _h4_one_bar("2026-01-29 12:00:00", 1.07, 1.07)

    predicate = make_kijun_d1_exit_predicate("EURUSD", df_h4, df_d1)
    pos = Position(
        position_id=1, pair="EURUSD", direction=Direction.LONG,
        entry_time=pd.Timestamp("2026-01-29", tz="UTC"),
        entry_price=1.10, size=1.0, sl_price=1.08, tp_price=None,
    )
    decision = predicate(pos, {"EURUSD": df_h4.iloc[0]}, df_h4.index[0])
    # FIXED impl: D1 close (1.15) > Kijun → no exit, even though H4 dipped.
    assert decision is None


def test_predicate_does_not_fire_for_short_positions() -> None:
    """The predicate is long-only (matches PR-E.1 design)."""
    days_warmup = 27
    rows = [
        (f"2026-01-{i + 1:02d}", 1.10, 1.20, 1.00, 1.10, 1.10, 1.20, 1.00, 1.10)
        for i in range(days_warmup)
    ]
    rows.append(("2026-01-28", 1.05, 1.20, 1.00, 1.05, 1.05, 1.20, 1.00, 1.05))
    df_d1 = _bidask_frame(rows)
    df_h4 = _h4_one_bar("2026-01-29 12:00:00", 1.10, 1.10)

    predicate = make_kijun_d1_exit_predicate("EURUSD", df_h4, df_d1)
    pos = Position(
        position_id=1, pair="EURUSD", direction=Direction.SHORT,
        entry_time=pd.Timestamp("2026-01-29", tz="UTC"),
        entry_price=1.10, size=1.0, sl_price=1.12, tp_price=None,
    )
    decision = predicate(pos, {"EURUSD": df_h4.iloc[0]}, df_h4.index[0])
    assert decision is None


def test_predicate_skips_other_pairs() -> None:
    """A per-pair predicate ignores positions on other pairs."""
    days_warmup = 27
    rows = [
        (f"2026-01-{i + 1:02d}", 1.10, 1.20, 1.00, 1.10, 1.10, 1.20, 1.00, 1.10)
        for i in range(days_warmup)
    ]
    rows.append(("2026-01-28", 1.05, 1.20, 1.00, 1.05, 1.05, 1.20, 1.00, 1.05))
    df_d1 = _bidask_frame(rows)
    df_h4 = _h4_one_bar("2026-01-29 12:00:00", 1.10, 1.10)

    predicate = make_kijun_d1_exit_predicate("EURUSD", df_h4, df_d1)
    pos = Position(
        position_id=1, pair="GBPUSD", direction=Direction.LONG,
        entry_time=pd.Timestamp("2026-01-29", tz="UTC"),
        entry_price=1.30, size=1.0, sl_price=1.28, tp_price=None,
    )
    decision = predicate(pos, {"GBPUSD": df_h4.iloc[0]}, df_h4.index[0])
    assert decision is None


# ── lag-1 alignment ──────────────────────────────────────────────


def test_lag1_alignment_same_day_d1_invisible() -> None:
    """Perturbing the D1 row for the SAME calendar day as the H4 bar must
    have ZERO impact on the predicate — D1 alignment is strictly prior."""
    days = 30
    rows_a = [
        (f"2026-01-{i + 1:02d}", 1.10, 1.20, 1.00, 1.10, 1.10, 1.20, 1.00, 1.10)
        for i in range(days)
    ]
    rows_b = list(rows_a)
    # Day 15 close perturbed wildly in b
    rows_b[14] = (
        "2026-01-15", 1.10, 1.20, 1.00, 99.99, 1.10, 1.20, 1.00, 99.99,
    )
    df_d1_a = _bidask_frame(rows_a)
    df_d1_b = _bidask_frame(rows_b)

    # H4 bar on day 15 — should NOT see the day-15 D1 close (uses day-14)
    df_h4 = _h4_one_bar("2026-01-15 12:00:00", 1.10, 1.10)

    d1_close_a, d1_kijun_a = _build_d1_lag1_close_and_kijun(df_h4.index, df_d1_a)
    d1_close_b, d1_kijun_b = _build_d1_lag1_close_and_kijun(df_h4.index, df_d1_b)

    # The day-15 H4 bar reads D1 from day-14 — the perturbation on day-15
    # must not appear in either output. Sanity-check by comparing the
    # whole series (NaN-aware equality).
    pd.testing.assert_series_equal(d1_close_a, d1_close_b, check_names=False)
    pd.testing.assert_series_equal(d1_kijun_a, d1_kijun_b, check_names=False)


def test_bid_side_kijun_differs_from_mid_when_spread_widens() -> None:
    """The fixed kijun uses bid-side OHLC. With a non-zero spread between
    bid and ask high/low, bid-Kijun != mid-Kijun. Sanity check that the
    Kijun computation reads bid columns."""
    rows = []
    for i in range(30):
        rows.append((
            f"2026-01-{i + 1:02d}",
            1.10, 1.20, 1.00, 1.10,   # bid OHLC
            1.20, 1.30, 1.10, 1.20,   # ask OHLC — shifted +0.10 (wide spread)
        ))
    df_d1 = _bidask_frame(rows)
    df_h4 = _h4_one_bar("2026-01-30 12:00:00", 1.10, 1.20)

    d1_close, d1_kijun = _build_d1_lag1_close_and_kijun(df_h4.index, df_d1)
    # Bid Kijun should be (1.20 + 1.00) / 2 = 1.10 (bid high/low over 26 bars)
    # Bid close at day-29 = 1.10
    assert d1_close.iloc[0] == pytest.approx(1.10)
    assert d1_kijun.iloc[0] == pytest.approx(1.10)
    # If we had used mid-OHLC, the values would be ~1.15 (shifted by half-spread)
    # — the test verifies bid-side semantics by NOT seeing that shift.
