"""Unit tests for core.signals.htf_alignment.

Covers:
  - basic alignment semantics under UTC storage convention
  - basic alignment semantics under 5ers EET storage convention
  - require_fully_closed True (default) vs False
  - out-of-range LTF timestamps → NaN
  - tz-awareness mismatch → ValueError
  - get_htf_row_at parity with get_htf_value_at per column
  - byte-identical-to-legacy-KH-24 D1-lag-1 alignment under UTC convention
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.signals.htf_alignment import (
    _coerce_to_index,
    get_htf_row_at,
    get_htf_value_at,
)


def _d1_panel_utc(days: int = 7) -> pd.DataFrame:
    """D1 panel under legacy UTC boundary convention.

    Each D1 bar is left-labelled at UTC 00:00 of its calendar date.
    """
    start = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    idx = pd.date_range(start, periods=days, freq="1D", tz="UTC")
    return pd.DataFrame({"d1_close": np.arange(days, dtype=float)}, index=idx)


def _d1_panel_eet(days: int = 7) -> pd.DataFrame:
    """D1 panel under 5ers EET boundary convention (winter, UTC+2).

    Each D1 bar is left-labelled at UTC 22:00 of the prior calendar
    date (i.e., EET 00:00 of its EET-day).
    """
    # First EET-day = Jan 2 (starts UTC 22:00 Jan 1)
    start = pd.Timestamp("2024-01-01 22:00:00", tz="UTC")
    idx = pd.date_range(start, periods=days, freq="1D", tz="UTC")
    return pd.DataFrame({"d1_close": np.arange(days, dtype=float)}, index=idx)


def _h4_index_utc(days: int = 5) -> pd.DatetimeIndex:
    """H4 LTF index under UTC convention (00, 04, 08, 12, 16, 20 each day)."""
    start = pd.Timestamp("2024-01-02 00:00:00", tz="UTC")
    return pd.date_range(start, periods=days * 6, freq="4h", tz="UTC")


def _h4_index_eet(days: int = 5) -> pd.DatetimeIndex:
    """H4 LTF index under 5ers EET winter convention (UTC 22, 02, 06, 10, 14, 18)."""
    start = pd.Timestamp("2024-01-01 22:00:00", tz="UTC")
    return pd.date_range(start, periods=days * 6, freq="4h", tz="UTC")


# ── basic semantics ──────────────────────────────────────────────────


def test_default_returns_prior_d1_for_h4_under_utc() -> None:
    """Default (require_fully_closed=True) gives the prior-calendar-day D1.

    Under UTC convention: H4 inside Jan 3 must see D1 of Jan 2 (= d1_close=1).
    """
    d1 = _d1_panel_utc(days=5)  # Jan 1..Jan 5, d1_close = 0..4
    h4 = pd.DatetimeIndex([pd.Timestamp("2024-01-03 04:00:00", tz="UTC")])
    out = get_htf_value_at(h4, d1, "d1_close")
    assert out.values[0] == 1.0  # Jan 2's D1


def test_default_returns_prior_d1_for_h4_under_eet() -> None:
    """Under 5ers EET convention: H4 inside EET-day-N must see D1 of EET-day-(N−1).

    EET-day-2 starts at UTC 22:00 Jan 1 (=d1_close index 0).
    H4 at EET 04:00 EET-day-3 = UTC 02:00 Jan 3.
    Must see D1 of EET-day-2 (d1_close index 0 → value 0).
    """
    d1 = _d1_panel_eet(days=5)  # EET-day-2..EET-day-6
    # H4 at "EET 04:00 EET-day-3" = UTC 02:00 Jan 3 (winter, UTC+2 offset)
    h4_inside_eet_day3 = pd.DatetimeIndex([pd.Timestamp("2024-01-03 02:00:00", tz="UTC")])
    out = get_htf_value_at(h4_inside_eet_day3, d1, "d1_close")
    # D1 EET-day-2 starts at UTC 22:00 Jan 1 = d1_close[0] = 0.
    assert out.values[0] == 0.0


def test_no_lookahead_at_first_h4_of_day_utc() -> None:
    """H4 at exactly day boundary (UTC 00:00 N) sees D1 of N−1, not N."""
    d1 = _d1_panel_utc(days=5)
    h4_at_boundary = pd.DatetimeIndex([pd.Timestamp("2024-01-03 00:00:00", tz="UTC")])
    out = get_htf_value_at(h4_at_boundary, d1, "d1_close")
    assert out.values[0] == 1.0  # Jan 2 D1, NOT Jan 3 D1


def test_no_lookahead_at_first_h4_of_day_eet() -> None:
    """H4 at exactly EET-day boundary sees PRIOR EET-day D1, not same-day."""
    d1 = _d1_panel_eet(days=5)
    # First H4 of EET-day-3 = UTC 22:00 Jan 2 (= same instant as D1 EET-day-3 start).
    h4_at_eet_day3_boundary = pd.DatetimeIndex(
        [pd.Timestamp("2024-01-02 22:00:00", tz="UTC")]
    )
    out = get_htf_value_at(h4_at_eet_day3_boundary, d1, "d1_close")
    # Must see D1 EET-day-2 (= d1_close[0] = 0.0), not D1 EET-day-3 (= d1_close[1] = 1.0).
    assert out.values[0] == 0.0


def test_require_fully_closed_false_returns_containing_bar() -> None:
    """require_fully_closed=False: returns the HTF bar CONTAINING the LTF ts.

    Arc 10 DLR pattern: wants D1 containing each H4, then applies its own offset.
    """
    d1 = _d1_panel_utc(days=5)
    h4_inside_jan3 = pd.DatetimeIndex([pd.Timestamp("2024-01-03 04:00:00", tz="UTC")])
    out = get_htf_value_at(h4_inside_jan3, d1, "d1_close", require_fully_closed=False)
    assert out.values[0] == 2.0  # Jan 3's D1 (containing the H4)


# ── out-of-range handling ────────────────────────────────────────────


def test_ltf_before_any_htf_returns_nan() -> None:
    """LTF timestamp before any HTF bar → NaN."""
    d1 = _d1_panel_utc(days=3)  # Jan 1..Jan 3
    h4_pre = pd.DatetimeIndex([pd.Timestamp("2023-12-31 12:00:00", tz="UTC")])
    out = get_htf_value_at(h4_pre, d1, "d1_close")
    assert np.isnan(out.values[0])


def test_ltf_at_first_htf_with_fully_closed_returns_nan() -> None:
    """LTF at the first HTF's own start: no prior bar exists → NaN under default."""
    d1 = _d1_panel_utc(days=3)
    h4_at_first = pd.DatetimeIndex([pd.Timestamp("2024-01-01 00:00:00", tz="UTC")])
    out = get_htf_value_at(h4_at_first, d1, "d1_close")
    # require_fully_closed=True: k=0 from searchsorted, then k=−1 → invalid → NaN
    assert np.isnan(out.values[0])


def test_ltf_at_first_htf_without_fully_closed_returns_first_bar() -> None:
    """LTF at the first HTF's start: with require_fully_closed=False, returns that bar."""
    d1 = _d1_panel_utc(days=3)
    h4_at_first = pd.DatetimeIndex([pd.Timestamp("2024-01-01 00:00:00", tz="UTC")])
    out = get_htf_value_at(h4_at_first, d1, "d1_close", require_fully_closed=False)
    assert out.values[0] == 0.0


# ── tz-awareness contract ────────────────────────────────────────────


def test_tz_naive_ltf_against_tz_aware_htf_raises() -> None:
    d1 = _d1_panel_utc(days=3)
    h4_naive = pd.DatetimeIndex([pd.Timestamp("2024-01-02 04:00:00")])  # no tz
    with pytest.raises(ValueError, match="tz-awareness"):
        get_htf_value_at(h4_naive, d1, "d1_close")


def test_tz_aware_ltf_against_tz_naive_htf_raises() -> None:
    d1 = _d1_panel_utc(days=3)
    d1.index = d1.index.tz_localize(None)  # strip tz
    h4 = pd.DatetimeIndex([pd.Timestamp("2024-01-02 04:00:00", tz="UTC")])
    with pytest.raises(ValueError, match="tz-awareness"):
        get_htf_value_at(h4, d1, "d1_close")


def test_tz_naive_both_sides_ok() -> None:
    """Tz-naive on both sides is allowed (the contract is *matched* awareness)."""
    d1 = _d1_panel_utc(days=3)
    d1.index = d1.index.tz_localize(None)
    h4_naive = pd.DatetimeIndex([pd.Timestamp("2024-01-02 04:00:00")])
    out = get_htf_value_at(h4_naive, d1, "d1_close")
    assert out.values[0] == 0.0  # Jan 1 D1


# ── coercion ─────────────────────────────────────────────────────────


def test_accepts_series_of_timestamps() -> None:
    d1 = _d1_panel_utc(days=5)
    h4_series = pd.Series(pd.to_datetime(["2024-01-03 04:00:00"], utc=True))
    out = get_htf_value_at(h4_series, d1, "d1_close")
    assert out.values[0] == 1.0


def test_coerce_to_index_preserves_tz() -> None:
    s = pd.Series(pd.to_datetime(["2024-01-01"], utc=True))
    idx = _coerce_to_index(s)
    assert idx.tz is not None


# ── unknown column ───────────────────────────────────────────────────


def test_missing_column_raises() -> None:
    d1 = _d1_panel_utc(days=3)
    h4 = pd.DatetimeIndex([pd.Timestamp("2024-01-02 04:00:00", tz="UTC")])
    with pytest.raises(KeyError, match="not in htf_panel.columns"):
        get_htf_value_at(h4, d1, "missing")


# ── get_htf_row_at ───────────────────────────────────────────────────


def test_row_at_matches_value_at_per_column() -> None:
    """get_htf_row_at returns the same values as per-column get_htf_value_at calls."""
    d1 = _d1_panel_utc(days=5).assign(d1_kijun=lambda d: d["d1_close"] * 2.0)
    h4 = _h4_index_utc(days=3)
    rows = get_htf_row_at(h4, d1)
    close_via_value = get_htf_value_at(h4, d1, "d1_close")
    kijun_via_value = get_htf_value_at(h4, d1, "d1_kijun")
    pd.testing.assert_series_equal(
        rows["d1_close"].rename("d1_close"),
        close_via_value.astype(rows["d1_close"].dtype),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        rows["d1_kijun"].rename("d1_kijun"),
        kijun_via_value.astype(rows["d1_kijun"].dtype),
        check_names=False,
    )


# ── byte-identical to legacy KH-24 D1-lag-1 idiom under UTC ──────────


def _legacy_kh24_d1_lag1(
    df_h4: pd.DataFrame, df_d1: pd.DataFrame, column: str
) -> np.ndarray:
    """Reproduces the legacy KH-24 _build_d1_lag1_arrays idiom (pre-fix).

    Used as the reference baseline for the byte-identical regression test:
    new utility output MUST match this under the UTC boundary convention.
    """
    d1 = pd.DataFrame(index=df_d1.index.copy())
    d1["_value"] = df_d1[column].values
    d1["_date"] = d1.index.normalize()
    d1 = d1.drop_duplicates(subset=["_date"], keep="last").reset_index(drop=True)

    shifted = pd.DataFrame(
        {
            "_date": df_h4.index.normalize() - pd.Timedelta(days=1),
            "_idx": np.arange(len(df_h4), dtype=np.int64),
        }
    ).sort_values("_date")
    merged = pd.merge_asof(
        shifted, d1[["_date", "_value"]], on="_date", direction="backward"
    )
    merged = merged.sort_values("_idx").reset_index(drop=True)
    return merged["_value"].values.astype(float)


def test_byte_identical_to_legacy_kh24_idiom_under_utc() -> None:
    """Canonical utility output == legacy KH-24 idiom output under UTC.

    Mandatory regression guard for Q2 mitigation (chat-approved):
    KH-24 live-deployment-adjacent code path must produce byte-identical
    output under the UTC convention (which is the convention KH-24
    currently uses at runtime).
    """
    # Realistic-size panel: 30 days of D1, ~28 days of H4 inside.
    d1 = _d1_panel_utc(days=30)
    h4 = _h4_index_utc(days=28)
    df_h4 = pd.DataFrame({"placeholder": np.zeros(len(h4))}, index=h4)

    legacy = _legacy_kh24_d1_lag1(df_h4, d1, "d1_close")
    canonical = get_htf_value_at(h4, d1, "d1_close").to_numpy()

    # Byte-identical under UTC convention.
    np.testing.assert_array_equal(canonical, legacy)


def test_byte_identical_when_ltf_extends_past_last_htf_under_utc() -> None:
    """Cover the case where LTF timestamps extend PAST the last HTF bar.

    Test fixture: D1 ends at Jan 28; H4 queried at Jan 29 12:00. Legacy
    KH-24 idiom picks D1 Jan 28 (the last D1, which fully closed at the
    Jan 29 00:00 boundary). The bar_end formulation must also pick Jan 28.

    Regression test against the earlier `k = k - 1` formulation, which
    incorrectly picked Jan 27 (one bar too early) in this case and broke
    test_kh24_kijun_d1.test_predicate_fires_when_prev_d1_close_below_prev_d1_kijun.
    """
    # D1 spans Jan 1..Jan 28 (28 days, d1_close = 0..27).
    start_d1 = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    d1_idx = pd.date_range(start_d1, periods=28, freq="1D", tz="UTC")
    d1 = pd.DataFrame({"d1_close": np.arange(28, dtype=float)}, index=d1_idx)

    # H4 at Jan 29 12:00 (past the last D1 by ~36 hours; D1 Jan 28 fully closed).
    h4 = pd.DatetimeIndex([pd.Timestamp("2024-01-29 12:00:00", tz="UTC")])
    df_h4 = pd.DataFrame({"placeholder": np.zeros(1)}, index=h4)

    legacy = _legacy_kh24_d1_lag1(df_h4, d1, "d1_close")
    canonical = get_htf_value_at(h4, d1, "d1_close").to_numpy()
    np.testing.assert_array_equal(canonical, legacy)
    # Specifically: both pick d1_close = 27 (Jan 28's D1), NOT 26 (Jan 27).
    assert canonical[0] == 27.0


def test_byte_identical_when_ltf_inside_last_htf_under_utc() -> None:
    """LTF timestamp inside the last HTF bar (still active) → both pick prior.

    H4 at Jan 28 12:00 with D1 spanning Jan 1..Jan 28. Last D1 (Jan 28)
    started at UTC 00:00 Jan 28 but doesn't end until UTC 00:00 Jan 29 —
    it's still active at Jan 28 12:00. Legacy picks D1 Jan 27 (= one day
    before H4's calendar date). Canonical must too.
    """
    start_d1 = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    d1_idx = pd.date_range(start_d1, periods=28, freq="1D", tz="UTC")
    d1 = pd.DataFrame({"d1_close": np.arange(28, dtype=float)}, index=d1_idx)

    h4 = pd.DatetimeIndex([pd.Timestamp("2024-01-28 12:00:00", tz="UTC")])
    df_h4 = pd.DataFrame({"placeholder": np.zeros(1)}, index=h4)

    legacy = _legacy_kh24_d1_lag1(df_h4, d1, "d1_close")
    canonical = get_htf_value_at(h4, d1, "d1_close").to_numpy()
    np.testing.assert_array_equal(canonical, legacy)
    # Both pick Jan 27 (= d1_close index 26).
    assert canonical[0] == 26.0


def test_canonical_disagrees_with_legacy_under_eet() -> None:
    """Under EET, canonical must DIFFER from the legacy normalize idiom.

    Legacy picks same-EET-day D1 (lookahead). Canonical picks prior-EET-day.
    Verifying the divergence proves the fix is doing the intended thing.
    """
    d1 = _d1_panel_eet(days=30)  # D1 EET-day-2..EET-day-31
    h4 = _h4_index_eet(days=28)  # H4 starting EET 00:00 EET-day-2
    df_h4 = pd.DataFrame({"placeholder": np.zeros(len(h4))}, index=h4)

    legacy = _legacy_kh24_d1_lag1(df_h4, d1, "d1_close")
    canonical = get_htf_value_at(h4, d1, "d1_close").to_numpy()

    # Must disagree on at least one position.
    diff_mask = ~np.isclose(legacy, canonical, equal_nan=True)
    assert diff_mask.any(), (
        "canonical and legacy must DIFFER under EET storage — if they agree, "
        "either the test fixture doesn't exercise the bug or the canonical "
        "utility has regressed to the buggy behaviour."
    )
