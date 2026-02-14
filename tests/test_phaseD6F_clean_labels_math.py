"""
Phase D-6F: Math correctness tests for clean opportunity labels.
Uses toy OHLC in-memory; no full dataset required.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from labels.clean_opportunity import clean_mfe_before_mae, compute_clean_labels_for_pair


def _make_toy_ohlc(n: int) -> pd.DataFrame:
    dates = pd.date_range("2019-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "date": dates,
        "open": np.arange(n, dtype=float) * 10.0,
        "high": np.arange(n, dtype=float) * 10.0 + 5.0,
        "low": np.arange(n, dtype=float) * 10.0 - 5.0,
        "close": np.arange(n, dtype=float) * 10.0,
        "volume": np.zeros(n),
    })


def test_first_breach_cutoff_excludes_breach_bar() -> None:
    """
    When breach happens on bar k and favorable spike is also on bar k,
    clean MFE must NOT include breach bar (strict j < k).
    Bar 0: adverse >= 1 (breach), favorable large; max over j<1 is empty -> 0.
    """
    ref_px = 100.0
    r_value = 10.0
    highs = np.array([200.0, 105.0, 102.0])
    lows = np.array([80.0, 100.0, 98.0])
    adverse_0 = (ref_px - lows[0]) / r_value
    favorable_0 = (highs[0] - ref_px) / r_value
    assert adverse_0 >= 1
    assert favorable_0 > 5

    mfe, breach, _ = clean_mfe_before_mae(
        highs, lows, ref_px, r_value, "long", x=1, h=3
    )
    assert breach == 1
    assert mfe == 0.0


def test_no_lookahead_beyond_window() -> None:
    """Put massive favorable on bar H+1; result for H unchanged."""
    ref_px = 100.0
    r_value = 10.0
    h = 5
    highs = np.array([101.0, 102.0, 103.0, 104.0, 105.0, 999.0])
    lows = np.array([99.0, 98.0, 97.0, 96.0, 95.0, 90.0])
    mfe_within, _, _ = clean_mfe_before_mae(
        highs[:h], lows[:h], ref_px, r_value, "long", x=1, h=h
    )
    mfe_extra, _, _ = clean_mfe_before_mae(
        highs[: h + 1], lows[: h + 1], ref_px, r_value, "long", x=1, h=h
    )
    assert mfe_within == mfe_extra


def test_ref_px_and_atr_t_usage() -> None:
    """ref_px = open[t+1], R uses ATR[t]."""
    df = _make_toy_ohlc(50)
    df.loc[1, "open"] = 123.45
    for i in range(2, 50):
        df.loc[i, "high"] = 100 + i
        df.loc[i, "low"] = 100 - i
        df.loc[i, "close"] = 100
    out = compute_clean_labels_for_pair(
        df, "X", date_start="2019-01-01", date_end="2026-01-01", atr_period=1
    )
    row0 = out[out["date"] == df["date"].iloc[0]].iloc[0]
    assert row0["ref_px"] == pytest.approx(123.45)


def test_symmetry_long_equals_short_on_mirror() -> None:
    """Mirror price p' = C - p; LONG on original equals SHORT on mirrored."""
    c = 100.0
    highs = np.array([105.0, 110.0, 108.0, 102.0])
    lows = np.array([98.0, 100.0, 95.0, 96.0])
    ref_px = 100.0
    r_value = 5.0

    mfe_long, b_l, p_l = clean_mfe_before_mae(
        highs, lows, ref_px, r_value, "long", x=1, h=4
    )
    high_mirror = c - lows
    low_mirror = c - highs
    ref_mirror = c - ref_px
    mfe_short, b_s, p_s = clean_mfe_before_mae(
        high_mirror, low_mirror, ref_mirror, r_value, "short", x=1, h=4
    )
    assert mfe_long == pytest.approx(mfe_short, rel=1e-9)
    assert b_l == b_s
    assert p_l == p_s


def test_clean_mfe_3r_before_mae_2r_false_when_2r_first() -> None:
    """2R adverse before 3R favorable must return False (clean_mfe < 3)."""
    ref_px = 100.0
    r_value = 10.0
    highs = np.array([105.0, 102.0, 103.0, 115.0, 120.0])
    lows = np.array([75.0, 98.0, 97.0, 96.0, 95.0])
    mfe, breach, _ = clean_mfe_before_mae(highs, lows, ref_px, r_value, "long", x=2, h=5)
    assert breach == 1
    assert mfe < 3.0


def test_clean_mfe_3r_before_mae_2r_true_when_3r_first() -> None:
    """3R favorable before any 2R adverse must return True (clean_mfe >= 3)."""
    ref_px = 100.0
    r_value = 10.0
    highs = np.array([135.0, 108.0, 105.0, 102.0, 101.0])
    lows = np.array([95.0, 92.0, 90.0, 88.0, 85.0])
    mfe, breach, _ = clean_mfe_before_mae(highs, lows, ref_px, r_value, "long", x=2, h=5)
    assert breach is None
    assert mfe >= 3.0


def test_clean_mfe_3r_before_mae_2r_synthetic_path() -> None:
    """compute_clean_labels_for_pair produces clean_mfe_3r_before_mae_2r columns."""
    n = 50
    df = pd.DataFrame({
        "date": pd.date_range("2019-01-01", periods=n, freq="D"),
        "open": 100.0,
        "high": 105.0,
        "low": 95.0,
        "close": 100.0,
        "volume": 0,
    })
    out = compute_clean_labels_for_pair(df, "X", date_start="2019-01-01", date_end="2026-01-01")
    assert "clean_mfe_3r_before_mae_2r_long_h10" in out.columns
    assert "clean_mfe_3r_before_mae_2r_short_h40" in out.columns
    for d in ("long", "short"):
        for h in (10, 20, 40):
            col = f"clean_mfe_3r_before_mae_2r_{d}_h{h}"
            assert col in out.columns
            vals = out[col].dropna()
            if len(vals) > 0:
                assert set(vals.unique()).issubset({True, False})
    assert "clean_mfe_4r_before_mae_2r_long_h40" in out.columns
    assert "clean_mfe_4r_before_mae_2r_short_h40" in out.columns
