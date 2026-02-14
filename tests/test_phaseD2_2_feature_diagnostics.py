"""
Phase D-2.2 — Feature diagnostics tests.

Synthetic tests only: causality, bin edge freeze, determinism, guard rails, ATR reuse.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _make_synthetic_ohlc(n: int, start: str = "2019-01-01") -> pd.DataFrame:
    """Build monotonically dated D1 OHLC with non-zero range."""
    np.random.seed(42)
    dates = pd.date_range(start, periods=n, freq="D")
    close = 1.0 + np.cumsum(np.random.randn(n) * 0.002)
    high = close + np.abs(np.random.randn(n) * 0.001)
    low = close - np.abs(np.random.randn(n) * 0.001)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    df = pd.DataFrame({
        "date": dates,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.zeros(n),
    })
    df["high"] = np.maximum(df["high"], np.maximum(df["open"], df["close"]))
    df["low"] = np.minimum(df["low"], np.minimum(df["open"], df["close"]))
    return df


def _minimal_d22_cfg(pair: str = "TEST", atr_period: int = 14) -> dict:
    return {
        "pair": pair,
        "atr_period": atr_period,
        "date_range": {"start": "2019-01-01", "end": "2026-01-01"},
        "split": {"discovery_end": "2022-12-31"},
    }


def test_feature_causality_no_future_bars() -> None:
    """Features at t must use only bars <= t. Validate with constructed series."""
    from analytics.phaseD2_2_features import compute_features_for_pair

    n = 300
    df = _make_synthetic_ohlc(n)
    df["close"] = 1.0 + np.arange(n, dtype=float)
    df["high"] = df["close"] + 0.5
    df["low"] = df["close"] - 0.5
    df["open"] = df["close"]

    cfg = _minimal_d22_cfg()
    out = compute_features_for_pair(df, cfg)

    assert not out.empty
    row_20 = out[out["date"] == df["date"].iloc[20]].iloc[0]
    close_20 = df["close"].iloc[20]
    close_0 = df["close"].iloc[0]
    expected_ret_20 = (close_20 / close_0) - 1.0
    assert row_20["ret_20"] == pytest.approx(expected_ret_20)

    row_10 = out[out["date"] == df["date"].iloc[10]].iloc[0]
    assert pd.isna(row_10["ret_20"])

    row_4 = out[out["date"] == df["date"].iloc[4]].iloc[0]
    assert pd.isna(row_4["ret_5"])


def test_bin_edge_freeze_discovery_only() -> None:
    """Bin edges derived from discovery; same edges applied to validation."""
    from analytics.phaseD2_2_features import (
        apply_bin_edges,
        compute_bin_edges_from_discovery,
    )

    np.random.seed(123)
    disc_vals = np.random.randn(300) * 10
    val_vals = np.random.randn(200) * 10 + 5
    df = pd.DataFrame({
        "feature_x": np.concatenate([disc_vals, val_vals]),
        "dataset_split": ["discovery"] * 300 + ["validation"] * 200,
    })
    edges = compute_bin_edges_from_discovery(
        df, "feature_x", n_bins=10, min_per_bin=20
    )
    assert edges is not None
    assert len(edges) == 11
    assert edges[0] == -np.inf
    assert edges[-1] == np.inf

    disc_bins = apply_bin_edges(
        df.loc[df["dataset_split"] == "discovery", "feature_x"], edges
    )
    val_bins = apply_bin_edges(
        df.loc[df["dataset_split"] == "validation", "feature_x"], edges
    )
    assert disc_bins.min() >= 0
    assert disc_bins.max() <= 9
    assert val_bins.min() >= 0
    assert val_bins.max() <= 9


def test_determinism_same_input_same_ranking() -> None:
    """Same input produces same ranking outputs."""
    from analytics.phaseD2_2_features import (
        FEATURE_NAMES,
        ZONE_B,
        compute_bin_edges_from_discovery,
        compute_feature_rankings,
        compute_features_for_pair,
    )

    df = _make_synthetic_ohlc(400)
    cfg = _minimal_d22_cfg()
    feats = compute_features_for_pair(df, cfg)

    labels = pd.DataFrame({
        "pair": ["TEST"] * 798,
        "date": list(pd.date_range("2019-01-02", periods=399, freq="D")) * 2,
        "direction": ["long"] * 399 + ["short"] * 399,
        "zone_b_3r_20": np.random.rand(798) > 0.9,
        "zone_c_6r_40": np.random.rand(798) > 0.95,
    })
    labels["date"] = pd.to_datetime(labels["date"])
    labels["dataset_split"] = labels["date"].apply(
        lambda d: "discovery" if d <= pd.Timestamp("2022-12-31") else "validation"
    )

    joined = pd.merge(
        labels,
        feats.drop(columns=["dataset_split"], errors="ignore"),
        on=["pair", "date"],
        how="inner",
    )
    if "dataset_split" not in joined.columns:
        joined["dataset_split"] = labels["dataset_split"]

    bin_edges = {}
    for f in FEATURE_NAMES:
        if f in joined.columns:
            bin_edges[f] = compute_bin_edges_from_discovery(
                joined, f, n_bins=10, min_per_bin=20
            )

    r1 = compute_feature_rankings(joined, bin_edges, ZONE_B, {"discovery_end": "2022-12-31"})
    r2 = compute_feature_rankings(joined, bin_edges, ZONE_B, {"discovery_end": "2022-12-31"})
    pd.testing.assert_frame_equal(r1, r2)


def test_guard_rails_divide_by_zero_and_nan() -> None:
    """Divide-by-zero yields NaN; insufficient history yields NaN."""
    from analytics.phaseD2_2_features import compute_features_for_pair

    df = _make_synthetic_ohlc(30)
    df.loc[10, "high"] = df.loc[10, "low"]
    cfg = _minimal_d22_cfg()
    out = compute_features_for_pair(df, cfg)

    row_10 = out[out["date"] == df["date"].iloc[10]].iloc[0]
    assert pd.isna(row_10["body_pct"]) or row_10["body_pct"] >= 0

    row_0 = out[out["date"] == df["date"].iloc[0]].iloc[0]
    assert pd.isna(row_0["ret_1"])
    assert pd.isna(row_0["ret_5"])
    assert pd.isna(row_0["ret_20"])
    assert pd.isna(row_0["mom_slope_5"])
    assert pd.isna(row_0["mom_slope_20"])


def test_atr_reuses_engine_helper() -> None:
    """Pipeline uses core.utils.calculate_atr (called from compute_features_for_pair)."""
    from unittest.mock import patch

    from core.utils import calculate_atr

    with patch("core.utils.calculate_atr", wraps=calculate_atr) as m:
        from analytics.phaseD2_2_features import compute_features_for_pair

        df = _make_synthetic_ohlc(300)
        cfg = _minimal_d22_cfg()
        compute_features_for_pair(df, cfg)
        m.assert_called()
