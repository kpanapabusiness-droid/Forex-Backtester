"""Tests for core.discovery.quantile_grid — known-input quantile reproducibility."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.discovery.quantile_grid import build_quantile_grid


def _matrix(values: list[float], col: str = "x") -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=len(values), freq="h", tz="UTC")
    return pd.DataFrame({col: values}, index=idx)


def test_quantiles_on_uniform_known_distribution():
    # 0..99 inclusive -> exact decile points known.
    m = _matrix(list(range(100)))
    grid = build_quantile_grid({"P1": m}, quantiles=(0.10, 0.50, 0.90))
    thresholds = grid.thresholds["x"]
    # numpy linear-interpolation: p10 of 0..99 -> 9.9, p50 -> 49.5, p90 -> 89.1
    assert abs(thresholds[0.10] - 9.9) < 1e-9
    assert abs(thresholds[0.50] - 49.5) < 1e-9
    assert abs(thresholds[0.90] - 89.1) < 1e-9
    assert grid.n_observations["x"] == 100


def test_pooling_across_pairs_is_sorted():
    """Pair iteration order is sorted; pool is concatenated in that order."""
    m1 = _matrix([float(v) for v in range(50)])     # 0..49
    m2 = _matrix([float(v) for v in range(50, 100)])  # 50..99
    grid_ab = build_quantile_grid({"AAA": m1, "BBB": m2}, quantiles=(0.50,))
    grid_ba = build_quantile_grid({"BBB": m2, "AAA": m1}, quantiles=(0.50,))
    # Same data pooled in same sorted order -> same thresholds regardless of dict order.
    assert grid_ab.thresholds["x"][0.50] == grid_ba.thresholds["x"][0.50]


def test_nan_excluded_from_quantile():
    m = _matrix([float("nan"), 0.0, 100.0, float("nan")])
    grid = build_quantile_grid({"P": m}, quantiles=(0.50,))
    assert grid.n_observations["x"] == 2
    assert abs(grid.thresholds["x"][0.50] - 50.0) < 1e-9


def test_features_with_data_filter():
    full = _matrix([float(i) for i in range(200)], col="full")
    sparse = _matrix([float("nan")] * 95 + [1.0, 2.0, 3.0, 4.0, 5.0], col="sparse")
    matrices = {"P": pd.concat([full, sparse], axis=1)}
    grid = build_quantile_grid(matrices, quantiles=(0.50,))
    assert "full" in grid.features_with_data(min_non_nan=100)
    assert "sparse" not in grid.features_with_data(min_non_nan=100)
